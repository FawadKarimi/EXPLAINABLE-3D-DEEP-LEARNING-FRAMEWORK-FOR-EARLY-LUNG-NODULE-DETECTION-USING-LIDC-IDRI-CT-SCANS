"""
Medical Lung Nodule Detection Web Application
Flask backend for CT scan upload and analysis
"""

from flask import Flask, render_template, request, jsonify, send_file
from werkzeug.utils import secure_filename
import os
import sys
from pathlib import Path
import torch
import numpy as np
import json
from datetime import datetime
import nibabel as nib
import io
import base64
from PIL import Image

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.hybrid_model import HybridModel
from data.preprocessing import PreprocessingPipeline
from explainability.xai_modules import MultiModalExplainer

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max file size
app.config['ALLOWED_EXTENSIONS'] = {'dcm', 'nii', 'nii.gz', 'zip'}

# Create upload directory
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Global model (loaded once)
model = None
explainer = None
device = None


def allowed_file(filename):
    return '.' in filename and \
           any(filename.lower().endswith(ext) for ext in ['.dcm', '.nii', '.nii.gz', '.zip'])


def load_model():
    """Load trained model"""
    global model, explainer, device
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    model = HybridModel(
        resnet_depth=18,
        use_transformer=True,
        use_multiscale=True,
        num_classes=2
    ).to(device)
    
    # Load checkpoint if exists
    checkpoint_path = Path('checkpoints/best_model.pth')
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
    
    model.eval()
    
    # Initialize explainer
    explainer = MultiModalExplainer(
        model=model,
        target_layer='resnet.resnet.layer4',
        fusion_weights=(0.40, 0.35, 0.25)
    )
    
    print(f"[OK] Model loaded on {device}")


def preprocess_scan(file_path):
    """Preprocess uploaded CT scan"""
    try:
        pipeline = PreprocessingPipeline(
            target_spacing=(1.0, 1.0, 1.0),
            window_level=-300,
            window_width=1400,
            apply_lung_segmentation=True
        )
        
        # Process scan
        result = pipeline.process(file_path, save_dir=None)
        
        # Ensure result is a dictionary
        if isinstance(result, dict):
            image = result.get('image')
            mask = result.get('mask')
            metadata = result.get('metadata', {})
        else:
            # If result is not a dict, assume it's just the image
            image = result
            mask = None
            metadata = {}
        
        # Validate image
        if image is None:
            raise ValueError("Preprocessing did not return an image")
        
        # Ensure metadata is a dictionary
        if not isinstance(metadata, dict):
            metadata = {}
        
        return image, mask, metadata
        
    except Exception as e:
        # If preprocessing fails completely, try to load the file directly
        print(f"Preprocessing error: {e}")
        print("Attempting direct file load...")
        
        try:
            import pydicom
            # Try loading as DICOM
            if file_path.lower().endswith('.dcm'):
                dcm = pydicom.dcmread(file_path)
                image = dcm.pixel_array.astype(np.float32)
                
                # Simple normalization
                image = (image - image.min()) / (image.max() - image.min() + 1e-8)
                
                # Add depth dimension if 2D
                if len(image.shape) == 2:
                    image = image[np.newaxis, ...]
                
                return image, None, {'spacing': [1.0, 1.0, 1.0]}
            else:
                # Try loading as NIfTI
                import nibabel as nib
                nii = nib.load(file_path)
                image = nii.get_fdata().astype(np.float32)
                
                # Simple normalization
                image = (image - image.min()) / (image.max() - image.min() + 1e-8)
                
                spacing = nii.header.get_zooms()
                return image, None, {'spacing': list(spacing[:3])}
                
        except Exception as load_error:
            raise Exception(f"Failed to load file: {str(load_error)}")



def analyze_scan(image_array):
    """Run model inference on preprocessed scan"""
    global model, explainer, device
    
    # Get image dimensions
    if len(image_array.shape) == 2:
        # 2D image - add depth dimension
        image_array = image_array[np.newaxis, ...]
    
    D, H, W = image_array.shape
    patch_size = 64
    
    # Check if image is large enough
    if D < patch_size or H < patch_size or W < patch_size:
        # Pad the image to minimum size
        pad_d = max(0, patch_size - D)
        pad_h = max(0, patch_size - H)
        pad_w = max(0, patch_size - W)
        
        image_array = np.pad(
            image_array,
            ((0, pad_d), (0, pad_h), (0, pad_w)),
            mode='constant',
            constant_values=0
        )
        D, H, W = image_array.shape
    
    # Extract center patch
    d_start = max(0, (D - patch_size) // 2)
    h_start = max(0, (H - patch_size) // 2)
    w_start = max(0, (W - patch_size) // 2)
    
    # Ensure we don't go out of bounds
    d_end = min(D, d_start + patch_size)
    h_end = min(H, h_start + patch_size)
    w_end = min(W, w_start + patch_size)
    
    patch = image_array[
        d_start:d_end,
        h_start:h_end,
        w_start:w_end
    ]
    
    # Ensure patch is exactly 64x64x64
    if patch.shape != (patch_size, patch_size, patch_size):
        # Pad if needed
        pad_d = patch_size - patch.shape[0]
        pad_h = patch_size - patch.shape[1]
        pad_w = patch_size - patch.shape[2]
        
        patch = np.pad(
            patch,
            ((0, pad_d), (0, pad_h), (0, pad_w)),
            mode='constant',
            constant_values=0
        )
    
    # Convert to tensor
    patch_tensor = torch.from_numpy(patch).float().unsqueeze(0).unsqueeze(0).to(device)
    
    # Verify shape
    expected_shape = (1, 1, patch_size, patch_size, patch_size)
    if patch_tensor.shape != expected_shape:
        raise ValueError(f"Patch shape mismatch: {patch_tensor.shape} != {expected_shape}")
    
    # Inference
    with torch.no_grad():
        output = model(patch_tensor)
        logits = output['logits']
        probs = torch.softmax(logits, dim=1)
        
        # Generate explanations (with error handling)
        try:
            explanations = explainer.explain(
                patch_tensor,
                target_class=None,
                return_individual=True
            )
            
            # Convert explanations to numpy
            explanation_maps = {
                key: val.cpu().numpy()[0] for key, val in explanations.items()
            }
        except Exception as exp_error:
            print(f"Explanation generation failed: {exp_error}")
            # Create dummy explanations
            explanation_maps = {
                'fused': np.zeros((patch_size, patch_size, patch_size))
            }
    
    # Extract results
    prediction = int(logits.argmax(dim=1).item())
    confidence = float(probs[0, prediction].item())
    
    return {
        'prediction': prediction,
        'confidence': confidence,
        'probability_benign': float(probs[0, 0].item()),
        'probability_malignant': float(probs[0, 1].item()),
        'explanations': explanation_maps
    }


def create_visualization(image_slice, explanation_slice):
    """Create visualization overlay"""
    # Normalize image
    img_norm = (image_slice - image_slice.min()) / (image_slice.max() - image_slice.min() + 1e-8)
    img_norm = (img_norm * 255).astype(np.uint8)
    
    # Create RGB image
    img_rgb = np.stack([img_norm] * 3, axis=-1)
    
    # Overlay heatmap
    heatmap = (explanation_slice * 255).astype(np.uint8)
    heatmap_colored = np.zeros_like(img_rgb)
    heatmap_colored[:, :, 0] = heatmap  # Red channel
    
    # Blend
    overlay = (0.6 * img_rgb + 0.4 * heatmap_colored).astype(np.uint8)
    
    # Convert to base64
    pil_img = Image.fromarray(overlay)
    buffer = io.BytesIO()
    pil_img.save(buffer, format='PNG')
    buffer.seek(0)
    img_base64 = base64.b64encode(buffer.getvalue()).decode()
    
    return f"data:image/png;base64,{img_base64}"


@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload and analysis"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type'}), 400
    
    try:
        # Save file
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        save_filename = f"{timestamp}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], save_filename)
        file.save(filepath)
        
        # Preprocess with error handling
        try:
            result = preprocess_scan(filepath)
            
            # Handle different return formats
            if isinstance(result, tuple):
                if len(result) == 3:
                    image, mask, metadata = result
                elif len(result) == 2:
                    image, mask = result
                    metadata = {}
                else:
                    image = result[0]
                    mask = None
                    metadata = {}
            elif isinstance(result, dict):
                image = result.get('image')
                mask = result.get('mask')
                metadata = result.get('metadata', {})
            else:
                raise ValueError("Unexpected preprocessing output format")
            
            # Ensure metadata is a dictionary
            if metadata is None:
                metadata = {}
                
        except Exception as preprocess_error:
            return jsonify({'error': f'Preprocessing failed: {str(preprocess_error)}'}), 500
        
        # Analyze
        try:
            results = analyze_scan(image)
        except Exception as analysis_error:
            return jsonify({'error': f'Analysis failed: {str(analysis_error)}'}), 500
        
        # Create visualizations (middle slices)
        try:
            mid_slice = image.shape[0] // 2
            img_slice = image[mid_slice]
            
            # Check if explanations exist
            if 'explanations' in results and 'fused' in results['explanations']:
                exp_slice = results['explanations']['fused'][mid_slice]
            else:
                # Create dummy explanation if not available
                exp_slice = np.zeros_like(img_slice)
            
            visualization = create_visualization(img_slice, exp_slice)
        except Exception as viz_error:
            # Use placeholder image if visualization fails
            visualization = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
        
        # Prepare response with safe metadata access
        response = {
            'success': True,
            'prediction': 'Malignant' if results['prediction'] == 1 else 'Benign',
            'confidence': round(results['confidence'] * 100, 2),
            'probability_benign': round(results['probability_benign'] * 100, 2),
            'probability_malignant': round(results['probability_malignant'] * 100, 2),
            'visualization': visualization,
            'metadata': {
                'spacing': metadata.get('spacing', [1.0, 1.0, 1.0]) if isinstance(metadata, dict) else [1.0, 1.0, 1.0],
                'shape': list(image.shape) if hasattr(image, 'shape') else [0, 0, 0],
                'timestamp': timestamp
            }
        }
        
        return jsonify(response)
    
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Error in upload_file: {error_details}")
        return jsonify({'error': f'Server error: {str(e)}'}), 500


if __name__ == '__main__':
    # Load model on startup
    load_model()
    
    # Run app
    app.run(debug=True, host='0.0.0.0', port=5000)
