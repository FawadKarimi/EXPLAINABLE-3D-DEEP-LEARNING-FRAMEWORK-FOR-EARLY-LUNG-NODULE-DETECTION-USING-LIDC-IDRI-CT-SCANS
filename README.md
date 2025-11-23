# Explainable 3D Deep Learning Framework for Early Lung Nodule Detection

A comprehensive deep learning framework for detecting lung nodules in CT scans using a hybrid 3D CNN-Transformer architecture with explainable AI capabilities.

## 🚀 Quick Start 
### Prerequisites
- Python 3.8+
- CUDA-capable GPU (tested on H100)
- LIDC-IDRI dataset downloaded

### Step 1: Upload Project 
```bash
# The folder structure should be:
lung_nodule_detection/
├── main.py
├── requirements.txt
├── config/
├── data/
├── models/
├── training/
├── explainability/
└── ...
```

### Step 2: Download LIDC-IDRI Dataset
1. Download the LIDC-IDRI dataset from: https://www.cancerimagingarchive.net/collection/lidc-idri/
2. Extract the dataset to a folder (e.g., `./data/LIDC-IDRI/`)

The dataset structure should look like:
```
data/
└── LIDC-IDRI/
    ├── LIDC-IDRI-0001/
    ├── LIDC-IDRI-0002/
    └── ...
```

### Step 3: Install Dependencies
```bash
# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Or install as package
pip install -e .
```

### Step 4: Run the Complete Pipeline
```bash
# Automatic mode - detects data and processes everything
python main.py --data_root ./data

# The script will automatically:
# 1. Find LIDC-IDRI dataset
# 2. Preprocess all CT scans (DICOM → NIfTI → Preprocessed)
# 3. Parse XML annotations and create JSON files
# 4. Split dataset (70% train, 15% val, 15% test)
# 5. Create data loaders
# 6. Initialize model
# 7. Start training
```

### Step 5: Monitor Training
Training progress will be displayed in the terminal. Checkpoints are saved to `./checkpoints/`.

## 📁 Project Structure

```
lung_nodule_detection/
│
├── main.py                    # Main pipeline script (run this!)
├── setup.py                   # Package setup
├── requirements.txt           # Python dependencies
├── README.md                  # This file
├── .gitignore                 # Git ignore rules
│
├── config/
│   ├── config.py              # Configuration management
│   └── config.yaml            # Default configuration
│
├── data/
│   ├── dataset.py             # Dataset class
│   ├── preprocessing.py       # Preprocessing pipeline
│   ├── augmentation.py        # Data augmentation
│   └── lidc_parser.py         # LIDC annotation parser
│
├── models/
│   ├── resnet3d.py            # 3D ResNet backbone
│   ├── transformer.py         # Multi-scale Transformer
│   └── hybrid_model.py        # Hybrid CNN-Transformer model
│
├── training/
│   └── trainer.py             # Training loop and utilities
│
├── explainability/
│   └── xai_modules.py         # XAI methods (Grad-CAM++, IG, Attention)
│
├── scripts/
│   ├── train.py               # Training script (alternative)
│   └── preprocess_lidc.py     # Preprocessing script (alternative)
│
├── checkpoints/               # Model checkpoints (created during training)
├── logs/                      # Training logs (created during training)
├── outputs/                   # Output files (created during training)
├── cache/                     # Preprocessed data cache
└── annotations/               # Annotation JSON files (created automatically)
```

## 🔧 Configuration

The project uses YAML-based configuration. Default settings are in `config/config.yaml`.

Key configuration options:
- **Data**: batch size, patch size, augmentation settings
- **Model**: ResNet depth, Transformer dimensions, fusion type
- **Training**: learning rates, epochs, scheduler, mixed precision
- **Evaluation**: metrics, size categories, XAI settings

To use a custom config:
```bash
python main.py --data_root ./data --config my_config.yaml
```

## 📊 Usage Examples

### Basic Training
```bash
# Automatic everything
python main.py --data_root ./data
```

### Skip Preprocessing (if already done)
```bash
python main.py --data_root ./data --skip_preprocessing
```

### Resume Training
```bash
python main.py --data_root ./data --resume checkpoints/best_model.pth
```

### Specify GPU
```bash
python main.py --data_root ./data --gpu_id 0
```

### Custom Configuration
```bash
python main.py --data_root ./data --config custom_config.yaml
```

## 🎯 Features

### Architecture
- **3D ResNet-18/34 Backbone**: Extracts multi-scale volumetric features
- **Multi-Scale Transformer**: Processes features at different resolutions
- **Hierarchical Fusion**: Combines CNN and Transformer features
- **Classification Head**: Binary nodule classification

### Data Processing
- **DICOM to NIfTI Conversion**: Standardized medical image format
- **Intensity Normalization**: Window/level normalization (-600/1400 HU)
- **Isotropic Resampling**: 1mm³ voxel spacing
- **Lung Segmentation**: Automatic lung field extraction
- **Data Augmentation**: Rotation, elastic deformation, noise, etc.

### Training Features
- **Mixed Precision Training**: Faster training with FP16
- **Differential Learning Rates**: Different LR for CNN and Transformer
- **Learning Rate Scheduling**: Cosine annealing with warm restarts
- **Gradient Clipping**: Prevents gradient explosion
- **Early Stopping**: Prevents overfitting
- **Hard Example Mining**: Focuses on difficult samples
- **XAI-Guided Loss**: Incorporates explainability into training

### Explainability
- **Grad-CAM++**: Visual explanations from CNN features
- **Integrated Gradients**: Attribution-based explanations
- **Attention Visualization**: Transformer attention maps
- **Multi-Modal Fusion**: Combines multiple explanation methods

## 📈 Training Output

The training script will:
1. Display progress bars for preprocessing
2. Show training/validation metrics each epoch
3. Save checkpoints periodically
4. Save best model based on validation loss
5. Log all metrics to console

Checkpoints are saved to `./checkpoints/`:
- `best_model.pth`: Best model based on validation loss
- `checkpoint_epoch_N.pth`: Periodic checkpoints
- `final_model.pth`: Final model after training
- `config.yaml`: Training configuration

## 🔍 Troubleshooting

### Dataset Not Found
```
❌ ERROR: Could not find LIDC-IDRI dataset!
```
**Solution**: Ensure the dataset is in one of these locations:
- `./data/LIDC-IDRI/`
- `./data/LIDC/`
- `./data/` (with LIDC-IDRI-XXXX folders directly)

Or specify the exact path:
```bash
python main.py --data_root /path/to/LIDC-IDRI
```

### Out of Memory
If you encounter CUDA out of memory:
1. Reduce batch size in `config/config.yaml`:
   ```yaml
   data:
     batch_size: 8  # Reduce from 16
   ```
2. Reduce patch size:
   ```yaml
   data:
     patch_size: [48, 48, 48]  # Reduce from [64, 64, 64]
   ```

### Preprocessing Takes Too Long
Preprocessing can take several hours for the full dataset. You can:
1. Process a subset first:
   ```bash
   # Process only first 10 scans for testing
   python scripts/preprocess_lidc.py --lidc_root ./data/LIDC-IDRI --scan_ids LIDC-IDRI-0001 LIDC-IDRI-0002 ...
   ```
2. Use `--skip_preprocessing` if already done:
   ```bash
   python main.py --data_root ./data --skip_preprocessing
   ```

## 📝 Citation

If you use this code, please cite:

```bibtex
@misc{lung_nodule_detection,
  title={Explainable 3D Deep Learning Framework for Early Lung Nodule Detection},
  author={Mohammad Fawad Karimi},
  year={2024}
}
```

## 📄 License

[Specify your license here]

## 👤 Author

Mohammad Fawad Karimi

## 🙏 Acknowledgments

- LIDC-IDRI dataset: https://www.cancerimagingarchive.net/collection/lidc-idri/
- PyTorch: https://pytorch.org/
- SimpleITK: https://simpleitk.org/

## 📧 Contact

[Your contact information]

---

## 🚨 Important Notes for Lab Usage

1. **Data Location**: Place LIDC-IDRI dataset in `./data/` folder
2. **Storage**: Preprocessed data requires significant disk space (~50-100GB)
3. **Time**: Full preprocessing takes 4-8 hours depending on dataset size
4. **GPU**: Ensure CUDA is properly installed and GPU is accessible
5. **Checkpoints**: Regularly save checkpoints (automatic every 5 epochs)

## 🎓 For University Lab Setup

1. **Upload entire project folder** to lab PC
2. **Download LIDC-IDRI** to `./data/` folder
3. **Run**: `python main.py --data_root ./data`
4. **Wait**: The script handles everything automatically
5. **Monitor**: Check terminal output for progress

That's it! The framework is designed to work out-of-the-box. 🎉

