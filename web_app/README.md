# Lung Nodule Detection Web Application

A professional medical web interface for AI-powered lung nodule detection and malignancy assessment.

## Features

- **Clean Medical Interface**: Professional design suitable for clinical settings
- **Drag & Drop Upload**: Easy CT scan upload (DICOM, NIfTI formats)
- **Real-time Analysis**: AI-powered nodule detection with explainability
- **Visual Results**: Heatmap overlays showing regions of interest
- **Detailed Metrics**: Confidence scores, probability distributions, and metadata

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Ensure your trained model checkpoint is available:
```
checkpoints/best_model.pth
```

3. Run the application:
```bash
python app.py
```

4. Open your browser and navigate to:
```
http://localhost:5000
```

## Usage

1. **Upload CT Scan**: Drag and drop or browse for DICOM/NIfTI files
2. **Wait for Analysis**: The AI model processes the scan automatically
3. **View Results**: See classification, confidence, and explainability heatmaps
4. **Review Details**: Check probability distributions and scan metadata

## File Structure

```
web_app/
├── app.py                 # Flask backend
├── requirements.txt       # Python dependencies
├── templates/
│   └── index.html        # Main interface
├── static/
│   ├── style.css         # Professional medical styling
│   └── script.js         # Interactive functionality
└── uploads/              # Temporary upload storage
```

## Medical Disclaimer

This AI system is designed to assist healthcare professionals in detecting lung nodules. Results should be reviewed by qualified radiologists and should not be used as the sole basis for clinical decisions.

## Technical Details

- **Backend**: Flask (Python)
- **Frontend**: HTML5, CSS3, JavaScript
- **AI Model**: Hybrid CNN-Transformer with explainability
- **Supported Formats**: DICOM (.dcm), NIfTI (.nii, .nii.gz), ZIP archives

## Security Notes

- Maximum upload size: 500MB
- Files are temporarily stored and should be cleaned periodically
- For production deployment, add authentication and HTTPS
