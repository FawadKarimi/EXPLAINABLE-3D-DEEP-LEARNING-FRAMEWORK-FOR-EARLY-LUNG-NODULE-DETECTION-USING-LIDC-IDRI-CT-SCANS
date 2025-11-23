# Quick Start Guide - Lung Nodule Detection Web App

## ✅ Installation Complete!

All required packages have been successfully installed.

## Running the Application

1. **Start the Flask server:**
   ```bash
   cd web_app
   python app.py
   ```

2. **Open your browser:**
   Navigate to: `http://localhost:5000`

3. **Upload a CT scan:**
   - Drag and drop a DICOM (.dcm) or NIfTI (.nii, .nii.gz) file
   - Or click "Browse Files" to select a file
   - Wait for the AI analysis to complete

4. **View Results:**
   - Classification (Benign/Malignant)
   - Confidence score
   - Probability distribution
   - Explainability heatmap
   - Scan metadata

## Important Notes

### Model Checkpoint
Make sure you have a trained model checkpoint at:
```
checkpoints/best_model.pth
```

If you don't have a trained model yet, you'll need to:
1. Preprocess your LIDC-IDRI data
2. Train the model using `scripts/train.py`
3. The best model will be saved automatically

### For Testing Without a Trained Model
You can modify `app.py` line 52-57 to skip loading the checkpoint:
```python
# Comment out these lines for testing:
# if checkpoint_path.exists():
#     checkpoint = torch.load(checkpoint_path, map_location=device)
#     model.load_state_dict(checkpoint['model_state_dict'])
```

## Troubleshooting

### Port Already in Use
If port 5000 is already in use, modify `app.py` line 233:
```python
app.run(debug=True, host='0.0.0.0', port=5001)  # Change port number
```

### Memory Issues
For large CT scans, you may need to increase the upload limit in `app.py` line 20:
```python
app.config['MAX_CONTENT_LENGTH'] = 1000 * 1024 * 1024  # 1GB
```

### CUDA/GPU Support
The app currently uses CPU by default. To enable GPU:
1. Install CUDA-enabled PyTorch
2. The code will automatically detect and use GPU if available

## File Structure

```
web_app/
├── app.py                 # Flask backend
├── requirements.txt       # Dependencies (all installed ✅)
├── templates/
│   └── index.html        # Medical interface
├── static/
│   ├── style.css         # Professional styling
│   └── script.js         # Interactive features
└── uploads/              # Temporary file storage
```

## Next Steps

1. **Test the interface:** Run the app and upload a sample CT scan
2. **Train your model:** Use the training scripts if you haven't already
3. **Customize:** Modify colors, text, or features as needed
4. **Deploy:** For production, add authentication and use a production server (gunicorn, nginx)

## Support

For issues or questions, refer to:
- Main README.md in project root
- Training documentation in `scripts/`
- Model documentation in `models/`

---
**Medical Disclaimer:** This AI system assists healthcare professionals. Results should be reviewed by qualified radiologists.
