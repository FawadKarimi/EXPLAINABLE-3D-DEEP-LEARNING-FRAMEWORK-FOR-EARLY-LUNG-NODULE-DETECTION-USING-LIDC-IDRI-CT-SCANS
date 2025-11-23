// Medical Lung Nodule Detection - Frontend JavaScript

// DOM Elements
const uploadZone = document.getElementById('uploadZone');
const fileInput = document.getElementById('fileInput');
const uploadSection = document.getElementById('uploadSection');
const resultsSection = document.getElementById('resultsSection');
const uploadProgress = document.getElementById('uploadProgress');
const progressFill = document.getElementById('progressFill');
const progressText = document.getElementById('progressText');

// Drag and Drop
uploadZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadZone.classList.add('dragover');
});

uploadZone.addEventListener('dragleave', () => {
    uploadZone.classList.remove('dragover');
});

uploadZone.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadZone.classList.remove('dragover');
    
    const files = e.dataTransfer.files;
    if (files.length > 0) {
        handleFileUpload(files[0]);
    }
});

// File Input Change
fileInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) {
        handleFileUpload(e.target.files[0]);
    }
});

// Handle File Upload
function handleFileUpload(file) {
    // Validate file
    const validExtensions = ['.dcm', '.nii', '.nii.gz', '.zip'];
    const isValid = validExtensions.some(ext => file.name.toLowerCase().endsWith(ext));
    
    if (!isValid) {
        alert('Invalid file type. Please upload a DICOM (.dcm), NIfTI (.nii, .nii.gz), or ZIP file.');
        return;
    }
    
    // Show progress
    uploadProgress.style.display = 'block';
    progressText.textContent = 'Uploading scan...';
    
    // Create FormData
    const formData = new FormData();
    formData.append('file', file);
    
    // Upload with progress
    const xhr = new XMLHttpRequest();
    
    xhr.upload.addEventListener('progress', (e) => {
        if (e.lengthComputable) {
            const percentComplete = (e.loaded / e.total) * 100;
            progressFill.style.width = percentComplete + '%';
        }
    });
    
    xhr.addEventListener('load', () => {
        if (xhr.status === 200) {
            const response = JSON.parse(xhr.responseText);
            
            if (response.success) {
                progressText.textContent = 'Processing complete!';
                setTimeout(() => {
                    displayResults(response);
                }, 500);
            } else {
                alert('Error: ' + (response.error || 'Unknown error'));
                resetUpload();
            }
        } else {
            const error = JSON.parse(xhr.responseText);
            alert('Error: ' + (error.error || 'Upload failed'));
            resetUpload();
        }
    });
    
    xhr.addEventListener('error', () => {
        alert('Network error. Please try again.');
        resetUpload();
    });
    
    // Update progress text during processing
    xhr.upload.addEventListener('loadend', () => {
        progressText.textContent = 'Analyzing scan with AI model...';
        progressFill.style.width = '100%';
    });
    
    xhr.open('POST', '/upload');
    xhr.send(formData);
}

// Display Results
function displayResults(data) {
    // Hide upload section
    uploadSection.style.display = 'none';
    
    // Show results section
    resultsSection.style.display = 'block';
    
    // Set prediction
    const predictionValue = document.getElementById('predictionValue');
    const resultBadge = document.getElementById('resultBadge');
    
    predictionValue.textContent = data.prediction;
    resultBadge.textContent = data.prediction;
    
    if (data.prediction === 'Malignant') {
        predictionValue.classList.add('malignant');
        resultBadge.classList.add('malignant');
    } else {
        predictionValue.classList.add('benign');
        resultBadge.classList.add('benign');
    }
    
    // Set confidence
    const confidenceFill = document.getElementById('confidenceFill');
    const confidenceValue = document.getElementById('confidenceValue');
    
    setTimeout(() => {
        confidenceFill.style.width = data.confidence + '%';
        confidenceValue.textContent = data.confidence + '%';
    }, 300);
    
    // Set probabilities
    const probBenignFill = document.getElementById('probBenignFill');
    const probBenignValue = document.getElementById('probBenignValue');
    const probMalignantFill = document.getElementById('probMalignantFill');
    const probMalignantValue = document.getElementById('probMalignantValue');
    
    setTimeout(() => {
        probBenignFill.style.width = data.probability_benign + '%';
        probBenignValue.textContent = data.probability_benign.toFixed(1) + '%';
        
        probMalignantFill.style.width = data.probability_malignant + '%';
        probMalignantValue.textContent = data.probability_malignant.toFixed(1) + '%';
    }, 400);
    
    // Set visualization
    const visualizationImage = document.getElementById('visualizationImage');
    visualizationImage.src = data.visualization;
    
    // Set metadata
    const metadataShape = document.getElementById('metadataShape');
    const metadataSpacing = document.getElementById('metadataSpacing');
    const metadataTime = document.getElementById('metadataTime');
    
    metadataShape.textContent = `${data.metadata.shape[0]} × ${data.metadata.shape[1]} × ${data.metadata.shape[2]}`;
    metadataSpacing.textContent = `${data.metadata.spacing[0].toFixed(2)} × ${data.metadata.spacing[1].toFixed(2)} × ${data.metadata.spacing[2].toFixed(2)} mm`;
    metadataTime.textContent = formatTimestamp(data.metadata.timestamp);
    
    // Scroll to results
    resultsSection.scrollIntoView({ behavior: 'smooth' });
}

// Reset Analysis
function resetAnalysis() {
    // Hide results
    resultsSection.style.display = 'none';
    
    // Show upload
    uploadSection.style.display = 'block';
    
    // Reset upload state
    resetUpload();
    
    // Scroll to top
    window.scrollTo({ top: 0, behavior: 'smooth' });
}

// Reset Upload State
function resetUpload() {
    uploadProgress.style.display = 'none';
    progressFill.style.width = '0%';
    progressText.textContent = 'Uploading...';
    fileInput.value = '';
}

// Format Timestamp
function formatTimestamp(timestamp) {
    // timestamp format: YYYYMMDD_HHMMSS
    const year = timestamp.substring(0, 4);
    const month = timestamp.substring(4, 6);
    const day = timestamp.substring(6, 8);
    const hour = timestamp.substring(9, 11);
    const minute = timestamp.substring(11, 13);
    
    return `${year}-${month}-${day} ${hour}:${minute}`;
}

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    console.log('Lung Nodule Detection System initialized');
});
