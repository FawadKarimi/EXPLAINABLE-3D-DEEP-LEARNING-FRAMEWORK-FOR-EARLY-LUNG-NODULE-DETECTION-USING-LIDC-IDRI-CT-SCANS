"""
Complete LIDC-IDRI Data Preprocessing Script
Processes real DICOM scans following thesis methodology (Chapter 3, Section 3.3)
File: scripts/preprocess_lidc.py

Run this script to preprocess the entire LIDC-IDRI dataset
"""

import os
import sys
from pathlib import Path
import argparse
import json
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.preprocessing import (
    DicomToNiftiConverter,
    PreprocessingPipeline
)


def find_dicom_series(patient_dir: Path) -> Path:
    """
    Find the CT series directory within a patient folder
    LIDC-IDRI structure: LIDC-IDRI-XXXX/study_date/series_number/
    """
    # Navigate through nested structure
    for study_dir in patient_dir.iterdir():
        if not study_dir.is_dir():
            continue
        
        for series_dir in study_dir.iterdir():
            if not series_dir.is_dir():
                continue
            
            # Check if contains DICOM files
            dicom_files = list(series_dir.glob('*.dcm'))
            if dicom_files:
                return series_dir
    
    return None


def preprocess_lidc_dataset(
    lidc_root: str,
    output_dir: str,
    nifti_dir: str = None,
    scan_list: list = None
):
    """
    Complete preprocessing pipeline for LIDC-IDRI dataset
    
    Steps (following thesis Section 3.3):
    1. DICOM to NIfTI conversion with HU standardization
    2. Intensity normalization (window/level: -600/1400 HU)
    3. Isotropic resampling (1mm³)
    4. Lung segmentation
    5. Save preprocessed arrays
    
    Args:
        lidc_root: Root directory of LIDC-IDRI dataset
        output_dir: Output directory for preprocessed files
        nifti_dir: Intermediate NIfTI storage (optional)
        scan_list: List of specific scans to process (None = all)
    """
    lidc_root = Path(lidc_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup NIfTI directory
    if nifti_dir is None:
        nifti_dir = output_dir / 'nifti_intermediate'
    nifti_dir = Path(nifti_dir)
    nifti_dir.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Find all patient directories
    if scan_list is None:
        patient_dirs = sorted([
            d for d in lidc_root.iterdir()
            if d.is_dir() and d.name.startswith('LIDC-IDRI-')
        ])
    else:
        patient_dirs = [lidc_root / scan_id for scan_id in scan_list]
    
    print("="*70)
    print("LIDC-IDRI Dataset Preprocessing")
    print("="*70)
    print(f"Source: {lidc_root}")
    print(f"Output: {output_dir}")
    print(f"Total scans to process: {len(patient_dirs)}")
    print("="*70)
    
    # Step 2: DICOM to NIfTI conversion
    print("\n[Step 1/2] Converting DICOM to NIfTI with HU standardization...")
    converter = DicomToNiftiConverter(str(nifti_dir))
    
    nifti_files = []
    failed_conversions = []
    
    for patient_dir in tqdm(patient_dirs, desc="Converting DICOM"):
        scan_id = patient_dir.name
        nifti_path = nifti_dir / f"{scan_id}.nii.gz"
        
        # Skip if already converted
        if nifti_path.exists():
            nifti_files.append(nifti_path)
            continue
        
        try:
            # Find DICOM series
            series_dir = find_dicom_series(patient_dir)
            
            if series_dir is None:
                print(f"[WARN] No DICOM series found for {scan_id}")
                failed_conversions.append(scan_id)
                continue
            
            # Convert
            output_path = converter.convert_series(
                str(series_dir),
                scan_id
            )
            nifti_files.append(Path(output_path))
            
        except Exception as e:
            print(f"[ERROR] Error converting {scan_id}: {e}")
            failed_conversions.append(scan_id)
    
    print(f"\n[OK] DICOM conversion complete:")
    print(f"   Successful: {len(nifti_files)}")
    print(f"   Failed: {len(failed_conversions)}")
    
    if failed_conversions:
        print(f"   Failed scans: {failed_conversions[:5]}...")
    
    # Step 3: Complete preprocessing pipeline
    print("\n[Step 2/2] Applying preprocessing pipeline...")
    print("   - Hounsfield Unit normalization (window: -600+/-700 HU)")
    print("   - Isotropic resampling to 1mm^3")
    print("   - Automatic lung segmentation")
    
    pipeline = PreprocessingPipeline(
        target_spacing=(1.0, 1.0, 1.0),  # 1mm³ isotropic (thesis spec)
        window_level=-300,                # Center of -1000 to +400 HU range (thesis Section 3.3.2)
        window_width=1400,                # Window width (thesis spec)
        apply_lung_segmentation=True
    )
    
    successful = 0
    failed_processing = []
    
    for nifti_path in tqdm(nifti_files, desc="Preprocessing"):
        scan_id = nifti_path.stem.replace('.nii', '')
        
        # Check if already processed
        output_image = output_dir / f"{scan_id}_image.npy"
        if output_image.exists():
            successful += 1
            continue
        
        try:
            # Process
            result = pipeline.process(
                str(nifti_path),
                save_dir=str(output_dir)
            )
            successful += 1
            
        except Exception as e:
            print(f"[ERROR] Error processing {scan_id}: {e}")
            failed_processing.append(scan_id)
    
    # Summary
    print("\n" + "="*70)
    print("Preprocessing Complete!")
    print("="*70)
    print(f"[OK] Successfully processed: {successful}/{len(nifti_files)} scans")
    
    if failed_processing:
        print(f"[ERROR] Failed: {len(failed_processing)} scans")
        print(f"   {failed_processing[:5]}...")
    
    print(f"\n[OUTPUT] Preprocessed files saved to: {output_dir}")
    print(f"   Format: SCAN-ID_image.npy (normalized CT)")
    print(f"           SCAN-ID_mask.npy (lung mask)")
    print(f"           SCAN-ID_metadata.json (spacing, origin, etc.)")
    
    # Create processing summary
    summary = {
        'total_scans': len(patient_dirs),
        'successful': successful,
        'failed_conversion': failed_conversions,
        'failed_processing': failed_processing,
        'preprocessing_params': {
            'target_spacing': [1.0, 1.0, 1.0],
            'window_level': -600,
            'window_width': 1400,
            'lung_segmentation': True
        }
    }
    
    with open(output_dir / 'preprocessing_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n[OK] Summary saved to: {output_dir / 'preprocessing_summary.json'}")


def main():
    parser = argparse.ArgumentParser(
        description='Preprocess LIDC-IDRI dataset for lung nodule detection'
    )
    parser.add_argument(
        '--lidc_root',
        type=str,
        required=True,
        help='Root directory of LIDC-IDRI dataset'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='processed_data',
        help='Output directory for preprocessed files'
    )
    parser.add_argument(
        '--nifti_dir',
        type=str,
        default=None,
        help='Directory for intermediate NIfTI files (default: output_dir/nifti_intermediate)'
    )
    parser.add_argument(
        '--scan_ids',
        type=str,
        nargs='+',
        default=None,
        help='Specific scan IDs to process (default: all)'
    )
    
    args = parser.parse_args()
    
    # Run preprocessing
    preprocess_lidc_dataset(
        lidc_root=args.lidc_root,
        output_dir=args.output_dir,
        nifti_dir=args.nifti_dir,
        scan_list=args.scan_ids
    )


if __name__ == "__main__":
    main()