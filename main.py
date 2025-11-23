"""
Main Pipeline Script for Lung Nodule Detection
Automatically handles: data detection, preprocessing, annotation parsing, and training
Run this script after downloading LIDC-IDRI dataset to the data folder
"""

import os
import sys

# Force UTF-8 encoding for standard output to support emojis on Windows
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import argparse
from pathlib import Path
import json
import torch
import numpy as np
import random

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from config.config import Config, get_default_config
from models.hybrid_model import create_model
from data.dataset import create_dataloaders
from training.trainer import Trainer
from data.lidc_parser import LIDCAnnotationParser
from data.preprocessing import PreprocessingPipeline, DicomToNiftiConverter
from scripts.preprocess_lidc import find_dicom_series


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def find_lidc_data(data_root: Path) -> Path:
    """
    Automatically find LIDC-IDRI dataset directory
    Looks for the first LIDC-IDRI-* folder recursively
    """
    if data_root.exists():
        # First check immediate folder
        lidc_folders = list(data_root.glob('LIDC-IDRI-*'))
        if lidc_folders:
            print(f"✅ Found {len(lidc_folders)} LIDC-IDRI scan folders in {data_root}")
            return data_root
            
        # Search recursively up to a reasonable depth
        for p in data_root.rglob('LIDC-IDRI-*'):
            if p.is_dir() and p.name.startswith("LIDC-IDRI-"):
                parent_dir = p.parent
                lidc_folders = list(parent_dir.glob('LIDC-IDRI-*'))
                print(f"✅ Found {len(lidc_folders)} LIDC-IDRI scan folders at: {parent_dir}")
                return parent_dir
    
    return None


def check_preprocessing_status(processed_dir: Path) -> bool:
    """Check if data has been preprocessed"""
    # Always return False so that preprocess_data checks files individually.
    # The inner loop already skips files that exist.
    return False


def check_annotations_status(annotations_dir: Path) -> bool:
    """Check if annotations have been created"""
    train_file = annotations_dir / 'train_annotations.json'
    val_file = annotations_dir / 'val_annotations.json'
    test_file = annotations_dir / 'test_annotations.json'
    
    return train_file.exists() and val_file.exists() and test_file.exists()


def preprocess_data(config: Config, lidc_root: Path, processed_dir: Path):
    """Preprocess LIDC-IDRI dataset"""
    print("\n" + "="*70)
    print("STEP 1: Data Preprocessing")
    print("="*70)
    
    # Check if already preprocessed
    if check_preprocessing_status(processed_dir):
        print(f"✅ Data already preprocessed in {processed_dir}")
        print(f"   Found preprocessed files. Skipping preprocessing...")
        return
    
    print(f"📁 LIDC-IDRI root: {lidc_root}")
    print(f"💾 Output directory: {processed_dir}")
    
    # Create output directory
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Convert DICOM to NIfTI
    nifti_dir = processed_dir / 'nifti_intermediate'
    nifti_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n🔄 Converting DICOM to NIfTI...")
    converter = DicomToNiftiConverter(str(nifti_dir))
    
    # Find all patient directories
    patient_dirs = sorted([
        d for d in lidc_root.iterdir()
        if d.is_dir() and d.name.startswith('LIDC-IDRI-')
    ])
    
    if not patient_dirs:
        raise ValueError(f"No LIDC-IDRI patient folders found in {lidc_root}")
    
    print(f"   Found {len(patient_dirs)} patient folders")
    
    nifti_files = []
    failed = []
    
    from tqdm import tqdm
    for patient_dir in tqdm(patient_dirs, desc="Converting DICOM"):
        scan_id = patient_dir.name
        nifti_path = nifti_dir / f"{scan_id}.nii.gz"
        
        if nifti_path.exists():
            nifti_files.append(nifti_path)
            continue
        
        try:
            series_dir = find_dicom_series(patient_dir)
            if series_dir is None:
                print(f"⚠️  No DICOM series found for {scan_id}")
                failed.append(scan_id)
                continue
            
            output_path = converter.convert_series(str(series_dir), scan_id)
            nifti_files.append(Path(output_path))
        except Exception as e:
            print(f"❌ Error converting {scan_id}: {e}")
            failed.append(scan_id)
    
    print(f"✅ Converted {len(nifti_files)} scans to NIfTI")
    if failed:
        print(f"⚠️  Failed to convert {len(failed)} scans")
    
    # Step 2: Apply preprocessing pipeline
    print("\n🔄 Applying preprocessing pipeline...")
    pipeline = PreprocessingPipeline(
        target_spacing=config.data.target_spacing,
        window_level=config.data.window_level,
        window_width=config.data.window_width,
        apply_lung_segmentation=True
    )
    
    successful = 0
    for nifti_path in tqdm(nifti_files, desc="Preprocessing"):
        scan_id = nifti_path.stem.replace('.nii', '')
        output_image = processed_dir / f"{scan_id}_image.npy"
        
        if output_image.exists():
            successful += 1
            continue
        
        try:
            pipeline.process(str(nifti_path), save_dir=str(processed_dir))
            successful += 1
        except Exception as e:
            print(f"❌ Error processing {scan_id}: {e}")
    
    print(f"✅ Successfully preprocessed {successful}/{len(nifti_files)} scans")
    print(f"💾 Preprocessed data saved to: {processed_dir}")


def create_annotations(config: Config, lidc_root: Path, annotations_dir: Path):
    """Create annotation files from LIDC XML files"""
    print("\n" + "="*70)
    print("STEP 2: Creating Annotations")
    print("="*70)
    
    # Check if already created
    if check_annotations_status(annotations_dir):
        print(f"✅ Annotations already exist in {annotations_dir}")
        return
    
    annotations_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"📁 Parsing LIDC-IDRI XML annotations...")
    
    # Create parser
    parser = LIDCAnnotationParser(str(lidc_root))
    
    # Process all scans
    all_annotations = parser.process_all_scans(
        output_file=str(annotations_dir / 'all_annotations.json')
    )
    
    # Split dataset
    print("\n🔄 Splitting dataset...")
    train_data, val_data, test_data = parser.split_dataset(
        all_annotations,
        train_ratio=config.data.train_ratio,
        val_ratio=config.data.val_ratio,
        test_ratio=config.data.test_ratio,
        random_seed=config.data.random_seed
    )
    
    # Save splits
    with open(annotations_dir / 'train_annotations.json', 'w') as f:
        json.dump(train_data, f, indent=2)
    
    with open(annotations_dir / 'val_annotations.json', 'w') as f:
        json.dump(val_data, f, indent=2)
    
    with open(annotations_dir / 'test_annotations.json', 'w') as f:
        json.dump(test_data, f, indent=2)
    
    print(f"✅ Annotations created and saved to: {annotations_dir}")
    print(f"   Train: {len(train_data)} scans")
    print(f"   Val: {len(val_data)} scans")
    print(f"   Test: {len(test_data)} scans")


def setup_environment(config: Config):
    """Setup training environment"""
    # Set device
    if config.device == 'cuda' and torch.cuda.is_available():
        device = torch.device(f'cuda:{config.gpu_id}')
        print(f"\n✅ Using GPU: {torch.cuda.get_device_name(config.gpu_id)}")
        print(f"   GPU Memory: {torch.cuda.get_device_properties(config.gpu_id).total_memory / 1e9:.2f} GB")
    else:
        device = torch.device('cpu')
        print("\n⚠️  Using CPU (training will be slow)")
    
    return device


def main():
    """Main pipeline - handles everything automatically"""
    parser = argparse.ArgumentParser(
        description='Lung Nodule Detection - Complete Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Automatic mode (detects data automatically)
  python main.py --data_root ./data
  
  # Specify exact paths
  python main.py --data_root ./data/LIDC-IDRI --skip_preprocessing
  
  # Resume training from checkpoint
  python main.py --data_root ./data --resume checkpoints/best_model.pth
        """
    )
    
    parser.add_argument(
        '--data_root',
        type=str,
        default='./data',
        help='Root directory containing LIDC-IDRI dataset (default: ./data)'
    )
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to config YAML file (default: uses default config)'
    )
    parser.add_argument(
        '--skip_preprocessing',
        action='store_true',
        help='Skip preprocessing if data is already processed'
    )
    parser.add_argument(
        '--skip_annotations',
        action='store_true',
        help='Skip annotation creation if already exists'
    )
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Resume training from checkpoint'
    )
    parser.add_argument(
        '--gpu_id',
        type=int,
        default=None,
        help='GPU ID to use (overrides config)'
    )
    
    args = parser.parse_args()
    
    print("="*70)
    print("Lung Nodule Detection - Complete Pipeline")
    print("="*70)
    
    # Load configuration
    if args.config and Path(args.config).exists():
        config = Config.load(args.config)
        print(f"✅ Loaded configuration from: {args.config}")
    else:
        config = get_default_config()
        print("✅ Using default configuration")
    
    # Override GPU ID if specified
    if args.gpu_id is not None:
        config.gpu_id = args.gpu_id
    
    # Set random seed
    set_seed(config.data.random_seed)
    print(f"✅ Random seed set to: {config.data.random_seed}")
    
    # Setup paths
    data_root = Path(args.data_root).resolve()
    processed_dir = Path(config.data.cache_dir).resolve()
    annotations_dir = Path('annotations').resolve()
    
    # Find LIDC data
    lidc_root = find_lidc_data(data_root)
    
    if lidc_root is None:
        print("\n❌ ERROR: Could not find LIDC-IDRI dataset!")
        print(f"   Searched in: {data_root}")
        print("\n   Please ensure:")
        print("   1. LIDC-IDRI dataset is downloaded")
        print("   2. Data is placed in one of these locations:")
        print(f"      - {data_root}/LIDC-IDRI/")
        print(f"      - {data_root}/LIDC/")
        print(f"      - {data_root}/ (with LIDC-IDRI-XXXX folders)")
        print("\n   Or specify the exact path with --data_root")
        return 1
    
    # Step 1: Preprocess data
    if not args.skip_preprocessing:
        try:
            preprocess_data(config, lidc_root, processed_dir)
        except Exception as e:
            print(f"\n❌ Preprocessing failed: {e}")
            import traceback
            traceback.print_exc()
            return 1
    else:
        print("\n⏭️  Skipping preprocessing (--skip_preprocessing)")
    
    # Step 2: Create annotations
    if not args.skip_annotations:
        try:
            create_annotations(config, lidc_root, annotations_dir)
        except Exception as e:
            print(f"\n❌ Annotation creation failed: {e}")
            import traceback
            traceback.print_exc()
            return 1
    else:
        print("\n⏭️  Skipping annotation creation (--skip_annotations)")
    
    # Step 3: Setup environment
    device = setup_environment(config)
    
    # Step 4: Create data loaders
    print("\n" + "="*70)
    print("STEP 3: Creating Data Loaders")
    print("="*70)
    
    train_annotation = annotations_dir / 'train_annotations.json'
    val_annotation = annotations_dir / 'val_annotations.json'
    test_annotation = annotations_dir / 'test_annotations.json'
    
    if not all([f.exists() for f in [train_annotation, val_annotation, test_annotation]]):
        print("❌ ERROR: Annotation files not found!")
        print("   Please run annotation creation first")
        return 1
    
    try:
        train_loader, val_loader, test_loader = create_dataloaders(
            data_dir=str(processed_dir),
            train_annotation=str(train_annotation),
            val_annotation=str(val_annotation),
            test_annotation=str(test_annotation),
            config=config,
            num_workers=config.data.num_workers
        )
        
        print(f"✅ Data loaders created:")
        print(f"   Train batches: {len(train_loader)}")
        print(f"   Val batches: {len(val_loader)}")
        print(f"   Test batches: {len(test_loader)}")
    except Exception as e:
        print(f"❌ Error creating data loaders: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Step 5: Create model
    print("\n" + "="*70)
    print("STEP 4: Creating Model")
    print("="*70)
    
    model = create_model(config)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"✅ Model created:")
    print(f"   Architecture: Hybrid 3D CNN-Transformer")
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    print(f"   Model size: {total_params * 4 / 1024 / 1024:.2f} MB")
    
    # Step 6: Create trainer
    print("\n" + "="*70)
    print("STEP 5: Initializing Trainer")
    print("="*70)
    
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=str(device),
        checkpoint_dir=config.checkpoint_dir
    )
    
    print(f"✅ Trainer initialized:")
    print(f"   Optimizer: AdamW")
    print(f"   LR (CNN): {config.training.lr_cnn}")
    print(f"   LR (Transformer): {config.training.lr_transformer}")
    print(f"   Scheduler: {config.training.scheduler}")
    print(f"   Mixed precision: {config.training.use_amp}")
    
    # Resume from checkpoint if specified
    if args.resume:
        if Path(args.resume).exists():
            print(f"\n📂 Resuming from checkpoint: {args.resume}")
            trainer.load_checkpoint(args.resume)
        else:
            print(f"⚠️  Checkpoint not found: {args.resume}")
    
    # Step 7: Start training
    print("\n" + "="*70)
    print("STEP 6: Starting Training")
    print("="*70)
    print(f"Epochs: {config.training.num_epochs}")
    print(f"Batch size: {config.data.batch_size}")
    print(f"Warmup epochs: {config.training.warmup_epochs}")
    print(f"Early stopping patience: {config.training.patience}")
    print("="*70 + "\n")
    
    try:
        trainer.train()
    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user")
        print("   Saving checkpoint...")
        trainer.save_checkpoint('interrupted.pth')
        print("   ✅ Checkpoint saved: interrupted.pth")
        return 0
    except Exception as e:
        print(f"\n❌ Training error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Save final model
    print("\n💾 Saving final model...")
    trainer.save_checkpoint('final_model.pth')
    config.save(Path(config.checkpoint_dir) / 'config.yaml')
    print("✅ Configuration saved")
    
    print("\n" + "="*70)
    print("✅ Training Complete!")
    print("="*70)
    print(f"Best validation loss: {trainer.best_val_loss:.4f}")
    print(f"Checkpoints saved to: {config.checkpoint_dir}")
    print(f"\nNext steps:")
    print(f"  1. Evaluate model: python scripts/evaluate.py --checkpoint {config.checkpoint_dir}/best_model.pth")
    print(f"  2. Generate explanations: python scripts/inference.py --checkpoint {config.checkpoint_dir}/best_model.pth")
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)

