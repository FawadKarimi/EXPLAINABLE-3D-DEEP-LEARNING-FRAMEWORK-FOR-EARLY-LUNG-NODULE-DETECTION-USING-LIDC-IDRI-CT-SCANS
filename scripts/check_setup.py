"""
Setup Verification Script
Checks if everything is configured correctly before training
"""

import sys
# Force UTF-8 encoding for standard output to support emojis on Windows
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from pathlib import Path
import torch

def check_python_version():
    """Check Python version"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8+ required")
        return False
    print(f"✅ Python {version.major}.{version.minor}.{version.micro}")
    return True

def check_dependencies():
    """Check if required packages are installed"""
    required = [
        'torch', 'torchvision', 'numpy', 'scipy',
        'SimpleITK', 'nibabel', 'pydicom', 'sklearn',
        'skimage', 'pandas', 'matplotlib', 'seaborn',
        'tqdm', 'yaml'
    ]
    
    missing = []
    for package in required:
        try:
            __import__(package)
        except ImportError:
            missing.append(package)
    
    if missing:
        print(f"❌ Missing packages: {', '.join(missing)}")
        print("   Run: pip install -r requirements.txt")
        return False
    
    print("✅ All required packages installed")
    return True

def check_cuda():
    """Check CUDA availability"""
    if not torch.cuda.is_available():
        print("⚠️  CUDA not available - will use CPU (very slow)")
        return False
    
    print(f"✅ CUDA available")
    print(f"   Device: {torch.cuda.get_device_name(0)}")
    print(f"   Version: {torch.version.cuda}")
    return True

def check_data_structure():
    """Check project structure"""
    required_dirs = [
        'config', 'data', 'models', 'training',
        'explainability', 'scripts'
    ]
    
    missing = []
    for dir_name in required_dirs:
        if not Path(dir_name).exists():
            missing.append(dir_name)
    
    if missing:
        print(f"❌ Missing directories: {', '.join(missing)}")
        return False
    
    print("✅ Project structure correct")
    return True

def check_lidc_data(data_root='./data'):
    """Check if LIDC-IDRI data exists"""
    data_path = Path(data_root)
    
    if not data_path.exists():
        print(f"⚠️  Data directory not found: {data_path}")
        print("   Please download LIDC-IDRI dataset and place it in ./data/")
        return False
    
    # Check for LIDC folders using rglob for nested structures
    for p in data_path.rglob('LIDC-IDRI-*'):
        if p.is_dir() and "LIDC-IDRI-" in p.name:
            parent_dir = p.parent
            lidc_folders = list(parent_dir.glob('LIDC-IDRI-*'))
            if lidc_folders:
                print(f"✅ Found LIDC-IDRI dataset: {len(lidc_folders)} scans at {parent_dir}")
                return True
                
    print(f"⚠️  No LIDC-IDRI folders found in {data_path}")
    print("   Please download LIDC-IDRI dataset")
    return False

def main():
    """Run all checks"""
    print("="*70)
    print("Setup Verification")
    print("="*70)
    
    checks = [
        ("Python Version", check_python_version),
        ("Dependencies", check_dependencies),
        ("CUDA", check_cuda),
        ("Project Structure", check_data_structure),
        ("LIDC Data", lambda: check_lidc_data()),
    ]
    
    results = []
    for name, check_func in checks:
        print(f"\n[{name}]")
        result = check_func()
        results.append((name, result))
    
    print("\n" + "="*70)
    print("Summary")
    print("="*70)
    
    all_passed = all(result for _, result in results)
    
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {name}")
    
    if all_passed:
        print("\n✅ All checks passed! Ready to train.")
        print("\nRun: python main.py --data_root ./data")
    else:
        print("\n⚠️  Some checks failed. Please fix issues above.")
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())

