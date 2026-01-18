"""
Quick setup script for CPU training
Removes old pickle files and re-processes data
"""

import shutil
from pathlib import Path
import config

def clean_processed_data():
    """Remove old processed data files"""
    print("=" * 80)
    print("CLEANING OLD DATA")
    print("=" * 80)
    
    files_to_remove = [
        config.TOKENIZER_EN_PATH,
        config.TOKENIZER_FR_PATH,
        config.PROCESSED_DATA_DIR / "train_dataset.pkl",
        config.PROCESSED_DATA_DIR / "val_dataset.pkl",
        config.PROCESSED_DATA_DIR / "test_dataset.pkl",
    ]
    
    for file_path in files_to_remove:
        if file_path.exists():
            file_path.unlink()
            print(f"✓ Removed {file_path.name}")
        else:
            print(f"  {file_path.name} not found (OK)")
    
    print("\n✓ Cleanup complete!")
    print("=" * 80)

def setup():
    """Main setup function"""
    print("\n" + "=" * 80)
    print("SETUP FOR CPU TRAINING")
    print("=" * 80)
    print("\nThis script will:")
    print("  1. Remove old processed data")
    print("  2. Re-process with CPU-optimized settings")
    print("  3. Prepare for training")
    print("\n" + "=" * 80)
    
    # Check if raw data exists
    if not config.RAW_DATA_FILE.exists():
        print("\n⚠️  Raw data not found!")
        print(f"Expected: {config.RAW_DATA_FILE}")
        print("\nPlease run: python download_data.py")
        return 1
    
    # Clean old data
    clean_processed_data()
    
    # Re-process data
    print("\n" + "=" * 80)
    print("RE-PROCESSING DATA")
    print("=" * 80)
    
    from utils.data_loader import prepare_data
    datasets, tokenizer_en, tokenizer_fr = prepare_data()
    
    print("\n" + "=" * 80)
    print("✓ SETUP COMPLETE!")
    print("=" * 80)
    print("\nYou can now train models:")
    print("  python train.py --model baseline --epochs 10")
    print("  python train.py --model attention --epochs 10")
    print("\n" + "=" * 80)
    
    return 0

if __name__ == "__main__":
    exit(setup())