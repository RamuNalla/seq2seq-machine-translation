import os
import requests
import zipfile
from pathlib import Path
from tqdm import tqdm
import config

def download_file(url: str, destination: Path):
    """
    Args:
        url: URL to download from
        destination: Path to save the file
    """
    print(f"Downloading from {url}...")
    
    # Some servers block requests that don't look like a browser. Set a
    # common User-Agent and a reasonable timeout. Allow redirects.
    headers = {
        'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 '
                      '(KHTML, like Gecko) Chrome/115.0 Safari/537.36'
    }
    response = requests.get(url, stream=True, headers=headers, timeout=30)
    response.raise_for_status()

    total_size = int(response.headers.get('content-length', 0))
    
    destination.parent.mkdir(parents=True, exist_ok=True)
    
    with open(destination, 'wb') as file, tqdm(
        desc=destination.name,
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as progress_bar:
        for data in response.iter_content(chunk_size=1024):
            size = file.write(data)
            progress_bar.update(size)
    
    print(f"✓ Downloaded to {destination}")

    # Quick verification: ensure downloaded file is a ZIP archive. Some
    # servers return HTML error pages (Not Acceptable / 406) instead of
    # the requested file; that's why extraction failed earlier.
    import zipfile as _zip
    if not _zip.is_zipfile(destination):
        # Read the start of the file to show a helpful message
        with open(destination, 'rb') as _f:
            start = _f.read(1024)
        # Try to decode printable prefix for the error message
        try:
            prefix = start.decode('utf-8', errors='replace')
        except Exception:
            prefix = str(start[:200])
        raise RuntimeError(
            f"Downloaded file is not a zip archive. The server may have "
            f"returned HTML or an error page. Sample start:\n{prefix}"
        )

def extract_zip(zip_path: Path, extract_to: Path):
    """
    Args:
        zip_path: Path to the zip file
        extract_to: Directory to extract to
    """
    print(f"Extracting {zip_path.name}...")
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    
    print(f"✓ Extracted to {extract_to}")


def verify_dataset(file_path: Path):
    """
    Verify the dataset file and print basic statistics
    
    Args:
        file_path: Path to the dataset file
    """
    print(f"\nVerifying dataset at {file_path}...")
    
    if not file_path.exists():
        print(f"✗ File not found: {file_path}")
        return False
    
    # Read and analyze the file
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    print(f"✓ Dataset verified!")
    print(f"  Total lines: {len(lines):,}")
    
    # Parse first few lines to show format
    print(f"\n📝 Sample data:")
    for i, line in enumerate(lines[:3]):
        parts = line.strip().split('\t')
        if len(parts) >= 2:
            en_text = parts[0]
            fr_text = parts[1]
            print(f"  {i+1}. EN: {en_text}")
            print(f"     FR: {fr_text}")
            print()
    
    # Basic statistics
    en_lengths = []
    fr_lengths = []
    for line in lines[:10000]:  # Sample first 10k for statistics
        parts = line.strip().split('\t')
        if len(parts) >= 2:
            en_lengths.append(len(parts[0].split()))
            fr_lengths.append(len(parts[1].split()))
    
    if en_lengths and fr_lengths:
        print(f"📊 Statistics (first 10,000 sentences):")
        print(f"  Average EN length: {sum(en_lengths)/len(en_lengths):.1f} words")
        print(f"  Average FR length: {sum(fr_lengths)/len(fr_lengths):.1f} words")
        print(f"  Max EN length: {max(en_lengths)} words")
        print(f"  Max FR length: {max(fr_lengths)} words")
    
    return True


def main():
    """Main function to download and prepare the dataset"""
    print("=" * 80)
    print("DOWNLOADING TATOEBA ENGLISH-FRENCH DATASET")
    print("=" * 80)
    print()
    
    # Define paths
    zip_path = config.RAW_DATA_DIR / "fra-eng.zip"
    data_file = config.RAW_DATA_FILE
    
    # Check if dataset already exists
    if data_file.exists():
        print(f"✓ Dataset already exists at {data_file}")
        user_input = input("Do you want to re-download? (y/N): ")
        if user_input.lower() != 'y':
            print("Using existing dataset.")
            verify_dataset(data_file)
            return
    
    try:
        # Download the zip file
        download_file(config.DATASET_URL, zip_path)
        
        # Extract the zip file
        extract_zip(zip_path, config.RAW_DATA_DIR)
        
        # Verify the dataset
        verify_dataset(data_file)
        
        # Clean up: remove the zip file
        if zip_path.exists():
            zip_path.unlink()
            print(f"\n✓ Cleaned up: removed {zip_path.name}")
        
        print("\n" + "=" * 80)
        print("✓ DATASET DOWNLOAD COMPLETE!")
        print("=" * 80)
        print(f"\n📁 Dataset location: {data_file}")
        print(f"📝 You can now run: python utils/data_loader.py")
        print()
        
    except Exception as e:
        print(f"\n✗ Error occurred: {e}")
        print("Please check your internet connection and try again.")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
