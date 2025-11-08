import os
from pathlib import Path

# ============================================================================
# PATHS
# ============================================================================
ROOT_DIR = Path(__file__).parent
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODEL_DIR = ROOT_DIR / "saved_models"
CHECKPOINT_DIR = ROOT_DIR / "checkpoints"
LOGS_DIR = ROOT_DIR / "logs"
VISUALIZATION_DIR = ROOT_DIR / "visualizations"

# Create directories if they don't exist
for dir_path in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, MODEL_DIR, 
                 CHECKPOINT_DIR, LOGS_DIR, VISUALIZATION_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# ============================================================================
# DATA CONFIGURATION
# ============================================================================
# Dataset URL
DATASET_URL = "http://www.manythings.org/anki/fra-eng.zip"
RAW_DATA_FILE = RAW_DATA_DIR / "fra.txt"
