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

# Preprocessing
MAX_LENGTH = 20  # Maximum sentence length (in words)
MIN_LENGTH = 3   # Minimum sentence length
VOCAB_SIZE_EN = 10000  # English vocabulary size
VOCAB_SIZE_FR = 10000  # French vocabulary size

# Special tokens
PAD_TOKEN = "<pad>"
SOS_TOKEN = "<sos>"  # Start of sequence
EOS_TOKEN = "<eos>"  # End of sequence
UNK_TOKEN = "<unk>"  # Unknown token

PAD_IDX = 0
SOS_IDX = 1
EOS_IDX = 2
UNK_IDX = 3

# Train/Val/Test split
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1

# ============================================================================
# MODEL HYPERPARAMETERS
# ============================================================================

# Common parameters for both models
EMBEDDING_DIM = 256
HIDDEN_DIM = 512
ENCODER_LAYERS = 2
DECODER_LAYERS = 2
DROPOUT = 0.3

# Attention-specific parameters
ATTENTION_DIM = 512  # Dimension of attention mechanism