"""
Configuration file for Neural Machine Translation Project
CPU-optimized version with reduced parameters
"""

import os
from pathlib import Path
import torch

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

# Preprocessing - REDUCED FOR CPU
MAX_LENGTH = 10  # Reduced from 20 for faster training
MIN_LENGTH = 3
VOCAB_SIZE_EN = 8000  # Reduced from 10000
VOCAB_SIZE_FR = 8000  # Reduced from 10000

# Special tokens
PAD_TOKEN = "<pad>"
SOS_TOKEN = "<sos>"
EOS_TOKEN = "<eos>"
UNK_TOKEN = "<unk>"

PAD_IDX = 0
SOS_IDX = 1
EOS_IDX = 2
UNK_IDX = 3

# Train/Val/Test split
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1

# ============================================================================
# MODEL HYPERPARAMETERS - CPU OPTIMIZED
# ============================================================================

# Common parameters - REDUCED FOR CPU
EMBEDDING_DIM = 128  # Reduced from 256
HIDDEN_DIM = 128     # Reduced from 512
ENCODER_LAYERS = 1   # Reduced from 2
DECODER_LAYERS = 1   # Reduced from 2
DROPOUT = 0.3

# Attention-specific parameters
ATTENTION_DIM = 256  # Reduced from 512

# ============================================================================
# TRAINING CONFIGURATION - CPU OPTIMIZED
# ============================================================================

# Training parameters - OPTIMIZED FOR CPU
BATCH_SIZE = 16      # Reduced from 64 for CPU
LEARNING_RATE = 0.001
NUM_EPOCHS = 5      # Reduced from 20 for faster initial training
GRADIENT_CLIP = 1.0
TEACHER_FORCING_RATIO = 0.5

# Optimizer
OPTIMIZER = "adam"
WEIGHT_DECAY = 1e-5

# Learning rate scheduler
USE_LR_SCHEDULER = True
LR_SCHEDULER_PATIENCE = 2  # Reduced from 3
LR_SCHEDULER_FACTOR = 0.5

# Early stopping
EARLY_STOPPING_PATIENCE = 3  # Reduced from 5

# ============================================================================
# DEVICE CONFIGURATION - FORCE CPU
# ============================================================================
# Force CPU for this configuration
DEVICE = torch.device("cpu")
NUM_WORKERS = 0  # Set to 0 for CPU to avoid multiprocessing issues on Windows

# ============================================================================
# LOGGING & CHECKPOINTING
# ============================================================================
LOG_INTERVAL = 50    # Reduced from 100
SAVE_INTERVAL = 1
VISUALIZE_ATTENTION_INTERVAL = 5

# ============================================================================
# EVALUATION CONFIGURATION
# ============================================================================
BEAM_SIZE = 3        # Reduced from 5
MAX_DECODE_LENGTH = 30  # Reduced from 50
BLEU_METRICS = ['bleu1', 'bleu2', 'bleu3', 'bleu4']

# ============================================================================
# API CONFIGURATION
# ============================================================================
API_HOST = "0.0.0.0"
API_PORT = 8000
API_WORKERS = 1

# Model paths for API
BASELINE_MODEL_PATH = MODEL_DIR / "baseline_lstm.pth"
ATTENTION_MODEL_PATH = MODEL_DIR / "attention_lstm.pth"
TOKENIZER_EN_PATH = PROCESSED_DATA_DIR / "tokenizer_en.pkl"
TOKENIZER_FR_PATH = PROCESSED_DATA_DIR / "tokenizer_fr.pkl"

# ============================================================================
# STREAMLIT CONFIGURATION
# ============================================================================
STREAMLIT_PORT = 8501
API_ENDPOINT = f"http://localhost:{API_PORT}"

# ============================================================================
# MODEL-SPECIFIC CONFIGURATIONS
# ============================================================================

class BaselineConfig:
    """Configuration for baseline LSTM encoder-decoder"""
    name = "baseline_lstm"
    embedding_dim = EMBEDDING_DIM
    hidden_dim = HIDDEN_DIM
    encoder_layers = ENCODER_LAYERS
    decoder_layers = DECODER_LAYERS
    dropout = DROPOUT
    use_bidirectional = False
    
    def __repr__(self):
        return f"BaselineConfig(hidden={self.hidden_dim}, layers={self.encoder_layers})"


class AttentionConfig:
    """Configuration for LSTM with Bahdanau attention"""
    name = "attention_lstm"
    embedding_dim = EMBEDDING_DIM
    hidden_dim = HIDDEN_DIM
    encoder_layers = ENCODER_LAYERS
    decoder_layers = DECODER_LAYERS
    dropout = DROPOUT
    attention_dim = ATTENTION_DIM
    use_bidirectional = True
    attention_type = "bahdanau"
    
    def __repr__(self):
        return f"AttentionConfig(hidden={self.hidden_dim}, attention={self.attention_dim})"


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_model_config(model_type: str):
    """Get configuration for specified model type"""
    if model_type.lower() == "baseline":
        return BaselineConfig()
    elif model_type.lower() == "attention":
        return AttentionConfig()
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def print_config():
    """Print all configuration parameters"""
    print("=" * 80)
    print("NEURAL MACHINE TRANSLATION - CONFIGURATION (CPU OPTIMIZED)")
    print("=" * 80)
    print(f"\n📁 PATHS:")
    print(f"  Root Dir: {ROOT_DIR}")
    print(f"  Data Dir: {DATA_DIR}")
    print(f"  Model Dir: {MODEL_DIR}")
    
    print(f"\n📊 DATA:")
    print(f"  Max Length: {MAX_LENGTH} (reduced for CPU)")
    print(f"  Vocab Size (EN): {VOCAB_SIZE_EN}")
    print(f"  Vocab Size (FR): {VOCAB_SIZE_FR}")
    
    print(f"\n🏗️ MODEL:")
    print(f"  Embedding Dim: {EMBEDDING_DIM} (reduced for CPU)")
    print(f"  Hidden Dim: {HIDDEN_DIM} (reduced for CPU)")
    print(f"  Encoder Layers: {ENCODER_LAYERS} (reduced for CPU)")
    print(f"  Decoder Layers: {DECODER_LAYERS} (reduced for CPU)")
    
    print(f"\n🎯 TRAINING:")
    print(f"  Batch Size: {BATCH_SIZE} (reduced for CPU)")
    print(f"  Learning Rate: {LEARNING_RATE}")
    print(f"  Epochs: {NUM_EPOCHS} (reduced for faster training)")
    print(f"  Device: {DEVICE}")
    print(f"  Num Workers: {NUM_WORKERS}")
    
    print(f"\n⚡ CPU OPTIMIZATIONS:")
    print(f"  ✓ Reduced model size (256 vs 512 hidden)")
    print(f"  ✓ Reduced layers (1 vs 2)")
    print(f"  ✓ Smaller batches (32 vs 64)")
    print(f"  ✓ No multiprocessing (workers=0)")
    print(f"  ✓ Shorter sequences (max_len=15 vs 20)")
    
    print("=" * 80)


if __name__ == "__main__":
    print_config()