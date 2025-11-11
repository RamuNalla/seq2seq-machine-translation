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
# Dataset URL (use HTTPS to avoid server rejecting non-secure requests)
DATASET_URL = "https://www.manythings.org/anki/fra-eng.zip"
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

# ============================================================================
# TRAINING CONFIGURATION
# ============================================================================

# Training parameters
BATCH_SIZE = 64
LEARNING_RATE = 0.001
NUM_EPOCHS = 20
GRADIENT_CLIP = 1.0  # Gradient clipping threshold
TEACHER_FORCING_RATIO = 0.5  # Probability of using teacher forcing

# Optimizer
OPTIMIZER = "adam"  # Options: 'adam', 'sgd', 'rmsprop'
WEIGHT_DECAY = 1e-5

# Learning rate scheduler
USE_LR_SCHEDULER = True
LR_SCHEDULER_PATIENCE = 3
LR_SCHEDULER_FACTOR = 0.5

# Early stopping
EARLY_STOPPING_PATIENCE = 5


# ============================================================================
# DEVICE CONFIGURATION
# ============================================================================
import torch
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_WORKERS = 4  # For data loading

# ============================================================================
# LOGGING & CHECKPOINTING
# ============================================================================
LOG_INTERVAL = 100  # Log every N batches
SAVE_INTERVAL = 1   # Save checkpoint every N epochs
VISUALIZE_ATTENTION_INTERVAL = 5  # Visualize attention every N epochs


# ============================================================================
# EVALUATION CONFIGURATION
# ============================================================================
BEAM_SIZE = 5  # Beam search width
MAX_DECODE_LENGTH = 50  # Maximum length for generated translations
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
    
    # Model-specific parameters
    use_bidirectional = False  # Unidirectional LSTM
    
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
    
    # Model-specific parameters
    use_bidirectional = True  # Bidirectional encoder for better context
    attention_type = "bahdanau"  # Options: 'bahdanau', 'luong'
    
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
    print("NEURAL MACHINE TRANSLATION - CONFIGURATION")
    print("=" * 80)
    print(f"\n📁 PATHS:")
    print(f"  Root Dir: {ROOT_DIR}")
    print(f"  Data Dir: {DATA_DIR}")
    print(f"  Model Dir: {MODEL_DIR}")
    
    print(f"\n📊 DATA:")
    print(f"  Max Length: {MAX_LENGTH}")
    print(f"  Vocab Size (EN): {VOCAB_SIZE_EN}")
    print(f"  Vocab Size (FR): {VOCAB_SIZE_FR}")
    print(f"  Train/Val/Test: {TRAIN_RATIO}/{VAL_RATIO}/{TEST_RATIO}")
    
    print(f"\n🏗️ MODEL:")
    print(f"  Embedding Dim: {EMBEDDING_DIM}")
    print(f"  Hidden Dim: {HIDDEN_DIM}")
    print(f"  Encoder Layers: {ENCODER_LAYERS}")
    print(f"  Decoder Layers: {DECODER_LAYERS}")
    print(f"  Dropout: {DROPOUT}")
    
    print(f"\n🎯 TRAINING:")
    print(f"  Batch Size: {BATCH_SIZE}")
    print(f"  Learning Rate: {LEARNING_RATE}")
    print(f"  Epochs: {NUM_EPOCHS}")
    print(f"  Device: {DEVICE}")
    print(f"  Teacher Forcing Ratio: {TEACHER_FORCING_RATIO}")
    
    print(f"\n🔧 API:")
    print(f"  Host: {API_HOST}:{API_PORT}")
    print(f"  Streamlit Port: {STREAMLIT_PORT}")
    
    print("=" * 80)


if __name__ == "__main__":
    print_config()