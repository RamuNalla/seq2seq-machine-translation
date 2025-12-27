"""
Baseline LSTM Encoder-Decoder Model (WITHOUT Attention)

This model demonstrates the "context vector bottleneck" problem:
- Encoder compresses entire input sentence into fixed-size context vector
- Decoder only sees this context vector at initialization
- Performance degrades on longer sentences
"""

import torch
import torch.nn as nn
import config

class Encoder(nn.Module):
    """
    LSTM Encoder
    Reads input sequence and produces final hidden/cell states as context
    """
    
    def __init__(self, vocab_size: int, embedding_dim: int, hidden_dim: int,
                 num_layers: int = 2, dropout: float = 0.3):
        """
        Initialize encoder
        
        Args:
            vocab_size: Size of vocabulary
            embedding_dim: Dimension of word embeddings
            hidden_dim: Dimension of LSTM hidden states
            num_layers: Number of LSTM layers
            dropout: Dropout probability
        """
        super(Encoder, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Word embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=config.PAD_IDX)
        
        # LSTM layer
        # NOTE: return_sequences=False (we only need final states)
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        self.dropout = nn.Dropout(dropout)