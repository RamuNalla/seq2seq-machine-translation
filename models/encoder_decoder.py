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

    def forward(self, src: torch.Tensor, src_lengths: torch.Tensor = None):
        """
        Forward pass
        
        Args:
            src: Source sequences (batch_size, src_len)
            src_lengths: Actual lengths of sequences (batch_size,)
            
        Returns:
            hidden: Final hidden states (num_layers, batch_size, hidden_dim)
            cell: Final cell states (num_layers, batch_size, hidden_dim)
        """
        # Embed source sequences
        embedded = self.embedding(src)  # (batch_size, src_len, embedding_dim)
        embedded = self.dropout(embedded)
        
        # Pack padded sequences if lengths provided (for efficiency)
        if src_lengths is not None:
            embedded = nn.utils.rnn.pack_padded_sequence(
                embedded, src_lengths.cpu(), batch_first=True, enforce_sorted=False
            )
        
        # Pass through LSTM
        # We only need the final hidden and cell states (the context vector)
        _, (hidden, cell) = self.lstm(embedded)
        
        # hidden: (num_layers, batch_size, hidden_dim)
        # cell: (num_layers, batch_size, hidden_dim)
        
        return hidden, cell

    class Decoder(nn.Module):
    """
    LSTM Decoder
    Takes context vector from encoder and generates target sequence
    """
    
    def __init__(self, vocab_size: int, embedding_dim: int, hidden_dim: int,
                 num_layers: int = 2, dropout: float = 0.3):
        """
        Initialize decoder
        
        Args:
            vocab_size: Size of target vocabulary
            embedding_dim: Dimension of word embeddings
            hidden_dim: Dimension of LSTM hidden states
            num_layers: Number of LSTM layers
            dropout: Dropout probability
        """
        super(Decoder, self).__init__()
        
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Word embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=config.PAD_IDX)
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # Output layer to project to vocabulary
        self.fc_out = nn.Linear(hidden_dim, vocab_size)
        
        self.dropout = nn.Dropout(dropout)

     def forward(self, input: torch.Tensor, hidden: torch.Tensor, cell: torch.Tensor):
        """
        Forward pass for one timestep
        
        Args:
            input: Current input token (batch_size, 1)
            hidden: Previous hidden state (num_layers, batch_size, hidden_dim)
            cell: Previous cell state (num_layers, batch_size, hidden_dim)
            
        Returns:
            output: Predictions for next token (batch_size, vocab_size)
            hidden: New hidden state (num_layers, batch_size, hidden_dim)
            cell: New cell state (num_layers, batch_size, hidden_dim)
        """
        # input: (batch_size, 1) - single token
        
        # Embed input token
        embedded = self.embedding(input)  # (batch_size, 1, embedding_dim)
        embedded = self.dropout(embedded)
        
        # Pass through LSTM with previous hidden/cell states
        output, (hidden, cell) = self.lstm(embedded, (hidden, cell))
        # output: (batch_size, 1, hidden_dim)
        
        # Project to vocabulary
        prediction = self.fc_out(output.squeeze(1))  # (batch_size, vocab_size)
        
        return prediction, hidden, cell