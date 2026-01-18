"""
LSTM Encoder-Decoder with Bahdanau Attention

Key Differences from Baseline:
1. Encoder returns ALL hidden states (not just final)
2. Decoder uses attention at EVERY timestep
3. Dynamic context vector computed for each output word
4. Much better performance on longer sentences
"""

import torch
import torch.nn as nn
import config
from models.attention_layer import BahdanauAttention

class AttentionEncoder(nn.Module):
    """
    Bidirectional LSTM Encoder with Attention Support. Returns all hidden states for attention mechanism
    """
    def __init__(self, vocab_size: int, embedding_dim: int, hidden_dim: int,
                 num_layers: int = 2, dropout: float = 0.3, bidirectional: bool = True):
        """
        Initialize encoder
        
        Args:
            vocab_size: Size of vocabulary
            embedding_dim: Dimension of word embeddings
            hidden_dim: Dimension of LSTM hidden states
            num_layers: Number of LSTM layers
            dropout: Dropout probability
            bidirectional: Use bidirectional LSTM
        """
        super(AttentionEncoder, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        
        # Word embedding
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=config.PAD_IDX)
        
        # Bidirectional LSTM
        # If bidirectional, each direction has hidden_dim//2 units, so concatenated output is hidden_dim
        lstm_hidden_dim = hidden_dim // self.num_directions
        
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=lstm_hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True
        )
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, src: torch.Tensor, src_lengths: torch.Tensor = None):
        """
        Forward pass
        
        Args:
            src: Source sequences (batch_size, src_len)
            src_lengths: Actual lengths (batch_size,)
            
        Returns:
            outputs: All hidden states (batch_size, src_len, hidden_dim)
            hidden: Final hidden state (num_layers, batch_size, hidden_dim)
            cell: Final cell state (num_layers, batch_size, hidden_dim)
        """
        # Embed
        embedded = self.embedding(src)  # (batch_size, src_len, embedding_dim)
        embedded = self.dropout(embedded)
        
        # Pack if lengths provided ()
        # Packing compresses the batch so the LSTM only processes real words, skipping the padding. 
        # This speeds up training and improves accuracy
        if src_lengths is not None:
            embedded = nn.utils.rnn.pack_padded_sequence(
                embedded, src_lengths.cpu(), batch_first=True, enforce_sorted=False
            )
        
        # CRITICAL: return_sequences=True (we need ALL hidden states)
        outputs, (hidden, cell) = self.lstm(embedded)
        
        # Unpack if we packed
        if src_lengths is not None:
            outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)
        
        # outputs: (batch_size, src_len, hidden_dim) (this holds the context of every single word)
        # hidden: (num_layers * num_directions, batch_size, lstm_hidden_dim)
        # cell: (num_layers * num_directions, batch_size, lstm_hidden_dim)
        
        # If bidirectional, concatenate forward and backward hidden states
        if self.bidirectional:
            # Reshape hidden and cell states
            # From: (num_layers * 2, batch_size, hidden_dim // 2)
            # To: (num_layers, batch_size, hidden_dim)
            hidden = self._concat_bidirectional(hidden)
            cell = self._concat_bidirectional(cell)
        
        return outputs, hidden, cell
    
    def _concat_bidirectional(self, state: torch.Tensor) -> torch.Tensor:
        """
        Concatenate forward and backward states
        
        Args:
            state: (num_layers * 2, batch_size, hidden_dim // 2)
            
        Returns:
            concatenated: (num_layers, batch_size, hidden_dim)
        """
        # state shape: (num_layers * 2, batch_size, hidden_dim // 2)
        # Reshape to (num_layers, 2, batch_size, hidden_dim // 2)
        state = state.view(self.num_layers, 2, -1, self.hidden_dim // 2)
        # Concatenate forward and backward: (num_layers, batch_size, hidden_dim)
        state = torch.cat([state[:, 0, :, :], state[:, 1, :, :]], dim=2)
        return state
    
class AttentionDecoder(nn.Module):
    """
    LSTM Decoder with Bahdanau Attention
    
    At each timestep:
    1. Compute attention context from encoder outputs
    2. Concatenate context with input embedding
    3. Feed through LSTM
    4. Predict next token
    """

    def __init__(self, vocab_size: int, embedding_dim: int, hidden_dim: int,
                 attention_dim: int, num_layers: int = 2, dropout: float = 0.3):
        """
        Initialize decoder with attention
        
        Args:
            vocab_size: Size of target vocabulary
            embedding_dim: Dimension of word embeddings
            hidden_dim: Dimension of LSTM hidden states
            attention_dim: Dimension of attention mechanism
            num_layers: Number of LSTM layers
            dropout: Dropout probability
        """
        super(AttentionDecoder, self).__init__()
        
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Word embedding
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=config.PAD_IDX)
        
        # Attention mechanism
        self.attention = BahdanauAttention(hidden_dim, attention_dim)
        
        # LSTM input is: embedding + context vector
        self.lstm = nn.LSTM(
            input_size=embedding_dim + hidden_dim,  # concatenated input of embedding dimension and context vector
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # Output layer
        # Input is: LSTM output (current decoder) + context vector + embedding
        self.fc_out = nn.Linear(hidden_dim + hidden_dim + embedding_dim, vocab_size)
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, input: torch.Tensor, hidden: torch.Tensor, cell: torch.Tensor,
                encoder_outputs: torch.Tensor, mask: torch.Tensor = None):
        """
        Forward pass for one timestep (one word)
        
        Args:
            input: Current input token (batch_size, 1)
            hidden: Previous hidden state (num_layers, batch_size, hidden_dim)
            cell: Previous cell state (num_layers, batch_size, hidden_dim)
            encoder_outputs: All encoder hidden states (batch_size, src_len, hidden_dim)
            mask: Mask for padded positions (batch_size, src_len)
            
        Returns:
            output: Predictions (batch_size, vocab_size)
            hidden: New hidden state (num_layers, batch_size, hidden_dim)
            cell: New cell state (num_layers, batch_size, hidden_dim)
            attention_weights: Attention weights (batch_size, src_len)
        """
        # input: (batch_size, 1)
        
        # Embed input token
        embedded = self.embedding(input)  # (batch_size, 1, embedding_dim)
        embedded = self.dropout(embedded)
        
        # Compute attention context
        # Use the top layer's hidden state for attention
        decoder_hidden_for_attention = hidden[-1]  # (batch_size, hidden_dim)
        
        # Get context vector and attention weights
        context, attention_weights = self.attention(
            decoder_hidden_for_attention, encoder_outputs, mask
        )
        # context: (batch_size, hidden_dim)
        # attention_weights: (batch_size, src_len)
        
        # Concatenate embedding and context
        context = context.unsqueeze(1)  # (batch_size, 1, hidden_dim)
        lstm_input = torch.cat([embedded, context], dim=2)
        # lstm_input: (batch_size, 1, embedding_dim + hidden_dim)
        
        # Pass through LSTM
        lstm_output, (hidden, cell) = self.lstm(lstm_input, (hidden, cell))
        # lstm_output: (batch_size, 1, hidden_dim)
        
        # Remove sequence dimension
        embedded = embedded.squeeze(1)  # (batch_size, embedding_dim)
        lstm_output = lstm_output.squeeze(1)  # (batch_size, hidden_dim)
        context = context.squeeze(1)  # (batch_size, hidden_dim)
        
        # Concatenate LSTM output, context, and embedding for final prediction
        prediction_input = torch.cat([lstm_output, context, embedded], dim=1)
        # prediction_input: (batch_size, hidden_dim + hidden_dim + embedding_dim)
        
        # Project to vocabulary
        output = self.fc_out(prediction_input)  # (batch_size, vocab_size)
        
        return output, hidden, cell, attention_weights

    

