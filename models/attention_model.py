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
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
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
    
class Seq2SeqAttention(nn.Module):
    """
    Complete Seq2Seq model with attention
    
    KEY IMPROVEMENT:
    Decoder receives ALL encoder hidden states and can dynamically
    attend to relevant parts of the input at each timestep.
    """
    
    def __init__(self, encoder: AttentionEncoder, decoder: AttentionDecoder, device: torch.device):
        """
        Initialize Seq2Seq with attention
        
        Args:
            encoder: AttentionEncoder module
            decoder: AttentionDecoder module
            device: Device to run on
        """
        super(Seq2SeqAttention, self).__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def create_mask(self, src: torch.Tensor) -> torch.Tensor:
        """
        Create mask for padded positions
        
        Args:
            src: Source sequences (batch_size, src_len)
            
        Returns:
            mask: (batch_size, src_len) with 1 for valid, 0 for padded
        """
        mask = (src != config.PAD_IDX).float()
        return mask

    def forward(self, src: torch.Tensor, tgt: torch.Tensor,
                src_lengths: torch.Tensor = None,
                teacher_forcing_ratio: float = 0.5):
        """
        Forward pass
        
        Args:
            src: Source sequences (batch_size, src_len)
            tgt: Target sequences (batch_size, tgt_len)
            src_lengths: Source sequence lengths
            teacher_forcing_ratio: Probability of using teacher forcing
            
        Returns:
            outputs: Predictions (batch_size, tgt_len, vocab_size)
            attention_weights: All attention weights (batch_size, tgt_len, src_len)
        """
        batch_size = src.size(0)
        tgt_len = tgt.size(1)
        vocab_size = self.decoder.vocab_size
        
        # Tensor to store outputs and attention weights
        outputs = torch.zeros(batch_size, tgt_len, vocab_size).to(self.device)
        attention_weights_all = torch.zeros(batch_size, tgt_len, src.size(1)).to(self.device)
        
        # Create mask for padding
        mask = self.create_mask(src)
        
        # Encode entire source sequence
        # CRITICAL: We get ALL hidden states
        encoder_outputs, hidden, cell = self.encoder(src, src_lengths)
        
        # First input is SOS token
        input = tgt[:, 0].unsqueeze(1)
        
        # Decode one token at a time
        for t in range(1, tgt_len):
            # Pass through decoder with attention
            output, hidden, cell, attention_weights = self.decoder(
                input, hidden, cell, encoder_outputs, mask
            )
            
            # Store predictions and attention weights
            outputs[:, t, :] = output
            attention_weights_all[:, t, :] = attention_weights
            
            # Teacher forcing
            teacher_force = torch.rand(1).item() < teacher_forcing_ratio
            
            if teacher_force:
                input = tgt[:, t].unsqueeze(1)
            else:
                input = output.argmax(1).unsqueeze(1)
        
        return outputs, attention_weights_all
    
    def translate(self, src: torch.Tensor, src_lengths: torch.Tensor = None,
                  max_length: int = 50):
        """
        Translate with attention (inference mode)
        
        Args:
            src: Source sequence (batch_size, src_len)
            src_lengths: Source sequence lengths
            max_length: Maximum length to generate
            
        Returns:
            translations: Generated sequences (batch_size, max_len)
            attention_weights: Attention weights (batch_size, max_len, src_len)
        """
        self.eval()
        batch_size = src.size(0)
        
        with torch.no_grad():
            # Create mask
            mask = self.create_mask(src)
            
            # Encode
            encoder_outputs, hidden, cell = self.encoder(src, src_lengths)
            
            # Start with SOS
            input = torch.full((batch_size, 1), config.SOS_IDX, dtype=torch.long).to(self.device)
            
            # Store translations and attention
            translations = [input]
            attention_weights_list = []
            
            # Generate tokens
            for _ in range(max_length):
                output, hidden, cell, attention_weights = self.decoder(
                    input, hidden, cell, encoder_outputs, mask
                )
                
                # Store attention
                attention_weights_list.append(attention_weights.unsqueeze(1))
                
                # Get predicted token
                predicted = output.argmax(1).unsqueeze(1)
                translations.append(predicted)
                
                # Stop if all sequences generated EOS
                if (predicted == config.EOS_IDX).all():
                    break
                
                input = predicted
            
            # Concatenate
            translations = torch.cat(translations, dim=1)
            if attention_weights_list:
                attention_weights_all = torch.cat(attention_weights_list, dim=1)
            else:
                attention_weights_all = None
        
        return translations, attention_weights_all
    
def create_attention_model(src_vocab_size: int, tgt_vocab_size: int,
                           device: torch.device) -> Seq2SeqAttention:
    """
    Factory function to create attention model
    
    Args:
        src_vocab_size: Source vocabulary size
        tgt_vocab_size: Target vocabulary size
        device: Device to run on
        
    Returns:
        Seq2SeqAttention model
    """
    model_config = config.AttentionConfig()
    
    encoder = AttentionEncoder(
        vocab_size=src_vocab_size,
        embedding_dim=model_config.embedding_dim,
        hidden_dim=model_config.hidden_dim,
        num_layers=model_config.encoder_layers,
        dropout=model_config.dropout,
        bidirectional=model_config.use_bidirectional
    )
    
    decoder = AttentionDecoder(
        vocab_size=tgt_vocab_size,
        embedding_dim=model_config.embedding_dim,
        hidden_dim=model_config.hidden_dim,
        attention_dim=model_config.attention_dim,
        num_layers=model_config.decoder_layers,
        dropout=model_config.dropout
    )
    
    model = Seq2SeqAttention(encoder, decoder, device).to(device)
    
    return model

# Test the model
if __name__ == "__main__":
    print("=" * 80)
    print("TESTING ATTENTION-BASED MODEL")
    print("=" * 80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    src_vocab_size = 5000
    tgt_vocab_size = 5000
    batch_size = 4
    src_len = 10
    tgt_len = 12
    
    # Create model
    model = create_attention_model(src_vocab_size, tgt_vocab_size, device)
    print(f"\nModel created on {device}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create dummy data
    src = torch.randint(0, src_vocab_size, (batch_size, src_len)).to(device)
    tgt = torch.randint(0, tgt_vocab_size, (batch_size, tgt_len)).to(device)
    src_lengths = torch.full((batch_size,), src_len, dtype=torch.long)
    
    print(f"\nInput shapes:")
    print(f"  Source: {src.shape}")
    print(f"  Target: {tgt.shape}")
    
    # Forward pass
    outputs, attention_weights = model(src, tgt, src_lengths, teacher_forcing_ratio=0.5)
    print(f"\nOutput shape: {outputs.shape}")
    print(f"Attention weights shape: {attention_weights.shape}")
    print(f"Expected attention: (batch={batch_size}, tgt_len={tgt_len}, src_len={src_len})")
    
    # Test translation
    translations, attn = model.translate(src, src_lengths, max_length=20)
    print(f"\nTranslation shape: {translations.shape}")
    if attn is not None:
        print(f"Translation attention shape: {attn.shape}")
    
    print("\n" + "=" * 80)
    print("✓ ATTENTION MODEL WORKING CORRECTLY")
    print("=" * 80)

    

