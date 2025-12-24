
import torch
import torch.nn as nn
import torch.nn.functional as F


class BahdanauAttention(nn.Module):
    """
    Bahdanau Attention (Additive Attention)
    
    At each decoder timestep t:
    1. Calculate attention scores for each encoder hidden state
    2. Apply softmax to get attention weights (sum to 1)
    3. Compute weighted sum of encoder states = context vector
    
    Formula:
        score(h_i, s_t) = v^T * tanh(W1 * h_i + W2 * s_t)
        alpha_i = softmax(scores)
        context = sum(alpha_i * h_i)
    """

    def __init__(self, hidden_dim: int, attention_dim: int):
        """
        Initialize attention mechanism
        
        Args:
            hidden_dim: Dimension of encoder/decoder hidden states
            attention_dim: Dimension of attention mechanism
        """
        super(BahdanauAttention, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.attention_dim = attention_dim

        # Linear transformations for attention score calculation
        # W1: transforms encoder hidden states
        self.W_encoder = nn.Linear(hidden_dim, attention_dim, bias=False)
        
        # W2: transforms decoder hidden state
        self.W_decoder = nn.Linear(hidden_dim, attention_dim, bias=False)
        
        # v: final linear layer to compute scalar score
        self.v = nn.Linear(attention_dim, 1, bias=False)

    def forward(self, decoder_hidden: torch.Tensor, encoder_outputs: torch.Tensor,
                mask: torch.Tensor = None) -> tuple:
        """
        Forward pass of attention mechanism

        Args:
            decoder_hidden: Current decoder hidden state
                           Shape: (batch_size, hidden_dim)
            encoder_outputs: All encoder hidden states
                           Shape: (batch_size, src_len, hidden_dim)
            mask: Mask for padded positions
                 Shape: (batch_size, src_len)
                 
        Returns:
            context: Context vector (weighted sum of encoder outputs)
                    Shape: (batch_size, hidden_dim)
            attention_weights: Attention weights for visualization
                             Shape: (batch_size, src_len)
        """

        batch_size = encoder_outputs.size(0)
        src_len = encoder_outputs.size(1)
        
        # Step 1: Transform encoder outputs
        # encoder_outputs: (batch_size, src_len, hidden_dim)
        # transformed_encoder: (batch_size, src_len, attention_dim)
        transformed_encoder = self.W_encoder(encoder_outputs)
        
        # Step 2: Transform decoder hidden state
        # decoder_hidden: (batch_size, hidden_dim)
        # transformed_decoder: (batch_size, attention_dim)
        transformed_decoder = self.W_decoder(decoder_hidden)
        
        # Step 3: Broadcast decoder hidden to match encoder outputs length
        # We need to add transformed_decoder to each position in transformed_encoder
        # transformed_decoder: (batch_size, attention_dim) -> (batch_size, 1, attention_dim)
        transformed_decoder = transformed_decoder.unsqueeze(1)
        
        # Step 4: Calculate attention scores
        # Add transformed encoder and decoder, apply tanh, then linear projection
        # energy: (batch_size, src_len, attention_dim)
        energy = torch.tanh(transformed_encoder + transformed_decoder)
        
        # Project to scalar scores
        # scores: (batch_size, src_len, 1) -> (batch_size, src_len)
        scores = self.v(energy).squeeze(-1)
        
        # Step 5: Apply mask if provided (for padded positions)
        if mask is not None:
            # Set scores for padded positions to very large negative value
            # so softmax gives them near-zero attention
            scores = scores.masked_fill(mask == 0, -1e10)
        
        # Step 6: Apply softmax to get attention weights
        # attention_weights: (batch_size, src_len)
        # These weights sum to 1 and indicate importance of each encoder position
        attention_weights = F.softmax(scores, dim=1)
        
        # Step 7: Compute context vector as weighted sum of encoder outputs
        # attention_weights: (batch_size, src_len) -> (batch_size, 1, src_len)
        # encoder_outputs: (batch_size, src_len, hidden_dim)
        # context: (batch_size, 1, hidden_dim) -> (batch_size, hidden_dim)
        context = torch.bmm(attention_weights.unsqueeze(1), encoder_outputs)
        context = context.squeeze(1)
        
        return context, attention_weights



"""
Bahdanau (Additive) Attention Mechanism
Implements attention in detail to understand how it works

Attention allows the decoder to "look back" at all encoder hidden states
and dynamically weight their importance at each decoding timestep.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BahdanauAttention(nn.Module):
    """
    Bahdanau Attention (Additive Attention)
    
    At each decoder timestep t:
    1. Calculate attention scores for each encoder hidden state
    2. Apply softmax to get attention weights (sum to 1)
    3. Compute weighted sum of encoder states = context vector
    
    Formula:
        score(h_i, s_t) = v^T * tanh(W1 * h_i + W2 * s_t)
        alpha_i = softmax(scores)
        context = sum(alpha_i * h_i)
    """
    
    def __init__(self, hidden_dim: int, attention_dim: int):
        """
        Initialize attention mechanism
        
        Args:
            hidden_dim: Dimension of encoder/decoder hidden states
            attention_dim: Dimension of attention mechanism
        """
        super(BahdanauAttention, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.attention_dim = attention_dim
        
        # Linear transformations for attention score calculation
        # W1: transforms encoder hidden states
        self.W_encoder = nn.Linear(hidden_dim, attention_dim, bias=False)
        
        # W2: transforms decoder hidden state
        self.W_decoder = nn.Linear(hidden_dim, attention_dim, bias=False)
        
        # v: final linear layer to compute scalar score
        self.v = nn.Linear(attention_dim, 1, bias=False)
        
    def forward(self, decoder_hidden: torch.Tensor, encoder_outputs: torch.Tensor,
                mask: torch.Tensor = None) -> tuple:
        """
        Forward pass of attention mechanism
        
        Args:
            decoder_hidden: Current decoder hidden state
                           Shape: (batch_size, hidden_dim)
            encoder_outputs: All encoder hidden states
                           Shape: (batch_size, src_len, hidden_dim)
            mask: Mask for padded positions
                 Shape: (batch_size, src_len)
                 
        Returns:
            context: Context vector (weighted sum of encoder outputs)
                    Shape: (batch_size, hidden_dim)
            attention_weights: Attention weights for visualization
                             Shape: (batch_size, src_len)
        """
        batch_size = encoder_outputs.size(0)
        src_len = encoder_outputs.size(1)
        
        # Step 1: Transform encoder outputs
        # encoder_outputs: (batch_size, src_len, hidden_dim)
        # transformed_encoder: (batch_size, src_len, attention_dim)
        transformed_encoder = self.W_encoder(encoder_outputs)
        
        # Step 2: Transform decoder hidden state
        # decoder_hidden: (batch_size, hidden_dim)
        # transformed_decoder: (batch_size, attention_dim)
        transformed_decoder = self.W_decoder(decoder_hidden)
        
        # Step 3: Broadcast decoder hidden to match encoder outputs length
        # We need to add transformed_decoder to each position in transformed_encoder
        # transformed_decoder: (batch_size, attention_dim) -> (batch_size, 1, attention_dim)
        transformed_decoder = transformed_decoder.unsqueeze(1)
        
        # Step 4: Calculate attention scores
        # Add transformed encoder and decoder, apply tanh, then linear projection
        # energy: (batch_size, src_len, attention_dim)
        energy = torch.tanh(transformed_encoder + transformed_decoder)
        
        # Project to scalar scores
        # scores: (batch_size, src_len, 1) -> (batch_size, src_len)
        scores = self.v(energy).squeeze(-1)
        
        # Step 5: Apply mask if provided (for padded positions)
        if mask is not None:
            # Set scores for padded positions to very large negative value
            # so softmax gives them near-zero attention
            scores = scores.masked_fill(mask == 0, -1e10)
        
        # Step 6: Apply softmax to get attention weights
        # attention_weights: (batch_size, src_len)
        # These weights sum to 1 and indicate importance of each encoder position
        attention_weights = F.softmax(scores, dim=1)
        
        # Step 7: Compute context vector as weighted sum of encoder outputs
        # attention_weights: (batch_size, src_len) -> (batch_size, 1, src_len)
        # encoder_outputs: (batch_size, src_len, hidden_dim)
        # context: (batch_size, 1, hidden_dim) -> (batch_size, hidden_dim)
        context = torch.bmm(attention_weights.unsqueeze(1), encoder_outputs)
        context = context.squeeze(1)
        
        return context, attention_weights


class LuongAttention(nn.Module):
    """
    Luong Attention (Multiplicative Attention)
    
    Alternative attention mechanism (simpler than Bahdanau)
    
    Formula:
        score(h_i, s_t) = h_i^T * W * s_t  (general)
        or score(h_i, s_t) = h_i^T * s_t    (dot)
        alpha_i = softmax(scores)
        context = sum(alpha_i * h_i)
    """
    
    def __init__(self, hidden_dim: int, attention_type: str = "general"):
        """
        Initialize Luong attention
        
        Args:
            hidden_dim: Dimension of hidden states
            attention_type: 'dot' or 'general'
        """
        super(LuongAttention, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.attention_type = attention_type
        
        if attention_type == "general":
            # Use a linear transformation
            self.W = nn.Linear(hidden_dim, hidden_dim, bias=False)
        elif attention_type != "dot":
            raise ValueError(f"Unknown attention type: {attention_type}")
        

    def forward(self, decoder_hidden: torch.Tensor, encoder_outputs: torch.Tensor,
                mask: torch.Tensor = None) -> tuple:
        """
        Forward pass of Luong attention
        
        Args:
            decoder_hidden: (batch_size, hidden_dim)
            encoder_outputs: (batch_size, src_len, hidden_dim)
            mask: (batch_size, src_len)
            
        Returns:
            context: (batch_size, hidden_dim)
            attention_weights: (batch_size, src_len)
        """
        # Transform decoder hidden if using general attention
        if self.attention_type == "general":
            # decoder_hidden: (batch_size, hidden_dim)
            transformed = self.W(decoder_hidden)  # (batch_size, hidden_dim)
            # Add dimension for batch matrix multiplication
            transformed = transformed.unsqueeze(1)  # (batch_size, 1, hidden_dim)
        else:  # dot attention
            transformed = decoder_hidden.unsqueeze(1)  # (batch_size, 1, hidden_dim)
        
        # Calculate scores using batch matrix multiplication
        # transformed: (batch_size, 1, hidden_dim)
        # encoder_outputs: (batch_size, src_len, hidden_dim)
        # scores: (batch_size, 1, src_len) -> (batch_size, src_len)
        scores = torch.bmm(transformed, encoder_outputs.transpose(1, 2)).squeeze(1)
        
        # Apply mask
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e10)
        
        # Softmax to get attention weights
        attention_weights = F.softmax(scores, dim=1)
        
        # Compute context vector
        context = torch.bmm(attention_weights.unsqueeze(1), encoder_outputs).squeeze(1)
        
        return context, attention_weights

def forward(self, decoder_hidden: torch.Tensor, encoder_outputs: torch.Tensor,
                mask: torch.Tensor = None) -> tuple:
        """
        Forward pass of Luong attention
        
        Args:
            decoder_hidden: (batch_size, hidden_dim)
            encoder_outputs: (batch_size, src_len, hidden_dim)
            mask: (batch_size, src_len)
            
        Returns:
            context: (batch_size, hidden_dim)
            attention_weights: (batch_size, src_len)
        """
        # Transform decoder hidden if using general attention
        if self.attention_type == "general":
            # decoder_hidden: (batch_size, hidden_dim)
            transformed = self.W(decoder_hidden)  # (batch_size, hidden_dim)
            # Add dimension for batch matrix multiplication
            transformed = transformed.unsqueeze(1)  # (batch_size, 1, hidden_dim)
        else:  # dot attention
            transformed = decoder_hidden.unsqueeze(1)  # (batch_size, 1, hidden_dim)
        
        # Calculate scores using batch matrix multiplication
        # transformed: (batch_size, 1, hidden_dim)
        # encoder_outputs: (batch_size, src_len, hidden_dim)
        # scores: (batch_size, 1, src_len) -> (batch_size, src_len)
        scores = torch.bmm(transformed, encoder_outputs.transpose(1, 2)).squeeze(1)
        
        # Apply mask
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e10)
        
        # Softmax to get attention weights
        attention_weights = F.softmax(scores, dim=1)
        
        # Compute context vector
        context = torch.bmm(attention_weights.unsqueeze(1), encoder_outputs).squeeze(1)
        
        return context, attention_weights


# Test the attention mechanisms
if __name__ == "__main__":
    print("=" * 80)
    print("TESTING ATTENTION MECHANISMS")
    print("=" * 80)
    
    # Parameters
    batch_size = 4
    src_len = 10
    hidden_dim = 512
    attention_dim = 256
    
    # Create dummy data
    decoder_hidden = torch.randn(batch_size, hidden_dim)
    encoder_outputs = torch.randn(batch_size, src_len, hidden_dim)
    
    # Create mask (simulate that last 3 positions are padding)
    mask = torch.ones(batch_size, src_len)
    mask[:, -3:] = 0
    
    print(f"\nInput shapes:")
    print(f"  Decoder hidden: {decoder_hidden.shape}")
    print(f"  Encoder outputs: {encoder_outputs.shape}")
    print(f"  Mask: {mask.shape}")
    
    # Test Bahdanau Attention
    print("\n" + "-" * 80)
    print("Bahdanau Attention:")
    print("-" * 80)
    bahdanau = BahdanauAttention(hidden_dim, attention_dim)
    context_b, weights_b = bahdanau(decoder_hidden, encoder_outputs, mask)