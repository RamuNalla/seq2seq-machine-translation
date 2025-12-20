
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