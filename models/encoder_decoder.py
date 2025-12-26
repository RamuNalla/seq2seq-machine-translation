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