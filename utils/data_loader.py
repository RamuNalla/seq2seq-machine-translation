import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import pandas as pd
import numpy as np
from typing import List, Tuple, Dict
from pathlib import Path
import pickle

import config
from utils.tokenizer import Tokenizer, create_tokenizers

class TranslationDataset(Dataset):
    """
    PyTorch Dataset for translation pairs
    """
    
    def __init__(self, en_sentences: List[str], fr_sentences: List[str],
                 tokenizer_en: Tokenizer, tokenizer_fr: Tokenizer,
                 max_length: int = config.MAX_LENGTH):
        """
        Initialize dataset
        
        Args:
            en_sentences: List of English sentences
            fr_sentences: List of French sentences
            tokenizer_en: English tokenizer
            tokenizer_fr: French tokenizer
            max_length: Maximum sequence length
        """
        self.tokenizer_en = tokenizer_en
        self.tokenizer_fr = tokenizer_fr
        self.max_length = max_length
        
        # Filter by length and encode
        self.pairs = []
        for en, fr in zip(en_sentences, fr_sentences):
            # Encode sequences
            en_indices = tokenizer_en.encode(en, add_sos=False, add_eos=True)
            fr_indices = tokenizer_fr.encode(fr, add_sos=True, add_eos=True)
            
            # Filter by length
            if len(en_indices) <= max_length and len(fr_indices) <= max_length:
                if len(en_indices) >= config.MIN_LENGTH and len(fr_indices) >= config.MIN_LENGTH:
                    self.pairs.append((en_indices, fr_indices))
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        en_indices, fr_indices = self.pairs[idx]
        return torch.tensor(en_indices, dtype=torch.long), torch.tensor(fr_indices, dtype=torch.long)
