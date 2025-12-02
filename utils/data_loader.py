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


def collate_fn(batch):
    """
    Custom collate function to pad sequences in a batch
    
    Args:
        batch: List of (source, target) tuples
        
    Returns:
        Padded source and target tensors with lengths
    """
    src_batch, tgt_batch = zip(*batch)
    
    # Pad sequences
    src_padded = pad_sequence(src_batch, batch_first=True, padding_value=config.PAD_IDX)
    tgt_padded = pad_sequence(tgt_batch, batch_first=True, padding_value=config.PAD_IDX)
    
    # Get lengths (before padding)
    src_lengths = torch.tensor([len(seq) for seq in src_batch], dtype=torch.long)
    tgt_lengths = torch.tensor([len(seq) for seq in tgt_batch], dtype=torch.long)
    
    return src_padded, tgt_padded, src_lengths, tgt_lengths

def load_raw_data(file_path: Path) -> Tuple[List[str], List[str]]:
    """
    Load raw data from Tatoeba file
    
    Args:
        file_path: Path to fra.txt
        
    Returns:
        Tuple of (English sentences, French sentences)
    """
    print(f"Loading data from {file_path}...")
    
    en_sentences = []
    fr_sentences = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                en_sentences.append(parts[0])
                fr_sentences.append(parts[1])
    
    print(f"✓ Loaded {len(en_sentences):,} sentence pairs")
    return en_sentences, fr_sentences
