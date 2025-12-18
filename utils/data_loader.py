import torch
import sys
import os
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import pandas as pd
import numpy as np
from typing import List, Tuple, Dict
from pathlib import Path
import pickle

# Add parent directory to path to import config
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
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


def split_data(en_sentences: List[str], fr_sentences: List[str]) -> Dict[str, Tuple[List[str], List[str]]]:
    """
    Split data into train/val/test sets
    
    Args:
        en_sentences: English sentences
        fr_sentences: French sentences
        
    Returns:
        Dictionary with 'train', 'val', 'test' keys
    """
    print("\nSplitting data...")
    
    total = len(en_sentences)
    indices = np.random.permutation(total)
    
    train_size = int(total * config.TRAIN_RATIO)
    val_size = int(total * config.VAL_RATIO)
    
    train_idx = indices[:train_size]
    val_idx = indices[train_size:train_size + val_size]
    test_idx = indices[train_size + val_size:]
    
    splits = {
        'train': (
            [en_sentences[i] for i in train_idx],
            [fr_sentences[i] for i in train_idx]
        ),
        'val': (
            [en_sentences[i] for i in val_idx],
            [fr_sentences[i] for i in val_idx]
        ),
        'test': (
            [en_sentences[i] for i in test_idx],
            [fr_sentences[i] for i in test_idx]
        )
    }
    
    print(f"✓ Train: {len(train_idx):,} pairs")
    print(f"✓ Val:   {len(val_idx):,} pairs")
    print(f"✓ Test:  {len(test_idx):,} pairs")
    
    return splits


def prepare_data():
    """
    Main function to prepare all data
    Loads raw data, creates tokenizers, and saves processed data
    """
    print("=" * 80)
    print("DATA PREPARATION")
    print("=" * 80)
    
    # Load raw data
    en_sentences, fr_sentences = load_raw_data(config.RAW_DATA_FILE)
    
    # Split data
    splits = split_data(en_sentences, fr_sentences)
    
    # Create tokenizers using training data only
    train_en, train_fr = splits['train']
    tokenizer_en, tokenizer_fr = create_tokenizers(train_en, train_fr)
    
    # Save tokenizers
    tokenizer_en.save(config.TOKENIZER_EN_PATH)
    tokenizer_fr.save(config.TOKENIZER_FR_PATH)
    
    # Create and save datasets
    print("\n" + "=" * 80)
    print("CREATING DATASETS")
    print("=" * 80)
    
    datasets = {}
    for split_name, (en_sents, fr_sents) in splits.items():
        print(f"\nCreating {split_name} dataset...")
        dataset = TranslationDataset(en_sents, fr_sents, tokenizer_en, tokenizer_fr)
        datasets[split_name] = dataset
        print(f"✓ {split_name}: {len(dataset):,} pairs (after filtering)")
        
        # Save dataset
        dataset_path = config.PROCESSED_DATA_DIR / f"{split_name}_dataset.pkl"
        with open(dataset_path, 'wb') as f:
            pickle.dump(dataset, f)
        print(f"✓ Saved to {dataset_path}")
    
    # Print sample data
    print("\n" + "=" * 80)
    print("SAMPLE DATA")
    print("=" * 80)
    
    train_dataset = datasets['train']
    for i in range(3):
        src, tgt = train_dataset[i]
        en_text = tokenizer_en.decode(src.tolist())
        fr_text = tokenizer_fr.decode(tgt.tolist())
        print(f"\nPair {i+1}:")
        print(f"  EN: {en_text}")
        print(f"  FR: {fr_text}")
        print(f"  EN indices: {src.tolist()[:10]}...")
        print(f"  FR indices: {tgt.tolist()[:10]}...")
    
    print("\n" + "=" * 80)
    print("✓ DATA PREPARATION COMPLETE")
    print("=" * 80)
    
    return datasets, tokenizer_en, tokenizer_fr


def load_datasets() -> Tuple[Dict[str, Dataset], Tokenizer, Tokenizer]:
    """
    Load preprocessed datasets and tokenizers
    
    Returns:
        Tuple of (datasets dict, English tokenizer, French tokenizer)
    """
    print("Loading preprocessed data...")
    
    # Load tokenizers
    tokenizer_en = Tokenizer.load(config.TOKENIZER_EN_PATH)
    tokenizer_fr = Tokenizer.load(config.TOKENIZER_FR_PATH)
    
    # Load datasets
    datasets = {}
    for split in ['train', 'val', 'test']:
        dataset_path = config.PROCESSED_DATA_DIR / f"{split}_dataset.pkl"
        with open(dataset_path, 'rb') as f:
            datasets[split] = pickle.load(f)
        print(f"✓ Loaded {split}: {len(datasets[split]):,} pairs")
    
    return datasets, tokenizer_en, tokenizer_fr


def create_dataloaders(datasets: Dict[str, Dataset], batch_size: int = config.BATCH_SIZE) -> Dict[str, DataLoader]:
    """
    Create DataLoaders for all splits
    
    Args:
        datasets: Dictionary of datasets
        batch_size: Batch size
        
    Returns:
        Dictionary of DataLoaders
    """
    dataloaders = {
        'train': DataLoader(
            datasets['train'],
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=config.NUM_WORKERS,
            pin_memory=True
        ),
        'val': DataLoader(
            datasets['val'],
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=config.NUM_WORKERS,
            pin_memory=True
        ),
        'test': DataLoader(
            datasets['test'],
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=config.NUM_WORKERS,
            pin_memory=True
        )
    }
    
    return dataloaders

# Main execution
if __name__ == "__main__":
    # Check if data is already processed
    if (config.TOKENIZER_EN_PATH.exists() and 
        config.TOKENIZER_FR_PATH.exists() and
        (config.PROCESSED_DATA_DIR / "train_dataset.pkl").exists()):
        
        print("Preprocessed data already exists!")
        user_input = input("Reprocess data? (y/N): ")
        if user_input.lower() != 'y':
            datasets, tokenizer_en, tokenizer_fr = load_datasets()
            print("\n✓ Loaded existing data")
        else:
            datasets, tokenizer_en, tokenizer_fr = prepare_data()
    else:
        datasets, tokenizer_en, tokenizer_fr = prepare_data()
    
    # Test dataloader
    print("\n" + "=" * 80)
    print("TESTING DATALOADER")
    print("=" * 80)
    
    dataloaders = create_dataloaders(datasets)
    train_loader = dataloaders['train']
    
    # Get one batch
    src, tgt, src_len, tgt_len = next(iter(train_loader))
    print(f"\nBatch shapes:")
    print(f"  Source: {src.shape} (batch_size, max_src_len)")
    print(f"  Target: {tgt.shape} (batch_size, max_tgt_len)")
    print(f"  Source lengths: {src_len}")
    print(f"  Target lengths: {tgt_len}")
    
    print("\n✓ Data loading complete!")
