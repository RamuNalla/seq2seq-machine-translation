"""
Custom Tokenizer for NMT Project
Handles tokenization, vocabulary building, and sequence encoding/decoding
"""

import re
import pickle
from collections import Counter
from typing import List, Dict, Tuple
import unicodedata
import config

class Tokenizer:
    """
    Custom tokenizer for machine translation
    Handles word-level tokenization with special tokens
    """

    def __init__(self, vocab_size: int = 10000, language: str = "en"):  # Initialize tokenizer

        """
        Initialize tokenizer
        
        Args:
            vocab_size: Maximum vocabulary size
            language: Language code ('en' or 'fr')
        """
        self.vocab_size = vocab_size
        self.language = language
        
        # Special tokens
        self.pad_token = config.PAD_TOKEN
        self.sos_token = config.SOS_TOKEN
        self.eos_token = config.EOS_TOKEN
        self.unk_token = config.UNK_TOKEN
        
        # Vocabularies
        self.word2idx = {
            self.pad_token: config.PAD_IDX,
            self.sos_token: config.SOS_IDX,
            self.eos_token: config.EOS_IDX,
            self.unk_token: config.UNK_IDX
        }
        self.idx2word = {v: k for k, v in self.word2idx.items()}
        
        # Statistics
        self.word_freq = Counter()

    def normalize_text(self, text: str) -> str:     # Normalize text: lowercase, remove accents, normalize unicode
        
        """
        Args:
            text: Input text
            
        Returns:
            Normalized text
        """
        # Lowercase
        text = text.lower()
        
        # Remove accents (NFD normalization)
        text = unicodedata.normalize('NFD', text)
        text = ''.join(char for char in text if unicodedata.category(char) != 'Mn')
        
        # Add space between words and punctuation
        # Example: "hello." -> "hello ."
        text = re.sub(r"([.!?])", r" \1", text)
        text = re.sub(r"[^a-zA-Z.!?]+", r" ", text)
        
        # Remove extra spaces
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def tokenize(self, text: str) -> List[str]:     # Tokenize text into words
        """
        Args:
            text: Input text
            
        Returns:
            List of tokens
        """
        text = self.normalize_text(text)
        tokens = text.split()
        return tokens
    
    def build_vocabulary(self, sentences: List[str]):       #Build vocabulary from a list of sentences
        """
        Args:
            sentences: List of sentences
        """
        print(f"Building vocabulary for {self.language}...")
        
        # Count word frequencies
        for sentence in sentences:
            tokens = self.tokenize(sentence)
            self.word_freq.update(tokens)
        
        # Get most common words (excluding special tokens)
        most_common = self.word_freq.most_common(self.vocab_size - 4)  # -4 for special tokens
        
        # Build word2idx (special tokens already added in __init__)
        for idx, (word, freq) in enumerate(most_common, start=4):
            self.word2idx[word] = idx
            self.idx2word[idx] = word
        
        print(f"✓ Vocabulary built: {len(self.word2idx)} words")
        print(f"  Most common words: {most_common[:10]}")
    

    def encode(self, text: str, add_sos: bool = False, add_eos: bool = False) -> List[int]:     # Encode text to indices
        """
        Args:
            text: Input text
            add_sos: Whether to add SOS token
            add_eos: Whether to add EOS token
            
        Returns:
            List of indices
        """
        tokens = self.tokenize(text)
        
        # Convert tokens to indices
        indices = []
        if add_sos:
            indices.append(config.SOS_IDX)
        
        for token in tokens:
            idx = self.word2idx.get(token, config.UNK_IDX)
            indices.append(idx)
        
        if add_eos:
            indices.append(config.EOS_IDX)
        
        return indices