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

    def decode(self, indices: List[int], remove_special: bool = True) -> str:
        """
        Decode indices to text
        
        Args:
            indices: List of indices
            remove_special: Whether to remove special tokens
            
        Returns:
            Decoded text
        """
        tokens = []
        for idx in indices:
            # Stop at EOS token
            if idx == config.EOS_IDX and remove_special:
                break
            
            token = self.idx2word.get(idx, self.unk_token)
            
            # Skip special tokens if requested
            if remove_special and token in [self.pad_token, self.sos_token, self.eos_token]:
                continue
            
            tokens.append(token)
        
        text = ' '.join(tokens)
        return text
    
    def batch_encode(self, texts: List[str], add_sos: bool = False, 
                    add_eos: bool = False) -> List[List[int]]:
        """
        Encode a batch of texts
        
        Args:
            texts: List of texts
            add_sos: Whether to add SOS token
            add_eos: Whether to add EOS token
            
        Returns:
            List of encoded sequences
        """
        return [self.encode(text, add_sos, add_eos) for text in texts]
    
    def batch_decode(self, batch_indices: List[List[int]], 
                    remove_special: bool = True) -> List[str]:
        """
        Decode a batch of indices
        
        Args:
            batch_indices: List of index sequences
            remove_special: Whether to remove special tokens
            
        Returns:
            List of decoded texts
        """
        return [self.decode(indices, remove_special) for indices in batch_indices]
    
    def save(self, path: str):
        """Save tokenizer to file"""
        with open(path, 'wb') as f:
            pickle.dump(self, f)
        print(f"✓ Tokenizer saved to {path}")

    @staticmethod
    def load(path: str) -> 'Tokenizer':
        """Load tokenizer from file"""
        with open(path, 'rb') as f:
            tokenizer = pickle.load(f)
        print(f"✓ Tokenizer loaded from {path}")
        return tokenizer
    
    def __len__(self):
        """Return vocabulary size"""
        return len(self.word2idx)
    
    def __repr__(self):
        return f"Tokenizer(vocab_size={len(self.word2idx)}, language={self.language})"
    

def create_tokenizers(en_sentences: List[str], fr_sentences: List[str]) -> Tuple[Tokenizer, Tokenizer]:
    """
    Create and build tokenizers for both languages
    
    Args:
        en_sentences: List of English sentences
        fr_sentences: List of French sentences
        
    Returns:
        Tuple of (English tokenizer, French tokenizer)
    """
    print("=" * 80)
    print("CREATING TOKENIZERS")
    print("=" * 80)
    
    # Create English tokenizer
    tokenizer_en = Tokenizer(vocab_size=config.VOCAB_SIZE_EN, language="en")
    tokenizer_en.build_vocabulary(en_sentences)
    
    print()
    
    # Create French tokenizer
    tokenizer_fr = Tokenizer(vocab_size=config.VOCAB_SIZE_FR, language="fr")
    tokenizer_fr.build_vocabulary(fr_sentences)
    
    print("\n" + "=" * 80)
    print("✓ TOKENIZERS CREATED")
    print("=" * 80)
    
    return tokenizer_en, tokenizer_fr