"""
Model service for loading and running inference
Handles model loading, caching, and translation
"""

import torch
import numpy as np
from typing import Dict, Optional, List
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

import config
from utils.tokenizer import Tokenizer
from models.encoder_decoder import create_baseline_model
from models.attention_model import create_attention_model


class ModelService:
    """
    Service for model loading and inference
    Manages both baseline and attention models
    """
    
    def __init__(self):
        """Initialize model service"""
        self.device = config.DEVICE
        self.tokenizer_en = None
        self.tokenizer_fr = None
        self.baseline_model = None
        self.attention_model = None
        
        print(f"Model service initialized on device: {self.device}")
    
    def load_models(self):
        """Load tokenizers and models from saved files"""
        print("Loading models...")
        
        # Load tokenizers
        try:
            self.tokenizer_en = Tokenizer.load(config.TOKENIZER_EN_PATH)
            self.tokenizer_fr = Tokenizer.load(config.TOKENIZER_FR_PATH)
            print(f"✓ Loaded tokenizers")
            print(f"  English vocab size: {len(self.tokenizer_en)}")
            print(f"  French vocab size: {len(self.tokenizer_fr)}")
        except Exception as e:
            print(f"✗ Error loading tokenizers: {e}")
            raise
        
        src_vocab_size = len(self.tokenizer_en)
        tgt_vocab_size = len(self.tokenizer_fr)
        
        # Load baseline model
        if config.BASELINE_MODEL_PATH.exists():
            try:
                self.baseline_model = create_baseline_model(
                    src_vocab_size, tgt_vocab_size, self.device
                )
                checkpoint = torch.load(
                    config.BASELINE_MODEL_PATH, 
                    map_location=self.device,
                    weights_only=False
                )
                self.baseline_model.load_state_dict(checkpoint['model_state_dict'])
                self.baseline_model.eval()
                
                params = sum(p.numel() for p in self.baseline_model.parameters())
                print(f"✓ Loaded baseline model ({params:,} parameters)")
            except Exception as e:
                print(f"⚠ Could not load baseline model: {e}")
                self.baseline_model = None
        else:
            print(f"⚠ Baseline model not found at {config.BASELINE_MODEL_PATH}")
        
        # Load attention model
        if config.ATTENTION_MODEL_PATH.exists():
            try:
                self.attention_model = create_attention_model(
                    src_vocab_size, tgt_vocab_size, self.device
                )
                checkpoint = torch.load(
                    config.ATTENTION_MODEL_PATH,
                    map_location=self.device,
                    weights_only=False
                )
                self.attention_model.load_state_dict(checkpoint['model_state_dict'])
                self.attention_model.eval()
                
                params = sum(p.numel() for p in self.attention_model.parameters())
                print(f"✓ Loaded attention model ({params:,} parameters)")
            except Exception as e:
                print(f"⚠ Could not load attention model: {e}")
                self.attention_model = None
        else:
            print(f"⚠ Attention model not found at {config.ATTENTION_MODEL_PATH}")
        
        if not self.baseline_model and not self.attention_model:
            raise RuntimeError("No models could be loaded!")
        
        print("✓ Model service ready")
    
    def translate(self, text: str, model_type: str = 'attention') -> Dict:
        """
        Translate English text to French
        
        Args:
            text: Source text in English
            model_type: 'baseline' or 'attention'
            
        Returns:
            Dictionary with translation and optionally attention weights
        """
        # Validate input
        if not text or not text.strip():
            raise ValueError("Input text cannot be empty")
        
        # Select model
        if model_type == 'attention':
            model = self.attention_model
        elif model_type == 'baseline':
            model = self.baseline_model
        else:
            raise ValueError(f"Invalid model type: {model_type}")
        
        if model is None:
            raise ValueError(f"{model_type} model not loaded")
        
        # Encode source text
        src_indices = self.tokenizer_en.encode(text, add_eos=True)
        src_tensor = torch.tensor([src_indices], dtype=torch.long).to(self.device)
        src_len = torch.tensor([len(src_indices)], dtype=torch.long)
        
        # Translate
        with torch.no_grad():
            if model_type == 'attention':
                translation, attention_weights = model.translate(
                    src_tensor, src_len, max_length=config.MAX_DECODE_LENGTH
                )
            else:
                translation = model.translate(
                    src_tensor, src_len, max_length=config.MAX_DECODE_LENGTH
                )
                attention_weights = None
        
        # Decode translation
        translation_indices = translation[0].cpu().tolist()
        translation_text = self.tokenizer_fr.decode(translation_indices, remove_special=True)
        
        # Prepare result
        result = {
            'translation': translation_text,
            'source': text,
            'model_type': model_type
        }
        
        # Add attention weights if available
        if attention_weights is not None:
            # Get tokens for visualization
            src_tokens = self.tokenizer_en.tokenize(text) + ['<eos>']
            tgt_tokens = translation_text.split()
            
            if len(tgt_tokens) > 0:
                # Extract and trim attention weights
                attn = attention_weights[0].cpu().numpy()
                attn = attn[:len(tgt_tokens), :len(src_tokens)]
                
                result['attention_weights'] = attn.tolist()
                result['source_tokens'] = src_tokens
                result['target_tokens'] = tgt_tokens
        
        return result
    
    def batch_translate(self, texts: List[str], model_type: str = 'attention') -> List[Dict]:
        """
        Translate multiple texts
        
        Args:
            texts: List of source texts
            model_type: 'baseline' or 'attention'
            
        Returns:
            List of translation results
        """
        results = []
        for text in texts:
            try:
                result = self.translate(text, model_type)
                results.append(result)
            except Exception as e:
                results.append({
                    'error': str(e),
                    'source': text,
                    'translation': None
                })
        return results
    
    def get_model_info(self, model_type: str) -> Optional[Dict]:
        """
        Get information about a model
        
        Args:
            model_type: 'baseline' or 'attention'
            
        Returns:
            Dictionary with model information or None
        """
        model = self.attention_model if model_type == 'attention' else self.baseline_model
        
        if model is None:
            return None
        
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        return {
            'name': model.__class__.__name__,
            'type': model_type,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'device': str(self.device),
            'vocab_size_en': len(self.tokenizer_en),
            'vocab_size_fr': len(self.tokenizer_fr),
        }
    
    def compare_translations(self, text: str) -> Dict:
        """
        Compare translations from both models
        
        Args:
            text: Source text
            
        Returns:
            Dictionary with both translations
        """
        results = {}
        
        # Translate with baseline
        if self.baseline_model:
            try:
                results['baseline'] = self.translate(text, 'baseline')
            except Exception as e:
                results['baseline'] = {'error': str(e)}
        
        # Translate with attention
        if self.attention_model:
            try:
                results['attention'] = self.translate(text, 'attention')
            except Exception as e:
                results['attention'] = {'error': str(e)}
        
        results['source'] = text
        return results


# Test the service
if __name__ == "__main__":
    print("=" * 80)
    print("TESTING MODEL SERVICE")
    print("=" * 80)
    
    # Create service
    service = ModelService()
    
    # Load models
    try:
        service.load_models()
    except Exception as e:
        print(f"\n✗ Error: {e}")
        print("\nMake sure you have:")
        print("  1. Trained the models (run train.py)")
        print("  2. Processed the data (run utils/data_loader.py)")
        exit(1)
    
    # Test translation
    print("\n" + "=" * 80)
    print("TESTING TRANSLATIONS")
    print("=" * 80)
    
    test_sentences = [
        "Hello, how are you?",
        "I love learning about machine translation.",
        "The weather is beautiful today."
    ]
    
    for sentence in test_sentences:
        print(f"\n📝 Source: {sentence}")
        
        # Try baseline
        if service.baseline_model:
            result = service.translate(sentence, 'baseline')
            print(f"🔵 Baseline: {result['translation']}")
        
        # Try attention
        if service.attention_model:
            result = service.translate(sentence, 'attention')
            print(f"🟣 Attention: {result['translation']}")
            if 'attention_weights' in result:
                print(f"   ✓ Attention weights available: {len(result['target_tokens'])} x {len(result['source_tokens'])}")
    
    # Print model info
    print("\n" + "=" * 80)
    print("MODEL INFORMATION")
    print("=" * 80)
    
    for model_type in ['baseline', 'attention']:
        info = service.get_model_info(model_type)
        if info:
            print(f"\n{model_type.upper()} MODEL:")
            for key, value in info.items():
                print(f"  {key}: {value}")
    
    print("\n" + "=" * 80)
    print("✓ MODEL SERVICE TEST COMPLETE")
    print("=" * 80)