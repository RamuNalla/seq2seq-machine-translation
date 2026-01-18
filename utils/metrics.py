"""
Evaluation metrics for machine translation
Implements BLEU score calculation and model comparison utilities
"""

import torch
from typing import List, Dict, Tuple
from sacrebleu.metrics import BLEU
from tqdm import tqdm
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from utils.tokenizer import Tokenizer


class BLEUScorer:
    """
    BLEU score calculator using sacrebleu
    
    BLEU (Bilingual Evaluation Understudy) measures how similar 
    the machine translation is to reference human translations.
    
    BLEU-n measures n-gram precision:
    - BLEU-1: unigram (single word) precision
    - BLEU-2: bigram (2-word phrase) precision
    - BLEU-3: trigram precision
    - BLEU-4: 4-gram precision (most commonly reported)
    """
    
    def __init__(self):
        """Initialize BLEU scorer"""
        self.bleu = BLEU()
    
    def calculate(self, hypotheses: List[str], references: List[List[str]]) -> Dict[str, float]:
        """
        Calculate BLEU scores
        
        Args:
            hypotheses: List of predicted translations
            references: List of reference translations (each can have multiple refs)
                       Format: [[ref1], [ref2], ...] or [[ref1a, ref1b], [ref2a], ...]
            
        Returns:
            Dictionary with BLEU scores
        """
        # Ensure references are in correct format (list of lists)
        if references and not isinstance(references[0], list):
            references = [[ref] for ref in references]
        
        # Calculate corpus-level BLEU score
        try:
            bleu_score = self.bleu.corpus_score(hypotheses, references)
            
            return {
                'bleu': bleu_score.score,  # Overall BLEU score
                'bleu1': bleu_score.precisions[0],  # Unigram precision
                'bleu2': bleu_score.precisions[1],  # Bigram precision
                'bleu3': bleu_score.precisions[2],  # Trigram precision
                'bleu4': bleu_score.precisions[3],  # 4-gram precision
                'bp': bleu_score.bp,  # Brevity penalty
                'sys_len': bleu_score.sys_len,  # System (hypothesis) length
                'ref_len': bleu_score.ref_len,  # Reference length
            }
        except Exception as e:
            print(f"Error calculating BLEU: {e}")
            return {
                'bleu': 0.0,
                'bleu1': 0.0,
                'bleu2': 0.0,
                'bleu3': 0.0,
                'bleu4': 0.0,
                'bp': 0.0,
                'sys_len': 0,
                'ref_len': 0,
            }
    
    def calculate_sentence_bleu(self, hypothesis: str, reference: str) -> float:
        """
        Calculate BLEU for a single sentence
        
        Args:
            hypothesis: Predicted translation
            reference: Reference translation
            
        Returns:
            BLEU score (0-100)
        """
        score = self.bleu.sentence_score(hypothesis, [reference])
        return score.score

def evaluate_model(model, dataloader, tokenizer_en: Tokenizer, 
                   tokenizer_fr: Tokenizer, device: torch.device,
                   max_samples: int = None) -> Dict[str, float]:
    """
    Evaluate model on a dataset and calculate BLEU scores
    
    Args:
        model: Translation model (baseline or attention)
        dataloader: DataLoader with evaluation data
        tokenizer_en: English tokenizer
        tokenizer_fr: French tokenizer
        device: Device to run on
        max_samples: Maximum number of samples to evaluate (None = all)
        
    Returns:
        Dictionary with BLEU scores and other metrics
    """
    model.eval()
    scorer = BLEUScorer()
    
    all_hypotheses = []  # Model predictions
    all_references = []  # Ground truth
    
    print("Generating translations...")
    
    with torch.no_grad():
        for batch_idx, (src, tgt, src_len, tgt_len) in enumerate(tqdm(dataloader, desc="Evaluating")):
            # Check max samples limit
            if max_samples and len(all_hypotheses) >= max_samples:
                break
            
            src = src.to(device)
            
            # Generate translations
            # Check if model has translate method (for inference)
            if hasattr(model, 'translate'):
                # Use model's translate method
                if 'attention' in model.__class__.__name__.lower():
                    translations, _ = model.translate(src, src_len, max_length=config.MAX_DECODE_LENGTH)
                else:
                    translations = model.translate(src, src_len, max_length=config.MAX_DECODE_LENGTH)
            else:
                # Use forward pass with no teacher forcing
                tgt = tgt.to(device)
                output = model(src, tgt, src_len, teacher_forcing_ratio=0.0)
                translations = output.argmax(dim=-1)
            
            # Decode predictions and references
            for i in range(len(src)):
                # Decode hypothesis (predicted translation)
                hyp_indices = translations[i].cpu().tolist()
                hyp = tokenizer_fr.decode(hyp_indices, remove_special=True)
                
                # Decode reference (ground truth)
                ref_indices = tgt[i].cpu().tolist()
                ref = tokenizer_fr.decode(ref_indices, remove_special=True)
                
                all_hypotheses.append(hyp)
                all_references.append([ref])
    
    print(f"\nEvaluated {len(all_hypotheses)} translations")
    
    # Calculate BLEU scores
    bleu_scores = scorer.calculate(all_hypotheses, all_references)
    
    return bleu_scores

def compare_models(baseline_model, attention_model, dataloader,
                   tokenizer_en: Tokenizer, tokenizer_fr: Tokenizer,
                   device: torch.device, num_samples: int = 100) -> Dict:
    """
    Compare baseline and attention models side-by-side
    
    Args:
        baseline_model: Baseline LSTM model
        attention_model: Attention-based model
        dataloader: DataLoader with test data
        tokenizer_en: English tokenizer
        tokenizer_fr: French tokenizer
        device: Device to run on
        num_samples: Number of samples to compare
        
    Returns:
        Dictionary with comparison results
    """
    print("=" * 80)
    print("COMPARING MODELS")
    print("=" * 80)
    
    # Evaluate baseline model
    print("\n1. Evaluating Baseline LSTM...")
    baseline_scores = evaluate_model(
        baseline_model, dataloader, tokenizer_en, tokenizer_fr, device, num_samples
    )
    
    # Evaluate attention model
    print("\n2. Evaluating LSTM with Attention...")
    attention_scores = evaluate_model(
        attention_model, dataloader, tokenizer_en, tokenizer_fr, device, num_samples
    )
    
    # Print comparison
    print("\n" + "=" * 80)
    print("RESULTS COMPARISON")
    print("=" * 80)
    
    metrics = ['bleu', 'bleu1', 'bleu2', 'bleu3', 'bleu4']
    
    print(f"\n{'Metric':<15} {'Baseline':<15} {'Attention':<15} {'Improvement':<15}")
    print("-" * 60)
    
    for metric in metrics:
        baseline_val = baseline_scores[metric]
        attention_val = attention_scores[metric]
        improvement = ((attention_val - baseline_val) / baseline_val * 100) if baseline_val > 0 else 0
        
        print(f"{metric.upper():<15} {baseline_val:<15.2f} {attention_val:<15.2f} {improvement:>+.2f}%")
    
    print("\n" + "=" * 80)
    
    return {
        'baseline': baseline_scores,
        'attention': attention_scores,
        'improvement': {
            metric: attention_scores[metric] - baseline_scores[metric]
            for metric in metrics
        }
    }


def analyze_by_length(model, dataloader, tokenizer_en: Tokenizer,
                     tokenizer_fr: Tokenizer, device: torch.device) -> Dict:
    """
    Analyze model performance by sentence length
    
    This shows how attention helps more on longer sentences
    
    Args:
        model: Translation model
        dataloader: DataLoader
        tokenizer_en: English tokenizer
        tokenizer_fr: French tokenizer
        device: Device
        
    Returns:
        Dictionary with performance by length bucket
    """
    model.eval()
    scorer = BLEUScorer()
    
    # Length buckets: short (1-10), medium (11-20), long (21+)
    buckets = {
        'short': {'hyps': [], 'refs': [], 'range': (1, 10)},
        'medium': {'hyps': [], 'refs': [], 'range': (11, 20)},
        'long': {'hyps': [], 'refs': [], 'range': (21, 100)},
    }
    
    with torch.no_grad():
        for src, tgt, src_len, tgt_len in tqdm(dataloader, desc="Analyzing by length"):
            src = src.to(device)
            
            # Generate translations
            if hasattr(model, 'translate'):
                if 'attention' in model.__class__.__name__.lower():
                    translations, _ = model.translate(src, src_len)
                else:
                    translations = model.translate(src, src_len)
            else:
                tgt = tgt.to(device)
                output = model(src, tgt, src_len, teacher_forcing_ratio=0.0)
                translations = output.argmax(dim=-1)
            
            # Process each sample
            for i in range(len(src)):
                length = src_len[i].item()
                
                # Determine bucket
                bucket_name = None
                for name, bucket in buckets.items():
                    if bucket['range'][0] <= length <= bucket['range'][1]:
                        bucket_name = name
                        break
                
                if bucket_name:
                    hyp = tokenizer_fr.decode(translations[i].cpu().tolist(), remove_special=True)
                    ref = tokenizer_fr.decode(tgt[i].cpu().tolist(), remove_special=True)
                    
                    buckets[bucket_name]['hyps'].append(hyp)
                    buckets[bucket_name]['refs'].append([ref])
    
    # Calculate BLEU for each bucket
    results = {}
    for name, bucket in buckets.items():
        if bucket['hyps']:
            scores = scorer.calculate(bucket['hyps'], bucket['refs'])
            results[name] = {
                'count': len(bucket['hyps']),
                'bleu': scores['bleu'],
                'bleu4': scores['bleu4'],
                'range': bucket['range']
            }
    
    return results

# Example usage and testing
if __name__ == "__main__":
    print("=" * 80)
    print("TESTING BLEU SCORER")
    print("=" * 80)
    
    # Create scorer
    scorer = BLEUScorer()
    
    # Test case 1: Perfect match
    print("\nTest 1: Perfect match")
    hyps = ["hello world"]
    refs = [["hello world"]]
    scores = scorer.calculate(hyps, refs)
    print(f"Hypotheses: {hyps}")
    print(f"References: {refs}")
    print(f"BLEU Score: {scores['bleu']:.2f} (should be 100.0)")
    
    # Test case 2: Partial match
    print("\nTest 2: Partial match")
    hyps = ["hello beautiful world"]
    refs = [["hello world"]]
    scores = scorer.calculate(hyps, refs)
    print(f"Hypotheses: {hyps}")
    print(f"References: {refs}")
    print(f"BLEU Score: {scores['bleu']:.2f}")
    
    # Test case 3: Multiple sentences
    print("\nTest 3: Multiple sentences")
    hyps = [
        "the cat sat on the mat",
        "i love machine learning",
        "hello how are you"
    ]
    refs = [
        ["the cat is on the mat"],
        ["i love machine learning"],
        ["hello how are you today"]
    ]
    scores = scorer.calculate(hyps, refs)
    print(f"BLEU Score: {scores['bleu']:.2f}")
    print(f"BLEU-1: {scores['bleu1']:.2f}")
    print(f"BLEU-2: {scores['bleu2']:.2f}")
    print(f"BLEU-3: {scores['bleu3']:.2f}")
    print(f"BLEU-4: {scores['bleu4']:.2f}")
    
    print("\n" + "=" * 80)
    print("✓ BLEU SCORER TESTS COMPLETE")
    print("=" * 80)

