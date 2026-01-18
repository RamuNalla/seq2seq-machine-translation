"""
Attention visualization utilities - SIMPLIFIED ROBUST VERSION
Creates heatmaps using matplotlib (more compatible)
"""

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for compatibility
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
from pathlib import Path
from typing import List, Tuple, Optional
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def plot_attention(attention_weights: np.ndarray, 
                   src_tokens: List[str], 
                   tgt_tokens: List[str],
                   save_path: Optional[Path] = None,
                   title: str = "Attention Weights") -> plt.Figure:
    """
    Plot attention heatmap using matplotlib/seaborn
    
    Args:
        attention_weights: Attention weights matrix (tgt_len, src_len)
        src_tokens: List of source tokens
        tgt_tokens: List of target tokens
        save_path: Path to save figure (optional)
        title: Plot title
        
    Returns:
        Matplotlib figure
    """
    # Create figure
    fig, ax = plt.subplots(figsize=(max(10, len(src_tokens) * 0.8), 
                                    max(8, len(tgt_tokens) * 0.6)))
    
    # Create heatmap
    sns.heatmap(
        attention_weights,
        xticklabels=src_tokens,
        yticklabels=tgt_tokens,
        cmap='YlOrRd',
        annot=len(src_tokens) < 15 and len(tgt_tokens) < 15,
        fmt='.2f',
        cbar_kws={'label': 'Attention Weight'},
        linewidths=0.5,
        linecolor='gray',
        ax=ax,
        vmin=0,
        vmax=1
    )
    
    # Styling
    ax.set_xlabel('Source Tokens (English)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Target Tokens (French)', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    
    # Rotate labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    plt.setp(ax.get_yticklabels(), rotation=0)
    
    # Tight layout
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved attention plot to {save_path}")
    
    return fig


def visualize_translation(model, 
                          src_sentence: str,
                          tokenizer_en,
                          tokenizer_fr,
                          device: torch.device) -> Tuple[str, Optional[plt.Figure]]:
    """
    Translate a sentence and visualize attention weights
    
    Args:
        model: Attention-based model
        src_sentence: Source sentence in English
        tokenizer_en: English tokenizer
        tokenizer_fr: French tokenizer
        device: Device to run on
        
    Returns:
        Tuple of (translation string, attention figure or None)
    """
    model.eval()
    
    # Encode source sentence
    src_indices = tokenizer_en.encode(src_sentence, add_eos=True)
    src_tensor = torch.tensor([src_indices], dtype=torch.long).to(device)
    src_len = torch.tensor([len(src_indices)], dtype=torch.long)
    
    # Translate
    with torch.no_grad():
        if hasattr(model, 'translate'):
            translation, attention_weights = model.translate(src_tensor, src_len)
        else:
            raise ValueError("Model doesn't support translation with attention visualization")
    
    # Decode translation
    translation_indices = translation[0].cpu().tolist()
    translation_text = tokenizer_fr.decode(translation_indices, remove_special=True)
    
    # Get tokens for visualization
    src_tokens = tokenizer_en.tokenize(src_sentence) + ['<eos>']
    tgt_tokens = translation_text.split()
    
    # Create attention plot
    if attention_weights is not None and len(tgt_tokens) > 0:
        # Extract attention weights
        attn = attention_weights[0].cpu().numpy()
        
        # Trim to actual lengths
        attn = attn[:len(tgt_tokens), :len(src_tokens)]
        
        # Create plot
        fig = plot_attention(
            attn, 
            src_tokens, 
            tgt_tokens,
            title=f"Translation: '{src_sentence}'"
        )
    else:
        fig = None
    
    return translation_text, fig


def plot_training_curves(train_losses: List[float],
                         val_losses: List[float],
                         bleu_scores: List[dict] = None,
                         save_path: Optional[Path] = None) -> plt.Figure:
    """
    Plot training and validation loss curves
    
    Args:
        train_losses: List of training losses per epoch
        val_losses: List of validation losses per epoch
        bleu_scores: List of BLEU score dictionaries
        save_path: Path to save figure
        
    Returns:
        Matplotlib figure
    """
    if bleu_scores:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    else:
        fig, ax1 = plt.subplots(1, 1, figsize=(10, 5))
    
    # Loss curves
    epochs = range(1, len(train_losses) + 1)
    ax1.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2, marker='o')
    ax1.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2, marker='s')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # BLEU scores
    if bleu_scores:
        bleu_epochs = [score['epoch'] for score in bleu_scores]
        bleu_values = [score['bleu'] for score in bleu_scores]
        
        ax2.plot(bleu_epochs, bleu_values, 'g-', label='BLEU Score', 
                linewidth=2, marker='D')
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('BLEU Score', fontsize=12)
        ax2.set_title('BLEU Score over Training', fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved training curves to {save_path}")
    
    return fig


def plot_model_comparison(baseline_scores: dict,
                         attention_scores: dict,
                         save_path: Optional[Path] = None) -> plt.Figure:
    """
    Plot bar chart comparing baseline and attention models
    
    Args:
        baseline_scores: BLEU scores for baseline model
        attention_scores: BLEU scores for attention model
        save_path: Path to save figure
        
    Returns:
        Matplotlib figure
    """
    metrics = ['bleu1', 'bleu2', 'bleu3', 'bleu4']
    baseline_vals = [baseline_scores[m] for m in metrics]
    attention_vals = [attention_scores[m] for m in metrics]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bars1 = ax.bar(x - width/2, baseline_vals, width, label='Baseline LSTM', 
                   color='steelblue', alpha=0.8)
    bars2 = ax.bar(x + width/2, attention_vals, width, label='LSTM + Attention',
                   color='coral', alpha=0.8)
    
    ax.set_xlabel('BLEU Metric', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('Model Comparison: BLEU Scores', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([m.upper() for m in metrics])
    ax.legend()
    ax.grid(True, axis='y', alpha=0.3)
    
    # Add value labels on bars
    def autolabel(bars):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom',
                       fontsize=9)
    
    autolabel(bars1)
    autolabel(bars2)
    
    plt.tight_layout()
    
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved comparison plot to {save_path}")
    
    return fig


# Testing
if __name__ == "__main__":
    print("=" * 80)
    print("TESTING VISUALIZATION UTILITIES")
    print("=" * 80)
    
    # Test attention heatmap
    print("\nTest 1: Attention Heatmap")
    attention = np.random.rand(5, 7)
    attention = attention / attention.sum(axis=1, keepdims=True)
    
    src_tokens = ['hello', 'how', 'are', 'you', '?', '<eos>']
    tgt_tokens = ['bonjour', 'comment', 'allez', 'vous', '?']
    
    try:
        fig = plot_attention(
            attention, 
            src_tokens, 
            tgt_tokens,
            save_path=config.VISUALIZATION_DIR / "test_attention.png"
        )
        plt.close(fig)
        print("✓ Attention heatmap test passed")
    except Exception as e:
        print(f"✗ Attention heatmap test failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test training curves
    print("\nTest 2: Training Curves")
    train_losses = [2.5, 2.0, 1.5, 1.2, 1.0, 0.9, 0.8]
    val_losses = [2.6, 2.1, 1.6, 1.3, 1.1, 1.0, 0.95]
    bleu_scores = [
        {'epoch': 5, 'bleu': 25.5},
        {'epoch': 10, 'bleu': 32.8},
    ]
    
    try:
        fig = plot_training_curves(
            train_losses,
            val_losses,
            bleu_scores,
            save_path=config.VISUALIZATION_DIR / "test_training_curves.png"
        )
        plt.close(fig)
        print("✓ Training curves test passed")
    except Exception as e:
        print(f"✗ Training curves test failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test model comparison
    print("\nTest 3: Model Comparison")
    baseline = {'bleu1': 45, 'bleu2': 32, 'bleu3': 23, 'bleu4': 17}
    attention = {'bleu1': 58, 'bleu2': 48, 'bleu3': 39, 'bleu4': 32}
    
    try:
        fig = plot_model_comparison(
            baseline,
            attention,
            save_path=config.VISUALIZATION_DIR / "test_comparison.png"
        )
        plt.close(fig)
        print("✓ Model comparison test passed")
    except Exception as e:
        print(f"✗ Model comparison test failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 80)
    print("✓ VISUALIZATION TESTS COMPLETE")
    print(f"✓ Plots saved to {config.VISUALIZATION_DIR}")
    print("=" * 80)