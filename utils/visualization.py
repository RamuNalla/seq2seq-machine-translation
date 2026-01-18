"""
Attention visualization utilities
Creates heatmaps and plots to visualize attention weights
"""

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
    Plot attention heatmap
    
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
        cmap='YlOrRd',  # Yellow-Orange-Red colormap
        annot=len(src_tokens) < 15 and len(tgt_tokens) < 15,  # Show values if small
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
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    # Tight layout
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved attention plot to {save_path}")
    
    return fig


def plot_attention_comparison(baseline_translation: str,
                              attention_translation: str,
                              attention_weights: np.ndarray,
                              src_tokens: List[str],
                              tgt_tokens: List[str],
                              source_sentence: str,
                              save_path: Optional[Path] = None) -> plt.Figure:
    """
    Plot comparison between baseline and attention models with attention heatmap
    
    Args:
        baseline_translation: Translation from baseline model
        attention_translation: Translation from attention model
        attention_weights: Attention weights (tgt_len, src_len)
        src_tokens: Source tokens
        tgt_tokens: Target tokens
        source_sentence: Original source sentence
        save_path: Path to save figure
        
    Returns:
        Matplotlib figure
    """
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(3, 1, height_ratios=[1, 1, 3], hspace=0.4)
    
    # Source sentence
    ax1 = fig.add_subplot(gs[0])
    ax1.text(0.5, 0.5, f"Source (EN): {source_sentence}", 
             ha='center', va='center', fontsize=12, wrap=True)
    ax1.axis('off')
    
    # Translations comparison
    ax2 = fig.add_subplot(gs[1])
    comparison_text = f"Baseline: {baseline_translation}\n\nAttention: {attention_translation}"
    ax2.text(0.5, 0.5, comparison_text, 
             ha='center', va='center', fontsize=11, 
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    ax2.axis('off')
    
    # Attention heatmap
    ax3 = fig.add_subplot(gs[2])
    sns.heatmap(
        attention_weights,
        xticklabels=src_tokens,
        yticklabels=tgt_tokens,
        cmap='YlOrRd',
        annot=len(src_tokens) < 12,
        fmt='.2f',
        cbar_kws={'label': 'Attention Weight'},
        linewidths=0.5,
        ax=ax3
    )
    ax3.set_xlabel('Source Tokens', fontsize=11)
    ax3.set_ylabel('Target Tokens', fontsize=11)
    ax3.set_title('Attention Weights Heatmap', fontsize=12, fontweight='bold')
    
    plt.setp(ax3.get_xticklabels(), rotation=45, ha='right')
    
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved comparison plot to {save_path}")
    
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


def plot_length_analysis(length_results: dict,
                         save_path: Optional[Path] = None) -> plt.Figure:
    """
    Plot BLEU scores by sentence length
    
    Args:
        length_results: Dictionary with results by length bucket
        save_path: Path to save figure
        
    Returns:
        Matplotlib figure
    """
    categories = list(length_results.keys())
    bleu_scores = [length_results[cat]['bleu'] for cat in categories]
    counts = [length_results[cat]['count'] for cat in categories]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # BLEU scores by length
    ax1.bar(categories, bleu_scores, color=['lightblue', 'steelblue', 'darkblue'], alpha=0.8)
    ax1.set_xlabel('Sentence Length', fontsize=12)
    ax1.set_ylabel('BLEU Score', fontsize=12)
    ax1.set_title('BLEU Score by Sentence Length', fontsize=14, fontweight='bold')
    ax1.grid(True, axis='y', alpha=0.3)
    
    for i, (cat, score) in enumerate(zip(categories, bleu_scores)):
        ax1.text(i, score + 1, f'{score:.1f}', ha='center', fontweight='bold')
    
    # Sample counts
    ax2.bar(categories, counts, color=['lightcoral', 'indianred', 'darkred'], alpha=0.8)
    ax2.set_xlabel('Sentence Length', fontsize=12)
    ax2.set_ylabel('Number of Samples', fontsize=12)
    ax2.set_title('Sample Distribution by Length', fontsize=14, fontweight='bold')
    ax2.grid(True, axis='y', alpha=0.3)
    
    for i, (cat, count) in enumerate(zip(categories, counts)):
        ax2.text(i, count + max(counts)*0.02, f'{count}', ha='center', fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved length analysis plot to {save_path}")
    
    return fig


# Testing
if __name__ == "__main__":
    print("=" * 80)
    print("TESTING VISUALIZATION UTILITIES")
    print("=" * 80)
    
    # Test attention heatmap
    print("\nTest 1: Attention Heatmap")
    attention = np.random.rand(5, 7)
    attention = attention / attention.sum(axis=1, keepdims=True)  # Normalize rows
    
    src_tokens = ['hello', 'how', 'are', 'you', '?', '<eos>']
    tgt_tokens = ['bonjour', 'comment', 'allez', 'vous', '?']
    
    fig = plot_attention(
        attention, 
        src_tokens, 
        tgt_tokens,
        save_path=config.VISUALIZATION_DIR / "test_attention.png"
    )
    plt.close(fig)
    
    # Test training curves
    print("\nTest 2: Training Curves")
    train_losses = [2.5, 2.0, 1.5, 1.2, 1.0, 0.9, 0.8]
    val_losses = [2.6, 2.1, 1.6, 1.3, 1.1, 1.0, 0.95]
    bleu_scores = [
        {'epoch': 5, 'bleu': 25.5},
        {'epoch': 10, 'bleu': 32.8},
    ]
    
    fig = plot_training_curves(
        train_losses,
        val_losses,
        bleu_scores,
        save_path=config.VISUALIZATION_DIR / "test_training_curves.png"
    )
    plt.close(fig)
    
    # Test model comparison
    print("\nTest 3: Model Comparison")
    baseline = {'bleu1': 45, 'bleu2': 32, 'bleu3': 23, 'bleu4': 17}
    attention = {'bleu1': 58, 'bleu2': 48, 'bleu3': 39, 'bleu4': 32}
    
    fig = plot_model_comparison(
        baseline,
        attention,
        save_path=config.VISUALIZATION_DIR / "test_comparison.png"
    )
    plt.close(fig)
    
    print("\n" + "=" * 80)
    print("✓ VISUALIZATION TESTS COMPLETE")
    print(f"✓ Plots saved to {config.VISUALIZATION_DIR}")
    print("=" * 80)