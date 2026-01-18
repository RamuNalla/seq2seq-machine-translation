"""
Training script for NMT models
Fixed for PyTorch compatibility and CPU training
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import argparse
from pathlib import Path
from tqdm import tqdm
import time
import json
import numpy as np

import config
from utils.data_loader import load_datasets, create_dataloaders
from models.encoder_decoder import create_baseline_model
from models.attention_model import create_attention_model
from utils.metrics import evaluate_model, BLEUScorer
from utils.visualization import visualize_translation


class Trainer:
    """
    Trainer class for NMT models
    Handles training, validation, checkpointing, and logging
    """
    
    def __init__(self, model_type: str, num_epochs: int = 10):
        """
        Initialize trainer
        
        Args:
            model_type: 'baseline' or 'attention'
            num_epochs: Number of epochs to train
        """
        self.model_type = model_type
        self.num_epochs = num_epochs
        self.device = config.DEVICE
        
        # Load data
        print("=" * 80)
        print(f"INITIALIZING TRAINER FOR {model_type.upper()} MODEL")
        print("=" * 80)
        
        self.datasets, self.tokenizer_en, self.tokenizer_fr = load_datasets()
        self.dataloaders = create_dataloaders(self.datasets, config.BATCH_SIZE)
        
        # Create model
        src_vocab_size = len(self.tokenizer_en)
        tgt_vocab_size = len(self.tokenizer_fr)
        
        if model_type == 'baseline':
            self.model = create_baseline_model(src_vocab_size, tgt_vocab_size, self.device)
            self.model_save_path = config.BASELINE_MODEL_PATH
        else:
            self.model = create_attention_model(src_vocab_size, tgt_vocab_size, self.device)
            self.model_save_path = config.ATTENTION_MODEL_PATH
        
        print(f"\n✓ Model: {self.model.__class__.__name__}")
        print(f"✓ Total parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"✓ Device: {self.device}")
        
        # Loss function (ignore padding tokens)
        self.criterion = nn.CrossEntropyLoss(ignore_index=config.PAD_IDX)
        
        # Optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY
        )
        
        # Learning rate scheduler - FIXED: Removed verbose parameter
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=config.LR_SCHEDULER_FACTOR,
            patience=config.LR_SCHEDULER_PATIENCE
        )
        
        # TensorBoard writer
        self.writer = SummaryWriter(config.LOGS_DIR / model_type)
        
        # Training state
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.train_losses = []
        self.val_losses = []
        self.bleu_scores = []
        self.last_lr = config.LEARNING_RATE
        
    def train_epoch(self, epoch: int) -> float:
        """
        Train for one epoch
        
        Args:
            epoch: Current epoch number
            
        Returns:
            Average training loss
        """
        self.model.train()
        epoch_loss = 0
        num_batches = len(self.dataloaders['train'])
        
        # Progress bar
        progress_bar = tqdm(
            self.dataloaders['train'],
            desc=f'Epoch {epoch}/{self.num_epochs}',
            leave=True
        )
        
        for batch_idx, (src, tgt, src_len, tgt_len) in enumerate(progress_bar):
            # Move to device
            src = src.to(self.device)
            tgt = tgt.to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            if self.model_type == 'attention':
                output, _ = self.model(src, tgt, src_len, config.TEACHER_FORCING_RATIO)
            else:
                output = self.model(src, tgt, src_len, config.TEACHER_FORCING_RATIO)
            
            # Reshape for loss calculation
            output_reshaped = output[:, 1:].reshape(-1, output.shape[-1])
            tgt_reshaped = tgt[:, 1:].reshape(-1)
            
            # Calculate loss
            loss = self.criterion(output_reshaped, tgt_reshaped)
            
            # Backward pass
            loss.backward()
            
            # Clip gradients
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), config.GRADIENT_CLIP)
            
            # Update weights
            self.optimizer.step()
            
            # Track loss
            batch_loss = loss.item()
            epoch_loss += batch_loss
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f'{batch_loss:.4f}',
                'avg_loss': f'{epoch_loss/(batch_idx+1):.4f}'
            })
            
            # Log to TensorBoard
            if (batch_idx + 1) % config.LOG_INTERVAL == 0:
                global_step = (epoch - 1) * num_batches + batch_idx
                self.writer.add_scalar('Train/Batch_Loss', batch_loss, global_step)
        
        avg_epoch_loss = epoch_loss / num_batches
        return avg_epoch_loss
    
    def validate(self) -> float:
        """
        Validate the model
        
        Returns:
            Average validation loss
        """
        self.model.eval()
        epoch_loss = 0
        num_batches = len(self.dataloaders['val'])
        
        with torch.no_grad():
            for src, tgt, src_len, tgt_len in tqdm(self.dataloaders['val'], desc='Validating', leave=False):
                src = src.to(self.device)
                tgt = tgt.to(self.device)
                
                # Forward pass (no teacher forcing during validation)
                if self.model_type == 'attention':
                    output, _ = self.model(src, tgt, src_len, teacher_forcing_ratio=0.0)
                else:
                    output = self.model(src, tgt, src_len, teacher_forcing_ratio=0.0)
                
                # Reshape for loss
                output_reshaped = output[:, 1:].reshape(-1, output.shape[-1])
                tgt_reshaped = tgt[:, 1:].reshape(-1)
                
                # Calculate loss
                loss = self.criterion(output_reshaped, tgt_reshaped)
                epoch_loss += loss.item()
        
        avg_epoch_loss = epoch_loss / num_batches
        return avg_epoch_loss
    
    def evaluate_bleu(self) -> dict:
        """
        Evaluate BLEU scores on validation set
        
        Returns:
            Dictionary with BLEU scores
        """
        print("\nCalculating BLEU scores...")
        bleu_scores = evaluate_model(
            self.model,
            self.dataloaders['val'],
            self.tokenizer_en,
            self.tokenizer_fr,
            self.device
        )
        return bleu_scores
    
    def save_checkpoint(self, epoch: int, val_loss: float, is_best: bool = False):
        """
        Save model checkpoint
        
        Args:
            epoch: Current epoch
            val_loss: Validation loss
            is_best: Whether this is the best model so far
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_loss': val_loss,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'bleu_scores': self.bleu_scores,
        }
        
        # Save regular checkpoint
        checkpoint_path = config.CHECKPOINT_DIR / f"{self.model_type}_epoch_{epoch}.pth"
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            torch.save(checkpoint, self.model_save_path)
            print(f"  ✓ Saved best model to {self.model_save_path}")
    
    def visualize_sample_translations(self, epoch: int):
        """
        Visualize attention for sample translations
        
        Args:
            epoch: Current epoch number
        """
        if self.model_type != 'attention':
            return
        
        # Sample sentences for visualization
        sample_sentences = [
            "Hello, how are you?",
            "I love learning.",
            "The weather is beautiful."
        ]
        
        for idx, sentence in enumerate(sample_sentences):
            try:
                translation, fig = visualize_translation(
                    self.model,
                    sentence,
                    self.tokenizer_en,
                    self.tokenizer_fr,
                    self.device
                )
                
                if fig:
                    save_path = config.VISUALIZATION_DIR / f"attention_epoch_{epoch}_sample_{idx+1}.png"
                    fig.savefig(save_path, dpi=150, bbox_inches='tight')
                    print(f"  ✓ Saved attention visualization to {save_path}")
                    
                    # Close figure to free memory
                    import matplotlib.pyplot as plt
                    plt.close(fig)
            except Exception as e:
                print(f"  ✗ Error visualizing sample {idx+1}: {e}")
    
    def train(self):
        """
        Main training loop
        """
        print("\n" + "=" * 80)
        print("STARTING TRAINING")
        print("=" * 80)
        
        for epoch in range(1, self.num_epochs + 1):
            start_time = time.time()
            
            # Train
            train_loss = self.train_epoch(epoch)
            self.train_losses.append(train_loss)
            
            # Validate
            val_loss = self.validate()
            self.val_losses.append(val_loss)
            
            # Learning rate scheduling
            self.scheduler.step(val_loss)
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Check if LR changed
            if current_lr != self.last_lr:
                print(f"\n  ⚡ Learning rate reduced: {self.last_lr:.6f} → {current_lr:.6f}")
                self.last_lr = current_lr
            
            # Calculate BLEU scores every 5 epochs or last epoch
            if epoch % 5 == 0 or epoch == self.num_epochs:
                bleu_scores = self.evaluate_bleu()
                self.bleu_scores.append({
                    'epoch': epoch,
                    **bleu_scores
                })
                
                # Log BLEU scores
                for metric, score in bleu_scores.items():
                    if metric in ['bleu', 'bleu1', 'bleu2', 'bleu3', 'bleu4']:
                        self.writer.add_scalar(f'BLEU/{metric}', score, epoch)
            else:
                bleu_scores = None
            
            # Log metrics
            self.writer.add_scalar('Loss/train', train_loss, epoch)
            self.writer.add_scalar('Loss/val', val_loss, epoch)
            self.writer.add_scalar('Learning_Rate', current_lr, epoch)
            
            # Calculate epoch time
            epoch_time = time.time() - start_time
            
            # Print epoch summary
            print(f'\n{"=" * 80}')
            print(f'Epoch {epoch}/{self.num_epochs} Summary')
            print(f'{"=" * 80}')
            print(f'  Train Loss:    {train_loss:.4f}')
            print(f'  Val Loss:      {val_loss:.4f}')
            print(f'  Learning Rate: {current_lr:.6f}')
            print(f'  Time:          {epoch_time:.2f}s')
            
            if bleu_scores:
                print(f'  BLEU Scores:')
                for metric, score in bleu_scores.items():
                    if metric in ['bleu', 'bleu1', 'bleu2', 'bleu3', 'bleu4']:
                        print(f'    {metric}: {score:.2f}')
            
            # Check if this is the best model
            is_best = val_loss < self.best_val_loss
            
            if is_best:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                print(f'  ✓ New best model! (Val Loss: {val_loss:.4f})')
            else:
                self.patience_counter += 1
                print(f'  No improvement for {self.patience_counter} epoch(s)')
            
            # Save checkpoint
            if epoch % config.SAVE_INTERVAL == 0 or is_best:
                self.save_checkpoint(epoch, val_loss, is_best)
            
            # Visualize attention
            if self.model_type == 'attention' and epoch % config.VISUALIZE_ATTENTION_INTERVAL == 0:
                print("\nGenerating attention visualizations...")
                self.visualize_sample_translations(epoch)
            
            # Early stopping
            if self.patience_counter >= config.EARLY_STOPPING_PATIENCE:
                print(f'\n{"=" * 80}')
                print(f'Early stopping triggered after {epoch} epochs')
                print(f'Best validation loss: {self.best_val_loss:.4f}')
                print(f'{"=" * 80}')
                break
            
            print(f'{"=" * 80}\n')
        
        # Training complete
        self.writer.close()
        
        print("\n" + "=" * 80)
        print("TRAINING COMPLETE!")
        print("=" * 80)
        print(f'✓ Best validation loss: {self.best_val_loss:.4f}')
        print(f'✓ Model saved to: {self.model_save_path}')
        
        # Save training history
        history = {
            'model_type': self.model_type,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'bleu_scores': self.bleu_scores,
            'best_val_loss': self.best_val_loss
        }
        
        history_path = config.LOGS_DIR / f"{self.model_type}_history.json"
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)
        print(f'✓ Training history saved to: {history_path}')
        print("=" * 80 + "\n")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description='Train Neural Machine Translation Model',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train baseline model for 10 epochs
  python train.py --model baseline --epochs 10
  
  # Train attention model with custom batch size
  python train.py --model attention --epochs 10 --batch_size 16
        """
    )
    
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        choices=['baseline', 'attention'],
        help='Model type: baseline (without attention) or attention (with Bahdanau attention)'
    )
    
    parser.add_argument(
        '--epochs',
        type=int,
        default=10,
        help='Number of training epochs (default: 10)'
    )
    
    parser.add_argument(
        '--batch_size',
        type=int,
        default=None,
        help=f'Batch size (default: {config.BATCH_SIZE})'
    )
    
    parser.add_argument(
        '--lr',
        type=float,
        default=None,
        help=f'Learning rate (default: {config.LEARNING_RATE})'
    )
    
    args = parser.parse_args()
    
    # Override config if specified
    if args.batch_size:
        config.BATCH_SIZE = args.batch_size
    if args.lr:
        config.LEARNING_RATE = args.lr
    
    # Print configuration
    print("\n" + "=" * 80)
    print("TRAINING CONFIGURATION")
    print("=" * 80)
    print(f"Model Type:        {args.model}")
    print(f"Epochs:            {args.epochs}")
    print(f"Batch Size:        {config.BATCH_SIZE}")
    print(f"Learning Rate:     {config.LEARNING_RATE}")
    print(f"Device:            {config.DEVICE}")
    print(f"Resume Training:   False")
    print("=" * 80 + "\n")
    
    # Create trainer
    trainer = Trainer(args.model, args.epochs)
    
    # Start training
    trainer.train()


if __name__ == "__main__":
    main()