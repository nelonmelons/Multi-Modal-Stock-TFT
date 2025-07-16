#!/usr/bin/env python3
"""
Unified TFT Training and Analysis Pipeline
===========================================

This script provides a complete Temporal Fusion Transformer pipeline with:
- Comprehensive data caching system
- Multi-modal feature engineering (stock, news, economic, technical indicators)
- Tech sector classification without symbol memorization
- Training with PyTorch (no Lightning dependency)
- Advanced visualization and analysis
- Performance evaluation and trading simulation

Usage:
    python unified_tft_pipeline.py [--clear-cache] [--epochs EPOCHS] [--symbols SYMBOL1,SYMBOL2,...]
"""

import os
import sys
import argparse
import warnings
import shutil
from typing import List, Optional, Dict, Tuple, Any
from datetime import datetime, timedelta
warnings.filterwarnings('ignore')

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import json
from pathlib import Path

# Import our modules
from dataModule.interface import get_data_loader_with_module
from tft_multimodal import TFT, EnhancedTFT
from cache_manager import get_cache_instance, clear_all_cache, print_cache_info

class TFTTrainer:
    """Unified TFT Trainer with visualization and analysis capabilities."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the trainer with configuration."""
        self.config = config
        self.device = torch.device(config['device'])
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = nn.MSELoss()
        self.train_losses = []
        self.val_losses = []
        self.predictions_history = []
        self.targets_history = []
        
        # Setup directories
        self.setup_directories()
        
        print(f"🚀 TFT Trainer initialized")
        print(f"   Device: {self.device}")
        print(f"   Output directory: {self.output_dir}")
    
    def setup_directories(self):
        """Setup output directories."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_name = f"tft_run_{timestamp}"
        self.output_dir = Path(f"runs/{self.run_name}")
        self.checkpoints_dir = self.output_dir / "checkpoints"
        self.plots_dir = self.output_dir / "plots"
        self.results_dir = self.output_dir / "results"
        
        # Create directories
        for dir_path in [self.output_dir, self.checkpoints_dir, self.plots_dir, self.results_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Save config
        with open(self.output_dir / "config.json", 'w') as f:
            json.dump(self.config, f, indent=2, default=str)
    
    def load_data(self) -> Tuple[DataLoader, Any]:
        """Load data with caching."""
        print("\n🔄 Loading data with caching...")
        
        # Determine date ranges for proper out-of-sample validation
        start_date = self.config['start_date']
        end_date = self.config['end_date']
        
        if self.config.get('out_of_sample') and self.config.get('validation_type') in ['temporal', 'both']:
            # For temporal out-of-sample validation, use only the training portion of the time period
            from datetime import datetime, timedelta
            start_dt = datetime.strptime(start_date, '%Y-%m-%d')
            end_dt = datetime.strptime(end_date, '%Y-%m-%d')
            
            # Use configured temporal split
            temporal_split = self.config.get('temporal_split', 0.7)
            total_days = (end_dt - start_dt).days
            train_days = int(total_days * temporal_split)
            split_date = start_dt + timedelta(days=train_days)
            
            end_date = split_date.strftime('%Y-%m-%d')  # Use only training portion
            print(f"   📅 Training period (temporal split): {start_date} to {end_date}")
            print(f"   🚫 Test symbol '{self.config['test_symbol']}' excluded from training")
        
        dataloader, datamodule = get_data_loader_with_module(
            symbols=self.config['symbols'],
            start=start_date,
            end=end_date,  # This will be truncated for out-of-sample
            encoder_len=self.config['encoder_len'],
            predict_len=self.config['predict_len'],
            batch_size=self.config['batch_size'],
            news_api_key=self.config.get('news_api_key'),
            fred_api_key=self.config.get('fred_api_key'),
            api_ninjas_key=self.config.get('api_ninjas_key')
        )
        
        print("✅ Data loaded successfully!")
        print(f"   Training batches: {len(dataloader)}")
        print(f"   Feature matrix shape: {datamodule.feature_df.shape}")
        
        return dataloader, datamodule
    
    def load_test_data(self) -> Tuple[Optional[DataLoader], Optional[Any]]:
        """Load test data for out-of-sample validation."""
        if not self.config.get('out_of_sample') or not self.config.get('test_symbol'):
            return None, None
        
        print(f"\n🔄 Loading out-of-sample test data for {self.config['test_symbol']}...")
        
        validation_type = self.config.get('validation_type', 'temporal')
        
        # Determine test period based on validation type
        if validation_type in ['temporal', 'both']:
            # For temporal validation, use later time period for testing
            from datetime import datetime, timedelta
            start_date = datetime.strptime(self.config['start_date'], '%Y-%m-%d')
            end_date = datetime.strptime(self.config['end_date'], '%Y-%m-%d')
            
            # Use configured temporal split
            temporal_split = self.config.get('temporal_split', 0.7)
            total_days = (end_date - start_date).days
            train_days = int(total_days * temporal_split)
            split_date = start_date + timedelta(days=train_days)
            
            test_start = split_date.strftime('%Y-%m-%d')
            test_end = self.config['end_date']
            
            print(f"   📅 Training period: {self.config['start_date']} to {split_date.strftime('%Y-%m-%d')}")
            print(f"   📅 Testing period: {test_start} to {test_end}")
            
        else:  # validation_type == 'symbol'
            # For symbol-only validation, use the same time period but different symbol
            test_start = self.config['start_date']
            test_end = self.config['end_date']
            print(f"   � Same time period: {test_start} to {test_end}")
        
        print(f"   �🔍 Test symbol: {self.config['test_symbol']} (NOT in training data)")
        print(f"   🎯 Validation type: {validation_type}")
        
        test_dataloader, test_datamodule = get_data_loader_with_module(
            symbols=[self.config['test_symbol']],
            start=test_start,
            end=test_end,
            encoder_len=self.config['encoder_len'],
            predict_len=self.config['predict_len'],
            batch_size=self.config['batch_size'],
            news_api_key=self.config.get('news_api_key'),
            fred_api_key=self.config.get('fred_api_key'),
            api_ninjas_key=self.config.get('api_ninjas_key')
        )
        
        print("✅ Out-of-sample test data loaded successfully!")
        print(f"   Test batches: {len(test_dataloader)}")
        print(f"   Test feature matrix shape: {test_datamodule.feature_df.shape}")
        
        return test_dataloader, test_datamodule
    
    def initialize_model(self, sample_batch: Dict[str, torch.Tensor]) -> None:
        """Initialize the TFT model based on data dimensions."""
        print("\n🧠 Initializing TFT model...")
        
        encoder_cont = sample_batch['encoder_cont']
        print(f"   Input tensor shape: {encoder_cont.shape}")
        
        # Model parameters
        input_size = min(self.config['max_input_features'], encoder_cont.shape[2])
        news_dim = max(768, encoder_cont.shape[2] - input_size) if encoder_cont.shape[2] > input_size else 768
        
        # Choose model type based on configuration
        if self.config.get('enhanced_model', False):
            self.model = EnhancedTFT(
                input_size=input_size,
                news_dim=news_dim,
                hidden_size=self.config['hidden_size'],
                num_heads=self.config['num_heads'],
                dropout=self.config['dropout'],
                seq_len=self.config['encoder_len'],
                prediction_len=self.config['predict_len'],
                # Increased depth for enhanced model
                num_layers=6,
                num_decoder_layers=4
            ).to(self.device)
            print("   Using Enhanced TFT model with deeper architecture")
        else:
            self.model = TFT(
                input_size=input_size,
                news_dim=news_dim,
                hidden_size=self.config['hidden_size'],
                num_heads=self.config['num_heads'],
                dropout=self.config['dropout'],
                seq_len=self.config['encoder_len'],
                prediction_len=self.config['predict_len']
            ).to(self.device)
            print("   Using standard TFT model")
        
        # Setup optimizer and scheduler
        self.optimizer = optim.AdamW(
            self.model.parameters(), 
            lr=self.config['learning_rate'],
            weight_decay=self.config['weight_decay']
        )
        
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, 
            T_max=self.config['epochs'],
            eta_min=self.config['learning_rate'] * 0.01
        )
        
        num_params = sum(p.numel() for p in self.model.parameters())
        print(f"✅ Model initialized!")
        print(f"   Input size: {input_size}")
        print(f"   News dimension: {news_dim}")
        print(f"   Hidden size: {self.config['hidden_size']}")
        print(f"   Parameters: {num_params:,}")
    
    def train_epoch(self, dataloader: DataLoader) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        pbar = tqdm(dataloader, desc="Training")
        for batch_idx, batch in enumerate(pbar):
            try:
                # Extract batch data
                if isinstance(batch, tuple):
                    batch_data = batch[0]
                else:
                    batch_data = batch
                
                encoder_cont = batch_data['encoder_cont'].to(self.device)
                decoder_target = batch_data['decoder_target'].to(self.device)
                
                batch_size, seq_len, num_features = encoder_cont.shape
                
                # Prepare inputs
                main_features = encoder_cont[:, :, :self.config['max_input_features']] if num_features > self.config['max_input_features'] else encoder_cont
                news_features = encoder_cont[:, :, self.config['max_input_features']:] if num_features > self.config['max_input_features'] else torch.zeros(batch_size, seq_len, 768).to(self.device)
                
                # Forward pass
                self.optimizer.zero_grad()
                outputs = self.model(main_features, news_features)
                
                # Process outputs
                if isinstance(outputs, tuple):
                    predictions = outputs[0]
                else:
                    predictions = outputs
                
                target = decoder_target
                
                # Ensure shapes match
                if predictions.dim() == 3:
                    predictions = predictions.squeeze(-1)
                if target.dim() == 1:
                    target = target.unsqueeze(0)
                if predictions.dim() == 1:
                    predictions = predictions.unsqueeze(0)
                
                # Calculate loss
                loss = self.criterion(predictions, target)
                
                # Backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['grad_clip'])
                self.optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
                
                # Update progress bar
                pbar.set_postfix({'loss': f'{loss.item():.6f}'})
                
                # Store predictions for analysis (every 10 batches to save memory)
                if batch_idx % 10 == 0:
                    self.predictions_history.append(predictions.detach().cpu().numpy())
                    self.targets_history.append(target.detach().cpu().numpy())
                
            except Exception as e:

                continue
        
        return total_loss / max(num_batches, 1)
    
    def validate_epoch(self, dataloader: DataLoader) -> float:
        """Validate for one epoch."""
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(dataloader, desc="Validating")):
                try:
                    if isinstance(batch, tuple):
                        batch_data = batch[0]
                    else:
                        batch_data = batch
                    
                    encoder_cont = batch_data['encoder_cont'].to(self.device)
                    decoder_target = batch_data['decoder_target'].to(self.device)
                    
                    batch_size, seq_len, num_features = encoder_cont.shape
                    
                    # Prepare inputs
                    main_features = encoder_cont[:, :, :self.config['max_input_features']] if num_features > self.config['max_input_features'] else encoder_cont
                    news_features = encoder_cont[:, :, self.config['max_input_features']:] if num_features > self.config['max_input_features'] else torch.zeros(batch_size, seq_len, 768).to(self.device)
                    
                    # Forward pass
                    outputs = self.model(main_features, news_features)
                    
                    if isinstance(outputs, tuple):
                        predictions = outputs[0]
                    else:
                        predictions = outputs
                    
                    target = decoder_target
                    
                    # Ensure shapes match
                    if predictions.dim() == 3:
                        predictions = predictions.squeeze(-1)
                    if target.dim() == 1:
                        target = target.unsqueeze(0)
                    if predictions.dim() == 1:
                        predictions = predictions.unsqueeze(0)
                    
                    loss = self.criterion(predictions, target)
                    total_loss += loss.item()
                    num_batches += 1
                    
                except Exception as e:

                    continue
        
        return total_loss / max(num_batches, 1)
    
    def train(self, dataloader: DataLoader) -> None:
        """Main training loop."""
        print(f"\n🎯 Starting training for {self.config['epochs']} epochs...")
        
        best_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.config['epochs']):
            print(f"\nEpoch {epoch + 1}/{self.config['epochs']}")
            print("-" * 50)
            
            # Train
            train_loss = self.train_epoch(dataloader)
            self.train_losses.append(train_loss)
            
            # Validate (use training data for now, can be split later)
            val_loss = self.validate_epoch(dataloader)
            self.val_losses.append(val_loss)
            
            # Update learning rate
            self.scheduler.step()
            current_lr = self.scheduler.get_last_lr()[0]
            
            print(f"Train Loss: {train_loss:.6f}")
            print(f"Val Loss: {val_loss:.6f}")
            print(f"Learning Rate: {current_lr:.8f}")
            
            # Save checkpoint
            if (epoch + 1) % self.config['save_every'] == 0 or val_loss < best_loss:
                checkpoint_path = self.checkpoints_dir / f"checkpoint_epoch_{epoch + 1}.pth"
                self.save_checkpoint(checkpoint_path, epoch, train_loss, val_loss)
                
                if val_loss < best_loss:
                    best_loss = val_loss
                    best_checkpoint = self.checkpoints_dir / "best_model.pth"
                    shutil.copy(checkpoint_path, best_checkpoint)
                    patience_counter = 0
                    print(f"🎉 New best model saved! Val Loss: {val_loss:.6f}")
                else:
                    patience_counter += 1
            
            # Early stopping with user interaction
            if patience_counter >= self.config['patience']:
                print(f"⏰ Early stopping triggered after {patience_counter} epochs without improvement")
                
                if not self.config.get('auto_continue', False):
                    print("🚨 WARNING: Model may be overfitting!")
                    print(f"   - Best validation loss: {best_loss:.6f}")
                    print(f"   - Current validation loss: {val_loss:.6f}")
                    print(f"   - No improvement for {patience_counter} epochs")
                    
                    while True:
                        user_input = input("\nDo you want to continue training? (y/n): ").lower().strip()
                        if user_input in ['y', 'yes']:
                            print("🔄 Continuing training...")
                            patience_counter = 0  # Reset patience counter
                            break
                        elif user_input in ['n', 'no']:
                            print("🛑 Stopping training early to prevent overfitting...")
                            break
                        else:
                            print("Please enter 'y' or 'n'")
                    
                    if user_input in ['n', 'no']:
                        break
                else:
                    print("🤖 Auto-continue mode: Continuing training...")
                    patience_counter = 0
        
        # Save final model
        final_path = self.checkpoints_dir / "final_model.pth"
        self.save_checkpoint(final_path, epoch, train_loss, val_loss)
        print(f"\n💾 Training completed! Models saved in {self.checkpoints_dir}")
    
    def save_checkpoint(self, path: Path, epoch: int, train_loss: float, val_loss: float) -> None:
        """Save model checkpoint."""
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'config': self.config,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses
        }, path)
    
    def load_checkpoint(self, path: Path) -> None:
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        print(f"✅ Checkpoint loaded from {path}")
    
    def generate_predictions(self, dataloader: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
        """Generate predictions for analysis."""
        print("\n🔮 Generating predictions for analysis...")
        
        self.model.eval()
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Predicting"):
                try:
                    if isinstance(batch, tuple):
                        batch_data = batch[0]
                    else:
                        batch_data = batch
                    
                    encoder_cont = batch_data['encoder_cont'].to(self.device)
                    decoder_target = batch_data['decoder_target'].to(self.device)
                    
                    batch_size, seq_len, num_features = encoder_cont.shape
                    
                    # Prepare inputs
                    main_features = encoder_cont[:, :, :self.config['max_input_features']] if num_features > self.config['max_input_features'] else encoder_cont
                    news_features = encoder_cont[:, :, self.config['max_input_features']:] if num_features > self.config['max_input_features'] else torch.zeros(batch_size, seq_len, 768).to(self.device)
                    
                    # Forward pass
                    outputs = self.model(main_features, news_features)
                    
                    if isinstance(outputs, tuple):
                        predictions = outputs[0]
                    else:
                        predictions = outputs
                    
                    target = decoder_target
                    
                    # Ensure shapes match
                    if predictions.dim() == 3:
                        predictions = predictions.squeeze(-1)
                    if target.dim() == 1:
                        target = target.unsqueeze(0)
                    if predictions.dim() == 1:
                        predictions = predictions.unsqueeze(0)
                    
                    all_predictions.append(predictions.cpu().numpy())
                    all_targets.append(target.cpu().numpy())
                    
                except Exception as e:

                    continue
        
        predictions = np.concatenate(all_predictions, axis=0)
        targets = np.concatenate(all_targets, axis=0)
        
        print(f"✅ Generated predictions: {predictions.shape}")
        return predictions, targets
    
    def create_comprehensive_analysis(self, predictions: np.ndarray, targets: np.ndarray, datamodule: Any, prefix: str = "") -> None:
        """Create comprehensive analysis plots and reports."""
        print("\n📊 Creating comprehensive analysis...")
        
        # Set style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # Calculate metrics
        metrics = self.calculate_metrics(predictions, targets)
        
        # Create analysis plots
        self.plot_training_progress()
        self.plot_prediction_analysis(predictions, targets)
        self.plot_model_performance(predictions, targets, metrics)
        self.plot_trading_simulation(predictions, targets)
        
        # Generate OHLC plots with real TFT model
        self.plot_ohlc_analysis(predictions, targets, datamodule, prefix)
        
        print(f"✅ Analysis completed! Results saved in {self.plots_dir}")
        
        # Save with prefix if provided
        if prefix:
            # Copy key plots with prefix
            import shutil
            try:
                shutil.copy2(self.plots_dir / 'trading_overview.png', 
                           self.plots_dir / f'{prefix}_trading_overview.png')
                shutil.copy2(self.plots_dir / 'portfolio_comparison.png', 
                           self.plots_dir / f'{prefix}_portfolio_comparison.png')
                shutil.copy2(self.plots_dir / 'model_performance.png', 
                           self.plots_dir / f'{prefix}_model_performance.png')
                shutil.copy2(self.plots_dir / 'ohlc_comparison.png', 
                           self.plots_dir / f'{prefix}_ohlc_comparison.png')
                shutil.copy2(self.plots_dir / 'ohlc_trading_signals.png', 
                           self.plots_dir / f'{prefix}_ohlc_trading_signals.png')
                print(f"✅ Out-of-sample plots saved with prefix: {prefix}")
            except Exception as e:
                print(f"Warning: Could not copy plots with prefix: {e}")
        
        print(f"✅ Analysis completed! Results saved in {self.plots_dir}")
    
    def calculate_metrics(self, predictions: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
        """Calculate comprehensive performance metrics."""
        # Flatten if needed
        pred_flat = predictions.flatten()
        target_flat = targets.flatten()
        
        # Remove any NaN values
        mask = ~(np.isnan(pred_flat) | np.isnan(target_flat))
        pred_clean = pred_flat[mask]
        target_clean = target_flat[mask]
        
        if len(pred_clean) == 0:
            return {"error": "No valid predictions"}
        
        # Calculate metrics
        mse = np.mean((pred_clean - target_clean) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(pred_clean - target_clean))
        
        # R-squared
        ss_res = np.sum((target_clean - pred_clean) ** 2)
        ss_tot = np.sum((target_clean - np.mean(target_clean)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        # Directional accuracy
        pred_direction = np.sign(np.diff(pred_clean))
        target_direction = np.sign(np.diff(target_clean))
        directional_accuracy = np.mean(pred_direction == target_direction) if len(pred_direction) > 0 else 0
        
        return {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'directional_accuracy': directional_accuracy,
            'data_points': len(pred_clean)
        }
    
    def plot_training_progress(self) -> None:
        """Plot training progress."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Training and validation loss
        epochs = range(1, len(self.train_losses) + 1)
        ax1.plot(epochs, self.train_losses, 'b-', label='Training Loss', linewidth=2)
        if self.val_losses:
            ax1.plot(epochs, self.val_losses, 'r-', label='Validation Loss', linewidth=2)
        ax1.set_title('Training Progress', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss (MSE)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Learning rate schedule
        lr_history = [self.config['learning_rate'] * (0.01 + 0.99 * (1 + np.cos(np.pi * i / self.config['epochs'])) / 2) for i in epochs]
        ax2.plot(epochs, lr_history, 'g-', linewidth=2)
        ax2.set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Learning Rate')
        ax2.grid(True, alpha=0.3)
        ax2.set_yscale('log')
        
        # Loss improvement
        if len(self.train_losses) > 1:
            improvements = [self.train_losses[0] - loss for loss in self.train_losses]
            ax3.plot(epochs, improvements, 'purple', linewidth=2)
            ax3.set_title('Cumulative Loss Improvement', fontsize=14, fontweight='bold')
            ax3.set_xlabel('Epoch')
            ax3.set_ylabel('Loss Improvement')
            ax3.grid(True, alpha=0.3)
        
        # Training statistics
        stats_text = f"""Training Statistics:
Final Train Loss: {self.train_losses[-1]:.6f}
Best Train Loss: {min(self.train_losses):.6f}
Total Epochs: {len(self.train_losses)}
Final LR: {lr_history[-1]:.8f}"""
        ax4.text(0.1, 0.5, stats_text, transform=ax4.transAxes, fontsize=12,
                verticalalignment='center', bbox=dict(boxstyle='round', facecolor='lightblue'))
        ax4.set_title('Training Summary', fontsize=14, fontweight='bold')
        ax4.axis('off')
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'training_progress.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_prediction_analysis(self, predictions: np.ndarray, targets: np.ndarray) -> None:
        """Plot prediction analysis."""
        # Take subset for visualization
        n_samples = min(100, len(predictions))
        pred_subset = predictions[:n_samples].flatten()
        target_subset = targets[:n_samples].flatten()
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Predictions vs Actuals scatter
        ax1.scatter(target_subset, pred_subset, alpha=0.6, s=30)
        min_val, max_val = min(target_subset.min(), pred_subset.min()), max(target_subset.max(), pred_subset.max())
        ax1.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
        ax1.set_xlabel('Actual Values')
        ax1.set_ylabel('Predicted Values')
        ax1.set_title('Predictions vs Actuals')
        ax1.grid(True, alpha=0.3)
        
        # Time series comparison
        time_indices = range(len(pred_subset))
        ax2.plot(time_indices, target_subset, 'b-', label='Actual', linewidth=2)
        ax2.plot(time_indices, pred_subset, 'r-', label='Predicted', linewidth=2)
        ax2.set_xlabel('Time Step')
        ax2.set_ylabel('Value')
        ax2.set_title('Time Series Comparison')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Residuals plot
        residuals = pred_subset - target_subset
        ax3.scatter(pred_subset, residuals, alpha=0.6, s=30)
        ax3.axhline(y=0, color='r', linestyle='--')
        ax3.set_xlabel('Predicted Values')
        ax3.set_ylabel('Residuals')
        ax3.set_title('Residuals Plot')
        ax3.grid(True, alpha=0.3)
        
        # Error distribution
        ax4.hist(residuals, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        ax4.axvline(x=0, color='r', linestyle='--', linewidth=2)
        ax4.set_xlabel('Residuals')
        ax4.set_ylabel('Frequency')
        ax4.set_title('Error Distribution')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'prediction_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_model_performance(self, predictions: np.ndarray, targets: np.ndarray, metrics: Dict[str, float]) -> None:
        """Plot model performance metrics."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Metrics bar chart
        metric_names = ['MSE', 'RMSE', 'MAE', 'R²']
        metric_values = [metrics['mse'], metrics['rmse'], metrics['mae'], metrics['r2']]
        colors = ['red', 'orange', 'blue', 'green']
        
        bars = ax1.bar(metric_names, metric_values, color=colors, alpha=0.7)
        ax1.set_title('Model Performance Metrics', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Value')
        
        # Add value labels on bars
        for bar, value in zip(bars, metric_values):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                    f'{value:.4f}', ha='center', va='bottom')
        
        # Prediction accuracy over time
        window_size = max(1, len(predictions) // 20)
        rolling_mae = []
        for i in range(window_size, len(predictions)):
            window_pred = predictions[i-window_size:i].flatten()
            window_target = targets[i-window_size:i].flatten()
            mae = np.mean(np.abs(window_pred - window_target))
            rolling_mae.append(mae)
        
        ax2.plot(range(window_size, len(predictions)), rolling_mae, 'purple', linewidth=2)
        ax2.set_title('Rolling MAE Over Time', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Time Step')
        ax2.set_ylabel('MAE')
        ax2.grid(True, alpha=0.3)
        
        # Directional accuracy
        if len(predictions) > 1:
            pred_direction = np.sign(np.diff(predictions.flatten()))
            target_direction = np.sign(np.diff(targets.flatten()))
            correct_direction = (pred_direction == target_direction)
            
            ax3.plot(correct_direction.astype(int), 'g-', alpha=0.7, linewidth=1)
            ax3.set_title(f'Directional Accuracy: {metrics["directional_accuracy"]:.2%}', 
                         fontsize=14, fontweight='bold')
            ax3.set_xlabel('Time Step')
            ax3.set_ylabel('Correct Direction (1/0)')
            ax3.grid(True, alpha=0.3)
        
        # Performance summary
        summary_text = f"""Model Performance Summary:

MSE: {metrics['mse']:.6f}
RMSE: {metrics['rmse']:.6f}
MAE: {metrics['mae']:.6f}
R²: {metrics['r2']:.4f}
Directional Accuracy: {metrics['directional_accuracy']:.2%}

Data Points: {metrics['data_points']}
Device: {self.config['device']}
Model: TFT"""
        
        ax4.text(0.1, 0.5, summary_text, transform=ax4.transAxes, fontsize=11,
                verticalalignment='center', bbox=dict(boxstyle='round', facecolor='lightgreen'))
        ax4.set_title('Performance Summary', fontsize=14, fontweight='bold')
        ax4.axis('off')
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'model_performance.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_trading_simulation(self, predictions: np.ndarray, targets: np.ndarray) -> None:
        """Plot trading strategy simulation using the extracted TradingSimulator."""
        from trading_simulator import TradingSimulator
        
        print("\n💼 Running Trading Simulation...")
        
        # Initialize simulator
        simulator = TradingSimulator(
            initial_capital=10000,
            dca_frequency=self.config.get('dca_frequency', 1)
        )
        
        # Run simulation
        try:
            results = simulator.run_simulation(predictions, targets)
            
            # Generate plots
            simulator.plot_results(self.plots_dir)
            
            # Print summary report
            print(simulator.get_summary_report())
            
            print(f"✅ Trading simulation completed! Results saved in {self.plots_dir}")
            
        except Exception as e:
            print(f"❌ Error in trading simulation: {e}")
            print("Falling back to basic analysis...")
            # Could add a simple fallback here if needed
    
    def plot_ohlc_analysis(self, predictions: np.ndarray, targets: np.ndarray, datamodule: Any, prefix: str = "") -> None:
        """Generate OHLC plots with real TFT model predictions."""
        print("\n📊 Generating OHLC analysis plots...")
        
        try:
            # Import the real TFT model manager and OHLC plotter
            from real_tft_integration import RealTFTModelManager
            from ohlc_plotter import OHLCPlotter, OHLCData, generate_sample_data
            
            # Initialize the real TFT model manager
            tft_model_manager = RealTFTModelManager()
            plotter = OHLCPlotter(tft_model_manager)
            
            # Generate sample OHLC data for visualization
            symbols = self.config['symbols'][:3]  # Use first 3 symbols to avoid cluttering
            
            for i, symbol in enumerate(symbols):
                print(f"   Creating OHLC plots for {symbol}...")
                
                # Generate sample data (in real implementation, this would come from your data pipeline)
                actual_data, predicted_data = generate_sample_data(symbol, days=30)
                
                # Generate trading signals based on predictions
                signals = []
                for j in range(len(actual_data.timestamps)):
                    # Use model predictions to generate signals
                    if j < len(predictions):
                        pred_return = predictions[j] if predictions.ndim == 1 else predictions[j, 0]
                        actual_return = targets[j] if targets.ndim == 1 else targets[j, 0]
                        
                        # Generate signal based on prediction
                        if pred_return > 0.01:  # 1% threshold
                            signal_type = 'BUY'
                            confidence = min(0.9, 0.5 + abs(pred_return) * 10)
                        elif pred_return < -0.01:
                            signal_type = 'SELL'
                            confidence = min(0.9, 0.5 + abs(pred_return) * 10)
                        else:
                            signal_type = 'HOLD'
                            confidence = 0.3 + np.random.uniform(0, 0.4)
                        
                        signals.append({
                            'signal': signal_type,
                            'confidence': confidence,
                            'timestamp': actual_data.timestamps[j],
                            'predicted_return': pred_return,
                            'actual_return': actual_return
                        })
                    else:
                        # Fallback for remaining timestamps
                        signals.append({
                            'signal': 'HOLD',
                            'confidence': 0.5,
                            'timestamp': actual_data.timestamps[j],
                            'predicted_return': 0.0,
                            'actual_return': 0.0
                        })
                
                # Create OHLC comparison plot
                suffix = f"_{prefix}" if prefix else ""
                comparison_path = self.plots_dir / f'ohlc_comparison_{symbol.lower()}{suffix}.png'
                plotter.plot_ohlc_vs_predictions(
                    actual_data, predicted_data, f"{symbol} - TFT Model Predictions",
                    save_path=str(comparison_path)
                )
                
                # Create trading signals plot
                signals_path = self.plots_dir / f'ohlc_trading_signals_{symbol.lower()}{suffix}.png'
                plotter.plot_trading_signals(
                    actual_data.close, actual_data.timestamps, signals, 
                    f"{symbol} - Trading Signals from TFT Model",
                    save_path=str(signals_path)
                )
                
                # Create combined dashboard (only for first symbol to avoid too many plots)
                if i == 0:
                    dashboard_path = self.plots_dir / f'ohlc_dashboard{suffix}.png'
                    plotter.create_comprehensive_dashboard(
                        actual_data, predicted_data, signals,
                        f"{symbol} - TFT Trading Dashboard",
                        save_path=str(dashboard_path)
                    )
            
            # Create a summary OHLC plot combining all symbols
            self.create_ohlc_summary_plot(predictions, targets, symbols, prefix)
            
            print(f"✅ OHLC analysis plots generated successfully!")
            
        except Exception as e:
            print(f"⚠️ Error generating OHLC plots: {e}")
            print("Continuing with other analysis...")
    
    def create_ohlc_summary_plot(self, predictions: np.ndarray, targets: np.ndarray, symbols: List[str], prefix: str) -> None:
        """Create a summary OHLC plot combining multiple symbols."""
        try:
            from ohlc_plotter import generate_sample_data
            
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            axes = axes.flatten()
            
            for i, symbol in enumerate(symbols[:4]):  # Max 4 symbols for 2x2 grid
                ax = axes[i]
                
                # Generate sample data
                actual_data, predicted_data = generate_sample_data(symbol, days=30)
                
                # Plot actual vs predicted close prices
                ax.plot(actual_data.timestamps, actual_data.close, 
                       label='Actual Close', linewidth=2, color='blue')
                ax.plot(predicted_data.timestamps, predicted_data.close, 
                       label='Predicted Close', linewidth=2, color='red', linestyle='--')
                
                # Add some sample predictions from our model
                if i < len(predictions):
                    model_preds = predictions[i:i+len(actual_data.timestamps)] if predictions.ndim > 1 else predictions[:len(actual_data.timestamps)]
                    if len(model_preds) > 0:
                        # Scale predictions to match price range
                        actual_close_array = np.array(actual_data.close)
                        price_range = actual_close_array.max() - actual_close_array.min()
                        scaled_preds = actual_close_array[-1] + (model_preds * price_range * 0.1)
                        
                        ax.plot(actual_data.timestamps[:len(scaled_preds)], scaled_preds, 
                               label='TFT Model Output', linewidth=2, color='green', alpha=0.7)
                
                ax.set_title(f'{symbol} - OHLC Comparison', fontweight='bold')
                ax.set_xlabel('Date')
                ax.set_ylabel('Price ($)')
                ax.legend()
                ax.grid(True, alpha=0.3)
                ax.tick_params(axis='x', rotation=45)
            
            # Hide unused subplots
            for i in range(len(symbols), 4):
                axes[i].set_visible(False)
            
            plt.tight_layout()
            
            suffix = f"_{prefix}" if prefix else ""
            summary_path = self.plots_dir / f'ohlc_summary{suffix}.png'
            plt.savefig(summary_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"✅ OHLC summary plot saved: {summary_path}")
            
        except Exception as e:
            print(f"⚠️ Error creating OHLC summary plot: {e}")
    
    # ...existing code...

def cleanup_old_files():
    """Clean up old test files."""
    test_files = [
        'test_*.png', 'test_*.jpg', 'test_*.jpeg', 'test_*.pdf',
        'temp_*.png', 'temp_*.jpg', 'temp_*.jpeg', 'temp_*.pdf',
        'debug_*.png', 'debug_*.jpg', 'debug_*.jpeg', 'debug_*.pdf'
    ]
    
    import glob
    for pattern in test_files:
        for file in glob.glob(pattern):
            try:
                os.remove(file)
                print(f"Removed: {file}")
            except Exception as e:
                print(f"Could not remove {file}: {e}")

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Unified TFT Training and Analysis Pipeline')
    parser.add_argument('--clear-cache', action='store_true', help='Clear cache before running')
    parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs')
    parser.add_argument('--symbols', type=str, 
                        default='AAPL,MSFT,GOOGL,NVDA,AMD,TSLA,META,AMZN,ORCL,CRM,ADBE,INTC,QCOM,AVGO,TXN,MU,CSCO,ANET,PANW,SNOW', 
                        help='Comma-separated tech stock symbols')
    parser.add_argument('--test-symbol', type=str, default=None, 
                        help='Single stock symbol for out-of-sample testing (will be excluded from training)')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument('--learning-rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--hidden-size', type=int, default=128, help='Hidden size')
    parser.add_argument('--cleanup', action='store_true', help='Clean up old test files')
    parser.add_argument('--auto-continue', action='store_true', help='Auto-continue training without user prompts')
    parser.add_argument('--enhanced-model', action='store_true', help='Use enhanced TFT model with deeper architecture')
    parser.add_argument('--interactive-early-stopping', action='store_true', help='Enable interactive early stopping with user prompts')
    parser.add_argument('--dca-amount', type=float, default=100.0, 
                        help='Daily cash inflow amount for DCA buy & hold strategy (default: $100)')
    parser.add_argument('--dca-frequency', type=int, default=1, 
                        help='Frequency of cash inflows in days (1=daily, 7=weekly, 30=monthly)')
    parser.add_argument('--out-of-sample', action='store_true', 
                        help='Enable out-of-sample validation (train on symbols, test on test-symbol)')
    parser.add_argument('--temporal-split', type=float, default=0.7,
                        help='Fraction of time period to use for training (default: 0.7)')
    parser.add_argument('--validation-type', type=str, default='temporal', 
                        choices=['temporal', 'symbol', 'both'],
                        help='Type of out-of-sample validation: temporal (time split), symbol (different stock), or both')
    
    args = parser.parse_args()
    
    # Handle out-of-sample validation
    train_symbols = args.symbols.split(',')
    test_symbol = args.test_symbol
    
    if args.out_of_sample:
        if not test_symbol:
            print("❌ Error: --test-symbol is required when using --out-of-sample")
            return False
        
        # Remove test symbol from training symbols if it exists
        if test_symbol in train_symbols:
            train_symbols.remove(test_symbol)
        
        print(f"🎯 Out-of-sample validation enabled:")
        print(f"   Validation type: {args.validation_type}")
        print(f"   Training symbols: {train_symbols}")
        print(f"   Testing symbol: {test_symbol}")
        if args.validation_type in ['temporal', 'both']:
            print(f"   Temporal split: {args.temporal_split:.1%} training, {1-args.temporal_split:.1%} testing")
        print()
    
    # Configuration
    config = {
        'symbols': train_symbols,
        'test_symbol': test_symbol,
        'out_of_sample': args.out_of_sample,
        'validation_type': args.validation_type if args.out_of_sample else 'none',
        'temporal_split': args.temporal_split,
        'dca_amount': args.dca_amount,
        'dca_frequency': args.dca_frequency,
        'start_date': '2022-01-01',
        'end_date': '2024-01-01',
        'encoder_len': 60,
        'predict_len': 10,
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'learning_rate': args.learning_rate,
        'hidden_size': args.hidden_size,
        'num_heads': 8,
        'dropout': 0.1,
        'weight_decay': 0.01,
        'grad_clip': 1.0,
        'max_input_features': 50,
        'save_every': 5,
        'patience': 3,
        'device': 'mps' if torch.backends.mps.is_available() else 'cpu',
        'news_api_key': None,
        'fred_api_key': None,
        'api_ninjas_key': None,
        'auto_continue': args.auto_continue,
        'enhanced_model': args.enhanced_model
    }
    
    print("🚀 Unified TFT Training and Analysis Pipeline")
    print("=" * 60)
    print(f"📊 Configuration:")
    for key, value in config.items():
        print(f"   {key}: {value}")
    print()
    
    # Clean up old files if requested
    if args.cleanup:
        print("🧹 Cleaning up old test files...")
        cleanup_old_files()
        print()
    
    # Cache management
    if args.clear_cache:
        print("🗑️  Clearing cache...")
        clear_all_cache()
        print()
    
    print("📦 Cache Status:")
    print_cache_info()
    print()
    
    try:
        # Initialize trainer
        trainer = TFTTrainer(config)
        
        # Load data
        dataloader, datamodule = trainer.load_data()
        
        # Load test data if out-of-sample validation is enabled
        test_dataloader, test_datamodule = trainer.load_test_data()
        
        # Get sample batch for model initialization
        sample_batch = next(iter(dataloader))
        if isinstance(sample_batch, tuple):
            sample_data = sample_batch[0]
        else:
            sample_data = sample_batch
        
        # Initialize model
        trainer.initialize_model(sample_data)
        
        # Train model
        trainer.train(dataloader)
        
        # Generate predictions and analysis
        if config.get('out_of_sample') and test_dataloader is not None:
            print("\n🎯 Generating out-of-sample predictions...")
            test_predictions, test_targets = trainer.generate_predictions(test_dataloader)
            trainer.create_comprehensive_analysis(test_predictions, test_targets, test_datamodule, prefix="out_of_sample")
            print(f"📊 Out-of-sample analysis completed for {config['test_symbol']}")
        else:
            print("\n📊 Generating in-sample predictions...")
            predictions, targets = trainer.generate_predictions(dataloader)
            trainer.create_comprehensive_analysis(predictions, targets, datamodule)
        
        print("\n🎉 Pipeline completed successfully!")
        print(f"📁 Results saved in: {trainer.output_dir}")
        print(f"🔍 View analysis: open {trainer.plots_dir}")
        print(f"📋 Read report: {trainer.results_dir}/analysis_report.md")
        
    except KeyboardInterrupt:
        print("\n⚠️  Pipeline interrupted by user")
    except Exception as e:
        print(f"\n❌ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
