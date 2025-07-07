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
        
        dataloader, datamodule = get_data_loader_with_module(
            symbols=self.config['symbols'],
            start=self.config['start_date'],
            end=self.config['end_date'],
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
        
        test_dataloader, test_datamodule = get_data_loader_with_module(
            symbols=[self.config['test_symbol']],
            start=self.config['start_date'],
            end=self.config['end_date'],
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
                num_layers=3,
                num_decoder_layers=2
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
                print(f"Error in batch {batch_idx}: {e}")
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
                    print(f"Error in validation batch {batch_idx}: {e}")
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
                    print(f"Error in prediction batch: {e}")
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
        self.create_feature_analysis(datamodule)
        
        # Generate report
        self.generate_analysis_report(metrics, datamodule)
        
        # Save with prefix if provided
        if prefix:
            # Copy key plots with prefix
            import shutil
            try:
                shutil.copy2(self.plots_dir / 'trading_simulation.png', 
                           self.plots_dir / f'{prefix}_trading_simulation.png')
                shutil.copy2(self.plots_dir / 'model_performance.png', 
                           self.plots_dir / f'{prefix}_model_performance.png')
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
        """Plot advanced trading strategy simulation with sophisticated risk management."""
        # Simple trading strategy based on predictions
        pred_flat = predictions.flatten()
        target_flat = targets.flatten()
        
        # Ensure we have enough data
        if len(target_flat) < 2:
            print("Warning: Not enough data for trading simulation")
            return
        
        # Debug: Print some basic statistics
        print(f"Debug: Target values - Min: {np.min(target_flat):.4f}, Max: {np.max(target_flat):.4f}, Mean: {np.mean(target_flat):.4f}")
        print(f"Debug: Prediction values - Min: {np.min(pred_flat):.4f}, Max: {np.max(pred_flat):.4f}, Mean: {np.mean(pred_flat):.4f}")
        
        # Since the model outputs are percentage returns (from pct_change), we can use them directly
        # for both direction prediction and risk assessment to determine optimal portfolio allocation
        
        # Cap returns to realistic daily ranges (e.g., +/- 5% per day maximum)
        max_daily_return = 0.05  # 5% max daily return
        pred_returns = np.clip(pred_flat, -max_daily_return, max_daily_return)
        actual_returns = np.clip(target_flat, -max_daily_return, max_daily_return)
        
        # Calculate directional accuracy
        pred_direction = np.sign(pred_returns)
        actual_direction = np.sign(actual_returns)
        direction_correctness = (pred_direction == actual_direction).astype(float)
        
        # ===== ADVANCED RISK ASSESSMENT =====
        
        # 1. Prediction Confidence based on magnitude
        prediction_confidence = np.abs(pred_returns)
        
        # 2. Calculate rolling volatility for dynamic risk adjustment
        window = min(20, len(actual_returns) // 4)  # Use 20-day or 1/4 of data
        rolling_volatility = np.array([
            np.std(actual_returns[max(0, i-window):i+1]) if i >= window//2 
            else np.std(actual_returns[:window]) 
            for i in range(len(actual_returns))
        ])
        
        # 3. Market regime detection using volatility
        median_vol = np.median(rolling_volatility)
        high_vol_threshold = median_vol * 1.5
        low_vol_threshold = median_vol * 0.7
        
        market_regime = np.where(rolling_volatility > high_vol_threshold, 'high_vol',
                                np.where(rolling_volatility < low_vol_threshold, 'low_vol', 'normal'))
        
        # 4. Prediction Error Analysis (if we have enough history)
        if len(pred_returns) > 10:
            prediction_errors = np.abs(pred_returns - actual_returns)
            rolling_pred_accuracy = np.array([
                1 - np.mean(prediction_errors[max(0, i-10):i+1]) if i >= 5
                else 1 - np.mean(prediction_errors[:10])
                for i in range(len(prediction_errors))
            ])
            rolling_pred_accuracy = np.clip(rolling_pred_accuracy, 0.1, 0.9)  # Reasonable bounds
        else:
            rolling_pred_accuracy = np.full(len(pred_returns), 0.55)
        
        # Normalize confidence to create risk scores (0 to 1)
        if np.max(prediction_confidence) > 0:
            risk_scores = prediction_confidence / np.max(prediction_confidence)
        else:
            risk_scores = np.zeros_like(prediction_confidence)
        
        # ===== ENHANCED PORTFOLIO ALLOCATION =====
        
        # Estimate win probability from multiple factors
        base_win_prob = 0.52  # Slightly better than random
        confidence_boost = risk_scores * 0.15  # Up to 15% boost for high confidence
        accuracy_boost = (rolling_pred_accuracy - 0.5) * 0.2  # Historical accuracy adjustment
        estimated_win_prob = base_win_prob + confidence_boost + accuracy_boost
        estimated_win_prob = np.clip(estimated_win_prob, 0.45, 0.75)  # Reasonable bounds
        
        # Kelly Criterion with risk adjustments
        kelly_fractions = 2 * estimated_win_prob - 1
        kelly_fractions = np.clip(kelly_fractions, 0, 1)  # Don't go short or over-leverage
        
        # Risk adjustment based on market regime
        regime_multiplier = np.where(market_regime == 'high_vol', 0.5,      # Reduce in high volatility
                                   np.where(market_regime == 'low_vol', 1.2,  # Increase in low volatility
                                           1.0))                               # Normal in normal volatility
        
        # Conservative Kelly scaling with volatility adjustment
        conservative_factor = 0.25  # Base conservative factor
        volatility_adjustment = np.clip(1 / (1 + rolling_volatility * 10), 0.2, 1.0)  # Reduce when volatile
        
        position_fractions = kelly_fractions * conservative_factor * regime_multiplier * volatility_adjustment
        
        # Minimum and maximum position limits
        min_position = 0.0   # Can hold cash if no good opportunities
        max_position = 0.6   # Maximum 60% of portfolio in any single trade
        position_fractions = np.clip(position_fractions, min_position, max_position)
        
        # ===== STRATEGY RETURNS CALCULATION =====
        
        # Calculate strategy returns based on position fraction and actual returns
        # Position fraction determines how much of the portfolio is at risk
        strategy_returns = position_fractions * actual_returns * np.sign(pred_returns)
        
        # For buy & hold with DCA (Dollar Cost Averaging) - simulate regular cash inflows
        dca_amount = self.config.get('dca_amount', 100.0)  # Default $100 per inflow
        dca_frequency = self.config.get('dca_frequency', 1)  # Default daily
        
        # Calculate when cash inflows occur
        cash_inflow_days = np.arange(0, len(actual_returns), dca_frequency)
        cash_inflows = np.zeros(len(actual_returns))
        cash_inflows[cash_inflow_days] = dca_amount
        
        # For comparison, also calculate traditional lump-sum buy & hold
        lump_sum_buy_hold_returns = actual_returns
        
        # Debug: Print enhanced statistics
        print(f"Debug: Direction correctness rate: {np.mean(direction_correctness):.2%}")
        print(f"Debug: Risk scores - Min: {np.min(risk_scores):.4f}, Max: {np.max(risk_scores):.4f}, Mean: {np.mean(risk_scores):.4f}")
        print(f"Debug: Win probabilities - Min: {np.min(estimated_win_prob):.4f}, Max: {np.max(estimated_win_prob):.4f}, Mean: {np.mean(estimated_win_prob):.4f}")
        print(f"Debug: Position fractions - Min: {np.min(position_fractions):.4f}, Max: {np.max(position_fractions):.4f}, Mean: {np.mean(position_fractions):.4f}")
        print(f"Debug: Market regime - High Vol: {np.mean(market_regime == 'high_vol'):.1%}, Low Vol: {np.mean(market_regime == 'low_vol'):.1%}")
        print(f"Debug: Volatility - Min: {np.min(rolling_volatility):.4f}, Max: {np.max(rolling_volatility):.4f}, Mean: {np.mean(rolling_volatility):.4f}")
        print(f"Debug: DCA Cash Inflows - Amount: ${dca_amount:.2f}, Frequency: {dca_frequency} days")
        print(f"Debug: Total DCA inflows: {len(cash_inflow_days)}, Total amount: ${np.sum(cash_inflows):.2f}")
        print(f"Debug: Strategy returns - Min: {np.min(strategy_returns):.4f}, Max: {np.max(strategy_returns):.4f}, Mean: {np.mean(strategy_returns):.4f}")
        print(f"Debug: Lump-sum Buy & Hold returns - Min: {np.min(lump_sum_buy_hold_returns):.4f}, Max: {np.max(lump_sum_buy_hold_returns):.4f}, Mean: {np.mean(lump_sum_buy_hold_returns):.4f}")
        
        # ===== PORTFOLIO SIMULATION =====
        
        initial_capital = 10000
        # Remove artificial caps - let returns be what they are naturally
        max_portfolio_value = initial_capital * 500  # Higher emergency brake
        min_portfolio_value = initial_capital * 0.01   # Keep emergency brake
        
        # Calculate portfolio values over time
        portfolio_values = [initial_capital]
        
        # DCA Buy & Hold: Start with initial capital, add cash inflows regularly
        dca_buy_hold_values = [initial_capital]
        dca_buy_hold_cash = initial_capital  # Track uninvested cash
        dca_buy_hold_shares = 0  # Track shares owned
        
        # Traditional lump-sum buy & hold for comparison
        lump_sum_buy_hold_values = [initial_capital]
        
        # Simulate initial stock price and track it
        initial_stock_price = 100.0  # Assume $100 initial stock price
        stock_prices = [initial_stock_price]
        
        for i in range(len(strategy_returns)):
            # Strategy portfolio
            new_portfolio_value = portfolio_values[-1] * (1 + strategy_returns[i])
            new_portfolio_value = np.clip(new_portfolio_value, min_portfolio_value, max_portfolio_value)
            portfolio_values.append(new_portfolio_value)
            
            # Update stock price
            new_stock_price = stock_prices[-1] * (1 + lump_sum_buy_hold_returns[i])
            stock_prices.append(new_stock_price)
            
            # DCA Buy & Hold portfolio
            # Add cash inflow if it's a DCA day
            if i < len(cash_inflows) and cash_inflows[i] > 0:
                dca_buy_hold_cash += cash_inflows[i]
            
            # Buy shares with available cash (DCA approach)
            if dca_buy_hold_cash > 0:
                shares_to_buy = dca_buy_hold_cash / new_stock_price
                dca_buy_hold_shares += shares_to_buy
                dca_buy_hold_cash = 0  # All cash invested
            
            # Calculate portfolio value
            new_dca_value = dca_buy_hold_shares * new_stock_price + dca_buy_hold_cash
            new_dca_value = np.clip(new_dca_value, min_portfolio_value, max_portfolio_value)
            dca_buy_hold_values.append(new_dca_value)
            
            # Traditional lump-sum buy & hold
            new_lump_sum_value = lump_sum_buy_hold_values[-1] * (1 + lump_sum_buy_hold_returns[i])
            new_lump_sum_value = np.clip(new_lump_sum_value, min_portfolio_value, max_portfolio_value)
            lump_sum_buy_hold_values.append(new_lump_sum_value)
        
        # Convert to numpy arrays for easier handling
        portfolio_values = np.array(portfolio_values)
        dca_buy_hold_values = np.array(dca_buy_hold_values)
        lump_sum_buy_hold_values = np.array(lump_sum_buy_hold_values)
        
        # Calculate DCA returns for metrics (excluding additional cash contributions)
        dca_invested_amount = initial_capital + np.sum(cash_inflows)
        dca_buy_hold_returns = np.diff(dca_buy_hold_values) / dca_buy_hold_values[:-1]
        
        # Debug: Print portfolio statistics
        print(f"Debug: Portfolio values - Min: {np.min(portfolio_values):.2f}, Max: {np.max(portfolio_values):.2f}, Final: {portfolio_values[-1]:.2f}")
        print(f"Debug: DCA Buy & Hold - Invested: ${dca_invested_amount:.2f}, Final: ${dca_buy_hold_values[-1]:.2f}")
        print(f"Debug: Lump-sum Buy & Hold - Final: ${lump_sum_buy_hold_values[-1]:.2f}")
        print(f"Debug: DCA shares owned: {dca_buy_hold_shares:.4f}, Final stock price: ${stock_prices[-1]:.2f}")
        
        # ===== ADVANCED PERFORMANCE METRICS =====
        
        # Calculate additional risk metrics
        strategy_sharpe = np.mean(strategy_returns) / np.std(strategy_returns) if np.std(strategy_returns) > 0 else 0
        lump_sum_sharpe = np.mean(lump_sum_buy_hold_returns) / np.std(lump_sum_buy_hold_returns) if np.std(lump_sum_buy_hold_returns) > 0 else 0
        dca_sharpe = np.mean(dca_buy_hold_returns) / np.std(dca_buy_hold_returns) if len(dca_buy_hold_returns) > 0 and np.std(dca_buy_hold_returns) > 0 else 0
        
        # Sortino ratio (downside deviation)
        downside_strategy = strategy_returns[strategy_returns < 0]
        downside_lump_sum = lump_sum_buy_hold_returns[lump_sum_buy_hold_returns < 0]
        downside_dca = dca_buy_hold_returns[dca_buy_hold_returns < 0] if len(dca_buy_hold_returns) > 0 else np.array([])
        
        strategy_sortino = np.mean(strategy_returns) / np.std(downside_strategy) if len(downside_strategy) > 0 and np.std(downside_strategy) > 0 else 0
        lump_sum_sortino = np.mean(lump_sum_buy_hold_returns) / np.std(downside_lump_sum) if len(downside_lump_sum) > 0 and np.std(downside_lump_sum) > 0 else 0
        dca_sortino = np.mean(dca_buy_hold_returns) / np.std(downside_dca) if len(downside_dca) > 0 and np.std(downside_dca) > 0 else 0
        
        # Maximum drawdown
        max_drawdown_strategy = np.max(np.maximum.accumulate(portfolio_values) - portfolio_values) / np.max(portfolio_values) * 100
        max_drawdown_lump_sum = np.max(np.maximum.accumulate(lump_sum_buy_hold_values) - lump_sum_buy_hold_values) / np.max(lump_sum_buy_hold_values) * 100
        max_drawdown_dca = np.max(np.maximum.accumulate(dca_buy_hold_values) - dca_buy_hold_values) / np.max(dca_buy_hold_values) * 100
        
        # Total returns
        strategy_total_return = ((portfolio_values[-1] - initial_capital) / initial_capital) * 100
        lump_sum_total_return = ((lump_sum_buy_hold_values[-1] - initial_capital) / initial_capital) * 100
        dca_total_return = ((dca_buy_hold_values[-1] - dca_invested_amount) / dca_invested_amount) * 100
        
        # Calmar ratio (return / max drawdown)
        strategy_calmar = strategy_total_return / max_drawdown_strategy if max_drawdown_strategy > 0 else 0
        lump_sum_calmar = lump_sum_total_return / max_drawdown_lump_sum if max_drawdown_lump_sum > 0 else 0
        dca_calmar = dca_total_return / max_drawdown_dca if max_drawdown_dca > 0 else 0
        
        # Win rate and profit factor
        positive_trades = strategy_returns[strategy_returns > 0]
        negative_trades = strategy_returns[strategy_returns < 0]
        win_rate = len(positive_trades) / len(strategy_returns) if len(strategy_returns) > 0 else 0
        profit_factor = np.sum(positive_trades) / abs(np.sum(negative_trades)) if len(negative_trades) > 0 and np.sum(negative_trades) != 0 else np.inf
        
        # ===== PLOTTING =====
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Returns comparison with regime overlay
        time_indices = range(len(target_flat))
        ax1.plot(time_indices, actual_returns * 100, 'b-', label='Actual Returns (%)', linewidth=1, alpha=0.7)
        ax1.plot(time_indices, pred_returns * 100, 'r--', label='Predicted Returns (%)', linewidth=1, alpha=0.7)
        
        # Color-code background by market regime
        regime_colors = {'high_vol': 'red', 'low_vol': 'green', 'normal': 'gray'}
        start_idx = 0
        for i, regime in enumerate(market_regime):
            if i == 0 or market_regime[i-1] != regime:
                start_idx = i
            if i == len(market_regime)-1 or market_regime[i+1] != regime:
                end_idx = i
                ax1.axvspan(start_idx, end_idx, alpha=0.1, color=regime_colors[regime])
        
        ax1.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax1.set_title('Daily Returns: Predicted vs Actual\n(Red: High Vol, Green: Low Vol, Gray: Normal)', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Returns (%)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Portfolio value with drawdown
        time_indices_portfolio = range(len(portfolio_values))
        ax2.plot(time_indices_portfolio, [initial_capital] * len(time_indices_portfolio), 'g--', label='Initial Capital', linewidth=1)
        ax2.plot(time_indices_portfolio, portfolio_values, 'purple', label='Enhanced TFT Strategy', linewidth=2)
        ax2.plot(time_indices_portfolio, dca_buy_hold_values, 'orange', label='DCA Buy & Hold', linewidth=2)
        ax2.plot(time_indices_portfolio, lump_sum_buy_hold_values, 'blue', label='Lump-sum Buy & Hold', linewidth=2, alpha=0.7)
        
        # Add drawdown shading
        strategy_peak = np.maximum.accumulate(portfolio_values)
        strategy_drawdown = (strategy_peak - portfolio_values) / strategy_peak
        ax2.fill_between(time_indices_portfolio, portfolio_values, strategy_peak, 
                        where=(strategy_drawdown > 0), alpha=0.3, color='red', label='Drawdown')
        
        ax2.set_title('Portfolio Value Over Time with Drawdown', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Portfolio Value ($)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Strategy performance comparison
        strategies = ['Enhanced TFT', 'DCA Buy & Hold', 'Lump-sum B&H']
        returns = [strategy_total_return, dca_total_return, lump_sum_total_return]
        colors = ['red' if r < 0 else 'green' for r in returns]
        
        bars = ax3.bar(strategies, returns, color=colors, alpha=0.7)
        ax3.set_title('Strategy Returns Comparison', fontsize=14, fontweight='bold')
        ax3.set_ylabel('Return (%)')
        ax3.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        
        # Add value labels
        for bar, ret in zip(bars, returns):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.5 if height >= 0 else height - 1,
                    f'{ret:.2f}%', ha='center', va='bottom' if height >= 0 else 'top')
        
        # 4. Enhanced trading summary
        num_trades = len(strategy_returns)
        avg_position_fraction = np.mean(position_fractions)
        max_position_fraction_used = np.max(position_fractions)
        avg_win_prob = np.mean(estimated_win_prob)
        avg_volatility = np.mean(rolling_volatility)
        
        summary_text = f"""Enhanced Trading Performance Summary:

📊 RETURNS & RISK:
Starting Capital: ${initial_capital:,}
Final Portfolio Value: ${portfolio_values[-1]:,.2f} (Enhanced TFT)
Final DCA Buy & Hold Value: ${dca_buy_hold_values[-1]:,.2f}
Final Lump-sum B&H Value: ${lump_sum_buy_hold_values[-1]:,.2f}

💰 INVESTMENT DETAILS:
DCA Total Invested: ${dca_invested_amount:,.2f}
DCA Frequency: Every {dca_frequency} day(s)
DCA Amount per Investment: ${dca_amount:.2f}

📈 RETURNS:
TFT Strategy Return: {strategy_total_return:.2f}%
DCA Buy & Hold Return: {dca_total_return:.2f}%
Lump-sum B&H Return: {lump_sum_total_return:.2f}%
Outperformance vs DCA: {strategy_total_return - dca_total_return:.2f}%
Outperformance vs Lump-sum: {strategy_total_return - lump_sum_total_return:.2f}%

🎯 RISK METRICS:
Sharpe Ratio: {strategy_sharpe:.3f} vs {dca_sharpe:.3f} (DCA) vs {lump_sum_sharpe:.3f} (Lump-sum)
Sortino Ratio: {strategy_sortino:.3f} vs {dca_sortino:.3f} (DCA) vs {lump_sum_sortino:.3f} (Lump-sum)
Calmar Ratio: {strategy_calmar:.3f} vs {dca_calmar:.3f} (DCA) vs {lump_sum_calmar:.3f} (Lump-sum)

Max Drawdown: {max_drawdown_strategy:.2f}% vs {max_drawdown_dca:.2f}% (DCA) vs {max_drawdown_lump_sum:.2f}% (Lump-sum)

� TRADING METRICS:
Number of Trades: {num_trades}
Win Rate: {win_rate:.2%}
Profit Factor: {profit_factor:.2f}
Avg Daily Return: {np.mean(strategy_returns)*100:.3f}%

🔧 STRATEGY DETAILS:
Avg Position Fraction: {avg_position_fraction:.1%}
Max Position Fraction: {max_position_fraction_used:.1%}
Avg Win Probability: {avg_win_prob:.1%}
Avg Volatility: {avg_volatility:.3f}

Strategy: Enhanced Kelly Criterion with volatility adjustment
Risk Management: Dynamic position sizing (0-60%) + regime detection
Buy & Hold: DCA with ${dca_amount:.0f} every {dca_frequency} day(s)"""
        
        ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, fontsize=9,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightcyan'))
        ax4.set_title('Enhanced Trading Summary', fontsize=14, fontweight='bold')
        ax4.axis('off')
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'trading_simulation.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_feature_analysis(self, datamodule: Any) -> None:
        """Create feature importance and data analysis plots."""
        try:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
            
            # Dataset overview
            feature_df = datamodule.feature_df
            
            # Feature statistics
            numeric_cols = feature_df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                feature_stats = feature_df[numeric_cols].describe()
                
                # Plot feature distribution (sample of features)
                sample_features = numeric_cols[:10] if len(numeric_cols) > 10 else numeric_cols
                sample_data = feature_df[sample_features].iloc[:1000]  # Sample data for speed
                
                ax1.boxplot([sample_data[col].dropna() for col in sample_features], 
                           labels=sample_features, vert=True)
                ax1.set_title('Feature Distribution (Sample)', fontsize=14, fontweight='bold')
                ax1.set_ylabel('Value')
                ax1.tick_params(axis='x', rotation=45)
                ax1.grid(True, alpha=0.3)
            
            # Symbol distribution
            if 'symbol' in feature_df.columns:
                symbol_counts = feature_df['symbol'].value_counts()
                ax2.pie(symbol_counts.values, labels=symbol_counts.index, autopct='%1.1f%%')
                ax2.set_title('Data Distribution by Symbol', fontsize=14, fontweight='bold')
            elif 'sector' in feature_df.columns:
                sector_counts = feature_df['sector'].value_counts()
                ax2.pie(sector_counts.values, labels=sector_counts.index, autopct='%1.1f%%')
                ax2.set_title('Data Distribution by Sector', fontsize=14, fontweight='bold')
            
            # Data timeline
            if 'date' in feature_df.columns:
                feature_df['date'] = pd.to_datetime(feature_df['date'], errors='coerce')
                daily_counts = feature_df.groupby(feature_df['date'].dt.date).size()
                ax3.plot(daily_counts.index, daily_counts.values, linewidth=2)
                ax3.set_title('Data Points Over Time', fontsize=14, fontweight='bold')
                ax3.set_xlabel('Date')
                ax3.set_ylabel('Number of Records')
                ax3.tick_params(axis='x', rotation=45)
                ax3.grid(True, alpha=0.3)
            
            # Dataset summary
            summary_text = f"""Dataset Summary:

Total Records: {len(feature_df):,}
Features: {len(feature_df.columns)}
Symbols: {self.config['symbols']}
Date Range: {self.config['start_date']} to {self.config['end_date']}

Encoder Length: {self.config['encoder_len']}
Prediction Length: {self.config['predict_len']}
Batch Size: {self.config['batch_size']}

Tech Sector Focus: ✓
Caching Enabled: ✓
Multi-modal Features: ✓"""
            
            ax4.text(0.1, 0.5, summary_text, transform=ax4.transAxes, fontsize=11,
                    verticalalignment='center', bbox=dict(boxstyle='round', facecolor='lightyellow'))
            ax4.set_title('Dataset Summary', fontsize=14, fontweight='bold')
            ax4.axis('off')
            
            plt.tight_layout()
            plt.savefig(self.plots_dir / 'feature_analysis.png', dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            print(f"Warning: Could not create feature analysis plot: {e}")
    
    def generate_analysis_report(self, metrics: Dict[str, float], datamodule: Any) -> None:
        """Generate comprehensive analysis report."""
        report_path = self.results_dir / "analysis_report.md"
        
        with open(report_path, 'w') as f:
            f.write(f"""# TFT Training and Analysis Report

## Run Information
- **Run Name**: {self.run_name}
- **Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- **Device**: {self.config['device']}
- **Model**: Temporal Fusion Transformer (TFT)

## Configuration
```json
{json.dumps(self.config, indent=2, default=str)}
```

## Dataset Summary
- **Symbols**: {', '.join(self.config['symbols'])}
- **Date Range**: {self.config['start_date']} to {self.config['end_date']}
- **Total Records**: {len(datamodule.feature_df):,}
- **Features**: {len(datamodule.feature_df.columns)}
- **Encoder Length**: {self.config['encoder_len']} days
- **Prediction Length**: {self.config['predict_len']} days

## Model Performance

### Training Results
- **Total Epochs**: {len(self.train_losses)}
- **Final Training Loss**: {self.train_losses[-1]:.6f}
- **Best Training Loss**: {min(self.train_losses):.6f}
- **Total Parameters**: {sum(p.numel() for p in self.model.parameters()):,} if self.model else 'N/A'

### Prediction Metrics
- **MSE**: {metrics.get('mse', 'N/A'):.6f}
- **RMSE**: {metrics.get('rmse', 'N/A'):.6f}
- **MAE**: {metrics.get('mae', 'N/A'):.6f}
- **R²**: {metrics.get('r2', 'N/A'):.4f}
- **Directional Accuracy**: {metrics.get('directional_accuracy', 'N/A'):.2%}
- **Data Points**: {metrics.get('data_points', 'N/A'):,}

## Key Features
- ✅ **Multi-modal data**: Stock prices, news sentiment, economic indicators, technical analysis
- ✅ **Sector-based classification**: Tech sector mapping without symbol memorization
- ✅ **Comprehensive caching**: Intelligent data caching for faster iterations
- ✅ **Advanced visualization**: Training progress, prediction analysis, trading simulation
- ✅ **Robust training**: AdamW optimizer, cosine learning rate scheduling, gradient clipping

## Files Generated
- **Model Checkpoints**: `{self.checkpoints_dir}/`
- **Analysis Plots**: `{self.plots_dir}/`
- **Configuration**: `{self.output_dir}/config.json`
- **This Report**: `{report_path}`

## Usage
To load the trained model:
```python
import torch
from tft_multimodal import TFT

# Load checkpoint
checkpoint = torch.load('{self.checkpoints_dir}/best_model.pth')
config = checkpoint['config']

# Initialize model
model = TFT(
    input_size=config['max_input_features'],
    news_dim=768,
    hidden_size=config['hidden_size'],
    num_heads=config['num_heads'],
    dropout=config['dropout'],
    seq_len=config['encoder_len'],
    prediction_len=config['predict_len']
)

# Load weights
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
```

---
*Generated by Unified TFT Pipeline*
""")
        
        print(f"📋 Analysis report saved to {report_path}")


def cleanup_old_files():
    """Clean up old test files."""
    files_to_remove = [
        'debug_dataloader.py',
        'test_full_pipeline.py',
        'test_improvements.py',
        'train_quick.py',
        'train_simple.py',
        'train_full_pipeline.py',
        'cache_utils.py'
    ]
    
    removed_count = 0
    for file in files_to_remove:
        file_path = Path(file)
        if file_path.exists():
            file_path.unlink()
            removed_count += 1
            print(f"   Removed: {file}")
    
    # Remove old plots
    old_plots = ['training_loss.png']
    for plot in old_plots:
        plot_path = Path(plot)
        if plot_path.exists():
            plot_path.unlink()
            removed_count += 1
            print(f"   Removed: {plot}")
    
    # Remove old checkpoints
    old_checkpoints = list(Path('.').glob('checkpoint_epoch_*.pth')) + list(Path('.').glob('final_*.pth'))
    for checkpoint in old_checkpoints:
        checkpoint.unlink()
        removed_count += 1
        print(f"   Removed: {checkpoint}")
    
    print(f"🗑️  Cleaned up {removed_count} old files")


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
        print(f"   Training symbols: {train_symbols}")
        print(f"   Testing symbol: {test_symbol}")
        print()
    
    # Configuration
    config = {
        'symbols': train_symbols,
        'test_symbol': test_symbol,
        'out_of_sample': args.out_of_sample,
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
