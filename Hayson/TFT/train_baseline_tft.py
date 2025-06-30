#!/usr/bin/env python3
"""
Simple TFT Training Script with Baseline Features

This script demonstrates proper TFT training using the baseline implementation.
Uses simple features and proper TFT data structure.
"""

# =============================================================================
# CONFIGURATION - Define stock symbols here
# =============================================================================
# Default stock symbols to use throughout the pipeline
DEFAULT_SYMBOLS = ['NFLX']  # Primary symbol(s) for training - UPDATED
FALLBACK_SYMBOL = 'AAPL'   # Fallback symbol for evaluation functions

# You can easily change to other symbols like:
# DEFAULT_SYMBOLS = ['AAPL']
# DEFAULT_SYMBOLS = ['SPY'] 
# DEFAULT_SYMBOLS = ['MSFT', 'GOOGL']  # Multiple symbols
# =============================================================================

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

from baseline_tft import BaselineTFT, QuantileLoss, create_baseline_data


class TFTDataset(Dataset):
    """Dataset for TFT training."""
    
    def __init__(self, batch_data: Dict[str, torch.Tensor]):
        self.data = batch_data
        self.length = batch_data['encoder_cont'].shape[0]
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        return {key: value[idx] for key, value in self.data.items()}


def create_dataloaders(symbols: List[str], 
                      start_date: str, 
                      end_date: str,
                      encoder_length: int = 30,
                      prediction_length: int = 1,
                      batch_size: int = 32,
                      train_split: float = 0.8) -> Tuple[DataLoader, DataLoader]:
    """Create train and validation dataloaders."""
    
    print("📊 Creating dataloaders...")
    
    # Create dataset
    batch_data = create_baseline_data(
        symbols=symbols,
        start_date=start_date,
        end_date=end_date,
        encoder_length=encoder_length,
        prediction_length=prediction_length
    )
    
    # Split train/validation
    total_samples = batch_data['encoder_cont'].shape[0]
    train_size = int(total_samples * train_split)
    
    train_data = {key: value[:train_size] for key, value in batch_data.items()}
    val_data = {key: value[train_size:] for key, value in batch_data.items()}
    
    # Create datasets
    train_dataset = TFTDataset(train_data)
    val_dataset = TFTDataset(val_data)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False
    )
    
    print(f"   ✅ Train samples: {len(train_dataset)}")
    print(f"   ✅ Validation samples: {len(val_dataset)}")
    print(f"   ✅ Train batches: {len(train_loader)}")
    print(f"   ✅ Validation batches: {len(val_loader)}")
    
    return train_loader, val_loader


def create_dataloaders_with_options(symbols: List[str], 
                                   start_date: str, 
                                   end_date: str,
                                   encoder_length: int = 30,
                                   prediction_length: int = 1,
                                   batch_size: int = 32,
                                   train_split: float = 0.8,
                                   use_enhanced_pipeline: bool = False,
                                   news_api_key: Optional[str] = None,
                                   fred_api_key: Optional[str] = None,
                                   api_ninjas_key: Optional[str] = None) -> Tuple[DataLoader, DataLoader, Dict]:
    """Create dataloaders with option for enhanced pipeline or simple baseline."""
    
    if use_enhanced_pipeline:
        print("📊 Attempting enhanced dataModule pipeline...")
        try:
            # Import and use enhanced pipeline
            from dataModule.interface import get_data_loader_with_module
            
            train_loader, data_module = get_data_loader_with_module(
                symbols=symbols,
                start=start_date,
                end=end_date,
                encoder_len=encoder_length,
                predict_len=prediction_length,
                batch_size=batch_size,
                news_api_key=news_api_key,
                fred_api_key=fred_api_key,
                api_ninjas_key=api_ninjas_key
            )
            
            val_loader = data_module.val_dataloader()
            
            # Extract enhanced feature info
            train_dataset = data_module.train_dataset
            feature_info = {
                'static_categorical_cardinalities': [len(train_dataset.data[col].unique()) 
                                                   for col in train_dataset.static_categoricals],
                'num_static_real': len(train_dataset.static_reals),
                'num_time_varying_real_known': len(train_dataset.time_varying_known_reals),
                'num_time_varying_real_unknown': len(train_dataset.time_varying_unknown_reals),
                'pipeline_type': 'enhanced'
            }
            
            print(f"   ✅ Enhanced pipeline successful!")
            print(f"      Features: Static-Cat({len(train_dataset.static_categoricals)}), Static-Real({len(train_dataset.static_reals)})")
            print(f"                Known({len(train_dataset.time_varying_known_reals)}), Unknown({len(train_dataset.time_varying_unknown_reals)})")
            
            return train_loader, val_loader, feature_info
            
        except Exception as e:
            print(f"⚠️ Enhanced pipeline failed: {e}")
            print("🔄 Falling back to simple baseline...")
    
    # Use simple baseline pipeline
    print("📊 Using simple baseline pipeline...")
    train_loader, val_loader = create_dataloaders(
        symbols=symbols,
        start_date=start_date,
        end_date=end_date,
        encoder_length=encoder_length,
        prediction_length=prediction_length,
        batch_size=batch_size,
        train_split=train_split
    )
    
    # Simple feature info
    feature_info = {
        'static_categorical_cardinalities': [len(symbols)],
        'num_static_real': 1,
        'num_time_varying_real_known': 3,
        'num_time_varying_real_unknown': 8,
        'pipeline_type': 'simple_baseline'
    }
    
    return train_loader, val_loader, feature_info


def train_baseline_tft(symbols: List[str] = None,
                      start_date: str = '2020-01-01', 
                      end_date: str = '2023-12-31',
                      epochs: int = 50,
                      batch_size: int = 32,
                      learning_rate: float = 1e-3,
                      encoder_length: int = 30,
                      prediction_length: int = 1,
                      hidden_size: int = 64,
                      device: str = 'auto',
                      use_enhanced_pipeline: bool = False,
                      news_api_key: Optional[str] = None,
                      fred_api_key: Optional[str] = None,
                      api_ninjas_key: Optional[str] = None):
    """Train TFT model with integrated dataModule pipeline."""
    
    # Use default symbols if none provided
    if symbols is None:
        symbols = DEFAULT_SYMBOLS.copy()
    
    pipeline_type = "Enhanced DataModule" if use_enhanced_pipeline else "Simple Baseline"
    print(f"🚀 Starting TFT Training with {pipeline_type}")
    print("=" * 60)
    print(f"Symbols: {symbols}")
    print(f"Date range: {start_date} to {end_date}")
    print(f"Architecture: {hidden_size}D hidden, {encoder_length} encoder, {prediction_length} prediction")
    print(f"Training: {epochs} epochs, batch size {batch_size}, LR {learning_rate}")
    if use_enhanced_pipeline:
        print(f"🔗 DataModule Features: Stock + Technical + Events + Economic + News")
        api_status = f"News={'✓' if news_api_key else '✗'}, FRED={'✓' if fred_api_key else '✗'}, Ninjas={'✓' if api_ninjas_key else '✗'}"
        print(f"🔑 API Keys: {api_status}")
    else:
        print(f"📊 Simple Features: Basic OHLCV + Technical indicators")
    
    # Setup device
    if device == 'auto':
        if torch.backends.mps.is_available():
            device = 'mps'
        elif torch.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'
    
    device = torch.device(device)
    print(f"🔧 Using device: {device}")
    
    # Create dataloaders
    if use_enhanced_pipeline:
        train_loader, val_loader, feature_info = create_dataloaders_with_options(
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            encoder_length=encoder_length,
            prediction_length=prediction_length,
            batch_size=batch_size,
            news_api_key=news_api_key,
            fred_api_key=fred_api_key,
            api_ninjas_key=api_ninjas_key
        )
    else:
        train_loader, val_loader = create_dataloaders(
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            encoder_length=encoder_length,
            prediction_length=prediction_length,
            batch_size=batch_size
        )
        
        # Simple feature info
        feature_info = {
            'static_categorical_cardinalities': [len(symbols)],
            'num_static_real': 1,
            'num_time_varying_real_known': 3,
            'num_time_varying_real_unknown': 8,
            'pipeline_type': 'simple_baseline'
        }
    
    # Initialize model
    model = BaselineTFT(
        static_categorical_cardinalities=feature_info['static_categorical_cardinalities'],
        num_static_real=feature_info['num_static_real'],
        num_time_varying_real_known=feature_info['num_time_varying_real_known'],
        num_time_varying_real_unknown=feature_info['num_time_varying_real_unknown'],
        hidden_size=hidden_size,
        encoder_length=encoder_length,
        prediction_length=prediction_length,
        quantiles=[0.1, 0.25, 0.5, 0.75, 0.9]
    ).to(device)
    
    print(f"🏗️  Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss and optimizer
    criterion = QuantileLoss([0.1, 0.25, 0.5, 0.75, 0.9])
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10
    )
    
    # Training history
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    print("\n🏋️ Starting training...")
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_batches = 0
        
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]", leave=False)
        
        for batch in train_pbar:
            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # Check for NaN/inf in input batch
            input_valid = True
            for key, tensor in batch.items():
                if torch.isnan(tensor).any() or torch.isinf(tensor).any():
                    print(f"⚠️  Found NaN/inf values in input batch '{key}', skipping batch")
                    input_valid = False
                    break
            
            if not input_valid:
                continue
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(batch)
            loss = criterion(outputs['prediction'], batch['target'])
            
            # Check for NaN loss
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"⚠️  NaN/inf loss detected, skipping batch")
                print(f"   Prediction range: [{outputs['prediction'].min():.4f}, {outputs['prediction'].max():.4f}]")
                print(f"   Target range: [{batch['target'].min():.4f}, {batch['target'].max():.4f}]")
                continue
            
            # Backward pass
            loss.backward()
            
            # Check for NaN gradients
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                print(f"⚠️  NaN/inf gradients detected, skipping update")
                optimizer.zero_grad()
                continue
                
            optimizer.step()
            
            # Update metrics
            train_loss += loss.item()
            train_batches += 1
            
            # Update progress bar
            train_pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'avg_loss': f"{train_loss/train_batches:.4f}" if train_batches > 0 else "0.0000"
            })
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_batches = 0
        
        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]", leave=False)
        
        with torch.no_grad():
            for batch in val_pbar:
                batch = {k: v.to(device) for k, v in batch.items()}
                
                # Check for NaN/inf in validation batch
                input_valid = True
                for key, tensor in batch.items():
                    if torch.isnan(tensor).any() or torch.isinf(tensor).any():
                        input_valid = False
                        break
                
                if not input_valid:
                    continue
                
                outputs = model(batch)
                loss = criterion(outputs['prediction'], batch['target'])
                
                # Check for valid loss
                if not (torch.isnan(loss) or torch.isinf(loss)):
                    val_loss += loss.item()
                    val_batches += 1
                
                val_pbar.set_postfix({
                    'loss': f"{loss.item():.4f}" if not (torch.isnan(loss) or torch.isinf(loss)) else "NaN",
                    'avg_loss': f"{val_loss/val_batches:.4f}" if val_batches > 0 else "0.0000"
                })
        
        # Calculate average losses
        avg_train_loss = train_loss / max(train_batches, 1)
        avg_val_loss = val_loss / max(val_batches, 1)
        
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        
        # Learning rate scheduling
        scheduler.step(avg_val_loss)
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'best_baseline_tft.pth')
        
        # Print epoch summary
        print(f"Epoch {epoch+1:3d}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        
        # Early stopping check
        if epoch > 20 and avg_val_loss > min(val_losses[-10:]) * 1.1:
            print("Early stopping triggered!")
            break
    
    print("\n✅ Training completed!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    
    # Load best model if it exists
    import os
    if os.path.exists('best_baseline_tft.pth'):
        model.load_state_dict(torch.load('best_baseline_tft.pth'))
        print("📦 Loaded best model weights")
    else:
        print("⚠️  No saved model found, using current model state")
    
    # Create training plot
    create_training_plot(train_losses, val_losses)
    
    # Generate predictions and analysis
    analyze_model_performance(model, val_loader, device, symbols, start_date, end_date)
    
    return model, train_losses, val_losses


def create_training_plot(train_losses: List[float], val_losses: List[float]):
    """Create training progress plot."""
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    plt.plot(train_losses, label='Training Loss', color='blue', alpha=0.8)
    plt.plot(val_losses, label='Validation Loss', color='red', alpha=0.8)
    plt.title('Training Progress')
    plt.xlabel('Epoch')
    plt.ylabel('Quantile Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 2)
    plt.plot(np.log(train_losses), label='Log Training Loss', color='blue', alpha=0.8)
    plt.plot(np.log(val_losses), label='Log Validation Loss', color='red', alpha=0.8)
    plt.title('Training Progress (Log Scale)')
    plt.xlabel('Epoch')
    plt.ylabel('Log Quantile Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 3)
    improvement = np.array(val_losses[1:]) - np.array(val_losses[:-1])
    plt.plot(improvement, color='green', alpha=0.8)
    plt.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    plt.title('Validation Loss Improvement')
    plt.xlabel('Epoch')
    plt.ylabel('Loss Change')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 4)
    plt.text(0.1, 0.8, f'Final Training Loss: {train_losses[-1]:.4f}', transform=plt.gca().transAxes)
    plt.text(0.1, 0.7, f'Final Validation Loss: {val_losses[-1]:.4f}', transform=plt.gca().transAxes)
    plt.text(0.1, 0.6, f'Best Validation Loss: {min(val_losses):.4f}', transform=plt.gca().transAxes)
    plt.text(0.1, 0.5, f'Training Epochs: {len(train_losses)}', transform=plt.gca().transAxes)
    plt.text(0.1, 0.4, f'Overfitting Ratio: {val_losses[-1]/train_losses[-1]:.2f}', transform=plt.gca().transAxes)
    plt.title('Training Summary')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('baseline_tft_training.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("📊 Training plot saved as 'baseline_tft_training.png'")


def analyze_model_performance(model: BaselineTFT, val_loader: DataLoader, device: torch.device,
                            symbols: List[str] = None, start_date: str = '2020-01-01', 
                            end_date: str = '2023-12-31'):
    """Analyze model performance and create visualizations including absolute price reconstruction."""
    
    # Use default symbols if none provided
    if symbols is None:
        symbols = DEFAULT_SYMBOLS.copy()
    
    print("📈 Analyzing model performance...")
    
    # Compare batch vs sequential prediction methods
    comparison_results = compare_prediction_methods(
        model=model,
        val_loader=val_loader, 
        device=device,
        num_sequential_steps=50  # Predict 50 days sequentially
    )
    
    # Use sequential predictions for main analysis (no future peeking)
    predictions = comparison_results['sequential_predictions']
    targets = comparison_results['sequential_targets']
    attention_weights = np.array([h['attention'] for h in comparison_results['attention_history']])
    
    # Also keep batch results for comparison
    batch_predictions = comparison_results['batch_predictions']
    batch_targets = comparison_results['batch_targets']
    
    # Reconstruct absolute prices for visualization
    print("📊 Reconstructing absolute prices from returns...")
    try:
        # Get original price data for reconstruction
        import yfinance as yf
        
        # Use first symbol for price reconstruction demo
        symbol = symbols[0] if symbols else FALLBACK_SYMBOL
        ticker = yf.Ticker(symbol)
        hist = ticker.history(start=start_date, end=end_date)
        
        if not hist.empty:
            # Get closing prices and align with our predictions
            close_prices = hist['Close'].values
            
            # Calculate the starting price index for our validation set
            # Assuming we're using the latter part of the data for validation
            val_start_idx = int(len(close_prices) * 0.8)  # 80% train split
            
            # Get a reasonable number of samples for visualization
            viz_samples = min(len(predictions), len(targets))
            
            # Use actual closing prices as starting points for reconstruction
            if val_start_idx + viz_samples < len(close_prices):
                starting_prices = close_prices[val_start_idx:val_start_idx + viz_samples]
                
                # Reconstruct sequential predictions (no future peeking)
                seq_pred_prices = starting_prices * (1 + predictions[:viz_samples])
                seq_actual_prices = starting_prices * (1 + targets[:viz_samples])
                
                # Also reconstruct batch predictions for comparison
                batch_viz_samples = min(len(batch_predictions), viz_samples)
                batch_pred_prices = starting_prices[:batch_viz_samples] * (1 + batch_predictions[:batch_viz_samples])
                batch_actual_prices = starting_prices[:batch_viz_samples] * (1 + batch_targets[:batch_viz_samples])
                
                # Create comprehensive comparison plot
                plt.figure(figsize=(20, 15))
                
                # Main comparison: Sequential vs Batch predictions
                plt.subplot(4, 3, 1)
                time_points = range(len(seq_pred_prices))
                plt.plot(time_points, seq_actual_prices, label=f'Actual {symbol} Price', 
                        color='blue', linewidth=2, alpha=0.8)
                plt.plot(time_points, seq_pred_prices, label=f'Sequential Pred (No Peeking)', 
                        color='red', linewidth=2, alpha=0.8)
                plt.plot(time_points[:batch_viz_samples], batch_pred_prices, label=f'Batch Pred (With Peeking)', 
                        color='orange', linewidth=2, alpha=0.6, linestyle='--')
                plt.xlabel('Time Steps')
                plt.ylabel('Stock Price ($)')
                plt.title(f'{symbol} Sequential vs Batch Prediction Comparison')
                plt.legend()
                plt.grid(True, alpha=0.3)
                
                # Calculate metrics for sequential predictions
                seq_price_mae = np.mean(np.abs(seq_pred_prices - seq_actual_prices))
                seq_price_rmse = np.sqrt(np.mean((seq_pred_prices - seq_actual_prices)**2))
                seq_price_mape = np.mean(np.abs((seq_pred_prices - seq_actual_prices) / seq_actual_prices)) * 100
                
                # Calculate metrics for batch predictions
                batch_price_mae = np.mean(np.abs(batch_pred_prices - batch_actual_prices))
                batch_price_rmse = np.sqrt(np.mean((batch_pred_prices - batch_actual_prices)**2))
                batch_price_mape = np.mean(np.abs((batch_pred_prices - batch_actual_prices) / batch_actual_prices)) * 100
                
                plt.text(0.02, 0.98, f'Sequential (No Peeking):\nMAE: ${seq_price_mae:.2f}\nRMSE: ${seq_price_rmse:.2f}\nMAPE: {seq_price_mape:.1f}%\n\nBatch (With Peeking):\nMAE: ${batch_price_mae:.2f}\nRMSE: ${batch_price_rmse:.2f}\nMAPE: {batch_price_mape:.1f}%', 
                        transform=plt.gca().transAxes, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
                
                # Future Peeking Impact Visualization
                plt.subplot(4, 3, 2)
                metrics_comparison = ['MAE', 'RMSE', 'MAPE']
                sequential_vals = [seq_price_mae, seq_price_rmse, seq_price_mape]
                batch_vals = [batch_price_mae, batch_price_rmse, batch_price_mape]
                
                x = np.arange(len(metrics_comparison))
                width = 0.35
                
                plt.bar(x - width/2, sequential_vals, width, label='Sequential (No Peeking)', 
                       color='red', alpha=0.7)
                plt.bar(x + width/2, batch_vals, width, label='Batch (With Peeking)', 
                       color='orange', alpha=0.7)
                
                plt.xlabel('Metrics')
                plt.ylabel('Values')
                plt.title('Future Peeking Impact on Price Prediction')
                plt.xticks(x, metrics_comparison)
                plt.legend()
                plt.grid(True, alpha=0.3, axis='y')
                
                # Sequential prediction error distribution
                plt.subplot(4, 3, 3)
                seq_price_errors = seq_pred_prices - seq_actual_prices
                plt.hist(seq_price_errors, bins=20, alpha=0.7, color='lightcoral', edgecolor='black')
                plt.xlabel('Sequential Price Prediction Error ($)')
                plt.ylabel('Frequency')
                plt.title('Sequential Prediction Error Distribution')
                plt.axvline(x=0, color='red', linestyle='--', alpha=0.8)
                plt.grid(True, alpha=0.3)
                
                print(f"✅ Price reconstruction completed for {symbol}")
                print(f"   Sequential (No Peeking) - MAE: ${seq_price_mae:.2f}, RMSE: ${seq_price_rmse:.2f}, MAPE: {seq_price_mape:.1f}%")
                print(f"   Batch (With Peeking) - MAE: ${batch_price_mae:.2f}, RMSE: ${batch_price_rmse:.2f}, MAPE: {batch_price_mape:.1f}%")
                print(f"   Performance degradation: MAE +{((seq_price_mae-batch_price_mae)/batch_price_mae*100):.1f}%, RMSE +{((seq_price_rmse-batch_price_rmse)/batch_price_rmse*100):.1f}%")
                
                # Reconstruct predicted prices: P_t+1 = P_t * (1 + return_t+1)
                pred_prices = starting_prices * (1 + predictions[:viz_samples])
                actual_prices = starting_prices * (1 + targets[:viz_samples])
                
                # Create absolute price comparison plot
                plt.figure(figsize=(18, 12))
                
                # Original plot layout (2x3) + new absolute price plot (make it 3x3)
                
                # Absolute Price Comparison - Main feature
                plt.subplot(3, 3, 1)
                time_points = range(len(pred_prices))
                plt.plot(time_points, actual_prices, label=f'Actual {symbol} Price', 
                        color='blue', linewidth=2, alpha=0.8)
                plt.plot(time_points, pred_prices, label=f'Predicted {symbol} Price', 
                        color='red', linewidth=2, alpha=0.8)
                plt.xlabel('Time Steps')
                plt.ylabel('Stock Price ($)')
                plt.title(f'{symbol} Predicted vs Actual Prices (Absolute)')
                plt.legend()
                plt.grid(True, alpha=0.3)
                
                # Calculate price-based metrics
                price_mae = np.mean(np.abs(pred_prices - actual_prices))
                price_rmse = np.sqrt(np.mean((pred_prices - actual_prices)**2))
                price_mape = np.mean(np.abs((pred_prices - actual_prices) / actual_prices)) * 100
                
                plt.text(0.02, 0.98, f'Price MAE: ${price_mae:.2f}\nPrice RMSE: ${price_rmse:.2f}\nPrice MAPE: {price_mape:.1f}%', 
                        transform=plt.gca().transAxes, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
                
                # Price Error Distribution
                plt.subplot(3, 3, 2)
                price_errors = pred_prices - actual_prices
                plt.hist(price_errors, bins=20, alpha=0.7, color='lightcoral', edgecolor='black')
                plt.xlabel('Price Prediction Error ($)')
                plt.ylabel('Frequency')
                plt.title('Price Prediction Error Distribution')
                plt.axvline(x=0, color='red', linestyle='--', alpha=0.8)
                plt.grid(True, alpha=0.3)
                
                # Percentage Error in Prices
                plt.subplot(3, 3, 3)
                pct_errors = (pred_prices - actual_prices) / actual_prices * 100
                plt.scatter(actual_prices, pct_errors, alpha=0.6, s=20, color='orange')
                plt.axhline(y=0, color='red', linestyle='--', alpha=0.8)
                plt.xlabel('Actual Price ($)')
                plt.ylabel('Percentage Error (%)')
                plt.title('Price Prediction Percentage Error')
                plt.grid(True, alpha=0.3)
                
                print(f"✅ Price reconstruction completed for {symbol}")
                print(f"   Price MAE: ${price_mae:.2f}")
                print(f"   Price RMSE: ${price_rmse:.2f}")
                print(f"   Price MAPE: {price_mape:.1f}%")
            else:
                print("⚠️  Not enough validation data for price reconstruction")
                plt.figure(figsize=(15, 12))
        else:
            print(f"⚠️  Could not fetch price data for {symbol}")
            plt.figure(figsize=(15, 12))
    except Exception as e:
        print(f"⚠️  Error in price reconstruction: {e}")
        print("📊 Continuing with returns-based analysis...")
        plt.figure(figsize=(15, 12))
    
    # Continue with existing analysis (adjust subplot positions)
    subplot_start = 4 if 'seq_pred_prices' in locals() else 1
    subplot_layout = (4, 3) if 'seq_pred_prices' in locals() else (2, 3)
    subplot_layout = (3, 3) if 'pred_prices' in locals() else (2, 3)
    
    # Predictions vs Targets
    plt.subplot(*subplot_layout, subplot_start)
    plt.scatter(targets, predictions, alpha=0.6, s=20)
    min_val = min(np.min(targets), np.min(predictions))
    max_val = max(np.max(targets), np.max(predictions))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)
    plt.xlabel('Actual Returns')
    plt.ylabel('Predicted Returns')
    plt.title('Predictions vs Actuals (Returns)')
    plt.grid(True, alpha=0.3)
    
    # Calculate R²
    correlation = np.corrcoef(targets, predictions)[0, 1]
    r_squared = correlation ** 2
    plt.text(0.05, 0.95, f'R² = {r_squared:.3f}', transform=plt.gca().transAxes,
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    # Residuals
    plt.subplot(*subplot_layout, subplot_start + 1)
    residuals = predictions - targets
    plt.scatter(predictions, residuals, alpha=0.6, s=20)
    plt.axhline(y=0, color='red', linestyle='--', alpha=0.8)
    plt.xlabel('Predicted Returns')
    plt.ylabel('Residuals')
    plt.title('Residuals Plot')
    plt.grid(True, alpha=0.3)
    
    # Error distribution
    plt.subplot(*subplot_layout, subplot_start + 2)
    plt.hist(residuals, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    plt.xlabel('Residuals')
    plt.ylabel('Frequency')
    plt.title('Error Distribution')
    plt.axvline(x=0, color='red', linestyle='--', alpha=0.8)
    plt.grid(True, alpha=0.3)
    
    # Time series comparison (sequential predictions)
    plt.subplot(*subplot_layout, subplot_start + 3)
    sample_size = min(len(targets), len(predictions))
    plt.plot(targets[:sample_size], label='Actual Returns', color='blue', alpha=0.8)
    plt.plot(predictions[:sample_size], label='Sequential Predictions (No Peeking)', color='red', alpha=0.8)
    plt.title('Sequential Prediction: Returns Time Series')
    plt.xlabel('Time Steps')
    plt.ylabel('Returns')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Attention heatmap (average across samples)
    plt.subplot(*subplot_layout, subplot_start + 4)
    avg_attention = np.mean(attention_weights, axis=0)
    plt.imshow(avg_attention, cmap='Blues', aspect='auto')
    plt.colorbar()
    plt.xlabel('Encoder Time Steps')
    plt.ylabel('Decoder Time Steps')
    plt.title('Average Attention Weights')
    
    # Performance metrics
    plt.subplot(*subplot_layout, subplot_start + 5)
    mae = np.mean(np.abs(residuals))
    rmse = np.sqrt(np.mean(residuals**2))
    mape = np.mean(np.abs(residuals / (np.abs(targets) + 1e-8))) * 100
    
    # Directional accuracy
    actual_direction = np.sign(targets[1:]) 
    pred_direction = np.sign(predictions[1:])
    directional_accuracy = np.mean(actual_direction == pred_direction) * 100
    
    metrics_text = f'''Sequential Prediction Metrics (No Future Peeking):

MAE: {mae:.4f}
RMSE: {rmse:.4f}
MAPE: {mape:.2f}%
R²: {r_squared:.3f}
Correlation: {correlation:.3f}
Directional Accuracy: {directional_accuracy:.1f}%

Data Points: {len(predictions)}
Model: Baseline TFT (Sequential)
Prediction Method: Walk-forward, No Future Peeking
'''
    
    plt.text(0.1, 0.5, metrics_text, fontsize=10, verticalalignment='center',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('baseline_tft_analysis_sequential.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("📊 Performance analysis saved as 'baseline_tft_analysis_sequential.png'")
    print(f"📈 Sequential Prediction Performance Summary (No Future Peeking):")
    print(f"   MAE: {mae:.4f}")
    print(f"   RMSE: {rmse:.4f}")
    print(f"   R²: {r_squared:.3f}")
    print(f"   Directional Accuracy: {directional_accuracy:.1f}%")
    print(f"   Sequential Steps: {len(predictions)}")
    
    # Show comparison with batch method
    print(f"\n🔍 Future Peeking Impact Summary:")
    for metric, degradation in comparison_results['degradation'].items():
        print(f"   {metric.upper()} degradation: +{degradation:.1f}%")
    
    # Optional: Run trading simulation on sequential predictions
    try:
        trading_results = simulate_trading_strategy(predictions, targets)
        print(f"🏆 Sequential trading simulation completed successfully!")
    except Exception as e:
        print(f"⚠️  Trading simulation failed: {e}")


def simulate_trading_strategy(predictions: np.ndarray, actuals: np.ndarray, 
                            initial_capital: float = 10000.0, 
                            buy_threshold: float = 0.001,
                            sell_threshold: float = -0.001) -> Dict:
    """
    Simulate a simple trading strategy based on TFT predictions.
    
    Args:
        predictions: Predicted returns
        actuals: Actual returns
        initial_capital: Starting portfolio value
        buy_threshold: Minimum predicted return to trigger buy
        sell_threshold: Maximum predicted return to trigger sell
        
    Returns:
        Dictionary with trading simulation results
    """
    print("💰 Running trading simulation...")
    
    # Convert returns to prices for visualization
    initial_price = 100.0
    actual_prices = [initial_price]
    pred_prices = [initial_price]
    
    for i in range(len(actuals)):
        actual_prices.append(actual_prices[-1] * (1 + actuals[i]))
        pred_prices.append(pred_prices[-1] * (1 + predictions[i]))
    
    actual_prices = np.array(actual_prices[1:])
    pred_prices = np.array(pred_prices[1:])
    
    # Trading simulation
    portfolio_value = initial_capital
    position = 0  # 0=cash, 1=long, -1=short
    trades = []
    portfolio_values = [portfolio_value]
    
    for i in range(len(predictions)):
        if i < len(predictions) - 1:
            predicted_return = predictions[i]
            actual_return = actuals[i]
            
            # Trading decisions based on predicted returns
            if predicted_return > buy_threshold and position != 1:
                if position == -1:
                    trades.append(('close_short', i, actual_prices[i]))
                trades.append(('buy', i, actual_prices[i]))
                position = 1
            elif predicted_return < sell_threshold and position != -1:
                if position == 1:
                    trades.append(('sell', i, actual_prices[i]))
                trades.append(('short', i, actual_prices[i]))
                position = -1
            
            # Update portfolio value based on actual returns
            if position == 1:  # Long position
                portfolio_value *= (1 + actual_return)
            elif position == -1:  # Short position
                portfolio_value *= (1 - actual_return)
            
            portfolio_values.append(portfolio_value)
    
    # Calculate performance metrics
    total_return = (portfolio_value - initial_capital) / initial_capital * 100
    buy_hold_return = (actual_prices[-1] - actual_prices[0]) / actual_prices[0] * 100
    
    # Create trading visualization
    plt.figure(figsize=(15, 10))
    
    # Price chart with trades
    plt.subplot(2, 2, 1)
    plt.plot(actual_prices, label='Actual Price', color='blue', linewidth=2, alpha=0.8)
    plt.plot(pred_prices, label='Predicted Price', color='red', linewidth=2, alpha=0.6, linestyle='--')
    
    # Mark trades on the chart
    for trade_type, idx, price in trades:
        if 'buy' in trade_type:
            plt.scatter(idx, price, color='green', marker='^', s=100, alpha=0.8, label='Buy' if idx == trades[0][1] else "")
        elif 'sell' in trade_type or 'short' in trade_type:
            plt.scatter(idx, price, color='red', marker='v', s=100, alpha=0.8, label='Sell/Short' if idx == trades[0][1] else "")
    
    plt.title('Trading Strategy Performance', fontsize=14, fontweight='bold')
    plt.xlabel('Time Steps')
    plt.ylabel('Stock Price ($)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Portfolio value over time
    plt.subplot(2, 2, 2)
    plt.plot(portfolio_values, color='green', linewidth=2)
    plt.axhline(y=initial_capital, color='gray', linestyle='--', alpha=0.8, label='Initial Capital')
    plt.title('Portfolio Value Over Time', fontsize=14, fontweight='bold')
    plt.xlabel('Time Steps')
    plt.ylabel('Portfolio Value ($)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Performance comparison
    plt.subplot(2, 2, 3)
    strategies = ['TFT Strategy', 'Buy & Hold']
    returns = [total_return, buy_hold_return]
    colors = ['green' if total_return > buy_hold_return else 'red', 'blue']
    
    bars = plt.bar(strategies, returns, color=colors, alpha=0.7)
    plt.title('Strategy Performance Comparison', fontsize=14, fontweight='bold')
    plt.ylabel('Total Return (%)')
    plt.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, value in zip(bars, returns):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                f'{value:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    # Trading statistics
    plt.subplot(2, 2, 4)
    stats_text = f'''Trading Statistics:

Total Trades: {len(trades)}
Final Portfolio: ${portfolio_value:.2f}
Total Return: {total_return:.2f}%
Buy & Hold Return: {buy_hold_return:.2f}%
Excess Return: {total_return - buy_hold_return:.2f}%

Max Predicted Return: {np.max(predictions):.3f}
Min Predicted Return: {np.min(predictions):.3f}
Strategy Alpha: {total_return - buy_hold_return:.2f}%
'''
    
    plt.text(0.1, 0.5, stats_text, fontsize=10, verticalalignment='center',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('baseline_tft_trading_simulation_sequential.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"💰 Sequential Trading Simulation Results:")
    print(f"   Strategy Return: {total_return:.2f}%")
    print(f"   Buy & Hold Return: {buy_hold_return:.2f}%")
    print(f"   Excess Return: {total_return - buy_hold_return:.2f}%")
    print(f"   Total Trades: {len(trades)}")
    print("📊 Trading simulation saved as 'baseline_tft_trading_simulation_sequential.png'")
    
    return {
        'portfolio_value': portfolio_value,
        'total_return': total_return,
        'buy_hold_return': buy_hold_return,
        'excess_return': total_return - buy_hold_return,
        'trades': trades,
        'portfolio_values': portfolio_values
    }


def create_sequential_batch(encoder_cont: torch.Tensor, 
                           decoder_cont: torch.Tensor,
                           static_cat: torch.Tensor,
                           static_real: torch.Tensor) -> Dict[str, torch.Tensor]:
    """Create a single batch for sequential prediction."""
    batch = {
        'encoder_cont': encoder_cont.unsqueeze(0),  # Add batch dimension
        'decoder_cont': decoder_cont.unsqueeze(0),
        'static_categorical': static_cat.unsqueeze(0),
        'static_real': static_real.unsqueeze(0)
    }
    return batch


def update_sequence_with_prediction(encoder_cont: torch.Tensor,
                                   decoder_cont: torch.Tensor,
                                   predicted_return: float,
                                   current_step: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Update the sequence by rolling forward one step and incorporating the prediction.
    
    Note: encoder_cont has 8 features, decoder_cont has 3 features.
    We need to properly construct the new encoder timestep from available information.
    
    Args:
        encoder_cont: Current encoder continuous features [seq_len, num_encoder_features]
        decoder_cont: Current decoder continuous features [pred_len, num_decoder_features]
        predicted_return: The predicted return value
        current_step: Current prediction step for time indexing
        
    Returns:
        Updated encoder_cont, decoder_cont for next prediction
    """
    # Encoder features: ['open_norm', 'high_norm', 'low_norm', 'close_norm', 
    #                   'volume_norm', 'returns_norm', 'sma_20_norm', 'rsi_14_norm']
    # Decoder features: ['day_of_week_norm', 'month_norm', 'time_idx_norm']
    
    # Roll encoder sequence forward (remove first timestep)
    new_encoder_cont = encoder_cont[1:].clone()  # [29, 8]
    
    # Create new timestep for encoder by extending the last encoder timestep
    # We'll use the last known values and update what we can predict
    last_encoder_timestep = encoder_cont[-1].clone()  # [8]
    
    # Update the return value (index 5 in encoder features)
    last_encoder_timestep[5] = predicted_return  # returns_norm
    
    # For OHLC features, we can simulate a simple price update based on the predicted return
    # Assume close price changes by the predicted return, and OHLC follow simple patterns
    if predicted_return != 0:
        # Update close price (index 3) based on return
        last_encoder_timestep[3] = torch.clamp(last_encoder_timestep[3] * (1 + predicted_return), 0, 1)
        
        # Update open (index 0) to be close to previous close
        last_encoder_timestep[0] = last_encoder_timestep[3]
        
        # Update high (index 1) and low (index 2) with some simple logic
        last_encoder_timestep[1] = max(last_encoder_timestep[0], last_encoder_timestep[3])  # high
        last_encoder_timestep[2] = min(last_encoder_timestep[0], last_encoder_timestep[3])  # low
    
    # For technical indicators (SMA, RSI), keep them similar to last timestep for simplicity
    # In practice, you'd want to recalculate these based on new price data
    
    # Add the new timestep to encoder
    new_encoder_cont = torch.cat([
        new_encoder_cont,  # [29, 8]
        last_encoder_timestep.unsqueeze(0)  # [1, 8]
    ], dim=0)  # [30, 8]
    
    # Update decoder for next prediction
    new_decoder_cont = decoder_cont.clone()
    
    # Update time index in decoder (index 2)
    # Increment by 1 day
    current_time_idx = decoder_cont[0, 2].item()
    new_decoder_cont[0, 2] = min(current_time_idx + 0.001, 1.0)  # Small increment, capped at 1.0
    
    return new_encoder_cont, new_decoder_cont


def sequential_prediction(model: BaselineTFT, 
                         initial_data: Dict[str, torch.Tensor],
                         num_steps: int,
                         device: torch.device) -> Tuple[np.ndarray, List[Dict]]:
    """
    Perform true sequential prediction without future peeking.
    
    Args:
        model: Trained TFT model
        initial_data: Initial 30-day sequence to start prediction
        num_steps: Number of future steps to predict
        device: Device to run inference on
        
    Returns:
        predictions: Array of predicted returns
        attention_history: List of attention weights for each step
    """
    print(f"🔮 Performing sequential prediction for {num_steps} steps...")
    
    model.eval()
    predictions = []
    attention_history = []
    
    # Extract initial sequences (take first sample from batch)
    current_encoder_cont = initial_data['encoder_cont'][0].clone()  # [30, num_features]
    current_decoder_cont = initial_data['decoder_cont'][0].clone()  # [1, num_features]
    current_static_cat = initial_data['static_categorical'][0].clone()
    current_static_real = initial_data['static_real'][0].clone()
    
    with torch.no_grad():
        for step in range(num_steps):
            # Create batch for current sequence
            batch = create_sequential_batch(
                current_encoder_cont,
                current_decoder_cont,
                current_static_cat,
                current_static_real
            )
            
            # Move to device
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # Get prediction
            outputs = model(batch)
            
            # Extract prediction (use median quantile 0.5)
            pred_quantiles = outputs['prediction'][0, 0]  # [num_quantiles]
            predicted_return = pred_quantiles[2].item()  # Middle quantile (0.5)
            
            predictions.append(predicted_return)
            
            # Store attention weights
            attention_weights = outputs['attention_weights'][0].cpu().numpy()
            attention_history.append({
                'step': step,
                'attention': attention_weights,
                'prediction': predicted_return
            })
            
            # Update sequence for next prediction (if not last step)
            if step < num_steps - 1:
                current_encoder_cont, current_decoder_cont = update_sequence_with_prediction(
                    current_encoder_cont,
                    current_decoder_cont, 
                    predicted_return,
                    step
                )
            
            if (step + 1) % 10 == 0:
                print(f"   Step {step + 1}/{num_steps}: Predicted return = {predicted_return:.4f}")
    
    print(f"✅ Sequential prediction completed!")
    return np.array(predictions), attention_history


def get_sequential_validation_data(val_loader: DataLoader, device: torch.device) -> Tuple[Dict, np.ndarray]:
    """
    Extract initial sequence and true future returns for sequential validation.
    
    Returns:
        initial_data: First sequence from validation set
        true_returns: Array of true future returns for comparison
    """
    print("📊 Extracting sequential validation data...")
    
    # Get all validation data
    all_data = []
    all_targets = []
    
    for batch in val_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        all_data.append(batch)
        all_targets.append(batch['target'].cpu().numpy())
    
    # Use first batch, first sequence as initial data
    initial_data = {k: v[:1] for k, v in all_data[0].items()}  # Take first sample only
    
    # Collect all targets for comparison (this represents the "true future")
    true_returns = np.concatenate(all_targets, axis=0).flatten()
    
    print(f"   Initial sequence shape: {initial_data['encoder_cont'].shape}")
    print(f"   True returns available: {len(true_returns)}")
    
    return initial_data, true_returns


def compare_prediction_methods(model: BaselineTFT, 
                              val_loader: DataLoader, 
                              device: torch.device,
                              num_sequential_steps: int = 50) -> Dict:
    """
    Compare traditional batch prediction vs sequential prediction to show future peeking impact.
    
    Returns:
        Dictionary with comparison results
    """
    print("🔍 Comparing batch vs sequential prediction methods...")
    
    # Method 1: Traditional batch prediction (with future peeking)
    print("\n1️⃣ Traditional batch prediction (with future peeking):")
    model.eval()
    batch_predictions = []
    batch_targets = []
    
    with torch.no_grad():
        for batch in val_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(batch)
            
            # Use median quantile
            pred = outputs['prediction'][:, :, 2].cpu().numpy().flatten()
            target = batch['target'].cpu().numpy().flatten()
            
            batch_predictions.extend(pred)
            batch_targets.extend(target)
    
    batch_predictions = np.array(batch_predictions)
    batch_targets = np.array(batch_targets)
    
    # Method 2: Sequential prediction (no future peeking)
    print("\n2️⃣ Sequential prediction (no future peeking):")
    initial_data, true_returns = get_sequential_validation_data(val_loader, device)
    
    # Limit sequential steps to available data
    max_steps = min(num_sequential_steps, len(true_returns))
    
    sequential_predictions, attention_history = sequential_prediction(
        model, initial_data, max_steps, device
    )
    
    # Use first max_steps of true returns for comparison
    sequential_targets = true_returns[:max_steps]
    
    # Calculate metrics for both methods
    def calculate_metrics(predictions, targets, method_name):
        mae = np.mean(np.abs(predictions - targets))
        rmse = np.sqrt(np.mean((predictions - targets)**2))
        corr = np.corrcoef(predictions, targets)[0, 1] if len(predictions) > 1 else 0
        
        # Directional accuracy
        pred_direction = np.sign(predictions[1:])
        true_direction = np.sign(targets[1:]) 
        dir_acc = np.mean(pred_direction == true_direction) * 100 if len(predictions) > 1 else 0
        
        print(f"   {method_name} Metrics:")
        print(f"     MAE: {mae:.4f}")
        print(f"     RMSE: {rmse:.4f}")
        print(f"     Correlation: {corr:.3f}")
        print(f"     Directional Accuracy: {dir_acc:.1f}%")
        print(f"     Samples: {len(predictions)}")
        
        return {
            'mae': mae,
            'rmse': rmse, 
            'correlation': corr,
            'directional_accuracy': dir_acc,
            'samples': len(predictions)
        }
    
    # Calculate metrics
    batch_metrics = calculate_metrics(
        batch_predictions[:max_steps], 
        batch_targets[:max_steps], 
        "Batch (Future Peeking)"
    )
    
    sequential_metrics = calculate_metrics(
        sequential_predictions,
        sequential_targets,
        "Sequential (No Peeking)"
    )
    
    # Performance degradation analysis
    mae_degradation = (sequential_metrics['mae'] - batch_metrics['mae']) / batch_metrics['mae'] * 100
    rmse_degradation = (sequential_metrics['rmse'] - batch_metrics['rmse']) / batch_metrics['rmse'] * 100
    corr_degradation = (batch_metrics['correlation'] - sequential_metrics['correlation']) / abs(batch_metrics['correlation']) * 100
    
    print(f"\n📊 Future Peeking Impact:")
    print(f"   MAE degradation: +{mae_degradation:.1f}%")
    print(f"   RMSE degradation: +{rmse_degradation:.1f}%") 
    print(f"   Correlation degradation: -{corr_degradation:.1f}%")
    
    return {
        'batch_predictions': batch_predictions[:max_steps],
        'batch_targets': batch_targets[:max_steps],
        'batch_metrics': batch_metrics,
        'sequential_predictions': sequential_predictions,
        'sequential_targets': sequential_targets,
        'sequential_metrics': sequential_metrics,
        'attention_history': attention_history,
        'degradation': {
            'mae': mae_degradation,
            'rmse': rmse_degradation,
            'correlation': corr_degradation
        }
    }


if __name__ == "__main__":
    print("🎯 Enhanced TFT Training Script with DataModule Pipeline")
    print("=" * 60)
    
    # Load API keys from environment
    import os
    from dotenv import load_dotenv
    load_dotenv()
    
    # Configuration - Using quantitative baseline by default
    config = {
        'symbols': DEFAULT_SYMBOLS,  # Use centralized configuration
        'start_date': '2020-01-01',
        'end_date': '2023-12-31',
        'epochs': 30,
        'batch_size': 32,
        'learning_rate': 1e-3,
        'encoder_length': 30,
        'prediction_length': 1,
        'hidden_size': 64,
        # Enhanced DataModule options
        'use_enhanced_pipeline': False,  # Using quantitative baseline by default
        'news_api_key': os.getenv('NEWS_API_KEY'),
        'fred_api_key': os.getenv('FRED_API_KEY'), 
        'api_ninjas_key': os.getenv('API_NINJAS_KEY')
    }
    
    print("Configuration:")
    for key, value in config.items():
        if 'api_key' in key:
            print(f"  {key}: {'[SET]' if value else '[NOT SET]'}")
        else:
            print(f"  {key}: {value}")
    
    print(f"\n📊 Data Pipeline: {'Enhanced DataModule' if config['use_enhanced_pipeline'] else 'Simple Baseline'}")
    if config['use_enhanced_pipeline']:
        print("   ✓ Stock OHLCV data (yfinance)")
        print("   ✓ Technical indicators (22 indicators)")
        print("   ✓ Corporate events (earnings, splits, dividends)")
        print(f"   {'✓' if config['fred_api_key'] else '✗'} Economic indicators (FRED)")
        print(f"   {'✓' if config['news_api_key'] else '✗'} News sentiment (BERT embeddings)")
        print("   ✓ Advanced feature engineering")
    
    try:
        # Train model with enhanced dataModule pipeline
        model, train_losses, val_losses = train_baseline_tft(**config)
        
        print("\n🎉 Training completed successfully!")
        print("Files generated:")
        print("  📄 best_baseline_tft.pth - Trained model weights")
        print("  📊 baseline_tft_training.png - Training progress")
        print("  📈 baseline_tft_analysis.png - Performance analysis")
        print("  💰 baseline_tft_trading_simulation.png - Trading simulation")
        
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
