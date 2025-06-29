#!/usr/bin/env python3
"""
TFT Model Training and Visualization for M1 Mac

Optimized for Apple Silicon with MPS support and robust error handling.
"""

import os
import sys
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dataModule.interface import get_data_loader_with_module
from pytorch_forecasting import TemporalFusionTransformer
from pytorch_forecasting.metrics import MAE

# Use lightning directly for better compatibility
try:
    import lightning as L
    import lightning.pytorch as pl
    from lightning.pytorch.callbacks import EarlyStopping
    print("Using lightning (new version)")
except ImportError:
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import EarlyStopping
    L = pl
    print("Using pytorch_lightning (legacy version)")

import dotenv
dotenv.load_dotenv()


def setup_device():
    """Setup device with MPS support for M1 Mac."""
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        accelerator = "mps"
        print("🚀 Using Apple Silicon MPS acceleration")
    else:
        device = torch.device("cpu")
        accelerator = "cpu"
        print("💻 Using CPU (MPS not available)")
    
    return device, accelerator


def main():
    """Main training and visualization function."""
    
    print("🚀 TFT MODEL TRAINING WITH M1 OPTIMIZATION")
    print("=" * 60)
    
    # Setup device
    device, accelerator = setup_device()
    
    # Configuration
    symbols = ['AAPL', 'MSFT']  # Start with 2 symbols for faster training
    start_date = '2023-01-01'
    end_date = '2024-01-01'
    
    # Optimized hyperparameters for M1 Mac
    encoder_len = 30
    predict_len = 7
    batch_size = 16  # Smaller batch for MPS
    hidden_size = 32  # Smaller for faster training
    max_epochs = 5    # Quick training for demo
    
    print(f"📊 Configuration:")
    print(f"   Symbols: {symbols}")
    print(f"   Device: {device}")
    print(f"   Encoder length: {encoder_len}")
    print(f"   Prediction length: {predict_len}")
    print(f"   Batch size: {batch_size}")
    print(f"   Max epochs: {max_epochs}")
    
    # Load data
    print(f"\n🔍 Loading data...")
    try:
        dataloader, datamodule = get_data_loader_with_module(
            symbols=symbols,
            start=start_date,
            end=end_date,
            encoder_len=encoder_len,
            predict_len=predict_len,
            batch_size=batch_size,
            news_api_key=os.getenv('NEWS_API_KEY'),
            fred_api_key=os.getenv('FRED_API_KEY'),
            api_ninjas_key=os.getenv('API_NINJAS_KEY')
        )
        print("✅ Data loaded successfully")
        print(f"   Training samples: {len(datamodule.train_dataset)}")
        print(f"   Validation samples: {len(datamodule.val_dataset)}")
        
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return
    
    # Create and train model
    print(f"\n🎯 Creating TFT model...")
    try:
        model = TemporalFusionTransformer.from_dataset(
            datamodule.train_dataset,
            learning_rate=0.05,  # Higher LR for faster convergence
            hidden_size=hidden_size,
            attention_head_size=2,  # Smaller for speed
            dropout=0.1,
            hidden_continuous_size=8,
            loss=MAE(),
            log_interval=1,
            reduce_on_plateau_patience=3,
        )
        
        print(f"✅ Model created")
        print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        print(f"\n🏋️ Training model for {max_epochs} epochs...")
        
        # Configure PyTorch Lightning Trainer with proper MPS support
        print("Using PyTorch Lightning with MPS configuration...")
        
        # Setup trainer with MPS-specific settings
        if accelerator == "mps":
            # Key MPS fix: Configure precision and strategy
            trainer = pl.Trainer(
                accelerator="mps",
                devices=1,
                max_epochs=max_epochs,
                gradient_clip_val=0.1,
                precision="16-mixed",  # Use mixed precision for MPS
                enable_checkpointing=False,
                logger=False,
                enable_progress_bar=True,
                log_every_n_steps=1,
                # MPS-specific settings to avoid placeholder storage issues
                strategy="auto",  # Let Lightning handle device strategy
                sync_batchnorm=False,  # Disable batch norm sync for single device
                enable_model_summary=False,
            )
        else:
            trainer = pl.Trainer(
                accelerator="cpu",
                devices=1,
                max_epochs=max_epochs,
                gradient_clip_val=0.1,
                enable_checkpointing=False,
                logger=False,
                enable_progress_bar=True,
                log_every_n_steps=1,
            )
        
        # Add early stopping callback
        early_stop_callback = EarlyStopping(
            monitor="train_loss",
            min_delta=0.001,
            patience=3,
            verbose=True,
            mode="min"
        )
        
        # Train the model
        print("Starting training with Lightning Trainer...")
        trainer.fit(
            model, 
            train_dataloaders=datamodule.train_dataloader(),
            val_dataloaders=datamodule.val_dataloader()
        )
        
        print("✅ Training completed successfully!")
        
    except Exception as e:
        print(f"⚠️ Training error: {e}")
        # DO NOT use mock model - fail properly
        print("❌ Training failed. Cannot proceed without real model.")
        return
    
    # Generate predictions and visualize
    print(f"\n🔮 Generating predictions...")
    predictions, actuals = generate_predictions(model, datamodule, device)
    
    print(f"\n📊 Creating visualizations...")
    create_visualizations(predictions, actuals, datamodule)
    
    print(f"\n💰 Running trading simulation...")
    simulate_trading(predictions, actuals, datamodule)
    
    print("\n✅ Analysis complete! Check generated files:")
    print("   - tft_analysis.png")
    print("   - trading_performance.png")


def generate_predictions(model, datamodule, device):
    """Generate predictions from trained TFT model."""
    
    print("   Using trained TFT model...")
    model.eval()
    model = model.to(device)
    
    predictions_list = []
    actuals_list = []
    
    with torch.no_grad():
        for i, batch in enumerate(datamodule.val_dataloader()):
            if i >= 10:  # Process more batches for better analysis
                break
            
            try:
                # Handle batch format
                if isinstance(batch, dict):
                    x = batch.get('x')
                    y = batch.get('y')
                elif isinstance(batch, tuple):
                    x, y = batch[0], batch[1]
                else:
                    continue
                
                if x is not None:
                    x = x.to(device)
                    pred = model(x)
                    
                    # Extract prediction value
                    if isinstance(pred, dict):
                        pred = pred.get('prediction', pred.get('output', pred))
                    elif isinstance(pred, tuple):
                        pred = pred[0]
                    
                    if torch.is_tensor(pred):
                        predictions_list.append(pred.cpu().detach())
                
                if y is not None:
                    if isinstance(y, tuple):
                        y = y[0]
                    if torch.is_tensor(y):
                        actuals_list.append(y.detach())
            
            except Exception as e:
                print(f"   Batch {i} error: {e}")
                continue
    
    # Combine predictions
    if predictions_list:
        predictions = torch.cat(predictions_list, dim=0).numpy().flatten()
        print(f"   ✅ Generated {len(predictions)} real predictions")
    else:
        raise ValueError("No predictions generated from model")
        
    if actuals_list:
        actuals = torch.cat(actuals_list, dim=0).numpy().flatten()
    else:
        raise ValueError("No actual values found")
    
    return predictions[:100], actuals[:100]  # Limit size for visualization


def create_visualizations(predictions, actuals, datamodule):
    """Create comprehensive visualizations."""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('TFT Model Analysis', fontsize=16, fontweight='bold')
    
    # 1. Predictions vs Actuals
    ax1 = axes[0, 0]
    ax1.scatter(actuals, predictions, alpha=0.6, s=50, color='blue')
    min_val, max_val = min(min(actuals), min(predictions)), max(max(actuals), max(predictions))
    ax1.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, label='Perfect')
    ax1.set_xlabel('Actual Returns')
    ax1.set_ylabel('Predicted Returns')
    ax1.set_title('Predictions vs Actuals')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Add metrics
    correlation = np.corrcoef(actuals, predictions)[0, 1]
    mae = np.mean(np.abs(actuals - predictions))
    ax1.text(0.05, 0.95, f'Corr: {correlation:.3f}\\nMAE: {mae:.4f}', 
             transform=ax1.transAxes, bbox=dict(boxstyle='round', facecolor='wheat'))
    
    # 2. Time series comparison
    ax2 = axes[0, 1]
    time_steps = range(len(predictions))
    ax2.plot(time_steps, actuals, 'b-', label='Actual', linewidth=2)
    ax2.plot(time_steps, predictions, 'r-', label='Predicted', linewidth=2, alpha=0.7)
    ax2.set_xlabel('Time Steps')
    ax2.set_ylabel('Returns')
    ax2.set_title('Time Series Comparison')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Error distribution
    ax3 = axes[1, 0]
    errors = actuals - predictions
    ax3.hist(errors, bins=20, alpha=0.7, color='lightcoral', edgecolor='black')
    ax3.axvline(0, color='red', linestyle='--', label='Zero Error')
    ax3.axvline(np.mean(errors), color='blue', linestyle='--', label=f'Mean: {np.mean(errors):.4f}')
    ax3.set_xlabel('Prediction Error')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Error Distribution')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Directional accuracy
    ax4 = axes[1, 1]
    actual_dir = np.sign(actuals)
    pred_dir = np.sign(predictions)
    directional_acc = np.mean(actual_dir == pred_dir)
    
    confusion = {
        'Correct Up': np.sum((actual_dir > 0) & (pred_dir > 0)),
        'Correct Down': np.sum((actual_dir < 0) & (pred_dir < 0)),
        'Wrong Up': np.sum((actual_dir < 0) & (pred_dir > 0)),
        'Wrong Down': np.sum((actual_dir > 0) & (pred_dir < 0))
    }
    
    colors = ['green', 'darkgreen', 'red', 'darkred']
    bars = ax4.bar(confusion.keys(), confusion.values(), color=colors)
    ax4.set_title(f'Directional Accuracy: {directional_acc:.1%}')
    ax4.set_ylabel('Count')
    ax4.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('tft_analysis.png', dpi=300, bbox_inches='tight')
    print("✅ Analysis saved as 'tft_analysis.png'")
    plt.show()


def simulate_trading(predictions, actuals, datamodule):
    """Simulate trading strategy using predictions."""
    
    initial_capital = 100000
    portfolio_values = [initial_capital]
    cash = initial_capital
    position = 0
    trades = []
    
    # Get price data for simulation
    price_data = datamodule.feature_df[['close']].dropna()
    prices = price_data['close'].values[-len(predictions):]
    
    for i, (pred, price) in enumerate(zip(predictions, prices)):
        # Trading logic
        if pred > 0.01 and cash > price * 100:  # Buy signal
            shares = min(100, int(cash * 0.1 / price))  # 10% position
            cost = shares * price * 1.001  # Include fees
            
            if cash >= cost:
                position += shares
                cash -= cost
                trades.append(('BUY', shares, price))
        
        elif pred < -0.01 and position > 0:  # Sell signal
            shares = min(position, 50)  # Partial sell
            proceeds = shares * price * 0.999  # Include fees
            
            position -= shares
            cash += proceeds
            trades.append(('SELL', shares, price))
        
        # Update portfolio value
        total_value = cash + position * price
        portfolio_values.append(total_value)
    
    # Calculate metrics
    total_return = (portfolio_values[-1] - initial_capital) / initial_capital
    
    # Visualize trading performance
    plt.figure(figsize=(12, 6))
    plt.plot(range(len(portfolio_values)), portfolio_values, 'b-', linewidth=2, label='Portfolio')
    plt.axhline(y=initial_capital, color='r', linestyle='--', alpha=0.7, label='Initial Capital')
    plt.fill_between(range(len(portfolio_values)), initial_capital, portfolio_values, 
                    alpha=0.3, color='green' if portfolio_values[-1] > initial_capital else 'red')
    plt.title(f'Trading Performance (Return: {total_return:.1%})', fontsize=14, fontweight='bold')
    plt.xlabel('Time Steps')
    plt.ylabel('Portfolio Value ($)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('trading_performance.png', dpi=300, bbox_inches='tight')
    print("✅ Trading performance saved as 'trading_performance.png'")
    plt.show()
    
    print(f"📊 Trading Results:")
    print(f"   Initial capital: ${initial_capital:,.2f}")
    print(f"   Final value: ${portfolio_values[-1]:,.2f}")
    print(f"   Total return: {total_return:.2%}")
    print(f"   Number of trades: {len(trades)}")


if __name__ == "__main__":
    main()
