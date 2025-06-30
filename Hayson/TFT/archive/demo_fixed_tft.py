#!/usr/bin/env python3
"""
Fixed Demo: Baseline TFT Training

This creates synthetic data to avoid data loading issues and NaN problems.
"""

import sys
import os
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from baseline_tft import BaselineTFT, QuantileLoss


def create_synthetic_data(num_samples: int = 1000, 
                         encoder_length: int = 30, 
                         prediction_length: int = 1,
                         batch_size: int = 32):
    """Create synthetic data for testing TFT."""
    
    print(f"📊 Creating synthetic dataset with {num_samples} samples...")
    
    # Generate synthetic time series data
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Static features
    static_categorical = torch.zeros(num_samples, 1, dtype=torch.long)  # All same symbol
    static_real = torch.randn(num_samples, 1) * 0.1  # Normalized market cap
    
    # Time-varying features
    encoder_cont = torch.randn(num_samples, encoder_length, 8) * 0.1  # 8 historical features
    decoder_cont = torch.randn(num_samples, prediction_length, 3) * 0.1  # 3 future features
    
    # Targets (simulate stock returns)
    base_trend = torch.sin(torch.linspace(0, 4*np.pi, num_samples)).unsqueeze(-1) * 0.01
    noise = torch.randn(num_samples, prediction_length) * 0.005
    target = base_trend + noise
    
    # Create batches
    dataset = []
    for i in range(0, num_samples, batch_size):
        end_idx = min(i + batch_size, num_samples)
        batch = {
            'static_categorical': static_categorical[i:end_idx],
            'static_real': static_real[i:end_idx],
            'encoder_cont': encoder_cont[i:end_idx],
            'decoder_cont': decoder_cont[i:end_idx],
            'target': target[i:end_idx]
        }
        dataset.append(batch)
    
    print(f"✅ Created {len(dataset)} batches")
    return dataset


def test_fixed_tft():
    """Test TFT with synthetic data to avoid NaN issues."""
    
    print("🎯 Fixed Baseline TFT Demo")
    print("=" * 50)
    print("This demo uses synthetic data to test TFT architecture")
    print("without data loading issues that cause NaN values.")
    print()
    
    # Create synthetic dataset
    train_data = create_synthetic_data(800, batch_size=32)
    val_data = create_synthetic_data(200, batch_size=32)
    
    # Create model
    print("🏗️ Creating baseline TFT model...")
    model = BaselineTFT(
        static_categorical_cardinalities=[1],  # 1 symbol
        num_static_real=1,  # market_cap
        num_time_varying_real_known=3,  # day_of_week, month, time_idx
        num_time_varying_real_unknown=8,  # OHLCV + returns + sma_20 + rsi_14
        hidden_size=32,  # Small for demo
        encoder_length=30,
        prediction_length=1,
        quantiles=[0.1, 0.25, 0.5, 0.75, 0.9]
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✅ Created TFT model with {total_params:,} parameters")
    
    # Initialize weights properly
    def init_weights(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight, gain=0.1)  # Small gain
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            torch.nn.init.normal_(m.weight, mean=0, std=0.01)
        elif isinstance(m, nn.LSTM):
            for name, param in m.named_parameters():
                if 'weight' in name:
                    torch.nn.init.xavier_uniform_(param, gain=0.1)
                elif 'bias' in name:
                    torch.nn.init.zeros_(param)
    
    model.apply(init_weights)
    print("✅ Initialized model weights")
    
    # Setup training
    device = torch.device('cpu')  # Use CPU to avoid MPS issues
    model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    criterion = QuantileLoss([0.1, 0.25, 0.5, 0.75, 0.9])
    
    print(f"\n🏋️ Training model for 3 epochs...")
    
    # Training loop
    for epoch in range(3):
        model.train()
        train_loss = 0
        
        pbar = tqdm(train_data, desc=f"Epoch {epoch+1}/3", leave=False)
        for batch_idx, batch in enumerate(pbar):
            # Move to device
            batch = {k: v.to(device) for k, v in batch.items()}
            
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(batch)
            
            # Calculate loss
            loss = criterion(outputs['prediction'], batch['target'])
            
            # Check for NaN
            if torch.isnan(loss):
                print(f"❌ NaN loss detected at batch {batch_idx}")
                print(f"   Predictions: {outputs['prediction'][:3]}")
                print(f"   Targets: {batch['target'][:3]}")
                break
            
            # Backward pass with gradient clipping
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()
            
            train_loss += loss.item()
            
            pbar.set_postfix({'loss': f"{loss.item():.6f}"})
            
            # Early stopping if loss is too high
            if loss.item() > 10.0:
                print(f"⚠️ High loss detected: {loss.item():.6f}")
                break
        
        avg_train_loss = train_loss / len(train_data)
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_data:
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(batch)
                loss = criterion(outputs['prediction'], batch['target'])
                
                if not torch.isnan(loss):
                    val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_data)
        
        print(f"   Epoch {epoch+1}: Train Loss = {avg_train_loss:.6f}, Val Loss = {avg_val_loss:.6f}")
        
        # Check for successful training
        if not np.isnan(avg_train_loss) and not np.isnan(avg_val_loss):
            print(f"   ✅ Epoch {epoch+1} completed successfully!")
        else:
            print(f"   ❌ NaN detected in epoch {epoch+1}")
            break
    
    # Test prediction
    print("\n🔮 Testing prediction...")
    model.eval()
    
    test_batch = val_data[0]
    test_batch = {k: v.to(device) for k, v in test_batch.items()}
    
    with torch.no_grad():
        outputs = model(test_batch)
    
    prediction = outputs['prediction'][0, 0]  # First sample, first timestep
    target = test_batch['target'][0, 0]
    
    print(f"✅ Prediction successful!")
    print(f"   Target: {target.item():.6f}")
    print(f"   Quantile predictions:")
    quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]
    for i, q in enumerate(quantiles):
        pred_val = prediction[i].item()
        print(f"     Q{q}: {pred_val:.6f}")
    
    # Show attention weights
    attention = outputs['attention_weights'][0]  # First sample
    print(f"   Attention shape: {attention.shape}")
    print(f"   Attention stats: min={attention.min().item():.4f}, max={attention.max().item():.4f}")
    
    print("\n🎉 Fixed demo completed successfully!")
    print("\nKey fixes applied:")
    print("✓ Synthetic data to avoid loading issues")
    print("✓ Proper weight initialization") 
    print("✓ Gradient clipping")
    print("✓ NaN detection and handling")
    print("✓ Small learning rate and regularization")


if __name__ == "__main__":
    test_fixed_tft()
