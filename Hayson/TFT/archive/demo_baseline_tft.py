#!/usr/bin/env python3
"""
Simple Example: Baseline TFT Training

This example demonstrates how to train a proper TFT model with simple features:
- Static: symbol, market_cap
- Known future: day_of_week, month, time_idx
- Unknown past: OHLCV, returns, sma_20, rsi_14

Run this to see the difference between the old SimpleTFT and the new BaselineTFT.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the updated TFT training functions
from tft_pure_torch_m1 import create_simple_tft_model, setup_device, prepare_data_for_training, train_model
from baseline_tft import create_baseline_data


def demo_baseline_tft():
    """Simple demo of the baseline TFT implementation."""
    
    print("🎯 Baseline TFT Demo")
    print("=" * 50)
    print("This demo shows a proper TFT implementation with:")
    print("✓ Variable Selection Networks (VSNs)")
    print("✓ Gated Residual Networks (GRNs)")
    print("✓ LSTM encoder-decoder architecture")
    print("✓ Interpretable multi-head attention")
    print("✓ Quantile prediction outputs")
    print("✓ Simple baseline features only")
    print()
    
    # Setup device
    device = setup_device()
    
    # Create simple baseline dataset with fallback to synthetic data
    print("📊 Creating baseline dataset...")
    try:
        # Try to create data using the baseline function
        batch_data = create_baseline_data(
            symbols=['AAPL'],
            start_date='2023-01-01',
            end_date='2023-06-30',
            encoder_length=30,
            prediction_length=1
        )
        
        # Check for NaN values and use synthetic data if issues found
        has_nan = any(torch.isnan(tensor).any() for tensor in batch_data.values() if torch.is_tensor(tensor))
        
        if has_nan:
            print("⚠️ NaN values detected in real data, falling back to synthetic data...")
            raise ValueError("NaN values in data")
        
        # Split into individual samples for training
        total_samples = batch_data['encoder_cont'].shape[0]
        train_size = int(total_samples * 0.8)
        
        train_data = []
        val_data = []
        
        for i in range(train_size):
            sample = {key: value[i:i+1] for key, value in batch_data.items()}
            train_data.append(sample)
        
        for i in range(train_size, total_samples):
            sample = {key: value[i:i+1] for key, value in batch_data.items()}
            val_data.append(sample)
            
        print(f"✅ Created {len(train_data)} training samples from real data")
        print(f"✅ Created {len(val_data)} validation samples from real data")
        
    except Exception as e:
        print(f"❌ Failed to create real dataset: {e}")
        print("🔄 Falling back to synthetic data for demo...")
        
        # Import the synthetic data function
        from demo_fixed_tft import create_synthetic_data
        
        # Create synthetic data
        train_data_batches = create_synthetic_data(800, batch_size=1)  # Individual samples
        val_data_batches = create_synthetic_data(200, batch_size=1)
        
        # Convert to list format
        train_data = []
        val_data = []
        
        for batch in train_data_batches:
            train_data.append(batch)
        
        for batch in val_data_batches:
            val_data.append(batch)
            
        print(f"✅ Created {len(train_data)} synthetic training samples")
        print(f"✅ Created {len(val_data)} synthetic validation samples")
    
    # Create baseline TFT model
    print("\n🏗️ Creating baseline TFT model...")
    try:
        model = create_simple_tft_model(
            num_symbols=1,
            hidden_size=32,  # Small for demo
            encoder_length=30,
            prediction_length=1
        )
        
        total_params = sum(p.numel() for p in model.parameters())
        print(f"✅ Created TFT model with {total_params:,} parameters")
        
        # Show model architecture overview
        print("\n📋 Model Architecture:")
        print("   Static Features:")
        print("     - Symbol ID (categorical)")
        print("     - Market cap (real)")
        print("   Known Future:")
        print("     - Day of week, month, time_idx")
        print("   Unknown Past:")
        print("     - OHLCV, returns, SMA(20), RSI(14)")
        print("   Outputs:")
        print("     - 5 quantiles (0.1, 0.25, 0.5, 0.75, 0.9)")
        
    except Exception as e:
        print(f"❌ Failed to create model: {e}")
        return
    
    # Train the model
    print("\n🏋️ Training baseline TFT...")
    try:
        train_losses, val_losses = train_model(
            model=model,
            train_data=train_data,
            val_data=val_data,
            device=device,
            epochs=5,  # Short demo
            lr=0.001
        )
        
        print(f"✅ Training completed!")
        print(f"   Final train loss: {train_losses[-1]:.4f}")
        print(f"   Final val loss: {val_losses[-1]:.4f}")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Test prediction
    print("\n🔮 Testing prediction...")
    try:
        model.eval()
        
        # Take a sample from validation data
        sample = val_data[0]
        
        # Move to device
        batch = {k: v.to(device) for k, v in sample.items()}
        
        # Forward pass
        with torch.no_grad():
            outputs = model(batch)
        
        # Show results
        prediction = outputs['prediction']
        target = batch['target']
        
        print(f"✅ Prediction successful!")
        print(f"   Target: {target.item():.4f}")
        print(f"   Quantile predictions:")
        quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]
        for i, q in enumerate(quantiles):
            pred_val = prediction[0, 0, i].item()
            print(f"     Q{q}: {pred_val:.4f}")
        
        # Show attention weights
        attention = outputs['attention_weights']
        print(f"   Attention shape: {attention.shape}")
        print(f"   Max attention: {attention.max().item():.4f}")
        
        # Show variable selection weights
        hist_weights = outputs['historical_weights']
        print(f"   Historical variable weights shape: {hist_weights.shape}")
        
        future_weights = outputs['future_weights']
        print(f"   Future variable weights shape: {future_weights.shape}")
        
    except Exception as e:
        print(f"❌ Prediction failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n🎉 Demo completed successfully!")
    print()
    print("Key improvements over SimpleTFT:")
    print("✓ Proper TFT architecture with all components")
    print("✓ Quantile outputs for uncertainty estimation")
    print("✓ Variable selection for interpretability")
    print("✓ Separate encoder/decoder structure")
    print("✓ Static covariate handling")
    print("✓ Attention weights for analysis")
    print()
    print("Next steps:")
    print("• Run train_baseline_tft.py for full training")
    print("• Experiment with different features")
    print("• Analyze attention patterns")
    print("• Compare with other models")


if __name__ == "__main__":
    # Import torch here to avoid issues
    import torch
    
    demo_baseline_tft()
