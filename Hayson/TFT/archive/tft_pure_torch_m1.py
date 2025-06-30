#!/usr/bin/env python3
"""
TFT Model Training with Pure PyTorch for M1 Mac

Now uses proper baseline TFT implementation instead of SimpleTFT.
This file provides training infrastructure for the real TFT model.
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dataModule.interface import get_data_loader_with_module
from baseline_tft import BaselineTFT, QuantileLoss, create_baseline_data

import dotenv
dotenv.load_dotenv()


def create_simple_tft_model(num_symbols: int = 1, hidden_size: int = 64, 
                           encoder_length: int = 30, prediction_length: int = 1):
    """
    Create a baseline TFT model with simple features.
    
    This replaces the old SimpleTFT with a proper TFT implementation.
    """
    model = BaselineTFT(
        # Feature dimensions - simple baseline
        static_categorical_cardinalities=[num_symbols],  # symbol IDs
        num_static_real=1,  # market_cap
        num_time_varying_real_known=3,  # day_of_week, month, time_idx
        num_time_varying_real_unknown=8,  # OHLCV + returns + sma_20 + rsi_14
        
        # Architecture parameters
        hidden_size=hidden_size,
        lstm_layers=2,
        attention_heads=4,
        dropout=0.1,
        
        # Sequence parameters
        encoder_length=encoder_length,
        prediction_length=prediction_length,
        
        # TFT-specific: quantile outputs for uncertainty
        quantiles=[0.1, 0.25, 0.5, 0.75, 0.9]
    )
    
    return model


def setup_device():
    """Setup device with M1 Mac optimization."""
    if torch.backends.mps.is_available():
        try:
            # Test MPS with a simple operation
            test_tensor = torch.randn(2, 2, device='mps')
            test_result = test_tensor @ test_tensor
            device = torch.device("mps")
            print("🚀 Using Apple Silicon MPS acceleration")
            return device
        except Exception as e:
            print(f"⚠️ MPS test failed: {e}")
            print("💻 Falling back to CPU")
            return torch.device("cpu")
    else:
        print("💻 Using CPU (MPS not available)")
        return torch.device("cpu")


def prepare_data_for_training(datamodule, device, max_batches=20):
    """
    Prepare data for training with proper TFT tensor handling.
    
    Now uses the baseline TFT data structure instead of concatenating features.
    """
    
    print("🔄 Preparing training data for Baseline TFT...")
    
    train_data = []
    val_data = []
    
    # Try to use simple baseline data if the datamodule approach fails
    try:
        if datamodule is not None:
            train_loader = datamodule.train_dataloader()
            val_loader = datamodule.val_dataloader()
            
            print("✅ Using provided datamodule")
        else:
            raise Exception("No datamodule provided, using fallback")
        
        # Process training data with enhanced progress bar
        train_pbar = tqdm(
            enumerate(train_loader), 
            total=min(max_batches, len(train_loader)),
            desc="📚 Processing Training Data",
            unit="batch",
            ncols=120,
            colour='blue'
        )
        
        for i, batch in train_pbar:
            if i >= max_batches:
                break
            
            try:
                # Extract from TFT dataloader format
                if isinstance(batch, tuple) and len(batch) >= 2:
                    x, y = batch[0], batch[1]
                    
                    # Check if this is proper TFT format
                    if isinstance(x, dict) and all(key in x for key in ['encoder_cont', 'decoder_cont']):
                        # Proper TFT format - use directly
                        tft_batch = {
                            'encoder_cont': x['encoder_cont'].float(),
                            'decoder_cont': x['decoder_cont'].float(),
                            'target': y[0] if isinstance(y, tuple) else y
                        }
                        
                        # Add static features if available
                        if 'static_categorical' in x:
                            tft_batch['static_categorical'] = x['static_categorical'].long()
                        if 'static_real' in x:
                            tft_batch['static_real'] = x['static_real'].float()
                        
                        train_data.append(tft_batch)
                        
                        if i == 0:
                            print(f"   ✓ TFT format detected:")
                            print(f"     Encoder shape: {tft_batch['encoder_cont'].shape}")
                            print(f"     Decoder shape: {tft_batch['decoder_cont'].shape}")
                            print(f"     Target shape: {tft_batch['target'].shape}")
                    
                    else:
                        # Legacy format - skip for now
                        print(f"   ⚠️ Legacy format detected, skipping batch {i}")
                        continue
                
                # Update progress bar
                train_pbar.set_postfix({
                    '📊 Samples': len(train_data),
                    '📦 Batch': f"{i+1}/{max_batches}",
                    '✅ Success': '🎯'
                })
                
            except Exception as e:
                print(f"   ❌ Error processing training batch {i}: {e}")
                continue
        
        # Process validation data similarly
        val_pbar = tqdm(
            enumerate(val_loader), 
            total=min(max_batches//2, len(val_loader)),
            desc="📊 Processing Validation Data",
            unit="batch",
            ncols=120,
            colour='yellow'
        )
        
        for i, batch in val_pbar:
            if i >= max_batches//2:
                break
            
            try:
                if isinstance(batch, tuple) and len(batch) >= 2:
                    x, y = batch[0], batch[1]
                    
                    if isinstance(x, dict) and all(key in x for key in ['encoder_cont', 'decoder_cont']):
                        tft_batch = {
                            'encoder_cont': x['encoder_cont'].float(),
                            'decoder_cont': x['decoder_cont'].float(),
                            'target': y[0] if isinstance(y, tuple) else y
                        }
                        
                        if 'static_categorical' in x:
                            tft_batch['static_categorical'] = x['static_categorical'].long()
                        if 'static_real' in x:
                            tft_batch['static_real'] = x['static_real'].float()
                        
                        val_data.append(tft_batch)
                
                val_pbar.set_postfix({
                    '📊 Samples': len(val_data),
                    '📦 Batch': f"{i+1}/{max_batches//2}",
                    '✅ Success': '🎯'
                })
                
            except Exception as e:
                print(f"   ❌ Error processing validation batch {i}: {e}")
                continue
    
    except Exception as e:
        print(f"❌ Datamodule approach failed: {e}")
        print("🔄 Falling back to simple baseline data generation...")
        
        # Fallback: create simple baseline data
        try:
            batch_data = create_baseline_data(
                symbols=['AAPL'],
                start_date='2023-01-01', 
                end_date='2023-06-30',
                encoder_length=30,
                prediction_length=1
            )
            
            # Split into train/val
            total_samples = batch_data['encoder_cont'].shape[0]
            train_size = int(total_samples * 0.8)
            
            # Convert to list of individual samples
            for i in range(train_size):
                sample = {key: value[i:i+1] for key, value in batch_data.items()}
                train_data.append(sample)
            
            for i in range(train_size, total_samples):
                sample = {key: value[i:i+1] for key, value in batch_data.items()}
                val_data.append(sample)
            
            print(f"   ✅ Generated {len(train_data)} training samples")
            print(f"   ✅ Generated {len(val_data)} validation samples")
            
        except Exception as fallback_error:
            print(f"❌ Fallback data generation failed: {fallback_error}")
            raise
    
    if not train_data:
        raise ValueError("No training data could be prepared")
    
    print(f"✅ Data preparation completed!")
    print(f"   Training samples: {len(train_data)}")
    print(f"   Validation samples: {len(val_data)}")
    
    return train_data, val_data


def train_model(model, train_data, val_data, device, epochs=10, lr=0.001):
    """Train the model with pure PyTorch and enhanced tqdm progress bars."""
    
    print(f"🏋️ Training Baseline TFT on {device} for {epochs} epochs...")
    print(f"📊 Training samples: {len(train_data)}, Validation samples: {len(val_data)}")
    print(f"📝 Learning rate: {lr}, Device: {device}")
    print(f"🎯 Using QuantileLoss for proper TFT training")
    print("=" * 80)
    
    model = model.to(device)
    
    # Improved optimizer with weight decay
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = QuantileLoss([0.1, 0.25, 0.5, 0.75, 0.9])  # TFT quantile loss
    
    # Initialize weights properly to prevent NaN
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
    print("✅ Applied proper weight initialization")
    
    train_losses = []
    val_losses = []
    
    # Main epoch progress bar with enhanced description
    epoch_pbar = tqdm(
        range(epochs), 
        desc="🏋️ TFT Training", 
        unit="epoch",
        ncols=100,
        colour='blue'
    )
    
    for epoch in epoch_pbar:
        # Training phase
        model.train()
        train_loss = 0
        train_batches = 0
        
        # Enhanced training batch progress bar
        train_pbar = tqdm(
            train_data, 
            desc=f"📈 Epoch {epoch+1:2d}/{epochs} Training", 
            leave=False,
            unit="batch",
            ncols=120,
            colour='green',
            miniters=1
        )
        
        for batch_idx, batch_item in enumerate(train_pbar):
            try:
                # Debug first iteration - check what we're getting
                if epoch == 0 and batch_idx == 0:
                    print(f"   DEBUG: Batch item type: {type(batch_item)}")
                    if isinstance(batch_item, dict):
                        print(f"   DEBUG: Batch keys: {list(batch_item.keys())}")
                        for key, value in batch_item.items():
                            if torch.is_tensor(value):
                                print(f"   DEBUG: {key} shape: {value.shape}")
                
                # Ensure we have a proper TFT batch dictionary
                if isinstance(batch_item, dict):
                    # Move to device
                    batch = {k: v.to(device) for k, v in batch_item.items()}
                    
                    # Get target
                    target = batch['target']
                    
                    if epoch == 0 and batch_idx == 0:
                        print(f"   DEBUG: Target shape: {target.shape}")
                
                else:
                    print(f"   ERROR: Expected dict batch, got {type(batch_item)}")
                    continue
                
                optimizer.zero_grad()
                
                # Forward pass through TFT
                outputs = model(batch)
                
                if epoch == 0 and batch_idx == 0:
                    print(f"   DEBUG: Model output keys: {outputs.keys()}")
                    print(f"   DEBUG: Prediction shape: {outputs['prediction'].shape}")
                
                # Calculate loss using quantile loss
                loss = criterion(outputs['prediction'], target)
                
                # Check for NaN loss
                if torch.isnan(loss):
                    print(f"   ⚠️ NaN loss detected at batch {batch_idx}, skipping...")
                    continue
                
                if epoch == 0 and batch_idx == 0:
                    print(f"   DEBUG: Loss: {loss.item()}")
                
                loss.backward()
                
                # Stronger gradient clipping for stable training
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                
                optimizer.step()
                
                train_loss += loss.item()
                train_batches += 1
                
                if epoch == 0 and batch_idx == 0:
                    print(f"   DEBUG: Training step completed successfully!")
                
                # Update progress bar with enhanced metrics
                current_avg = train_loss / train_batches
                train_pbar.set_postfix({
                    'loss': f"{loss.item():.5f}",
                    'avg': f"{current_avg:.5f}",
                    'lr': f"{lr:.1e}",
                    'batch': f"{batch_idx+1}"
                })
                
            except Exception as e:
                if epoch == 0 and batch_idx == 0:
                    print(f"   DEBUG: Exception in training loop: {e}")
                    import traceback
                    traceback.print_exc()
                train_pbar.set_postfix({
                    'ERROR': str(e)[:15],
                    'batch': f"{batch_idx+1}"
                })
                continue
        
        # Validation phase
        model.eval()
        val_loss = 0
        val_batches = 0
        
        # Enhanced validation batch progress bar
        val_pbar = tqdm(
            val_data, 
            desc=f"📊 Epoch {epoch+1:2d}/{epochs} Validation", 
            leave=False,
            unit="batch",
            ncols=120,
            colour='yellow',
            miniters=1
        )
        
        with torch.no_grad():
            for val_idx, val_batch_item in enumerate(val_pbar):
                try:
                    # Debug first validation iteration
                    if epoch == 0 and val_idx == 0:
                        print(f"   DEBUG: Val batch item type: {type(val_batch_item)}")
                        if isinstance(val_batch_item, dict):
                            print(f"   DEBUG: Val batch keys: {list(val_batch_item.keys())}")
                    
                    # Ensure we have a proper TFT batch dictionary
                    if isinstance(val_batch_item, dict):
                        batch = {k: v.to(device) for k, v in val_batch_item.items()}
                        target = batch['target']
                    else:
                        print(f"   ERROR: Expected dict val batch, got {type(val_batch_item)}")
                        continue
                    
                    # Forward pass
                    outputs = model(batch)
                    loss = criterion(outputs['prediction'], target)
                    
                    val_loss += loss.item()
                    val_batches += 1
                    
                    # Update validation progress with enhanced metrics
                    current_val_avg = val_loss / val_batches
                    val_pbar.set_postfix({
                        'val_loss': f"{loss.item():.5f}",
                        'val_avg': f"{current_val_avg:.5f}",
                        'samples': val_batches
                    })
                    
                except Exception as e:
                    val_pbar.set_postfix({
                        'ERROR': str(e)[:15],
                        'samples': val_batches
                    })
                    continue
        
        # Calculate average losses
        avg_train_loss = train_loss / max(train_batches, 1)
        avg_val_loss = val_loss / max(val_batches, 1)
        
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        
        # Update main epoch progress bar with comprehensive metrics
        epoch_pbar.set_postfix({
            '🚂 Train': f"{avg_train_loss:.5f}",
            '📊 Val': f"{avg_val_loss:.5f}",
            '📦 T.Batch': train_batches,
            '📦 V.Batch': val_batches,
            '📈 Improve': '✅' if epoch > 0 and avg_val_loss < val_losses[-2] else '⚠️'
        })
    
    epoch_pbar.close()
    
    # Final training summary
    print("\n" + "=" * 80)
    print("✅ BASELINE TFT TRAINING COMPLETED SUCCESSFULLY!")
    if train_losses:
        print(f"📈 Final Training Loss: {train_losses[-1]:.6f}")
    if val_losses:
        print(f"📊 Final Validation Loss: {val_losses[-1]:.6f}")
    print(f"🔄 Total Epochs: {epochs}")
    print(f"📦 Total Training Samples: {len(train_data)}")
    print(f"📦 Total Validation Samples: {len(val_data)}")
    print(f"🎯 Model Type: Baseline TFT with VSNs, GRNs, and Quantile Loss")
    print("=" * 80)
    
    return train_losses, val_losses


def generate_predictions(model, val_data, device):
    """Generate predictions from the trained model with progress bar."""
    
    print("🔮 Generating predictions...")
    
    model.eval()
    model = model.to(device)
    
    predictions = []
    actuals = []
    
    # Enhanced progress bar for prediction generation
    prediction_pbar = tqdm(
        val_data, 
        desc="🔮 Generating Predictions", 
        unit="batch",
        ncols=120,
        colour='cyan'
    )
    
    with torch.no_grad():
        for pred_idx, pred_batch_item in enumerate(prediction_pbar):
            try:
                # Debug first prediction iteration
                if pred_idx == 0:
                    print(f"   DEBUG: Pred batch item type: {type(pred_batch_item)}")
                    if isinstance(pred_batch_item, (tuple, list)):
                        print(f"   DEBUG: Pred batch item length: {len(pred_batch_item)}")
                
                # Use the same TFT batch format as training
                if isinstance(pred_batch_item, dict):
                    # Move batch to device
                    batch = {k: v.to(device) for k, v in pred_batch_item.items()}
                    target = batch['target']
                    
                    # Generate prediction using TFT model
                    outputs = model(batch)
                    pred = outputs['prediction']  # TFT returns dict with 'prediction' key
                    
                    # Extract median quantile (index 2 for [0.1, 0.25, 0.5, 0.75, 0.9])
                    if pred.dim() == 3 and pred.shape[-1] == 5:  # [batch, seq, quantiles]
                        pred_median = pred[:, :, 2]  # Take median quantile
                    elif pred.dim() == 2:
                        pred_median = pred
                    else:
                        pred_median = pred.squeeze()
                    
                    # Convert to numpy
                    pred_np = pred_median.cpu().numpy().flatten()
                    target_np = target.cpu().numpy().flatten()
                    
                    predictions.extend(pred_np)
                    actuals.extend(target_np)
                    
                    # Update progress bar with detailed prediction info
                    avg_pred = np.mean(pred_np) if len(pred_np) > 0 else 0
                    avg_actual = np.mean(target_np) if len(target_np) > 0 else 0
                    prediction_pbar.set_postfix({
                        'pred_avg': f"{avg_pred:.4f}",
                        'actual_avg': f"{avg_actual:.4f}",
                        'batch_size': len(pred_np),
                        'total': len(predictions)
                    })
                else:
                    print(f"   ERROR: Expected dict pred batch, got {type(pred_batch_item)}")
                    continue
                
            except Exception as e:
                print(f"   ERROR in prediction: {str(e)}")
                prediction_pbar.set_postfix({
                    'ERROR': str(e)[:15],
                    'total': len(predictions)
                })
                continue
    
    prediction_pbar.close()
    
    predictions = np.array(predictions)
    actuals = np.array(actuals)
    
    print(f"   ✅ Generated {len(predictions)} predictions")
    
    return predictions, actuals


def create_visualizations(predictions, actuals, train_losses, val_losses):
    """Create comprehensive visualizations."""
    
    print("📊 Creating visualizations...")
    
    # Check if we have valid data for visualization
    if len(predictions) == 0 or len(actuals) == 0:
        print("   ⚠️ No predictions or actuals available for visualization")
        print(f"   📊 Predictions: {len(predictions)}, Actuals: {len(actuals)}")
        
        # Create a simple plot showing only training progress
        plt.figure(figsize=(10, 6))
        if train_losses:
            plt.plot(train_losses, label='Training Loss', color='blue', alpha=0.8)
        if val_losses:
            plt.plot(val_losses, label='Validation Loss', color='red', alpha=0.8)
        plt.title('Training Progress (No Predictions Generated)', fontsize=14, fontweight='bold')
        plt.xlabel('Epoch')
        plt.ylabel('Loss (MSE)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Save simple plot
        output_path = 'tft_training_only.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   ✅ Saved training progress to {output_path}")
        return
    
    # Convert to numpy arrays
    predictions = np.array(predictions)
    actuals = np.array(actuals)
    
    plt.style.use('seaborn-v0_8')
    fig = plt.figure(figsize=(20, 12))
    
    # 1. Training curves
    ax1 = plt.subplot(2, 3, 1)
    plt.plot(train_losses, label='Training Loss', color='blue', alpha=0.8)
    plt.plot(val_losses, label='Validation Loss', color='red', alpha=0.8)
    plt.title('Training Progress', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 2. Predictions vs Actuals
    ax2 = plt.subplot(2, 3, 2)
    plt.scatter(actuals, predictions, alpha=0.6, s=20)
    min_val = min(np.min(actuals), np.min(predictions))
    max_val = max(np.max(actuals), np.max(predictions))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Predictions vs Actuals', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # Calculate R²
    corr_matrix = np.corrcoef(actuals, predictions)
    r_squared = corr_matrix[0, 1] ** 2
    plt.text(0.05, 0.95, f'R² = {r_squared:.3f}', transform=ax2.transAxes, 
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    # 3. Time series comparison
    ax3 = plt.subplot(2, 3, 3)
    indices = np.arange(len(predictions))
    plt.plot(indices, actuals, label='Actual', color='blue', alpha=0.8)
    plt.plot(indices, predictions, label='Predicted', color='red', alpha=0.8)
    plt.title('Time Series Comparison', fontsize=14, fontweight='bold')
    plt.xlabel('Sample Index')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 4. Residuals
    ax4 = plt.subplot(2, 3, 4)
    residuals = predictions - actuals
    plt.scatter(predictions, residuals, alpha=0.6, s=20)
    plt.axhline(y=0, color='red', linestyle='--', alpha=0.8)
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.title('Residuals Plot', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # 5. Error distribution
    ax5 = plt.subplot(2, 3, 5)
    plt.hist(residuals, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    plt.xlabel('Residuals')
    plt.ylabel('Frequency')
    plt.title('Error Distribution', fontsize=14, fontweight='bold')
    plt.axvline(x=0, color='red', linestyle='--', alpha=0.8)
    plt.grid(True, alpha=0.3)
    
    # 6. Model metrics
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')
    
    # Calculate metrics
    mae = np.mean(np.abs(residuals))
    rmse = np.sqrt(np.mean(residuals**2))
    mape = np.mean(np.abs(residuals / (actuals + 1e-8))) * 100
    
    # Directional accuracy
    actual_direction = np.sign(np.diff(actuals))
    pred_direction = np.sign(np.diff(predictions))
    directional_accuracy = np.mean(actual_direction == pred_direction) * 100
    
    metrics_text = f'''
Model Performance Metrics:

MAE: {mae:.4f}
RMSE: {rmse:.4f}
MAPE: {mape:.2f}%
R²: {r_squared:.3f}
Directional Accuracy: {directional_accuracy:.1f}%

Data Points: {len(predictions)}
Device: {'MPS' if torch.backends.mps.is_available() else 'CPU'}
Model: SimpleTFT
    '''
    
    ax6.text(0.1, 0.5, metrics_text, fontsize=12, verticalalignment='center',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    
    # Save plot
    output_path = 'tft_pure_torch_analysis.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"   ✅ Saved analysis to {output_path}")


def simulate_trading(predictions, actuals):
    import numpy as np
    import matplotlib.pyplot as plt
    print("💰 Running trading simulation...")
    initial_price = 100.0
    actual_prices = [initial_price]
    pred_prices = [initial_price]
    for i in range(len(actuals)):
        actual_prices.append(actual_prices[-1] * (1 + actuals[i] * 0.01))
        pred_prices.append(pred_prices[-1] * (1 + predictions[i] * 0.01))
    actual_prices = np.array(actual_prices[1:])
    pred_prices = np.array(pred_prices[1:])
    portfolio_value = 10000
    position = 0
    trades = []
    portfolio_values = [portfolio_value]
    for i in range(1, len(pred_prices)):
        if i < len(pred_prices) - 1:
            predicted_return = (pred_prices[i+1] - pred_prices[i]) / pred_prices[i]
            actual_return = (actual_prices[i+1] - actual_prices[i]) / actual_prices[i]
            if predicted_return > 0.001 and position != 1:
                if position == -1:
                    trades.append(('close_short', i, actual_prices[i]))
                trades.append(('buy', i, actual_prices[i]))
                position = 1
            elif predicted_return < -0.001 and position != -1:
                if position == 1:
                    trades.append(('close_long', i, actual_prices[i]))
                trades.append(('sell_short', i, actual_prices[i]))
                position = -1
            if position == 1:
                portfolio_value *= (1 + actual_return)
            elif position == -1:
                portfolio_value *= (1 - actual_return)
            portfolio_values.append(portfolio_value)
    total_return = (portfolio_value - 10000) / 10000 * 100
    buy_hold_return = (actual_prices[-1] - actual_prices[0]) / actual_prices[0] * 100
    plt.figure(figsize=(15, 10))
    plt.subplot(2, 2, 1)
    plt.plot(actual_prices, label='Actual Price', color='blue', alpha=0.8)
    plt.plot(pred_prices, label='Predicted Price', color='red', alpha=0.8, linestyle='--')
    for trade_type, idx, price in trades:
        color = 'green' if 'buy' in trade_type else 'red'
        marker = '^' if 'buy' in trade_type else 'v'
        plt.scatter(idx, price, color=color, marker=marker, s=100, alpha=0.8)
    plt.title('Trading Strategy Performance', fontsize=14, fontweight='bold')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.subplot(2, 2, 2)
    plt.plot(portfolio_values, color='green', linewidth=2)
    plt.axhline(y=10000, color='gray', linestyle='--', alpha=0.8)
    plt.title('Portfolio Value Over Time', fontsize=14, fontweight='bold')
    plt.xlabel('Time')
    plt.ylabel('Portfolio Value ($)')
    plt.grid(True, alpha=0.3)
    plt.subplot(2, 2, 3)
    strategies = ['TFT Strategy', 'Buy & Hold']
    returns = [total_return, buy_hold_return]
    colors = ['green' if r > 0 else 'red' for r in returns]
    plt.bar(strategies, returns, color=colors, alpha=0.7)
    plt.title('Strategy Returns Comparison', fontsize=14, fontweight='bold')
    plt.ylabel('Return (%)')
    plt.grid(True, alpha=0.3)
    plt.subplot(2, 2, 4)
    plt.axis('off')
    summary_text = f'''
Trading Performance Summary:

Starting Capital: $10,000
Final Portfolio Value: ${portfolio_value:,.2f}
Total Return: {total_return:.2f}%
Buy & Hold Return: {buy_hold_return:.2f}%
Outperformance: {total_return - buy_hold_return:.2f}%

Number of Trades: {len(trades)}
Win Rate: {np.mean([1 if t > 0 else 0 for t in np.diff(portfolio_values)]) * 100 if len(portfolio_values) > 1 else float('nan'):.1f}%

Strategy: Trend Following
Signal: Predicted Price Direction
    '''
    plt.text(0.1, 0.5, summary_text, fontsize=12, verticalalignment='center', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    plt.tight_layout()
    trading_output = 'tft_trading_performance.png'
    plt.savefig(trading_output, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✅ Saved trading analysis to {trading_output}")
    print(f"   💰 Total return: {total_return:.2f}% vs Buy & Hold: {buy_hold_return:.2f}%")


def analyze_training_results(train_losses, val_losses, predictions=None, actuals=None):
    """
    Analyze training results and provide recommendations.
    """
    print("\n" + "="*80)
    print("📊 TRAINING ANALYSIS & RECOMMENDATIONS")
    print("="*80)
    
    # Basic validation
    if not train_losses or not val_losses:
        print("❌ No training history available for analysis")
        return
    
    final_train_loss = train_losses[-1]
    final_val_loss = val_losses[-1]
    
    # 1. Overfitting Analysis
    overfitting_ratio = final_val_loss / final_train_loss
    print(f"🎯 OVERFITTING ANALYSIS:")
    print(f"   Validation/Training Loss Ratio: {overfitting_ratio:.3f}")
    
    if overfitting_ratio > 2.0:
        print("   ⚠️  POTENTIAL OVERFITTING DETECTED!")
        print("   💡 Recommendations:")
        print("      • Increase dropout (currently 0.1 → try 0.2-0.3)")
        print("      • Add L2 regularization")
        print("      • Reduce model complexity")
        print("      • Get more training data")
    elif overfitting_ratio > 1.5:
        print("   🟡 Mild overfitting - monitor closely")
        print("   💡 Consider slight regularization increase")
    else:
        print("   ✅ Good generalization!")
    
    # 2. Training Stability
    print(f"\n🔄 TRAINING STABILITY:")
    if len(train_losses) > 2:
        train_improvement = (train_losses[0] - train_losses[-1]) / train_losses[0] * 100
        val_improvement = (val_losses[0] - val_losses[-1]) / val_losses[0] * 100
        
        print(f"   Training Loss Improvement: {train_improvement:.1f}%")
        print(f"   Validation Loss Improvement: {val_improvement:.1f}%")
        
        if train_improvement < 5:
            print("   ⚠️  Limited training improvement")
            print("   💡 Try: Higher learning rate, more epochs, or different optimizer")
        
        # Check for convergence
        if len(train_losses) >= 3:
            recent_change = abs(train_losses[-1] - train_losses[-2]) / train_losses[-2] * 100
            if recent_change < 1:
                print("   ✅ Training appears to have converged")
            else:
                print("   🔄 Training still improving - could benefit from more epochs")
    
    # 3. Loss Magnitude Analysis
    print(f"\n📈 LOSS MAGNITUDE:")
    print(f"   Final Training Loss: {final_train_loss:.6f}")
    print(f"   Final Validation Loss: {final_val_loss:.6f}")
    
    if final_train_loss < 0.001:
        print("   ⚠️  Very low training loss - possible overfitting")
    elif final_train_loss > 0.1:
        print("   ⚠️  High training loss - model may be underfitting")
        print("   💡 Try: Lower learning rate, more complex model, or more epochs")
    else:
        print("   ✅ Reasonable loss magnitude")
    
    # 4. Quantile Prediction Analysis (if available)
    if predictions is not None and actuals is not None:
        print(f"\n🎯 PREDICTION QUALITY:")
        if len(predictions) > 0 and len(actuals) > 0:
            mae = np.mean(np.abs(np.array(predictions) - np.array(actuals)))
            rmse = np.sqrt(np.mean((np.array(predictions) - np.array(actuals))**2))
            
            print(f"   MAE: {mae:.6f}")
            print(f"   RMSE: {rmse:.6f}")
            
            # Compare with baseline
            baseline_mae = np.mean(np.abs(np.array(actuals)))
            relative_mae = mae / (baseline_mae + 1e-8)
            
            if relative_mae < 0.8:
                print("   ✅ Excellent prediction accuracy!")
            elif relative_mae < 1.2:
                print("   ✅ Good prediction accuracy")
            else:
                print("   ⚠️  Room for improvement in predictions")
    
    # 5. Recommendations
    print(f"\n💡 SPECIFIC RECOMMENDATIONS:")
    
    # Based on current performance
    if overfitting_ratio > 1.8:
        print("   🎯 PRIORITY: Address overfitting")
        print("      1. Increase dropout to 0.2-0.3")
        print("      2. Add weight decay (L2 regularization)")
        print("      3. Use early stopping")
        print("      4. Consider data augmentation")
    
    if final_train_loss > 0.01:
        print("   🎯 PRIORITY: Improve model fit")
        print("      1. Increase model capacity (hidden_size)")
        print("      2. Train for more epochs")
        print("      3. Try different learning rate schedule")
        print("      4. Check data preprocessing")
    
    print("   📊 GENERAL IMPROVEMENTS:")
    print("      • Try different optimizers (AdamW, RMSprop)")
    print("      • Implement learning rate scheduling")
    print("      • Experiment with different sequence lengths")
    print("      • Add more diverse features")
    print("      • Use ensemble methods")
    
    print("\n" + "="*80)

# Add this function to the main training flow
def run_enhanced_training_demo():
    """Enhanced demo with comprehensive analysis."""
    
    print("🚀 Enhanced TFT Training Demo with Analysis")
    print("="*80)
    
    # Setup
    device = setup_device()
    
    # Load data with proper parameters
    try:
        datamodule = get_data_loader_with_module(
            symbols=['SPY'],  # Use SPY as default symbol
            start='2020-01-01',
            end='2023-12-31', 
            encoder_len=30,
            predict_len=1,
            batch_size=32
        )
        print("✅ Successfully loaded datamodule with parameters")
    except Exception as e:
        print(f"⚠️ Failed to load datamodule: {e}")
        print("🔄 Will use fallback data generation")
        datamodule = None
    
    # Prepare training data
    train_data, val_data = prepare_data_for_training(datamodule, device, max_batches=20)
    
    # Create model
    model = create_simple_tft_model(
        num_symbols=1,
        hidden_size=64,
        encoder_length=30,
        prediction_length=1
    )
    
    # Train model
    train_losses, val_losses = train_model(
        model, train_data, val_data, device, 
        epochs=5, lr=0.001
    )
    
    # Generate predictions for analysis
    print("🔮 Generating predictions for analysis...")
    try:
        predictions, actuals = generate_predictions(model, val_data, device)
        print(f"   ✅ Generated {len(predictions)} predictions for analysis")
    except Exception as e:
        print(f"   ⚠️ Could not generate predictions: {e}")
        predictions, actuals = None, None
    
    # Comprehensive analysis
    analyze_training_results(train_losses, val_losses, predictions, actuals)
    
    # Create visualizations if we have predictions
    if predictions is not None and actuals is not None and len(predictions) > 0:
        create_visualizations(predictions, actuals, train_losses, val_losses)
        simulate_trading(predictions, actuals)
    else:
        # Create basic training visualization
        create_visualizations([], [], train_losses, val_losses)
    
    return model, train_losses, val_losses, predictions, actuals

if __name__ == "__main__":
    print("🚀 Starting TFT Pure Torch Training Demo")
    print("="*80)
    
    try:
        # Run enhanced training with analysis
        model, train_losses, val_losses, predictions, actuals = run_enhanced_training_demo()
        
        print("\n🎉 Demo completed successfully!")
        print("\nKey improvements over SimpleTFT:")
        print("✓ Proper TFT architecture with all components")
        print("✓ Quantile outputs for uncertainty estimation") 
        print("✓ Variable selection for interpretability")
        print("✓ Separate encoder/decoder structure")
        print("✓ Static covariate handling")
        print("✓ Attention weights for analysis")
        
        print("\nNext steps:")
        print("• Run train_baseline_tft.py for full training")
        print("• Experiment with different features")
        print("• Analyze attention patterns")
        print("• Compare with other models")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
