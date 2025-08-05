#!/usr/bin/env python3
"""
Deep Learning Baselines for Time Series Forecasting
===================================================

This script provides a framework for training and evaluating

Available Models:
- LSTM
- GRU
- GRU with Soft Alignment (Attention)
- Encoder-Decoder Transformer
- Encoder-only Transformer

Usage:
    python deeplearning_baselines.py
"""

import argparse
import json
import os
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import math
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.optim as optim
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

import matplotlib
matplotlib.use('Agg')  # Set non-GUI backend before importing pyplot
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
from matplotlib.collections import LineCollection
import matplotlib.dates as mdates

# Try to import scipy for statistical analysis

from scipy import stats
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import math
# Add current directory to path to import project modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from dataModule.interface import get_data_loader
from cache_manager import print_cache_info, clear_all_cache

warnings.filterwarnings('ignore')

# --- Model Definitions ---

class BaseModel(nn.Module):
    """Abstract base model for all baselines."""
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        self.predict_len = config['predict_len']
        self.horizon_days = config.get('horizon_days', config['predict_len'])

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        raise NotImplementedError
    
    def autoregressive_predict(self, batch: Dict[str, torch.Tensor], steps: Optional[int] = None) -> torch.Tensor:
        """
        Autoregressive prediction for horizon-based forecasting.
        Default implementation falls back to standard forward pass.
        """
        if steps is None:
            steps = self.predict_len
        return self.forward(batch)

class LSTMModel(BaseModel):
    """LSTM forecasting model with autoregressive capability."""
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.hidden_size = config['hidden_size']
        self.num_layers = config.get('num_layers', 2)
        
        self.lstm = nn.LSTM(
            input_size=config['input_feature_dim'],
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=config['dropout']
        )
        # Output single step predictions for autoregressive forecasting
        self.fc = nn.Linear(self.hidden_size, 1)
        # Alternative: direct multi-step prediction
        self.fc_multi = nn.Linear(self.hidden_size, self.predict_len)

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Direct multi-step prediction."""
        x_past = batch['x_past_features']
        _, (h_n, _) = self.lstm(x_past)
        # Use the hidden state from the last layer for multi-step prediction
        out = self.fc_multi(h_n[-1])
        return out
    
    def autoregressive_predict(self, batch: Dict[str, torch.Tensor], steps: Optional[int] = None) -> torch.Tensor:
        """Autoregressive prediction step by step."""
        if steps is None:
            steps = self.predict_len
            
        x_past = batch['x_past_features']
        batch_size = x_past.size(0)
        
        # Initialize predictions tensor
        predictions = torch.zeros(batch_size, steps, device=x_past.device)
        
        # Get initial hidden state
        lstm_out, (h_n, c_n) = self.lstm(x_past)
        
        # Predict step by step
        for step in range(steps):
            # Predict next value using current hidden state
            pred = self.fc(h_n[-1]).squeeze(-1)  # (batch_size,)
            predictions[:, step] = pred
            
            # Update hidden state with prediction (simplified - could be improved)
            # In practice, you might want to use the prediction as input for next step
            pred_input = pred.unsqueeze(1).unsqueeze(2)  # (batch_size, 1, 1)
            # Pad to match input feature dimension if needed
            if pred_input.size(-1) < x_past.size(-1):
                padding = torch.zeros(batch_size, 1, x_past.size(-1) - 1, device=x_past.device)
                pred_input = torch.cat([pred_input, padding], dim=-1)
            
            # Pass through LSTM to update hidden state
            _, (h_n, c_n) = self.lstm(pred_input, (h_n, c_n))
        
        return predictions

class GRUModel(BaseModel):
    """GRU forecasting model with horizon-based capabilities."""
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.hidden_size = config['hidden_size']
        self.num_layers = config.get('num_layers', 2)

        self.gru = nn.GRU(
            input_size=config['input_feature_dim'],
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=config['dropout']
        )
        # Multi-step prediction head
        self.fc = nn.Linear(self.hidden_size, self.predict_len)
        # Single-step prediction for autoregressive mode
        self.fc_single = nn.Linear(self.hidden_size, 1)

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        x_past = batch['x_past_features']
        _, h_n = self.gru(x_past)
        out = self.fc(h_n[-1])
        return out

class SoftAlignGRUModel(BaseModel):
    """GRU model with soft attention mechanism and horizon-based prediction."""
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.hidden_size = config['hidden_size']
        self.num_layers = config.get('num_layers', 2)

        self.gru = nn.GRU(
            input_size=config['input_feature_dim'],
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=config['dropout']
        )
        
        # Attention mechanism
        self.attn = nn.Linear(self.hidden_size, self.hidden_size)
        self.v = nn.Parameter(torch.rand(self.hidden_size))
        
        # Multi-horizon prediction head
        self.fc = nn.Linear(self.hidden_size, self.predict_len)

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        x_past = batch['x_past_features']
        outputs, h_n = self.gru(x_past)
        
        # Attention mechanism
        energy = torch.tanh(self.attn(outputs))
        attn_weights = torch.softmax(torch.einsum('bij,j->bi', energy, self.v), dim=1)
        context = torch.einsum('bi,bij->bj', attn_weights, outputs)
        
        # Predict all horizon steps at once
        out = self.fc(context)
        return out

class EncDecTransformerModel(BaseModel):
    """Encoder-Decoder Transformer model with horizon-based prediction."""
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.transformer = nn.Transformer(
            d_model=config['hidden_size'],
            nhead=config['num_heads'],
            num_encoder_layers=config.get('num_layers', 2),
            num_decoder_layers=config.get('num_layers', 2),
            dim_feedforward=config['hidden_size'] * 4,
            dropout=config['dropout'],
            batch_first=True
        )
        self.input_proj = nn.Linear(config['input_feature_dim'], config['hidden_size'])
        # Predict full horizon at once
        self.output_proj = nn.Linear(config['hidden_size'], 1)
        self.pos_encoder = nn.Parameter(torch.randn(1, config['encoder_len'], config['hidden_size']))
        self.pos_decoder = nn.Parameter(torch.randn(1, config['predict_len'], config['hidden_size']))
        # Robustness enhancements
        self.layer_norm = nn.LayerNorm(config['hidden_size'])
        self.dropout_layer = nn.Dropout(config['dropout'])

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        src = self.input_proj(batch['x_past_features']) + self.pos_encoder
        
        # Decoder input for horizon prediction
        tgt = torch.zeros(src.size(0), self.predict_len, src.size(2), device=src.device)
        tgt = tgt + self.pos_decoder

        output = self.transformer(src, tgt)
        # Output shape: (batch_size, predict_len, 1) -> (batch_size, predict_len)
        return self.output_proj(output).squeeze(-1)

class EncoderOnlyTransformerModel(BaseModel):
    """Encoder-only Transformer model."""
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config['hidden_size'],
            nhead=config['num_heads'],
            dim_feedforward=config['hidden_size'] * 4,
            dropout=config['dropout'],
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.get('num_layers', 2)
        )
        self.input_proj = nn.Linear(config['input_feature_dim'], config['hidden_size'])
        # Enhanced decoding head: two-layer MLP with non-linear activation and dropout
        self.fc = nn.Sequential(
            nn.Linear(config['hidden_size'], config['hidden_size']),
            nn.ReLU(),
            nn.Dropout(config['dropout']),
            nn.Linear(config['hidden_size'], self.predict_len)
        )
        self.pos_encoder = nn.Parameter(torch.randn(1, config['encoder_len'], config['hidden_size']))
        # Robustness enhancements
        self.layer_norm = nn.LayerNorm(config['hidden_size'])
        self.dropout_layer = nn.Dropout(config['dropout'])

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        # Input projection with positional encoding
        src = self.input_proj(batch['x_past_features']) + self.pos_encoder
        src = self.layer_norm(self.dropout_layer(src))
        # Transformer encoding
        encoder_output = self.transformer_encoder(src)  # (batch, seq_len, hidden_size)
        # Use last time step's hidden state for prediction
        last_state = encoder_output[:, -1, :]  # (batch, hidden_size)
        # MLP decoding head
        return self.fc(last_state)  # (batch, predict_len)

# --- Training and Validation Functions ---

def train_model(model: nn.Module, loader: DataLoader, criterion: nn.Module, optimizer: torch.optim.Optimizer, epochs: int, device: torch.device) -> float:
    """Generic training loop for a model."""
    model.train()
    avg_loss = float('nan')
    for epoch in range(epochs):
        total_loss = 0
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}")
        for x, y in pbar:
            optimizer.zero_grad()
            
            # Create the single feature tensor
            x_past_features = torch.cat([x['encoder_cont'], x['encoder_cat'].float()], dim=-1).to(device)
            
            # The batch for the model only needs the past features
            batch = {'x_past_features': x_past_features}
            
            predictions = model(batch)
            # FIXED: Use the correct target format for horizon-based prediction
            # y is a list where y[0] contains the actual targets
            if isinstance(y, (list, tuple)):
                targets = y[0].to(device)
            else:
                targets = y.to(device)
            
            # Ensure targets match prediction dimensions
            if targets.dim() == 1:
                targets = targets.unsqueeze(-1)
            if targets.shape[-1] == 1 and predictions.shape[-1] > 1:
                # If targets are single values but we predict multiple horizons,
                # we need to repeat the target or use sequence targets
                print(f"Warning: Target shape {targets.shape} doesn't match prediction shape {predictions.shape}")
                targets = targets.repeat(1, predictions.shape[-1])
            
            loss = criterion(predictions, targets)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")
        avg_loss = total_loss / len(loader) if len(loader) > 0 else float('nan')
        print(f"Epoch {epoch+1} finished. Average Training Loss: {avg_loss:.4f}")
    # Return the final epoch's average loss
    return avg_loss

def validate_model(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device, horizon_days: int = 1, debug_mode: bool = True) -> Tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    """Validation loop to get predictions and targets with enhanced debugging."""
    model.eval()
    all_preds, all_targets, all_last_known = [], [], []
    
    print(f"\n--- Starting Validation (Horizon: {horizon_days} days) ---")
    
    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(tqdm(loader, desc="Validating")):
            # Create the single feature tensor
            x_past_features = torch.cat([x['encoder_cont'], x['encoder_cat'].float()], dim=-1).to(device)
            
            # The batch for the model only needs the past features
            batch = {'x_past_features': x_past_features}
            
            predictions = model(batch)
            
            # FIXED: Handle target format correctly for horizon-based prediction
            if isinstance(y, (list, tuple)):
                targets = y[0]
            else:
                targets = y
            
            # The last known price is the last value in the encoder_target sequence
            last_known_price = x['encoder_target'][:, -1]
            
            # Debug first batch
            if batch_idx == 0 and debug_mode:
                print(f"🔍 Batch {batch_idx} Debug Info:")
                print(f"   Input features shape: {x_past_features.shape}")
                print(f"   Encoder target shape: {x['encoder_target'].shape}")
                print(f"   Prediction shape: {predictions.shape}")
                print(f"   Target shape: {targets.shape}")
                print(f"   Last known price shape: {last_known_price.shape}")
                
                # Sample values for leakage detection
                last_price_sample = last_known_price[0].cpu().item()
                pred_sample = predictions[0, 0].cpu().item() if predictions.dim() > 1 else predictions[0].cpu().item()
                target_sample = targets[0, 0].cpu().item() if targets.dim() > 1 else targets[0].cpu().item()
                
                print(f"   Last encoder price (sample 0): {last_price_sample:.4f}")
                print(f"   First prediction (sample 0): {pred_sample:.4f}")
                print(f"   First target (sample 0): {target_sample:.4f}")
                
                # Check if prediction is suspiciously close to last encoder price
                if abs(last_price_sample) > 1e-6:  # Avoid division by zero
                    print(f"   Target change vs last encoder: {((target_sample/last_price_sample - 1) * 100):.2f}%")
                    print(f"   Prediction change vs last encoder: {((pred_sample/last_price_sample - 1) * 100):.2f}%")
                    
                    # Check for potential data leakage
                    pred_vs_encoder_diff = abs(pred_sample - last_price_sample) / abs(last_price_sample)
                    target_vs_encoder_diff = abs(target_sample - last_price_sample) / abs(last_price_sample)
                    
                    if pred_vs_encoder_diff < 0.001:  # Less than 0.1% difference
                        print(f"   ⚠️  WARNING: Prediction suspiciously close to last encoder price!")
                    if target_vs_encoder_diff < 0.001:
                        print(f"   ⚠️  WARNING: Target suspiciously close to last encoder price!")
                else:
                    print(f"   📊 Note: Data appears to be normalized (last encoder price ≈ 0)")
                    print(f"   📊 Target value: {target_sample:.4f}")
                    print(f"   📊 Prediction value: {pred_sample:.4f}")
                    print(f"   📊 Absolute difference: {abs(pred_sample - target_sample):.4f}")
                    
                    # For normalized data, check if predictions are suspiciously close to targets
                    if abs(pred_sample - target_sample) < 0.001:
                        print(f"   ⚠️  WARNING: Prediction suspiciously close to target (possible overfitting/leakage)!")
                    else:
                        print(f"   ✅ Prediction vs target difference appears reasonable")

            all_preds.append(predictions.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
            all_last_known.append(last_known_price.cpu().numpy())

    if not all_preds:
        return None, None, None

    predictions = np.concatenate(all_preds, axis=0)
    targets = np.concatenate(all_targets, axis=0)
    last_known_prices = np.concatenate(all_last_known, axis=0)
    
    print(f"📊 Validation Dataset Summary:")
    print(f"   Total samples: {predictions.shape[0]}")
    if predictions.ndim > 1:
        print(f"   Horizon length: {predictions.shape[1]}")
    
    # Enhanced data leakage detection
    print(f"\n🔒 Data Leakage Analysis:")
    first_predictions = predictions[:, 0] if predictions.ndim > 1 else predictions
    first_targets = targets[:, 0] if targets.ndim > 1 else targets
    
    # Check if data is normalized (last known prices near zero)
    is_normalized = np.mean(np.abs(last_known_prices)) < 0.1
    print(f"   Data appears to be normalized: {is_normalized}")
    print(f"   Mean absolute last encoder price: {np.mean(np.abs(last_known_prices)):.6f}")
    print(f"   Mean absolute prediction: {np.mean(np.abs(first_predictions)):.6f}")
    print(f"   Mean absolute target: {np.mean(np.abs(first_targets)):.6f}")
    
    # Calculate correlations (safe for normalized data)
    try:
        pred_encoder_corr = np.corrcoef(first_predictions, last_known_prices)[0, 1]
        target_encoder_corr = np.corrcoef(first_targets, last_known_prices)[0, 1]
        pred_target_corr = np.corrcoef(first_predictions, first_targets)[0, 1]
        
        print(f"   Prediction vs Last Encoder Price correlation: {pred_encoder_corr:.4f}")
        print(f"   Target vs Last Encoder Price correlation: {target_encoder_corr:.4f}")
        print(f"   Prediction vs Target correlation: {pred_target_corr:.4f}")
        
        if not is_normalized:
            # Traditional leakage checks for non-normalized data
            if pred_encoder_corr > 0.99:
                print(f"   ❌ CRITICAL: Predictions are nearly identical to last encoder prices!")
            elif pred_encoder_corr > 0.95:
                print(f"   ⚠️  WARNING: Predictions are suspiciously correlated with last encoder prices!")
            else:
                print(f"   ✅ Predictions appear independent of last encoder prices")
        else:
            # For normalized data, check prediction-target correlation
            print(f"   📊 Note: Data is normalized - focusing on prediction quality")
            
        if pred_target_corr > 0.99:
            print(f"   ❌ CRITICAL: Predictions are nearly perfect (possible data leakage)!")
        elif pred_target_corr > 0.95:
            print(f"   ⚠️  WARNING: Predictions are suspiciously accurate!")
        elif pred_target_corr > 0.5:
            print(f"   ✅ Good prediction accuracy (correlation: {pred_target_corr:.3f})")
        elif pred_target_corr > 0.0:
            print(f"   📈 Moderate prediction accuracy (correlation: {pred_target_corr:.3f})")
        else:
            print(f"   ⚠️  Poor prediction accuracy (correlation: {pred_target_corr:.3f})")
    except Exception as e:
        print(f"   ❌ Error calculating correlations: {e}")
    
    # Debug information about shapes
    print(f"[DEBUG] Final validation shapes:")
    print(f"  Predictions: {predictions.shape}")
    print(f"  Targets: {targets.shape}")
    print(f"  Last known prices: {last_known_prices.shape}")
    
    return predictions, targets, last_known_prices

def plot_candlestick_predictions(predictions: np.ndarray, targets: np.ndarray, last_known_prices: np.ndarray, 
                                model_name: str, plot_dir: str, symbol: str):
    """Generates candlestick-style plot comparing predictions vs actual prices."""
    os.makedirs(plot_dir, exist_ok=True)
    predict_len = predictions.shape[1]
    
    print(f"    - Generating candlestick prediction plot...")
    
    try:
        # Check if predictions need to be converted to absolute prices
        preds = np.asarray(predictions, dtype=np.float32)
        trues = np.asarray(targets, dtype=np.float32)
        last_prices = np.asarray(last_known_prices, dtype=np.float32)
        
        # Convert relative predictions to absolute prices if needed
        if np.abs(preds).mean() < np.abs(trues).mean() * 0.1:
            preds = preds + last_prices[:, None]
        
        # Select a subset of samples for clarity (max 50 samples)
        n_samples = min(50, len(preds))
        indices = np.linspace(0, len(preds)-1, n_samples, dtype=int)
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12))
        
        # --- Top plot: Individual sample trajectories ---
        x_days = np.arange(predict_len)
        
        for i in indices[:10]:  # Show first 10 samples
            # Actual trajectory
            ax1.plot(x_days, trues[i], 'b-', alpha=0.6, linewidth=1.5, label='Actual' if i == indices[0] else "")
            # Predicted trajectory  
            ax1.plot(x_days, preds[i], 'r--', alpha=0.6, linewidth=1.5, label='Predicted' if i == indices[0] else "")
            
            # Add start point (last known price)
            ax1.scatter([-1], [last_prices[i]], c='green', s=30, alpha=0.8, 
                       label='Last Known Price' if i == indices[0] else "")
        
        ax1.set_title(f'{symbol} - {model_name}: Price Trajectory Comparison (Sample Paths)')
        ax1.set_xlabel('Days into Forecast')
        ax1.set_ylabel('Price ($)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # --- Bottom plot: Candlestick-style comparison ---
        # Create OHLC-style data for each day in the horizon
        days = np.arange(predict_len)
        
        # Calculate daily statistics across all samples
        actual_open = trues[:, 0] if predict_len > 0 else last_prices
        actual_close = trues[:, -1] if predict_len > 0 else last_prices
        actual_high = np.max(trues, axis=1) if predict_len > 1 else trues[:, 0]
        actual_low = np.min(trues, axis=1) if predict_len > 1 else trues[:, 0]
        
        pred_open = preds[:, 0] if predict_len > 0 else last_prices
        pred_close = preds[:, -1] if predict_len > 0 else last_prices
        pred_high = np.max(preds, axis=1) if predict_len > 1 else preds[:, 0]
        pred_low = np.min(preds, axis=1) if predict_len > 1 else preds[:, 0]
        
        # Calculate percentiles for each day across all samples
        actual_percentiles = np.percentile(trues, [10, 25, 50, 75, 90], axis=0)
        pred_percentiles = np.percentile(preds, [10, 25, 50, 75, 90], axis=0)
        
        # Plot as box plots with whiskers
        box_width = 0.3
        
        for day in days:
            # Actual data (blue boxes)
            ax2.add_patch(Rectangle((day - box_width/2, actual_percentiles[1, day]), 
                                   box_width, actual_percentiles[3, day] - actual_percentiles[1, day],
                                   facecolor='lightblue', edgecolor='blue', alpha=0.7))
            
            # Predicted data (red boxes)
            ax2.add_patch(Rectangle((day + box_width/2, pred_percentiles[1, day]), 
                                   box_width, pred_percentiles[3, day] - pred_percentiles[1, day],
                                   facecolor='lightcoral', edgecolor='red', alpha=0.7))
            
            # Median lines
            ax2.plot([day - box_width/2, day + box_width/2], 
                    [actual_percentiles[2, day], actual_percentiles[2, day]], 
                    'b-', linewidth=2)
            ax2.plot([day - box_width/2, day + box_width/2], 
                    [pred_percentiles[2, day], pred_percentiles[2, day]], 
                    'r-', linewidth=2)
            
            # Whiskers (10th to 90th percentiles)
            ax2.plot([day, day], [actual_percentiles[0, day], actual_percentiles[4, day]], 
                    'b-', linewidth=1, alpha=0.8)
            ax2.plot([day, day], [pred_percentiles[0, day], pred_percentiles[4, day]], 
                    'r-', linewidth=1, alpha=0.8)
        
        # Connect medians with lines
        ax2.plot(days, actual_percentiles[2, :], 'b-', linewidth=2, label='Actual Median', alpha=0.8)
        ax2.plot(days, pred_percentiles[2, :], 'r--', linewidth=2, label='Predicted Median', alpha=0.8)
        
        ax2.set_title(f'{symbol} - {model_name}: Price Distribution Comparison (Box Plot Style)')
        ax2.set_xlabel('Days into Forecast')
        ax2.set_ylabel('Price ($)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, f'{symbol}_{model_name}_candlestick_predictions.png'), dpi=150)
        plt.close(fig)
        print("    - Candlestick prediction plot saved.")
        
    except Exception as e:
        print(f"    - Error generating candlestick plot: {e}")

def calculate_model_complexity_metrics(model: nn.Module) -> Dict[str, Any]:
    """Calculate various complexity metrics for a model."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Calculate memory usage (approximate)
    param_memory = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_memory = sum(b.numel() * b.element_size() for b in model.buffers())
    total_memory = param_memory + buffer_memory
    
    # Count layers by type
    layer_counts = {}
    for name, module in model.named_modules():
        layer_type = type(module).__name__
        layer_counts[layer_type] = layer_counts.get(layer_type, 0) + 1
    
    return {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'memory_mb': total_memory / (1024 * 1024),
        'layer_counts': layer_counts
    }

# --- Plotting Functions ---
def plot_evaluation_suite(predictions: np.ndarray, targets: np.ndarray, last_known_prices: np.ndarray, model_name: str, plot_dir: str, symbol: str):
    """Generates and saves a suite of evaluation plots for a model."""
    os.makedirs(plot_dir, exist_ok=True)
    predict_len = predictions.shape[1]
    fig = None
    print(f"  Plotting for {model_name}. Predictions shape: {predictions.shape}, Targets shape: {targets.shape}")

    try:
        # Check if predictions need to be converted to absolute prices
        preds = np.asarray(predictions, dtype=np.float32)
        trues = np.asarray(targets, dtype=np.float32)
        last_prices = np.asarray(last_known_prices, dtype=np.float32)
        
        # Convert relative predictions to absolute prices if needed
        if np.abs(preds).mean() < np.abs(trues).mean() * 0.1:
            preds = preds + last_prices[:, None]
        
        # --- 1. Enhanced Price Prediction Trajectory Plot ---
        print("    - Generating enhanced price prediction trajectory plot...")
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
        
        # Top subplot: Sample trajectories
        num_examples = min(10, len(preds))
        indices = np.random.choice(len(preds), num_examples, replace=False)
        # Generate colors for each trajectory
        colors = []
        for i in range(num_examples):
            colors.append(plt.cm.get_cmap('viridis')(i / max(1, num_examples - 1)))
        
        for idx, i in enumerate(indices):
            # Create full trajectory including last known price
            days = np.arange(-1, predict_len)  # -1 for last known, 0 to predict_len-1 for forecast
            actual_trajectory = np.concatenate(([last_prices[i]], trues[i]))
            pred_trajectory = np.concatenate(([last_prices[i]], preds[i]))
            
            ax1.plot(days, actual_trajectory, '--', color=colors[idx], alpha=0.7, 
                    label=f'Actual {idx+1}' if idx < 3 else "")
            ax1.plot(days, pred_trajectory, '-', color=colors[idx], alpha=0.8, 
                    label=f'Predicted {idx+1}' if idx < 3 else "")
        
        ax1.axvline(x=0, color='red', linestyle=':', alpha=0.8, label='Prediction Start')
        ax1.set_title(f'{symbol} - {model_name}: Price Prediction Trajectories (Sample)')
        ax1.set_xlabel('Days from Prediction Start')
        ax1.set_ylabel('Price ($)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Bottom subplot: Average trajectory with confidence intervals
        mean_actual = np.mean(trues, axis=0)
        mean_pred = np.mean(preds, axis=0)
        std_actual = np.std(trues, axis=0)
        std_pred = np.std(preds, axis=0)
        
        days_forecast = np.arange(predict_len)
        
        ax2.fill_between(days_forecast, mean_actual - std_actual, mean_actual + std_actual, 
                        alpha=0.3, color='blue', label='Actual ±1σ')
        ax2.fill_between(days_forecast, mean_pred - std_pred, mean_pred + std_pred, 
                        alpha=0.3, color='orange', label='Predicted ±1σ')
        ax2.plot(days_forecast, mean_actual, '--', color='blue', linewidth=2, label='Actual Mean')
        ax2.plot(days_forecast, mean_pred, '-', color='orange', linewidth=2, label='Predicted Mean')
        
        ax2.set_title(f'{symbol} - {model_name}: Average Prediction Performance')
        ax2.set_xlabel('Days into Forecast Horizon')
        ax2.set_ylabel('Price ($)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, f'{symbol}_{model_name}_price_predictions.png'), dpi=150)
        plt.close(fig)
        print("    - Enhanced price prediction plot saved.")

        # --- 2. Final Price Scatter Plot ---
        print("    - Generating final price scatter plot...")
        final_preds, final_targets = preds[:, -1], trues[:, -1]
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Create scatter plot with alpha based on density
        ax.scatter(final_targets, final_preds, alpha=0.6, s=20, edgecolors='none')
        
        # Perfect prediction line
        min_val = np.minimum(np.min(final_targets), np.min(final_preds))
        max_val = np.maximum(np.max(final_targets), np.max(final_preds))
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
        
        # Calculate and display R²
        try:
            r2_final = r2_score(final_targets, final_preds)
            ax.text(0.05, 0.95, f'R² = {r2_final:.3f}', transform=ax.transAxes, 
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        except:
            pass
        
        ax.set_title(f'{symbol} - {model_name}: Final Day Prediction vs. Actual')
        ax.set_xlabel('Actual Final Price ($)')
        ax.set_ylabel('Predicted Final Price ($)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, f'{symbol}_{model_name}_final_price_scatter.png'), dpi=150)
        plt.close(fig)
        print("    - Final price scatter plot saved.")

        # --- 3. Enhanced Cumulative Returns Comparison ---
        print("    - Generating enhanced cumulative returns plot...")
        
        # Calculate returns for different horizons
        horizons_to_test = [1, predict_len//4, predict_len//2, predict_len-1] if predict_len > 4 else [1, predict_len-1]
        horizons_to_test = [h for h in horizons_to_test if h < predict_len]
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()
        
        for idx, horizon in enumerate(horizons_to_test[:4]):
            if idx >= len(axes):
                break
                
            # Calculate returns
            actual_returns = (trues[:, horizon] - last_prices) / (last_prices + 1e-9)
            pred_returns = (preds[:, horizon] - last_prices) / (last_prices + 1e-9)
            
            # Simple strategy: go long if predicted return > 0
            pred_signals = (pred_returns > 0).astype(int)
            strategy_returns = pred_signals * actual_returns
            
            # Cumulative returns
            cum_actual = np.cumsum(actual_returns)
            cum_strategy = np.cumsum(strategy_returns)
            
            ax = axes[idx]
            ax.plot(cum_actual, label='Buy and Hold', color='blue', linewidth=2)
            ax.plot(cum_strategy, label=f'{model_name} Strategy', color='orange', linewidth=2)
            
            # Calculate strategy metrics
            strategy_sharpe = np.mean(strategy_returns) / (np.std(strategy_returns) + 1e-9) * np.sqrt(252)
            buy_hold_sharpe = np.mean(actual_returns) / (np.std(actual_returns) + 1e-9) * np.sqrt(252)
            
            ax.set_title(f'Horizon: {horizon+1} Day(s)\nStrategy Sharpe: {strategy_sharpe:.2f}, B&H Sharpe: {buy_hold_sharpe:.2f}')
            ax.set_xlabel('Time (Samples)')
            ax.set_ylabel('Cumulative Return')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Hide unused subplots
        for idx in range(len(horizons_to_test), len(axes)):
            axes[idx].set_visible(False)
        
        plt.suptitle(f'{symbol} - {model_name}: Cumulative Returns Comparison')
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, f'{symbol}_{model_name}_cumulative_returns.png'), dpi=150)
        plt.close(fig)
        print("    - Enhanced cumulative returns plot saved.")

        # --- 4. Enhanced Error Analysis ---
        print("    - Generating enhanced error analysis...")
        errors = preds - trues  # Use corrected predictions
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Error distribution
        values = errors.flatten()
        n, bins, patches = ax1.hist(values, bins=50, density=True, alpha=0.7, color='skyblue', edgecolor='black')
        ax1.axvline(0, color='red', linestyle='--', linewidth=2, label='Zero Error')
        ax1.set_title('Distribution of Prediction Errors')
        ax1.set_xlabel('Prediction Error ($)')
        ax1.set_ylabel('Density')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Error over forecast horizon
        mean_abs_error_by_horizon = np.mean(np.abs(errors), axis=0)
        std_abs_error_by_horizon = np.std(np.abs(errors), axis=0)
        days = np.arange(1, predict_len + 1)
        
        ax2.plot(days, mean_abs_error_by_horizon, 'o-', color='red', linewidth=2, label='Mean Absolute Error')
        ax2.fill_between(days, 
                        mean_abs_error_by_horizon - std_abs_error_by_horizon,
                        mean_abs_error_by_horizon + std_abs_error_by_horizon,
                        alpha=0.3, color='red', label='±1σ')
        ax2.set_title('Error Growth Over Forecast Horizon')
        ax2.set_xlabel('Days into Forecast')
        ax2.set_ylabel('Mean Absolute Error ($)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # R² over forecast horizon
        r2_by_horizon = []
        for i in range(predict_len):
            try:
                r2_h = r2_score(trues[:, i], preds[:, i])
                r2_by_horizon.append(r2_h)
            except:
                r2_by_horizon.append(0)
        
        ax3.plot(days, r2_by_horizon, 's-', color='green', linewidth=2, label='R² Score')
        ax3.axhline(0, color='black', linestyle=':', alpha=0.5)
        ax3.set_title('Prediction Quality Over Forecast Horizon')
        ax3.set_xlabel('Days into Forecast')
        ax3.set_ylabel('R² Score')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Residuals vs predicted (diagnostic)
        sample_size = min(1000, len(preds))  # Sample for performance
        indices = np.random.choice(len(preds), sample_size, replace=False)
        pred_sample = preds[indices].flatten()
        error_sample = errors[indices].flatten()
        
        ax4.scatter(pred_sample, error_sample, alpha=0.5, s=10)
        ax4.axhline(0, color='red', linestyle='--', linewidth=2)
        ax4.set_title('Residuals vs. Predicted Values')
        ax4.set_xlabel('Predicted Price ($)')
        ax4.set_ylabel('Residual (Predicted - Actual)')
        ax4.grid(True, alpha=0.3)
        
        plt.suptitle(f'{symbol} - {model_name}: Error Analysis')
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, f'{symbol}_{model_name}_error_analysis.png'), dpi=150)
        plt.close(fig)
        print("    - Enhanced error analysis plot saved.")

        # --- 5. Horizon-Specific Performance Metrics ---
        print("    - Generating horizon-specific performance metrics...")
        
        # Calculate directional accuracy for each horizon
        directional_accuracy = []
        for i in range(predict_len):
            actual_direction = trues[:, i] > last_prices
            pred_direction = preds[:, i] > last_prices
            accuracy = np.mean(actual_direction == pred_direction)
            directional_accuracy.append(accuracy)
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 10))
        
        # Directional accuracy over horizon
        ax1.plot(days, directional_accuracy, 'o-', color='purple', linewidth=2, markersize=6)
        ax1.axhline(0.5, color='red', linestyle='--', alpha=0.7, label='Random Guess')
        ax1.set_title('Directional Accuracy Over Forecast Horizon')
        ax1.set_xlabel('Days into Forecast')
        ax1.set_ylabel('Directional Accuracy')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 1)
        
        # Correlation over horizon
        correlations = []
        for i in range(predict_len):
            try:
                corr = np.corrcoef(trues[:, i], preds[:, i])[0, 1]
                correlations.append(corr)
            except:
                correlations.append(0)
        
        ax2.plot(days, correlations, 's-', color='orange', linewidth=2, markersize=6)
        ax2.axhline(0, color='black', linestyle=':', alpha=0.5)
        ax2.set_title('Correlation Over Forecast Horizon')
        ax2.set_xlabel('Days into Forecast')
        ax2.set_ylabel('Pearson Correlation')
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(-1, 1)
        
        # Relative MSE vs baseline
        baseline_preds_local = np.tile(last_prices[:, None], (1, predict_len))  # Constant last price prediction
        baseline_mse_by_horizon = np.mean((baseline_preds_local - trues)**2, axis=0)
        model_mse_by_horizon = np.mean((preds - trues)**2, axis=0)
        relative_mse = model_mse_by_horizon / (baseline_mse_by_horizon + 1e-9)
        
        ax3.plot(days, relative_mse, '^-', color='red', linewidth=2, markersize=6)
        ax3.axhline(1, color='black', linestyle='--', alpha=0.7, label='Baseline Performance')
        ax3.set_title('Relative MSE vs. Naive Baseline')
        ax3.set_xlabel('Days into Forecast')
        ax3.set_ylabel('Model MSE / Baseline MSE')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.set_yscale('log')
        
        # Prediction confidence intervals
        pred_percentiles = np.percentile(preds, [10, 25, 50, 75, 90], axis=0)
        actual_percentiles = np.percentile(trues, [10, 25, 50, 75, 90], axis=0)
        
        ax4.fill_between(days, pred_percentiles[0], pred_percentiles[4], alpha=0.2, color='blue', label='Pred 10-90%')
        ax4.fill_between(days, pred_percentiles[1], pred_percentiles[3], alpha=0.3, color='blue', label='Pred 25-75%')
        ax4.plot(days, pred_percentiles[2], '-', color='blue', linewidth=2, label='Pred Median')
        
        ax4.fill_between(days, actual_percentiles[0], actual_percentiles[4], alpha=0.2, color='green', label='Actual 10-90%')
        ax4.fill_between(days, actual_percentiles[1], actual_percentiles[3], alpha=0.3, color='green', label='Actual 25-75%')
        ax4.plot(days, actual_percentiles[2], '--', color='green', linewidth=2, label='Actual Median')
        
        ax4.set_title('Prediction vs. Actual Distribution')
        ax4.set_xlabel('Days into Forecast')
        ax4.set_ylabel('Price ($)')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.suptitle(f'{symbol} - {model_name}: Horizon-Specific Performance Metrics')
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, f'{symbol}_{model_name}_horizon_metrics.png'), dpi=150)
        plt.close(fig)
        print("    - Horizon-specific performance metrics plot saved.")
        
        # --- 6. Candlestick-style Predictions Plot ---
        plot_candlestick_predictions(predictions, targets, last_known_prices, model_name, plot_dir, symbol)
        
    except Exception as e:
        print(f"An error occurred during plotting for {model_name}: {e}")
        if fig is not None and plt.fignum_exists(fig.number):
            plt.close(fig)

def plot_all_model_comparison(all_model_results: Dict[str, Dict], plot_dir: str, symbol: str, model_complexity: Optional[Dict[str, Dict[str, Any]]] = None):
    """Generates and saves a comprehensive comparison plot for all models."""
    os.makedirs(plot_dir, exist_ok=True)
    model_names = list(all_model_results.keys())
    num_models = len(model_names)
    
    if num_models == 0:
        print("No model results to compare.")
        return

    fig = None  # Initialize fig to None to avoid unbound error
    try:
        predict_len = all_model_results[model_names[0]]['predictions'].shape[1]

        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle(f'{symbol} - All Models Comparison (Horizon: {predict_len} days)', fontsize=16)

        # Prepare corrected predictions for all models
        model_data = {}
        for model_name, results in all_model_results.items():
            preds = np.asarray(results['predictions'], dtype=np.float32)
            trues = np.asarray(results['targets'], dtype=np.float32)
            last_prices = np.asarray(results['last_known_prices'], dtype=np.float32)
            
            # Convert relative predictions to absolute if needed
            if np.abs(preds).mean() < np.abs(trues).mean() * 0.1:
                preds = preds + last_prices[:, None]
            
            model_data[model_name] = {'preds': preds, 'trues': trues, 'last_prices': last_prices}

        # --- 1. Final Horizon R² Comparison ---
        ax = axes[0, 0]
        r2_scores = []
        for model_name, data in model_data.items():
            try:
                r2 = r2_score(data['trues'][:, -1], data['preds'][:, -1])
                r2_scores.append(r2)
            except:
                r2_scores.append(0)
        
        bars = ax.bar(model_names, r2_scores, color=['skyblue', 'lightcoral', 'lightgreen', 'gold', 'plum'][:len(model_names)])
        ax.set_title(f'R² Score - Final Day Prediction')
        ax.set_ylabel('R² Score')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, score in zip(bars, r2_scores):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{score:.3f}', ha='center', va='bottom')

        # --- 2. Error over Horizon (MAE) ---
        ax = axes[0, 1]
        for model_name, data in model_data.items():
            errors = np.abs(data['preds'] - data['trues'])
            mae_over_horizon = np.mean(errors, axis=0)
            ax.plot(np.arange(1, predict_len + 1), mae_over_horizon, marker='o', linestyle='-', 
                   label=model_name, linewidth=2, markersize=4)
        ax.set_title('Mean Absolute Error Over Horizon')
        ax.set_xlabel('Days into Forecast')
        ax.set_ylabel('MAE ($)')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # --- 3. R² over Horizon ---
        ax = axes[0, 2]
        for model_name, data in model_data.items():
            r2_over_horizon = []
            for i in range(predict_len):
                try:
                    r2_h = r2_score(data['trues'][:, i], data['preds'][:, i])
                    r2_over_horizon.append(r2_h)
                except:
                    r2_over_horizon.append(0)
            ax.plot(np.arange(1, predict_len + 1), r2_over_horizon, marker='s', linestyle='-', 
                   label=model_name, linewidth=2, markersize=4)
        ax.axhline(0, color='black', linestyle=':', alpha=0.5)
        ax.set_title('R² Score Over Horizon')
        ax.set_xlabel('Days into Forecast')
        ax.set_ylabel('R² Score')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # --- 4. Directional Accuracy Comparison ---
        ax = axes[1, 0]
        accuracies = []
        for model_name, data in model_data.items():
            # Direction relative to last known price
            actual_direction = data['trues'] > data['last_prices'][:, None]
            pred_direction = data['preds'] > data['last_prices'][:, None]
            accuracy = np.mean(actual_direction == pred_direction)
            accuracies.append(accuracy)
        
        bars = ax.bar(model_names, accuracies, color=['skyblue', 'lightcoral', 'lightgreen', 'gold', 'plum'][:len(model_names)])
        ax.axhline(0.5, color='red', linestyle='--', alpha=0.7, label='Random Guess')
        ax.set_title('Overall Directional Accuracy')
        ax.set_ylabel('Accuracy Score')
        ax.tick_params(axis='x', rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)
        
        # Add value labels on bars
        for bar, acc in zip(bars, accuracies):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{acc:.3f}', ha='center', va='bottom')

        # --- 5. Final Price Correlation ---
        ax = axes[1, 1]
        correlations = []
        for model_name, data in model_data.items():
            try:
                corr = np.corrcoef(data['trues'][:, -1], data['preds'][:, -1])[0, 1]
                correlations.append(corr)
            except:
                correlations.append(0)
        
        bars = ax.bar(model_names, correlations, color=['skyblue', 'lightcoral', 'lightgreen', 'gold', 'plum'][:len(model_names)])
        ax.set_title('Final Day Price Correlation')
        ax.set_ylabel('Pearson Correlation')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-1, 1)
        
        # Add value labels on bars
        for bar, corr in zip(bars, correlations):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{corr:.3f}', ha='center', va='bottom')

        # --- 6. Model Complexity vs Performance ---
        ax = axes[1, 2]
        if model_complexity:
            param_counts = []
            final_r2_scores = []
            
            for model_name in model_names:
                if model_name in model_complexity:
                    param_counts.append(model_complexity[model_name]['total_params'])
                    # Get final R² score
                    try:
                        data = model_data[model_name]
                        r2 = r2_score(data['trues'][:, -1], data['preds'][:, -1])
                        final_r2_scores.append(r2)
                    except:
                        final_r2_scores.append(0)
            
            if param_counts and final_r2_scores:
                # Create scatter plot
                colors = ['skyblue', 'lightcoral', 'lightgreen', 'gold', 'plum'][:len(model_names)]
                scatter = ax.scatter(param_counts, final_r2_scores, 
                                   c=colors, s=100, alpha=0.7, edgecolors='black')
                
                # Add model name labels
                for i, model_name in enumerate(model_names):
                    if i < len(param_counts):
                        ax.annotate(model_name, (param_counts[i], final_r2_scores[i]), 
                                  xytext=(5, 5), textcoords='offset points', fontsize=8)
                
                ax.set_xlabel('Model Parameters (count)')
                ax.set_ylabel('Final R² Score')
                ax.set_title('Model Complexity vs Performance')
                ax.grid(True, alpha=0.3)
                
                # Add trend line if we have enough points
                if len(param_counts) > 2:
                    z = np.polyfit(param_counts, final_r2_scores, 1)
                    p = np.poly1d(z)
                    x_trend = np.linspace(min(param_counts), max(param_counts), 100)
                    ax.plot(x_trend, p(x_trend), "r--", alpha=0.7, label=f'Trend (slope: {z[0]:.2e})')
                    ax.legend()
            else:
                ax.text(0.5, 0.5, 'Model Complexity\nData Not Available', 
                       ha='center', va='center', transform=ax.transAxes, fontsize=12)
        else:
            ax.text(0.5, 0.5, 'Model Complexity\nvs Performance\n(complexity data not provided)', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=12)
            ax.set_title('Model Complexity vs Performance')
            ax.set_xlabel('Model Parameters')
            ax.set_ylabel('Final R² Score')

        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, f'{symbol}_all_models_comparison.png'), dpi=150)
        plt.close(fig) # Close the figure explicitly
    except Exception as e:
        print(f"An error occurred during plotting: {e}")
        # If a figure is open, close it to prevent hangs
        if fig is not None and plt.fignum_exists(fig.number):
            plt.close(fig)

# --- Main Execution ---

def run_deep_learning_baselines():
    """Main function to run the deep learning baseline pipeline."""
    parser = argparse.ArgumentParser(description='Deep Learning Baselines for Time Series Forecasting')
    parser.add_argument('--clear-cache', action='store_true', help='Clear cache before running')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--symbol', type=str, default='AAPL', help='Stock symbol to train on.')
    parser.add_argument('--horizon-days', type=int, default=30, help='Prediction horizon in days (default: 30)')
    parser.add_argument('--skip-training', action='store_true', help='Skip training and generate plots from a results directory.')
    parser.add_argument('--run-dir', type=str, default=None, help='Specify a run directory to generate plots from (requires --skip-training).')
    args = parser.parse_args()

    if args.skip_training and not args.run_dir:
        print("ERROR: --run-dir must be specified when using --skip-training.")
        sys.exit(1)

    # --- Configuration ---
    config = {
        'symbol': args.symbol,
        'start_date': '2020-01-01',
        'end_date': '2020-12-31',
        'val_start_date': '2021-01-01',
        'val_end_date': '2021-12-31',
        'encoder_len': 90,
        'predict_len': args.horizon_days,  # Use horizon parameter
        'horizon_days': args.horizon_days,  # Store horizon for reference
        'batch_size': 64,
        'hidden_size': 64,
        'num_layers': 2,
        'num_heads': 4,
        'dropout': 0.1,
        'learning_rate': 0.001,
        'epochs': args.epochs,
        'patience': 3,
        'device': 'mps' if torch.backends.mps.is_available() else 'cpu'
    }
    
    # --- Directory Setup ---
    base_results_dir = "deeplearning_baseline_runs"
    run_dir = args.run_dir

    if not args.skip_training:
        # Create a new unique directory for this run
        run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = os.path.join(base_results_dir, f"{config['symbol']}_{run_timestamp}")
        os.makedirs(run_dir, exist_ok=True)
        print(f"📂 All artifacts for this run will be saved in: {run_dir}")

        # Save config to the run directory
        config_filepath = os.path.join(run_dir, 'config.json')
        with open(config_filepath, 'w') as f:
            json.dump(config, f, indent=4)
        print(f"💾 Configuration saved to {config_filepath}")

    results_filepath = os.path.join(run_dir, f"{config['symbol']}_baseline_results.npz")

    if not args.skip_training:
        if args.clear_cache:
            print("Clearing cache...")
            clear_all_cache()
            print("Cache cleared.")

        print("📦 Cache Status:")
        print_cache_info()
        print()

        print("--- Starting Deep Learning Baseline Pipeline ---")
        print(f"Configuration: {json.dumps(config, indent=4)}")
        device = torch.device(config['device'])

        # --- Data Loading with Enhanced Debugging ---
        print("\n--- Loading Data ---")
        print(f"📊 Training period: {config['start_date']} to {config['end_date']}")
        print(f"📊 Validation period: {config['val_start_date']} to {config['val_end_date']}")
        print(f"📊 Horizon length: {config['horizon_days']} days")
        print(f"📊 Encoder length: {config['encoder_len']} days")
        
        train_loader = get_data_loader(
            symbols=[config['symbol']], start=config['start_date'], end=config['end_date'],
            encoder_len=config['encoder_len'], predict_len=config['predict_len'], batch_size=config['batch_size']
        )
        validation_loader = get_data_loader(
            symbols=[config['symbol']], start=config['val_start_date'], end=config['val_end_date'],
            encoder_len=config['encoder_len'], predict_len=config['predict_len'], batch_size=config['batch_size']
        )

        # --- Enhanced Data Inspection ---
        print("\n--- Inspecting Data Quality and Leakage Prevention ---")
        try:
            # Check training data
            train_sample_x, train_sample_y = next(iter(train_loader))
            print(f"📈 Training data sample:")
            print(f"   Batch size: {train_sample_x['encoder_cont'].shape[0]}")
            print(f"   Encoder sequence length: {train_sample_x['encoder_cont'].shape[1]}")
            print(f"   Feature dimensions: {train_sample_x['encoder_cont'].shape[2]}")
            
            if isinstance(train_sample_y, (list, tuple)):
                target_shape = train_sample_y[0].shape
                target_sample = train_sample_y[0][:3].cpu().numpy()
            else:
                target_shape = train_sample_y.shape
                target_sample = train_sample_y[:3].cpu().numpy()
            print(f"   Target shape: {target_shape}")
            print(f"   Sample targets (first 3 samples): {target_sample}")
            
            # Check encoder target (last known prices)
            encoder_target = train_sample_x['encoder_target'][:3, -5:].cpu().numpy()
            print(f"   Last 5 encoder prices (first 3 samples): {encoder_target}")
            
            # Check validation data
            val_sample_x, val_sample_y = next(iter(validation_loader))
            print(f"📉 Validation data sample:")
            print(f"   Batch size: {val_sample_x['encoder_cont'].shape[0]}")
            if isinstance(val_sample_y, (list, tuple)):
                val_target_sample = val_sample_y[0][:3].cpu().numpy()
            else:
                val_target_sample = val_sample_y[:3].cpu().numpy()
            print(f"   Sample targets (first 3 samples): {val_target_sample}")
            
            val_encoder_target = val_sample_x['encoder_target'][:3, -5:].cpu().numpy()
            print(f"   Last 5 encoder prices (first 3 samples): {val_encoder_target}")
            
            # Data leakage check: ensure no overlap in time periods
            print(f"🔒 Data leakage check:")
            print(f"   Training ends: {config['end_date']}")
            print(f"   Validation starts: {config['val_start_date']}")
            
            train_end = datetime.strptime(config['end_date'], '%Y-%m-%d')
            val_start = datetime.strptime(config['val_start_date'], '%Y-%m-%d')
            gap_days = (val_start - train_end).days
            print(f"   Time gap: {gap_days} days")
            
            if gap_days < 0:
                print(f"   ❌ ERROR: Validation period overlaps with training period!")
                raise ValueError("Data leakage detected: validation period overlaps with training period")
            elif gap_days == 0:
                print(f"   ⚠️  WARNING: No gap between training and validation")
            else:
                print(f"   ✅ No temporal leakage detected")
                
        except Exception as e:
            print(f"ERROR during data inspection: {e}")
            sys.exit(1)

        # --- Determine input_feature_dim from data ---
        print("\n--- Determining model input dimensions from data ---")
        try:
            x, y = next(iter(train_loader))
            sample_features = torch.cat([x['encoder_cont'], x['encoder_cat'].float()], dim=-1)
            config['input_feature_dim'] = sample_features.shape[-1]
            print(f"Determined input_feature_dim: {config['input_feature_dim']}")
        except (StopIteration, KeyError) as e:
            print(f"ERROR: Could not determine input dimensions from data loader: {e}")
            sys.exit(1)

        # --- Model Training and Evaluation ---
        print("\n--- Training and Evaluating Models ---")
        models_to_run = {
            'LSTM': LSTMModel, 'GRU': GRUModel, 'SoftAlignGRU': SoftAlignGRUModel,
            'EncDecTransformer': EncDecTransformerModel, 'EncoderOnlyTransformer': EncoderOnlyTransformerModel
        }
        all_model_results = {}
        metrics_stats = {}
        model_complexity_data = {}

        for model_name, model_class in models_to_run.items():
            print(f"\n--- Running pipeline for {model_name} ---")
            model = model_class(config).to(device)
            
            # Calculate model complexity metrics
            complexity_metrics = calculate_model_complexity_metrics(model)
            model_complexity_data[model_name] = complexity_metrics
            print(f"📊 {model_name} Complexity:")
            print(f"   Parameters: {complexity_metrics['total_params']:,}")
            print(f"   Memory: {complexity_metrics['memory_mb']:.2f} MB")
            
            # Use AdamW optimizer for better generalization
            optimizer = AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=0.01)
            # Cosine annealing learning rate scheduler
            scheduler = CosineAnnealingLR(optimizer, T_max=config['epochs'], eta_min=1e-6)
            criterion = nn.MSELoss()

            print(f"Training {model_name} with early stopping (patience={config.get('patience')})...")
            # Early stopping based on validation loss
            best_val_loss = math.inf
            epochs_no_improve = 0
            train_loss = None
            for epoch in range(config['epochs']):
                train_loss = train_model(model, train_loader, criterion, optimizer, 1, device)
                # Step scheduler after each epoch
                scheduler.step()
                # Validate
                predictions, targets, _ = validate_model(model, validation_loader, criterion, device, 
                                                        horizon_days=config['horizon_days'], debug_mode=False)
                if predictions is None or targets is None:
                    print(f"Validation returned no results at epoch {epoch+1}")
                    break
                # Ensure types before flattening
                assert isinstance(predictions, np.ndarray) and isinstance(targets, np.ndarray), \
                    f"Invalid validation outputs at epoch {epoch+1}: {type(predictions)}, {type(targets)}"
                # Flatten arrays safely
                preds_flat = np.asarray(predictions).flatten()
                targets_flat = np.asarray(targets).flatten()
                val_loss = mean_squared_error(targets_flat, preds_flat)
                print(f"Epoch {epoch+1}: val_loss={val_loss:.6f}")
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    epochs_no_improve = 0
                    # Save best checkpoint
                    torch.save(model.state_dict(), os.path.join(run_dir, f'{model_name}_best.pt'))
                else:
                    epochs_no_improve += 1
                    if epochs_no_improve >= config.get('patience', 3):
                        print(f"Early stopping at epoch {epoch+1}")
                        break
            print(f"✅ {model_name} training complete. Best val_loss: {best_val_loss:.4f}")

            print(f"Validating {model_name}...")
            predictions, targets, last_known_prices = validate_model(model, validation_loader, criterion, device,
                                                                     horizon_days=config['horizon_days'], debug_mode=True)
            
            if predictions is not None:
                all_model_results[model_name] = {
                    "predictions": predictions, "targets": targets, "last_known_prices": last_known_prices
                }
                # Count parameters
                num_params = sum(p.numel() for p in model.parameters())
                
                # Convert to numpy arrays for metrics calculation
                preds = np.asarray(predictions, dtype=np.float32)
                trues = np.asarray(targets, dtype=np.float32)
                last_prices = np.asarray(last_known_prices, dtype=np.float32)
                
                # Debug: Check if predictions are absolute prices or relative changes
                print(f"[DEBUG] Raw predictions range: {preds.min():.4f}-{preds.max():.4f}")
                print(f"[DEBUG] Raw targets range: {trues.min():.4f}-{trues.max():.4f}")
                print(f"[DEBUG] Last known prices range: {last_prices.min():.4f}-{last_prices.max():.4f}")
                
                # Ensure predictions and targets are in the same scale (absolute prices)
                # If predictions are relative changes, convert to absolute prices
                if np.abs(preds).mean() < np.abs(trues).mean() * 0.1:  # Heuristic to detect relative predictions
                    print("[DEBUG] Converting relative predictions to absolute prices")
                    preds = preds + last_prices[:, None]
                else:
                    print("[DEBUG] Predictions appear to be absolute prices already")
                
                print(f"[DEBUG] Final preds range: {preds.min():.4f}-{preds.max():.4f}")
                print(f"[DEBUG] Final trues range: {trues.min():.4f}-{trues.max():.4f}")
                
                # Compute baseline predictions (constant last known price)
                baseline_preds = np.tile(last_prices[:, None], (1, preds.shape[1]))
                
                # Compute per-horizon metrics
                mse_per_horizon = np.mean((preds - trues)**2, axis=0)
                mae_per_horizon = np.mean(np.abs(preds - trues), axis=0)
                
                # Compute per-horizon R² and correlation
                r2_per_horizon = []
                corr_per_horizon = []
                for i in range(preds.shape[1]):
                    yt, yp = trues[:, i], preds[:, i]
                    try:
                        # Calculate R² for this horizon
                        r2_h = r2_score(yt, yp)
                        r2_per_horizon.append(float(r2_h))
                    except Exception:
                        r2_per_horizon.append(float('nan'))
                    
                    try:
                        # Calculate Pearson correlation for this horizon
                        corr_h = np.corrcoef(yt, yp)[0, 1]
                        corr_per_horizon.append(float(corr_h))
                    except Exception:
                        corr_per_horizon.append(float('nan'))
                
                # Overall metrics across all horizons
                preds_flat = preds.flatten()
                trues_flat = trues.flatten()
                baseline_flat = baseline_preds.flatten()
                
                # Overall R² calculation
                try:
                    overall_r2 = r2_score(trues_flat, preds_flat)
                except Exception:
                    overall_r2 = float('nan')
                
                # Baseline metrics for comparison
                try:
                    baseline_r2 = r2_score(trues_flat, baseline_flat)
                    baseline_mse = mean_squared_error(trues_flat, baseline_flat)
                except Exception:
                    baseline_r2 = float('nan')
                    baseline_mse = float('nan')
                
                # Average metrics
                avg_mse = float(np.mean(mse_per_horizon))
                avg_mae = float(np.mean(mae_per_horizon))
                avg_r2 = float(np.nanmean(r2_per_horizon))  # Use nanmean to handle NaN values
                
                metrics_stats[model_name] = {
                    'train_loss': train_loss,
                    'val_mse': avg_mse,
                    'mse_over_time': mse_per_horizon.tolist(),
                    'val_mae': avg_mae,
                    'mae_over_time': mae_per_horizon.tolist(),
                    'r2': overall_r2,
                    'r2_over_time': r2_per_horizon,
                    'avg_r2': avg_r2,
                    'corr_over_time': corr_per_horizon,
                    'baseline_r2': baseline_r2,
                    'baseline_mse': baseline_mse,
                    'num_params': num_params
                }
                
                print(f"📊 {model_name} stats -> params: {num_params}")
                print(f"    MSE: {avg_mse:.4f}, MAE: {avg_mae:.4f}")
                print(f"    Overall R²: {overall_r2:.4f}, Avg R²: {avg_r2:.4f}")
                print(f"    Baseline R²: {baseline_r2:.4f}, Baseline MSE: {baseline_mse:.4f}")
                # Save model checkpoint
                checkpoint_path = os.path.join(run_dir, f'{model_name}_checkpoint.pt')
                torch.save(model.state_dict(), checkpoint_path)
                print(f"💾 Checkpoint saved to {checkpoint_path}")
                print(f"✅ {model_name} pipeline finished successfully.")
            else:
                print(f"⚠️ Skipping {model_name} due to validation returning no results.")

        # --- Summary Table ---
        print("\nModel Performance Summary:")
        header = f"{'Model':<20}{'Train Loss':<15}{'Val MSE':<15}{'Val MAE':<15}{'Overall R²':<12}{'Avg R²':<10}{'Params':<10}"
        print(header)
        print('-' * len(header))
        for m, stats_ in metrics_stats.items():
            # Safely fetch metrics with fallback for legacy keys
            train_loss = stats_.get('train_loss', float('nan'))
            val_mse = stats_.get('val_mse', stats_.get('mse', float('nan')))
            val_mae = stats_.get('val_mae', stats_.get('mae', float('nan')))
            overall_r2 = stats_.get('r2', float('nan'))
            avg_r2 = stats_.get('avg_r2', overall_r2)
            num_params = stats_.get('num_params', 0)
            print(f"{m:<20}{train_loss:<15.4f}{val_mse:<15.4f}{val_mae:<15.4f}{overall_r2:<12.4f}{avg_r2:<10.4f}{num_params:<10}")
        
        print(f"\nPrediction Horizon: {config['horizon_days']} days")
        print(f"Encoder Length: {config['encoder_len']} days")

        # --- Save metrics summary to file ---
        metrics_filepath = os.path.join(run_dir, 'metrics_summary.json')
        with open(metrics_filepath, 'w') as f:
            json.dump(metrics_stats, f, indent=4)
        print(f"💾 Metrics saved to {metrics_filepath}")
        # --- Save results to file ---
        print(f"\n--- Saving results to {results_filepath} ---")
        # np.savez_compressed needs keyword arguments for each array
        save_dict = {}
        for model_name, data in all_model_results.items():
            save_dict[f"{model_name}_predictions"] = data['predictions']
            save_dict[f"{model_name}_targets"] = data['targets']
            save_dict[f"{model_name}_last_known_prices"] = data['last_known_prices']
        np.savez_compressed(results_filepath, **save_dict)
        print("✅ Results saved successfully.")
        
        # --- Generate plots with model complexity data ---
        print("Generating final model comparison plot with complexity analysis...")
        plot_all_model_comparison(all_model_results, run_dir, config['symbol'], model_complexity_data)

    # --- Generate plots from the saved file ---
    generate_plots_from_file(results_filepath, config['symbol'], run_dir)

def generate_plots_from_file(results_filepath: str, symbol: str, plot_dir: str):
    """Loads results from a file and generates all evaluation plots."""
    print(f"\n--- Generating plots from {results_filepath} ---")
    
    try:
        # Plots will be saved in the specified plot_dir (the run directory)
        os.makedirs(plot_dir, exist_ok=True)
        print(f"Plots will be saved in: '{plot_dir}'")

        results_data = np.load(results_filepath, allow_pickle=True)
        print(f"⚙️ Found result keys in {results_filepath}: {results_data.files}")
        
        all_model_results = {}
        # Correctly extract model names from the keys in the npz file
        model_names = sorted(list(set(k.rsplit('_', 1)[0] for k in results_data.keys())))

        for model_name in model_names:
            if f"{model_name}_predictions" in results_data:
                print(f"Found results for model: {model_name}")
                all_model_results[model_name] = {
                    "predictions": results_data[f"{model_name}_predictions"],
                    "targets": results_data[f"{model_name}_targets"],
                    "last_known_prices": results_data[f"{model_name}_last_known_prices"]
                }

        if not all_model_results:
            print("No model results found in the file.")
            return

        # --- Generate plots for each model ---
        for model_name, results in all_model_results.items():
            print(f"Generating plots for {model_name}...")
            plot_evaluation_suite(
                predictions=results['predictions'],
                targets=results['targets'],
                last_known_prices=results['last_known_prices'],
                model_name=model_name,
                plot_dir=plot_dir,
                symbol=symbol
            )

        # --- Generate final comparison plot ---
        print("Generating final model comparison plot...")
        plot_all_model_comparison(all_model_results, plot_dir, symbol)

        print("\n✅ All plots generated successfully.")

    except FileNotFoundError:
        print(f"ERROR: Results file not found at {results_filepath}. Run training first or check the path.")
    except Exception as e:
        print(f"An error occurred while generating plots: {e}")

# --- Entry Point ---

if __name__ == '__main__':
    run_deep_learning_baselines()
