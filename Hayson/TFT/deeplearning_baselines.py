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
from typing import Any, Dict, List, Tuple

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

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        raise NotImplementedError

class LSTMModel(BaseModel):
    """Simple LSTM forecasting model."""
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
        self.fc = nn.Linear(self.hidden_size, self.predict_len)

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        x_past = batch['x_past_features']
        _, (h_n, _) = self.lstm(x_past)
        # Use the hidden state from the last layer
        out = self.fc(h_n[-1])
        return out

class GRUModel(BaseModel):
    """Simple GRU forecasting model."""
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
        self.fc = nn.Linear(self.hidden_size, self.predict_len)

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        x_past = batch['x_past_features']
        _, h_n = self.gru(x_past)
        out = self.fc(h_n[-1])
        return out

class SoftAlignGRUModel(BaseModel):
    """GRU model with a soft attention mechanism."""
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
        
        self.fc = nn.Linear(self.hidden_size, self.predict_len)

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        x_past = batch['x_past_features']
        outputs, h_n = self.gru(x_past)
        
        # Attention
        energy = torch.tanh(self.attn(outputs))
        attn_weights = torch.softmax(torch.einsum('bij,j->bi', energy, self.v), dim=1)
        context = torch.einsum('bi,bij->bj', attn_weights, outputs)
        
        out = self.fc(context)
        return out

class EncDecTransformerModel(BaseModel):
    """Encoder-Decoder Transformer model."""
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
        self.output_proj = nn.Linear(config['hidden_size'], 1) # Predict one step at a time
        self.pos_encoder = nn.Parameter(torch.randn(1, config['encoder_len'], config['hidden_size']))
        self.pos_decoder = nn.Parameter(torch.randn(1, config['predict_len'], config['hidden_size']))
        # Robustness enhancements
        self.layer_norm = nn.LayerNorm(config['hidden_size'])
        self.dropout_layer = nn.Dropout(config['dropout'])

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        src = self.input_proj(batch['x_past_features']) + self.pos_encoder
        
        # Decoder input starts with a start token (zeros)
        tgt = torch.zeros(src.size(0), self.predict_len, src.size(2), device=src.device)
        tgt = tgt + self.pos_decoder

        output = self.transformer(src, tgt)
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
            targets = y[0].to(device)
            
            loss = criterion(predictions, targets)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")
        avg_loss = total_loss / len(loader) if len(loader) > 0 else float('nan')
        print(f"Epoch {epoch+1} finished. Average Training Loss: {avg_loss:.4f}")
    # Return the final epoch's average loss
    return avg_loss

def validate_model(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> Tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    """Validation loop to get predictions and targets."""
    model.eval()
    all_preds, all_targets, all_last_known = [], [], []
    
    with torch.no_grad():
        for x, y in tqdm(loader, desc="Validating"):
            # Create the single feature tensor
            x_past_features = torch.cat([x['encoder_cont'], x['encoder_cat'].float()], dim=-1).to(device)
            
            # The batch for the model only needs the past features
            batch = {'x_past_features': x_past_features}
            
            predictions = model(batch)
            targets = y[0]
            # The last known price is the last value in the encoder_target sequence
            last_known_price = x['encoder_target'][:, -1]

            all_preds.append(predictions.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
            all_last_known.append(last_known_price.cpu().numpy())

    if not all_preds:
        return None, None, None

    predictions = np.concatenate(all_preds, axis=0)
    targets = np.concatenate(all_targets, axis=0)
    last_known_prices = np.concatenate(all_last_known, axis=0)
    
    return predictions, targets, last_known_prices

# --- Plotting Functions ---
def plot_evaluation_suite(predictions: np.ndarray, targets: np.ndarray, last_known_prices: np.ndarray, model_name: str, plot_dir: str, symbol: str):
    """Generates and saves a suite of evaluation plots for a model."""
    os.makedirs(plot_dir, exist_ok=True)
    predict_len = predictions.shape[1]
    fig = None
    print(f"  Plotting for {model_name}. Predictions shape: {predictions.shape}, Targets shape: {targets.shape}")

    try:
        # --- 1. Price Prediction Trajectory Plot ---
        print("    - Generating price prediction trajectory plot...")
        fig, ax = plt.subplots(figsize=(15, 7))
        num_examples = min(5, len(predictions))
        for i in range(num_examples):
            full_actual = np.concatenate(([last_known_prices[i]], targets[i]))
            full_pred = np.concatenate(([last_known_prices[i]], predictions[i]))
            time_steps = np.arange(len(full_actual))
            ax.plot(time_steps, full_actual, '--', label=f'Actual Trajectory {i+1}')
            ax.plot(time_steps, full_pred, '-', label=f'Predicted Trajectory {i+1}')
        ax.set_title(f'{symbol} - {model_name}: Price Prediction Trajectories')
        ax.set_xlabel('Time Steps (0 = Last Known Price)')
        ax.set_ylabel('Price')
        ax.legend()
        ax.grid(True)
        plt.savefig(os.path.join(plot_dir, f'{symbol}_{model_name}_price_predictions.png'))
        plt.close(fig)
        print("    - Price prediction plot saved.")

        # --- 2. Final Price Scatter Plot ---
        print("    - Generating final price scatter plot...")
        final_preds, final_targets = predictions[:, -1], targets[:, -1]
        fig, ax = plt.subplots(figsize=(8, 8))
        sns.scatterplot(x=final_targets, y=final_preds, alpha=0.6, ax=ax)
        ax.plot([min(final_targets), max(final_targets)], [min(final_targets), max(final_targets)], 'r--')
        ax.set_title(f'{symbol} - {model_name}: Final Predicted Price vs. Actual')
        ax.set_xlabel('Actual Final Price')
        ax.set_ylabel('Predicted Final Price')
        ax.grid(True)
        plt.savefig(os.path.join(plot_dir, f'{symbol}_{model_name}_final_price_scatter.png'))
        plt.close(fig)
        print("    - Final price scatter plot saved.")

        # --- 3. Cumulative Returns Comparison ---
        print("    - Generating cumulative returns plot...")
        actual_returns = (targets[:, 0] - last_known_prices) / (last_known_prices + 1e-9)
        pred_returns = (predictions[:, 0] - last_known_prices) / (last_known_prices + 1e-9)
        pred_signals = (pred_returns > 0).astype(int)
        strategy_returns = pred_signals * actual_returns
        
        fig = plt.figure(figsize=(15, 7))
        plt.plot(np.cumsum(actual_returns), label='Buy and Hold Cumulative Returns', color='royalblue')
        plt.plot(np.cumsum(strategy_returns), label=f'{model_name} Strategy Cumulative Returns', color='darkorange')
        plt.title(f'{symbol} - {model_name}: Cumulative Returns Comparison (1-Day Horizon)')
        plt.xlabel('Time (Samples)')
        plt.ylabel('Cumulative Return')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(plot_dir, f'{symbol}_{model_name}_cumulative_returns.png'))
        plt.close(fig)
        print("    - Cumulative returns plot saved.")

        # --- 4. Error Distribution Plot ---
        print("    - Generating error distribution plot...")
        errors = predictions - targets
        try:
            fig, ax = plt.subplots(figsize=(10, 6))
            print("      - Figure created. Building histogram bars...")
            values = errors.flatten()
            counts, bin_edges = np.histogram(values, bins=50, density=True)
            bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
            ax.bar(bin_centers, counts, width=(bin_edges[1] - bin_edges[0]), alpha=0.7, color='blue')
            ax.set_title(f'{symbol} - {model_name}: Distribution of Prediction Errors')
            ax.set_xlabel('Prediction Error (Predicted - Actual)')
            ax.set_ylabel('Density')
            ax.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join(plot_dir, f'{symbol}_{model_name}_error_distribution.png'))
            plt.close(fig)
            print("    - Error distribution plot saved.")
        except Exception as e_hist:
            print(f"    - Error distribution plot skipped due to error: {e_hist}")
            if 'fig' in locals() and fig is not None:
                plt.close(fig)

        # --- 5. Detailed Residual Analysis ---
        print("    - Generating residual analysis plot...")
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        sns.scatterplot(x=predictions.flatten(), y=errors.flatten(), alpha=0.5, ax=axes[0])
        axes[0].axhline(0, color='red', linestyle='--')
        axes[0].set_title('Residuals vs. Predicted Values')
        axes[0].set_xlabel('Predicted Price')
        axes[0].set_ylabel('Residuals')
        stats.probplot(errors.flatten(), dist="norm", plot=axes[1])
        axes[1].set_title('Q-Q Plot of Residuals')
        fig.suptitle(f'{symbol} - {model_name}: Residual Analysis')
        plt.tight_layout(rect=(0, 0.03, 1, 0.95))
        plt.savefig(os.path.join(plot_dir, f'{symbol}_{model_name}_residual_analysis.png'))
        plt.close(fig)
        print("    - Residual analysis plot saved.")
        
    except Exception as e:
        print(f"An error occurred during plotting for {model_name}: {e}")
        if fig is not None and plt.fignum_exists(fig.number):
            plt.close(fig)

def plot_all_model_comparison(all_model_results: Dict[str, Dict], plot_dir: str, symbol: str):
    """Generates and saves a comparison plot for all models."""
    os.makedirs(plot_dir, exist_ok=True)
    model_names = list(all_model_results.keys())
    num_models = len(model_names)
    
    if num_models == 0:
        print("No model results to compare.")
        return

    fig = None  # Initialize fig to None to avoid unbound error
    try:
        predict_len = all_model_results[model_names[0]]['predictions'].shape[1]

        fig, axes = plt.subplots(2, 2, figsize=(18, 15))
        fig.suptitle(f'{symbol} - All Models Comparison', fontsize=16)

        # --- 1. Predicted vs Actual Returns ---
        ax = axes[0, 0]
        for model_name, results in all_model_results.items():
            actual_returns = (results['targets'][:, -1] - results['last_known_prices']) / results['last_known_prices']
            pred_returns = (results['predictions'][:, -1] - results['last_known_prices']) / results['last_known_prices']
            sns.kdeplot(x=actual_returns, y=pred_returns, ax=ax, label=model_name, fill=True, alpha=0.2)
        ax.set_title('Predicted vs. Actual Returns (Final Horizon)')
        ax.set_xlabel('Actual Returns')
        ax.set_ylabel('Predicted Returns')
        ax.legend()

        # --- 2. Error over Horizon (MAE) ---
        ax = axes[0, 1]
        for model_name, results in all_model_results.items():
            errors = np.abs(results['predictions'] - results['targets'])
            mae_over_horizon = np.mean(errors, axis=0)
            ax.plot(np.arange(1, predict_len + 1), mae_over_horizon, marker='o', linestyle='-', label=model_name)
        ax.set_title('Mean Absolute Error (MAE) over Horizon')
        ax.set_xlabel('Prediction Horizon (Steps)')
        ax.set_ylabel('MAE')
        ax.legend()
        ax.grid(True)

        # --- 3. Residuals Distribution ---
        ax = axes[1, 0]
        for model_name, results in all_model_results.items():
            errors = (results['predictions'] - results['targets']).flatten()
            sns.kdeplot(errors, ax=ax, label=model_name, fill=True, alpha=0.2)
        ax.set_title('Distribution of Residuals (All Horizons)')
        ax.set_xlabel('Error (Predicted - Actual)')
        ax.legend()

        # --- 4. Directional Accuracy ---
        ax = axes[1, 1]
        accuracies = []
        for model_name, results in all_model_results.items():
            actual_direction = (results['targets'] > np.roll(results['targets'], 1, axis=1))[:, 1:]
            pred_direction = (results['predictions'] > np.roll(results['targets'], 1, axis=1))[:, 1:]
            accuracy = np.mean(actual_direction == pred_direction)
            accuracies.append(accuracy)
        
        sns.barplot(x=model_names, y=accuracies, ax=ax)
        ax.set_title('Overall Directional Accuracy')
        ax.set_ylabel('Accuracy Score')
        ax.tick_params(axis='x', rotation=45)
        ax.set_ylim(0, 1)

        plt.tight_layout(rect=(0, 0.03, 1, 0.95))
        plt.savefig(os.path.join(plot_dir, f'{symbol}_all_models_comparison.png'))
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
        'predict_len': 30,
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

        # --- Data Loading ---
        print("\n--- Loading Data ---")
        train_loader = get_data_loader(
            symbols=[config['symbol']], start=config['start_date'], end=config['end_date'],
            encoder_len=config['encoder_len'], predict_len=config['predict_len'], batch_size=config['batch_size']
        )
        validation_loader = get_data_loader(
            symbols=[config['symbol']], start=config['val_start_date'], end=config['val_end_date'],
            encoder_len=config['encoder_len'], predict_len=config['predict_len'], batch_size=config['batch_size']
        )

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

        for model_name, model_class in models_to_run.items():
            print(f"\n--- Running pipeline for {model_name} ---")
            model = model_class(config).to(device)
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
                predictions, targets, _ = validate_model(model, validation_loader, criterion, device)
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
            predictions, targets, last_known_prices = validate_model(model, validation_loader, criterion, device)
            
            if predictions is not None:
                all_model_results[model_name] = {
                    "predictions": predictions, "targets": targets, "last_known_prices": last_known_prices
                }
                # Count parameters
                num_params = sum(p.numel() for p in model.parameters())
                # Compute metrics
                preds = np.asarray(predictions)
                trues = np.asarray(targets)
                # Reconstruct absolute predictions from model outputs and last known price
                preds = np.asarray(predictions)
                trues = np.asarray(targets)
                last_prices = np.asarray(last_known_prices)
                # Model outputs are absolute future prices or deltas? Add last known price to get absolute
                preds = preds + last_prices[:, None]
                # Debug: print min/max of predictions, targets, and last known prices
                print(f"[DEBUG] preds range: {preds.min():.4f}-{preds.max():.4f}, trues range: {trues.min():.4f}-{trues.max():.4f}, last_prices range: {last_prices.min():.4f}-{last_prices.max():.4f}")
                # Compute naive baseline predictions (constant last known price)
                baseline_preds = np.tile(last_prices[:, None], (1, preds.shape[1]))
                # Flatten for metrics
                baseline_flat = baseline_preds.flatten()
                true_flat = trues.flatten()
                # Baseline metrics
                baseline_mse = mean_squared_error(true_flat, baseline_flat)
                try:
                    baseline_r2 = r2_score(true_flat, baseline_flat)
                except Exception:
                    baseline_r2 = float('nan')
                print(f"[DEBUG] Baseline MSE: {baseline_mse:.4f}, Baseline R2: {baseline_r2:.4f}")

                # Compute per-horizon MSE and MAE on absolute values
                mse = np.mean((preds - trues)**2, axis=0).tolist()
                mae = np.mean(np.abs(preds - trues), axis=0).tolist()

                # Compute per-horizon Pearson correlation for diagnostics
                corr_list = []
                for i in range(preds.shape[1]):
                    yt, yp = trues[:, i], preds[:, i]
                    try:
                        corr_list.append(float(np.corrcoef(yt, yp)[0,1]))
                    except Exception:
                        corr_list.append(float('nan'))
                # Single R2 over all absolute predictions
                trues_flat = trues.flatten()
                preds_flat = preds.flatten()
                # Compute unified R² across all horizons using variance-weighted multioutput
                try:
                    r2_flat = r2_score(trues, preds, multioutput='variance_weighted')
                except Exception:
                    r2_flat = float('nan')

                avg_mse = float(np.mean(mse))
                avg_mae = float(np.mean(mae))
                metrics_stats[model_name] = {
                    'train_loss': train_loss,
                    'val_mse': avg_mse,
                    'mse_over_time': mse,
                    'val_mae': avg_mae,
                    'mae_over_time': mae,
                    'r2': r2_flat,
                    'corr_over_time': corr_list,
                    'num_params': num_params
                }
                print(f"📊 {model_name} stats -> params: {num_params}, avg_mse: {avg_mse:.4f}, avg_mae: {avg_mae:.4f}, r2: {r2_flat:.4f}")
                # Save model checkpoint
                checkpoint_path = os.path.join(run_dir, f'{model_name}_checkpoint.pt')
                torch.save(model.state_dict(), checkpoint_path)
                print(f"💾 Checkpoint saved to {checkpoint_path}")
                print(f"✅ {model_name} pipeline finished successfully.")
            else:
                print(f"⚠️ Skipping {model_name} due to validation returning no results.")

        # --- Summary Table ---
        print("\nModel Performance Summary:")
        header = f"{'Model':<20}{'Train Loss':<15}{'Val MSE':<15}{'Val MAE':<15}{'Avg R2':<10}"
        print(header)
        print('-' * len(header))
        for m, stats_ in metrics_stats.items():
            # Safely fetch metrics with fallback for legacy keys
            train_loss = stats_.get('train_loss', float('nan'))
            val_mse = stats_.get('val_mse', stats_.get('mse', float('nan')))
            val_mae = stats_.get('val_mae', stats_.get('mae', float('nan')))
            avg_r2 = stats_.get('avg_r2', stats_.get('r2', float('nan')))
            print(f"{m:<20}{train_loss:<15.4f}{val_mse:<15.4f}{val_mae:<15.4f}{avg_r2:<10.4f}")

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
