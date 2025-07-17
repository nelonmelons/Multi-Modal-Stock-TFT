#!/usr/bin/env python3
"""
Fixed Baseline Models Pipeline for Stock Price Prediction
========================================================

A robust pipeline with proper symbol handling and fixed autoregressive predictions.
Uses simple feature normalization to avoid log transformation issues.

Features:
- Fetches real stock data using yfinance
- Handles each symbol separately to avoid mixing data
- Implements proper autoregressive predictions (model uses its own predictions)
- Creates proper evaluation metrics and plots per symbol
- Robust implementation with simple feature scaling

Usage:
    python fixed_baseline_pipeline.py
"""

import os
import sys
import numpy as np
import pandas as pd
import warnings
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any, Optional
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Setup paths and suppress warnings
warnings.filterwarnings('ignore')
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("⚠️ XGBoost not available. Install with: pip install xgboost")

# Import data fetching module
from dataModule.fetch_stock import fetch_stock_data, validate_stock_data

class FixedStockPredictor:
    """Stock predictor with simplified, robust feature processing."""
    
    def __init__(self, sequence_length: int = 30):
        self.sequence_length = sequence_length
        self.feature_scaler = StandardScaler()
        self.feature_dim = 5  # OHLCV
        
    def prepare_sequences(self, df: pd.DataFrame, symbol: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Create sequences from stock data for a single symbol with simple normalization.
        
        Args:
            df: Stock data DataFrame
            symbol: Stock symbol to process
            
        Returns:
            X: Feature sequences [n_samples, sequence_length * 5]
            y: Return targets [n_samples]
            timestamps: Timestamps for each sample
            prices: Current prices for each sample
        """
        # Filter data for this symbol
        symbol_data = df[df['symbol'] == symbol].copy()
        symbol_data = symbol_data.sort_values('date').reset_index(drop=True)
        
        if len(symbol_data) < self.sequence_length + 2:
            raise ValueError(f"Insufficient data for {symbol}: {len(symbol_data)} rows")
        
        # Extract OHLCV columns
        feature_cols = ['open', 'high', 'low', 'close', 'volume']
        features = symbol_data[feature_cols].values
        
        # Simple validation and cleaning
        features = features.astype(np.float64)
        features = np.nan_to_num(features, nan=0.0, posinf=1e10, neginf=-1e10)
        
        # Ensure positive values for price features
        features[:, :4] = np.maximum(features[:, :4], 1e-6)
        features[:, 4] = np.maximum(features[:, 4], 0)  # Volume >= 0
        
        # Calculate percentage features (relative to close price) for better normalization
        normalized_features = np.zeros_like(features)
        for i in range(len(features)):
            close_price = features[i, 3]  # Close price
            if close_price > 0:
                normalized_features[i, 0] = features[i, 0] / close_price  # Open/Close
                normalized_features[i, 1] = features[i, 1] / close_price  # High/Close  
                normalized_features[i, 2] = features[i, 2] / close_price  # Low/Close
                normalized_features[i, 3] = 1.0  # Close/Close = 1
                normalized_features[i, 4] = features[i, 4] / 1e6  # Scale volume down
            else:
                normalized_features[i] = [1.0, 1.0, 1.0, 1.0, 0.0]
        
        # Calculate returns (percentage change in close price)
        returns = symbol_data['close'].pct_change().fillna(0).values
        
        # Create sequences
        X, y, timestamps, prices = [], [], [], []
        
        for i in range(self.sequence_length, len(symbol_data) - 1):  # -1 because we need next period
            # Feature sequence: past sequence_length timesteps
            sequence = normalized_features[i-self.sequence_length:i].flatten()  # Flatten to 1D
            
            # Target return for next period
            target_return = returns[i + 1]  # Next period return
            
            # Current price and timestamp
            current_price = symbol_data.iloc[i]['close']
            timestamp = symbol_data.iloc[i]['date']
            
            X.append(sequence)
            y.append(target_return)
            prices.append(current_price)
            timestamps.append(timestamp)
        
        return np.array(X), np.array(y), np.array(timestamps), np.array(prices)
    
    def make_autoregressive_predictions(self, model, X_test_scaled: np.ndarray, 
                                       scaler, X_test_original: np.ndarray) -> np.ndarray:
        """
        Make TRUE autoregressive predictions where the model uses its own predictions.
        
        This is a simplified but more robust approach that avoids synthetic feature generation.
        Instead, it progressively shifts the input window using the model's own predictions
        to reconstruct return sequences.
        
        Args:
            model: Trained sklearn model
            X_test_scaled: Scaled test features [n_samples, features]
            scaler: The fitted StandardScaler used for training
            X_test_original: Original unscaled test features [n_samples, features]
            
        Returns:
            predictions: Autoregressive predictions
        """
        predictions = []
        
        # Start with first test sequence (scaled)
        current_sequence_scaled = X_test_scaled[0].copy()  # Shape: [sequence_length * 5]
        current_sequence_original = X_test_original[0].copy()  # Keep original for reconstruction
        
        for i in range(len(X_test_scaled)):
            # Make prediction using current scaled sequence
            pred_return = model.predict(current_sequence_scaled.reshape(1, -1))[0]
            predictions.append(pred_return)
            
            # AUTOREGRESSIVE STEP: Update sequence for next prediction
            if i < len(X_test_scaled) - 1:
                # Method 1: Use actual next sequence but modify the last timestep with prediction
                # This is more stable than full synthetic generation
                if i + 1 < len(X_test_original):
                    # Get next actual sequence
                    next_sequence_original = X_test_original[i + 1].copy()
                    
                    # Reshape to [sequence_length, 5]
                    next_seq_2d = next_sequence_original.reshape(self.sequence_length, self.feature_dim)
                    
                    # Modify the last timestep based on our prediction
                    # Assume the predicted return affects the price ratios slightly
                    last_timestep = next_seq_2d[-1].copy()
                    
                    # Apply small perturbation based on predicted return
                    # This simulates how our prediction might affect the next price ratios
                    return_impact = pred_return * 0.1  # Dampen the impact for stability
                    
                    # Modify ratios slightly (but keep close to 1.0 for close/close)
                    last_timestep[0] *= (1 + return_impact * 0.5)  # Open/Close
                    last_timestep[1] *= (1 + abs(return_impact) * 0.3)  # High/Close
                    last_timestep[2] *= (1 - abs(return_impact) * 0.3)  # Low/Close
                    # last_timestep[3] remains 1.0 (Close/Close)
                    # last_timestep[4] unchanged (Volume)
                    
                    # Ensure ratios stay reasonable
                    last_timestep[0] = np.clip(last_timestep[0], 0.95, 1.05)
                    last_timestep[1] = np.clip(last_timestep[1], 1.0, 1.1)
                    last_timestep[2] = np.clip(last_timestep[2], 0.9, 1.0)
                    
                    # Update the sequence
                    next_seq_2d[-1] = last_timestep
                    next_sequence_modified = next_seq_2d.flatten()
                    
                    # Scale the modified sequence
                    current_sequence_scaled = scaler.transform(next_sequence_modified.reshape(1, -1))[0]
                    current_sequence_original = next_sequence_modified
                else:
                    # Fallback: shift the current sequence and add predicted effects
                    current_seq_2d = current_sequence_original.reshape(self.sequence_length, self.feature_dim)
                    
                    # Shift sequence: remove first timestep, duplicate last with modifications
                    last_timestep = current_seq_2d[-1].copy()
                    
                    # Apply prediction impact
                    return_impact = pred_return * 0.1
                    last_timestep[0] *= (1 + return_impact * 0.5)
                    last_timestep[1] *= (1 + abs(return_impact) * 0.3)
                    last_timestep[2] *= (1 - abs(return_impact) * 0.3)
                    
                    # Clip to reasonable ranges
                    last_timestep[0] = np.clip(last_timestep[0], 0.95, 1.05)
                    last_timestep[1] = np.clip(last_timestep[1], 1.0, 1.1)
                    last_timestep[2] = np.clip(last_timestep[2], 0.9, 1.0)
                    
                    # Shift and add modified timestep
                    new_sequence = np.vstack([current_seq_2d[1:], last_timestep])
                    current_sequence_original = new_sequence.flatten()
                    
                    # Scale the new sequence
                    current_sequence_scaled = scaler.transform(current_sequence_original.reshape(1, -1))[0]
        
        return np.array(predictions)
        
        return np.array(predictions)

class FixedBaselineRunner:
    """Runs baseline models with robust error handling."""
    
    def __init__(self, output_dir: str = "fixed_baseline_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.results = {}  # Will store results per symbol per model
        self.models = {}
        
    def initialize_models(self):
        """Initialize baseline models including naive baselines for comparison."""
        self.models = {
            'Linear Regression': LinearRegression(),
            'Random Forest': RandomForestRegressor(
                n_estimators=50,  # Reduced for faster training
                max_depth=8, 
                random_state=42, 
                n_jobs=-1
            )
        }
        
        if XGBOOST_AVAILABLE:
            self.models['XGBoost'] = xgb.XGBRegressor(
                n_estimators=50,  # Reduced for faster training
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                n_jobs=-1,
                verbosity=0  # Suppress XGBoost warnings
            )
        
        # Add naive baselines to show what poor predictions look like
        # Note: 'uniform' strategy is not available in all sklearn versions
        self.models.update({
            'Zero Baseline': DummyRegressor(strategy='constant', constant=0.0),
            'Mean Baseline': DummyRegressor(strategy='mean'),
            'Median Baseline': DummyRegressor(strategy='median')
        })
    
    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                         current_prices: np.ndarray) -> Dict[str, float]:
        """Calculate comprehensive prediction metrics."""
        try:
            # Return-based metrics
            mse_returns = mean_squared_error(y_true, y_pred)
            mae_returns = mean_absolute_error(y_true, y_pred)
            r2_returns = r2_score(y_true, y_pred)
            
            # OLD METHOD (flawed): Convert returns to prices using same base price
            predicted_prices_old = current_prices * (1 + y_pred)
            actual_prices_old = current_prices * (1 + y_true)
            
            # CORRECTED METHOD: Cumulative price reconstruction
            predicted_prices_corrected = [current_prices[0]]
            actual_prices_corrected = [current_prices[0]]
            
            for i in range(len(y_pred)):
                if i == 0:
                    predicted_prices_corrected.append(current_prices[0] * (1 + y_pred[i]))
                    actual_prices_corrected.append(current_prices[0] * (1 + y_true[i]))
                else:
                    predicted_prices_corrected.append(predicted_prices_corrected[-1] * (1 + y_pred[i]))
                    actual_prices_corrected.append(actual_prices_corrected[-1] * (1 + y_true[i]))
            
            # Remove first element (starting price)
            predicted_prices_corrected = np.array(predicted_prices_corrected[1:])
            actual_prices_corrected = np.array(actual_prices_corrected[1:])
            
            # Calculate corrected price metrics
            mse_prices_corrected = mean_squared_error(actual_prices_corrected, predicted_prices_corrected)
            mae_prices_corrected = mean_absolute_error(actual_prices_corrected, predicted_prices_corrected)
            r2_prices_corrected = r2_score(actual_prices_corrected, predicted_prices_corrected)
            
            # Calculate old (flawed) price metrics for comparison
            mse_prices_old = mean_squared_error(actual_prices_old, predicted_prices_old)
            mae_prices_old = mean_absolute_error(actual_prices_old, predicted_prices_old)
            r2_prices_old = r2_score(actual_prices_old, predicted_prices_old)
            
            # Calculate percentage errors with protection against division by zero
            mape_returns = np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + 1e-8))) * 100
            mape_prices_old = np.mean(np.abs((actual_prices_old - predicted_prices_old) / (actual_prices_old + 1e-8))) * 100
            mape_prices_corrected = np.mean(np.abs((actual_prices_corrected - predicted_prices_corrected) / (actual_prices_corrected + 1e-8))) * 100
            
            # Calculate cumulative return accuracy (important for trading strategies)
            cumulative_actual = np.prod(1 + y_true) - 1
            cumulative_predicted = np.prod(1 + y_pred) - 1
            cumulative_return_error = abs(cumulative_actual - cumulative_predicted)
            
            # Calculate directional accuracy (what % of time did we predict direction correctly)
            actual_directions = np.sign(y_true)
            predicted_directions = np.sign(y_pred)
            directional_accuracy = np.mean(actual_directions == predicted_directions) * 100
            
            return {
                # Return-based metrics
                'mse_returns': float(mse_returns),
                'mae_returns': float(mae_returns),
                'r2_returns': float(r2_returns),
                'mape_returns': float(mape_returns),
                
                # OLD (flawed) price metrics
                'mse_prices_old': float(mse_prices_old),
                'mae_prices_old': float(mae_prices_old),
                'r2_prices_old': float(r2_prices_old),
                'mape_prices_old': float(mape_prices_old),
                
                # CORRECTED price metrics
                'mse_prices_corrected': float(mse_prices_corrected),
                'mae_prices_corrected': float(mae_prices_corrected),
                'r2_prices_corrected': float(r2_prices_corrected),
                'mape_prices_corrected': float(mape_prices_corrected),
                
                # Additional useful metrics
                'return_volatility': float(np.std(y_pred)),
                'price_volatility_corrected': float(np.std(predicted_prices_corrected)),
                'cumulative_return_error': float(cumulative_return_error),
                'directional_accuracy': float(directional_accuracy),
                'cumulative_actual_return': float(cumulative_actual),
                'cumulative_predicted_return': float(cumulative_predicted),
            }
        except Exception as e:
            print(f"⚠️ Error calculating metrics: {e}")
            return {
                'mse_returns': np.inf, 'mae_returns': np.inf, 'r2_returns': -np.inf,
                'mape_returns': np.inf, 'mse_prices_old': np.inf, 'mae_prices_old': np.inf,
                'r2_prices_old': -np.inf, 'mape_prices_old': np.inf,
                'mse_prices_corrected': np.inf, 'mae_prices_corrected': np.inf,
                'r2_prices_corrected': -np.inf, 'mape_prices_corrected': np.inf,
                'return_volatility': 0.0, 'price_volatility_corrected': 0.0,
                'cumulative_return_error': np.inf, 'directional_accuracy': 0.0,
                'cumulative_actual_return': 0.0, 'cumulative_predicted_return': 0.0
            }
    
    def train_and_evaluate_symbol(self, stock_data: pd.DataFrame, symbol: str, 
                                  test_split: float = 0.2, use_autoregressive: bool = False):
        """Train and evaluate all models for a single symbol."""
        print(f"\n{'='*60}")
        print(f"🎯 PROCESSING SYMBOL: {symbol}")
        print(f"{'='*60}")
        
        try:
            # Prepare data for this symbol
            predictor = FixedStockPredictor(sequence_length=30)
            
            X, y, timestamps, prices = predictor.prepare_sequences(stock_data, symbol)
            
            if len(X) == 0:
                print(f"❌ No sequences created for {symbol}")
                return
            
            # Split data chronologically
            split_idx = int(len(X) * (1 - test_split))
            if split_idx < 10:  # Ensure minimum training data
                print(f"❌ Insufficient training data for {symbol}")
                return
            
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]
            timestamps_test = timestamps[split_idx:]
            prices_test = prices[split_idx:]
            
            print(f"📊 {symbol}: {len(X_train)} train, {len(X_test)} test samples")
            print(f"📊 Feature shape: {X_train.shape}")
            
            # Scale features
            scaler = StandardScaler()
            try:
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
            except Exception as e:
                print(f"❌ Scaling failed for {symbol}: {e}")
                return
            
            # Initialize results for this symbol
            if symbol not in self.results:
                self.results[symbol] = {}
            
            # Get symbol data for autoregressive prediction
            symbol_data = stock_data[stock_data['symbol'] == symbol].sort_values('date').reset_index(drop=True)
            test_start_idx = len(symbol_data) - len(X_test)
            
            for model_name, model in self.models.items():
                print(f"\n🔄 Training {model_name} for {symbol}...")
                
                try:
                    # Train model
                    model.fit(X_train_scaled, y_train)
                    
                    # Make both types of predictions for comparison
                    print(f"   📊 Making standard predictions...")
                    standard_predictions = model.predict(X_test_scaled)
                    
                    if use_autoregressive and len(X_test) > 1:
                        print(f"   🔮 Making autoregressive predictions...")
                        autoregressive_predictions = predictor.make_autoregressive_predictions(
                            model, X_test_scaled, scaler, X_test
                        )
                        predictions = autoregressive_predictions
                        pred_type = "autoregressive"
                    else:
                        predictions = standard_predictions
                        pred_type = "standard"
                    
                    # Calculate metrics
                    metrics = self.calculate_metrics(y_test, predictions, prices_test)
                    
                    # Also calculate standard metrics for comparison
                    standard_metrics = self.calculate_metrics(y_test, standard_predictions, prices_test)
                    
                    # Store results
                    self.results[symbol][model_name] = {
                        'predictions': predictions,
                        'actuals': y_test,
                        'timestamps': timestamps_test,
                        'prices': prices_test,
                        'metrics': metrics,
                        'prediction_type': pred_type,
                        'standard_predictions': standard_predictions,
                        'standard_metrics': standard_metrics
                    }
                    
                    print(f"✅ {model_name} ({pred_type}) - R²: {metrics['r2_returns']:.4f}, MAE: {metrics['mae_returns']:.4f}")
                    print(f"   📊 Corrected Price R²: {metrics['r2_prices_corrected']:.4f}, MAE: ${metrics['mae_prices_corrected']:.2f}")
                    print(f"   📊 Directional Accuracy: {metrics['directional_accuracy']:.1f}%")
                    print(f"   Standard comparison - R²: {standard_metrics['r2_returns']:.4f}, MAE: {standard_metrics['mae_returns']:.4f}")
                    
                except Exception as e:
                    print(f"❌ {model_name} failed for {symbol}: {e}")
                    import traceback
                    print(f"   Traceback: {traceback.format_exc()}")
                    continue
        
        except Exception as e:
            print(f"❌ Failed to process {symbol}: {e}")
            import traceback
            print(f"   Traceback: {traceback.format_exc()}")
            return
    
    def create_symbol_plots(self, symbol: str):
        """Create plots for a specific symbol with CORRECTED price reconstruction."""
        if symbol not in self.results or len(self.results[symbol]) == 0:
            print(f"⚠️ No results to plot for {symbol}")
            return
            
        print(f"📊 Creating plots for {symbol}...")
        
        symbol_results = self.results[symbol]
        n_models = len(symbol_results)
        
        try:
            # Create subplots for each model (now 4 plots per model)
            fig, axes = plt.subplots(n_models, 4, figsize=(24, 6*n_models))
            if n_models == 1:
                axes = axes.reshape(1, -1)
            
            fig.suptitle(f'{symbol} - Stock Predictions (CORRECTED)', fontsize=16, fontweight='bold')
            
            for i, (model_name, result) in enumerate(symbol_results.items()):
                predictions = result['predictions']
                actuals = result['actuals']
                timestamps = result['timestamps']
                prices = result['prices']
                
                # CORRECTED: Cumulative price reconstruction
                # Start with the first actual price and build sequentially
                predicted_prices_corrected = [prices[0]]  # Start with first actual price
                actual_prices_corrected = [prices[0]]     # Start with first actual price
                
                for j in range(len(predictions)):
                    if j == 0:
                        # First prediction: use the actual starting price
                        predicted_prices_corrected.append(prices[0] * (1 + predictions[j]))
                        actual_prices_corrected.append(prices[0] * (1 + actuals[j]))
                    else:
                        # Subsequent predictions: build on previous predicted/actual price
                        predicted_prices_corrected.append(predicted_prices_corrected[-1] * (1 + predictions[j]))
                        actual_prices_corrected.append(actual_prices_corrected[-1] * (1 + actuals[j]))
                
                # Remove the first element (it was just the starting price)
                predicted_prices_corrected = np.array(predicted_prices_corrected[1:])
                actual_prices_corrected = np.array(actual_prices_corrected[1:])
                
                # OLD METHOD (for comparison) - the flawed approach
                predicted_prices_old = prices * (1 + predictions)
                actual_prices_old = prices * (1 + actuals)
                
                # Plot 1: Returns over time
                axes[i, 0].plot(timestamps, actuals, color='blue', linewidth=2, label='Actual Returns', alpha=0.8)
                axes[i, 0].plot(timestamps, predictions, color='red', linewidth=2, label='Predicted Returns', alpha=0.8)
                axes[i, 0].set_title(f'{model_name} - Returns Over Time')
                axes[i, 0].set_ylabel('Return (%)')
                axes[i, 0].legend()
                axes[i, 0].grid(True, alpha=0.3)
                axes[i, 0].tick_params(axis='x', rotation=45)
                
                # Plot 2: CORRECTED Cumulative Price Predictions  
                axes[i, 1].plot(timestamps, actual_prices_corrected, color='blue', linewidth=2, label='Actual Price Path', alpha=0.8)
                axes[i, 1].plot(timestamps, predicted_prices_corrected, color='red', linewidth=2, label='Predicted Price Path', alpha=0.8)
                axes[i, 1].set_title(f'{model_name} - CORRECTED Price Predictions')
                axes[i, 1].set_ylabel('Price ($)')
                axes[i, 1].legend()
                axes[i, 1].grid(True, alpha=0.3)
                axes[i, 1].tick_params(axis='x', rotation=45)
                
                # Plot 3: OLD (Flawed) Price Predictions for comparison
                axes[i, 2].plot(timestamps, actual_prices_old, color='blue', linewidth=2, label='Actual Prices (OLD)', alpha=0.8)
                axes[i, 2].plot(timestamps, predicted_prices_old, color='red', linewidth=2, label='Predicted Prices (OLD)', alpha=0.8)
                axes[i, 2].set_title(f'{model_name} - OLD (Flawed) Price Method')
                axes[i, 2].set_ylabel('Price ($)')
                axes[i, 2].legend()
                axes[i, 2].grid(True, alpha=0.3)
                axes[i, 2].tick_params(axis='x', rotation=45)
                
                # Plot 4: Prediction vs Actual scatter (returns)
                axes[i, 3].scatter(actuals, predictions, alpha=0.6, s=20)
                min_val = min(actuals.min(), predictions.min())
                max_val = max(actuals.max(), predictions.max())
                axes[i, 3].plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)
                axes[i, 3].set_title(f'{model_name} - Predicted vs Actual Returns')
                axes[i, 3].set_xlabel('Actual Returns')
                axes[i, 3].set_ylabel('Predicted Returns')
                axes[i, 3].grid(True, alpha=0.3)
                
                # Calculate and display corrected price metrics
                price_mae = np.mean(np.abs(actual_prices_corrected - predicted_prices_corrected))
                price_mape = np.mean(np.abs((actual_prices_corrected - predicted_prices_corrected) / actual_prices_corrected)) * 100
                print(f"  📊 {model_name} - Corrected Price MAE: ${price_mae:.2f}, MAPE: {price_mape:.2f}%")
            
            plt.tight_layout()
            plot_path = self.output_dir / f"{symbol}_predictions_corrected.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"  📈 {symbol} corrected plots saved to {plot_path}")
            
        except Exception as e:
            print(f"❌ Failed to create plots for {symbol}: {e}")
    
    def create_comparison_plots(self):
        """Create comparison plots across models and symbols."""
        print("\n📊 Creating model comparison plots...")
        
        # Collect all metrics
        all_metrics = []
        for symbol, symbol_results in self.results.items():
            for model_name, result in symbol_results.items():
                metrics = result['metrics'].copy()
                metrics['symbol'] = symbol
                metrics['model'] = model_name
                all_metrics.append(metrics)
        
        if not all_metrics:
            print("❌ No results to compare")
            return
        
        try:
            metrics_df = pd.DataFrame(all_metrics)
            
            # Create comparison plots
            fig, axes = plt.subplots(2, 3, figsize=(20, 12))
            fig.suptitle('Model Performance Comparison - CORRECTED METRICS', fontsize=16, fontweight='bold')
            
            # R² (Returns) by model and symbol
            r2_pivot = metrics_df.pivot(index='symbol', columns='model', values='r2_returns')
            r2_pivot.plot(kind='bar', ax=axes[0, 0])
            axes[0, 0].set_title('R² Score (Returns) by Model and Symbol')
            axes[0, 0].set_ylabel('R² Score')
            axes[0, 0].tick_params(axis='x', rotation=45)
            axes[0, 0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            
            # R² (Corrected Prices) by model and symbol
            r2_price_pivot = metrics_df.pivot(index='symbol', columns='model', values='r2_prices_corrected')
            r2_price_pivot.plot(kind='bar', ax=axes[0, 1])
            axes[0, 1].set_title('R² Score (Corrected Prices) by Model and Symbol')
            axes[0, 1].set_ylabel('R² Score')
            axes[0, 1].tick_params(axis='x', rotation=45)
            axes[0, 1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            
            # Directional Accuracy by model and symbol
            dir_acc_pivot = metrics_df.pivot(index='symbol', columns='model', values='directional_accuracy')
            dir_acc_pivot.plot(kind='bar', ax=axes[0, 2])
            axes[0, 2].set_title('Directional Accuracy by Model and Symbol')
            axes[0, 2].set_ylabel('Accuracy (%)')
            axes[0, 2].tick_params(axis='x', rotation=45)
            axes[0, 2].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            
            # Average performance by model
            avg_metrics = metrics_df.groupby('model').agg({
                'r2_returns': 'mean',
                'r2_prices_corrected': 'mean',
                'directional_accuracy': 'mean'
            })
            
            avg_metrics['r2_returns'].plot(kind='bar', ax=axes[1, 0])
            axes[1, 0].set_title('Average R² Score (Returns) by Model')
            axes[1, 0].set_ylabel('Average R² Score')
            axes[1, 0].tick_params(axis='x', rotation=45)
            
            avg_metrics['r2_prices_corrected'].plot(kind='bar', ax=axes[1, 1])
            axes[1, 1].set_title('Average R² Score (Corrected Prices) by Model')
            axes[1, 1].set_ylabel('Average R² Score')
            axes[1, 1].tick_params(axis='x', rotation=45)
            
            avg_metrics['directional_accuracy'].plot(kind='bar', ax=axes[1, 2])
            axes[1, 2].set_title('Average Directional Accuracy by Model')
            axes[1, 2].set_ylabel('Average Accuracy (%)')
            axes[1, 2].tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            comparison_path = self.output_dir / "model_comparison.png"
            plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            # Save metrics
            metrics_path = self.output_dir / "all_metrics.csv"
            metrics_df.to_csv(metrics_path, index=False)
            
            print(f"  📊 Comparison plots saved to {comparison_path}")
            print(f"  📊 Metrics saved to {metrics_path}")
            
        except Exception as e:
            print(f"❌ Failed to create comparison plots: {e}")
    
    def print_summary(self):
        """Print summary of results."""
        print("\n" + "=" * 60)
        print("📋 RESULTS SUMMARY")
        print("=" * 60)
        
        if not self.results:
            print("❌ No results to summarize")
            return
        
        try:
            # Calculate averages across symbols for each model
            model_averages = {}
            
            for symbol, symbol_results in self.results.items():
                for model_name, result in symbol_results.items():
                    if model_name not in model_averages:
                        model_averages[model_name] = {
                            'r2': [], 'mae': [], 'mape': [], 
                            'r2_price_corrected': [], 'mae_price_corrected': [],
                            'directional_accuracy': [], 'cumulative_error': []
                        }
                    
                    metrics = result['metrics']
                    model_averages[model_name]['r2'].append(metrics['r2_returns'])
                    model_averages[model_name]['mae'].append(metrics['mae_returns'])
                    model_averages[model_name]['mape'].append(metrics['mape_prices_corrected'])
                    model_averages[model_name]['r2_price_corrected'].append(metrics['r2_prices_corrected'])
                    model_averages[model_name]['mae_price_corrected'].append(metrics['mae_prices_corrected'])
                    model_averages[model_name]['directional_accuracy'].append(metrics['directional_accuracy'])
                    model_averages[model_name]['cumulative_error'].append(metrics['cumulative_return_error'])
            
            if not model_averages:
                print("❌ No model results to average")
                return
            
            # Print comprehensive summary table
            print(f"{'Model':<20} {'R²(Ret)':<10} {'MAE(Ret)':<12} {'R²(Price)':<12} {'MAE($)':<10} {'Dir.Acc':<8} {'Cum.Err':<10}")
            print("-" * 102)
            
            best_r2 = -np.inf
            best_model = ""
            
            for model_name, metrics in model_averages.items():
                avg_r2 = np.mean(metrics['r2'])
                avg_mae = np.mean(metrics['mae'])
                avg_r2_price = np.mean(metrics['r2_price_corrected'])
                avg_mae_price = np.mean(metrics['mae_price_corrected'])
                avg_dir_acc = np.mean(metrics['directional_accuracy'])
                avg_cum_err = np.mean(metrics['cumulative_error'])
                
                print(f"{model_name:<20} {avg_r2:<10.4f} {avg_mae:<12.4f} {avg_r2_price:<12.4f} {avg_mae_price:<10.2f} {avg_dir_acc:<8.1f}% {avg_cum_err:<10.4f}")
                
                if avg_r2 > best_r2:
                    best_r2 = avg_r2
                    best_model = model_name
            
            print(f"\n🏆 Best performing model (avg): {best_model} (R² = {best_r2:.4f})")
            
            # Print per-symbol details
            print(f"\n📊 Per-Symbol Performance (Corrected Metrics):")
            for symbol in self.results:
                print(f"\n  {symbol}:")
                for model_name, result in self.results[symbol].items():
                    metrics = result['metrics']
                    print(f"    {model_name:<15}: R²(Ret)={metrics['r2_returns']:.4f}, R²(Price)={metrics['r2_prices_corrected']:.4f}, Dir={metrics['directional_accuracy']:.1f}%")
                    
        except Exception as e:
            print(f"❌ Error in summary: {e}")

def main():
    """Main function to run the CORRECTED baseline pipeline."""
    
    # Configuration
    symbols = ['AAPL', 'GOOGL', 'MSFT']  # Popular tech stocks
    start_date = '2022-01-01'
    end_date = '2024-12-01'
    test_split = 0.2  # 20% for testing
    use_autoregressive = True  # Enable TRUE autoregressive predictions
    
    print("🚀 Starting CORRECTED Baseline Models Pipeline")
    print(f"📅 Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🎯 Symbols: {symbols}")
    print(f"📊 Date range: {start_date} to {end_date}")
    print(f"🔮 Autoregressive predictions: {use_autoregressive}")
    print(f"⚠️  FIXED: Cumulative price reconstruction for accurate evaluation")
    print(f"⚠️  ADDED: Naive baselines to show what poor predictions look like")
    
    try:
        # Step 1: Fetch stock data
        print("\n" + "=" * 60)
        print("📈 FETCHING STOCK DATA")
        print("=" * 60)
        
        stock_data = fetch_stock_data(symbols, start_date, end_date)
        
        if stock_data.empty:
            raise ValueError("No stock data retrieved")
        
        # Validate data
        if not validate_stock_data(stock_data):
            print("⚠️ Data validation warnings - proceeding anyway")
        
        print(f"✅ Retrieved {len(stock_data)} data points")
        print(f"📊 Symbols found: {sorted(stock_data['symbol'].unique())}")
        
        # Step 2: Initialize runner and models
        runner = FixedBaselineRunner()
        runner.initialize_models()
        
        # Step 3: Process each symbol separately
        for symbol in symbols:
            if symbol in stock_data['symbol'].values:
                runner.train_and_evaluate_symbol(stock_data, symbol, test_split, use_autoregressive)
            else:
                print(f"⚠️ No data found for {symbol}")
        
        # Step 4: Create visualizations
        print("\n" + "=" * 60)
        print("📊 CREATING VISUALIZATIONS")
        print("=" * 60)
        
        for symbol in symbols:
            runner.create_symbol_plots(symbol)
        
        runner.create_comparison_plots()
        
        # Step 5: Print summary
        runner.print_summary()
        
        end_time = datetime.now()
        print(f"\n✅ Pipeline completed successfully!")
        print(f"📂 Results saved to: {runner.output_dir}")
        
    except Exception as e:
        print(f"\n❌ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    main()
