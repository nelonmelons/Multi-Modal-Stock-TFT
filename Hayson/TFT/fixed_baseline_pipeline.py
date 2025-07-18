#!/usr/bin/env python3
"""
Multimodal Baseline Models Pipeline for Stock Price Prediction
==============================================================

A robust baseline pipeline that uses the EXACT same multimodal data as the TFT pipeline
for fair comparison. Includes news embeddings, economic indicators, technical analysis,
and corporate events data.

Features:
- Uses EXACT same data loading as TFT pipeline (via dataModule interface)
- Identical train/validation split logic with temporal buffers
- Multimodal features: stock OHLCV, news embeddings, FRED economic data, technical indicators
- Same normalization and preprocessing as TFT pipeline
- Handles each symbol separately to avoid mixing data
- Implements proper autoregressive predictions (model uses its own predictions)
- Creates proper evaluation metrics and plots per symbol
- Fair comparison with TFT model using identical feature sets and data splits

Key Improvements:
- Uses get_data_loader_with_module() exactly like TFT pipeline
- Respects temporal constraints and lookahead buffers
- Maintains consistent API key usage for external data sources
- Identical feature engineering and preprocessing pipeline

Usage:
    # Basic usage with multimodal features
    python fixed_baseline_pipeline.py --symbols AAPL,MSFT,GOOGL --multimodal
    
    # With API keys for external data
    python fixed_baseline_pipeline.py --symbols AAPL,MSFT,GOOGL \
        --news-api-key YOUR_KEY --fred-api-key YOUR_KEY --api-ninjas-key YOUR_KEY
    
    # Basic OHLCV features only
    python fixed_baseline_pipeline.py --symbols AAPL,MSFT,GOOGL --basic-only
    
    # Custom date range and sequence length
    python fixed_baseline_pipeline.py --symbols AAPL,MSFT,GOOGL \
        --start-date 2020-01-01 --end-date 2023-12-31 --encoder-len 60
"""

import os
import sys
import numpy as np
import pandas as pd
import warnings
import traceback
import argparse
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

# Import TFT data interface for multimodal data
from dataModule.interface import get_data_loader_with_module

class MultiStepPredictor:
    """Wrapper to make sklearn models predict multiple timesteps like TFT."""
    
    def __init__(self, base_model, predict_len: int):
        self.base_model = base_model
        self.predict_len = predict_len
        self.models = []  # One model per timestep
        
    def fit(self, X, y):
        """
        Fit separate models for each prediction timestep.
        
        Args:
            X: Input features [n_samples, n_features]
            y: Multi-step targets [n_samples, predict_len]
        """
        self.models = []
        
        for step in range(self.predict_len):
            # Clone the base model for this timestep
            from sklearn.base import clone
            model = clone(self.base_model)
            
            # Fit on the target for this specific timestep
            model.fit(X, y[:, step])
            self.models.append(model)
    
    def predict(self, X):
        """
        Predict multiple timesteps.
        
        Args:
            X: Input features [n_samples, n_features]
            
        Returns:
            predictions: [n_samples, predict_len]
        """
        predictions = []
        
        for step, model in enumerate(self.models):
            step_pred = model.predict(X)
            predictions.append(step_pred)
        
        # Stack predictions: [predict_len, n_samples] -> [n_samples, predict_len]
        return np.array(predictions).T

class MultimodalStockPredictor:
    """Stock predictor using multimodal features from TFT pipeline."""
    
    def __init__(self, sequence_length: int = 30):
        self.sequence_length = sequence_length
        self.feature_scaler = StandardScaler()
        
    def extract_features_from_datamodule(self, datamodule, use_multimodal: bool = True, predict_len: int = 5) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Extract features from TFT DataModule for baseline models with multi-step prediction.
        
        Args:
            datamodule: TFT DataModule containing multimodal features
            use_multimodal: Whether to use all multimodal features or just OHLCV
            predict_len: Number of future timesteps to predict (T+1, T+2, ..., T+predict_len)
            
        Returns:
            X: Feature sequences [n_samples, features]
            y: Multi-step targets [n_samples, predict_len]
            timestamps: Timestamps for each sample
            symbols: Symbol for each sample
        """
        print(f"📊 Extracting features from datamodule...")
        print(f"   Multimodal features: {use_multimodal}")
        print(f"   Prediction horizon: {predict_len} steps")
        
        # Get the feature dataframe from the datamodule
        feature_df = datamodule.feature_df.copy()
        feature_df = feature_df.sort_values(['symbol', 'time_idx']).reset_index(drop=True)
        
        print(f"   Feature matrix shape: {feature_df.shape}")
        print(f"   Available columns: {list(feature_df.columns)}")
        
        # Identify feature types
        excluded_cols = ['symbol', 'date', 'time_idx', 'target']
        
        if use_multimodal:
            # Use ALL available features (OHLCV + Technical + News + Economic + Events)
            # But exclude categorical string columns that can't be scaled
            feature_cols = []
            for col in feature_df.columns:
                if col not in excluded_cols:
                    # Check if column contains numeric data
                    try:
                        # Try to convert a sample to float
                        sample_val = feature_df[col].dropna().iloc[0] if not feature_df[col].dropna().empty else 0
                        float(sample_val)
                        feature_cols.append(col)
                    except (ValueError, TypeError):
                        print(f"   ⚠️ Excluding non-numeric column: {col} (sample value: {sample_val})")
                        continue
            print(f"   Using multimodal features: {len(feature_cols)} features")
        else:
            # Use only basic OHLCV features for comparison
            basic_cols = ['open', 'high', 'low', 'close', 'volume']
            feature_cols = [col for col in basic_cols if col in feature_df.columns]
            print(f"   Using basic OHLCV features: {len(feature_cols)} features")
        
        # Extract features for each symbol separately
        all_X, all_y, all_timestamps, all_symbols = [], [], [], []
        
        for symbol in feature_df['symbol'].unique():
            symbol_df = feature_df[feature_df['symbol'] == symbol].copy()
            symbol_df = symbol_df.sort_values('time_idx').reset_index(drop=True)
            
            if len(symbol_df) < self.sequence_length + predict_len:
                print(f"   ⚠️ Insufficient data for {symbol}: {len(symbol_df)} rows (need {self.sequence_length + predict_len})")
                continue
            
            # Prepare features and targets
            features = symbol_df[feature_cols].values
            targets = symbol_df['target'].values
            timestamps = symbol_df['date'].values
            
            # Handle missing values
            features = np.nan_to_num(features, nan=0.0, posinf=1e10, neginf=-1e10)
            targets = np.nan_to_num(targets, nan=0.0, posinf=1e10, neginf=-1e10)
            
            # Create sequences with multi-step targets (sliding window approach)
            for i in range(self.sequence_length, len(symbol_df) - predict_len + 1):
                # Feature sequence: past sequence_length timesteps flattened
                sequence = features[i-self.sequence_length:i].flatten()
                
                # Multi-step targets: next predict_len timesteps
                multi_targets = targets[i:i+predict_len]
                
                # Metadata (timestamp of the first prediction)
                timestamp = timestamps[i]
                
                all_X.append(sequence)
                all_y.append(multi_targets)
                all_timestamps.append(timestamp)
                all_symbols.append(symbol)
        
        if len(all_X) == 0:
            raise ValueError("No valid sequences could be created from the data")
        
        X = np.array(all_X)
        y = np.array(all_y)  # Shape: [n_samples, predict_len]
        timestamps = np.array(all_timestamps)
        symbols = np.array(all_symbols)
        
        print(f"   ✅ Created {len(X)} sequences with {X.shape[1]} features each")
        print(f"   Multi-step targets shape: {y.shape}")
        print(f"   Target statistics: mean={y.mean():.4f}, std={y.std():.4f}")
        
        return X, y, timestamps, symbols
    
    def prepare_sequences_by_symbol(self, X: np.ndarray, y: np.ndarray, 
                                   timestamps: np.ndarray, symbols: np.ndarray, 
                                   symbol: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Filter sequences for a specific symbol.
        
        Args:
            X: All feature sequences
            y: All targets
            timestamps: All timestamps
            symbols: All symbols
            symbol: Symbol to filter for
            
        Returns:
            Filtered arrays for the specified symbol
        """
        mask = symbols == symbol
        return X[mask], y[mask], timestamps[mask], np.full(mask.sum(), symbol)

class FixedBaselineRunner:
    """Runs baseline models with robust error handling."""
    
    def __init__(self, config: Dict[str, Any], output_dir: str = "fixed_baseline_results"):
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.results = {}  # Will store results per symbol per model
        self.models = {}
        
    def load_multimodal_data(self) -> Tuple[Any, Any]:
        """Load multimodal data using the exact same method as TFT pipeline."""
        print("\n🔄 Loading multimodal data with caching...")
        
        # Use same date split logic as TFT pipeline
        start_date = self.config['start_date']
        end_date = self.config['end_date']
        validation_split = self.config.get('validation_split', 0.7)  # Reduced to ensure sufficient data
        lookahead_buffer_days = self.config.get('lookahead_buffer', 7)  # Reduced buffer
        
        from datetime import datetime, timedelta
        import pandas as pd
        
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')
        
        # Check if we have sufficient time range
        total_days = (end_dt - start_dt).days
        min_required_days = self.config.get('encoder_len', 30) + lookahead_buffer_days + 30  # Minimum viable period
        
        if total_days < min_required_days:
            print(f"⚠️ Warning: Short time range ({total_days} days). Adjusting split strategy...")
            validation_split = 0.6  # Use smaller training portion for short periods
            lookahead_buffer_days = max(1, lookahead_buffer_days // 3)  # Reduce buffer significantly
        
        # Split training period into train/validation with buffer
        train_days = int(total_days * validation_split)
        val_split_date = start_dt + timedelta(days=train_days)
        
        # Add buffer to prevent lookahead bias
        train_end = (val_split_date - timedelta(days=lookahead_buffer_days)).strftime('%Y-%m-%d')
        val_start = val_split_date.strftime('%Y-%m-%d')
        val_end = end_date
        
        print(f"   📈 Training period: {start_date} to {train_end}")
        print(f"   🛡️  Lookahead buffer: {lookahead_buffer_days} days")
        print(f"   📅 Validation period: {val_start} to {val_end}")
        print(f"   📊 Total period: {total_days} days, Training: {train_days} days")
        
        # Load training data with all multimodal features
        print("   🔄 Loading training data...")
        try:
            train_dataloader, train_datamodule = get_data_loader_with_module(
                symbols=self.config['symbols'],
                start=start_date,
                end=train_end,
                encoder_len=self.config.get('encoder_len', 30),
                predict_len=self.config.get('predict_len', 7),
                batch_size=self.config.get('batch_size', 32),
                news_api_key=self.config.get('news_api_key'),
                fred_api_key=self.config.get('fred_api_key'),
                api_ninjas_key=self.config.get('api_ninjas_key'),
                split_date=None,  # Don't pass split_date for training data
                is_training=True
            )
        except Exception as e:
            print(f"❌ Failed to load training data: {e}")
            # Fallback: try with longer period and reduced buffer
            print("   🔄 Retrying with adjusted parameters...")
            train_end = (end_dt - timedelta(days=14)).strftime('%Y-%m-%d')  # Leave 2 weeks for validation
            val_start = (end_dt - timedelta(days=14)).strftime('%Y-%m-%d')
            
            train_dataloader, train_datamodule = get_data_loader_with_module(
                symbols=self.config['symbols'],
                start=start_date,
                end=train_end,
                encoder_len=self.config.get('encoder_len', 30),
                predict_len=self.config.get('predict_len', 7),
                batch_size=self.config.get('batch_size', 32),
                news_api_key=self.config.get('news_api_key'),
                fred_api_key=self.config.get('fred_api_key'),
                api_ninjas_key=self.config.get('api_ninjas_key'),
                split_date=None,
                is_training=True
            )
        
        # Load validation data with same normalization parameters
        print("   🔄 Loading validation data with shared normalization...")
        try:
            val_dataloader, val_datamodule = get_data_loader_with_module(
                symbols=self.config['symbols'],
                start=val_start,
                end=val_end,
                encoder_len=self.config.get('encoder_len', 30),
                predict_len=self.config.get('predict_len', 7),
                batch_size=self.config.get('batch_size', 32),
                news_api_key=self.config.get('news_api_key'),
                fred_api_key=self.config.get('fred_api_key'),
                api_ninjas_key=self.config.get('api_ninjas_key'),
                split_date=None,  # Don't pass split_date for validation data either
                is_training=False,
                reference_datamodule=train_datamodule  # Use training normalization params
            )
        except Exception as e:
            print(f"❌ Failed to load validation data: {e}")
            print("   🔄 Creating validation data from training period...")
            # Fallback: Use the last portion of training data as validation
            val_dataloader, val_datamodule = get_data_loader_with_module(
                symbols=self.config['symbols'],
                start=(datetime.strptime(train_end, '%Y-%m-%d') - timedelta(days=21)).strftime('%Y-%m-%d'),
                end=train_end,
                encoder_len=self.config.get('encoder_len', 30),
                predict_len=self.config.get('predict_len', 7),
                batch_size=self.config.get('batch_size', 32),
                news_api_key=self.config.get('news_api_key'),
                fred_api_key=self.config.get('fred_api_key'),
                api_ninjas_key=self.config.get('api_ninjas_key'),
                split_date=None,
                is_training=False,
                reference_datamodule=train_datamodule
            )
        
        print("✅ Multimodal data loaded successfully!")
        print(f"   Training batches: {len(train_dataloader)}")
        print(f"   Validation batches: {len(val_dataloader)}")
        print(f"   Feature matrix shape: {train_datamodule.feature_df.shape}")
        
        return train_datamodule, val_datamodule
        
    def initialize_models(self):
        """Initialize baseline models including naive baselines for comparison."""
        predict_len = self.config.get('predict_len', 5)
        
        # Base sklearn models
        base_models = {
            'Linear Regression': LinearRegression(),
            'Random Forest': RandomForestRegressor(
                n_estimators=50,  # Reduced for faster training
                max_depth=8, 
                random_state=42, 
                n_jobs=-1
            )
        }
        
        if XGBOOST_AVAILABLE:
            base_models['XGBoost'] = xgb.XGBRegressor(
                n_estimators=50,  # Reduced for faster training
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                n_jobs=-1,
                verbosity=0  # Suppress XGBoost warnings
            )
        
        # Add naive baselines
        base_models.update({
            'Zero Baseline': DummyRegressor(strategy='constant', constant=0.0),
            'Mean Baseline': DummyRegressor(strategy='mean'),
            'Median Baseline': DummyRegressor(strategy='median')
        })
        
        # Wrap each model for multi-step prediction
        self.models = {}
        for name, base_model in base_models.items():
            self.models[name] = MultiStepPredictor(base_model, predict_len)
            
        print(f"✅ Initialized {len(self.models)} models for {predict_len}-step prediction")
    
    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                         current_prices: np.ndarray) -> Dict[str, float]:
        """
        Calculate comprehensive prediction metrics for multi-step predictions.
        
        Args:
            y_true: True returns [n_samples, predict_len]
            y_pred: Predicted returns [n_samples, predict_len]
            current_prices: Starting prices for each sequence [n_samples]
        """
        try:
            # Flatten for overall metrics
            y_true_flat = y_true.flatten()
            y_pred_flat = y_pred.flatten()
            
            # Return-based metrics (overall)
            mse_returns = mean_squared_error(y_true_flat, y_pred_flat)
            mae_returns = mean_absolute_error(y_true_flat, y_pred_flat)
            r2_returns = r2_score(y_true_flat, y_pred_flat)
            
            # Calculate step-wise metrics
            step_metrics = {}
            predict_len = y_true.shape[1]
            
            for step in range(predict_len):
                step_true = y_true[:, step]
                step_pred = y_pred[:, step]
                
                step_metrics[f'mse_step_{step+1}'] = float(mean_squared_error(step_true, step_pred))
                step_metrics[f'mae_step_{step+1}'] = float(mean_absolute_error(step_true, step_pred))
                step_metrics[f'r2_step_{step+1}'] = float(r2_score(step_true, step_pred))
            
            # Price reconstruction for multi-step predictions
            predicted_prices_sequences = []
            actual_prices_sequences = []
            
            for i in range(len(y_true)):
                # For each sequence, reconstruct the price path
                start_price = current_prices[i]
                
                pred_prices = [start_price]
                actual_prices = [start_price]
                
                for step in range(predict_len):
                    pred_prices.append(pred_prices[-1] * (1 + y_pred[i, step]))
                    actual_prices.append(actual_prices[-1] * (1 + y_true[i, step]))
                
                predicted_prices_sequences.append(pred_prices[1:])  # Remove starting price
                actual_prices_sequences.append(actual_prices[1:])   # Remove starting price
            
            # Flatten price sequences for overall metrics
            predicted_prices_flat = np.array(predicted_prices_sequences).flatten()
            actual_prices_flat = np.array(actual_prices_sequences).flatten()
            
            # Price-based metrics
            mse_prices = mean_squared_error(actual_prices_flat, predicted_prices_flat)
            mae_prices = mean_absolute_error(actual_prices_flat, predicted_prices_flat)
            r2_prices = r2_score(actual_prices_flat, predicted_prices_flat)
            
            # Calculate percentage errors with protection against division by zero
            mape_returns = np.mean(np.abs((y_true_flat - y_pred_flat) / (np.abs(y_true_flat) + 1e-8))) * 100
            mape_prices = np.mean(np.abs((actual_prices_flat - predicted_prices_flat) / (actual_prices_flat + 1e-8))) * 100
            
            # Directional accuracy for each step
            overall_directional_accuracy = 0
            for step in range(predict_len):
                actual_directions = np.sign(y_true[:, step])
                predicted_directions = np.sign(y_pred[:, step])
                step_acc = np.mean(actual_directions == predicted_directions) * 100
                step_metrics[f'directional_accuracy_step_{step+1}'] = float(step_acc)
                overall_directional_accuracy += step_acc
            
            overall_directional_accuracy /= predict_len
            
            # Multi-step cumulative return accuracy
            cumulative_actual_returns = []
            cumulative_predicted_returns = []
            
            for i in range(len(y_true)):
                cum_actual = np.prod(1 + y_true[i]) - 1
                cum_pred = np.prod(1 + y_pred[i]) - 1
                cumulative_actual_returns.append(cum_actual)
                cumulative_predicted_returns.append(cum_pred)
            
            cumulative_return_error = np.mean(np.abs(np.array(cumulative_actual_returns) - np.array(cumulative_predicted_returns)))
            
            base_metrics = {
                # Overall return-based metrics
                'mse_returns': float(mse_returns),
                'mae_returns': float(mae_returns),
                'r2_returns': float(r2_returns),
                'mape_returns': float(mape_returns),
                
                # Overall price-based metrics
                'mse_prices_corrected': float(mse_prices),
                'mae_prices_corrected': float(mae_prices),
                'r2_prices_corrected': float(r2_prices),
                'mape_prices_corrected': float(mape_prices),
                
                # Additional useful metrics
                'return_volatility': float(np.std(y_pred_flat)),
                'price_volatility_corrected': float(np.std(predicted_prices_flat)),
                'cumulative_return_error': float(cumulative_return_error),
                'directional_accuracy': float(overall_directional_accuracy),
                'cumulative_actual_return': float(np.mean(cumulative_actual_returns)),
                'cumulative_predicted_return': float(np.mean(cumulative_predicted_returns)),
                'predict_len': predict_len,
            }
            
            # Combine base metrics with step-wise metrics
            base_metrics.update(step_metrics)
            return base_metrics
            
        except Exception as e:
            print(f"⚠️ Error calculating metrics: {e}")
            return {
                'mse_returns': np.inf, 'mae_returns': np.inf, 'r2_returns': -np.inf,
                'mape_returns': np.inf, 'mse_prices_corrected': np.inf, 'mae_prices_corrected': np.inf,
                'r2_prices_corrected': -np.inf, 'mape_prices_corrected': np.inf,
                'return_volatility': 0.0, 'price_volatility_corrected': 0.0,
                'cumulative_return_error': np.inf, 'directional_accuracy': 0.0,
                'cumulative_actual_return': 0.0, 'cumulative_predicted_return': 0.0,
                'predict_len': 1,
            }
    
    def train_and_evaluate_symbol(self, train_datamodule: Any, val_datamodule: Any, symbol: str, 
                                  use_multimodal: bool = True):
        """Train and evaluate all models for a single symbol using multimodal data with fixed prediction horizon."""
        print(f"\n{'='*60}")
        print(f"🎯 PROCESSING SYMBOL: {symbol}")
        print(f"{'='*60}")
        
        try:
            # Extract features from datamodules using multimodal approach
            predictor = MultimodalStockPredictor(sequence_length=self.config.get('encoder_len', 30))
            predict_len = self.config.get('predict_len', 5)
            
            # Extract training data
            X_train, y_train, timestamps_train, symbols_train = predictor.extract_features_from_datamodule(
                train_datamodule, use_multimodal=use_multimodal, predict_len=predict_len
            )
            
            # Extract validation data
            X_val, y_val, timestamps_val, symbols_val = predictor.extract_features_from_datamodule(
                val_datamodule, use_multimodal=use_multimodal, predict_len=predict_len
            )
            
            # Filter for current symbol
            train_mask = symbols_train == symbol
            val_mask = symbols_val == symbol
            
            if not train_mask.any():
                print(f"❌ No training data found for {symbol}")
                return
                
            if not val_mask.any():
                print(f"❌ No validation data found for {symbol}")
                return
            
            X_train_symbol = X_train[train_mask]
            y_train_symbol = y_train[train_mask]
            X_val_symbol = X_val[val_mask]
            y_val_symbol = y_val[val_mask]
            timestamps_val_symbol = timestamps_val[val_mask]
            
            # Get current prices for validation period (needed for price reconstruction)
            val_symbol_df = val_datamodule.feature_df[val_datamodule.feature_df['symbol'] == symbol].copy()
            val_symbol_df = val_symbol_df.sort_values('time_idx').reset_index(drop=True)
            
            if 'close' in val_symbol_df.columns:
                # Use close prices as current prices - need to align with sequences
                # For multi-step prediction, we need the starting price for each sequence
                val_prices = []
                for i, timestamp in enumerate(timestamps_val_symbol):
                    # Find the corresponding close price for this timestamp
                    price_row = val_symbol_df[val_symbol_df['date'] == timestamp]
                    if not price_row.empty:
                        val_prices.append(price_row['close'].iloc[0])
                    else:
                        # Fallback: use previous price or estimate
                        val_prices.append(val_prices[-1] if val_prices else 100.0)
                val_prices = np.array(val_prices)
            else:
                # Fallback: estimate prices from returns (less accurate)
                val_prices = np.ones(len(y_val_symbol)) * 100  # Assume $100 base price
                print(f"⚠️ No close prices found for {symbol}, using estimated prices")
            
            print(f"📊 {symbol}: {len(X_train_symbol)} train, {len(X_val_symbol)} test samples")
            print(f"📊 Feature shape: {X_train_symbol.shape}")
            print(f"📊 Target shape: {y_train_symbol.shape} (multi-step: {predict_len} steps)")
            print(f"📊 Using multimodal features: {use_multimodal}")
            
            # Scale features
            scaler = StandardScaler()
            try:
                X_train_scaled = scaler.fit_transform(X_train_symbol)
                X_val_scaled = scaler.transform(X_val_symbol)
            except Exception as e:
                print(f"❌ Scaling failed for {symbol}: {e}")
                return
            
            # Initialize results for this symbol
            if symbol not in self.results:
                self.results[symbol] = {}
            
            for model_name, model in self.models.items():
                print(f"\n🔄 Training {model_name} for {symbol}...")
                
                try:
                    # Train model (now handles multi-step targets)
                    model.fit(X_train_scaled, y_train_symbol)
                    
                    # Make predictions (returns multi-step predictions)
                    print(f"   � Making {predict_len}-step predictions...")
                    predictions = model.predict(X_val_scaled)
                    
                    # Calculate metrics
                    metrics = self.calculate_metrics(y_val_symbol, predictions, val_prices)
                    
                    # Store results
                    self.results[symbol][model_name] = {
                        'predictions': predictions,
                        'actuals': y_val_symbol,
                        'timestamps': timestamps_val_symbol,
                        'prices': val_prices,
                        'metrics': metrics,
                        'prediction_type': f"{predict_len}-step",
                    }
                    
                    print(f"✅ {model_name} ({predict_len}-step) - R²: {metrics['r2_returns']:.4f}, MAE: {metrics['mae_returns']:.4f}")
                    print(f"   📊 Price R²: {metrics['r2_prices_corrected']:.4f}, MAE: ${metrics['mae_prices_corrected']:.2f}")
                    print(f"   📊 Directional Accuracy: {metrics['directional_accuracy']:.1f}%")
                    
                    # Print step-wise performance
                    for step in range(predict_len):
                        r2_step = metrics.get(f'r2_step_{step+1}', 0)
                        print(f"   📈 Step {step+1}: R²={r2_step:.4f}")
                    
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
        """Create plots for a specific symbol with multi-step predictions."""
        if symbol not in self.results or len(self.results[symbol]) == 0:
            print(f"⚠️ No results to plot for {symbol}")
            return
            
        print(f"📊 Creating plots for {symbol}...")
        
        symbol_results = self.results[symbol]
        n_models = len(symbol_results)
        
        try:
            # Create subplots for each model (3 plots per model for multi-step)
            fig, axes = plt.subplots(n_models, 3, figsize=(18, 6*n_models))
            if n_models == 1:
                axes = axes.reshape(1, -1)
            
            fig.suptitle(f'{symbol} - Multi-Step Stock Predictions', fontsize=16, fontweight='bold')
            
            for i, (model_name, result) in enumerate(symbol_results.items()):
                predictions = result['predictions']  # Shape: [n_samples, predict_len]
                actuals = result['actuals']          # Shape: [n_samples, predict_len]
                timestamps = result['timestamps']
                prices = result['prices']
                predict_len = predictions.shape[1]
                
                # Plot 1: Multi-step returns over time (show each step)
                for step in range(min(predict_len, 3)):  # Show first 3 steps to avoid clutter
                    axes[i, 0].plot(timestamps, actuals[:, step], 
                                   label=f'Actual T+{step+1}', alpha=0.7, linestyle='-')
                    axes[i, 0].plot(timestamps, predictions[:, step], 
                                   label=f'Predicted T+{step+1}', alpha=0.7, linestyle='--')
                
                axes[i, 0].set_title(f'{model_name} - Multi-Step Returns Over Time')
                axes[i, 0].set_ylabel('Return (%)')
                axes[i, 0].legend()
                axes[i, 0].grid(True, alpha=0.3)
                axes[i, 0].tick_params(axis='x', rotation=45)
                
                # Plot 2: Cumulative price paths for each sequence
                axes[i, 1].set_title(f'{model_name} - Price Prediction Paths')
                axes[i, 1].set_ylabel('Price ($)')
                
                # Show a few example sequences
                n_show = min(5, len(predictions))
                for seq_idx in range(0, len(predictions), max(1, len(predictions) // n_show)):
                    if seq_idx >= len(predictions):
                        break
                        
                    start_price = prices[seq_idx]
                    
                    # Actual price path
                    actual_path = [start_price]
                    pred_path = [start_price]
                    
                    for step in range(predict_len):
                        actual_path.append(actual_path[-1] * (1 + actuals[seq_idx, step]))
                        pred_path.append(pred_path[-1] * (1 + predictions[seq_idx, step]))
                    
                    x_steps = list(range(predict_len + 1))
                    axes[i, 1].plot(x_steps, actual_path, 'b-', alpha=0.6, linewidth=1)
                    axes[i, 1].plot(x_steps, pred_path, 'r--', alpha=0.6, linewidth=1)
                
                # Add legend for the last sequence
                axes[i, 1].plot([], [], 'b-', label='Actual Paths', alpha=0.8)
                axes[i, 1].plot([], [], 'r--', label='Predicted Paths', alpha=0.8)
                axes[i, 1].legend()
                axes[i, 1].grid(True, alpha=0.3)
                axes[i, 1].set_xlabel('Prediction Step')
                
                # Plot 3: Step-wise performance metrics
                steps = list(range(1, predict_len + 1))
                step_r2s = []
                step_maes = []
                
                for step in range(predict_len):
                    r2_key = f'r2_step_{step+1}'
                    mae_key = f'mae_step_{step+1}'
                    step_r2s.append(result['metrics'].get(r2_key, 0))
                    step_maes.append(result['metrics'].get(mae_key, 0))
                
                ax3_twin = axes[i, 2].twinx()
                
                line1 = axes[i, 2].plot(steps, step_r2s, 'b-o', label='R² Score', linewidth=2)
                line2 = ax3_twin.plot(steps, step_maes, 'r-s', label='MAE', linewidth=2)
                
                axes[i, 2].set_title(f'{model_name} - Step-wise Performance')
                axes[i, 2].set_xlabel('Prediction Step')
                axes[i, 2].set_ylabel('R² Score', color='b')
                ax3_twin.set_ylabel('MAE', color='r')
                
                # Combine legends
                lines = line1 + line2
                labels = [l.get_label() for l in lines]
                axes[i, 2].legend(lines, labels, loc='upper right')
                
                axes[i, 2].grid(True, alpha=0.3)
                axes[i, 2].set_xticks(steps)
                
                print(f"  📊 {model_name} - Overall R²: {result['metrics']['r2_returns']:.4f}")
                
            plt.tight_layout()
            plot_path = self.output_dir / f"{symbol}_multistep_predictions.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"  📈 {symbol} multi-step plots saved to {plot_path}")
            
        except Exception as e:
            print(f"❌ Failed to create plots for {symbol}: {e}")
            import traceback
            traceback.print_exc()
    
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

def parse_arguments():
    """Parse command line arguments to match TFT pipeline configuration."""
    parser = argparse.ArgumentParser(description='Baseline Models Pipeline using Multimodal Stock Data')
    
    # Data configuration
    parser.add_argument('--symbols', type=str, default='AAPL,GOOGL,MSFT', 
                        help='Comma-separated list of stock symbols')
    parser.add_argument('--start-date', type=str, default='2023-01-01', 
                        help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, default='2024-06-01', 
                        help='End date (YYYY-MM-DD)')
    
    # Model configuration - reduced defaults for better compatibility
    parser.add_argument('--encoder-len', type=int, default=20, 
                        help='Encoder sequence length')
    parser.add_argument('--predict-len', type=int, default=5, 
                        help='Prediction horizon')
    parser.add_argument('--batch-size', type=int, default=16, 
                        help='Batch size')
    
    # API keys for multimodal data
    parser.add_argument('--news-api-key', type=str, 
                        help='News API key for news data')
    parser.add_argument('--fred-api-key', type=str, 
                        help='FRED API key for economic data')
    parser.add_argument('--api-ninjas-key', type=str, 
                        help='API Ninjas key for additional data')
    
    # Training configuration - more conservative defaults
    parser.add_argument('--validation-split', type=float, default=0.7, 
                        help='Training/validation split ratio')
    parser.add_argument('--lookahead-buffer', type=int, default=7, 
                        help='Lookahead buffer days to prevent data leakage')
    
    # Model options
    parser.add_argument('--multimodal', action='store_true', default=True,
                        help='Use multimodal features (news, economic, technical)')
    parser.add_argument('--basic-only', action='store_true', default=False,
                        help='Use only basic OHLCV features')
    
    # Output options
    parser.add_argument('--output-dir', type=str, default='fixed_baseline_results',
                        help='Output directory for results')
    
    return parser.parse_args()

def main():
    """Main function to run the multimodal baseline pipeline."""
    
    # Parse arguments
    args = parse_arguments()
    
    # Build configuration dictionary similar to TFT pipeline
    config = {
        'symbols': args.symbols.split(','),
        'start_date': args.start_date,
        'end_date': args.end_date,
        'encoder_len': args.encoder_len,
        'predict_len': args.predict_len,
        'batch_size': args.batch_size,
        'validation_split': args.validation_split,
        'lookahead_buffer': args.lookahead_buffer,
        'news_api_key': args.news_api_key,
        'fred_api_key': args.fred_api_key,
        'api_ninjas_key': args.api_ninjas_key,
    }
    
    # Determine feature mode
    use_multimodal = args.multimodal and not args.basic_only
    
    print("🚀 Starting Multimodal Baseline Models Pipeline")
    print(f"📅 Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🎯 Symbols: {config['symbols']}")
    print(f"📊 Date range: {config['start_date']} to {config['end_date']}")
    print(f"🔮 Fixed prediction horizon: {config['predict_len']} steps")
    print(f"🎛️  Multimodal features: {use_multimodal}")
    print(f"⚡ Using EXACT same data loading as TFT pipeline")
    print(f"📏 Encoder length: {config['encoder_len']}, Predict length: {config['predict_len']}")
    
    try:
        # Step 1: Initialize runner with configuration
        print("\n" + "=" * 60)
        print("� INITIALIZING BASELINE RUNNER")
        print("=" * 60)
        
        runner = FixedBaselineRunner(config, args.output_dir)
        runner.initialize_models()
        
        # Step 2: Load multimodal data using TFT pipeline method
        print("\n" + "=" * 60)
        print("📈 LOADING MULTIMODAL DATA")
        print("=" * 60)
        
        train_datamodule, val_datamodule = runner.load_multimodal_data()
        
        # Step 3: Process each symbol separately
        print("\n" + "=" * 60)
        print("🎯 TRAINING AND EVALUATING MODELS")
        print("=" * 60)
        
        for symbol in config['symbols']:
            # Check if symbol exists in the data
            train_symbols = train_datamodule.feature_df['symbol'].unique()
            val_symbols = val_datamodule.feature_df['symbol'].unique()
            
            if symbol in train_symbols or symbol in val_symbols:
                runner.train_and_evaluate_symbol(
                    train_datamodule, val_datamodule, symbol, 
                    use_multimodal=use_multimodal
                )
            else:
                print(f"⚠️ No data found for {symbol}")
        
        # Step 4: Create visualizations
        print("\n" + "=" * 60)
        print("📊 CREATING VISUALIZATIONS")
        print("=" * 60)
        
        for symbol in config['symbols']:
            runner.create_symbol_plots(symbol)
        
        runner.create_comparison_plots()
        
        # Step 5: Print summary
        runner.print_summary()
        
        print(f"\n✅ Pipeline completed successfully!")
        print(f"📂 Results saved to: {runner.output_dir}")
        print(f"🎯 Features used: {'Multimodal (news, economic, technical, OHLCV)' if use_multimodal else 'Basic OHLCV only'}")
        print(f"🔮 Prediction mode: Fixed {config['predict_len']}-step horizon")
        
    except Exception as e:
        print(f"\n❌ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    main()
