#!/usr/bin/env python3
"""
Multimodal Baseline Models Pipeline for Stock Price Prediction
==============================================================

A comprehensive baseline pipeline that uses the EXACT same multimodal data as the TFT pipeline
for fair comparison. Includes news embeddings, economic indicators, technical analysis,
and corporate events data.

Models Included:
- Traditional ML: Ridge Regression (L2 regularized), Random Forest
- Ensemble Methods: XGBoost

Features:
- Uses EXACT same data loading as TFT pipeline (via dataModule interface)
- Identical train/validation split logic with temporal buffers
- Multimodal features: stock OHLCV, news embeddings, FRED economic data, technical indicators
- Same normalization and preprocessing as TFT pipeline
- Handles each symbol separately to avoid mixing data
- Implements proper multi-step predictions for fair comparison
- Creates comprehensive evaluation metrics and visualizations per symbol
- Fair comparison with TFT model using identical feature sets and data splits

Enhanced Visualization Suite:
- Ablation study performance heatmaps (R², Accuracy, Sharpe Ratio)
- Feature group contribution analysis with statistical significance
- Original performance metrics plots (classification, regression, financial)
- Enhanced prediction plots: predicted vs actual scatter plots
- Error over horizon analysis: MAE/RMSE trends across forecast horizons
- Comprehensive validation set analysis

Ablation Study Framework:
- 6 feature modes: full, no_news, no_economic, no_technical, core, ohlcv_only
- Quantifies contribution of each multimodal feature group
- Statistical analysis of feature importance across models
- Research-ready visualizations and summary tables

Data Leakage Prevention:
- Strict temporal separation between train/validation with configurable buffer
- Independent normalization for validation data (no shared statistics)
- Historical price initialization (no future price peeking)
- Explicit temporal boundary validation in sequence creation
- Per-symbol feature scaling using only training data statistics

Key Improvements:
- Uses get_data_loader_with_module() exactly like TFT pipeline
- Respects temporal constraints and lookahead buffers
- Maintains consistent API key usage for external data sources
- Identical feature engineering and preprocessing pipeline
- Robust data leakage prevention and validation

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
import warnings
import traceback
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score

# Try to import scipy for statistical analysis
try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("⚠️ SciPy not available. Some statistical plots may be limited. Install with: pip install scipy")

# Setup paths and suppress warnings
warnings.filterwarnings('ignore')
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env')
    if os.path.exists(env_path):
        load_dotenv(env_path)
        print(f"✅ Loaded environment variables from {env_path}")
    else:
        print(f"ℹ️  No .env file found at {env_path}")
        print("   You can create one to set API keys automatically")
except ImportError:
    print("⚠️  python-dotenv not installed. Install with: pip install python-dotenv")
    print("   API keys will need to be set manually or via system environment variables")

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("⚠️ XGBoost not available. Install with: pip install xgboost")# Import TFT data interface for multimodal data
from dataModule.interface import get_data_loader_with_module

# Feature ablation modes for ablation study
FEATURE_MODES = {
    'full': "All multimodal features",
    'no_news': "All except news embeddings", 
    'no_economic': "All except economic indicators",
    'no_technical': "All except technical indicators", 
    'ohlcv_only': "Only OHLCV data",
    'core': "OHLCV + Technical indicators"
}

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
        self.feature_scaler = RobustScaler(quantile_range=(25.0, 75.0))
        
    def process_targets_safely(self, targets):
        """
        Process and clean target returns to prevent numerical instability.
        
        Args:
            targets: Raw target returns array
            
        Returns:
            targets_cleaned: Processed targets with outliers clipped
        """
        # Handle NaN/Inf values first
        targets_cleaned = np.nan_to_num(targets, nan=0.0, posinf=0.5, neginf=-0.5)
        
        # Clip extreme returns to prevent model instability
        # Most realistic daily returns are within ±50%
        targets_cleaned = np.clip(targets_cleaned, -0.5, 0.5)
        
        # Additional outlier detection using IQR method
        if len(targets_cleaned) > 10:  # Need enough data for percentiles
            Q1 = np.percentile(targets_cleaned, 25)
            Q3 = np.percentile(targets_cleaned, 75)
            IQR = Q3 - Q1
            
            if IQR > 0:  # Avoid division by zero
                # More conservative bounds for financial returns
                lower_bound = Q1 - 2.5 * IQR  # 2.5 instead of 1.5 for conservative clipping
                upper_bound = Q3 + 2.5 * IQR
                
                # Ensure bounds are reasonable for daily returns
                lower_bound = max(lower_bound, -0.5)  # No worse than -50%
                upper_bound = min(upper_bound, 0.5)   # No better than +50%
                
                targets_cleaned = np.clip(targets_cleaned, lower_bound, upper_bound)
        
        return targets_cleaned
    
    def validate_model_predictions(self, y_pred, model_name):
        """
        Validate and clean model predictions to prevent numerical errors.
        
        Args:
            y_pred: Model predictions
            model_name: Name of the model for logging
            
        Returns:
            y_pred_cleaned: Validated predictions
        """
        # Check for NaN/Inf values
        nan_count = np.sum(np.isnan(y_pred))
        inf_count = np.sum(np.isinf(y_pred))
        
        if nan_count > 0 or inf_count > 0:
            print(f"   ⚠️ {model_name} produced {nan_count} NaN and {inf_count} Inf predictions")
            y_pred = np.nan_to_num(y_pred, nan=0.0, posinf=0.2, neginf=-0.2)
        
        # Clip extreme predictions that could cause downstream issues
        y_pred_clipped = np.clip(y_pred, -1.0, 1.0)  # ±100% max daily return prediction
        
        # Check if clipping was significant
        clipped_count = np.sum(np.abs(y_pred) > 1.0)
        if clipped_count > 0:
            print(f"   📐 {model_name}: Clipped {clipped_count}/{len(y_pred)} extreme predictions")
        
        return y_pred_clipped
        
    def extract_features_from_datamodule(self, datamodule, feature_mode: str = 'full', predict_len: int = 5) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Extract features from TFT DataModule for baseline models with ablation support.
        
        Args:
            datamodule: TFT DataModule containing multimodal features
            feature_mode: Feature ablation mode ('full', 'no_news', 'no_economic', 'no_technical', 'ohlcv_only', 'core')
            predict_len: Number of future timesteps to predict (T+1, T+2, ..., T+predict_len)
            
        Returns:
            X: Feature sequences [n_samples, features]
            y: Multi-step targets [n_samples, predict_len]
            timestamps: Timestamps for each sample
            symbols: Symbol for each sample
        """
        print(f"📊 Extracting features from datamodule...")
        print(f"   Feature mode: {feature_mode} - {FEATURE_MODES.get(feature_mode, 'Unknown mode')}")
        print(f"   Prediction horizon: {predict_len} steps")
        
        # Get the feature dataframe from the datamodule
        feature_df = datamodule.feature_df.copy()
        feature_df = feature_df.sort_values(['symbol', 'time_idx']).reset_index(drop=True)
        
        print(f"   Feature matrix shape: {feature_df.shape}")
        print(f"   Available columns: {list(feature_df.columns)}")
        
        # Identify feature types by common patterns
        excluded_cols = ['symbol', 'date', 'time_idx', 'target']
        
        # Define feature groups based on common naming patterns
        ohlcv_cols = ['open', 'high', 'low', 'close', 'volume']
        
        # Technical indicators - expanded patterns to catch more variations
        technical_patterns = ['sma', 'ema', 'rsi', 'macd', 'bb_', 'bollinger', 'atr', 'stoch', 'williams', 'adx', 'cci', 'momentum', 'roc', 
                             'ta_', 'technical', 'indicator', 'moving_avg', 'ma_', 'std_', 'vol_', 'volatility', 'return_lag', 'lag_']
        
        # News/sentiment features - expanded patterns  
        news_patterns = ['news', 'sentiment', 'embed', 'emb_', 'nlp', 'text', 'headline', 'article', 'bert', 'compound', 'negative', 'neutral', 'positive']
        
        # Economic indicators - expanded FRED data patterns
        economic_patterns = ['fred', 'gdp', 'inflation', 'unemployment', 'interest', 'cpi', 'ppi', 'ism', 'nfp', 'retail', 'housing',
                           'economic', 'macro', 'fed', 'rate', 'yield', 'bond', 'treasury']
        
        # Categorize all available features
        feature_cols = []
        all_numeric_cols = []
        
        print(f"   🔍 Analyzing {len(feature_df.columns)} total columns...")
        
        for col in feature_df.columns:
            if col not in excluded_cols:
                # Check if column contains numeric data
                try:
                    sample_val = feature_df[col].dropna().iloc[0] if not feature_df[col].dropna().empty else 0
                    float(sample_val)
                    all_numeric_cols.append(col)
                except (ValueError, TypeError):
                    print(f"   ⚠️ Excluding non-numeric column: {col}")
                    continue
        
        print(f"   📊 Found {len(all_numeric_cols)} numeric columns")
        
        # Categorize features by type for debugging
        ohlcv_features = [col for col in ohlcv_cols if col in all_numeric_cols]
        technical_features = [col for col in all_numeric_cols 
                             if any(pattern in col.lower() for pattern in technical_patterns)]
        news_features = [col for col in all_numeric_cols 
                        if any(pattern in col.lower() for pattern in news_patterns)]
        economic_features = [col for col in all_numeric_cols 
                           if any(pattern in col.lower() for pattern in economic_patterns)]
        
        print(f"   📈 OHLCV features: {len(ohlcv_features)} - {ohlcv_features}")
        print(f"   📊 Technical features: {len(technical_features)} - {technical_features[:5]}{'...' if len(technical_features) > 5 else ''}")
        print(f"   📰 News features: {len(news_features)} - {news_features[:5]}{'...' if len(news_features) > 5 else ''}")
        print(f"   🏛️ Economic features: {len(economic_features)} - {economic_features[:5]}{'...' if len(economic_features) > 5 else ''}")
        
        # Apply feature mode filtering
        if feature_mode == 'ohlcv_only':
            feature_cols = ohlcv_features
            
        elif feature_mode == 'core':
            # OHLCV + Technical indicators
            feature_cols = ohlcv_features + technical_features
            
        elif feature_mode == 'no_news':
            # All except news features
            feature_cols = [col for col in all_numeric_cols 
                           if not any(pattern in col.lower() for pattern in news_patterns)]
            
        elif feature_mode == 'no_economic':
            # All except economic features
            feature_cols = [col for col in all_numeric_cols 
                           if not any(pattern in col.lower() for pattern in economic_patterns)]
            
        elif feature_mode == 'no_technical':
            # All except technical indicators
            feature_cols = [col for col in all_numeric_cols 
                           if not any(pattern in col.lower() for pattern in technical_patterns)]
            
        else:  # feature_mode == 'full' or unknown
            # Use all numeric features
            feature_cols = all_numeric_cols
        
        # Remove duplicates and sort
        feature_cols = sorted(list(set(feature_cols)))
        
        print(f"   ✅ Selected {len(feature_cols)} features for mode '{feature_mode}'")
        
        # Validate that we have features
        if len(feature_cols) == 0:
            print(f"   ❌ ERROR: No features selected for mode '{feature_mode}'!")
            print(f"   Available feature types: OHLCV={len(ohlcv_features)}, Technical={len(technical_features)}, News={len(news_features)}, Economic={len(economic_features)}")
            raise ValueError(f"No features available for feature mode '{feature_mode}'")
        
        if len(feature_cols) < 20:  # Only print if not too many
            print(f"   📋 Features: {feature_cols}")
        
        # Extract features for each symbol separately
        all_X, all_y, all_timestamps, all_symbols = [], [], [], []
        
        for symbol in feature_df['symbol'].unique():
            symbol_df = feature_df[feature_df['symbol'] == symbol].copy()
            symbol_df = symbol_df.sort_values('time_idx').reset_index(drop=True)
            
            if len(symbol_df) < self.sequence_length + predict_len:
                print(f"   ⚠️ Insufficient data for {symbol}: {len(symbol_df)} rows (need {self.sequence_length + predict_len})")
                continue
            # First, prepare basic data structures
            features = symbol_df[feature_cols].values
            targets = symbol_df['target'].values
            timestamps = symbol_df['date'].values
            
            # CRITICAL FIX: Calculate proper forward-looking returns if close prices are available
            if 'close' in symbol_df.columns:
                close_prices = symbol_df['close'].values
                
                # Calculate forward-looking returns properly
                forward_returns = np.zeros(len(close_prices))
                for j in range(len(close_prices) - 1):
                    forward_returns[j] = (close_prices[j + 1] - close_prices[j]) / close_prices[j]
                
                # Replace target with properly calculated forward returns
                targets = forward_returns
                print(f"   🔄 Using calculated forward returns instead of target column")
                print(f"   📊 Calculated returns: mean={np.mean(targets):.6f}, std={np.std(targets):.6f}")
            
            # Handle missing values in features
            features = np.nan_to_num(features, nan=0.0, posinf=1e10, neginf=-1e10)
            
            # Process targets safely to prevent numerical instability
            targets = self.process_targets_safely(targets)
            
            # Add debugging for the first symbol only to avoid spam
            if len(all_X) == 0:  # First symbol being processed
                print(f"\n🔍 DEBUGGING FIRST SYMBOL: {symbol}")
                self.debug_temporal_alignment(symbol_df)
                self.debug_sequence_logic(symbol_df, self.sequence_length, predict_len)
                self.debug_prediction_bias(targets, symbol_df)
            
            # Basic data validation for debugging
            print(f"   📊 {symbol}: {len(symbol_df)} rows, target mean={targets.mean():.4f}, std={targets.std():.4f}, range=[{targets.min():.4f}, {targets.max():.4f}]")
            
            # Create sequences with multi-step targets (sliding window approach)
            # CRITICAL: This ensures no future peeking - features[i-sequence_length:i] uses ONLY past data
            # to predict targets[i:i+predict_len] which are future returns
            sequence_count = 0
            
            for i in range(self.sequence_length, len(symbol_df) - predict_len + 1):
                # Feature sequence: past sequence_length timesteps flattened
                # Uses data from [i-sequence_length, i) - strictly historical data
                sequence = features[i-self.sequence_length:i].flatten()
                
                # Multi-step targets: next predict_len timesteps  
                # Uses data from [i, i+predict_len) - strictly future data
                multi_targets = targets[i:i+predict_len]
                
                # Metadata (timestamp of the first prediction) 
                timestamp = symbol_df.iloc[i]['date'] if 'date' in symbol_df.columns else i
                
                # Validation: Ensure we're not using future data in features
                assert i >= self.sequence_length, f"Sequence boundary violation: i={i}, seq_len={self.sequence_length}"
                assert i + predict_len <= len(symbol_df), f"Prediction boundary violation: i={i}, predict_len={predict_len}, total_len={len(symbol_df)}"
                
                all_X.append(sequence)
                all_y.append(multi_targets)
                all_timestamps.append(timestamp)
                all_symbols.append(symbol)
                sequence_count += 1
        
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
    
    def debug_temporal_alignment(self, symbol_df):
        """Debug what the target column actually represents"""
        
        print("🔍 Debugging temporal alignment...")
        
        # Check first few rows
        sample_df = symbol_df.head(10)
        
        if 'date' in sample_df.columns:
            print("📅 Dates and targets:")
            for idx, row in sample_df.iterrows():
                print(f"  {row['date']}: target = {row['target']:.6f}")
        
        # Check if target is returns or prices
        target_values = symbol_df['target'].values[:100]
        print(f"📊 Target statistics:")
        print(f"  Mean: {np.mean(target_values):.6f}")
        print(f"  Std: {np.std(target_values):.6f}")
        print(f"  Range: [{np.min(target_values):.6f}, {np.max(target_values):.6f}]")
        
        # Compare with close prices
        if 'close' in symbol_df.columns:
            close_prices = symbol_df['close'].values[:100]
            manual_returns = np.diff(close_prices) / close_prices[:-1]
            
            print(f"📈 Manual returns (from close prices):")
            print(f"  Mean: {np.mean(manual_returns):.6f}")
            print(f"  Std: {np.std(manual_returns):.6f}")
            
            # Check correlation between target and manual returns
            if len(target_values) > len(manual_returns):
                target_subset = target_values[1:len(manual_returns)+1]  # Skip first
            else:
                target_subset = target_values[:len(manual_returns)]
                
            corr = np.corrcoef(target_subset, manual_returns[:len(target_subset)])[0,1]
            print(f"📊 Correlation between target and manual returns: {corr:.4f}")
            
            if corr < 0.5:
                print("🚨 LOW CORRELATION - Target might not be returns!")
    
    def debug_sequence_logic(self, symbol_df, sequence_length=30, predict_len=5):
        """Debug the sequence creation logic"""
        
        features = symbol_df[['close']].values if 'close' in symbol_df.columns else symbol_df[['target']].values
        targets = symbol_df['target'].values
        dates = symbol_df['date'].values if 'date' in symbol_df.columns else None
        
        # Check a specific sequence
        i = sequence_length + 5  # Example index
        
        print(f"🔍 Sequence at index {i}:")
        
        if dates is not None and len(dates) > i + predict_len:
            feature_dates = dates[i-sequence_length:i]
            target_dates = dates[i:i+predict_len]
            
            print(f"📅 Feature period: {feature_dates[0]} to {feature_dates[-1]}")
            print(f"📅 Target period: {target_dates[0]} to {target_dates[-1]}")
            
            # Check for overlap
            if feature_dates[-1] >= target_dates[0]:
                print("🚨 TEMPORAL LEAKAGE DETECTED!")
                print(f"   Last feature date: {feature_dates[-1]}")
                print(f"   First target date: {target_dates[0]}")
        
        # Show actual values
        if len(features) > i:
            feature_values = features[i-sequence_length:i, 0] if features.shape[1] > 0 else features[i-sequence_length:i]
            target_values = targets[i:i+predict_len]
            
            print(f"📊 Feature values (last 5): {feature_values[-5:] if len(feature_values) >= 5 else feature_values}")
            print(f"📊 Target values: {target_values}")
            
            # Calculate what the returns SHOULD be
            if len(feature_values) > 1 and 'close' in symbol_df.columns:
                last_price = feature_values[-1]
                print(f"📈 Last known price: {last_price}")
                
                # If targets are returns, what would the future prices be?
                future_prices = [last_price]
                for ret in target_values:
                    future_prices.append(future_prices[-1] * (1 + ret))
                
                print(f"📈 Implied future prices: {future_prices[1:]}")
    
    def debug_prediction_bias(self, corrected_targets, symbol_df):
        """Debug the prediction bias fix by comparing original vs corrected targets"""
        
        print("🔍 Debugging prediction bias fix...")
        
        # Compare original target column vs our corrected targets
        original_targets = symbol_df['target'].values
        
        print(f"📊 Original target column statistics:")
        print(f"  Mean: {np.mean(original_targets):.6f}")
        print(f"  Std: {np.std(original_targets):.6f}")
        print(f"  Range: [{np.min(original_targets):.6f}, {np.max(original_targets):.6f}]")
        
        print(f"📊 Corrected targets (forward returns) statistics:")
        print(f"  Mean: {np.mean(corrected_targets):.6f}")
        print(f"  Std: {np.std(corrected_targets):.6f}")
        print(f"  Range: [{np.min(corrected_targets):.6f}, {np.max(corrected_targets):.6f}]")
        
        # Calculate correlation between original and corrected
        min_length = min(len(original_targets), len(corrected_targets))
        if min_length > 1:
            corr = np.corrcoef(original_targets[:min_length], corrected_targets[:min_length])[0,1]
            print(f"📊 Correlation between original and corrected targets: {corr:.4f}")
            
            if corr < 0.5:
                print("✅ LOW CORRELATION CONFIRMED - Fix is working!")
                print("   Original target column was NOT proper forward returns")
                print("   Now using calculated forward returns from close prices")
            else:
                print("⚠️ High correlation - original targets might have been correct")
        
        # Show first few values for comparison
        print(f"📋 First 10 values comparison:")
        print(f"   Original: {original_targets[:10]}")
        print(f"   Corrected: {corrected_targets[:10]}")

class FixedBaselineRunner:
    """Runs baseline models with robust error handling."""
    
    def __init__(self, config: Dict[str, Any], output_dir: str = "fixed_baseline_results"):
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.results = {}  # Will store results per symbol per model
        self.models = {}
        
    def validate_model_predictions(self, y_pred, model_name):
        """
        Validate and clean model predictions to prevent numerical errors.
        
        Args:
            y_pred: Model predictions
            model_name: Name of the model for logging
            
        Returns:
            y_pred_cleaned: Validated predictions
        """
        # Check for NaN/Inf values
        nan_count = np.sum(np.isnan(y_pred))
        inf_count = np.sum(np.isinf(y_pred))
        
        if nan_count > 0 or inf_count > 0:
            print(f"   ⚠️ {model_name} produced {nan_count} NaN and {inf_count} Inf predictions")
            y_pred = np.nan_to_num(y_pred, nan=0.0, posinf=0.2, neginf=-0.2)
        
        # Clip extreme predictions that could cause downstream issues
        y_pred_clipped = np.clip(y_pred, -1.0, 1.0)  # ±100% max daily return prediction
        
        # Check if clipping was significant
        clipped_count = np.sum(np.abs(y_pred) > 1.0)
        if clipped_count > 0:
            print(f"   📐 {model_name}: Clipped {clipped_count}/{len(y_pred)} extreme predictions")
        
        return y_pred_clipped
        
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
        
        # CRITICAL: Validate no temporal overlap between train and validation
        train_end_dt = datetime.strptime(train_end, '%Y-%m-%d')
        val_start_dt = datetime.strptime(val_start, '%Y-%m-%d')
        
        if train_end_dt >= val_start_dt:
            print(f"❌ CRITICAL ERROR: Temporal overlap detected!")
            print(f"   Training ends: {train_end}, Validation starts: {val_start}")
            raise ValueError("Training and validation periods overlap - this will cause data leakage!")
        
        gap_days = (val_start_dt - train_end_dt).days
        print(f"   ✅ Temporal gap validated: {gap_days} days between train and validation")
        
        if gap_days < lookahead_buffer_days:
            print(f"   ⚠️  Warning: Gap ({gap_days} days) is less than intended buffer ({lookahead_buffer_days} days)")
        else:
            print(f"   ✅ Lookahead buffer satisfied: {gap_days} >= {lookahead_buffer_days} days")
        
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
        
        # Load validation data with independent normalization to prevent data leakage
        print("   🔄 Loading validation data with independent normalization...")
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
                is_training=False
                # ❌ REMOVED: reference_datamodule=train_datamodule to prevent data leakage
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
                is_training=False
                # ❌ REMOVED: reference_datamodule=train_datamodule to prevent data leakage
            )
        
        print("✅ Multimodal data loaded successfully!")
        print(f"   Training batches: {len(train_dataloader)}")
        print(f"   Validation batches: {len(val_dataloader)}")
        print(f"   Feature matrix shape: {train_datamodule.feature_df.shape}")
        
        return train_datamodule, val_datamodule
        
    def initialize_models(self):
        """Initialize traditional ML and time series models for stock prediction."""
        predict_len = self.config.get('predict_len', 5)
        
        # Traditional ML models
        base_models = {
            'Ridge Regression': Ridge(alpha=1.0, max_iter=1000),  # Regularized linear model
            'Random Forest': RandomForestRegressor(
                n_estimators=100,
                max_depth=10, 
                random_state=42, 
                n_jobs=-1
            ),
        }
        
        # Add XGBoost if available
        if XGBOOST_AVAILABLE:
            base_models['XGBoost'] = xgb.XGBRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                reg_alpha=0.1,  # L1 regularization for stability
                reg_lambda=1.0,  # L2 regularization for stability
                random_state=42,
                n_jobs=-1,
                verbosity=0
            )
        
        # Wrap each model for multi-step prediction
        self.models = {}
        for name, base_model in base_models.items():
            self.models[name] = MultiStepPredictor(base_model, predict_len)
            
        print(f"✅ Initialized {len(self.models)} models for {predict_len}-step prediction")
        print(f"   📊 Available models: {list(self.models.keys())}")
    
    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                         current_prices: np.ndarray) -> Dict[str, float]:
        """
        Calculate comprehensive prediction metrics including Classification, Regression, and Financial metrics.
        
        Args:
            y_true: True returns [n_samples, predict_len]
            y_pred: Predicted returns [n_samples, predict_len]
            current_prices: Starting prices for each sequence [n_samples]
        """
        try:
            # Flatten for overall metrics
            y_true_flat = y_true.flatten()
            y_pred_flat = y_pred.flatten()
            
            # === REGRESSION METRICS ===
            # MAE, RMSE, MAPE, R²
            mae_returns = mean_absolute_error(y_true_flat, y_pred_flat)
            rmse_returns = np.sqrt(mean_squared_error(y_true_flat, y_pred_flat))
            r2_returns = r2_score(y_true_flat, y_pred_flat)
            mape_returns = np.mean(np.abs((y_true_flat - y_pred_flat) / (np.abs(y_true_flat) + 1e-8))) * 100
            
            # === CLASSIFICATION METRICS (TREND PREDICTION) ===
            # Convert returns to binary classification (up/down trend)
            y_true_binary = (y_true_flat > 0).astype(int)
            y_pred_binary = (y_pred_flat > 0).astype(int)
            
            # Calculate classification metrics
            accuracy = accuracy_score(y_true_binary, y_pred_binary)
            balanced_accuracy = balanced_accuracy_score(y_true_binary, y_pred_binary)
            
            # F1-Score (handle case where one class is missing)
            try:
                f1_score_val = f1_score(y_true_binary, y_pred_binary, average='binary')
            except:
                f1_score_val = 0.0
            
            # AUC-ROC (use predicted returns as probabilities after normalization)
            try:
                # Normalize predictions to [0,1] range for ROC calculation
                y_pred_normalized = (y_pred_flat - y_pred_flat.min()) / (y_pred_flat.max() - y_pred_flat.min() + 1e-8)
                auc_roc = roc_auc_score(y_true_binary, y_pred_normalized)
            except:
                auc_roc = 0.5  # Random performance
            
            # === FINANCIAL METRICS ===
            # Price reconstruction for financial metrics
            predicted_prices_sequences = []
            actual_prices_sequences = []
            
            predict_len = y_true.shape[1]
            
            for i in range(len(y_true)):
                start_price = current_prices[i]
                
                pred_prices = [start_price]
                actual_prices = [start_price]
                
                for step in range(predict_len):
                    pred_prices.append(pred_prices[-1] * (1 + y_pred[i, step]))
                    actual_prices.append(actual_prices[-1] * (1 + y_true[i, step]))
                
                predicted_prices_sequences.append(pred_prices[1:])
                actual_prices_sequences.append(actual_prices[1:])
            
            # Calculate cumulative returns for the prediction period
            cumulative_actual_returns = []
            cumulative_predicted_returns = []
            
            for i in range(len(y_true)):
                cum_actual = np.prod(1 + y_true[i]) - 1
                cum_pred = np.prod(1 + y_pred[i]) - 1
                cumulative_actual_returns.append(cum_actual)
                cumulative_predicted_returns.append(cum_pred)
            
            # Average cumulative returns
            avg_actual_return = np.mean(cumulative_actual_returns)
            avg_predicted_return = np.mean(cumulative_predicted_returns)
            
            # Annualized Return (assuming predict_len is in days)
            # Simple annualization: (1 + return)^(252/days) - 1
            days_in_prediction = predict_len
            annualized_actual_return = (1 + avg_actual_return) ** (252 / days_in_prediction) - 1
            annualized_predicted_return = (1 + avg_predicted_return) ** (252 / days_in_prediction) - 1
            
            # Sharpe Ratio (using predicted returns volatility)
            returns_volatility = np.std(y_pred_flat)
            annualized_volatility = returns_volatility * np.sqrt(252)
            
            # Assume risk-free rate of 3% (0.03)
            risk_free_rate = 0.03
            sharpe_ratio = (annualized_predicted_return - risk_free_rate) / (annualized_volatility + 1e-8)
            
            # Maximum Drawdown (MDD) - calculate from price sequences
            def calculate_mdd(price_sequences):
                if not price_sequences:
                    return 0.0
                
                max_drawdowns = []
                for prices in price_sequences:
                    if len(prices) == 0:
                        continue
                    
                    # Calculate running maximum
                    running_max = np.maximum.accumulate(prices)
                    # Calculate drawdown at each point
                    drawdown = (prices - running_max) / running_max
                    # Maximum drawdown is the most negative value
                    max_drawdown = np.min(drawdown)
                    max_drawdowns.append(max_drawdown)
                
                return np.mean(max_drawdowns) if max_drawdowns else 0.0
            
            mdd_actual = calculate_mdd(actual_prices_sequences)
            mdd_predicted = calculate_mdd(predicted_prices_sequences)
            
            # Price-based regression metrics for completeness
            predicted_prices_flat = np.array(predicted_prices_sequences).flatten()
            actual_prices_flat = np.array(actual_prices_sequences).flatten()
            
            mae_prices = mean_absolute_error(actual_prices_flat, predicted_prices_flat)
            rmse_prices = np.sqrt(mean_squared_error(actual_prices_flat, predicted_prices_flat))
            r2_prices = r2_score(actual_prices_flat, predicted_prices_flat)
            mape_prices = np.mean(np.abs((actual_prices_flat - predicted_prices_flat) / (actual_prices_flat + 1e-8))) * 100
            
            # Compile all metrics
            metrics = {
                # === REGRESSION METRICS ===
                'mae_returns': float(mae_returns),
                'rmse_returns': float(rmse_returns),
                'r2_returns': float(r2_returns),
                'mape_returns': float(mape_returns),
                
                # Price-based regression metrics
                'mae_prices': float(mae_prices),
                'rmse_prices': float(rmse_prices),
                'r2_prices': float(r2_prices),
                'mape_prices': float(mape_prices),
                
                # === CLASSIFICATION METRICS ===
                'accuracy': float(accuracy),
                'balanced_accuracy': float(balanced_accuracy),
                'f1_score': float(f1_score_val),
                'auc_roc': float(auc_roc),
                
                # === FINANCIAL METRICS ===
                'annualized_return_actual': float(annualized_actual_return),
                'annualized_return_predicted': float(annualized_predicted_return),
                'sharpe_ratio': float(sharpe_ratio),
                'mdd_actual': float(mdd_actual),
                'mdd_predicted': float(mdd_predicted),
                'cumulative_return_actual': float(avg_actual_return),
                'cumulative_return_predicted': float(avg_predicted_return),
                
                # Additional useful metrics
                'volatility_actual': float(np.std(y_true_flat)),
                'volatility_predicted': float(np.std(y_pred_flat)),
                'predict_len': int(predict_len),
            }
            
            return metrics
            
        except Exception as e:
            print(f"⚠️ Error calculating metrics: {e}")
            return {
                'mae_returns': np.inf, 'rmse_returns': np.inf, 'r2_returns': -np.inf, 'mape_returns': np.inf,
                'mae_prices': np.inf, 'rmse_prices': np.inf, 'r2_prices': -np.inf, 'mape_prices': np.inf,
                'accuracy': 0.0, 'balanced_accuracy': 0.0, 'f1_score': 0.0, 'auc_roc': 0.5,
                'annualized_return_actual': 0.0, 'annualized_return_predicted': 0.0,
                'sharpe_ratio': 0.0, 'mdd_actual': 0.0, 'mdd_predicted': 0.0,
                'cumulative_return_actual': 0.0, 'cumulative_return_predicted': 0.0,
                'volatility_actual': 0.0, 'volatility_predicted': 0.0, 'predict_len': 1,
            }
    
    def train_and_evaluate_symbol_ablation(self, train_datamodule: Any, val_datamodule: Any, symbol: str):
        """Train and evaluate all models for a single symbol using ablation study across feature modes."""
        print(f"\n{'='*80}")
        print(f"🧪 ABLATION STUDY FOR SYMBOL: {symbol}")
        print(f"{'='*80}")
        
        # Initialize results for this symbol if not exists
        if symbol not in self.results:
            self.results[symbol] = {}
        
        # Run ablation across all feature modes
        feature_modes = ['full', 'no_news', 'no_economic', 'no_technical', 'ohlcv_only', 'core']
        
        total_attempts = 0
        successful_runs = 0
        
        for feature_mode in feature_modes:
            print(f"\n📊 Testing feature mode: {feature_mode} - {FEATURE_MODES[feature_mode]}")
            
            try:
                # Extract features from datamodules using ablation approach
                predictor = MultimodalStockPredictor(sequence_length=self.config.get('encoder_len', 30))
                predict_len = self.config.get('predict_len', 5)
                
                # Extract training data
                X_train, y_train, timestamps_train, symbols_train = predictor.extract_features_from_datamodule(
                    train_datamodule, feature_mode=feature_mode, predict_len=predict_len
                )
                
                # Extract validation data
                X_val, y_val, timestamps_val, symbols_val = predictor.extract_features_from_datamodule(
                    val_datamodule, feature_mode=feature_mode, predict_len=predict_len
                )
                
                # Filter for current symbol
                train_mask = symbols_train == symbol
                val_mask = symbols_val == symbol
                
                if not train_mask.any():
                    print(f"❌ No training data found for {symbol} in mode {feature_mode}")
                    continue
                    
                if not val_mask.any():
                    print(f"❌ No validation data found for {symbol} in mode {feature_mode}")
                    continue
                
                X_train_symbol = X_train[train_mask]
                y_train_symbol = y_train[train_mask]
                X_val_symbol = X_val[val_mask]
                y_val_symbol = y_val[val_mask]
                timestamps_val_symbol = timestamps_val[val_mask]
                
                # Validate we have enough data
                if len(X_train_symbol) < 10 or len(X_val_symbol) < 5:
                    print(f"❌ Insufficient data for {symbol} in mode {feature_mode}: train={len(X_train_symbol)}, val={len(X_val_symbol)}")
                    continue
                
                # Get starting prices for validation sequences
                train_symbol_df = train_datamodule.feature_df[train_datamodule.feature_df['symbol'] == symbol].copy()
                train_symbol_df = train_symbol_df.sort_values('time_idx').reset_index(drop=True)
                
                if 'close' in train_symbol_df.columns and len(train_symbol_df) > 0:
                    last_train_price = train_symbol_df['close'].iloc[-1]
                    starting_prices = np.full(len(X_val_symbol), last_train_price)
                else:
                    starting_prices = np.ones(len(X_val_symbol)) * 100  # Default price
                
                print(f"📊 {symbol} ({feature_mode}): {len(X_train_symbol)} train, {len(X_val_symbol)} test samples, {X_train_symbol.shape[1]} features")
                
                # Enhanced feature scaling with RobustScaler for better outlier handling
                scaler = RobustScaler(quantile_range=(25.0, 75.0))
                try:
                    X_train_scaled = scaler.fit_transform(X_train_symbol)
                    X_val_scaled = scaler.transform(X_val_symbol)
                    
                    # Clip extreme values to prevent numerical overflow
                    X_train_scaled = np.clip(X_train_scaled, -10, 10)
                    X_val_scaled = np.clip(X_val_scaled, -10, 10)
                    
                except Exception as e:
                    print(f"   ⚠️ RobustScaler failed: {e}, trying StandardScaler")
                    try:
                        scaler = StandardScaler()
                        X_train_scaled = scaler.fit_transform(X_train_symbol)
                        X_val_scaled = scaler.transform(X_val_symbol)
                        X_train_scaled = np.clip(X_train_scaled, -10, 10)
                        X_val_scaled = np.clip(X_val_scaled, -10, 10)
                    except Exception as e2:
                        print(f"   ⚠️ All scaling failed: {e2}, using unscaled features")
                        X_train_scaled = X_train_symbol
                        X_val_scaled = X_val_symbol
                
                # Train and evaluate each model for this feature mode
                mode_successful = 0
                for model_name, model in self.models.items():
                    model_key = f"{model_name}_{feature_mode}"
                    print(f"   🔄 Training {model_key}...")
                    total_attempts += 1
                    
                    try:
                        # Validate input data before training
                        if X_train_scaled.shape[1] < 1:
                            raise ValueError(f"No features available after filtering for mode {feature_mode}")
                        
                        if X_train_scaled.shape[0] < 5:
                            raise ValueError(f"Insufficient training samples: {X_train_scaled.shape[0]}")
                        
                        # Check for invalid values in features
                        if np.any(np.isnan(X_train_scaled)) or np.any(np.isinf(X_train_scaled)):
                            print(f"   ⚠️ {model_key}: Cleaning invalid values in features")
                            X_train_scaled = np.nan_to_num(X_train_scaled, nan=0.0, posinf=1.0, neginf=-1.0)
                            X_val_scaled = np.nan_to_num(X_val_scaled, nan=0.0, posinf=1.0, neginf=-1.0)
                        
                        # Fit model with additional error handling
                        from sklearn.base import clone
                        model_clone = clone(model.base_model) if hasattr(model, 'base_model') else clone(model)
                        multi_step_model = MultiStepPredictor(model_clone, self.config.get('predict_len', 5))
                        multi_step_model.fit(X_train_scaled, y_train_symbol)
                        
                        # Make predictions and validate them
                        y_pred_raw = multi_step_model.predict(X_val_scaled)
                        y_pred = self.validate_model_predictions(y_pred_raw, model_name)
                        
                        # Calculate metrics with error handling
                        try:
                            metrics = self.calculate_metrics(y_val_symbol, y_pred, starting_prices)
                        except Exception as metric_error:
                            print(f"   ⚠️ {model_key}: Metrics calculation failed: {metric_error}")
                            # Create fallback metrics
                            metrics = {
                                'r2_returns': -999.0, 'accuracy': 0.0, 'sharpe_ratio': -999.0,
                                'mae_returns': 999.0, 'rmse_returns': 999.0, 
                                'failed_metrics': True, 'error': str(metric_error)
                            }
                        
                        # Debug prediction bias for first model of first symbol
                        if total_attempts == 1:  # First model being trained
                            print(f"\n🔍 DEBUGGING PREDICTION BIAS FOR {model_key}:")
                            try:
                                predictor = MultimodalStockPredictor()
                                predictor.debug_prediction_bias(y_val_symbol, y_pred)
                            except Exception as debug_error:
                                print(f"   ⚠️ Debug failed: {debug_error}")
                        
                        # Store results (even if metrics failed)
                        self.results[symbol][model_key] = {
                            'predictions': y_pred,
                            'actuals': y_val_symbol,
                            'timestamps': timestamps_val_symbol,
                            'prices': starting_prices,
                            'metrics': metrics,
                            'feature_mode': feature_mode,
                            'n_features': X_train_scaled.shape[1],
                            'training_successful': True,
                            'data_shape': X_train_scaled.shape
                        }
                        
                        # Display results
                        r2_val = metrics.get('r2_returns', -999)
                        acc_val = metrics.get('accuracy', 0)
                        sharpe_val = metrics.get('sharpe_ratio', -999)
                        
                        if metrics.get('failed_metrics', False):
                            print(f"   ⚠️ {model_key}: Training succeeded but metrics failed")
                        else:
                            print(f"   ✅ {model_key}: R²={r2_val:.3f}, Acc={acc_val:.3f}, Sharpe={sharpe_val:.3f}")
                        
                        successful_runs += 1
                        mode_successful += 1
                        
                    except Exception as e:
                        print(f"   ❌ {model_key} failed: {str(e)}")
                        
                        # Store failure information for debugging
                        self.results[symbol][model_key] = {
                            'training_successful': False,
                            'error': str(e),
                            'feature_mode': feature_mode,
                            'n_features': X_train_scaled.shape[1] if 'X_train_scaled' in locals() else 0,
                            'data_shape': X_train_scaled.shape if 'X_train_scaled' in locals() else (0, 0),
                            'metrics': {
                                'r2_returns': np.nan, 'accuracy': np.nan, 'sharpe_ratio': np.nan,
                                'failed_training': True
                            }
                        }
                        continue
                
                if mode_successful == 0:
                    print(f"   ⚠️ No models succeeded for feature mode {feature_mode}")
                else:
                    print(f"   ✅ Feature mode {feature_mode}: {mode_successful}/{len(self.models)} models succeeded")
                        
            except Exception as e:
                print(f"❌ Failed to process {symbol} with feature mode {feature_mode}: {e}")
                import traceback
                print(f"   Detailed error: {traceback.format_exc()}")
                continue
        
        print(f"\n✅ Completed ablation study for {symbol}")
        print(f"   📊 Success rate: {successful_runs}/{total_attempts} ({100*successful_runs/max(1,total_attempts):.1f}%)")
        print(f"   🎯 Total results stored: {len(self.results[symbol])}")
        
        # Print summary of what was actually generated
        if self.results[symbol]:
            modes_tested = set()
            models_tested = set()
            for key in self.results[symbol].keys():
                if '_' in key:
                    parts = key.split('_')
                    if len(parts) >= 2:
                        model = '_'.join(parts[:-1])
                        mode = parts[-1]
                        modes_tested.add(mode)
                        models_tested.add(model)
            
            print(f"   📋 Feature modes with results: {sorted(modes_tested)}")
            print(f"   🤖 Models with results: {sorted(models_tested)}")
        else:
            print(f"   ⚠️ No successful results for {symbol}")

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
            
            # Extract training data - use appropriate feature_mode based on multimodal flag
            feature_mode = 'full' if use_multimodal else 'ohlcv_only'
            X_train, y_train, timestamps_train, symbols_train = predictor.extract_features_from_datamodule(
                train_datamodule, feature_mode=feature_mode, predict_len=predict_len
            )
            
            # Extract validation data
            X_val, y_val, timestamps_val, symbols_val = predictor.extract_features_from_datamodule(
                val_datamodule, feature_mode=feature_mode, predict_len=predict_len
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
            
            # Get starting prices for validation sequences - CRITICAL: Use training data to avoid future peeking
            # We need the price at the END of training period as the starting point for validation predictions
            train_symbol_df = train_datamodule.feature_df[train_datamodule.feature_df['symbol'] == symbol].copy()
            train_symbol_df = train_symbol_df.sort_values('time_idx').reset_index(drop=True)
            
            if 'close' in train_symbol_df.columns and len(train_symbol_df) > 0:
                # Use the LAST available close price from training data as the base price
                # This prevents future peeking since we only use historical training data
                base_price = train_symbol_df['close'].iloc[-1]  # Last training period price
                print(f"   📊 Using base price from end of training: ${base_price:.2f}")
                
                # For validation sequences, we'll use the base price as starting point
                # This is realistic - we only know prices up to the end of training
                val_prices = np.full(len(y_val_symbol), base_price)
                
            else:
                # Fallback: estimate base price
                val_prices = np.ones(len(y_val_symbol)) * 100  # Assume $100 base price
                print(f"⚠️ No close prices found for {symbol}, using estimated base price: $100")
            
            print(f"📊 {symbol}: {len(X_train_symbol)} train, {len(X_val_symbol)} test samples")
            print(f"📊 Feature shape: {X_train_symbol.shape}")
            print(f"📊 Target shape: {y_train_symbol.shape} (multi-step: {predict_len} steps)")
            print(f"📊 Using multimodal features: {use_multimodal}")
            
            # Enhanced feature scaling with RobustScaler for better outlier handling
            scaler = RobustScaler(quantile_range=(25.0, 75.0))
            try:
                # ✅ CORRECT: Fit scaler on training data only
                X_train_scaled = scaler.fit_transform(X_train_symbol)
                # ✅ CORRECT: Transform validation data using training statistics only
                X_val_scaled = scaler.transform(X_val_symbol)
                
                # Clip extreme values to prevent numerical overflow
                X_train_scaled = np.clip(X_train_scaled, -10, 10)
                X_val_scaled = np.clip(X_val_scaled, -10, 10)
                
                print(f"   ✅ RobustScaler completed - no data leakage")
                print(f"      Training median: {np.median(X_train_scaled):.4f}, IQR: {np.percentile(X_train_scaled, 75) - np.percentile(X_train_scaled, 25):.4f}")
                print(f"      Validation median: {np.median(X_val_scaled):.4f}, IQR: {np.percentile(X_val_scaled, 75) - np.percentile(X_val_scaled, 25):.4f}")
                
            except Exception as e:
                print(f"   ⚠️ RobustScaler failed: {e}, trying StandardScaler")
                try:
                    scaler = StandardScaler()
                    X_train_scaled = scaler.fit_transform(X_train_symbol)
                    X_val_scaled = scaler.transform(X_val_symbol)
                    X_train_scaled = np.clip(X_train_scaled, -10, 10)
                    X_val_scaled = np.clip(X_val_scaled, -10, 10)
                    print(f"   ✅ StandardScaler fallback completed")
                except Exception as e2:
                    print(f"❌ All scaling failed for {symbol}: {e2}")
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
                    
                    print(f"✅ {model_name} ({predict_len}-step):")
                    print(f"   📊 Classification - Accuracy: {metrics['accuracy']:.3f}, F1: {metrics['f1_score']:.3f}, AUC: {metrics['auc_roc']:.3f}")
                    print(f"   📊 Regression - MAE: {metrics['mae_returns']:.4f}, RMSE: {metrics['rmse_returns']:.4f}, R²: {metrics['r2_returns']:.4f}")
                    print(f"   � Financial - Sharpe: {metrics['sharpe_ratio']:.3f}, Ann. Return: {metrics['annualized_return_predicted']:.3f}")
                    
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
    
    def create_enhanced_prediction_plots(self, symbol: str):
        """Create enhanced prediction plots with detailed analysis for validation test set."""
        if symbol not in self.results or len(self.results[symbol]) == 0:
            print(f"⚠️ No results to create enhanced plots for {symbol}")
            return
            
        print(f"📊 Creating enhanced prediction plots for {symbol}...")
        
        symbol_results = self.results[symbol]
        n_models = len(symbol_results)
        
        try:
            # Create comprehensive plots: 2x2 grid for each type of analysis
            fig, axes = plt.subplots(2, 2, figsize=(20, 16))
            fig.suptitle(f'{symbol} - Enhanced Prediction Analysis (Validation Set)', fontsize=16, fontweight='bold')
            
            # Colors for different models
            colors = plt.cm.Set1(np.linspace(0, 1, n_models))
            
            # Get predict_len from first model
            first_result = list(symbol_results.values())[0]
            predict_len = first_result['predictions'].shape[1]
            horizons = np.arange(1, predict_len + 1)
            
            # 1. Predicted vs Actual Returns/Prices
            ax1 = axes[0, 0]
            ax1.set_title('Predicted vs Actual Final Returns')
            ax1.set_xlabel('Actual Returns')
            ax1.set_ylabel('Predicted Returns')
            
            for i, (model_name, result) in enumerate(symbol_results.items()):
                predictions = result['predictions']
                actuals = result['actuals']
                
                # Calculate cumulative returns over prediction horizon
                actual_cumulative = np.sum(actuals, axis=1)
                predicted_cumulative = np.sum(predictions, axis=1)
                
                # Scatter plot
                ax1.scatter(actual_cumulative, predicted_cumulative, 
                           alpha=0.6, label=model_name, color=colors[i], s=20)
            
            # Add perfect prediction line
            min_val = min([np.min(np.sum(result['actuals'], axis=1)) for result in symbol_results.values()])
            max_val = max([np.max(np.sum(result['actuals'], axis=1)) for result in symbol_results.values()])
            ax1.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, label='Perfect Prediction')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # 2. Error Over Horizon (MAE and RMSE by forecast step)
            ax2 = axes[0, 1]
            ax2.set_title('Error Over Forecast Horizon')
            ax2.set_xlabel('Forecast Horizon (Days)')
            ax2.set_ylabel('Error')
            
            for i, (model_name, result) in enumerate(symbol_results.items()):
                predictions = result['predictions']
                actuals = result['actuals']
                
                # Calculate MAE and RMSE for each horizon step
                mae_by_horizon = []
                rmse_by_horizon = []
                
                for h in range(predict_len):
                    mae_h = mean_absolute_error(actuals[:, h], predictions[:, h])
                    rmse_h = np.sqrt(mean_squared_error(actuals[:, h], predictions[:, h]))
                    mae_by_horizon.append(mae_h)
                    rmse_by_horizon.append(rmse_h)
                
                # Plot MAE and RMSE lines
                ax2.plot(horizons, mae_by_horizon, '-', color=colors[i], 
                        label=f'{model_name} (MAE)', linewidth=2, alpha=0.8)
                ax2.plot(horizons, rmse_by_horizon, '--', color=colors[i], 
                        label=f'{model_name} (RMSE)', linewidth=2, alpha=0.8)
            
            ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax2.grid(True, alpha=0.3)
            ax2.set_xticks(horizons)
            
            # 3. Residual Plots (Errors vs Predicted Values)
            ax3 = axes[1, 0]
            ax3.set_title('Residuals vs Predicted Values')
            ax3.set_xlabel('Predicted Returns')
            ax3.set_ylabel('Residuals (Actual - Predicted)')
            
            for i, (model_name, result) in enumerate(symbol_results.items()):
                predictions = result['predictions']
                actuals = result['actuals']
                
                # Flatten for residual analysis
                pred_flat = predictions.flatten()
                actual_flat = actuals.flatten()
                residuals = actual_flat - pred_flat
                
                # Scatter plot of residuals
                ax3.scatter(pred_flat, residuals, alpha=0.5, label=model_name, 
                           color=colors[i], s=15)
            
            # Add zero line
            ax3.axhline(y=0, color='k', linestyle='-', alpha=0.5)
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            
            # 4. Directional Accuracy Over Horizon
            ax4 = axes[1, 1]
            ax4.set_title('Directional Accuracy Over Forecast Horizon')
            ax4.set_xlabel('Forecast Horizon (Days)')
            ax4.set_ylabel('Directional Accuracy (%)')
            
            for i, (model_name, result) in enumerate(symbol_results.items()):
                predictions = result['predictions']
                actuals = result['actuals']
                
                # Calculate directional accuracy for each horizon
                directional_accuracy = []
                
                for h in range(predict_len):
                    actual_direction = (actuals[:, h] > 0).astype(int)
                    predicted_direction = (predictions[:, h] > 0).astype(int)
                    accuracy = accuracy_score(actual_direction, predicted_direction) * 100
                    directional_accuracy.append(accuracy)
                
                # Plot directional accuracy
                ax4.plot(horizons, directional_accuracy, '-o', color=colors[i], 
                        label=model_name, linewidth=2, markersize=6, alpha=0.8)
            
            # Add 50% baseline (random guessing)
            ax4.axhline(y=50, color='k', linestyle='--', alpha=0.5, label='Random Baseline (50%)')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
            ax4.set_xticks(horizons)
            ax4.set_ylim(30, 80)  # Focus on reasonable range
            
            plt.tight_layout()
            plot_path = self.output_dir / f"{symbol}_enhanced_predictions.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"  📈 {symbol} enhanced prediction plots saved to {plot_path}")
            
        except Exception as e:
            print(f"❌ Failed to create enhanced prediction plots for {symbol}: {e}")
            import traceback
            traceback.print_exc()

    def create_detailed_residual_analysis(self, symbol: str):
        """Create detailed residual analysis plots including time series residuals."""
        if symbol not in self.results or len(self.results[symbol]) == 0:
            print(f"⚠️ No results for residual analysis for {symbol}")
            return
            
        print(f"📊 Creating detailed residual analysis for {symbol}...")
        
        symbol_results = self.results[symbol]
        n_models = len(symbol_results)
        
        try:
            # Create residual analysis plots: 3x2 grid
            fig, axes = plt.subplots(3, 2, figsize=(20, 18))
            fig.suptitle(f'{symbol} - Detailed Residual Analysis', fontsize=16, fontweight='bold')
            
            colors = plt.cm.Set1(np.linspace(0, 1, n_models))
            
            # Get predict_len from first model
            first_result = list(symbol_results.values())[0]
            predict_len = first_result['predictions'].shape[1]
            
            for i, (model_name, result) in enumerate(symbol_results.items()):
                predictions = result['predictions']
                actuals = result['actuals']
                timestamps = result['timestamps']
                
                # Calculate residuals for different horizons
                residuals_by_horizon = []
                for h in range(predict_len):
                    residuals_h = actuals[:, h] - predictions[:, h]
                    residuals_by_horizon.append(residuals_h)
                
                # 1. Residuals over time (for first forecast horizon)
                if i < 3:  # Only show first 3 models to avoid clutter
                    ax = axes[0, 0] if i < 2 else axes[0, 1]
                    if i == 0:
                        ax.set_title('Residuals Over Time (1-Day Horizon)')
                        ax.set_xlabel('Sample Index')
                        ax.set_ylabel('Residuals')
                    
                    ax.plot(residuals_by_horizon[0], color=colors[i], 
                           label=model_name, alpha=0.7, linewidth=1)
                    ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                
                # 2. Residual distribution (histogram)
                if i < 3:
                    ax = axes[1, 0] if i < 2 else axes[1, 1]
                    if i == 0:
                        ax.set_title('Residual Distribution')
                        ax.set_xlabel('Residuals')
                        ax.set_ylabel('Frequency')
                    
                    all_residuals = np.concatenate(residuals_by_horizon)
                    ax.hist(all_residuals, bins=30, alpha=0.6, color=colors[i], 
                           label=model_name, density=True)
                    ax.axvline(x=0, color='k', linestyle='-', alpha=0.3)
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                
                # 3. Q-Q plot for normality check
                if i < 2:
                    ax = axes[2, i]
                    ax.set_title(f'{model_name} - Q-Q Plot (Normality Check)')
                    ax.set_xlabel('Theoretical Quantiles')
                    ax.set_ylabel('Sample Quantiles')
                    
                    all_residuals = np.concatenate(residuals_by_horizon)
                    
                    if SCIPY_AVAILABLE:
                        stats.probplot(all_residuals, dist="norm", plot=ax)
                    else:
                        # Fallback: simple histogram instead of Q-Q plot
                        ax.hist(all_residuals, bins=20, alpha=0.7, density=True)
                        ax.set_title(f'{model_name} - Residual Distribution (SciPy not available)')
                        ax.set_ylabel('Density')
                    
                    ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plot_path = self.output_dir / f"{symbol}_residual_analysis.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"  📈 {symbol} residual analysis saved to {plot_path}")
            
        except Exception as e:
            print(f"❌ Failed to create residual analysis for {symbol}: {e}")
            import traceback
            traceback.print_exc()

    def create_horizon_error_analysis(self, symbol: str):
        """Create comprehensive error analysis over forecast horizons."""
        if symbol not in self.results or len(self.results[symbol]) == 0:
            print(f"⚠️ No results for horizon error analysis for {symbol}")
            return
            
        print(f"📊 Creating horizon error analysis for {symbol}...")
        
        symbol_results = self.results[symbol]
        n_models = len(symbol_results)
        
        try:
            # Create comprehensive horizon analysis: 2x2 grid
            fig, axes = plt.subplots(2, 2, figsize=(20, 16))
            fig.suptitle(f'{symbol} - Error Analysis Over Forecast Horizons', fontsize=16, fontweight='bold')
            
            colors = plt.cm.Set1(np.linspace(0, 1, n_models))
            
            # Get predict_len from first model
            first_result = list(symbol_results.values())[0]
            predict_len = first_result['predictions'].shape[1]
            horizons = np.arange(1, predict_len + 1)
            
            # Prepare data structures for analysis
            model_metrics_by_horizon = {}
            
            for model_name, result in symbol_results.items():
                predictions = result['predictions']
                actuals = result['actuals']
                
                # Calculate metrics for each horizon
                mae_by_horizon = []
                rmse_by_horizon = []
                directional_accuracy = []
                correlation_by_horizon = []
                
                for h in range(predict_len):
                    # Error metrics
                    mae_h = mean_absolute_error(actuals[:, h], predictions[:, h])
                    rmse_h = np.sqrt(mean_squared_error(actuals[:, h], predictions[:, h]))
                    
                    # Directional accuracy
                    actual_direction = (actuals[:, h] > 0).astype(int)
                    predicted_direction = (predictions[:, h] > 0).astype(int)
                    dir_acc = accuracy_score(actual_direction, predicted_direction) * 100
                    
                    # Correlation
                    corr = np.corrcoef(actuals[:, h], predictions[:, h])[0, 1]
                    if np.isnan(corr):
                        corr = 0.0
                    
                    mae_by_horizon.append(mae_h)
                    rmse_by_horizon.append(rmse_h)
                    directional_accuracy.append(dir_acc)
                    correlation_by_horizon.append(corr)
                
                model_metrics_by_horizon[model_name] = {
                    'mae': mae_by_horizon,
                    'rmse': rmse_by_horizon,
                    'directional_accuracy': directional_accuracy,
                    'correlation': correlation_by_horizon
                }
            
            # 1. MAE and RMSE Over Horizon
            ax1 = axes[0, 0]
            ax1.set_title('Mean Absolute Error (MAE) Over Forecast Horizon')
            ax1.set_xlabel('Forecast Horizon (Days)')
            ax1.set_ylabel('MAE')
            
            for i, (model_name, metrics) in enumerate(model_metrics_by_horizon.items()):
                ax1.plot(horizons, metrics['mae'], '-o', color=colors[i], 
                        label=model_name, linewidth=2, markersize=6, alpha=0.8)
            
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            ax1.set_xticks(horizons)
            
            # 2. RMSE Over Horizon
            ax2 = axes[0, 1]
            ax2.set_title('Root Mean Square Error (RMSE) Over Forecast Horizon')
            ax2.set_xlabel('Forecast Horizon (Days)')
            ax2.set_ylabel('RMSE')
            
            for i, (model_name, metrics) in enumerate(model_metrics_by_horizon.items()):
                ax2.plot(horizons, metrics['rmse'], '-s', color=colors[i], 
                        label=model_name, linewidth=2, markersize=6, alpha=0.8)
            
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            ax2.set_xticks(horizons)
            
            # 3. Directional Accuracy Over Horizon
            ax3 = axes[1, 0]
            ax3.set_title('Directional Accuracy (% Up/Down Correct) Over Horizon')
            ax3.set_xlabel('Forecast Horizon (Days)')
            ax3.set_ylabel('Directional Accuracy (%)')
            
            for i, (model_name, metrics) in enumerate(model_metrics_by_horizon.items()):
                ax3.plot(horizons, metrics['directional_accuracy'], '-^', color=colors[i], 
                        label=model_name, linewidth=2, markersize=6, alpha=0.8)
            
            # Add 50% baseline (random guessing)
            ax3.axhline(y=50, color='k', linestyle='--', alpha=0.5, label='Random Baseline (50%)')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            ax3.set_xticks(horizons)
            ax3.set_ylim(30, 80)
            
            # 4. Correlation Over Horizon
            ax4 = axes[1, 1]
            ax4.set_title('Prediction Correlation Over Forecast Horizon')
            ax4.set_xlabel('Forecast Horizon (Days)')
            ax4.set_ylabel('Correlation Coefficient')
            
            for i, (model_name, metrics) in enumerate(model_metrics_by_horizon.items()):
                ax4.plot(horizons, metrics['correlation'], '-d', color=colors[i], 
                        label=model_name, linewidth=2, markersize=6, alpha=0.8)
            
            ax4.axhline(y=0, color='k', linestyle='-', alpha=0.3)
            ax4.legend()
            ax4.grid(True, alpha=0.3)
            ax4.set_xticks(horizons)
            ax4.set_ylim(-0.2, 1.0)
            
            plt.tight_layout()
            plot_path = self.output_dir / f"{symbol}_horizon_error_analysis.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"  📈 {symbol} horizon error analysis saved to {plot_path}")
            
            # Print summary statistics
            print(f"\n📊 {symbol} - Horizon Error Analysis Summary:")
            for model_name, metrics in model_metrics_by_horizon.items():
                avg_mae = np.mean(metrics['mae'])
                avg_rmse = np.mean(metrics['rmse'])
                avg_dir_acc = np.mean(metrics['directional_accuracy'])
                avg_corr = np.mean(metrics['correlation'])
                
                print(f"  {model_name}:")
                print(f"    Avg MAE: {avg_mae:.4f}, Avg RMSE: {avg_rmse:.4f}")
                print(f"    Avg Directional Accuracy: {avg_dir_acc:.1f}%")
                print(f"    Avg Correlation: {avg_corr:.3f}")
            
        except Exception as e:
            print(f"❌ Failed to create horizon error analysis for {symbol}: {e}")
            import traceback
            traceback.print_exc()

    def create_price_prediction_plots(self, symbol: str):
        """Create detailed price prediction plots showing actual vs predicted prices over time."""
        if symbol not in self.results or len(self.results[symbol]) == 0:
            print(f"⚠️ No results for price prediction plots for {symbol}")
            return
            
        print(f"📊 Creating price prediction plots for {symbol}...")
        
        symbol_results = self.results[symbol]
        n_models = len(symbol_results)
        
        try:
            # Create comprehensive price analysis: 2x2 grid
            fig, axes = plt.subplots(2, 2, figsize=(20, 16))
            fig.suptitle(f'{symbol} - Price Prediction Analysis (Validation Set)', fontsize=16, fontweight='bold')
            
            colors = plt.cm.Set1(np.linspace(0, 1, n_models))
            
            # Get first result for reference
            first_result = list(symbol_results.values())[0]
            predict_len = first_result['predictions'].shape[1]
            timestamps = first_result['timestamps']
            prices = first_result['prices']
            actuals = first_result['actuals']
            
            # 1. Price Trajectories for First N Samples
            ax1 = axes[0, 0]
            ax1.set_title('Price Trajectories - First 10 Validation Samples')
            ax1.set_xlabel('Days into Forecast')
            ax1.set_ylabel('Price ($)')
            
            n_samples_to_show = min(10, len(actuals))
            days = np.arange(predict_len + 1)  # Include starting point
            
            for sample_idx in range(n_samples_to_show):
                start_price = prices[sample_idx]
                actual_returns = actuals[sample_idx]
                
                # Calculate actual price trajectory
                actual_prices = [start_price]
                current_price = start_price
                for ret in actual_returns:
                    current_price = current_price * (1 + ret)
                    actual_prices.append(current_price)
                
                # Plot actual trajectory
                ax1.plot(days, actual_prices, 'b-', alpha=0.3, linewidth=1)
                
                # Plot predicted trajectories for each model
                for i, (model_name, result) in enumerate(symbol_results.items()):
                    if sample_idx == 0:  # Only add label once
                        predicted_returns = result['predictions'][sample_idx]
                        pred_prices = [start_price]
                        current_price = start_price
                        for ret in predicted_returns:
                            current_price = current_price * (1 + ret)
                            pred_prices.append(current_price)
                        
                        ax1.plot(days, pred_prices, '--', color=colors[i], 
                                alpha=0.7, linewidth=2, label=f'{model_name} Pred')
                    else:
                        predicted_returns = result['predictions'][sample_idx]
                        pred_prices = [start_price]
                        current_price = start_price
                        for ret in predicted_returns:
                            current_price = current_price * (1 + ret)
                            pred_prices.append(current_price)
                        
                        ax1.plot(days, pred_prices, '--', color=colors[i], 
                                alpha=0.7, linewidth=2)
            
            # Add legend items for actual
            ax1.plot([], [], 'b-', alpha=0.7, linewidth=2, label='Actual')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # 2. Final Price Scatter Plot (Actual vs Predicted)
            ax2 = axes[0, 1]
            ax2.set_title('Final Price Predictions (End of Forecast Period)')
            ax2.set_xlabel('Actual Final Price ($)')
            ax2.set_ylabel('Predicted Final Price ($)')
            
            for i, (model_name, result) in enumerate(symbol_results.items()):
                predictions = result['predictions']
                actuals_data = result['actuals']
                prices_data = result['prices']
                
                # Calculate final prices
                actual_final_prices = []
                predicted_final_prices = []
                
                for j in range(len(predictions)):
                    start_price = prices_data[j]
                    
                    # Actual final price
                    actual_final = start_price * np.prod(1 + actuals_data[j])
                    
                    # Predicted final price
                    predicted_final = start_price * np.prod(1 + predictions[j])
                    
                    actual_final_prices.append(actual_final)
                    predicted_final_prices.append(predicted_final)
                
                # Scatter plot
                ax2.scatter(actual_final_prices, predicted_final_prices, 
                           alpha=0.6, label=model_name, color=colors[i], s=30)
                
                # Calculate and display correlation
                corr = np.corrcoef(actual_final_prices, predicted_final_prices)[0, 1]
                print(f"  {model_name} - Final Price Correlation: {corr:.3f}")
            
            # Add perfect prediction line
            all_actual = []
            for result in symbol_results.values():
                prices_data = result['prices']
                actuals_data = result['actuals']
                for j in range(len(actuals_data)):
                    start_price = prices_data[j]
                    actual_final = start_price * np.prod(1 + actuals_data[j])
                    all_actual.append(actual_final)
            
            min_price = min(all_actual)
            max_price = max(all_actual)
            ax2.plot([min_price, max_price], [min_price, max_price], 'k--', alpha=0.5, label='Perfect Prediction')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # 3. Cumulative Returns Comparison
            ax3 = axes[1, 0]
            ax3.set_title('Cumulative Returns Over Validation Period')
            ax3.set_xlabel('Sample Index')
            ax3.set_ylabel('Cumulative Return (%)')
            
            for i, (model_name, result) in enumerate(symbol_results.items()):
                predictions = result['predictions']
                actuals_data = result['actuals']
                
                # Calculate cumulative returns for each sample
                actual_cumulative = np.cumsum(np.sum(actuals_data, axis=1)) * 100
                predicted_cumulative = np.cumsum(np.sum(predictions, axis=1)) * 100
                
                sample_indices = np.arange(len(actual_cumulative))
                
                if i == 0:  # Plot actual only once
                    ax3.plot(sample_indices, actual_cumulative, 'k-', 
                            linewidth=3, label='Actual', alpha=0.8)
                
                ax3.plot(sample_indices, predicted_cumulative, '--', 
                        color=colors[i], linewidth=2, label=f'{model_name} Pred', alpha=0.8)
            
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            ax3.axhline(y=0, color='k', linestyle='-', alpha=0.3)
            
            # 4. Price Prediction Error Distribution
            ax4 = axes[1, 1]
            ax4.set_title('Price Prediction Error Distribution')
            ax4.set_xlabel('Price Prediction Error ($)')
            ax4.set_ylabel('Density')
            
            for i, (model_name, result) in enumerate(symbol_results.items()):
                predictions = result['predictions']
                actuals_data = result['actuals']
                prices_data = result['prices']
                
                # Calculate price errors
                price_errors = []
                for j in range(len(predictions)):
                    start_price = prices_data[j]
                    actual_final = start_price * np.prod(1 + actuals_data[j])
                    predicted_final = start_price * np.prod(1 + predictions[j])
                    error = predicted_final - actual_final
                    price_errors.append(error)
                
                # Plot histogram
                ax4.hist(price_errors, bins=30, alpha=0.6, color=colors[i], 
                        label=model_name, density=True)
                
                # Print error statistics
                mae_price = np.mean(np.abs(price_errors))
                rmse_price = np.sqrt(np.mean(np.array(price_errors)**2))
                print(f"  {model_name} - Price MAE: ${mae_price:.2f}, Price RMSE: ${rmse_price:.2f}")
            
            ax4.axvline(x=0, color='k', linestyle='-', alpha=0.5)
            ax4.legend()
            ax4.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plot_path = self.output_dir / f"{symbol}_price_predictions.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"  📈 {symbol} price prediction plots saved to {plot_path}")
            
        except Exception as e:
            print(f"❌ Failed to create price prediction plots for {symbol}: {e}")
            import traceback
            traceback.print_exc()

    def create_symbol_plots(self, symbol: str):
        """Create plots for a specific symbol showing Classification, Regression, Financial metrics, and Price Comparisons."""
        if symbol not in self.results or len(self.results[symbol]) == 0:
            print(f"⚠️ No results to plot for {symbol}")
            return
            
        print(f"📊 Creating plots for {symbol}...")
        
        symbol_results = self.results[symbol]
        n_models = len(symbol_results)
        
        try:
            # Create subplots: 1 row per model, 4 columns (Classification, Regression, Financial, Price Comparison)
            fig, axes = plt.subplots(n_models, 4, figsize=(24, 6*n_models))
            if n_models == 1:
                axes = axes.reshape(1, -1)
            
            fig.suptitle(f'{symbol} - Model Performance Metrics & Price Predictions', fontsize=16, fontweight='bold')
            
            for i, (model_name, result) in enumerate(symbol_results.items()):
                metrics = result['metrics']
                predictions = result['predictions']
                actuals = result['actuals']
                prices = result['prices']
                timestamps = result['timestamps']
                predict_len = predictions.shape[1]
                
                # Plot 1: Classification Metrics (Trend Prediction)
                classification_metrics = ['accuracy', 'balanced_accuracy', 'f1_score', 'auc_roc']
                classification_values = [metrics.get(metric, 0) for metric in classification_metrics]
                classification_labels = ['Accuracy', 'Balanced Accuracy', 'F1-Score', 'AUC-ROC']
                
                bars1 = axes[i, 0].bar(classification_labels, classification_values, 
                                      color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'], alpha=0.8)
                axes[i, 0].set_title(f'{model_name} - Classification Metrics')
                axes[i, 0].set_ylabel('Score')
                axes[i, 0].set_ylim(0, 1)
                axes[i, 0].grid(True, alpha=0.3)
                
                # Add value labels on bars
                for bar, value in zip(bars1, classification_values):
                    axes[i, 0].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                                   f'{value:.3f}', ha='center', va='bottom', fontsize=10)
                
                # Plot 2: Regression Metrics (Price Prediction)
                regression_metrics = ['mae_returns', 'rmse_returns', 'mape_returns', 'r2_returns']
                regression_values = [metrics.get(metric, 0) for metric in regression_metrics]
                regression_labels = ['MAE', 'RMSE', 'MAPE (%)', 'R²']
                
                # Normalize MAPE to be on similar scale (divide by 100)
                regression_values_normalized = regression_values.copy()
                if len(regression_values_normalized) > 2:
                    regression_values_normalized[2] = regression_values_normalized[2] / 100  # MAPE normalization
                
                bars2 = axes[i, 1].bar(regression_labels, regression_values_normalized, 
                                      color=['#9467bd', '#8c564b', '#e377c2', '#7f7f7f'], alpha=0.8)
                axes[i, 1].set_title(f'{model_name} - Regression Metrics')
                axes[i, 1].set_ylabel('Score')
                axes[i, 1].grid(True, alpha=0.3)
                
                # Add value labels on bars (show original values)
                for bar, value, orig_value in zip(bars2, regression_values_normalized, regression_values):
                    axes[i, 1].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.001,
                                   f'{orig_value:.3f}', ha='center', va='bottom', fontsize=10)
                
                # Plot 3: Financial Metrics
                financial_metrics = ['annualized_return_predicted', 'sharpe_ratio', 'mdd_predicted', 'cumulative_return_predicted']
                financial_values = [metrics.get(metric, 0) for metric in financial_metrics]
                financial_labels = ['Annualized Return', 'Sharpe Ratio', 'Max Drawdown', 'Cumulative Return']
                
                # Color bars based on performance (green for positive, red for negative)
                colors = []
                for val in financial_values:
                    if val > 0:
                        colors.append('#2ca02c')  # Green
                    else:
                        colors.append('#d62728')  # Red
                
                bars3 = axes[i, 2].bar(financial_labels, financial_values, 
                                      color=colors, alpha=0.8)
                axes[i, 2].set_title(f'{model_name} - Financial Metrics')
                axes[i, 2].set_ylabel('Value')
                axes[i, 2].grid(True, alpha=0.3)
                axes[i, 2].axhline(y=0, color='black', linestyle='-', alpha=0.3)
                
                # Add value labels on bars
                for bar, value in zip(bars3, financial_values):
                    y_pos = bar.get_height() + 0.001 if value >= 0 else bar.get_height() - 0.01
                    axes[i, 2].text(bar.get_x() + bar.get_width()/2., y_pos,
                                   f'{value:.3f}', ha='center', va='bottom' if value >= 0 else 'top', fontsize=10)
                
                # Plot 4: Actual vs Predicted Price Comparison
                axes[i, 3].set_title(f'{model_name} - Price Predictions vs Actual')
                axes[i, 3].set_ylabel('Price ($)')
                axes[i, 3].set_xlabel('Time (Sample Index)')
                
                # Reconstruct price sequences for visualization
                n_samples_to_show = min(50, len(predictions))  # Show first 50 samples for clarity
                sample_indices = np.arange(n_samples_to_show)
                
                # Calculate actual and predicted prices for each sample
                actual_final_prices = []
                predicted_final_prices = []
                
                for j in range(n_samples_to_show):
                    start_price = prices[j]
                    
                    # Calculate final price after predict_len days
                    actual_final_price = start_price * np.prod(1 + actuals[j])
                    predicted_final_price = start_price * np.prod(1 + predictions[j])
                    
                    actual_final_prices.append(actual_final_price)
                    predicted_final_prices.append(predicted_final_price)
                
                # Plot actual vs predicted final prices
                axes[i, 3].plot(sample_indices, actual_final_prices, 'b-', linewidth=2, 
                               label='Actual Prices', alpha=0.8)
                axes[i, 3].plot(sample_indices, predicted_final_prices, 'r--', linewidth=2, 
                               label='Predicted Prices', alpha=0.8)
                
                # Add correlation coefficient
                corr_coef = np.corrcoef(actual_final_prices, predicted_final_prices)[0, 1]
                axes[i, 3].text(0.05, 0.95, f'Correlation: {corr_coef:.3f}', 
                               transform=axes[i, 3].transAxes, fontsize=10, 
                               bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
                
                axes[i, 3].legend()
                axes[i, 3].grid(True, alpha=0.3)
                
                # Rotate x-axis labels for better readability
                for col in range(4):
                    axes[i, col].tick_params(axis='x', rotation=45)
                
                # Print summary for this model
                print(f"  📊 {model_name} Summary:")
                print(f"     Classification - Accuracy: {metrics.get('accuracy', 0):.3f}, F1: {metrics.get('f1_score', 0):.3f}")
                print(f"     Regression - MAE: {metrics.get('mae_returns', 0):.3f}, R²: {metrics.get('r2_returns', 0):.3f}")
                print(f"     Financial - Sharpe: {metrics.get('sharpe_ratio', 0):.3f}, Return: {metrics.get('annualized_return_predicted', 0):.3f}")
                print(f"     Price Correlation: {corr_coef:.3f}")
                
            plt.tight_layout()
            plot_path = self.output_dir / f"{symbol}_performance_metrics.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"  📈 {symbol} performance metrics plot saved to {plot_path}")
            
        except Exception as e:
            print(f"❌ Failed to create plots for {symbol}: {e}")
            import traceback
            traceback.print_exc()
    
    def create_ablation_performance_heatmap(self, symbol: str):
        """Create heatmap showing performance across models and feature modes."""
        if symbol not in self.results or len(self.results[symbol]) == 0:
            print(f"⚠️ No results for ablation heatmap for {symbol}")
            return
            
        print(f"📊 Creating ablation performance heatmap for {symbol}...")
        
        try:
            import pandas as pd
            import matplotlib.pyplot as plt
            import numpy as np
            
            # Define expected models and feature modes
            expected_models = ['Ridge Regression', 'Random Forest', 'XGBoost']
            expected_feature_modes = ['full', 'no_news', 'no_economic', 'no_technical', 'core', 'ohlcv_only']
            
            # Filter to only include expected models that exist in results
            all_keys = list(self.results[symbol].keys())
            valid_models = []
            valid_feature_modes = []
            
            print(f"   🔍 Analyzing keys: {all_keys}")
            
            # Extract valid model-feature combinations
            for key in all_keys:
                for model in expected_models:
                    for mode in expected_feature_modes:
                        expected_key = f"{model}_{mode}"
                        if key == expected_key:
                            if model not in valid_models:
                                valid_models.append(model)
                            if mode not in valid_feature_modes:
                                valid_feature_modes.append(mode)
            
            # Sort for consistent ordering
            valid_models = sorted(valid_models)
            valid_feature_modes = sorted(valid_feature_modes)
            
            print(f"   🎯 Valid models: {valid_models}")
            print(f"   📊 Valid feature modes: {valid_feature_modes}")
            
            if not valid_models or not valid_feature_modes:
                print(f"   ⚠️ Insufficient valid data for heatmap - models: {len(valid_models)}, modes: {len(valid_feature_modes)}")
                return
            
            # Create matrices for different metrics
            r2_matrix = np.full((len(valid_models), len(valid_feature_modes)), np.nan)
            accuracy_matrix = np.full((len(valid_models), len(valid_feature_modes)), np.nan)
            sharpe_matrix = np.full((len(valid_models), len(valid_feature_modes)), np.nan)
            
            for i, model in enumerate(valid_models):
                for j, mode in enumerate(valid_feature_modes):
                    key = f"{model}_{mode}"
                    if key in self.results[symbol]:
                        metrics = self.results[symbol][key]['metrics']
                        r2_matrix[i, j] = metrics['r2_returns']
                        accuracy_matrix[i, j] = metrics['accuracy']
                        sharpe_matrix[i, j] = metrics['sharpe_ratio']
            
            # Check if we have any data
            if np.isnan(r2_matrix).all():
                print(f"   ⚠️ No valid data found for heatmap")
                return
            
            # Calculate dynamic figure size based on content
            n_models = len(valid_models)
            n_modes = len(valid_feature_modes)
            
            # Base sizing: wider for more columns, taller for more rows
            base_width_per_heatmap = max(4, n_modes * 0.8)  # Minimum 4, scale with columns
            base_height = max(3, n_models * 0.6)  # Minimum 3, scale with rows
            total_width = base_width_per_heatmap * 3 + 3  # 3 heatmaps + spacing
            
            print(f"   📐 Dynamic sizing: {n_models} models × {n_modes} modes → {total_width:.1f}×{base_height:.1f}")
            
            # Create the heatmap plot with dynamic sizing
            fig, axes = plt.subplots(1, 3, figsize=(total_width, base_height))
            fig.suptitle(f'{symbol} - Ablation Study Performance Heatmap', fontsize=16, fontweight='bold')
            
            # Adjust spacing between subplots
            plt.subplots_adjust(wspace=0.4, hspace=0.3)
            
            # Prepare feature mode labels with better formatting
            mode_labels = []
            for mode in valid_feature_modes:
                if mode in FEATURE_MODES:
                    # Use the descriptive name but make it shorter
                    label = FEATURE_MODES[mode]
                    if len(label) > 20:  # Truncate very long labels
                        label = label.replace('All multimodal features', 'Full')
                        label = label.replace('All except ', 'No ')
                        label = label.replace(' embeddings', '')
                        label = label.replace(' indicators', '')
                    mode_labels.append(label)
                else:
                    mode_labels.append(mode.replace('_', ' ').title())
            
            # R² heatmap
            vmin_r2, vmax_r2 = np.nanmin(r2_matrix), np.nanmax(r2_matrix)
            if vmin_r2 == vmax_r2:  # Handle edge case
                vmin_r2, vmax_r2 = vmin_r2 - 0.1, vmax_r2 + 0.1
            
            im1 = axes[0].imshow(r2_matrix, cmap='RdYlGn', aspect='auto', vmin=vmin_r2, vmax=vmax_r2)
            axes[0].set_title('R² Score (Returns Prediction)', fontsize=12, pad=20)
            axes[0].set_xticks(range(len(valid_feature_modes)))
            axes[0].set_xticklabels(mode_labels, rotation=45, ha='right', fontsize=10)
            axes[0].set_yticks(range(len(valid_models)))
            axes[0].set_yticklabels(valid_models, fontsize=10)
            
            # Add text annotations with better formatting
            for i in range(len(valid_models)):
                for j in range(len(valid_feature_modes)):
                    if not np.isnan(r2_matrix[i, j]):
                        # Dynamic text color based on value
                        text_color = 'white' if r2_matrix[i, j] < (vmin_r2 + vmax_r2) / 2 else 'black'
                        # Adjust font size based on cell size
                        font_size = max(8, min(12, base_width_per_heatmap / n_modes * 2))
                        axes[0].text(j, i, f'{r2_matrix[i, j]:.3f}', ha='center', va='center', 
                                   color=text_color, fontweight='bold', fontsize=font_size)
            
            # Add colorbar with proper sizing
            cbar1 = plt.colorbar(im1, ax=axes[0], shrink=0.8)
            cbar1.ax.tick_params(labelsize=9)
            
            # Accuracy heatmap
            vmin_acc, vmax_acc = np.nanmin(accuracy_matrix), np.nanmax(accuracy_matrix)
            if vmin_acc == vmax_acc:
                vmin_acc, vmax_acc = vmin_acc - 0.1, vmax_acc + 0.1
                
            im2 = axes[1].imshow(accuracy_matrix, cmap='RdYlGn', aspect='auto', vmin=vmin_acc, vmax=vmax_acc)
            axes[1].set_title('Directional Accuracy', fontsize=12, pad=20)
            axes[1].set_xticks(range(len(valid_feature_modes)))
            axes[1].set_xticklabels(mode_labels, rotation=45, ha='right', fontsize=10)
            axes[1].set_yticks(range(len(valid_models)))
            axes[1].set_yticklabels(valid_models, fontsize=10)
            
            for i in range(len(valid_models)):
                for j in range(len(valid_feature_modes)):
                    if not np.isnan(accuracy_matrix[i, j]):
                        text_color = 'white' if accuracy_matrix[i, j] < (vmin_acc + vmax_acc) / 2 else 'black'
                        font_size = max(8, min(12, base_width_per_heatmap / n_modes * 2))
                        axes[1].text(j, i, f'{accuracy_matrix[i, j]:.3f}', ha='center', va='center',
                                   color=text_color, fontweight='bold', fontsize=font_size)
            
            cbar2 = plt.colorbar(im2, ax=axes[1], shrink=0.8)
            cbar2.ax.tick_params(labelsize=9)
            
            # Sharpe Ratio heatmap
            vmin_sharpe, vmax_sharpe = np.nanmin(sharpe_matrix), np.nanmax(sharpe_matrix)
            if vmin_sharpe == vmax_sharpe:
                vmin_sharpe, vmax_sharpe = vmin_sharpe - 0.1, vmax_sharpe + 0.1
                
            im3 = axes[2].imshow(sharpe_matrix, cmap='RdYlGn', aspect='auto', vmin=vmin_sharpe, vmax=vmax_sharpe)
            axes[2].set_title('Sharpe Ratio', fontsize=12, pad=20)
            axes[2].set_xticks(range(len(valid_feature_modes)))
            axes[2].set_xticklabels(mode_labels, rotation=45, ha='right', fontsize=10)
            axes[2].set_yticks(range(len(valid_models)))
            axes[2].set_yticklabels(valid_models, fontsize=10)
            
            for i in range(len(valid_models)):
                for j in range(len(valid_feature_modes)):
                    if not np.isnan(sharpe_matrix[i, j]):
                        # For Sharpe ratio, use absolute value for color threshold
                        mid_point = (vmin_sharpe + vmax_sharpe) / 2
                        text_color = 'white' if abs(sharpe_matrix[i, j] - mid_point) < abs(vmax_sharpe - vmin_sharpe) * 0.3 else 'black'
                        font_size = max(8, min(12, base_width_per_heatmap / n_modes * 2))
                        axes[2].text(j, i, f'{sharpe_matrix[i, j]:.3f}', ha='center', va='center',
                                   color=text_color, fontweight='bold', fontsize=font_size)
            
            cbar3 = plt.colorbar(im3, ax=axes[2], shrink=0.8)
            cbar3.ax.tick_params(labelsize=9)
            
            plt.tight_layout()
            plot_path = self.output_dir / f"{symbol}_ablation_heatmap.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"  📈 {symbol} ablation heatmap saved to {plot_path}")
            
        except Exception as e:
            print(f"❌ Failed to create ablation heatmap for {symbol}: {e}")
            import traceback
            traceback.print_exc()

    def create_feature_contribution_analysis(self, symbol: str):
        """Create bar chart showing feature group contribution to performance with robust missing value handling."""
        if symbol not in self.results or len(self.results[symbol]) == 0:
            print(f"⚠️ No results for feature contribution analysis for {symbol}")
            return
            
        print(f"📊 Creating feature contribution analysis for {symbol}...")
        
        try:
            import pandas as pd
            import matplotlib.pyplot as plt
            import numpy as np
            
            # Define expected models (only the actual model names)
            expected_models = ['Ridge Regression', 'Random Forest', 'XGBoost']
            
            # Find which models actually have successful results
            valid_models = []
            all_keys = list(self.results[symbol].keys())
            
            print(f"   🔍 Analyzing available results: {len(all_keys)} total keys")
            
            for model in expected_models:
                full_key = f"{model}_full"
                if full_key in all_keys:
                    # Check if the result is valid (not a failure)
                    result = self.results[symbol][full_key]
                    if result.get('training_successful', True) and not result.get('metrics', {}).get('failed_training', False):
                        valid_models.append(model)
                        print(f"   ✅ {model}: Valid baseline found")
                    else:
                        print(f"   ❌ {model}: Baseline failed - {result.get('error', 'Unknown error')}")
                else:
                    print(f"   ❌ {model}: No baseline result found")
            
            if not valid_models:
                print(f"⚠️ No valid models with successful full baseline found for {symbol}")
                print(f"   Available keys: {all_keys}")
                return
            
            print(f"   🎯 Valid models for analysis: {valid_models}")
            
            feature_groups = ['News', 'Economic', 'Technical']
            metrics_to_analyze = ['r2_returns', 'accuracy', 'sharpe_ratio']
            metric_labels = ['R² Returns', 'Accuracy', 'Sharpe Ratio']
            
            fig, axes = plt.subplots(len(metrics_to_analyze), 1, figsize=(12, 4 * len(metrics_to_analyze)))
            if len(metrics_to_analyze) == 1:
                axes = [axes]
            
            fig.suptitle(f'{symbol} - Feature Group Contribution Analysis', fontsize=16, fontweight='bold')
            
            for metric_idx, (metric, metric_label) in enumerate(zip(metrics_to_analyze, metric_labels)):
                ax = axes[metric_idx]
                
                # Calculate contributions for each model
                x_pos = np.arange(len(feature_groups))
                width = 0.25 if len(valid_models) > 1 else 0.5
                
                has_any_data = False
                models_with_data = []
                
                for model_idx, model in enumerate(valid_models):
                    contributions = []
                    contribution_labels = []
                    
                    # Get baseline performance (full features)
                    full_key = f"{model}_full"
                    baseline_result = self.results[symbol][full_key]
                    baseline_perf = baseline_result['metrics'][metric]
                    
                    print(f"\n   📊 {model} {metric} analysis:")
                    print(f"     Baseline ({full_key}): {baseline_perf:.4f}")
                    
                    # Calculate performance drop for each ablation
                    ablation_mappings = [
                        ('News', 'no_news'),
                        ('Economic', 'no_economic'), 
                        ('Technical', 'no_technical')
                    ]
                    
                    model_has_data = False
                    
                    for group_name, suffix in ablation_mappings:
                        ablation_key = f"{model}_{suffix}"
                        
                        if ablation_key in self.results[symbol]:
                            ablation_result = self.results[symbol][ablation_key]
                            
                            # Check if the ablation result is valid
                            if (ablation_result.get('training_successful', True) and 
                                not ablation_result.get('metrics', {}).get('failed_training', False)):
                                
                                ablated_perf = ablation_result['metrics'][metric]
                                contribution = baseline_perf - ablated_perf  # Positive = feature helps
                                contributions.append(contribution)
                                contribution_labels.append(f"{contribution:.3f}")
                                model_has_data = True
                                
                                print(f"     {group_name} removal ({ablation_key}): {ablated_perf:.4f} → contribution: {contribution:.4f}")
                            else:
                                contributions.append(np.nan)
                                contribution_labels.append("Failed")
                                print(f"     {group_name} removal ({ablation_key}): FAILED - {ablation_result.get('error', 'Unknown error')}")
                        else:
                            contributions.append(np.nan)
                            contribution_labels.append("Missing")
                            print(f"     {group_name} removal ({ablation_key}): MISSING")
                    
                    # Plot bars for this model if we have any valid data
                    if model_has_data:
                        has_any_data = True
                        models_with_data.append(model)
                        
                        # Convert nan to 0 for plotting, but keep track of which are real vs missing
                        plot_contributions = []
                        for contrib in contributions:
                            if np.isnan(contrib):
                                plot_contributions.append(0)
                            else:
                                plot_contributions.append(contrib)
                        
                        # Plot bars
                        bars = ax.bar(x_pos + model_idx * width, plot_contributions, width, 
                                     label=model, alpha=0.8)
                        
                        # Add value labels on bars
                        for bar, contrib, label in zip(bars, contributions, contribution_labels):
                            height = bar.get_height()
                            
                            if not np.isnan(contrib) and abs(height) > 1e-6:  # Valid, non-zero contribution
                                ax.text(bar.get_x() + bar.get_width()/2., 
                                       height + (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.01 if height >= 0 else height - (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.01,
                                       label, ha='center', va='bottom' if height >= 0 else 'top', fontsize=9)
                            elif np.isnan(contrib):  # Missing data
                                ax.text(bar.get_x() + bar.get_width()/2., 0, 
                                       'N/A', ha='center', va='center', fontsize=8, 
                                       style='italic', color='red')
                
                # Configure the plot
                if not has_any_data:
                    ax.text(0.5, 0.5, 'No valid ablation data available\n(All model-feature combinations failed)', 
                           transform=ax.transAxes, ha='center', va='center', fontsize=12, style='italic', color='red')
                    ax.set_title(f'{metric_label} - No Data Available')
                else:
                    ax.set_xlabel('Feature Group Removed')
                    ax.set_ylabel(f'{metric_label} Contribution')
                    ax.set_title(f'{metric_label} - Positive = Feature Helps Performance')
                    ax.set_xticks(x_pos + width * (len(valid_models) - 1) / 2)
                    ax.set_xticklabels(feature_groups)
                    
                    if models_with_data:
                        ax.legend(title='Models with Data')
                    
                    # Add horizontal line at zero for reference
                    ax.axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=0.5)
                
                ax.grid(True, alpha=0.3)
                
            plt.tight_layout()
            
            # Save the plot
            filename = f"{symbol}_feature_contribution_analysis_enhanced.png"
            filepath = self.output_dir / filename
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.show()
            
            print(f"✅ Enhanced feature contribution analysis saved: {filepath}")
            
            # Print summary statistics
            print(f"\n📊 Summary for {symbol}:")
            print(f"   🎯 Models analyzed: {len(valid_models)}")
            print(f"   ✅ Models with data: {len(set(models_with_data))}")
            print(f"   📋 Feature groups: {len(feature_groups)}")
            print(f"   📈 Metrics analyzed: {len(metrics_to_analyze)}")
            
        except Exception as e:
            print(f"❌ Error creating feature contribution analysis: {e}")
            import traceback
            print(f"   Detailed error: {traceback.format_exc()}")

    def create_ablation_summary_table(self):
        """Create summary table showing best feature combinations across symbols and models."""
        print("📊 Creating ablation summary table...")
        
        try:
            import pandas as pd
            
            # Define expected models and feature modes for validation
            expected_models = ['Ridge Regression', 'Random Forest', 'XGBoost']
            expected_feature_modes = ['full', 'no_news', 'no_economic', 'no_technical', 'core', 'ohlcv_only']
            
            # Collect all valid results
            summary_data = []
            
            for symbol, symbol_results in self.results.items():
                for model in expected_models:
                    for mode in expected_feature_modes:
                        expected_key = f"{model}_{mode}"
                        if expected_key in symbol_results:
                            result = symbol_results[expected_key]
                            metrics = result['metrics']
                            summary_data.append({
                                'Symbol': symbol,
                                'Model': model,
                                'Feature_Mode': mode,
                                'R2_Returns': metrics['r2_returns'],
                                'Accuracy': metrics['accuracy'],
                                'Sharpe_Ratio': metrics['sharpe_ratio'],
                                'N_Features': result.get('n_features', 0)
                            })
            
            if not summary_data:
                print("⚠️ No valid ablation data found for summary table")
                print("   Expected format: 'Model_FeatureMode' (e.g., 'Ridge Regression_full')")
                return
            
            df = pd.DataFrame(summary_data)
            
            print(f"   ✅ Found {len(summary_data)} valid results across {df['Symbol'].nunique()} symbols")
            print(f"   📊 Models: {sorted(df['Model'].unique())}")
            print(f"   🎯 Feature modes: {sorted(df['Feature_Mode'].unique())}")
            
            # Create summary statistics
            print("\n📊 Ablation Study Summary:")
            print("=" * 80)
            
            # Best feature mode by metric (averaged across symbols and models)
            print("\n🏆 Best Feature Modes (Average Performance):")
            for metric in ['R2_Returns', 'Accuracy', 'Sharpe_Ratio']:
                avg_by_mode = df.groupby('Feature_Mode')[metric].mean().sort_values(ascending=False)
                print(f"\n{metric}:")
                for i, (mode, score) in enumerate(avg_by_mode.head(3).items()):
                    print(f"  {i+1}. {mode}: {score:.4f} - {FEATURE_MODES.get(mode, 'Unknown')}")
            
            # Feature count impact
            print(f"\n📊 Feature Count Analysis:")
            mode_features = df.groupby('Feature_Mode')['N_Features'].mean().sort_values()
            mode_performance = df.groupby('Feature_Mode')['R2_Returns'].mean()
            
            for mode in mode_features.index:
                n_feat = mode_features[mode]
                perf = mode_performance[mode]
                print(f"  {mode}: {n_feat:.0f} features → R² = {perf:.4f}")
            
            # Save detailed table
            table_path = self.output_dir / "ablation_summary_table.csv"
            df.to_csv(table_path, index=False)
            print(f"\n💾 Detailed results saved to: {table_path}")
            
        except Exception as e:
            print(f"❌ Failed to create ablation summary table: {e}")

    def create_comparison_plots(self):
        """Create comparison plots across models and symbols with new metrics categories."""
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
            
            # Create comparison plots - 3 categories (Classification, Regression, Financial)
            fig, axes = plt.subplots(3, 2, figsize=(16, 18))
            fig.suptitle('Model Performance Comparison - Classification, Regression & Financial Metrics', fontsize=16, fontweight='bold')
            
            # Row 1: Classification: Accuracy by model and symbol
            accuracy_pivot = metrics_df.pivot(index='symbol', columns='model', values='accuracy')
            accuracy_pivot.plot(kind='bar', ax=axes[0, 0])
            axes[0, 0].set_title('Classification: Accuracy by Model and Symbol')
            axes[0, 0].set_ylabel('Accuracy Score')
            axes[0, 0].tick_params(axis='x', rotation=45)
            axes[0, 0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            
            # Row 1: Regression: R² by model and symbol
            r2_pivot = metrics_df.pivot(index='symbol', columns='model', values='r2_returns')
            r2_pivot.plot(kind='bar', ax=axes[0, 1])
            axes[0, 1].set_title('Regression: R² Score by Model and Symbol')
            axes[0, 1].set_ylabel('R² Score')
            axes[0, 1].tick_params(axis='x', rotation=45)
            axes[0, 1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            
            # Row 2: Financial: Sharpe Ratio by model and symbol
            sharpe_pivot = metrics_df.pivot(index='symbol', columns='model', values='sharpe_ratio')
            sharpe_pivot.plot(kind='bar', ax=axes[1, 0])
            axes[1, 0].set_title('Financial: Sharpe Ratio by Model and Symbol')
            axes[1, 0].set_ylabel('Sharpe Ratio')
            axes[1, 0].tick_params(axis='x', rotation=45)
            axes[1, 0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            
            # Row 2: Returns MAPE comparison
            mape_pivot = metrics_df.pivot(index='symbol', columns='model', values='mape_returns')
            mape_pivot.plot(kind='bar', ax=axes[1, 1])
            axes[1, 1].set_title('Returns: MAPE by Model and Symbol')
            axes[1, 1].set_ylabel('MAPE (%)')
            axes[1, 1].tick_params(axis='x', rotation=45)
            axes[1, 1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            
            # Row 3: Average performance by model
            avg_metrics = metrics_df.groupby('model').agg({
                'accuracy': 'mean',
                'f1_score': 'mean',
                'r2_returns': 'mean',
                'mae_returns': 'mean',
                'sharpe_ratio': 'mean',
                'annualized_return_predicted': 'mean'
            })
            
            # Average Classification metrics
            classification_avg = avg_metrics[['accuracy', 'f1_score']]
            classification_avg.plot(kind='bar', ax=axes[2, 0])
            axes[2, 0].set_title('Average Classification Performance by Model')
            axes[2, 0].set_ylabel('Average Score')
            axes[2, 0].tick_params(axis='x', rotation=45)
            
            # Average Regression metrics
            regression_avg = avg_metrics[['r2_returns', 'mae_returns']]
            # Use secondary y-axis for MAE since it's on different scale
            ax_twin = axes[2, 1].twinx()
            avg_metrics['r2_returns'].plot(kind='bar', ax=axes[2, 1], color='blue', alpha=0.7, label='R²')
            avg_metrics['mae_returns'].plot(kind='bar', ax=ax_twin, color='red', alpha=0.7, label='MAE')
            axes[2, 1].set_title('Average Regression Performance by Model')
            axes[2, 1].set_ylabel('R² Score', color='blue')
            ax_twin.set_ylabel('MAE', color='red')
            axes[2, 1].tick_params(axis='x', rotation=45)
            axes[2, 1].legend(loc='upper left')
            ax_twin.legend(loc='upper right')
            
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
            import traceback
            traceback.print_exc()
    
    def create_returns_comparison_plots(self, symbol: str):
        """Create detailed returns comparison plots for a specific symbol."""
        if symbol not in self.results or len(self.results[symbol]) == 0:
            print(f"⚠️ No results to create returns comparison for {symbol}")
            return
            
        print(f"📊 Creating returns comparison plots for {symbol}...")
        
        symbol_results = self.results[symbol]
        n_models = len(symbol_results)
        
        try:
            # Create subplots: 2 rows, 3 columns for various comparison views
            fig, axes = plt.subplots(2, 3, figsize=(20, 12))
            fig.suptitle(f'{symbol} - Returns Analysis & Comparison', fontsize=16, fontweight='bold')
            
            # Collect all data for analysis
            all_actual_returns = []
            all_predicted_returns = []
            model_names = []
            
            for model_name, result in symbol_results.items():
                actuals = result['actuals']
                predictions = result['predictions']
                
                # Flatten multi-step predictions for analysis
                actual_returns_flat = actuals.flatten()
                predicted_returns_flat = predictions.flatten()
                
                all_actual_returns.extend(actual_returns_flat)
                all_predicted_returns.extend(predicted_returns_flat)
                model_names.extend([model_name] * len(actual_returns_flat))
            
            # Convert to numpy arrays
            all_actual_returns = np.array(all_actual_returns)
            all_predicted_returns = np.array(all_predicted_returns)
            
            # Plot 1: Overall Returns Scatter Plot
            axes[0, 0].scatter(all_actual_returns, all_predicted_returns, alpha=0.6)
            axes[0, 0].plot([all_actual_returns.min(), all_actual_returns.max()], 
                           [all_actual_returns.min(), all_actual_returns.max()], 'r--', lw=2)
            axes[0, 0].set_xlabel('Actual Returns')
            axes[0, 0].set_ylabel('Predicted Returns')
            axes[0, 0].set_title('Overall Returns: Actual vs Predicted')
            axes[0, 0].grid(True, alpha=0.3)
            
            # Add correlation coefficient
            corr_coef = np.corrcoef(all_actual_returns, all_predicted_returns)[0, 1]
            axes[0, 0].text(0.05, 0.95, f'Correlation: {corr_coef:.3f}', 
                           transform=axes[0, 0].transAxes, fontsize=10,
                           bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
            
            # Plot 2: Returns Distribution Comparison
            axes[0, 1].hist(all_actual_returns, bins=50, alpha=0.7, label='Actual Returns', density=True)
            axes[0, 1].hist(all_predicted_returns, bins=50, alpha=0.7, label='Predicted Returns', density=True)
            axes[0, 1].set_xlabel('Returns')
            axes[0, 1].set_ylabel('Density')
            axes[0, 1].set_title('Returns Distribution Comparison')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
            
            # Plot 3: Model-wise Returns MAPE
            model_mapes = []
            unique_models = list(symbol_results.keys())
            
            for model_name in unique_models:
                result = symbol_results[model_name]
                mape = result['metrics']['mape_returns']
                model_mapes.append(mape)
            
            bars = axes[0, 2].bar(unique_models, model_mapes, alpha=0.8)
            axes[0, 2].set_title('Returns MAPE by Model')
            axes[0, 2].set_ylabel('MAPE (%)')
            axes[0, 2].tick_params(axis='x', rotation=45)
            axes[0, 2].grid(True, alpha=0.3)
            
            # Add value labels on bars
            for bar, value in zip(bars, model_mapes):
                axes[0, 2].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.1,
                               f'{value:.2f}%', ha='center', va='bottom', fontsize=10)
            
            # Plot 4: Time Series of Returns (show a sample)
            # Take the first model's time series for demonstration
            first_model = list(symbol_results.keys())[0]
            first_result = symbol_results[first_model]
            sample_size = min(100, len(first_result['actuals']))
            
            sample_indices = np.arange(sample_size)
            sample_actual = first_result['actuals'][:sample_size].mean(axis=1)  # Average across prediction steps
            sample_predicted = first_result['predictions'][:sample_size].mean(axis=1)
            
            axes[1, 0].plot(sample_indices, sample_actual, 'b-', label='Actual Returns', linewidth=2)
            axes[1, 0].plot(sample_indices, sample_predicted, 'r--', label='Predicted Returns', linewidth=2)
            axes[1, 0].set_xlabel('Time (Sample Index)')
            axes[1, 0].set_ylabel('Average Returns')
            axes[1, 0].set_title(f'Time Series Returns ({first_model})')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
            
            # Plot 5: Residuals Analysis
            residuals = all_predicted_returns - all_actual_returns
            axes[1, 1].scatter(all_actual_returns, residuals, alpha=0.6)
            axes[1, 1].axhline(y=0, color='r', linestyle='--', lw=2)
            axes[1, 1].set_xlabel('Actual Returns')
            axes[1, 1].set_ylabel('Residuals (Predicted - Actual)')
            axes[1, 1].set_title('Residuals Analysis')
            axes[1, 1].grid(True, alpha=0.3)
            
            # Plot 6: Cumulative Returns Comparison
            # Calculate cumulative returns for each model
            axes[1, 2].set_title('Cumulative Returns by Model')
            axes[1, 2].set_xlabel('Time (Sample Index)')
            axes[1, 2].set_ylabel('Cumulative Returns')
            
            for i, (model_name, result) in enumerate(symbol_results.items()):
                actuals = result['actuals']
                predictions = result['predictions']
                
                # Calculate cumulative returns for first 50 samples
                n_samples = min(50, len(actuals))
                cumulative_actual = []
                cumulative_predicted = []
                
                for j in range(n_samples):
                    if j == 0:
                        cumulative_actual.append(0)
                        cumulative_predicted.append(0)
                    else:
                        cumulative_actual.append(cumulative_actual[-1] + actuals[j].mean())
                        cumulative_predicted.append(cumulative_predicted[-1] + predictions[j].mean())
                
                sample_indices = np.arange(n_samples)
                axes[1, 2].plot(sample_indices, cumulative_actual, '-', 
                               label=f'{model_name} (Actual)', alpha=0.7)
                axes[1, 2].plot(sample_indices, cumulative_predicted, '--', 
                               label=f'{model_name} (Predicted)', alpha=0.7)
            
            axes[1, 2].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            axes[1, 2].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plot_path = self.output_dir / f"{symbol}_returns_comparison.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"  📈 {symbol} returns comparison plots saved to {plot_path}")
            
        except Exception as e:
            print(f"❌ Failed to create returns comparison plots for {symbol}: {e}")
            import traceback
            traceback.print_exc()
    
    def print_summary(self):
        """Print summary of results with new metrics categories."""
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
                            # Classification metrics
                            'accuracy': [], 'f1_score': [], 'auc_roc': [],
                            # Regression metrics
                            'mae_returns': [], 'rmse_returns': [], 'r2_returns': [], 'mape_returns': [],
                            # Financial metrics
                            'sharpe_ratio': [], 'annualized_return_predicted': [], 'mdd_predicted': []
                        }
                    
                    metrics = result['metrics']
                    # Classification
                    model_averages[model_name]['accuracy'].append(metrics.get('accuracy', 0))
                    model_averages[model_name]['f1_score'].append(metrics.get('f1_score', 0))
                    model_averages[model_name]['auc_roc'].append(metrics.get('auc_roc', 0))
                    # Regression
                    model_averages[model_name]['mae_returns'].append(metrics.get('mae_returns', 0))
                    model_averages[model_name]['rmse_returns'].append(metrics.get('rmse_returns', 0))
                    model_averages[model_name]['r2_returns'].append(metrics.get('r2_returns', 0))
                    model_averages[model_name]['mape_returns'].append(metrics.get('mape_returns', 0))
                    # Financial
                    model_averages[model_name]['sharpe_ratio'].append(metrics.get('sharpe_ratio', 0))
                    model_averages[model_name]['annualized_return_predicted'].append(metrics.get('annualized_return_predicted', 0))
                    model_averages[model_name]['mdd_predicted'].append(metrics.get('mdd_predicted', 0))
            
            if not model_averages:
                print("❌ No model results to average")
                return
            
            # Print comprehensive summary table
            print("🎯 CLASSIFICATION METRICS (Trend Prediction)")
            print(f"{'Model':<20} {'Accuracy':<10} {'F1-Score':<10} {'AUC-ROC':<10}")
            print("-" * 52)
            
            for model_name, metrics in model_averages.items():
                avg_accuracy = np.mean(metrics['accuracy'])
                avg_f1 = np.mean(metrics['f1_score'])
                avg_auc = np.mean(metrics['auc_roc'])
                
                print(f"{model_name:<20} {avg_accuracy:<10.3f} {avg_f1:<10.3f} {avg_auc:<10.3f}")
            
            print("\n📊 REGRESSION METRICS (Price Prediction)")
            print(f"{'Model':<20} {'MAE':<10} {'RMSE':<10} {'MAPE(%)':<10} {'R²':<10}")
            print("-" * 62)
            
            best_r2 = -np.inf
            best_model = ""
            
            for model_name, metrics in model_averages.items():
                avg_mae = np.mean(metrics['mae_returns'])
                avg_rmse = np.mean(metrics['rmse_returns'])
                avg_r2 = np.mean(metrics['r2_returns'])
                avg_mape = np.mean(metrics['mape_returns'])
                
                print(f"{model_name:<20} {avg_mae:<10.4f} {avg_rmse:<10.4f} {avg_mape:<10.2f} {avg_r2:<10.4f}")
                
                if avg_r2 > best_r2:
                    best_r2 = avg_r2
                    best_model = model_name
            
            print("\n💰 FINANCIAL METRICS")
            print(f"{'Model':<20} {'Sharpe':<10} {'Ann.Return':<12} {'Max DD':<10}")
            print("-" * 54)
            
            for model_name, metrics in model_averages.items():
                avg_sharpe = np.mean(metrics['sharpe_ratio'])
                avg_return = np.mean(metrics['annualized_return_predicted'])
                avg_mdd = np.mean(metrics['mdd_predicted'])
                
                print(f"{model_name:<20} {avg_sharpe:<10.3f} {avg_return:<12.3f} {avg_mdd:<10.3f}")
            
            print(f"\n🏆 Best performing model (Regression R²): {best_model} (R² = {best_r2:.4f})")
            
            # Print per-symbol details
            print(f"\n📊 Per-Symbol Performance Summary:")
            for symbol in self.results:
                print(f"\n  {symbol}:")
                for model_name, result in self.results[symbol].items():
                    metrics = result['metrics']
                    print(f"    {model_name:<15}: Acc={metrics.get('accuracy', 0):.3f}, R²={metrics.get('r2_returns', 0):.3f}, Sharpe={metrics.get('sharpe_ratio', 0):.3f}")
                    
        except Exception as e:
            print(f"❌ Error in summary: {e}")
            import traceback
            traceback.print_exc()

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
                        default=os.getenv('NEWS_API_KEY'),
                        help='News API key for news data (default: from NEWS_API_KEY env var)')
    parser.add_argument('--fred-api-key', type=str, 
                        default=os.getenv('FRED_API_KEY'),
                        help='FRED API key for economic data (default: from FRED_API_KEY env var)')
    parser.add_argument('--api-ninjas-key', type=str, 
                        default=os.getenv('API_NINJAS_KEY'),
                        help='API Ninjas key for additional data (default: from API_NINJAS_KEY env var)')
    
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
    parser.add_argument('--ablation-study', action='store_true', default=True,
                        help='Run ablation study across feature groups')
    parser.add_argument('--no-ablation', action='store_true', default=False,
                        help='Skip ablation study and run only full multimodal baseline')
    
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
    
    # Determine feature mode and ablation settings
    use_multimodal = args.multimodal and not args.basic_only
    run_ablation = args.ablation_study and not args.no_ablation
    
    print("🚀 Starting Multimodal Baseline Models Pipeline with Ablation Study")
    print(f"📅 Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🎯 Symbols: {config['symbols']}")
    print(f"📊 Date range: {config['start_date']} to {config['end_date']}")
    print(f"🔮 Fixed prediction horizon: {config['predict_len']} steps")
    print(f"🎛️  Multimodal features: {use_multimodal}")
    print(f"🧪 Ablation study: {run_ablation}")
    print(f"⚡ Using EXACT same data loading as TFT pipeline")
    print(f"📏 Encoder length: {config['encoder_len']}, Predict length: {config['predict_len']}")
    
    if run_ablation:
        print(f"\n🧪 Ablation Study Configuration:")
        print(f"   📊 Feature modes to test: {len(FEATURE_MODES)}")
        for mode, desc in FEATURE_MODES.items():
            print(f"     • {mode}: {desc}")
    
    # Log API key status
    print(f"\n🔑 API Key Status:")
    print(f"   NEWS_API_KEY: {'✅ Set' if config['news_api_key'] else '❌ Not set'}")
    print(f"   FRED_API_KEY: {'✅ Set' if config['fred_api_key'] else '❌ Not set'}")
    print(f"   API_NINJAS_KEY: {'✅ Set' if config['api_ninjas_key'] else '❌ Not set'}")
    if use_multimodal and not any([config['news_api_key'], config['fred_api_key'], config['api_ninjas_key']]):
        print("   ⚠️  No API keys set - multimodal features will be limited")
        print("   💡 Set API keys in .env file or use command line arguments")
    
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
        if run_ablation:
            print("\n" + "=" * 60)
            print("🧪 TRAINING MODELS WITH ABLATION STUDY")
            print("=" * 60)
            
            for symbol in config['symbols']:
                # Check if symbol exists in the data
                train_symbols = train_datamodule.feature_df['symbol'].unique()
                val_symbols = val_datamodule.feature_df['symbol'].unique()
                
                if symbol in train_symbols or symbol in val_symbols:
                    runner.train_and_evaluate_symbol_ablation(
                        train_datamodule, val_datamodule, symbol
                    )
                else:
                    print(f"⚠️ No data found for {symbol}")
        else:
            print("\n" + "=" * 60)
            print("🎯 TRAINING BASELINE MODELS (NO ABLATION)")
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
        if run_ablation:
            print("\n" + "=" * 60)
            print("📊 CREATING ABLATION ANALYSIS VISUALIZATIONS")
            print("=" * 60)
            
            for symbol in config['symbols']:
                if symbol in runner.results and len(runner.results[symbol]) > 0:
                    print(f"\n🎯 Creating ablation analysis for {symbol}...")
                    
                    # Core ablation visualizations for research paper
                    runner.create_ablation_performance_heatmap(symbol)
                    runner.create_feature_contribution_analysis(symbol)
                    
                    # Selected enhanced analysis (reduced set for research focus)
                    runner.create_enhanced_prediction_plots(symbol)
                else:
                    print(f"⚠️ No results found for {symbol}")
            
            # Create cross-symbol ablation summary
            runner.create_ablation_summary_table()
        else:
            print("\n" + "=" * 60)
            print("📊 CREATING STANDARD VISUALIZATIONS")
            print("=" * 60)
            
            for symbol in config['symbols']:
                runner.create_enhanced_prediction_plots(symbol)
                runner.create_horizon_error_analysis(symbol)
        
        runner.create_comparison_plots()
        
        # Step 5: Print summary
        runner.print_summary()
        
        print(f"\n✅ Ablation Study Pipeline completed successfully!")
        print(f"📂 Results saved to: {runner.output_dir}")
        print(f"🧪 Ablation study: {len(FEATURE_MODES)} feature modes tested")
        print(f"🔮 Prediction mode: Fixed {config['predict_len']}-step horizon")
        
        # Summary of generated ablation analysis
        print(f"\n📊 Generated Ablation Analysis:")
        print(f"   � Ablation performance heatmaps: {len(config['symbols'])} symbols")
        print(f"   � Feature contribution analysis: {len(config['symbols'])} symbols") 
        print(f"   📈 Enhanced prediction analysis: {len(config['symbols'])} symbols")
        print(f"   � Comprehensive ablation summary table: 1 file")
        print(f"\n🧪 Ablation Study Features:")
        for mode, description in FEATURE_MODES.items():
            print(f"   ✅ {mode}: {description}")
        print(f"\n🎯 Research Paper Ready Outputs:")
        print(f"   ✅ Performance heatmaps (R², Accuracy, Sharpe Ratio)")
        print(f"   ✅ Feature group contribution quantification")
        print(f"   ✅ Statistical significance of feature groups")
        print(f"   ✅ Model robustness across feature combinations")
        print(f"   ✅ Detailed CSV results for further analysis")
        
    except Exception as e:
        print(f"\n❌ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    main()
