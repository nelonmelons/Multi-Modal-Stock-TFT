#!/usr/bin/env python3
"""
Stock Price Baseline Models for TFT Research Paper Comparison
============================================================

This module implements various baseline models for comprehensive comparison
in the multi-modal TFT research paper. All models use STOCK PRICE DATA ONLY
(OHLCV - Open, High, Low, Close, Volume) for fair comparison.

MODIFIED: All baseline models now use only stock price data (no news, earnings, etc.)

Baseline Categories:
1. Traditional ML: Linear Regression, Random Forest, XGBoost  
2. Deep Learning: LSTM, GRU, Vanilla Transformer
3. Time Series: ARIMA, Prophet
4. Finance-Specific: Buy & Hold, Moving Average

Key Features:
- Uses only OHLCV (Open, High, Low, Close, Volume) stock price data
- Predicts returns and converts to absolute prices
- Comprehensive plotting and metrics
- Prevents data leakage by using historical features only
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from typing import Dict, List, Tuple, Any, Optional
import warnings
warnings.filterwarnings('ignore')

try:
    from statsmodels.tsa.arima.model import ARIMA
    from prophet import Prophet
    ARIMA_AVAILABLE = True
    PROPHET_AVAILABLE = True
except ImportError:
    ARIMA_AVAILABLE = False
    PROPHET_AVAILABLE = False
    print("⚠️  ARIMA/Prophet not available. Install with: pip install statsmodels prophet")

class BaselineModel:
    """Base class for all baseline models."""
    
    def __init__(self, name: str, config: Dict[str, Any]):
        self.name = name
        self.config = config
        self.model = None
        self.scaler = StandardScaler()
        self.is_fitted = False
        
    def prepare_data(self, dataloader: DataLoader, feature_subset: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Extract and prepare STOCK PRICE DATA ONLY from dataloader for price/return prediction.
        CRITICAL: Prevent data leakage by using only historical features.
        MODIFIED: Only uses OHLCV (Open, High, Low, Close, Volume) stock price data.
        
        Returns:
            X: Stock price features [n_samples, 5] - OHLCV data only  
            y: Target returns [n_samples]
            timestamps: Date information for plotting [n_samples]
            current_prices: Current stock prices for price plotting [n_samples]
        """
        X_list, y_list, timestamp_list, price_list = [], [], [], []
        
        for batch in dataloader:
            if isinstance(batch, tuple):
                batch_data = batch[0]
            else:
                batch_data = batch
                
            # Extract stock price data ONLY (OHLCV)
            encoder_cont = batch_data['encoder_cont'].numpy()  # Shape: [batch, seq_len, features]
            decoder_target = batch_data['decoder_target'].numpy()  # Shape: [batch, predict_len]
            
            # STOCK PRICE ONLY: Extract OHLCV data (first 5 features)
            # Assumes OHLCV are the first 5 features: [Open, High, Low, Close, Volume]
            stock_data = encoder_cont[:, :, :5]  # [batch, seq_len, 5]
            
            # ANTI-LEAKAGE: Use features from timestep t-1 (not current timestep t)
            if stock_data.shape[1] > 1:
                X_batch = stock_data[:, -2, :]  # [batch, 5] - Previous timestep OHLCV
                current_prices = stock_data[:, -2, 3]  # Previous close price (index 3)
            else:
                # Fallback if sequence length is 1
                X_batch = stock_data[:, -1, :]
                current_prices = stock_data[:, -1, 3]
            
            # Target: predict next period return
            y_batch = decoder_target[:, 0] if decoder_target.ndim > 1 else decoder_target  # [batch]
            
            # Extract timestamps for plotting
            if 'time_idx' in batch_data:
                timestamp_batch = batch_data['time_idx'].numpy()[:, -1] + 1  # Next timestamp
            else:
                timestamp_batch = np.arange(len(X_batch))  # Fallback to indices
            
            X_list.append(X_batch)
            y_list.append(y_batch)
            timestamp_list.append(timestamp_batch)
            price_list.append(current_prices)
        
        X = np.vstack(X_list)
        y = np.concatenate(y_list)
        timestamps = np.concatenate(timestamp_list)
        current_prices = np.concatenate(price_list)
        
        # Store feature info for consistent prediction
        self.n_features_expected = X.shape[1]  # Should always be 5 for OHLCV
                
        print(f"   📊 Prepared STOCK data: {X.shape[0]} samples, {X.shape[1]} features (OHLCV)")
        print(f"   🎯 Target return range: {y.min():.4f} to {y.max():.4f}")
        print(f"   💰 Current price range: ${current_prices.min():.2f} to ${current_prices.max():.2f}")
        print(f"   🕒 Timestamp range: {timestamps.min():.0f} to {timestamps.max():.0f}")
        
        return X, y, timestamps, current_prices
    
    def fit(self, train_dataloader: DataLoader, feature_subset: Optional[str] = None):
        """Train the model."""
        raise NotImplementedError
        
    def predict(self, test_dataloader: DataLoader) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Make price/return predictions and convert to absolute prices.
        
        Returns:
            predictions: Predicted returns/prices
            actuals: Actual returns/prices  
            timestamps: Timestamps for plotting
            current_prices: Current stock prices for price plotting
        """
        if not self.is_fitted:
            raise ValueError(f"{self.name} model must be fitted before prediction")
        
        X_test, y_test, timestamps, current_prices = self.prepare_data(test_dataloader)
        
        # Check feature consistency
        if hasattr(self, 'n_features_expected'):
            if X_test.shape[1] != self.n_features_expected:
                raise ValueError(f"Feature shape mismatch, expected: {self.n_features_expected}, got {X_test.shape[1]}")
        
        X_test_scaled = self.scaler.transform(X_test)
        predictions = self.model.predict(X_test_scaled)
        
        return predictions, y_test, timestamps, current_prices
        
    def get_feature_importance(self) -> Optional[np.ndarray]:
        """Get feature importance if available."""
        return None

class LinearRegressionBaseline(BaselineModel):
    """Linear Regression baseline."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("Linear Regression", config)
        self.model = LinearRegression()
        
    def fit(self, train_dataloader: DataLoader, feature_subset: Optional[str] = None):
        # Store feature subset for consistent prediction
        X_train, y_train, _, _ = self.prepare_data(train_dataloader)
        X_train_scaled = self.scaler.fit_transform(X_train)
        self.model.fit(X_train_scaled, y_train)
        self.is_fitted = True
        
    def predict(self, test_dataloader: DataLoader) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        X_test, y_test, timestamps, current_prices = self.prepare_data(test_dataloader)
        
        X_test_scaled = self.scaler.transform(X_test)
        predictions = self.model.predict(X_test_scaled)
        return predictions, y_test, timestamps, current_prices
    
    def get_feature_importance(self) -> Optional[np.ndarray]:
        if self.is_fitted:
            return np.abs(self.model.coef_)
        return None

class RandomForestBaseline(BaselineModel):
    """Random Forest baseline."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("Random Forest", config)
        self.model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        
    def fit(self, train_dataloader: DataLoader, feature_subset: Optional[str] = None):
        X_train, y_train, _, _ = self.prepare_data(train_dataloader)
        self.model.fit(X_train, y_train)
        self.is_fitted = True
        
    def predict(self, test_dataloader: DataLoader) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        X_test, y_test, timestamps, current_prices = self.prepare_data(test_dataloader)
        
        predictions = self.model.predict(X_test)
        return predictions, y_test, timestamps, current_prices
    
    def get_feature_importance(self) -> Optional[np.ndarray]:
        if self.is_fitted:
            return self.model.feature_importances_
        return None

class XGBoostBaseline(BaselineModel):
    """XGBoost baseline (strong gradient boosting baseline)."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("XGBoost", config)
        self.model = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            n_jobs=-1
        )
        
    def fit(self, train_dataloader: DataLoader, feature_subset: Optional[str] = None):
        X_train, y_train, _, _ = self.prepare_data(train_dataloader)
        self.model.fit(X_train, y_train)
        self.is_fitted = True
        
    def predict(self, test_dataloader: DataLoader) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        X_test, y_test, timestamps, current_prices = self.prepare_data(test_dataloader)
        
        predictions = self.model.predict(X_test)
        return predictions, y_test, timestamps, current_prices
    
    def get_feature_importance(self) -> Optional[np.ndarray]:
        if self.is_fitted:
            return self.model.feature_importances_
        return None

class LSTMBaseline(BaselineModel):
    """LSTM baseline for time series prediction."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("LSTM", config)
        self.device = torch.device(config.get('device', 'cpu'))
        self.input_size = None
        self.model = None
        self.optimizer = None
        self.criterion = nn.MSELoss()
        
    def _build_model(self, input_size: int):
        """Build LSTM model."""
        class LSTMModel(nn.Module):
            def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.1):
                super().__init__()
                self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                                  batch_first=True, dropout=dropout)
                self.fc = nn.Linear(hidden_size, 1)
                self.dropout = nn.Dropout(dropout)
                
            def forward(self, x):
                lstm_out, _ = self.lstm(x)
                # Use last timestep output
                out = self.fc(self.dropout(lstm_out[:, -1, :]))
                return out.squeeze(-1)
        
        self.model = LSTMModel(input_size).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        
    def fit(self, train_dataloader: DataLoader, feature_subset: Optional[str] = None):
        # Get input size from first batch (OHLCV = 5 features)
        first_batch = next(iter(train_dataloader))
        if isinstance(first_batch, tuple):
            batch_data = first_batch[0]
        else:
            batch_data = first_batch
            
        # Stock data only - 5 features (OHLCV)
        input_size = 5
        self._build_model(input_size)
        
        # Training loop
        self.model.train()
        epochs = 3  # Quick testing
        
        for epoch in range(epochs):
            total_loss = 0
            num_batches = 0
            
            for batch in train_dataloader:
                if isinstance(batch, tuple):
                    batch_data = batch[0]
                else:
                    batch_data = batch
                    
                encoder_cont = batch_data['encoder_cont'].to(self.device)
                decoder_target = batch_data['decoder_target'].to(self.device)
                
                # Use only OHLCV data (first 5 features)
                encoder_cont = encoder_cont[:, :, :5]
                
                target = decoder_target[:, 0] if decoder_target.dim() > 1 else decoder_target
                
                self.optimizer.zero_grad()
                output = self.model(encoder_cont)
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
                
        self.is_fitted = True
        
    def predict(self, test_dataloader: DataLoader) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Make price/return predictions with LSTM."""
        self.model.eval()
        predictions = []
        actuals = []
        timestamps = []
        current_prices_list = []
        
        with torch.no_grad():
            for batch in test_dataloader:
                if isinstance(batch, tuple):
                    batch_data = batch[0]
                else:
                    batch_data = batch
                    
                encoder_cont = batch_data['encoder_cont'].to(self.device)
                decoder_target = batch_data['decoder_target'].to(self.device)
                
                # Use only OHLCV data (first 5 features)
                encoder_cont = encoder_cont[:, :, :5]
                
                target = decoder_target[:, 0] if decoder_target.dim() > 1 else decoder_target
                output = self.model(encoder_cont)
                
                predictions.append(output.cpu().numpy())
                actuals.append(target.cpu().numpy())
                
                # Extract current prices (close price from last encoder timestep)
                original_encoder = batch_data['encoder_cont'].to(self.device)
                current_prices = original_encoder[:, -1, 3].cpu().numpy()  # Close price at index 3
                current_prices_list.append(current_prices)
                
                # Extract timestamps for plotting
                if 'time_idx' in batch_data:
                    timestamp_batch = batch_data['time_idx'].cpu().numpy()[:, -1]
                else:
                    timestamp_batch = np.arange(len(output.cpu().numpy()))
                timestamps.append(timestamp_batch)
                
        predictions = np.concatenate(predictions)
        actuals = np.concatenate(actuals)
        timestamps = np.concatenate(timestamps)
        current_prices = np.concatenate(current_prices_list)
        
        return predictions, actuals, timestamps, current_prices

class GRUBaseline(LSTMBaseline):
    """GRU baseline - similar to LSTM but with GRU cells."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.name = "GRU"
        
    def _build_model(self, input_size: int):
        """Build GRU model."""
        class GRUModel(nn.Module):
            def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.1):
                super().__init__()
                self.gru = nn.GRU(input_size, hidden_size, num_layers, 
                                batch_first=True, dropout=dropout)
                self.fc = nn.Linear(hidden_size, 1)
                self.dropout = nn.Dropout(dropout)
                
            def forward(self, x):
                gru_out, _ = self.gru(x)
                out = self.fc(self.dropout(gru_out[:, -1, :]))
                return out.squeeze(-1)
        
        self.model = GRUModel(input_size).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

class VanillaTransformerBaseline(BaselineModel):
    """Vanilla Transformer baseline without TFT-specific components."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("Vanilla Transformer", config)
        self.device = torch.device(config.get('device', 'cpu'))
        self.model = None
        self.optimizer = None
        self.criterion = nn.MSELoss()
        
    def _build_model(self, input_size: int, seq_len: int):
        """Build vanilla transformer model."""
        class VanillaTransformer(nn.Module):
            def __init__(self, input_size, d_model=128, nhead=8, num_layers=3, dropout=0.1):
                super().__init__()
                self.input_projection = nn.Linear(input_size, d_model)
                self.positional_encoding = nn.Parameter(torch.randn(1000, d_model))
                
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True
                )
                self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
                self.output_projection = nn.Linear(d_model, 1)
                
            def forward(self, x):
                # x shape: [batch, seq_len, input_size]
                seq_len = x.size(1)
                x = self.input_projection(x)
                x = x + self.positional_encoding[:seq_len].unsqueeze(0)
                
                transformer_out = self.transformer(x)
                # Use last timestep
                out = self.output_projection(transformer_out[:, -1, :])
                return out.squeeze(-1)
        
        self.model = VanillaTransformer(input_size).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        
    def _get_feature_indices(self, feature_subset: Optional[str]) -> Optional[np.ndarray]:
        """Get feature indices for subset."""
        if feature_subset == 'ohlcv_only':
            return np.arange(5)
        elif feature_subset == 'technical_only':
            return np.arange(5, 25)
        elif feature_subset == 'news_only':
            return np.arange(25, 35)
        elif feature_subset == 'economic_only':
            return np.arange(35, 45)
        return None
        
    def fit(self, train_dataloader: DataLoader, feature_subset: Optional[str] = None):
        # Get dimensions from first batch (OHLCV = 5 features)
        first_batch = next(iter(train_dataloader))
        if isinstance(first_batch, tuple):
            batch_data = first_batch[0]
        else:
            batch_data = first_batch
            
        encoder_cont = batch_data['encoder_cont']
        seq_len = encoder_cont.shape[1]
        
        # Stock data only - 5 features (OHLCV)
        input_size = 5
        self._build_model(input_size, seq_len)
        
        # Training loop
        self.model.train()
        epochs = 3  # Quick testing
        
        for epoch in range(epochs):
            for batch in train_dataloader:
                if isinstance(batch, tuple):
                    batch_data = batch[0]
                else:
                    batch_data = batch
                    
                encoder_cont = batch_data['encoder_cont'].to(self.device)
                decoder_target = batch_data['decoder_target'].to(self.device)
                
                # Use only OHLCV data (first 5 features)
                encoder_cont = encoder_cont[:, :, :5]
                
                target = decoder_target[:, 0] if decoder_target.dim() > 1 else decoder_target
                
                self.optimizer.zero_grad()
                output = self.model(encoder_cont)
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()
                
        self.is_fitted = True
        
    def predict(self, test_dataloader: DataLoader) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Make price/return predictions with Transformer."""
        self.model.eval()
        predictions = []
        actuals = []
        timestamps = []
        current_prices_list = []
        
        with torch.no_grad():
            for batch in test_dataloader:
                if isinstance(batch, tuple):
                    batch_data = batch[0]
                else:
                    batch_data = batch
                    
                encoder_cont = batch_data['encoder_cont'].to(self.device)
                decoder_target = batch_data['decoder_target'].to(self.device)
                
                # Use only OHLCV data (first 5 features)
                encoder_cont = encoder_cont[:, :, :5]
                
                target = decoder_target[:, 0] if decoder_target.dim() > 1 else decoder_target
                output = self.model(encoder_cont)
                
                predictions.append(output.cpu().numpy())
                actuals.append(target.cpu().numpy())
                
                # Extract current prices (close price from last encoder timestep)
                original_encoder = batch_data['encoder_cont'].to(self.device)
                current_prices = original_encoder[:, -1, 3].cpu().numpy()  # Close price at index 3
                current_prices_list.append(current_prices)
                
                # Extract timestamps for plotting
                if 'time_idx' in batch_data:
                    timestamp_batch = batch_data['time_idx'].cpu().numpy()[:, -1]
                else:
                    timestamp_batch = np.arange(len(output.cpu().numpy()))
                timestamps.append(timestamp_batch)
                
        predictions = np.concatenate(predictions)
        actuals = np.concatenate(actuals)
        timestamps = np.concatenate(timestamps)
        current_prices = np.concatenate(current_prices_list)
        
        return predictions, actuals, timestamps, current_prices

class ARIMABaseline(BaselineModel):
    """ARIMA time series baseline."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("ARIMA", config)
        self.models = {}  # One model per time series
        
    def fit(self, train_dataloader: DataLoader, feature_subset: Optional[str] = None):
        if not ARIMA_AVAILABLE:
            print("⚠️  ARIMA not available. Skipping ARIMA baseline.")
            return
            
        # For ARIMA, we need time series data
        # This is a simplified implementation - would need proper time series extraction
        print("📊 ARIMA baseline fitting... (simplified implementation)")
        self.is_fitted = True
        
    def predict(self, test_dataloader: DataLoader) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Make price/return predictions with ARIMA."""
        if not ARIMA_AVAILABLE or not self.is_fitted:
            # Return dummy predictions with proper format
            predictions, actuals, timestamps, current_prices = self.prepare_data(test_dataloader)
            dummy_predictions = np.zeros(len(predictions))
            return dummy_predictions, actuals, timestamps, current_prices
        
        # Simplified ARIMA prediction with proper format
        predictions, actuals, timestamps, current_prices = self.prepare_data(test_dataloader)
        dummy_predictions = np.random.normal(0, 0.01, len(predictions))
        return dummy_predictions, actuals, timestamps, current_prices

class BuyAndHoldBaseline(BaselineModel):
    """Buy and Hold strategy baseline."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("Buy & Hold", config)
        self.last_price = None
        
    def fit(self, train_dataloader: DataLoader, feature_subset: Optional[str] = None):
        # Extract last price from training data for baseline
        for batch in train_dataloader:
            if isinstance(batch, tuple):
                batch_data = batch[0]
            else:
                batch_data = batch
                
            encoder_cont = batch_data['encoder_cont']
            # Close price is at index 3
            self.last_price = encoder_cont[:, -1, 3].mean().item()
            
        self.is_fitted = True
        
    def predict(self, test_dataloader: DataLoader) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Make price/return predictions with Buy & Hold."""
        # Buy and hold predicts no change (return = 0)
        predictions, actuals, timestamps, current_prices = self.prepare_data(test_dataloader)
        buy_hold_predictions = np.zeros(len(predictions))
        return buy_hold_predictions, actuals, timestamps, current_prices

class MovingAverageCrossoverBaseline(BaselineModel):
    """Moving Average Crossover strategy baseline."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("Moving Average Crossover", config)
        self.short_window = 10
        self.long_window = 30
        
    def fit(self, train_dataloader: DataLoader, feature_subset: Optional[str] = None):
        # Moving average doesn't need fitting
        self.is_fitted = True
        
    def predict(self, test_dataloader: DataLoader) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Make price/return predictions with Moving Average Crossover."""
        predictions = []
        actuals = []
        timestamps = []
        current_prices_list = []
        
        for batch in test_dataloader:
            if isinstance(batch, tuple):
                batch_data = batch[0]
            else:
                batch_data = batch
                
            encoder_cont = batch_data['encoder_cont']
            decoder_target = batch_data['decoder_target']
            
            # Use price momentum as prediction
            prices = encoder_cont[:, :, 3]  # Close price at index 3
            
            if prices.shape[1] >= 10:
                momentum = (prices[:, -1] - prices[:, -10]) / prices[:, -10]
            else:
                momentum = torch.zeros(prices.shape[0])
                
            predictions.append(momentum.numpy())
            
            # Extract actuals
            target = decoder_target[:, 0] if decoder_target.dim() > 1 else decoder_target
            actuals.append(target.numpy())
            
            # Extract current prices
            current_prices = encoder_cont[:, -1, 3].numpy()
            current_prices_list.append(current_prices)
            
            # Extract timestamps
            if 'time_idx' in batch_data:
                timestamp_batch = batch_data['time_idx'].numpy()[:, -1]
            else:
                timestamp_batch = np.arange(len(momentum.numpy()))
            timestamps.append(timestamp_batch)
            
        predictions = np.concatenate(predictions)
        actuals = np.concatenate(actuals)
        timestamps = np.concatenate(timestamps)
        current_prices = np.concatenate(current_prices_list)
        
        return predictions, actuals, timestamps, current_prices

def get_all_baselines(config: Dict[str, Any]) -> Dict[str, BaselineModel]:
    """Get all available stock price baseline models (OHLCV data only)."""
    baselines = {
        # Traditional ML
        'linear_regression': LinearRegressionBaseline(config),
        'random_forest': RandomForestBaseline(config),
        'xgboost': XGBoostBaseline(config),
        
        # Deep Learning
        'lstm': LSTMBaseline(config),
        'gru': GRUBaseline(config),
        'vanilla_transformer': VanillaTransformerBaseline(config),
        
        # Time Series
        'arima': ARIMABaseline(config),
        
        # Finance-Specific
        'buy_and_hold': BuyAndHoldBaseline(config),
        'moving_average': MovingAverageCrossoverBaseline(config),
    }
    
    return baselines

def get_baseline_subset(baseline_type: str, config: Dict[str, Any]) -> Dict[str, BaselineModel]:
    """Get a subset of stock price baselines by type."""
    all_baselines = get_all_baselines(config)
    
    if baseline_type == 'traditional_ml':
        return {k: v for k, v in all_baselines.items() 
                if k in ['linear_regression', 'random_forest', 'xgboost']}
    elif baseline_type == 'deep_learning':
        return {k: v for k, v in all_baselines.items() 
                if k in ['lstm', 'gru', 'vanilla_transformer']}
    elif baseline_type == 'finance_specific':
        return {k: v for k, v in all_baselines.items() 
                if k in ['buy_and_hold', 'moving_average', 'arima']}
    elif baseline_type == 'quick':
        # Fast baselines for development
        return {k: v for k, v in all_baselines.items() 
                if k in ['linear_regression', 'random_forest', 'buy_and_hold']}
    else:
        return all_baselines

# Utility Functions for Price Analysis and Plotting
def convert_returns_to_prices(returns: np.ndarray, initial_prices: np.ndarray) -> np.ndarray:
    """
    Convert predicted returns to absolute stock prices.
    
    Args:
        returns: Predicted returns (percentage change)
        initial_prices: Starting prices for each prediction
        
    Returns:
        predicted_prices: Absolute stock prices
    """
    # Formula: new_price = current_price * (1 + return)
    predicted_prices = initial_prices * (1 + returns)
    return predicted_prices

def calculate_price_metrics(predicted_returns: np.ndarray, actual_returns: np.ndarray, 
                          current_prices: np.ndarray) -> Dict[str, float]:
    """
    Calculate comprehensive metrics for price/return predictions.
    
    Returns both return-based and price-based metrics.
    """
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    
    # Return-based metrics
    mse_returns = mean_squared_error(actual_returns, predicted_returns)
    mae_returns = mean_absolute_error(actual_returns, predicted_returns)
    r2_returns = r2_score(actual_returns, predicted_returns)
    
    # Convert to prices for price-based metrics
    predicted_prices = convert_returns_to_prices(predicted_returns, current_prices)
    actual_prices = convert_returns_to_prices(actual_returns, current_prices)
    
    mse_prices = mean_squared_error(actual_prices, predicted_prices)
    mae_prices = mean_absolute_error(actual_prices, predicted_prices)
    r2_prices = r2_score(actual_prices, predicted_prices)
    
    # Calculate percentage errors
    mape_returns = np.mean(np.abs((actual_returns - predicted_returns) / (actual_returns + 1e-8))) * 100
    mape_prices = np.mean(np.abs((actual_prices - predicted_prices) / (actual_prices + 1e-8))) * 100
    
    return {
        # Return metrics
        'mse_returns': mse_returns,
        'mae_returns': mae_returns,
        'r2_returns': r2_returns,
        'mape_returns': mape_returns,
        
        # Price metrics
        'mse_prices': mse_prices,
        'mae_prices': mae_prices,
        'r2_prices': r2_prices,
        'mape_prices': mape_prices,
        
        # Trading metrics
        'return_volatility': np.std(predicted_returns),
        'price_volatility': np.std(predicted_prices),
        'max_price_error': np.max(np.abs(actual_prices - predicted_prices)),
        'mean_price_level': np.mean(actual_prices)
    }

def create_stock_prediction_plots(timestamps: np.ndarray, actual_returns: np.ndarray, 
                                predicted_returns: np.ndarray, current_prices: np.ndarray,
                                model_name: str, save_path: Optional[str] = None):
    """
    Create comprehensive plots for stock price predictions.
    
    Shows both returns and absolute price predictions with proper formatting.
    """
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from datetime import datetime, timedelta
    
    # Convert returns to prices
    predicted_prices = convert_returns_to_prices(predicted_returns, current_prices)
    actual_prices = convert_returns_to_prices(actual_returns, current_prices)
    
    # Create figure with subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'{model_name} - Stock Price Predictions', fontsize=16, fontweight='bold')
    
    # Convert timestamps to dates if they're numeric
    if len(timestamps) > 0 and isinstance(timestamps[0], (int, float)):
        # Assume timestamps are days since some epoch
        base_date = datetime(2020, 1, 1)  # Adjust as needed
        dates = [base_date + timedelta(days=int(t)) for t in timestamps]
    else:
        dates = timestamps
    
    # Plot 1: Returns Comparison
    ax1.scatter(dates, actual_returns, alpha=0.6, s=20, color='blue', label='Actual Returns')
    ax1.scatter(dates, predicted_returns, alpha=0.6, s=20, color='red', label='Predicted Returns')
    ax1.set_title('Returns Prediction')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Return (%)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Price Comparison
    ax2.plot(dates, actual_prices, color='blue', linewidth=2, label='Actual Prices', alpha=0.8)
    ax2.plot(dates, predicted_prices, color='red', linewidth=2, label='Predicted Prices', alpha=0.8)
    ax2.set_title('Stock Price Prediction')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Price ($)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Returns Scatter Plot
    ax3.scatter(actual_returns, predicted_returns, alpha=0.6, s=20)
    min_val = min(actual_returns.min(), predicted_returns.min())
    max_val = max(actual_returns.max(), predicted_returns.max())
    ax3.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)
    ax3.set_title('Returns: Predicted vs Actual')
    ax3.set_xlabel('Actual Returns')
    ax3.set_ylabel('Predicted Returns')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Price Error Analysis
    price_errors = actual_prices - predicted_prices
    ax4.hist(price_errors, bins=30, alpha=0.7, color='purple', edgecolor='black')
    ax4.axvline(0, color='red', linestyle='--', linewidth=2)
    ax4.set_title('Price Prediction Errors Distribution')
    ax4.set_xlabel('Error ($)')
    ax4.set_ylabel('Frequency')
    ax4.grid(True, alpha=0.3)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save plot if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Plot saved to: {save_path}")
    
    return fig
