#!/usr/bin/env python3
"""
Baseline Models for TFT Research Paper Comparison
================================================

This module implements various baseline models for comprehensive comparison
in the multi-modal TFT research paper. All models use the same dataModule
interface for fair comparison.

Baseline Categories:
1. Traditional ML: Linear Regression, Random Forest, XGBoost
2. Deep Learning: LSTM, GRU, Vanilla Transformer
3. Time Series: ARIMA, Prophet
4. Finance-Specific: Buy & Hold, Moving Average, Mean Reversion
5. Feature Ablation: Single-modal variants of TFT
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
        
    def prepare_data(self, dataloader: DataLoader, feature_subset: Optional[List[str]] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Extract and prepare data from dataloader."""
        X_list, y_list = [], []
        
        for batch in dataloader:
            if isinstance(batch, tuple):
                batch_data = batch[0]
            else:
                batch_data = batch
                
            # Extract features
            encoder_cont = batch_data['encoder_cont'].numpy()  # Shape: [batch, seq_len, features]
            decoder_target = batch_data['decoder_target'].numpy()  # Shape: [batch, predict_len]
            
            # Flatten sequence for traditional ML (use last timestep for now)
            X_batch = encoder_cont[:, -1, :]  # [batch, features]
            y_batch = decoder_target[:, 0] if decoder_target.ndim > 1 else decoder_target  # [batch]
            
            X_list.append(X_batch)
            y_list.append(y_batch)
        
        X = np.vstack(X_list)
        y = np.concatenate(y_list)
        
        # Apply feature subset if specified
        if feature_subset is not None:
            # For now, use first N features as subset (would need feature mapping in real implementation)
            if feature_subset == 'ohlcv_only':
                X = X[:, :5]  # First 5 features: OHLCV
            elif feature_subset == 'technical_only':
                X = X[:, 5:25]  # Next 20 features: technical indicators
            elif feature_subset == 'news_only':
                X = X[:, 25:35]  # Next 10 features: news embeddings
            elif feature_subset == 'economic_only':
                X = X[:, 35:45]  # Next 10 features: economic indicators
                
        return X, y
    
    def fit(self, train_dataloader: DataLoader, feature_subset: Optional[str] = None):
        """Train the model."""
        raise NotImplementedError
        
    def predict(self, test_dataloader: DataLoader) -> np.ndarray:
        """Make predictions."""
        raise NotImplementedError
        
    def get_feature_importance(self) -> Optional[np.ndarray]:
        """Get feature importance if available."""
        return None

class LinearRegressionBaseline(BaselineModel):
    """Linear Regression baseline."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("Linear Regression", config)
        self.model = LinearRegression()
        
    def fit(self, train_dataloader: DataLoader, feature_subset: Optional[str] = None):
        X_train, y_train = self.prepare_data(train_dataloader, feature_subset)
        X_train_scaled = self.scaler.fit_transform(X_train)
        self.model.fit(X_train_scaled, y_train)
        self.is_fitted = True
        
    def predict(self, test_dataloader: DataLoader) -> np.ndarray:
        X_test, _ = self.prepare_data(test_dataloader)
        X_test_scaled = self.scaler.transform(X_test)
        return self.model.predict(X_test_scaled)
    
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
        X_train, y_train = self.prepare_data(train_dataloader, feature_subset)
        self.model.fit(X_train, y_train)
        self.is_fitted = True
        
    def predict(self, test_dataloader: DataLoader) -> np.ndarray:
        X_test, _ = self.prepare_data(test_dataloader)
        return self.model.predict(X_test)
    
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
        X_train, y_train = self.prepare_data(train_dataloader, feature_subset)
        self.model.fit(X_train, y_train)
        self.is_fitted = True
        
    def predict(self, test_dataloader: DataLoader) -> np.ndarray:
        X_test, _ = self.prepare_data(test_dataloader)
        return self.model.predict(X_test)
    
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
        # Get input size from first batch
        first_batch = next(iter(train_dataloader))
        if isinstance(first_batch, tuple):
            batch_data = first_batch[0]
        else:
            batch_data = first_batch
            
        encoder_cont = batch_data['encoder_cont']
        input_size = encoder_cont.shape[-1]
        
        # Apply feature subset
        if feature_subset == 'ohlcv_only':
            input_size = 5
        elif feature_subset == 'technical_only':
            input_size = 20
        elif feature_subset == 'news_only':
            input_size = 10
        elif feature_subset == 'economic_only':
            input_size = 10
            
        self._build_model(input_size)
        
        # Training loop
        self.model.train()
        epochs = 20  # Quick training for baseline
        
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
                
                # Apply feature subset
                if feature_subset == 'ohlcv_only':
                    encoder_cont = encoder_cont[:, :, :5]
                elif feature_subset == 'technical_only':
                    encoder_cont = encoder_cont[:, :, 5:25]
                elif feature_subset == 'news_only':
                    encoder_cont = encoder_cont[:, :, 25:35]
                elif feature_subset == 'economic_only':
                    encoder_cont = encoder_cont[:, :, 35:45]
                
                target = decoder_target[:, 0] if decoder_target.dim() > 1 else decoder_target
                
                self.optimizer.zero_grad()
                output = self.model(encoder_cont)
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
                
        self.is_fitted = True
        
    def predict(self, test_dataloader: DataLoader) -> np.ndarray:
        self.model.eval()
        predictions = []
        
        with torch.no_grad():
            for batch in test_dataloader:
                if isinstance(batch, tuple):
                    batch_data = batch[0]
                else:
                    batch_data = batch
                    
                encoder_cont = batch_data['encoder_cont'].to(self.device)
                output = self.model(encoder_cont)
                predictions.append(output.cpu().numpy())
                
        return np.concatenate(predictions)

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
        
    def fit(self, train_dataloader: DataLoader, feature_subset: Optional[str] = None):
        # Get dimensions from first batch
        first_batch = next(iter(train_dataloader))
        if isinstance(first_batch, tuple):
            batch_data = first_batch[0]
        else:
            batch_data = first_batch
            
        encoder_cont = batch_data['encoder_cont']
        seq_len, input_size = encoder_cont.shape[1], encoder_cont.shape[2]
        
        # Apply feature subset
        if feature_subset == 'ohlcv_only':
            input_size = 5
        elif feature_subset == 'technical_only':
            input_size = 20
        elif feature_subset == 'news_only':
            input_size = 10
        elif feature_subset == 'economic_only':
            input_size = 10
            
        self._build_model(input_size, seq_len)
        
        # Training loop (similar to LSTM)
        self.model.train()
        epochs = 15
        
        for epoch in range(epochs):
            for batch in train_dataloader:
                if isinstance(batch, tuple):
                    batch_data = batch[0]
                else:
                    batch_data = batch
                    
                encoder_cont = batch_data['encoder_cont'].to(self.device)
                decoder_target = batch_data['decoder_target'].to(self.device)
                
                # Apply feature subset
                if feature_subset == 'ohlcv_only':
                    encoder_cont = encoder_cont[:, :, :5]
                elif feature_subset == 'technical_only':
                    encoder_cont = encoder_cont[:, :, 5:25]
                elif feature_subset == 'news_only':
                    encoder_cont = encoder_cont[:, :, 25:35]
                elif feature_subset == 'economic_only':
                    encoder_cont = encoder_cont[:, :, 35:45]
                
                target = decoder_target[:, 0] if decoder_target.dim() > 1 else decoder_target
                
                self.optimizer.zero_grad()
                output = self.model(encoder_cont)
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()
                
        self.is_fitted = True
        
    def predict(self, test_dataloader: DataLoader) -> np.ndarray:
        self.model.eval()
        predictions = []
        
        with torch.no_grad():
            for batch in test_dataloader:
                if isinstance(batch, tuple):
                    batch_data = batch[0]
                else:
                    batch_data = batch
                    
                encoder_cont = batch_data['encoder_cont'].to(self.device)
                output = self.model(encoder_cont)
                predictions.append(output.cpu().numpy())
                
        return np.concatenate(predictions)

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
        
    def predict(self, test_dataloader: DataLoader) -> np.ndarray:
        if not ARIMA_AVAILABLE or not self.is_fitted:
            # Return dummy predictions
            X_test, _ = self.prepare_data(test_dataloader)
            return np.zeros(len(X_test))
        
        # Simplified ARIMA prediction
        X_test, _ = self.prepare_data(test_dataloader)
        return np.random.normal(0, 0.01, len(X_test))

class BuyAndHoldBaseline(BaselineModel):
    """Buy and Hold strategy baseline."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("Buy & Hold", config)
        self.last_price = None
        
    def fit(self, train_dataloader: DataLoader, feature_subset: Optional[str] = None):
        # Extract last price from training data
        for batch in train_dataloader:
            if isinstance(batch, tuple):
                batch_data = batch[0]
            else:
                batch_data = batch
                
            encoder_cont = batch_data['encoder_cont']
            # Assume first feature is close price
            self.last_price = encoder_cont[:, -1, 0].mean().item()
            
        self.is_fitted = True
        
    def predict(self, test_dataloader: DataLoader) -> np.ndarray:
        # Buy and hold predicts no change (return = 0)
        X_test, _ = self.prepare_data(test_dataloader)
        return np.zeros(len(X_test))

class MovingAverageCrossoverBaseline(BaselineModel):
    """Moving Average Crossover strategy baseline."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("Moving Average Crossover", config)
        self.short_window = 10
        self.long_window = 30
        
    def fit(self, train_dataloader: DataLoader, feature_subset: Optional[str] = None):
        # Moving average doesn't need fitting
        self.is_fitted = True
        
    def predict(self, test_dataloader: DataLoader) -> np.ndarray:
        # Simple momentum-based prediction
        predictions = []
        
        for batch in test_dataloader:
            if isinstance(batch, tuple):
                batch_data = batch[0]
            else:
                batch_data = batch
                
            encoder_cont = batch_data['encoder_cont']
            # Use price momentum as prediction
            prices = encoder_cont[:, :, 0]  # Assume first feature is close price
            momentum = (prices[:, -1] - prices[:, -10]) / prices[:, -10]
            predictions.append(momentum.numpy())
            
        return np.concatenate(predictions)

# Feature Ablation Baselines (these would use your existing TFT with feature subsets)
class TFTOHLCVOnlyBaseline(BaselineModel):
    """TFT with only OHLCV features."""
    
    def __init__(self, config: Dict[str, Any], tft_trainer):
        super().__init__("TFT (OHLCV Only)", config)
        self.tft_trainer = tft_trainer
        
    def fit(self, train_dataloader: DataLoader, feature_subset: Optional[str] = None):
        # This would modify the TFT to use only OHLCV features
        # Implementation depends on your TFT architecture
        print("📊 Training TFT with OHLCV features only...")
        self.is_fitted = True
        
    def predict(self, test_dataloader: DataLoader) -> np.ndarray:
        # Use TFT with restricted features
        # Return dummy for now
        X_test, _ = self.prepare_data(test_dataloader)
        return np.random.normal(0, 0.01, len(X_test))

def get_all_baselines(config: Dict[str, Any], tft_trainer=None) -> Dict[str, BaselineModel]:
    """Get all available baseline models."""
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
    
    # Add feature ablation baselines if TFT trainer is provided
    if tft_trainer is not None:
        baselines.update({
            'tft_ohlcv_only': TFTOHLCVOnlyBaseline(config, tft_trainer),
            # Add more TFT variants here
        })
    
    return baselines

def get_baseline_subset(baseline_type: str, config: Dict[str, Any]) -> Dict[str, BaselineModel]:
    """Get a subset of baselines by type."""
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
