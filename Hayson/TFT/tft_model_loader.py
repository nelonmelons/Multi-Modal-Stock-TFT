#!/usr/bin/env python3
"""
TFT Model Loader and Predictor for Live Trading and OHLC Plotting

This module provides functionality to load trained TFT models and generate OHLC predictions
for live trading systems and visualization.
"""

import os
import sys
import json
import torch
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
import logging

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from tft_multimodal import TFT, EnhancedTFT
    TFTMultiModal = EnhancedTFT  # Use EnhancedTFT as the main model class
except ImportError:
    # Fallback for testing
    TFT = None
    EnhancedTFT = None
    TFTMultiModal = None
from dataModule.interface import get_data_loader_with_module

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TFTModelLoader:
    """
    Handles loading of trained TFT models from checkpoints.
    """
    
    def __init__(self, checkpoint_path: str, config_path: Optional[str] = None):
        """
        Initialize the TFT model loader.
        
        Args:
            checkpoint_path: Path to the model checkpoint (.pth file)
            config_path: Path to the configuration file (optional)
        """
        self.checkpoint_path = checkpoint_path
        self.config_path = config_path or self._find_config_path()
        self.config = self._load_config()
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
        
    def _find_config_path(self) -> Optional[str]:
        """Find the config.json file in the same run directory."""
        run_dir = os.path.dirname(os.path.dirname(self.checkpoint_path))
        config_path = os.path.join(run_dir, 'config.json')
        return config_path if os.path.exists(config_path) else None
        
    def _load_config(self) -> Dict[str, Any]:
        """Load the configuration from JSON file."""
        if self.config_path and os.path.exists(self.config_path):
            with open(self.config_path, 'r') as f:
                return json.load(f)
        else:
            # Default configuration if not found
            return {
                'hidden_size': 128,
                'num_heads': 8,
                'dropout': 0.1,
                'encoder_len': 60,
                'predict_len': 10,
                'enhanced_model': True,
                'max_input_features': 50
            }
    
    def load_model(self) -> Any:
        """
        Load the TFT model from checkpoint.
        
        Returns:
            Loaded TFT model
        """
        if not os.path.exists(self.checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")
            
        # Load checkpoint
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        
        # Extract model configuration
        input_dim = checkpoint.get('input_dim', 20)  # Default fallback
        news_dim = checkpoint.get('news_dim', 384)
        events_dim = checkpoint.get('events_dim', 10)
        
        # Initialize model
        if TFTMultiModal is not None:
            self.model = TFTMultiModal(
                input_size=input_dim,
                news_dim=news_dim,
                hidden_size=self.config['hidden_size'],
                num_heads=self.config['num_heads'],
                dropout=self.config['dropout'],
                seq_len=self.config['encoder_len'],
                prediction_len=self.config['predict_len']
            )
        else:
            raise ImportError("TFT model classes not available")
        
        # Load model state
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        logger.info(f"Loaded TFT model from {self.checkpoint_path}")
        return self.model
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        return {
            'checkpoint_path': self.checkpoint_path,
            'config': self.config,
            'device': str(self.device),
            'model_loaded': self.model is not None
        }

class TFTPredictor:
    """
    Handles predictions using the loaded TFT model.
    """
    
    def __init__(self, model_loader: TFTModelLoader):
        """
        Initialize the TFT predictor.
        
        Args:
            model_loader: Loaded TFT model loader
        """
        self.model_loader = model_loader
        self.model = model_loader.model
        self.config = model_loader.config
        self.device = model_loader.device
        
        if self.model is None:
            raise ValueError("Model must be loaded before creating predictor")
    
    def predict_ohlc(self, 
                     input_data: torch.Tensor,
                     news_data: Optional[torch.Tensor] = None,
                     events_data: Optional[torch.Tensor] = None) -> Dict[str, np.ndarray]:
        """
        Generate OHLC predictions using the TFT model.
        
        Args:
            input_data: Historical price and technical indicator data
            news_data: Optional news embedding data
            events_data: Optional events data
            
        Returns:
            Dictionary containing OHLC predictions
        """
        if self.model is None:
            raise ValueError("Model must be loaded before making predictions")
            
        self.model.eval()
        with torch.no_grad():
            # Ensure input is on correct device
            input_data = input_data.to(self.device)
            
            # Handle optional inputs
            if news_data is not None:
                news_data = news_data.to(self.device)
            if events_data is not None:
                events_data = events_data.to(self.device)
            
            # Get predictions
            predictions = self.model(input_data, news_data, events_data)
            
            # Convert to numpy
            predictions = predictions.cpu().numpy()
            
            # Assume predictions are in format [batch, time, features]
            # where features might be [open, high, low, close] or similar
            batch_size, pred_len, n_features = predictions.shape
            
            if n_features >= 4:
                # If we have 4+ features, assume first 4 are OHLC
                ohlc_predictions = {
                    'open': predictions[:, :, 0],
                    'high': predictions[:, :, 1],
                    'low': predictions[:, :, 2],
                    'close': predictions[:, :, 3]
                }
            else:
                # If fewer features, replicate close price for missing components
                close_pred = predictions[:, :, 0]  # Assume first feature is close
                ohlc_predictions = {
                    'open': close_pred,
                    'high': close_pred * 1.02,  # Assume 2% higher for high
                    'low': close_pred * 0.98,   # Assume 2% lower for low
                    'close': close_pred
                }
            
            return ohlc_predictions
    
    def generate_trading_signals(self, 
                                input_data: torch.Tensor,
                                news_data: Optional[torch.Tensor] = None,
                                events_data: Optional[torch.Tensor] = None,
                                current_price: Optional[float] = None) -> Dict[str, Any]:
        """
        Generate trading signals based on TFT predictions.
        
        Args:
            input_data: Historical data for prediction
            news_data: Optional news data
            events_data: Optional events data
            current_price: Current market price
            
        Returns:
            Dictionary containing trading signals and analysis
        """
        # Get OHLC predictions
        ohlc_pred = self.predict_ohlc(input_data, news_data, events_data)
        
        # Calculate price direction and momentum
        close_pred = ohlc_pred['close'][0]  # First batch
        
        # Price trend analysis
        price_trend = np.diff(close_pred)
        momentum = np.mean(price_trend)
        volatility = np.std(price_trend)
        
        # Generate signal
        if momentum > 0.01:  # 1% positive momentum
            signal = 'BUY'
            confidence = min(abs(momentum) * 10, 1.0)
        elif momentum < -0.01:  # 1% negative momentum
            signal = 'SELL'
            confidence = min(abs(momentum) * 10, 1.0)
        else:
            signal = 'HOLD'
            confidence = 0.5
        
        # Risk assessment
        risk_level = 'LOW' if volatility < 0.02 else 'MEDIUM' if volatility < 0.05 else 'HIGH'
        
        # Position sizing (simplified Kelly criterion)
        win_prob = 0.5 + (confidence - 0.5) * 0.3  # Scale confidence to probability
        kelly_fraction = max(0.01, min(0.25, (win_prob - 0.5) / 0.5))  # Conservative Kelly
        
        return {
            'signal': signal,
            'confidence': confidence,
            'risk_level': risk_level,
            'kelly_fraction': kelly_fraction,
            'predicted_returns': momentum,
            'volatility': volatility,
            'price_predictions': {
                'next_close': close_pred[-1],
                'trend': 'BULLISH' if momentum > 0 else 'BEARISH',
                'ohlc_forecast': ohlc_pred
            }
        }
    
    def batch_predict(self, 
                     data_loader,
                     include_actual: bool = True) -> Dict[str, List]:
        """
        Generate predictions for a batch of data.
        
        Args:
            data_loader: Data loader with test data
            include_actual: Whether to include actual values for comparison
            
        Returns:
            Dictionary containing predictions and optionally actual values
        """
        predictions = []
        actuals = []
        
        if self.model is None:
            raise ValueError("Model must be loaded before making predictions")
        
        self.model.eval()
        with torch.no_grad():
            for batch in data_loader:
                if len(batch) >= 4:
                    inputs, news, events, targets = batch[:4]
                else:
                    inputs, targets = batch[0], batch[-1]
                    news, events = None, None
                
                # Get predictions
                pred = self.model(inputs.to(self.device), 
                                news.to(self.device) if news is not None else None,
                                events.to(self.device) if events is not None else None)
                
                predictions.append(pred.cpu().numpy())
                if include_actual:
                    actuals.append(targets.cpu().numpy())
        
        result = {'predictions': predictions}
        if include_actual:
            result['actuals'] = actuals
            
        return result

def get_latest_model_path(runs_dir: str = "runs") -> str:
    """
    Get the path to the latest trained model.
    
    Args:
        runs_dir: Directory containing training runs
        
    Returns:
        Path to the best model checkpoint
    """
    runs_path = os.path.join(os.path.dirname(__file__), runs_dir)
    if not os.path.exists(runs_path):
        raise FileNotFoundError(f"Runs directory not found: {runs_path}")
    
    # Get all run directories
    run_dirs = [d for d in os.listdir(runs_path) if d.startswith('tft_run_')]
    if not run_dirs:
        raise FileNotFoundError("No training runs found")
    
    # Sort by timestamp (assuming format tft_run_YYYYMMDD_HHMMSS)
    run_dirs.sort(reverse=True)
    
    # Check for best model in the latest run
    latest_run = os.path.join(runs_path, run_dirs[0])
    best_model_path = os.path.join(latest_run, 'checkpoints', 'best_model.pth')
    
    if os.path.exists(best_model_path):
        return best_model_path
    else:
        # Fallback to final model
        final_model_path = os.path.join(latest_run, 'checkpoints', 'final_model.pth')
        if os.path.exists(final_model_path):
            return final_model_path
        else:
            raise FileNotFoundError(f"No model checkpoints found in {latest_run}")

# Example usage
if __name__ == "__main__":
    # Load the latest model
    try:
        checkpoint_path = get_latest_model_path()
        print(f"Loading model from: {checkpoint_path}")
        
        # Initialize model loader
        model_loader = TFTModelLoader(checkpoint_path)
        model = model_loader.load_model()
        
        # Create predictor
        predictor = TFTPredictor(model_loader)
        
        # Example prediction with dummy data
        batch_size = 1
        encoder_len = model_loader.config['encoder_len']
        input_dim = 20  # Adjust based on your data
        
        dummy_input = torch.randn(batch_size, encoder_len, input_dim)
        dummy_news = torch.randn(batch_size, 384)  # News embedding dimension
        dummy_events = torch.randn(batch_size, 10)  # Events dimension
        
        # Generate OHLC predictions
        ohlc_pred = predictor.predict_ohlc(dummy_input, dummy_news, dummy_events)
        print("OHLC Predictions shape:", {k: v.shape for k, v in ohlc_pred.items()})
        
        # Generate trading signals
        signals = predictor.generate_trading_signals(dummy_input, dummy_news, dummy_events)
        print("Trading Signal:", signals['signal'])
        print("Confidence:", signals['confidence'])
        print("Risk Level:", signals['risk_level'])
        
    except Exception as e:

        print("Please ensure you have trained models in the runs/ directory")
