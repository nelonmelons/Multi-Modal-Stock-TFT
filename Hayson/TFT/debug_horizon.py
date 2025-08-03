#!/usr/bin/env python3
"""
Simple test script to debug horizon-based predictions
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Any

# Simplified model for testing
class TestLSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, predict_len: int):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, predict_len)
        self.predict_len = predict_len
    
    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        x = batch['x_past_features']
        _, (h_n, _) = self.lstm(x)
        return self.fc(h_n[-1])

def test_horizon_prediction():
    """Test horizon-based prediction with simple data."""
    print("=== Testing Horizon-Based Prediction ===")
    
    # Test parameters
    batch_size = 32
    input_size = 10
    hidden_size = 16
    encoder_len = 60
    predict_lens = [1, 5, 15, 30]
    
    for predict_len in predict_lens:
        print(f"\n--- Testing horizon length: {predict_len} ---")
        
        # Create model
        model = TestLSTM(input_size, hidden_size, predict_len)
        
        # Create synthetic data
        # Input features (batch_size, encoder_len, input_size)
        x_features = torch.randn(batch_size, encoder_len, input_size)
        
        # Create realistic price targets 
        # Simulate price changes that are small relative to price level
        base_price = 100.0
        daily_returns = torch.randn(batch_size, predict_len) * 0.02  # 2% daily volatility
        
        # Convert returns to absolute prices
        prices = torch.zeros(batch_size, predict_len)
        prices[:, 0] = base_price * (1 + daily_returns[:, 0])
        for i in range(1, predict_len):
            prices[:, i] = prices[:, i-1] * (1 + daily_returns[:, i])
        
        # Test forward pass
        batch = {'x_past_features': x_features}
        predictions = model(batch)
        
        print(f"  Input shape: {x_features.shape}")
        print(f"  Prediction shape: {predictions.shape}")
        print(f"  Target shape: {prices.shape}")
        print(f"  Expected shape: ({batch_size}, {predict_len})")
        
        # Test loss calculation
        criterion = nn.MSELoss()
        loss = criterion(predictions, prices)
        print(f"  Loss: {loss.item():.6f}")
        
        # Test scaling issues
        pred_range = torch.max(predictions) - torch.min(predictions)
        target_range = torch.max(prices) - torch.min(prices)
        print(f"  Prediction range: {pred_range.item():.4f}")
        print(f"  Target range: {target_range.item():.4f}")
        print(f"  Range ratio: {pred_range.item() / target_range.item():.4f}")
        
        # Check if model outputs are reasonable
        if pred_range < target_range * 0.01:
            print(f"  ⚠️  WARNING: Predictions may be too small relative to targets")
        elif pred_range > target_range * 100:
            print(f"  ⚠️  WARNING: Predictions may be too large relative to targets")
        else:
            print(f"  ✓ Prediction scale looks reasonable")

if __name__ == "__main__":
    test_horizon_prediction()
