#!/usr/bin/env python3
"""
Unified TFT Pipeline: Run this script to execute the full, robust, and interpretable TFT pipeline (data loading, training, prediction, interpretability, trading simulation, and plotting).
"""

print("Starting script...")

import os
import sys
from datetime import datetime, timedelta
import warnings

import torch
import torch.nn as nn
warnings.filterwarnings('ignore')

print("Basic imports done...")

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

print("About to import dataModule...")
try:
    from dataModule.interface import get_data_loader_with_module
    print("dataModule imported successfully")
except Exception as e:
    print(f"Error importing dataModule: {e}")

print("About to import tft_pure_torch_m1...")
try:
    from tft_multimodal import TFT, setup_device, prepare_data_for_training, train_model, generate_predictions, create_visualizations, simulate_trading
    print("tft_pure_torch_m1 imported successfully")
except Exception as e:
    print(f"Error importing tft_pure_torch_m1: {e}")


def main():
    print("\n🚀 UNIFIED TFT PIPELINE (ENHANCED)")
    device = setup_device()
    # Diverse dataset: more symbols, longer range
    symbols = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'META', 'TSLA', 'NVDA', 'JPM', 'V', 'UNH']
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=365*2)).strftime('%Y-%m-%d')
    encoder_len = 30
    predict_len = 3
    batch_size = 32
    print(f"Symbols: {symbols}")
    print(f"Date range: {start_date} to {end_date}")
    dataloader, datamodule = get_data_loader_with_module(
        symbols=symbols,
        start=start_date,
        end=end_date,
        encoder_len=encoder_len,
        predict_len=predict_len,
        batch_size=batch_size,
        news_api_key=os.getenv('NEWS_API_KEY'),
        fred_api_key=os.getenv('FRED_API_KEY'),
        api_ninjas_key=os.getenv('API_NINJAS_KEY')
    )
    train_data, val_data = prepare_data_for_training(datamodule, device)
    if not train_data:
        print("❌ No training data available"); return

    # Assume news embedding is last feature in x, and its dim is known (e.g. 800)
    example_x = train_data[0][0]
    news_dim = 800  # Set this to your actual news embedding dimension
    input_size = example_x.shape[-1] - news_dim
    print(f"   Input size (non-news): {input_size}, News dim: {news_dim}")

    model = TFT(
        input_size=input_size,
        news_dim=news_dim,
        hidden_size=64,
        num_heads=4,
        dropout=0.1,
        seq_len=encoder_len,
        prediction_len=1,
        news_downsample_dim=32
    )
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Model params: {total_params}")

    # Helper to split x into (non-news, news)
    def split_x(x):
        return x[..., :-news_dim], x[..., -news_dim:]

    # Wrap train/val data to provide (x, news), y
    train_data_split = [ (split_x(x), y) for x, y in train_data ]
    val_data_split = [ (split_x(x), y) for x, y in val_data ]

    def train_model_tft(model, train_data, val_data, device, epochs=10, lr=0.001):
        print(f"🏋️ Training model on {device} for {epochs} epochs...")
        model = model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.MSELoss()
        train_losses, val_losses = [], []
        for epoch in range(epochs):
            model.train()
            train_loss = 0
            for (x, news), y in train_data:
                x, news, y = x.to(device), news.to(device), y.to(device)
                if len(y.shape) == 1:
                    y = y.unsqueeze(-1)
                optimizer.zero_grad()
                pred = model(x, news)
                if pred.shape != y.shape:
                    pred = pred.view(y.shape)
                loss = criterion(pred, y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            train_losses.append(train_loss / len(train_data))
            # Validation
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for (x, news), y in val_data:
                    x, news, y = x.to(device), news.to(device), y.to(device)
                    if len(y.shape) == 1:
                        y = y.unsqueeze(-1)
                    pred = model(x, news)
                    if pred.shape != y.shape:
                        pred = pred.view(y.shape)
                    loss = criterion(pred, y)
                    val_loss += loss.item()
            val_losses.append(val_loss / len(val_data))
            print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_losses[-1]:.5f}, Val Loss: {val_losses[-1]:.5f}")
        return train_losses, val_losses

    def generate_predictions_tft(model, val_data, device):
        model.eval()
        model = model.to(device)
        predictions, actuals = [], []
        with torch.no_grad():
            for (x, news), y in val_data:
                x, news = x.to(device), news.to(device)
                pred = model(x, news)
                predictions.extend(pred.cpu().numpy().flatten())
                actuals.extend(y.numpy().flatten())
        return predictions, actuals

    train_losses, val_losses = train_model_tft(model, train_data_split, val_data_split, device, epochs=15, lr=0.001)
    predictions, actuals = generate_predictions_tft(model, val_data_split, device)
    create_visualizations(predictions, actuals, train_losses, val_losses)
    if len(predictions) > 0 and len(actuals) > 0:
        simulate_trading(predictions, actuals)
    else:
        print("⚠️ Skipping trading simulation - no predictions available")
    print("\n✅ Pipeline complete. See tft_pure_torch_analysis.png and tft_trading_performance.png for results.")

print("hayson is gay")
main()