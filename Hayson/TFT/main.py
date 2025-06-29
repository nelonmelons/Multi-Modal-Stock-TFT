#!/usr/bin/env python3
"""
Unified TFT Pipeline: Run this script to execute the full, robust, and interpretable TFT pipeline (data loading, training, prediction, interpretability, trading simulation, and plotting).
"""

import os
import sys
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from dataModule.interface import get_data_loader_with_module
from tft_pure_torch_m1 import SimpleTFT, setup_device, prepare_data_for_training, train_model, generate_predictions, create_visualizations, simulate_trading


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
    input_size = train_data[0][0].shape[-1]
    print(f"   Input size: {input_size}")
    model = SimpleTFT(input_size=input_size, hidden_size=64, num_heads=4, dropout=0.1, seq_len=encoder_len, prediction_len=1)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Model params: {total_params}")
    train_losses, val_losses = train_model(model, train_data, val_data, device, epochs=15, lr=0.001)
    predictions, actuals = generate_predictions(model, val_data, device)
    create_visualizations(predictions, actuals, train_losses, val_losses)
    if len(predictions) > 0 and len(actuals) > 0:
        simulate_trading(predictions, actuals)
    else:
        print("⚠️ Skipping trading simulation - no predictions available")
    print("\n✅ Pipeline complete. See tft_pure_torch_analysis.png and tft_trading_performance.png for results.")

if __name__ == "__main__":
    main()
