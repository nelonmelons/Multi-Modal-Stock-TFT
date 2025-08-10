#!/usr/bin/env python3
"""
Debug script to load and print training and validation data batches for TFT pipeline.
"""
import os
import sys
from datetime import datetime, timedelta

# Ensure local dataModule package is discoverable
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from dataModule.datamodule import NumericDataModule
from dataModule.fetch_stock import fetch_stock_data
from dataModule.fetch_events import fetch_events_data
from dataModule.fetch_news import fetch_news_embeddings
from dataModule.fetch_fred import fetch_fred_data
from dataModule.compute_ta import compute_technical_indicators
from dataModule.build_features import build_features

config = {
        'symbols': ['AAPL', 'MSFT', 'GOOGL'],
        'start_date': '2022-01-01',
        'end_date': '2022-03-01',
        'encoder_len': 60,
        'predict_len': 10,
        'batch_size': 16,
        'news_api_key': os.getenv('NEWS_API_KEY'),
        'fred_api_key': os.getenv('FRED_API_KEY'),
        'api_ninjas_key': os.getenv('API_NINJAS_KEY'),
        'lookahead_buffer': 5,
        'validation_split': 0.8,
    }


def main():
    # Configuration for debugging data loading
    
    print("Configuration for data loading:", config)

    # Calculate temporal splits
    start_dt = datetime.strptime(config['start_date'], '%Y-%m-%d')
    end_dt = datetime.strptime(config['end_date'], '%Y-%m-%d')
    total_days = (end_dt - start_dt).days
    train_days = int(total_days * config['validation_split'])
    val_split_dt = start_dt + timedelta(days=train_days)
    train_end = (val_split_dt - timedelta(days=config['lookahead_buffer'])).strftime('%Y-%m-%d')
    val_start = val_split_dt.strftime('%Y-%m-%d')
    val_end = config['end_date']

    print(f"Training period: {config['start_date']} -> {train_end}")
    print(f"Validation period: {val_start} -> {val_end}\n")

    # Fetch and build features for ENTIRE period (train + validation)
    print("Fetching data for entire period...")
    stock_df = fetch_stock_data(config['symbols'], config['start_date'], config['end_date'])
    events_df = fetch_events_data(config['symbols'], config['start_date'], config['end_date'], None, config['api_ninjas_key'])
    news_df = fetch_news_embeddings(config['symbols'], config['start_date'], config['end_date'], config['news_api_key'])
    fred_df = fetch_fred_data(config['start_date'], config['end_date'], config['fred_api_key'])
    ta_df = compute_technical_indicators(stock_df)
    feature_df = build_features(stock_df, events_df, news_df, ta_df, fred_df,
                                config['encoder_len'], config['predict_len'])
    
    print(f"Total feature matrix shape: {feature_df.shape}")
    print(f"Date range in features: {feature_df['date'].min()} to {feature_df['date'].max()}")
    
    # Initialize NumericDataModule with proper temporal split
    data_module = NumericDataModule(feature_df=feature_df,
                                   split_date=train_end,
                                   batch_size=config['batch_size'],
                                   date_col='date',
                                   target_col='target')
    data_module.setup()
    
    train_loader = data_module.train_loader
    val_loader = data_module.val_loader
    
    # Print summary statistics
    print(f"\n=== DATA SUMMARY ===")
    print(f"Training batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    
    # Print a few training batches
    print(f"\n=== TRAINING DATA BATCHES ===")
    for idx, (features, targets) in enumerate(train_loader):
        print(f"Train Batch {idx}: Features shape={features.shape}, Targets shape={targets.shape}")
        print(f"  Features sample: {features[0][:5]}...")  # First 5 features of first sample
        print(f"  Targets sample: {targets[0]}")
        if idx >= 2:  # Only show first 3 batches
            break
    
    # Print a few validation batches
    print(f"\n=== VALIDATION DATA BATCHES ===")
    for idx, (features, targets) in enumerate(val_loader):
        print(f"Val Batch {idx}: Features shape={features.shape}, Targets shape={targets.shape}")
        print(f"  Features sample: {features[0][:5]}...")  # First 5 features of first sample
        print(f"  Targets sample: {targets[0]}")
        if idx >= 2:  # Only show first 3 batches
            break


if __name__ == '__main__':
    main()
