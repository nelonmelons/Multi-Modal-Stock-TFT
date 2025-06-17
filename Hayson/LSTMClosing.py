import datetime
import numpy as np
import pandas as pd
import torch
from torch.utils import data as data_utils
from torch.utils.data import DataLoader
import os
from typing import List, Dict, Tuple, override, Optional, Callable, Union

import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset
import torch.nn as nn

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from dataModule import Data_Day_Hourly_StocksPrice # Added import


class StockPredictor(nn.Module):
    def __init__(self, input_dim=42, hidden_dim=64, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x, lengths):
        # x: [batch_size, seq_len, input_dim], lengths: [batch_size]
        packed = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        packed_out, (h_n, c_n) = self.lstm(packed)
        out = self.fc(h_n[-1])  # [batch_size, 1]
        return out.squeeze(-1)  #


def collate_fn(batch):
    # batch: list of (x, y), where x: [l, 42], y: scalar
    batch.sort(key=lambda x: x[0].shape[0], reverse=True)
    lengths = [x[0].shape[0] for x in batch]
    xs = [x[0] for x in batch]
    ys = torch.tensor([x[1] for x in batch], dtype=torch.float32)
    xs_padded = torch.nn.utils.rnn.pad_sequence(xs, batch_first=True)  # [batch, max_l, 42]
    return xs_padded, lengths, ys

def generate_predictions(model, data_loader, data_obj, stock_symbol, source_df_for_timestamps, seq_len_param): # Added params
    """
    Generate predictions for a given stock symbol using the trained model.
    
    Parameters:
    model: The trained model
    data_loader: DataLoader for the stock's dataset (e.g., test or validation set)
    data_obj: Data_Day_Hourly_StocksPrice instance (used for its methods if needed)
    stock_symbol: Stock symbol to predict (for context, if needed)
    source_df_for_timestamps: The DataFrame that was used to create the data_loader's dataset.
                              This df must have the 'timestamp' column.
                              It should be the raw hourly data for the prediction period,
                              having had add_return_columns applied if StockDataset expected it.
    seq_len_param: The sequence length used in the StockDataset.
    
    Returns:
    dict: Dictionary mapping timestamps (integer Unix seconds) to predicted values
    """
    model.eval()
    predictions = {}
    
    df = source_df_for_timestamps
    
    if df is None or df.empty:
        print(f"Warning ({stock_symbol}): source_df_for_timestamps is empty or None. Cannot generate predictions.")
        return predictions

    if 'timestamp' not in df.columns:
        print(f"Warning ({stock_symbol}): 'timestamp' column not in source_df_for_timestamps. Cannot map predictions.")
        return predictions
        
    try:
        timestamps_series = pd.to_datetime(df['timestamp'])
    except Exception as e:
        print(f"Warning ({stock_symbol}): Could not convert 'timestamp' column to datetime: {e}. Predictions may be misaligned.")
        return predictions

    hours_per_day = 7 # Should be consistent with StockDataset
    n_total_hourly_rows_in_df = len(df)
    n_total_days_in_df = n_total_hourly_rows_in_df // hours_per_day
    
    seq_len = seq_len_param

    current_prediction_index_in_dataset = 0 
    
    with torch.no_grad():
        for batch_x, lengths, batch_y in data_loader:
            outputs = model(batch_x, lengths)
            
            for i in range(len(outputs)):
                day_index_predicted = current_prediction_index_in_dataset + seq_len
                
                if day_index_predicted < n_total_days_in_df:
                    target_hour_index_in_original_df = (day_index_predicted + 1) * hours_per_day - 1
                    
                    if target_hour_index_in_original_df < n_total_hourly_rows_in_df:
                        timestamp_obj = timestamps_series.iloc[target_hour_index_in_original_df]
                        ts_key = int(pd.Timestamp(timestamp_obj).timestamp())
                        predictions[ts_key] = float(outputs[i].item())
                    else:
                        print(f"Warning ({stock_symbol}): Calculated target_hour_index {target_hour_index_in_original_df} out of bounds for df with {n_total_hourly_rows_in_df} rows. Day predicted: {day_index_predicted}.")
                else:
                    print(f"Warning ({stock_symbol}): Calculated day_index_predicted {day_index_predicted} (from item {current_prediction_index_in_dataset} + seq_len {seq_len}) is out of bounds for df with {n_total_days_in_df} days.")

                current_prediction_index_in_dataset += 1
    
    return predictions


def train_model_with_visualization(data, stock_symbols, epochs=5, lr=0.001, seq_len=24, batch_size=32):
    """
    Optimized training pipeline with visualization and monitoring.
    """
    print("=== Starting Optimized Training Pipeline ===")
    print(f"Using symbols: {stock_symbols}")
    
    # Get data loaders
    train_loader, test_loader, val_loader = data.get_loaders(
        stock_symbols=stock_symbols,
        seq_len=seq_len,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )
    
    if train_loader is None:
        print("ERROR: No training data available!")
        return None
    
    # Initialize model and optimizer
    model = StockPredictor(input_dim=42, hidden_dim=128, num_layers=2) # Assuming 42 features from 7 hours * 6 features/hour
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
    criterion = nn.MSELoss()
    
    train_losses = []
    val_losses = []
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    if train_loader: print(f"Training batches: {len(train_loader)}")
    if val_loader: print(f"Validation batches: {len(val_loader)}")
    
    for epoch in range(epochs):
        model.train()
        epoch_train_loss = 0.0
        num_batches = 0
        
        print(f"\\n--- Epoch {epoch + 1}/{epochs} ---")
        
        for batch_idx, (batch_x, lengths, batch_y) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(batch_x, lengths)
            loss = criterion(outputs, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_train_loss += loss.item()
            num_batches += 1
            
            if batch_idx % 20 == 0: # Log progress
                print(f"  Epoch {epoch+1}, Batch {batch_idx:3d}/{len(train_loader)}, Loss: {loss.item():.6f}, LR: {optimizer.param_groups[0]['lr']:.2e}")
        
        avg_train_loss = epoch_train_loss / num_batches if num_batches > 0 else 0
        train_losses.append(avg_train_loss)
        
        val_loss = 0.0
        if val_loader is not None:
            model.eval()
            with torch.no_grad():
                val_batches = 0
                for batch_x_val, lengths_val, batch_y_val in val_loader:
                    outputs_val = model(batch_x_val, lengths_val)
                    loss_val = criterion(outputs_val, batch_y_val)
                    val_loss += loss_val.item()
                    val_batches += 1
                val_loss /= val_batches if val_batches > 0 else 1
            val_losses.append(val_loss)
            scheduler.step(val_loss)
            print(f"  Epoch {epoch + 1} - Train Loss: {avg_train_loss:.6f}, Val Loss: {val_loss:.6f}")
        else:
            scheduler.step(avg_train_loss) # Step scheduler even if no val_loader, using train loss
            print(f"  Epoch {epoch + 1} - Train Loss: {avg_train_loss:.6f}")
    
    # Plot training curves
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss', color='blue')
    if val_losses:
        plt.plot(val_losses, label='Validation Loss', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    
    # Learning rate plot might need to store LR history if it changes per batch
    lr_history = [optimizer.param_groups[0]['lr']] * epochs # Simplified: assumes LR changes per epoch via scheduler
    plt.subplot(1, 2, 2)
    plt.plot(lr_history, label='Learning Rate')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate Schedule (End of Epoch)')
    plt.yscale('log')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

    if model is not None:
        print("\\n=== Generating Predictions and Visualizations ===")
        hours_per_day = 7 # Consistent with StockDataset

        for symbol in stock_symbols:
            print(f"\\n--- Processing symbol: {symbol} for prediction and visualization ---")
            
            data_s, type_of_data = data.get_from_symbol(symbol)
            
            df_for_prediction_source = None # This will be the df used to create StockDataset
            original_df_for_visualization = None # Raw df for the period

            if type_of_data == 'test/train' and data_s and isinstance(data_s, tuple):
                _, test_df_orig = data_s
                if test_df_orig is not None and not test_df_orig.empty:
                    original_df_for_visualization = test_df_orig.copy()
                    df_for_prediction_source = Data_Day_Hourly_StocksPrice.add_return_columns(test_df_orig.copy())
                else:
                    print(f"Test data for {symbol} is empty or None.")
            elif type_of_data == 'validation' and isinstance(data_s, pd.DataFrame):
                val_df_orig = data_s
                if val_df_orig is not None and not val_df_orig.empty:
                    original_df_for_visualization = val_df_orig.copy()
                    df_for_prediction_source = Data_Day_Hourly_StocksPrice.add_return_columns(val_df_orig.copy())
                else:
                    print(f"Validation data for {symbol} is empty or None.")
            else:
                print(f"No suitable data found for {symbol} (type: {type_of_data}).")

            if df_for_prediction_source is None or df_for_prediction_source.empty:
                print(f"Skipping {symbol} due to lack of data for prediction.")
                continue

            num_days_in_pred_df = len(df_for_prediction_source) // hours_per_day
            if num_days_in_pred_df <= seq_len:
                print(f"Not enough daily data in prediction period for {symbol} (days: {num_days_in_pred_df}, seq_len: {seq_len}). Skipping.")
                continue
            
            try:
                prediction_dataset = data.StockDataset(df_for_prediction_source, seq_len)
            except Exception as e:
                print(f"Error creating StockDataset for {symbol}: {e}. Skipping.")
                continue

            if len(prediction_dataset) == 0:
                 print(f"Prediction dataset for {symbol} is empty (source days: {num_days_in_pred_df}, seq_len: {seq_len}). Skipping.")
                 continue
            
            prediction_loader = DataLoader(prediction_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

            print(f"Generating predictions for {symbol} using {len(prediction_dataset)} sequences...")
            predictions_dict = generate_predictions(
                model, 
                prediction_loader, 
                data_obj=data, 
                stock_symbol=symbol, 
                source_df_for_timestamps=df_for_prediction_source,
                seq_len_param=seq_len
            )
            
            if not predictions_dict:
                print(f"No predictions generated for {symbol}.")
            else:
                print(f"Generated {len(predictions_dict)} predictions for {symbol}.")

            if original_df_for_visualization is not None and not original_df_for_visualization.empty:
                data.visualize(
                    stock_symbol=symbol,
                    pred=predictions_dict,
                    title_pre=f"Predictions for "
                    # visualize will use its internal logic to fetch data for 'actual' plot
                    # which should align if 'symbol' correctly points to test or validation data
                )
            else:
                print(f"No original data for visualization for {symbol}.")
    
    return model


def get_stock_symbols(data_obj, consistent_symbols_list):
    """
    Determines and returns a list of stock symbols to use based on availability
    in training/testing and validation datasets.
    """
    available_train_test = list(data_obj.symbols_test_train.keys())
    available_validation = list(data_obj.symbols_validation.keys())

    print(f"Available train/test symbols: {available_train_test[:10]}...")
    print(f"Available validation symbols: {available_validation[:5]}...")

    stock_symbols = []
    if available_train_test:
        for symbol in consistent_symbols_list:
            if symbol in available_train_test:
                stock_symbols.append(symbol)
        
        if not stock_symbols: # If none of our consistent symbols exist, use first 2 available
            stock_symbols = available_train_test[:2]

    if available_validation and len(stock_symbols) < 3:
        for symbol in consistent_symbols_list:
            if symbol in available_validation and symbol not in stock_symbols:
                stock_symbols.append(symbol)
                if len(stock_symbols) >= 3: # Ensure we don't add more than needed if consistent_symbols are plentiful
                    break

        # If still need more (i.e. less than 3 and consistent symbols didn't cover it)
        if len(stock_symbols) < 3 and available_validation:
            for symbol in available_validation:
                if symbol not in stock_symbols:
                    stock_symbols.append(symbol)
                    if len(stock_symbols) >= 3:
                        break

    if not stock_symbols:
        print("ERROR: No stock symbols available in the data!")
        # Consider raising an exception here instead of exiting directly
        # For now, following original logic:
        exit(1)

    print(f"Using symbols: {stock_symbols}")
    return stock_symbols


def run_main_logic():
    """
    Main logic for data loading, processing, and model training.
    """
    # Ensure the 'data' directory is accessible from where this script is run.
    # Typically, if LSTMbaseLIne.py is in Hayson/ and data is in Hayson/data/,
    # running from Hayson/ directory means 'data' is the correct relative path.
    data = Data_Day_Hourly_StocksPrice.from_dir('data')
    print("data:", data)

    # Use symbols that actually exist in the data - pick same stocks for consistency
    consistent_symbols = ['AAPL', 'AMD', 'AIG', 'AMZN', 'GOOGL', 'MSFT', 'NFLX', 'NVDA', 'TSLA', 'META', 'UBER', 'WMT', 'DIS', 'BA', 'INTC', 'IBM', 'ORCL', 'CSCO', 'VZ', 'T', 'JPM', 'V', 'JNJ', 'PG', 'HD', 'MA', 'UNH', 'BAC', 'PYPL', 'ADBE', 'CRM', 'KO', 'PEP', 'XOM', 'CVX', 'MCD', 'UBER', 'UL', 'UBS', 'TSLA']

    # get_stock_symbols is defined in this file
    stock_symbols_to_use = get_stock_symbols(data, consistent_symbols)

    if not stock_symbols_to_use:
        print("ERROR: No stock symbols determined for training. Exiting.")
        return

    print(f"Proceeding with symbols: {stock_symbols_to_use}")

    # train_model_with_visualization is defined in this file
    # Pass necessary parameters; using defaults for epochs, lr, etc.
    trained_model = train_model_with_visualization(
        data=data,
        stock_symbols=stock_symbols_to_use
    )

    if trained_model:
        print("Model training complete.")
    else:
        print("Model training failed or was skipped.")


if __name__ == "__main__":
    run_main_logic()
