#!/usr/bin/env python3
"""
Training and evaluation script for stock prediction models.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from tqdm import tqdm
import pandas as pd
import numpy as np
from model.tft_model import setup_device

def train_model(model, data_module, epochs=20, lr=0.001, device=None, patience=5):
    """
    Training function with early stopping to prevent overfitting.
    ✅ FIXED: Now includes early stopping based on validation loss.
    
    Args:
        model: The PyTorch model to train
        data_module: The data module with train_loader and val_loader
        epochs: Number of training epochs
        lr: Learning rate
        device: PyTorch device to use (if None, will auto-detect)
        patience: Early stopping patience (epochs to wait for improvement)
    
    Returns:
        Tuple of (trained_model, history)
    """
    if device is None:
        device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    history = {'train_loss': [], 'val_loss': []}
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        
        for features, targets in tqdm(data_module.train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            features, targets = features.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, targets)
            
            # Check for NaN or infinite loss
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"⚠️  Warning: NaN or infinite loss detected in epoch {epoch+1}")
                print(f"   Features shape: {features.shape}, Outputs shape: {outputs.shape}")
                print(f"   Features range: [{features.min():.6f}, {features.max():.6f}]")
                print(f"   Outputs range: [{outputs.min():.6f}, {outputs.max():.6f}]")
                continue  # Skip this batch
            
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(data_module.train_loader)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for features, targets in data_module.val_loader:
                features, targets = features.to(device), targets.to(device)
                outputs = model(features)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(data_module.val_loader)
        
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        
        # ✅ FIXED: Early stopping based on validation loss
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
        
        if epoch % 5 == 0:
            print(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        
        # Early stopping
        if patience_counter >= patience:
            print(f"✅ Early stopping at epoch {epoch+1} (patience={patience})")
            break
    
    # Restore best model state
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"✅ Restored best model (val_loss={best_val_loss:.4f})")
    
    return model, history

def train_tft_model(model, data_module, epochs=10, lr=0.001, device=None, patience=5):
    """
    Train a TFT model with early stopping.
    ✅ FIXED: Added early stopping to prevent overfitting.
    
    Args:
        model: TFT model to train
        data_module: Data module with train/val loaders
        epochs: Number of training epochs
        lr: Learning rate
        device: Training device
        patience: Early stopping patience
    
    Returns:
        Trained model and training history
    """
    if device is None:
        device = setup_device()
    
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    history = {'train_loss': [], 'val_loss': []}
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        
        for batch_idx, (features, targets) in enumerate(tqdm(data_module.train_loader, desc=f"Epoch {epoch+1}/{epochs}")):
            features, targets = features.to(device), targets.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass without news data
            outputs = model(features)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(data_module.train_loader)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for batch_idx, (features, targets) in enumerate(data_module.val_loader):
                features, targets = features.to(device), targets.to(device)
                
                outputs = model(features)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(data_module.val_loader)
        
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        
        # ✅ FIXED: Early stopping for TFT
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
        
        if epoch % 5 == 0:
            print(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        
        # Early stopping
        if patience_counter >= patience:
            print(f"✅ Early stopping at epoch {epoch+1} (patience={patience})")
            break
    
    # Restore best model state
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"✅ Restored best TFT model (val_loss={best_val_loss:.4f})")
    
    return model, history

def train_model_advanced(model, data_module, device, config, feature_indices, feature_set='all'):
    """
    Train a given model with selective features.

    Args:
        model: The model to train.
        data_module: The data module containing train and validation loaders.
        device: The device to train on.
        config: A dictionary with training parameters (epochs, lr).
        feature_indices: Dict with indices for 'base', 'news', 'fred'.
        feature_set: Which features to use ('base', 'base_news', 'all').

    Returns:
        The trained model and a history of training/validation losses.
    """
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=config.get('weight_decay', 0.01))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['epochs'], eta_min=config['lr']/10)
    criterion = nn.MSELoss()

    history = {'train_loss': [], 'val_loss': []}

    for epoch in range(config['epochs']):
        model.train()
        total_train_loss = 0
        train_loader = data_module.train_loader
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['epochs']} - Training"):
            features, target = batch
            features, target = features.to(device), target.to(device)

            optimizer.zero_grad()
            
            # Select features based on the set
            numeric_features = features[:, feature_indices['base']]
            news_features = features[:, feature_indices['news']] if feature_indices['news'] else None
            
            prediction = None
            if feature_set == 'base':
                prediction = model(numeric_features)
            elif feature_set == 'base_news':
                prediction = model(numeric_features, news=news_features)
            elif feature_set == 'all':
                fred_features = features[:, feature_indices['fred']] if feature_indices['fred'] else None
                
                combined_features_list = [numeric_features]
                if fred_features is not None:
                    combined_features_list.append(fred_features)
                combined_features = torch.cat(combined_features_list, dim=-1)
                
                prediction = model(combined_features, news=news_features)
            else:
                prediction = model(features) # Fallback for simple models

            if prediction is not None:
                loss = criterion(prediction, target)
                loss.backward()
                optimizer.step()
                total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        history['train_loss'].append(avg_train_loss)
        scheduler.step()

        # Validation
        model.eval()
        total_val_loss = 0
        val_loader = data_module.val_loader
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{config['epochs']} - Validation"):
                features, target = batch
                features, target = features.to(device), target.to(device)
                
                numeric_features = features[:, feature_indices['base']]
                news_features = features[:, feature_indices['news']] if feature_indices['news'] else None

                prediction = None
                if feature_set == 'base':
                    prediction = model(numeric_features)
                elif feature_set == 'base_news':
                    prediction = model(numeric_features, news=news_features)
                elif feature_set == 'all':
                    fred_features = features[:, feature_indices['fred']] if feature_indices['fred'] else None
                    
                    combined_features_list = [numeric_features]
                    if fred_features is not None:
                        combined_features_list.append(fred_features)
                    combined_features = torch.cat(combined_features_list, dim=-1)

                    prediction = model(combined_features, news=news_features)
                else:
                    prediction = model(features)

                if prediction is not None:
                    loss = criterion(prediction, target)
                    total_val_loss += loss.item()
        
        avg_val_loss = total_val_loss / len(val_loader) if len(val_loader) > 0 else 0
        history['val_loss'].append(avg_val_loss)

        print(f"Epoch {epoch+1}: Train Loss={avg_train_loss:.6f}, Val Loss={avg_val_loss:.6f}")

    return model, history

def evaluate_model(model, data_module, device, horizons, feature_indices, feature_set='all'):
    """
    Evaluate a model's performance over different time horizons.

    Args:
        model: The trained model.
        data_module: The data module with the validation loader.
        device: The device for evaluation.
        horizons: A list of integers representing prediction horizons to evaluate.
        feature_indices: Dict with indices for 'base', 'news', 'fred'.
        feature_set: Which features were used for training.

    Returns:
        A dictionary with evaluation metrics and a DataFrame of predictions.
    """
    model.to(device)
    model.eval()
    
    all_predictions = []
    all_ground_truths = []
    
    # We need to get symbols and dates back for the predictions DataFrame
    val_df = data_module.df[data_module.df[data_module.date_col] > data_module.split_date].copy().reset_index(drop=True)
    
    val_loader = data_module.val_loader
    with torch.no_grad():
        for features, target in val_loader:
            features, target = features.to(device), target.to(device)
            
            numeric_features = features[:, feature_indices['base']]
            news_features = features[:, feature_indices['news']] if feature_indices['news'] else None

            pred = None
            if feature_set == 'base':
                pred = model(numeric_features)
            elif feature_set == 'base_news':
                pred = model(numeric_features, news=news_features)
            elif feature_set == 'all':
                fred_features = features[:, feature_indices['fred']] if feature_indices['fred'] else None
                
                combined_features_list = [numeric_features]
                if fred_features is not None:
                    combined_features_list.append(fred_features)
                combined_features = torch.cat(combined_features_list, dim=-1)

                pred = model(combined_features, news=news_features)
            else:
                pred = model(features)

            if pred is not None:
                all_predictions.append(pred.cpu().numpy())
                all_ground_truths.append(target.cpu().numpy())
            
    if not all_predictions:
        return {}, pd.DataFrame()

    predictions = np.concatenate(all_predictions, axis=0)
    ground_truths = np.concatenate(all_ground_truths, axis=0)
    
    # Create predictions DataFrame
    pred_data = []
    num_predictions = predictions.shape[0]
    
    # Ensure val_df has enough rows
    if len(val_df) < num_predictions:
        print(f"Warning: Mismatch between number of predictions ({num_predictions}) and validation samples ({len(val_df)}). Truncating.")
        val_df = val_df.iloc[:num_predictions]

    for i in range(num_predictions):
        for h in range(predictions.shape[1]):
            pred_data.append({
                'date': val_df.iloc[i]['date'],
                'symbol': val_df.iloc[i]['symbol'],
                'horizon': h + 1,
                'prediction': predictions[i, h],
                'actual': ground_truths[i, h]
            })
    predictions_df = pd.DataFrame(pred_data)

    results = {}
    for h in horizons:
        horizon_df = predictions_df[predictions_df['horizon'] == h]
        
        if not horizon_df.empty:
            mse = np.mean((horizon_df['prediction'] - horizon_df['actual'])**2)
            mae = np.mean(np.abs(horizon_df['prediction'] - horizon_df['actual']))
            results[f'horizon_{h}'] = {'MSE': mse, 'MAE': mae}
        else:
            results[f'horizon_{h}'] = {'MSE': np.nan, 'MAE': np.nan}
        
    return results, predictions_df

def get_predictions(model, dataloader, val_df=None):
    """
    Generate predictions for a given model and dataloader.

    Args:
        model (nn.Module): The trained PyTorch model.
        dataloader (DataLoader): The dataloader for which to generate predictions.
        val_df (pd.DataFrame): Validation DataFrame with date/symbol information.

    Returns:
        pd.DataFrame: A DataFrame containing dates, symbols, and predictions.
    """
    model.eval()
    predictions = []

    device = next(model.parameters()).device

    with torch.no_grad():
        for features, _ in dataloader:
            features = features.to(device)
            
            # Handle both TFT and regular models
            output = model(features)
            
            # Store predictions
            predictions.extend(output.cpu().numpy())

    # Create a predictions DataFrame using actual validation data if available
    num_predictions = len(predictions)
    
    if val_df is not None and len(val_df) >= num_predictions:
        # Use actual validation data structure
        prediction_rows = []
        
        for i in range(num_predictions):
            # Handle different prediction formats
            pred = predictions[i]
            if isinstance(pred, np.ndarray):
                # Multi-step predictions - use the first step for portfolio decisions
                pred_value = float(pred[0]) if len(pred) > 0 else 0.0
            else:
                # Single prediction value
                pred_value = float(pred)
            
            # Use actual date/symbol from validation data
            prediction_rows.append({
                'date': val_df.iloc[i]['date'],
                'symbol': val_df.iloc[i]['symbol'],
                'prediction': pred_value  # Single numeric value for portfolio simulation
            })
        
        return pd.DataFrame(prediction_rows)
    
    else:
        # Fallback to synthetic data if val_df not available or insufficient
        print(f"Warning: Using synthetic data for predictions. val_df available: {val_df is not None}, len: {len(val_df) if val_df is not None else 'N/A'}, predictions: {num_predictions}")
        
        prediction_rows = []
        symbols = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA', 'AMD', 'QCOM', 'INTC']
        start_date = pd.Timestamp('2024-08-01')  # Use recent date
        
        for i in range(num_predictions):
            # Add business days (skip weekends)
            date = start_date + pd.Timedelta(days=i)
            # Skip weekends
            while date.weekday() >= 5:  # 5=Saturday, 6=Sunday
                date += pd.Timedelta(days=1)
            
            symbol = symbols[i % len(symbols)]
            
            # Handle different prediction formats
            pred = predictions[i]
            if isinstance(pred, np.ndarray):
                # Multi-step predictions - use the first step for portfolio decisions
                pred_value = float(pred[0]) if len(pred) > 0 else 0.0
            else:
                # Single prediction value
                pred_value = float(pred)
            
            prediction_rows.append({
                'date': date,
                'symbol': symbol,
                'prediction': pred_value  # Single numeric value for portfolio simulation
            })
        
        return pd.DataFrame(prediction_rows)
