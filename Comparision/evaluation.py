#!/usr/bin/env python3
"""
Comprehensive evaluation module for multi-horizon stock prediction models.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any
from prettytable import PrettyTable
import torch


def evaluate_multi_horizon_predictions(model, data_module, horizons: List[int] = [1, 5, 10, 15, 20]) -> Dict[str, Any]:
    """
    Evaluate model predictions over multiple time horizons.
    
    Args:
        model: Trained model
        data_module: Data module with validation data
        horizons: List of prediction horizons to evaluate
        
    Returns:
        Dict containing MSE/MAE for each horizon and detailed predictions
    """
    model.eval()
    device = next(model.parameters()).device if hasattr(model, 'parameters') else None
    
    all_predictions = []
    all_targets = []
    all_metadata = []
    
    val_df = data_module.val_df if hasattr(data_module, 'val_df') else None
    
    with torch.no_grad():
        for batch_idx, (features, targets) in enumerate(data_module.val_loader):
            if device:
                features, targets = features.to(device), targets.to(device)
            
            # Get predictions
            if hasattr(model, 'news_dim') and model.news_dim > 0:
                # TFT model with news
                predictions = model(features, news=None)
            else:
                # Regular model
                predictions = model(features)
            
            # Convert to numpy
            pred_np = predictions.cpu().numpy() if device else predictions.numpy()
            target_np = targets.cpu().numpy() if device else targets.numpy()
            
            all_predictions.append(pred_np)
            all_targets.append(target_np)
            
            # Store metadata if available
            batch_size = features.shape[0]
            start_idx = batch_idx * batch_size
            
            if val_df is not None and start_idx < len(val_df):
                end_idx = min(start_idx + batch_size, len(val_df))
                batch_metadata = val_df.iloc[start_idx:end_idx][['date', 'symbol']].to_dict('records')
                all_metadata.extend(batch_metadata)
    
    # Concatenate all predictions and targets
    predictions_array = np.concatenate(all_predictions, axis=0)  # Shape: (samples, horizons)
    targets_array = np.concatenate(all_targets, axis=0)         # Shape: (samples, horizons)
    
    # Calculate metrics for each horizon
    results = {}
    detailed_predictions = []
    
    max_horizons = min(predictions_array.shape[1], len(horizons))
    
    for i, horizon in enumerate(horizons[:max_horizons]):
        pred_h = predictions_array[:, i]
        target_h = targets_array[:, i]
        
        # Calculate MSE and MAE
        mse = np.mean((pred_h - target_h) ** 2)
        mae = np.mean(np.abs(pred_h - target_h))
        
        # Calculate additional metrics
        rmse = np.sqrt(mse)
        mape = np.mean(np.abs((target_h - pred_h) / (target_h + 1e-8))) * 100
        
        results[f'horizon_{horizon}'] = {
            'MSE': mse,
            'MAE': mae,
            'RMSE': rmse,
            'MAPE': mape
        }
        
        # Store detailed predictions
        for j in range(len(pred_h)):
            metadata = all_metadata[j] if j < len(all_metadata) else {'date': f'sample_{j}', 'symbol': 'unknown'}
            detailed_predictions.append({
                'date': metadata['date'],
                'symbol': metadata['symbol'],
                'horizon': horizon,
                'prediction': float(pred_h[j]),
                'actual': float(target_h[j]),
                'absolute_error': abs(float(pred_h[j]) - float(target_h[j])),
                'squared_error': (float(pred_h[j]) - float(target_h[j])) ** 2
            })
    
    return {
        'horizon_metrics': results,
        'detailed_predictions': pd.DataFrame(detailed_predictions),
        'summary_stats': {
            'total_samples': len(predictions_array),
            'horizons_evaluated': max_horizons,
            'avg_mse': np.mean([results[f'horizon_{h}']['MSE'] for h in horizons[:max_horizons]]),
            'avg_mae': np.mean([results[f'horizon_{h}']['MAE'] for h in horizons[:max_horizons]])
        }
    }


def evaluate_sklearn_multi_horizon(model, X_val, y_val, val_df, horizons: List[int] = [1, 5, 10, 15, 20]) -> Dict[str, Any]:
    """
    Evaluate sklearn model predictions over multiple horizons.
    
    Args:
        model: Trained sklearn model
        X_val: Validation features
        y_val: Validation targets (multi-step)
        val_df: Validation DataFrame with metadata
        horizons: List of horizons to evaluate
        
    Returns:
        Dict containing evaluation results
    """
    # Get predictions - sklearn models typically predict single step, we'll replicate
    base_predictions = model.predict(X_val)
    
    # Create multi-step predictions by replicating single-step prediction
    max_horizon = max(horizons)
    if len(base_predictions.shape) == 1:
        # Single output - replicate for all horizons
        predictions_array = np.tile(base_predictions.reshape(-1, 1), (1, max_horizon))
    else:
        # Multi-output - use as is or truncate/pad
        predictions_array = base_predictions[:, :max_horizon]
        if predictions_array.shape[1] < max_horizon:
            # Pad with last prediction if needed
            last_col = predictions_array[:, -1:] 
            padding = np.tile(last_col, (1, max_horizon - predictions_array.shape[1]))
            predictions_array = np.concatenate([predictions_array, padding], axis=1)
    
    # Ensure y_val has the right shape
    if len(y_val.shape) == 1:
        # Single step targets - can't evaluate multi-horizon, use single horizon only
        horizons = [1]
        targets_array = y_val.reshape(-1, 1)
        predictions_array = predictions_array[:, :1]
    else:
        targets_array = y_val[:, :max_horizon]
    
    # Calculate metrics for each horizon
    results = {}
    detailed_predictions = []
    
    for i, horizon in enumerate(horizons):
        if i >= predictions_array.shape[1]:
            break
            
        pred_h = predictions_array[:, i]
        target_h = targets_array[:, i] if i < targets_array.shape[1] else targets_array[:, 0]
        
        # Calculate metrics
        mse = np.mean((pred_h - target_h) ** 2)
        mae = np.mean(np.abs(pred_h - target_h))
        rmse = np.sqrt(mse)
        mape = np.mean(np.abs((target_h - pred_h) / (target_h + 1e-8))) * 100
        
        results[f'horizon_{horizon}'] = {
            'MSE': mse,
            'MAE': mae,
            'RMSE': rmse,
            'MAPE': mape
        }
        
        # Store detailed predictions
        for j in range(len(pred_h)):
            if j < len(val_df):
                metadata = val_df.iloc[j]
                date = metadata.get('date', f'sample_{j}')
                symbol = metadata.get('symbol', 'unknown')
            else:
                date, symbol = f'sample_{j}', 'unknown'
                
            detailed_predictions.append({
                'date': date,
                'symbol': symbol,
                'horizon': horizon,
                'prediction': float(pred_h[j]),
                'actual': float(target_h[j]),
                'absolute_error': abs(float(pred_h[j]) - float(target_h[j])),
                'squared_error': (float(pred_h[j]) - float(target_h[j])) ** 2
            })
    
    return {
        'horizon_metrics': results,
        'detailed_predictions': pd.DataFrame(detailed_predictions),
        'summary_stats': {
            'total_samples': len(predictions_array),
            'horizons_evaluated': len([h for h in horizons if h <= predictions_array.shape[1]]),
            'avg_mse': np.mean([results[f'horizon_{h}']['MSE'] for h in horizons if f'horizon_{h}' in results]),
            'avg_mae': np.mean([results[f'horizon_{h}']['MAE'] for h in horizons if f'horizon_{h}' in results])
        }
    }


def create_horizon_comparison_table(all_model_results: Dict[str, Dict], horizons: List[int]) -> str:
    """
    Create a comprehensive table comparing all models across multiple horizons.
    
    Args:
        all_model_results: Dictionary of model results
        horizons: List of horizons evaluated
        
    Returns:
        Formatted table string
    """
    # Create output string manually to avoid PrettyTable type issues
    output = "="*100 + "\n"
    output += "📊 MULTI-HORIZON PREDICTION EVALUATION - MEAN SQUARED ERROR (MSE)\n"
    output += "="*100 + "\n"
    
    # Header
    mse_header = f"{'Model':<18}"
    for h in horizons:
        mse_header += f"{'H' + str(h) + '_MSE':>12}"
    mse_header += f"{'Avg_MSE':>12}\n"
    output += mse_header
    output += "-" * len(mse_header) + "\n"
    
    # MSE rows
    for model_name, results in all_model_results.items():
        if 'horizon_metrics' not in results:
            continue
            
        horizon_metrics = results['horizon_metrics']
        
        mse_row = f"{model_name:<18}"
        mse_values = []
        for h in horizons:
            if f'horizon_{h}' in horizon_metrics:
                mse_val = horizon_metrics[f'horizon_{h}']['MSE']
                mse_row += f"{mse_val:>12.6f}"
                mse_values.append(mse_val)
            else:
                mse_row += f"{'N/A':>12}"
        
        avg_mse = np.mean(mse_values) if mse_values else 0.0
        mse_row += f"{avg_mse:>12.6f}\n"
        output += mse_row
    
    output += "\n" + "="*100 + "\n"
    output += "📊 MULTI-HORIZON PREDICTION EVALUATION - MEAN ABSOLUTE ERROR (MAE)\n"
    output += "="*100 + "\n"
    
    # Header
    mae_header = f"{'Model':<18}"
    for h in horizons:
        mae_header += f"{'H' + str(h) + '_MAE':>12}"
    mae_header += f"{'Avg_MAE':>12}\n"
    output += mae_header
    output += "-" * len(mae_header) + "\n"
    
    # MAE rows
    for model_name, results in all_model_results.items():
        if 'horizon_metrics' not in results:
            continue
            
        horizon_metrics = results['horizon_metrics']
        
        mae_row = f"{model_name:<18}"
        mae_values = []
        for h in horizons:
            if f'horizon_{h}' in horizon_metrics:
                mae_val = horizon_metrics[f'horizon_{h}']['MAE']
                mae_row += f"{mae_val:>12.6f}"
                mae_values.append(mae_val)
            else:
                mae_row += f"{'N/A':>12}"
        
        avg_mae = np.mean(mae_values) if mae_values else 0.0
        mae_row += f"{avg_mae:>12.6f}\n"
        output += mae_row
    
    return output


def create_detailed_horizon_analysis(detailed_predictions_df: pd.DataFrame, horizons: List[int]) -> str:
    """
    Create detailed analysis of predictions by horizon.
    
    Args:
        detailed_predictions_df: DataFrame with all detailed predictions
        horizons: List of horizons
        
    Returns:
        Analysis string
    """
    output = "="*80 + "\n"
    output += "🔍 DETAILED HORIZON ANALYSIS\n"
    output += "="*80 + "\n"
    
    for horizon in horizons:
        h_data = detailed_predictions_df[detailed_predictions_df['horizon'] == horizon]
        if h_data.empty:
            continue
            
        output += f"\n--- Horizon {horizon} Days ---\n"
        output += f"Samples: {len(h_data)}\n"
        output += f"MSE: {h_data['squared_error'].mean():.6f}\n"
        output += f"MAE: {h_data['absolute_error'].mean():.6f}\n"
        output += f"RMSE: {np.sqrt(h_data['squared_error'].mean()):.6f}\n"
        
        # Best and worst predictions
        best_pred = h_data.loc[h_data['absolute_error'].idxmin()]
        worst_pred = h_data.loc[h_data['absolute_error'].idxmax()]
        
        output += f"Best Prediction: {best_pred['symbol']} on {best_pred['date']} "
        output += f"(Pred: {best_pred['prediction']:.4f}, Actual: {best_pred['actual']:.4f}, Error: {best_pred['absolute_error']:.4f})\n"
        
        output += f"Worst Prediction: {worst_pred['symbol']} on {worst_pred['date']} "
        output += f"(Pred: {worst_pred['prediction']:.4f}, Actual: {worst_pred['actual']:.4f}, Error: {worst_pred['absolute_error']:.4f})\n"
    
    return output
