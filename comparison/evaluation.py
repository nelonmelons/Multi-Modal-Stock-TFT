#!/usr/bin/env python3
"""
Comprehensive evaluation module for multi-horizon stock prediction models.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any
from prettytable import PrettyTable
import torch
from sklearn.metrics import r2_score


def _directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute directional accuracy (sign match rate)."""
    return float(np.mean(np.sign(y_true) == np.sign(y_pred))) if len(y_true) else np.nan


def evaluate_multi_horizon_predictions(model, data_module, horizons: List[int] = [1, 5, 21], split: str = 'test') -> Dict[str, Any]:
    """
    Evaluate model predictions over multiple time horizons on chosen split.
    Assumes model outputs full prediction_len steps aligned with target_0..target_{T-1}.

    Args:
        model: Trained model
        data_module: Data module with loaders and dfs
        horizons: List of horizons (in days) to evaluate
        split: 'val' or 'test'

    Returns:
        Dict containing metrics and detailed predictions
    """
    model.eval()
    device = next(model.parameters()).device if hasattr(model, 'parameters') else None

    # Select loader and reference df by split
    if split == 'test' and hasattr(data_module, 'test_loader') and data_module.test_loader is not None:
        loader = data_module.test_loader
        ref_df = getattr(data_module, 'test_df', None)
    else:
        loader = data_module.val_loader
        ref_df = getattr(data_module, 'val_df', None)

    all_predictions, all_targets, all_metadata = [], [], []

    with torch.no_grad():
        for batch_idx, (features, targets) in enumerate(loader):
            if device:
                features, targets = features.to(device), targets.to(device)
            # Forward
            predictions = model(features)
            pred_np = predictions.detach().cpu().numpy() if device else predictions.numpy()
            target_np = targets.detach().cpu().numpy() if device else targets.numpy()
            all_predictions.append(pred_np)
            all_targets.append(target_np)

            # Metadata
            batch_size = features.shape[0]
            start_idx = batch_idx * batch_size
            if ref_df is not None and start_idx < len(ref_df):
                end_idx = min(start_idx + batch_size, len(ref_df))
                batch_metadata = ref_df.iloc[start_idx:end_idx][['date', 'symbol']].to_dict('records')
                # If smaller due to last batch truncation, pad
                while len(batch_metadata) < batch_size:
                    batch_metadata.append({'date': None, 'symbol': None})
                all_metadata.extend(batch_metadata[:batch_size])

    if not all_predictions:
        return {'horizon_metrics': {}, 'detailed_predictions': pd.DataFrame(), 'summary_stats': {}}

    predictions_array = np.concatenate(all_predictions, axis=0)  # (N, T)
    targets_array = np.concatenate(all_targets, axis=0)          # (N, T)

    # Build detailed predictions using true column indices h-1
    detailed_rows = []
    metrics = {}

    for h in horizons:
        col_idx = max(h - 1, 0)
        if col_idx >= predictions_array.shape[1]:
            continue
        pred_h = predictions_array[:, col_idx]
        true_h = targets_array[:, col_idx]
        rmse = float(np.sqrt(np.mean((pred_h - true_h) ** 2)))
        mae = float(np.mean(np.abs(pred_h - true_h)))
        r2 = float(r2_score(true_h, pred_h)) if len(true_h) > 1 else np.nan
        da = _directional_accuracy(true_h, pred_h)
        metrics[f'horizon_{h}'] = {'RMSE': rmse, 'MAE': mae, 'R2': r2, 'DA': da}

        # Detailed
        for j in range(len(pred_h)):
            meta = all_metadata[j] if j < len(all_metadata) else {'date': None, 'symbol': None}
            detailed_rows.append({
                'date': meta['date'],
                'symbol': meta['symbol'],
                'horizon': h,
                'prediction': float(pred_h[j]),
                'actual': float(true_h[j]),
                'absolute_error': float(abs(pred_h[j] - true_h[j])),
                'squared_error': float((pred_h[j] - true_h[j]) ** 2),
            })

    detailed_df = pd.DataFrame(detailed_rows)

    return {
        'horizon_metrics': metrics,
        'detailed_predictions': detailed_df,
        'summary_stats': {
            'total_samples': predictions_array.shape[0],
            'horizons_evaluated': len(metrics),
            'avg_rmse': np.nanmean([m['RMSE'] for m in metrics.values()]) if metrics else np.nan,
            'avg_mae': np.nanmean([m['MAE'] for m in metrics.values()]) if metrics else np.nan,
        }
    }



def create_horizon_comparison_table(all_model_results: Dict[str, Dict], horizons: List[int]) -> str:
    """
    Create a comprehensive table comparing all models across multiple horizons.
    Shows RMSE for compactness.
    """
    output = "="*100 + "\n"
    output += "[CHARTS] MULTI-HORIZON PREDICTION EVALUATION - RMSE\n"
    output += "="*100 + "\n"

    header = f"{'Model':<18}" + "".join([f"{'H'+str(h)+'_RMSE':>12}" for h in horizons]) + f"{'Avg_RMSE':>12}\n"
    output += header + "-" * len(header) + "\n"

    for model_name, results in all_model_results.items():
        if 'horizon_metrics' not in results:
            continue
        row = f"{model_name:<18}"
        rmses = []
        for h in horizons:
            key = f'horizon_{h}'
            if key in results['horizon_metrics']:
                rmse_val = results['horizon_metrics'][key]['RMSE']
                row += f"{rmse_val:>12.6f}"
                rmses.append(rmse_val)
            else:
                row += f"{'N/A':>12}"
        avg_rmse = np.mean(rmses) if rmses else 0.0
        row += f"{avg_rmse:>12.6f}\n"
        output += row
    return output


def create_detailed_horizon_analysis(detailed_predictions_df: pd.DataFrame, horizons: List[int]) -> str:
    """Unchanged aside from docstring."""
    output = "="*80 + "\n"
    output += "[SEARCH] DETAILED HORIZON ANALYSIS\n"
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
        best_pred = h_data.loc[h_data['absolute_error'].idxmin()]
        worst_pred = h_data.loc[h_data['absolute_error'].idxmax()]
        output += f"Best Prediction: {best_pred['symbol']} on {best_pred['date']} (Pred: {best_pred['prediction']:.4f}, Actual: {best_pred['actual']:.4f}, Error: {best_pred['absolute_error']:.4f})\n"
        output += f"Worst Prediction: {worst_pred['symbol']} on {worst_pred['date']} (Pred: {worst_pred['prediction']:.4f}, Actual: {worst_pred['actual']:.4f}, Error: {worst_pred['absolute_error']:.4f})\n"
    return output
