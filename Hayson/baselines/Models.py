# ===========================
# Imports & Dependencies
# ===========================

# Core Python & OS
import os
import inspect
from datetime import datetime

# Scientific Computing
import numpy as np
import pandas as pd

# PyTorch & Data Utilities
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch import nn

# Visualization
import matplotlib.pyplot as plt

# Metrics
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Project Modules
from basics import *
from EncOnly import CausalTransformer
from EncDec import EncDecTransformer
from SoftAlignment import SoftAlignModelLoop
from dataModule import Data_Day_Hourly_StocksPrice

# Typing
from typing import Optional, Union, cast, Dict

# ===========================

class FullModel(nn.Module):
    def __init__(self, model_type: str, input_dim=35, embed_dim=128):
        super().__init__()
        self.embedding = EmbedderGRNWrapper(input_dim=input_dim, embed_dim=embed_dim)

        if model_type == 'causal':
            self.model = CausalTransformer(input_dim=input_dim, embed_dim=embed_dim)
        elif model_type == 'encdec':
            self.model = EncDecTransformer(embed_dim=embed_dim, n_heads=4)
        elif model_type == 'softalign':
            # use loop version for memory-efficient attention
            self.model = SoftAlignModelLoop(embed_dim=embed_dim)
        else:
            raise ValueError("Invalid model_type")
        

        print(f"Using model type: {model_type}")

        print(f"Model initialized with input_dim={input_dim}, embed_dim={embed_dim}")

        print(f"Parameters Count: {sum(p.numel() for p in self.model.parameters() if p.requires_grad)}")

        # maps (B, T, embed_dim) → (B, T, input_dim)
        self.decoder = Decoder(embed_dim, input_dim)

    def forward(self,
                x: torch.Tensor,
                tgt: Optional[torch.Tensor] = None,
                src_mask: Optional[torch.Tensor] = None,
                src_padding_mask: Optional[torch.Tensor] = None,
                max_len: int = 1
                ) -> torch.Tensor:
        x_embed = self.embedding(x)

        model_args = {
            "x": x_embed,
            "tgt": tgt,
            "src_mask": src_mask,
            "src_padding_mask": src_padding_mask,
            "max_len": max_len,
        }

        sig = inspect.signature(self.model.forward)
        filtered_args = {k: v for k, v in model_args.items() if k in sig.parameters and v is not None}

        out = self.model(**filtered_args)
        out = self.decoder(out)
        return out.squeeze(1)  # assuming output sequence length = 1


def train_full_model(
    model_type: str,
    data_dir: str,
    batch_size: int = 32,
    lr: float = 1e-3,
    epochs: int = 50,
    device: Optional[str] = None,
    save_path: str = "best_model.pt"
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    data_tensors, masks = get_data_and_mask(data_dir)
    x_train, y_train = data_tensors.train_input, data_tensors.train_target.squeeze(1)
    # Determine closing index for last hour of each 7-hour block
    feature_dim = x_train.size(-1)
    features_per_hour = 5
    word_len = feature_dim // features_per_hour
    closing_idx = (word_len - 1) * features_per_hour + 3  # last hour close position
    # Compute normalization stats for closing price at block end
    train_close = y_train[:, closing_idx]
    mean_close_train = train_close.mean().item()
    std_close_train = train_close.std().item()

    x_val, y_val = data_tensors.val_input, data_tensors.val_target.squeeze(1)
    train_mask = masks['train']
    val_mask = masks['val']

    print("Got data tensors:")
    print(data_tensors.summarize())

    train_ds = TensorDataset(x_train, y_train, train_mask)
    val_ds = TensorDataset(x_val, y_val, val_mask)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    model = FullModel(model_type, input_dim=x_train.size(-1), embed_dim=128)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    best_val_loss = float('inf')

    # Early stopping and LR scheduler setup
    patience = 5  # Number of epochs to wait for improvement
    wait = 0
    best_epoch = 0
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    for epoch in range(1, epochs + 1):
        # Training
        model.train()
        total_loss = 0.0
        for (xb, yb, mb) in tqdm(train_loader, desc=f"Epoch {epoch:>2}"):
            xb, yb, mb = xb.to(device), yb.to(device), mb.to(device)

            optimizer.zero_grad()
            src_padding_mask = (mb == 0)  # True where padding

            preds = model(xb, src_padding_mask=src_padding_mask)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * xb.size(0)

        avg_train = total_loss / x_train.size(0)

        # Validation
        model.eval()
        total_val = 0.0
        with torch.no_grad():
            for bidx, (xb, yb, mb) in enumerate(val_loader):
                xb, yb, mb = xb.to(device), yb.to(device), mb.to(device)
                src_padding_mask = (mb == 0)

                preds = model(xb, src_padding_mask=src_padding_mask)
                total_val += criterion(preds, yb).item() * xb.size(0)

        avg_val = total_val / x_val.size(0)
        print(f"Epoch {epoch:>2}: Train Loss={avg_train:.4f}, Val Loss={avg_val:.4f}")

        scheduler.step()

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            best_epoch = epoch
            wait = 0
            torch.save(model.state_dict(), save_path)
            print(f"  ↪️  New best model saved (val loss {best_val_loss:.4f})")
        else:
            wait += 1
            print(f"  No improvement for {wait} epochs.")
            if wait >= patience:
                print(f"Early stopping at epoch {epoch}. Best epoch was {best_epoch}.")
                break

    print("Training complete.")
    # Load best model before returning
    model.load_state_dict(torch.load(save_path))

    # Evaluate on validation set and plot metrics
    model.eval()
    y_true_list, y_pred_list = [], []
    with torch.no_grad():
        for xb, yb, mb in val_loader:
            xb, yb, mb = xb.to(device), yb.to(device), mb.to(device)
            preds = model(xb, src_padding_mask=(mb == 0))
            y_true_list.append(yb.cpu().numpy())
            y_pred_list.append(preds.cpu().numpy())
    y_true = np.concatenate(y_true_list, axis=0)
    y_pred = np.concatenate(y_pred_list, axis=0)
    # Ensure arrays are 2D for sklearn metrics
    if y_true.ndim > 2:
        y_true = np.squeeze(y_true, axis=1)
    if y_pred.ndim > 2:
        y_pred = np.squeeze(y_pred, axis=1)

    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    # Create timestamped directory for plots
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    plot_dir = os.path.join('plots', ts)
    os.makedirs(plot_dir, exist_ok=True)

    # Plot metrics
    plt.figure(figsize=(6, 4))
    metrics = {'MSE': mse, 'MAE': mae, 'R2': r2}
    plt.bar(list(metrics.keys()), list(metrics.values()), color=['blue', 'orange', 'green'])
    plt.title('Validation Metrics')
    plt.ylabel('Value')
    plt.tight_layout()
    plot_path = os.path.join(plot_dir, 'validation_metrics.png')
    plt.savefig(plot_path)
    plt.close()
    print(f"Validation metrics plot saved to {plot_path}")

    # Additional diagnostic plots
    plt.figure(figsize=(12, 10))
    # Predictions vs Actuals
    plt.subplot(2, 2, 1)
    plt.scatter(y_true, y_pred, alpha=0.5, color='pink')
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    plt.plot(lims, lims, '--r')
    plt.title('Predictions vs Actuals')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')

    # Time Series Comparison
    plt.subplot(2, 2, 2)
    n_plot = min(100, len(y_true))
    plt.plot(range(n_plot), y_true[:n_plot], label='Actual', color='blue')
    plt.plot(range(n_plot), y_pred[:n_plot], label='Predicted', color='red')
    plt.title('Time Series Comparison')
    plt.xlabel('Time Step')
    plt.ylabel('Value')
    plt.legend()

    # Residuals Plot
    plt.subplot(2, 2, 3)
    residuals = y_true - y_pred
    plt.scatter(y_pred, residuals, alpha=0.5, color='pink')
    plt.hlines(0, xmin=y_pred.min(), xmax=y_pred.max(), colors='r', linestyles='--')
    plt.title('Residuals Plot')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')

    # Error Distribution
    plt.subplot(2, 2, 4)
    # flatten residuals for histogram
    residuals_flat = residuals.flatten()
    plt.hist(residuals_flat, bins=50, color='skyblue', edgecolor='black')
    plt.axvline(0, color='r', linestyle='--')
    plt.title('Error Distribution')
    plt.xlabel('Residuals')
    plt.ylabel('Frequency')

    plt.tight_layout()
    diag_path = os.path.join(plot_dir, 'validation_diagnostics.png')
    plt.savefig(diag_path)
    plt.close()
    print(f"Validation diagnostics plot saved to {diag_path}")

    # Additional R2 on flattened data
    r2_flat = r2_score(y_true.flatten(), y_pred.flatten())
    print(f"R2 (flattened): {r2_flat:.4f}")

    # Plot model prediction vs actual closing prices (denormalized)
    # use closing index from training (last hour in block)
    closing_idx_eval = closing_idx  # already computed above
    actual_close_norm = y_true[:, closing_idx_eval]
    pred_close_norm = y_pred[:, closing_idx_eval]
    # Denormalize using training stats
    actual_close = actual_close_norm * std_close_train + mean_close_train
    pred_close = pred_close_norm * std_close_train + mean_close_train
    plt.figure(figsize=(8, 4))
    plt.plot(actual_close, label='Actual Close', color='blue')
    plt.plot(pred_close, label='Predicted Close', color='red', alpha=0.7)
    plt.title('Denormalized Closing Prices: Actual vs Predicted')
    plt.xlabel('Sample')
    plt.ylabel('Price')
    plt.legend()
    close_path = os.path.join(plot_dir, 'closing_comparison_denorm.png')
    plt.tight_layout()
    plt.savefig(close_path)
    plt.close()
    print(f"Denormalized closing price comparison saved to {close_path}")

    # Global classification, regression, and financial evaluation on denormalized close series
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score
    # Prepare actual and predicted denormalized series
    actual = actual_close
    pred = pred_close
    # Trend classification
    actual_delta = np.diff(actual)
    pred_delta = pred[1:] - actual[:-1]
    y_true_trend = actual_delta > 0
    y_pred_trend = pred_delta > 0
    y_score = pred_delta  # use delta as score for ROC
    class_metrics = {
        'accuracy': accuracy_score(y_true_trend, y_pred_trend),
        'balanced_accuracy': balanced_accuracy_score(y_true_trend, y_pred_trend),
        'f1_score': f1_score(y_true_trend, y_pred_trend),
        'auc_roc': roc_auc_score(y_true_trend, y_score)
    }
    # Regression metrics
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mape = np.mean(np.abs((actual - pred) / actual)) * 100
    reg_metrics = {
        'mae': mean_absolute_error(actual, pred),
        'rmse': rmse,
        'mape': mape,
        'r2': r2_score(actual, pred)
    }
    # Financial metrics: strategy based on trend signals
    returns = actual[1:] / actual[:-1] - 1
    positions = np.where(pred_delta > 0, 1, -1)
    strat_returns = positions * returns
    cum_returns = np.cumprod(1 + strat_returns) - 1
    cumulative_return = cum_returns[-1]
    ann_return = (1 + cumulative_return) ** (252 / len(returns)) - 1
    sr = np.mean(strat_returns) / np.std(strat_returns) * np.sqrt(252) if np.std(strat_returns) > 0 else np.nan
    drawdown = cum_returns - np.maximum.accumulate(cum_returns)
    mdd = drawdown.min()
    fin_metrics = {
        'cumulative_return': cumulative_return,
        'annualized_return': ann_return,
        'sharpe_ratio': sr,
        'max_drawdown': mdd
    }
    # Save all metrics to CSV
    all_metrics = {**class_metrics, **reg_metrics, **fin_metrics}
    metrics_df = pd.DataFrame([all_metrics])
    metrics_csv = os.path.join(plot_dir, 'evaluation_metrics.csv')
    metrics_df.to_csv(metrics_csv, index=False)
    print(f"Evaluation metrics saved to {metrics_csv}")
    # Per-symbol sliding-window prediction and visualization
    from dataModule import Data_Day_Hourly_StocksPrice
    data = Data_Day_Hourly_StocksPrice.from_dir(data_dir)
    # Define parameters for daily blocks
    word_len = 7
    features_per_hour = 5
    close_feat_idx = 3  # index of 'close' in each hourly block
    # Compute index in flattened vector for closing price of last hour in block
    closing_idx_block = (word_len - 1) * features_per_hour + close_feat_idx
    # Limit visualization to a few symbols to avoid clutter
    max_symbols = 3
    symbol_items = list(data.symbols_test_train.items())[:max_symbols]
    for symbol, idx in symbol_items:
        # Directly access train and test DataFrames by index
        try:
            train_df = data.train[idx]
            test_df = data.test[idx]
        except (IndexError, KeyError):
            continue
        # Verify DataFrame types
        if not isinstance(test_df, pd.DataFrame):
            continue
        # Build daily blocks of 7 hours × 5 features
        arr = test_df[['open', 'high', 'low', 'close', 'volume']].values
        blocks = []
        for start in range(0, len(arr), word_len):
            block = arr[start:start + word_len]
            if block.shape[0] == word_len:
                blocks.append(torch.tensor(block.reshape(-1), dtype=torch.float32))
        if not blocks:
            continue
        pred_dict = {}
        # Sliding-window inference over blocks
        # Prepare raw close series and causal stats for inversion
        raw_close = test_df['close']
        pct = raw_close.pct_change().fillna(0)
        exp_vol = pct.expanding().std(ddof=0).replace(0, 1)
        mean_series = raw_close.expanding().mean()
        std_series = raw_close.expanding().std(ddof=0).replace(0, 1)
        for i in range(1, len(blocks)):
            seq_blocks = blocks[max(0, i - MAX_SEQ_LEN):i]
            seq_tensor = torch.stack(seq_blocks).unsqueeze(0).to(device)
            with torch.no_grad():
                out = model(seq_tensor, src_padding_mask=None)
            # Normalized predicted closing feature
            pred_norm = float(out[0, closing_idx_block].item())
            # Determine timestamp and index for prediction
            idx_ts = min(i * word_len + (word_len - 1), len(test_df) - 1)
            ts = test_df['timestamp'].iloc[idx_ts]
            epoch = int(pd.to_datetime(ts).timestamp())
            # Invert causal normalization: z = (x-mean)/std * vol -> x = (z/vol)*std + mean
            vol_i = exp_vol.iloc[idx_ts]
            mean_i = mean_series.iloc[idx_ts]
            std_i = std_series.iloc[idx_ts]
            pred_denorm = (pred_norm / vol_i) * std_i + mean_i
            pred_dict[epoch] = pred_denorm
        # Visualize per-symbol predictions (normalized scale)
        save_v = os.path.join(plot_dir, f"{symbol}_visualize.png")
        data.visualize(symbol, pred=pred_dict, save_path=save_v)
        print(f"Visualization for {symbol} saved to {save_v}")
    # Autoregressive forecasting without data leakage
    print("Generating autoregressive forecast...")
    # Use the first validation sequence as seed
    data_tensors, _ = get_data_and_mask(data_dir)
    x_init = data_tensors.val_input[:1].to(device)
    seq = x_init
    ar_horizon = 20  # number of future steps to forecast
    closing_idx_ar = closing_idx  # same closing index in output vector
    ar_preds = []
    for _ in range(ar_horizon):
        with torch.no_grad():
            out = model(seq, src_padding_mask=None)
        # take closing feature
        val = out.cpu().numpy().flatten()[closing_idx_ar]
        ar_preds.append(val)
        # append prediction as next input, remove oldest
        next_feat = torch.from_numpy(out.cpu().numpy()).unsqueeze(1).to(device)
        seq = torch.cat([seq[:, 1:, :], next_feat], dim=1)
    ar_preds = np.array(ar_preds)
    # denormalize
    ar_preds_denorm = ar_preds * std_close_train + mean_close_train
    plt.figure(figsize=(8, 4))
    plt.plot(ar_preds_denorm, label='AR Forecast', marker='o')
    plt.title('Autoregressive Forecast (Denormalized)')
    plt.xlabel('Forecast Step')
    plt.ylabel('Close Price')
    plt.legend()
    ar_path = os.path.join(plot_dir, 'ar_forecast_denorm.png')
    plt.tight_layout()
    plt.savefig(ar_path)
    plt.close()
    print(f"Autoregressive forecast plot saved to {ar_path}")

    # Autoregressive forecast across the test period for the first symbol
    first_symbol = symbol_items[0][0] if symbol_items else None
    if first_symbol:
        data_vis = Data_Day_Hourly_StocksPrice.from_dir(data_dir)
        idx_sym = data_vis.symbols_test_train.get(first_symbol)
        if idx_sym is not None:
            test_df = data_vis.test[idx_sym]
            # Build blocks from test_df as for sliding-window
            arr = test_df[['open', 'high', 'low', 'close', 'volume']].values
            blocks = [torch.tensor(arr[i:i+word_len].reshape(-1), dtype=torch.float32)
                      for i in range(0, len(arr), word_len) if arr[i:i+word_len].shape[0]==word_len]
            ar_pred_dict = {}
            # Generate AR predictions using only previous actual blocks
            for i in range(1, len(blocks)):
                # Determine timestamp index at start of block i
                ts_idx = min((i-1) * word_len, len(test_df) - 1)
                # Run model on previous blocks
                seq_blocks = blocks[max(0, i-MAX_SEQ_LEN):i]
                seq_tensor = torch.stack(seq_blocks).unsqueeze(0).to(device)
                with torch.no_grad():
                    out = model(seq_tensor, src_padding_mask=None)
                pred_norm = float(out[0, closing_idx_block].item())
                # Invert causal normalization at ts_idx
                raw_close = test_df['close']
                pct = raw_close.pct_change().fillna(0)
                vol = pct.expanding().std(ddof=0).replace(0,1).iloc[ts_idx]
                mean = raw_close.expanding().mean().iloc[ts_idx]
                std = raw_close.expanding().std(ddof=0).replace(0,1).iloc[ts_idx]
                pred_price = (pred_norm/vol)*std + mean
                # Map to timestamp at block start
                ts = test_df['timestamp'].iloc[ts_idx]
                epoch = int(pd.to_datetime(ts).timestamp())
                ar_pred_dict[epoch] = pred_price
            # Visualize AR predictions overlayed
            save_ar = os.path.join(plot_dir, f"{first_symbol}_ar_forecast.png")
            data_vis.visualize(first_symbol, pred=ar_pred_dict, title_pre="AR Forecast: ", save_path=save_ar, show=False)
            print(f"Autoregressive forecast visualization saved to {save_ar}")
            # Save AR predictions to CSV
            csv_path = os.path.join(plot_dir, f"{first_symbol}_ar_predictions.csv")
            pd.DataFrame.from_dict(ar_pred_dict, orient='index', columns=['predicted']).to_csv(csv_path)
            print(f"Autoregressive predictions saved to {csv_path}")
    return model


"""
IMPORTANT, if you are on mps, run this before training:

export PYTORCH_ENABLE_MPS_FALLBACK=1
"""

if __name__ == "__main__":
    model = train_full_model(
        model_type="softalign",  # 'causal', 'encdec', or 'softalign'
        data_dir="data",
        batch_size=32,
        lr=1e-5,
        epochs=3,
        save_path="best_encdec.pt",
        device="mps"
    )