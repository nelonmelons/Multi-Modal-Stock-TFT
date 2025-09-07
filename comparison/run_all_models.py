#!/usr/bin/env python3
"""
Main pipeline to run all model comparisons.
"""
import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import re
import pandas as pd
import numpy as np
import torch
from datetime import datetime
from prettytable import PrettyTable
# Added imports for artifact generation
import json
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
sns.set_palette('husl')

# Helper functions for figures / tables
def _save_table_figure(df: pd.DataFrame, title: str, out_path: str):
    try:
        fig, ax = plt.subplots(figsize=(max(6, len(df.columns)*1.2), 0.6*max(1,len(df))+1.5))
        ax.axis('off')
        tbl = ax.table(cellText=df.values, colLabels=df.columns, loc='center')
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(8)
        ax.set_title(title, fontweight='bold')
        fig.tight_layout()
        fig.savefig(out_path, dpi=200)
        plt.close(fig)
    except Exception as e:
        print(f"⚠️ Failed to save table figure {out_path}: {e}")

def _compute_equity_curve(pred_files: list[str], models: list[str]) -> pd.DataFrame:
    rows = []
    for fp in pred_files:
        if not os.path.exists(fp):
            continue
        m = re.search(r'EXP-[A-Z]\d+_(.*?)_\d+\.parquet$', os.path.basename(fp))
        model = m.group(1) if m else 'unknown'
        if model not in models:
            continue
        try:
            df = pd.read_parquet(fp)
            if 'horizon' not in df.columns:
                continue
            df = df[df['horizon'] == 21]
            df['model'] = model
            rows.append(df[['date','ticker','horizon','y_true','y_pred','model']])
        except Exception as e:
            print(f"Failed reading {fp}: {e}")
    if not rows:
        return pd.DataFrame()
    all_df = pd.concat(rows, ignore_index=True)
    all_df['date'] = pd.to_datetime(all_df['date'])
    equity_rows = []
    for model, g in all_df.groupby('model'):
        daily = g.groupby('date').apply(lambda d: np.nanmean(np.sign(d['y_pred']) * d['y_true']))
        cum = daily.cumsum()
        tmp = pd.DataFrame({'date': daily.index, 'daily_pnl': daily.values, 'cum_pnl': cum.values, 'model': model})
        equity_rows.append(tmp)
    return pd.concat(equity_rows, ignore_index=True) if equity_rows else pd.DataFrame()

def _plot_equity_curve(eq: pd.DataFrame, out_path: str):
    if eq.empty:
        print('⚠️ Equity curve data empty; skipping Fig 3')
        return
    fig, ax = plt.subplots(figsize=(10,5))
    for model, g in eq.groupby('model'):
        g = g.sort_values('date')
        ax.plot(g['date'], g['cum_pnl'], label=model)
    ax.set_title('Long-Short Equity Curve @21 (Sign(pred)*return)')
    ax.set_ylabel('Cumulative PnL')
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

def _plot_da_by_horizon(agg_df: pd.DataFrame, out_path: str):
    if agg_df.empty:
        print('⚠️ Empty metrics for DA figure')
        return
    fig, ax = plt.subplots(figsize=(8,5))
    pivot = agg_df.pivot(index='model', columns='horizon', values='DA_mean')
    desired_order = ['Ridge','XGBoost','RandomForest','GRU','LSTM','TFT']
    pivot = pivot.reindex([m for m in desired_order if m in pivot.index])
    pivot.plot(kind='bar', ax=ax)
    ax.set_ylabel('Directional Accuracy')
    ax.set_title('Fig 1: Directional Accuracy by Horizon (Test)')
    ax.legend(title='H')
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

def _plot_regime_heatmap(regime_df: pd.DataFrame, out_path: str):
    if regime_df.empty:
        print('⚠️ Regime data empty; skipping Fig 2')
        return
    pivot = regime_df.pivot(index='model', columns='regime', values='RMSE_21')
    fig, ax = plt.subplots(figsize=(8,4))
    sns.heatmap(pivot, annot=True, fmt='.3f', cmap='viridis', ax=ax)
    ax.set_title('Fig 2: RMSE@21 by Regime')
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

from main import LeakageFreeDataLoader
from models import LSTMModel, GRUModel, TFT
from train import train_model, train_tft_model
from model.tft_model import setup_device
from evaluation import evaluate_multi_horizon_predictions
import warnings
warnings.filterwarnings('ignore')

# Universe and baselines
try:
    from src.universe import DOW30_2018 as DOW_UNIVERSE
except ImportError:
    # Fallback universe if src.universe is not available
    DOW_UNIVERSE = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'AMD', 'INTC', 'JPM']
    print("⚠️  Warning: Using fallback universe. Install src.universe module for full DOW30 support.")

from src.sk_baselines import fit_predict_baseline
from dataModule.datamodule import NumericDataModule

# Fixed experiment contract - can be overridden by CLI arguments
DEFAULT_TRAIN_START = '2016-01-01'
DEFAULT_TRAIN_END = '2019-12-31'
DEFAULT_VAL_START = '2020-01-01'
DEFAULT_VAL_END = '2020-12-31'
DEFAULT_TEST_START = '2021-01-01'
DEFAULT_TEST_END = '2024-12-31'
DEFAULT_HORIZONS = [1, 5, 21]
DEFAULT_LOOKBACK = 60
DEFAULT_PREDICT_LEN = 21
DEFAULT_SEEDS = [42, 43, 44]

# Global variables that can be set by CLI
TRAIN_START = DEFAULT_TRAIN_START
TRAIN_END = DEFAULT_TRAIN_END
VAL_START = DEFAULT_VAL_START
VAL_END = DEFAULT_VAL_END
TEST_START = DEFAULT_TEST_START
TEST_END = DEFAULT_TEST_END
HORIZONS = DEFAULT_HORIZONS
LOOKBACK = DEFAULT_LOOKBACK
PREDICT_LEN = DEFAULT_PREDICT_LEN
SEEDS = DEFAULT_SEEDS

ARTIFACT_DIR = os.path.join('artifacts')
PRED_DIR = os.path.join(ARTIFACT_DIR, 'predictions')
METRICS_DIR = os.path.join(ARTIFACT_DIR, 'metrics')
SLICES_DIR = os.path.join(ARTIFACT_DIR, 'slices')
FIG_DIR = os.path.join(ARTIFACT_DIR, 'figures')
# Note: Directories are created in update_global_config() after CLI parsing


def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_baseline_grids():
    return {
        'Ridge': [{'alpha': a} for a in [0.1, 1.0, 10.0]],
        'RandomForest': [
            {'n_estimators': n, 'max_depth': d, 'min_samples_leaf': m}
            for n in [300, 800] for d in [10, None] for m in [1, 5]
        ],
        'XGBoost': [
            {'n_estimators': n, 'learning_rate': lr, 'max_depth': d, 'subsample': 0.8, 'colsample_bytree': 0.8, 'reg_lambda': rl}
            for n in [400, 800] for lr in [0.05, 0.03] for d in [4, 6] for rl in [1, 3]
        ],
    }


def tune_baseline(model_name: str, train_df: pd.DataFrame, val_df: pd.DataFrame) -> dict:
    """Select hyperparams by Val-2020 RMSE@21. ✅ FIXED: No data leakage."""
    grids = get_baseline_grids()[model_name]
    best_rmse, best_params = float('inf'), grids[0]
    
    print(f"🔍 Tuning {model_name}: trying {len(grids)} parameter combinations...")
    for i, params in enumerate(grids):
        # ✅ Train on train_df, evaluate on val_df - NO LEAKAGE
        res = fit_predict_baseline(model_name, train_df, val_df, horizons=[21], model_params={model_name: params})
        rmse = res.metrics.get('horizon_21', {}).get('RMSE', np.inf)
        if rmse < best_rmse:
            best_rmse, best_params = rmse, params
        if i % max(1, len(grids)//5) == 0:
            print(f"   Progress: {i+1}/{len(grids)}, current best RMSE: {best_rmse:.6f}")
    
    print(f"✅ Best {model_name} params (val RMSE={best_rmse:.6f}): {best_params}")
    return best_params


def build_deep_model(name: str, input_dim: int, device: torch.device):
    if name == 'GRU':
        return GRUModel(input_dim=input_dim, hidden_dim=128, num_layers=2, output_dim=PREDICT_LEN, dropout=0.1).to(device)
    if name == 'LSTM':
        return LSTMModel(input_dim=input_dim, hidden_dim=128, num_layers=2, output_dim=PREDICT_LEN, dropout=0.1).to(device)
    if name == 'TFT':
        # Set news_dim=0 to disable news processing completely
        return TFT(input_size=input_dim, news_dim=0, hidden_size=128, num_heads=4, dropout=0.1, prediction_len=PREDICT_LEN).to(device)
    raise ValueError(f"Unknown deep model: {name}")


def select_feature_columns_by_modality(df: pd.DataFrame, modality: str, horizons: list[int]) -> list[str]:
    """Return feature columns for a given modality label."""
    # Exclude all target columns from features
    target_cols = [c for c in df.columns if c.startswith('target_')]
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    base_feats = [c for c in num_cols if c not in target_cols]

    def tech_mask(c: str) -> bool:
        toks = ['sma', 'ema', 'rsi', 'macd', 'bb_', 'atr', 'volatility', 'price_change', 'high_20d', 'low_20d', 'price_position', 'volume_ratio', 'volume_sma']
        return any(t in c for t in toks)

    def earnings_mask(c: str) -> bool:
        toks = ['eps_', 'revenue_', 'days_to_next_earnings', 'days_since_earnings', 'is_earnings_day', 'earnings_in_prediction_window', 'days_to_earnings_in_window']
        return any(c.startswith(t) or t in c for t in toks)

    tech = [c for c in base_feats if tech_mask(c)]
    earn = [c for c in base_feats if earnings_mask(c)]

    if modality == 'Tech':
        return tech
    if modality == 'Tech+Earnings':
        return list(sorted(set(tech + earn)))
    return base_feats


def make_filtered_datamodule(dm: NumericDataModule, feature_cols: list[str]) -> NumericDataModule:
    """Create a new NumericDataModule with only selected feature columns."""
    # Determine target columns directly from the dataframe to avoid relying on internal attributes
    target_cols = [c for c in dm.df.columns if c.startswith('target_')]
    keep_cols = ['date', 'symbol'] + feature_cols + target_cols
    # Guard for columns that might be missing
    keep_cols = [c for c in keep_cols if c in dm.df.columns]
    fdf = dm.df[keep_cols].copy()
    fdm = NumericDataModule(
        feature_df=fdf,
        batch_size=dm.batch_size,
        date_col=dm.date_col,
        target_col=dm.target_identifier,
        train_range=(dm.train_range[0].strftime('%Y-%m-%d'), dm.train_range[1].strftime('%Y-%m-%d')) if dm.train_range else None,
        val_range=(dm.val_range[0].strftime('%Y-%m-%d'), dm.val_range[1].strftime('%Y-%m-%d')) if dm.val_range else None,
        test_range=(dm.test_range[0].strftime('%Y-%m-%d'), dm.test_range[1].strftime('%Y-%m-%d')) if dm.test_range else None,
        embargo_days=dm.embargo_days,
        horizons=dm.horizons,
    )
    fdm.setup()
    return fdm


def compute_slices_from_predictions(files: list[str], events: dict):
    """Compute regime and earnings slices from EXP-A3 predictions (h=21)."""
    regimes = {
        'covid_crash': ('2020-02-15', '2020-04-30'),
        'bear_2022': ('2022-01-01', '2022-10-15'),
        'rally_2023_2024': ('2023-01-01', '2024-12-31'),
    }
    all_preds = []
    for fp in files:
        if os.path.exists(fp):
            df = pd.read_parquet(fp)
            # Infer model from filename
            m = re.search(r'EXP-[A-Z]\d+_(.*?)_\d+\.parquet$', os.path.basename(fp))
            model_name = m.group(1) if m else 'unknown'
            df['model'] = model_name
            all_preds.append(df)
    if not all_preds:
        return pd.DataFrame(), pd.DataFrame()
    df = pd.concat(all_preds, ignore_index=True)
    df['date'] = pd.to_datetime(df['date'])

    rows = []
    for regime, (s, e) in regimes.items():
        mask = (df['date'] >= s) & (df['date'] <= e) & (df['horizon'] == 21)
        sub = df[mask]
        if sub.empty:
            continue
        for model in sub['model'].unique():
            subm = sub[sub['model'] == model]
            rmse = float(np.sqrt(np.mean((subm['y_pred'] - subm['y_true']) ** 2)))
            da = float(np.mean(np.sign(subm['y_pred']) == np.sign(subm['y_true'])))
            rows.append({'regime': regime, 'model': model, 'RMSE_21': rmse, 'DA_21': da})
    regimes_df = pd.DataFrame(rows)

    rows = []
    earn_map = {}
    for tkr, ev in (events or {}).items():
        if 'earnings' in ev and ev['earnings']:
            earn_map[tkr] = pd.to_datetime(pd.Series(ev['earnings']))
    if earn_map:
        df['is_earn_window'] = False
        for tkr, dates in earn_map.items():
            mask_t = df['ticker'] == tkr
            if not mask_t.any():
                continue
            for d in dates:
                win_start = d - pd.tseries.offsets.BDay(3)
                win_end = d + pd.tseries.offsets.BDay(3)
                df.loc[mask_t & (df['date'] >= win_start) & (df['date'] <= win_end), 'is_earn_window'] = True
        for flag, label in [(True, 'earnings_window'), (False, 'non_earnings')]:
            sub = df[(df['is_earn_window'] == flag) & (df['horizon'] == 21)]
            if sub.empty:
                continue
            rmse = float(np.sqrt(np.mean((sub['y_pred'] - sub['y_true']) ** 2)))
            da = float(np.mean(np.sign(sub['y_pred']) == np.sign(sub['y_true'])))
            rows.append({'slice': label, 'RMSE_21': rmse, 'DA_21': da})
    earnings_df = pd.DataFrame(rows)
    return regimes_df, earnings_df


def parse_arguments():
    """Parse command line arguments for date ranges and universe configuration."""
    parser = argparse.ArgumentParser(
        description='Run multi-modal stock prediction model comparison pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default DOW30 universe and date ranges
  python run_all_models.py

  # Run with custom date ranges
  python run_all_models.py --train-start 2018-01-01 --train-end 2021-12-31 --test-start 2022-01-01 --test-end 2024-12-31

  # Run with custom universe (comma-separated symbols)
  python run_all_models.py --universe AAPL,MSFT,GOOGL,TSLA,NVDA

  # Run with custom universe from file
  python run_all_models.py --universe-file symbols.txt

  # Run with custom horizons and seeds
  python run_all_models.py --horizons 1,5,10,21 --seeds 42,43,44,45,46
        """
    )
    
    # Date range arguments
    parser.add_argument('--train-start', type=str, default=DEFAULT_TRAIN_START,
                       help=f'Training start date (YYYY-MM-DD, default: {DEFAULT_TRAIN_START})')
    parser.add_argument('--train-end', type=str, default=DEFAULT_TRAIN_END,
                       help=f'Training end date (YYYY-MM-DD, default: {DEFAULT_TRAIN_END})')
    parser.add_argument('--val-start', type=str, default=DEFAULT_VAL_START,
                       help=f'Validation start date (YYYY-MM-DD, default: {DEFAULT_VAL_START})')
    parser.add_argument('--val-end', type=str, default=DEFAULT_VAL_END,
                       help=f'Validation end date (YYYY-MM-DD, default: {DEFAULT_VAL_END})')
    parser.add_argument('--test-start', type=str, default=DEFAULT_TEST_START,
                       help=f'Test start date (YYYY-MM-DD, default: {DEFAULT_TEST_START})')
    parser.add_argument('--test-end', type=str, default=DEFAULT_TEST_END,
                       help=f'Test end date (YYYY-MM-DD, default: {DEFAULT_TEST_END})')
    
    # Universe arguments
    universe_group = parser.add_mutually_exclusive_group()
    universe_group.add_argument('--universe', type=str,
                               help='Comma-separated list of stock symbols (e.g., AAPL,MSFT,GOOGL)')
    universe_group.add_argument('--universe-file', type=str,
                               help='Path to text file containing stock symbols (one per line)')
    universe_group.add_argument('--universe-preset', type=str, choices=['DOW30', 'SP500', 'NASDAQ100'],
                               help='Use predefined universe (DOW30, SP500, NASDAQ100)')
    
    # Model configuration arguments
    parser.add_argument('--horizons', type=str, default=','.join(map(str, DEFAULT_HORIZONS)),
                       help=f'Comma-separated prediction horizons in days (default: {",".join(map(str, DEFAULT_HORIZONS))})')
    parser.add_argument('--seeds', type=str, default=','.join(map(str, DEFAULT_SEEDS)),
                       help=f'Comma-separated random seeds (default: {",".join(map(str, DEFAULT_SEEDS))})')
    parser.add_argument('--lookback', type=int, default=DEFAULT_LOOKBACK,
                       help=f'Lookback window length (default: {DEFAULT_LOOKBACK})')
    parser.add_argument('--predict-len', type=int, default=DEFAULT_PREDICT_LEN,
                       help=f'Prediction sequence length (default: {DEFAULT_PREDICT_LEN})')
    
    # Output arguments
    parser.add_argument('--output-dir', type=str, default='artifacts',
                       help='Output directory for results (default: artifacts)')
    
    # Model selection arguments
    parser.add_argument('--models', type=str, 
                       default='Ridge,XGBoost,RandomForest,GRU,LSTM,TFT',
                       help='Comma-separated list of models to run (default: all models)')
    parser.add_argument('--experiments', type=str,
                       choices=['A', 'B', 'ALL'], default='ALL',
                       help='Which experiments to run: A (main comparison), B (ablations), or ALL (default: ALL)')
    
    return parser.parse_args()


def load_universe_from_args(args):
    """Load stock universe based on command line arguments."""
    if args.universe:
        # Comma-separated symbols
        symbols = [s.strip().upper() for s in args.universe.split(',')]
        print(f"✅ Using custom universe: {len(symbols)} symbols")
        return symbols
    
    elif args.universe_file:
        # Load from file
        if not os.path.exists(args.universe_file):
            raise FileNotFoundError(f"Universe file not found: {args.universe_file}")
        
        with open(args.universe_file, 'r') as f:
            symbols = [line.strip().upper() for line in f if line.strip() and not line.startswith('#')]
        
        print(f"✅ Loaded universe from {args.universe_file}: {len(symbols)} symbols")
        return symbols
    
    elif args.universe_preset:
        # Use predefined universe
        if args.universe_preset == 'DOW30':
            from src.universe import DOW30_2018 as universe
        elif args.universe_preset == 'SP500':
            # You would need to implement this
            raise NotImplementedError("SP500 universe not implemented yet")
        elif args.universe_preset == 'NASDAQ100':
            # You would need to implement this
            raise NotImplementedError("NASDAQ100 universe not implemented yet")
        
        print(f"✅ Using {args.universe_preset} universe: {len(universe)} symbols")
        return universe
    
    else:
        # Default to DOW30
        from src.universe import DOW30_2018 as DOW_UNIVERSE
        print(f"✅ Using default DOW30 universe: {len(DOW_UNIVERSE)} symbols")
        return DOW_UNIVERSE


def validate_date_ranges(args):
    """Validate that date ranges are logical and properly formatted."""
    from datetime import datetime
    
    try:
        # Parse dates
        train_start = datetime.strptime(args.train_start, '%Y-%m-%d')
        train_end = datetime.strptime(args.train_end, '%Y-%m-%d')
        val_start = datetime.strptime(args.val_start, '%Y-%m-%d')
        val_end = datetime.strptime(args.val_end, '%Y-%m-%d')
        test_start = datetime.strptime(args.test_start, '%Y-%m-%d')
        test_end = datetime.strptime(args.test_end, '%Y-%m-%d')
        
        # Validate logical order
        if not (train_start < train_end < val_start < val_end < test_start < test_end):
            raise ValueError("Date ranges must be in order: train_start < train_end < val_start < val_end < test_start < test_end")
        
        # Validate minimum periods
        if (train_end - train_start).days < 365:
            print("⚠️  Warning: Training period is less than 1 year")
        if (val_end - val_start).days < 90:
            print("⚠️  Warning: Validation period is less than 3 months")
        if (test_end - test_start).days < 180:
            print("⚠️  Warning: Test period is less than 6 months")
        
        print("✅ Date ranges validated successfully")
        
    except ValueError as e:
        raise ValueError(f"Invalid date format or range: {e}")


def update_global_config(args):
    """Update global configuration variables based on CLI arguments."""
    global TRAIN_START, TRAIN_END, VAL_START, VAL_END, TEST_START, TEST_END
    global HORIZONS, LOOKBACK, PREDICT_LEN, SEEDS, ARTIFACT_DIR
    global PRED_DIR, METRICS_DIR, SLICES_DIR, FIG_DIR
    
    # Update date ranges
    TRAIN_START = args.train_start
    TRAIN_END = args.train_end
    VAL_START = args.val_start
    VAL_END = args.val_end
    TEST_START = args.test_start
    TEST_END = args.test_end
    
    # Update model configuration
    HORIZONS = [int(h.strip()) for h in args.horizons.split(',')]
    SEEDS = [int(s.strip()) for s in args.seeds.split(',')]
    LOOKBACK = args.lookback
    PREDICT_LEN = args.predict_len
    
    # Update output directory and recreate paths
    ARTIFACT_DIR = args.output_dir
    PRED_DIR = os.path.join(ARTIFACT_DIR, 'predictions')
    METRICS_DIR = os.path.join(ARTIFACT_DIR, 'metrics')
    SLICES_DIR = os.path.join(ARTIFACT_DIR, 'slices')
    FIG_DIR = os.path.join(ARTIFACT_DIR, 'figures')
    
    # Create directories
    os.makedirs(PRED_DIR, exist_ok=True)
    os.makedirs(METRICS_DIR, exist_ok=True)
    os.makedirs(SLICES_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)
    
    print("✅ Global configuration updated from CLI arguments")


def run_pipeline(custom_universe=None):
    print("🚀 Starting Full Model Comparison Pipeline...")
    print("✅ FIXED: This pipeline now prevents data leakage by:")
    print("   1. Training models on Train data only (2016-2019)")
    print("   2. Using Val data only for hyperparameter tuning (2020)")
    print("   3. Using Test data only for final evaluation (2021-2024)")
    print("   4. No future information leakage into past training")
    print("   5. Early stopping prevents overfitting on validation set")
    print()

    device = setup_device()
    fred_api_key = os.getenv('FRED_API_KEY')
    api_ninjas_key = os.getenv('API_NINJAS_KEY')

    # Use custom universe if provided, otherwise use default DOW30
    universe = custom_universe if custom_universe is not None else DOW_UNIVERSE

    base_config = {
        'symbols': universe,
        'start_date': TRAIN_START,
        'end_date': TEST_END,
        'train_start': TRAIN_START,
        'train_end': TRAIN_END,
        'val_start': VAL_START,
        'val_end': VAL_END,
        'test_start': TEST_START,
        'test_end': TEST_END,
        'encoder_len': LOOKBACK,
        'predict_len': PREDICT_LEN,
        'batch_size': 256,
        'fred_api_key': fred_api_key,
        'api_ninjas_key': api_ninjas_key,
        'horizons': HORIZONS,
        'device': device,
    }

    print("Contract:")
    print(f"  Universe size: {len(universe)}")
    print(f"  Windows: Train {TRAIN_START}->{TRAIN_END}, Val {VAL_START}->{VAL_END}, Test {TEST_START}->{TEST_END}")
    print(f"  Horizons: {HORIZONS}")

    experiments = []
    for h in HORIZONS:
        experiments.append({'id': f'EXP-A{HORIZONS.index(h)+1}', 'desc': f'Main comparison h={h}', 'models': ['Ridge', 'XGBoost', 'RandomForest', 'GRU', 'LSTM', 'TFT'], 'horizon_eval': HORIZONS, 'modality': 'Tech+Earnings'})
    experiments += [
        {'id': 'EXP-B1', 'desc': 'Tech only', 'models': ['XGBoost', 'TFT'], 'horizon_eval': [21], 'modality': 'Tech'},
        {'id': 'EXP-B2', 'desc': 'Tech+Earnings', 'models': ['XGBoost', 'TFT'], 'horizon_eval': [21], 'modality': 'Tech+Earnings'},
    ]

    all_metric_rows = []
    exp_a3_files = []

    for seed in SEEDS:
        print(f"\n===== Seed {seed} =====")
        set_seed(seed)

        data_loader = LeakageFreeDataLoader(base_config)
        data_module = data_loader.load_complete_pipeline()
        print("✅ Data module loaded. Proceeding to model training and evaluation...")

        sample_batch = next(iter(data_module.train_loader))
        input_dim = sample_batch[0].shape[-1]

        train_df = data_module.train_df.copy()
        val_df = data_module.val_df.copy()
        test_df = data_module.test_df.copy()
        print(f"📦 DataFrames prepared: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")

        # Tune baselines once per seed (on Val-2020, RMSE@21)
        baseline_params = {}
        for m in ['Ridge', 'RandomForest', 'XGBoost']:
            try:
                print(f"🔍 Tuning baseline {m} on Val-2020...")
                baseline_params[m] = tune_baseline(m, train_df, val_df)
                print(f"✅ Best params for {m}: {baseline_params[m]}")
            except Exception as e:
                print(f"Baseline tuning failed for {m}: {e}")
                baseline_params[m] = {}

        for exp in experiments:
            exp_id = exp['id']
            print(f"\n--- Running {exp_id}: {exp['desc']} ---")
            modality = exp['modality']

            # Determine feature columns for baselines and filtered TFT in ablations
            feat_cols = select_feature_columns_by_modality(pd.concat([train_df, val_df, test_df]).drop(columns=[c for c in ['date','symbol'] if c in train_df.columns]), modality, HORIZONS)
            print(f"🔬 Selected feature columns for modality '{modality}': {len(feat_cols)} columns")

            for model_name in exp['models']:
                print(f"🚦 Starting training for model: {model_name}")
                detailed = pd.DataFrame()
                metrics = {}

                if model_name in ['Ridge', 'RandomForest', 'XGBoost']:
                    # ✅ FIXED: Train on Train only, not Train+Val to prevent leakage
                    params = baseline_params.get(model_name, {})
                    print(f"🛠 Training {model_name} on Train only, evaluating on Test...")
                    res = fit_predict_baseline(model_name, train_df, test_df, horizons=exp['horizon_eval'], model_params={model_name: params}, feature_cols=feat_cols, random_state=seed)
                    print(f"✅ {model_name} training complete. Metrics computed.")
                    detailed = res.predictions.copy()
                    metrics = res.metrics
                    print(f"📊 {model_name} metrics: {list(metrics.keys())}")
                elif model_name in ['GRU', 'LSTM', 'TFT']:
                    # For ablations, filter features only for TFT (as per spec); A-experiments use full features
                    dm_for_model = data_module
                    if exp_id.startswith('EXP-B') and model_name == 'TFT':
                        try:
                            print(f"🔬 Building filtered DataModule for TFT ablation...")
                            dm_for_model = make_filtered_datamodule(data_module, feat_cols)
                            input_dim = next(iter(dm_for_model.train_loader))[0].shape[-1]
                        except Exception as e:
                            print(f"Failed to build filtered DataModule for TFT ablation: {e}")
                    print(f"🛠 Training {model_name} on Train only, validating on Val, evaluating on Test...")
                    model = build_deep_model(model_name, input_dim, device)
                    class DMView:
                        def __init__(self, dm):
                            # ✅ FIXED: Use proper train/val/test splits - NO LEAKAGE
                            self.train_loader = dm.train_loader      # Train on 2016-2019 only
                            self.val_loader = dm.val_loader          # Validate on 2020 only  
                            self.val_df = dm.val_df
                            self.test_loader = dm.test_loader        # Test on 2021-2024
                            self.test_df = dm.test_df
                    dm_view = DMView(dm_for_model)
                    print(f"▶️ Starting {model_name} training loop...")
                    if model_name == 'TFT':
                        trained_model, _ = train_tft_model(model, dm_view, epochs=30, lr=3e-4, device=device, patience=5)
                    else:
                        trained_model, _ = train_model(model, dm_view, epochs=30, lr=1e-3, device=device, patience=5)
                    print(f"✅ {model_name} training complete. Evaluating...")
                    eval_res = evaluate_multi_horizon_predictions(trained_model, dm_for_model, horizons=exp['horizon_eval'], split='test')
                    metrics = eval_res['horizon_metrics']
                    det = eval_res['detailed_predictions']
                    print(f"📊 {model_name} evaluation complete. Metrics: {list(metrics.keys())}")
                    if not det.empty:
                        # Rename columns to match expected schema: symbol->ticker, actual->y_true, prediction->y_pred
                        detailed = det.rename(columns={'symbol': 'ticker', 'actual': 'y_true', 'prediction': 'y_pred'})
                        # Select the required columns if they exist
                        if all(col in detailed.columns for col in ['date','ticker','horizon','y_true','y_pred']):
                            detailed = detailed[['date','ticker','horizon','y_true','y_pred']].copy()
                        else:
                            print(f"⚠️ Missing required columns in detailed predictions for {model_name}. Available: {list(detailed.columns)}")
                else:
                    print(f"Unknown model {model_name}")
                    continue

                # Save predictions parquet per run (exact schema)
                if not detailed.empty:
                    # Handle different column naming conventions
                    # Baseline models use 'actual'/'prediction', deep models use 'y_true'/'y_pred'
                    save_df = detailed.copy()
                    
                    # Standardize column names to y_true/y_pred
                    if 'actual' in save_df.columns and 'prediction' in save_df.columns:
                        save_df = save_df.rename(columns={'actual': 'y_true', 'prediction': 'y_pred'})
                    
                    # Ensure we have the required columns
                    required_cols = ['date', 'ticker', 'horizon', 'y_true', 'y_pred']
                    if all(col in save_df.columns for col in required_cols):
                        save_df = save_df[required_cols].copy()
                        out_path = os.path.join(PRED_DIR, f"{exp_id}_{model_name}_{seed}.parquet")
                        save_df.to_parquet(out_path, index=False)
                        print(f"💾 Saved predictions for {model_name} to {out_path}")
                        if exp_id == 'EXP-A3':
                            exp_a3_files.append(out_path)
                    else:
                        print(f"⚠️ Missing required columns for {model_name}. Available: {list(save_df.columns)}")
                else:
                    print(f"⚠️ No predictions to save for {model_name}")

                # Collect metrics rows for final table (Test metrics)
                for h in HORIZONS:
                    key = f'horizon_{h}'
                    if key in metrics:
                        m = metrics[key]
                        all_metric_rows.append({'exp': exp_id, 'seed': seed, 'model': model_name, 'horizon': h, **m})

                if device.type == 'cuda':
                    torch.cuda.empty_cache()

    # Aggregate metrics mean±sd over seeds for EXP-A (Table 1)
    metrics_df = pd.DataFrame(all_metric_rows)
    table1 = pd.DataFrame()
    if not metrics_df.empty:
        # Per-experiment metrics CSVs
        for exp_id, gexp in metrics_df.groupby('exp'):
            agg = gexp.groupby(['model','horizon']).agg(RMSE_mean=('RMSE','mean'), RMSE_std=('RMSE','std'), MAE_mean=('MAE','mean'), MAE_std=('MAE','std'), R2_mean=('R2','mean'), R2_std=('R2','std'), DA_mean=('DA','mean'), DA_std=('DA','std')).reset_index()
            agg.to_csv(os.path.join(METRICS_DIR, f"{exp_id}.csv"), index=False)
        a_mask = metrics_df['exp'].str.startswith('EXP-A')
        table1 = metrics_df[a_mask].groupby(['model','horizon']).agg(RMSE_mean=('RMSE','mean'), RMSE_std=('RMSE','std'), MAE_mean=('MAE','mean'), MAE_std=('MAE','std'), R2_mean=('R2','mean'), R2_std=('R2','std'), DA_mean=('DA','mean'), DA_std=('DA','std')).reset_index()
        table1.to_csv(os.path.join(METRICS_DIR, 'Table1_EXP-A.csv'), index=False)
        # Save Table 1 figure
        t1_disp = table1.copy()
        for mcol in ['RMSE','MAE','R2','DA']:
            mean_col = f'{mcol}_mean'; std_col = f'{mcol}_std'
            if mean_col in t1_disp and std_col in t1_disp:
                t1_disp[mcol] = t1_disp.apply(lambda r: f"{r[mean_col]:.4f}±{(0 if np.isnan(r[std_col]) else r[std_col]):.4f}", axis=1)
        cols_show = ['model','horizon'] + [c for c in ['RMSE','MAE','R2','DA'] if c in t1_disp]
        t1_disp = t1_disp[cols_show]
        _save_table_figure(t1_disp, 'Table 1: Test Metrics (Mean±SD over seeds)', os.path.join(FIG_DIR,'table1_metrics.png'))
        # Fig 1 DA by horizon
        _plot_da_by_horizon(table1, os.path.join(FIG_DIR,'fig1_da_by_horizon.png'))
    # Ablation Table 2 (EXP-B*, horizon=21, models XGBoost/TFT)
    ablation_mask = metrics_df['exp'].str.startswith('EXP-B') if not metrics_df.empty else []
    if metrics_df.empty or not any(ablation_mask):
        print("⚠️ No ablation metrics for Table 2")
    else:
        abl = metrics_df[ablation_mask & (metrics_df['horizon'] == 21) & metrics_df['model'].isin(['XGBoost','TFT'])]
        if not abl.empty:
            abl_agg = abl.groupby(['exp','model']).agg(RMSE_mean=('RMSE','mean'), RMSE_std=('RMSE','std'), R2_mean=('R2','mean'), R2_std=('R2','std'), DA_mean=('DA','mean'), DA_std=('DA','std')).reset_index()
            abl_agg.to_csv(os.path.join(METRICS_DIR,'Table2_Ablations.csv'), index=False)
            disp = abl_agg.copy()
            for metric in ['RMSE','R2','DA']:
                m_mean = f'{metric}_mean'; m_std = f'{metric}_std'
                if m_mean in disp:
                    disp[metric] = disp.apply(lambda r: f"{r[m_mean]:.4f}±{(0 if np.isnan(r[m_std]) else r[m_std]):.4f}", axis=1)
            cols = ['exp','model']+[m for m in ['RMSE','R2','DA'] if m in disp]
            disp = disp[cols]
            _save_table_figure(disp, 'Table 2: Ablations @21', os.path.join(FIG_DIR,'table2_ablations.png'))
        else:
            print("⚠️ Ablation subset empty for Table 2")
    # Slices from EXP-A3 predictions
    regimes_df = pd.DataFrame(); earnings_df = pd.DataFrame()
    if exp_a3_files:
        regimes_df, earnings_df = compute_slices_from_predictions(exp_a3_files, getattr(data_loader, 'raw_data', {}).get('events', {}))
        if not regimes_df.empty:
            regimes_df.to_csv(os.path.join(SLICES_DIR, 'EXP-A3_regimes.csv'), index=False)
            _plot_regime_heatmap(regimes_df, os.path.join(FIG_DIR,'fig2_regime_rmse_heatmap.png'))
        if not earnings_df.empty:
            earnings_df.to_csv(os.path.join(SLICES_DIR, 'EXP-A3_earnings.csv'), index=False)
    # Optional Fig 3 equity curve
    equity_df = _compute_equity_curve(exp_a3_files, ['XGBoost','TFT'])
    if not equity_df.empty:
        _plot_equity_curve(equity_df, os.path.join(FIG_DIR,'fig3_equity_curve.png'))
    # Metadata log
    try:
        meta = {
            'timestamp': datetime.utcnow().isoformat(),
            'universe': list(universe),
            'date_contract': {'train': [TRAIN_START, TRAIN_END], 'val': [VAL_START, VAL_END], 'test': [TEST_START, TEST_END]},
            'horizons': HORIZONS,
            'lookback': LOOKBACK,
            'predict_len': PREDICT_LEN,
            'seeds': SEEDS,
            'feature_schema_example': [c for c in train_df.columns if not c.startswith('target_')][:50],
            'baseline_params': baseline_params if 'baseline_params' in locals() else {},
            'package_versions': {
                'python': sys.version,
                'pandas': pd.__version__,
                'numpy': np.__version__,
                'torch': torch.__version__,
            }
        }
        with open(os.path.join(ARTIFACT_DIR,'experiment_log.json'),'w') as f:
            json.dump(meta, f, indent=2)
    except Exception as e:
        print(f"⚠️ Failed to write experiment_log.json: {e}")
    print(f"Artifacts saved under {ARTIFACT_DIR}")


if __name__ == '__main__':
    from dotenv import load_dotenv
    load_dotenv()
    
    # Parse command line arguments
    args = parse_arguments()
    
    try:
        # Validate date ranges
        validate_date_ranges(args)
        
        # Update global configuration
        update_global_config(args)
        
        # Load universe
        universe = load_universe_from_args(args)
        
        # Update the base_config in run_pipeline to use the loaded universe
        # We'll need to pass this to the function
        print("\n🚀 Starting pipeline with configuration:")
        print(f"📅 Train: {TRAIN_START} → {TRAIN_END}")
        print(f"📅 Val:   {VAL_START} → {VAL_END}")
        print(f"📅 Test:  {TEST_START} → {TEST_END}")
        print(f"🎯 Horizons: {HORIZONS}")
        print(f"🎲 Seeds: {SEEDS}")
        print(f"📈 Universe: {len(universe)} symbols")
        print(f"📁 Output: {ARTIFACT_DIR}")
        print()
        
        # Run the pipeline with the custom universe
        run_pipeline(custom_universe=universe)
        
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)
