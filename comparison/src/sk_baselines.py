"""
Scikit-learn/XGBoost tabular baselines for multi-horizon stock return prediction.

- Models: Ridge, RandomForest, XGBoost
- Input: one row per (date, ticker) at time t using engineered features
- Targets: target_{h-1} for each horizon h
- Scaling: StandardScaler fit on Train only
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Optional XGBoost
try:
    from xgboost import XGBRegressor  # type: ignore
    HAS_XGB = True
except Exception:  # ImportError or runtime issues
    HAS_XGB = False


@dataclass
class TabularResult:
    metrics: Dict[str, Dict[str, float]]
    predictions: pd.DataFrame


def _make_model(name: str, params: Optional[Dict]=None, random_state: Optional[int]=None):
    params = params or {}
    if name.lower() == 'ridge':
        # Ridge is deterministic, but we pass random_state for consistency
        model = Ridge(**params)
    elif name.lower() == 'randomforest':
        # Add random_state for reproducible randomness across seeds
        if 'random_state' not in params and random_state is not None:
            params['random_state'] = random_state
        model = RandomForestRegressor(**params)
    elif name.lower() == 'xgboost':
        if not HAS_XGB:
            raise RuntimeError("xgboost is not installed")
        # Sensible defaults; caller may override
        default = dict(n_estimators=400, learning_rate=0.05, max_depth=4, subsample=0.8, colsample_bytree=0.8, reg_lambda=1)
        default.update(params)
        # Add random_state for reproducible randomness across seeds
        if 'random_state' not in default and random_state is not None:
            default['random_state'] = random_state
        model = XGBRegressor(**default, objective='reg:squarederror', n_jobs=4, tree_method='hist')
    else:
        raise ValueError(f"Unknown baseline model: {name}")
    return Pipeline([
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("model", model),
    ])


def fit_predict_baseline(
    model_name: str,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    horizons: List[int],
    feature_cols: Optional[List[str]] = None,
    model_params: Optional[Dict[str, Dict]] = None,
    random_state: Optional[int] = None,
) -> TabularResult:
    """
    Fit a tabular baseline separately for each horizon and produce predictions/metrics.

    Returns
    -------
    TabularResult with metrics per horizon and long-form predictions DataFrame
    """
    assert all(f"target_{h-1}" in train_df.columns for h in horizons), "Missing target columns in train_df"
    assert all(c in train_df.columns for c in ['date','symbol']), "Need date/symbol columns"

    # Feature columns: default to all numeric except targets
    if feature_cols is None:
        num_cols = train_df.select_dtypes(include=[np.number]).columns.tolist()
        target_cols = [f"target_{h-1}" for h in horizons]
        feature_cols = [c for c in num_cols if c not in target_cols]

    model_params = model_params or {}

    preds_long = []
    metrics: Dict[str, Dict[str, float]] = {}

    # Ensure aligned indices
    X_test_base = test_df[feature_cols].values

    for h in horizons:
        y_col = f"target_{h-1}"
        X_train = train_df[feature_cols].values
        y_train = train_df[y_col].values
        y_test = test_df[y_col].values if y_col in test_df.columns else np.full(len(test_df), np.nan)

        pipe = _make_model(model_name, params=model_params.get(model_name, {}), random_state=random_state)
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test_base)

        # Metrics
        rmse = float(np.sqrt(mean_squared_error(y_test, y_pred))) if np.isfinite(y_test).all() else np.nan
        mae = float(mean_absolute_error(y_test, y_pred)) if np.isfinite(y_test).all() else np.nan
        r2 = float(r2_score(y_test, y_pred)) if np.isfinite(y_test).all() else np.nan
        da = float(np.mean(np.sign(y_test) == np.sign(y_pred))) if np.isfinite(y_test).all() else np.nan
        metrics[f"horizon_{h}"] = {"RMSE": rmse, "MAE": mae, "R2": r2, "DA": da}

        # Long-form predictions
        part = pd.DataFrame({
            'date': test_df['date'].values,
            'ticker': test_df['symbol'].astype(str).values,
            'horizon': h,
            'actual': y_test.astype(float),
            'prediction': y_pred.astype(float),
        })
        preds_long.append(part)

    preds_df = pd.concat(preds_long, ignore_index=True) if preds_long else pd.DataFrame()
    return TabularResult(metrics=metrics, predictions=preds_df)
