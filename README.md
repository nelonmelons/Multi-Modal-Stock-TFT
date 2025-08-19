# 📈 Stock TFT: Temporal Fusion Transformer for Stock Price Prediction

## Multi-Modal Stock Model Comparison (comparison/)

This repo’s active codebase lives in `comparison/`. It implements a leakage-safe, multi-horizon stock prediction pipeline with caching, classical baselines, and deep models.

Checklist for this update

- Summarize comparison directory and entry points
- Document data pipeline, models, experiments, and artifacts
- Provide Windows-friendly setup and run steps

### Overview

- Entry points:
  - `comparison/run_all_models.py`: end-to-end experiments (data → train → eval → artifacts)
  - `comparison/main.py`: builds the leakage-free data module only
- Core modules:
  - `comparison/models.py`: LSTM, GRU, Transformer, TFT implementations
  - `comparison/train.py`: training loops for generic and TFT models
  - `comparison/evaluation.py`: multi-horizon metrics (RMSE, MAE, R2, DA) and detailed outputs
  - `comparison/cache_manager.py`: unified cache for stock, news, FRED, TA, events, and features
  - `comparison/dataModule/`: data loaders, feature building, symbol universe, etc. (imported in code)

### Data pipeline (leakage-safe)

- Fixed windows: Train 2016-01-01→2019-12-31, Val 2020-01-01→2020-12-31, Test 2021-01-01→2024-12-31
- Horizons evaluated: [1, 5, 21]; lookback (encoder_len) = 60; predict_len = 21
- Universe: `src.universe.DOW30_2018` (imported)
- Sources and features (via `comparison/main.py`):
  - Prices (yfinance), corporate events, news embeddings, FRED, technicals
  - Features built with strict temporal splits; no future peeking
- Caching: `comparison/cache/` auto-populates with `.pkl` and `.json` + metadata

### Models

- Baselines (from `src.sk_baselines`): Ridge, RandomForest, XGBoost (tuned on 2020 Val for RMSE@21)
- Deep models (`comparison/models.py`): GRUModel, LSTMModel, TransformerModel, TFT
- Device selection via `model.tft_model.setup_device()`

### Experiments and artifacts

- Orchestrated in `comparison/run_all_models.py` across seeds [42, 43, 44]
- Experiment sets:
  - EXP-A1/A2/A3: Main comparisons across horizons [1,5,21] with models: Ridge, XGBoost, RandomForest, GRU, LSTM, TFT
  - EXP-B1/B2/B3: Ablations by modality (Tech, Tech+Earnings, Tech+Earnings+News) for XGBoost/TFT @21
- Outputs under `comparison/artifacts/`:
  - `predictions/EXP-<ID>_<Model>_<Seed>.parquet` (columns: date, ticker, horizon, y_true, y_pred)
  - `metrics/*.csv` (per-experiment metrics and Table 1/2 aggregations)
  - `figures/*.png` (DA-by-horizon bar, regime heatmap, equity curve, table images)
  - `slices/*.csv` (regime and earnings-window slices for EXP-A3)
  - `experiment_log.json` (universe, contract, package versions, params)

### Setup (Windows PowerShell)

1. Create and activate a venv

```powershell
cd comparison
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Install dependencies

```powershell
pip install -r requirements.txt
```

3. Configure API keys (either `.env` in `comparison/` or session env vars)

```powershell
# .env is supported by run_all_models.py
# For session-only vars:
$env:NEWS_API_KEY = "<key>"
$env:FRED_API_KEY = "<key>"
$env:API_NINJAS_KEY = "<key>"
```

Notes

- `TA-Lib` and related packages may require platform-specific wheels on Windows.
- Data is cached in `comparison/cache/`; delete files there to refresh.

### Run

- Full experiment suite (creates predictions/metrics/slices/figures under `comparison/artifacts/`):

```powershell
python run_all_models.py
```

- Data-only smoke test (builds features and loaders, uses cache):

```powershell
python main.py
```

### Repository layout (focused on comparison/)

```
comparison/
    run_all_models.py     # orchestrates experiments and artifact generation
    main.py               # builds leakage-free data module and features
    models.py             # GRU, LSTM, Transformer, TFT
    train.py              # training loops for deep models
    evaluation.py         # metrics and detailed predictions
    cache_manager.py      # unified caching layer
    requirements.txt      # Python dependencies for comparison pipeline
    artifacts/            # outputs: metrics, predictions, figures, slices, log
    cache/                # cached raw and derived data (.pkl/.json + meta)
    dataModule/           # loaders, feature building, adapters (imports used)
    src/                  # universe and baseline helpers (imports used)
```

### Minimal customization

- Edit experiment lists, seeds, and horizons in `comparison/run_all_models.py`
- Adjust lookback/predict_len and batch size in the base config block
