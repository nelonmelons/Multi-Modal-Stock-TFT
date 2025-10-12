# Benchmarking Transformers and Baselines for Multi-Horizon Stock Return Prediction with Technical and Earnings Features

Acccepted to **IEEE ICITEE 2025 - JSCI** - 17th International Conference on Information Technology and Electrical Engineering - (The Joint Symposium on Computational Intelligence)

This repository contains the code for our research comparing different neural architectures (GRU, LSTM, Transformer, TFT) against classical ML baselines for stock return prediction. We focus on multi-horizon forecasting using technical indicators and earnings data.

## Getting Started

The main code is in `comparison/`. Two key entry points:

- `run_all_models.py` - runs the full experimental pipeline
- `main.py` - just builds the data module for testing

Key files:

- `models.py` - all the neural network implementations
- `train.py` - training loops
- `evaluation.py` - metrics and analysis
- `cache_manager.py` - handles data caching

## Data Setup

We use a clean train/validation/test split to avoid look-ahead bias:

- Training: 2016-2019
- Validation: 2020
- Testing: 2021-2024

The dataset includes DOW 30 stocks with:

- Price data from Yahoo Finance
- Technical indicators (RSI, MACD, etc.)
- Earnings announcements
- News sentiment embeddings
- FRED economic indicators

Prediction horizons: 1, 5, and 21 days ahead.

## Models

**Baselines**: Ridge regression, Random Forest, XGBoost  
**Neural networks**: GRU, LSTM, Transformer, Temporal Fusion Transformer

All models are tuned on the validation set and evaluated on unseen test data.

## Experiments

We run two main experiment sets:

**EXP-A**: Compare all models across different prediction horizons  
**EXP-B**: Ablation study on input features (technical only vs. technical + earnings vs. all features)

Results are saved to `comparison/artifacts/` with:

- Prediction files (`.parquet`)
- Metrics summaries (`.csv`)
- Visualizations (`.png`)
- Detailed analysis by market regime

## Setup

1. Clone and navigate to comparison folder:

```bash
cd comparison
```

2. Create virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

3. Install requirements:

```powershell
pip install -r requirements.txt
```

4. Set up API keys in `.env` file:

```
NEWS_API_KEY=your_key_here
FRED_API_KEY=your_key_here
API_NINJAS_KEY=your_key_here
```

## Running Experiments

The pipeline supports extensive CLI customization. Get help with:

```powershell
python run_all_models.py --help
```

### Basic Usage

Run with defaults (DOW 30, 2016-2024 data, all models):

```powershell
python run_all_models.py
```

Test data loading only:

```powershell
python main.py
```

### Customizing Date Ranges

```powershell
# Custom training/test periods
python run_all_models.py --train-start 2018-01-01 --train-end 2021-12-31 --test-start 2022-01-01 --test-end 2024-12-31
```

### Customizing Stock Universe

```powershell
# Tech stocks only
python run_all_models.py --universe AAPL,MSFT,GOOGL,TSLA,NVDA

# From file (one symbol per line)
python run_all_models.py --universe-file my_stocks.txt

# Use preset universe
python run_all_models.py --universe-preset SP500
```

### Customizing Models and Experiments

```powershell
# Run specific models only
python run_all_models.py --models XGBoost,TFT,LSTM

# Run only main comparison experiments (not ablations)
python run_all_models.py --experiments A

# Custom prediction horizons and seeds
python run_all_models.py --horizons 1,5,10,21 --seeds 42,43,44,45,46
```

### Model Configuration

```powershell
# Adjust model parameters
python run_all_models.py --lookback 120 --predict-len 30

# Custom output directory
python run_all_models.py --output-dir my_results
```

### Available CLI Options

- **Date ranges**: `--train-start`, `--train-end`, `--val-start`, `--val-end`, `--test-start`, `--test-end`
- **Stock universe**: `--universe`, `--universe-file`, `--universe-preset` (DOW30/SP500/NASDAQ100)
- **Model selection**: `--models` (Ridge,XGBoost,RandomForest,GRU,LSTM,TFT)
- **Experiments**: `--experiments` (A=main comparison, B=ablations, ALL=both)
- **Prediction setup**: `--horizons`, `--seeds`, `--lookback`, `--predict-len`
- **Output**: `--output-dir`

## Results

The code generates all tables and figures from our paper. Key metrics include RMSE, R², and directional accuracy across different market conditions and prediction horizons.

## Notes

- First run will take time to download and cache data
- TA-Lib may need special installation on Windows
- All data is cached in `comparison/cache/` for faster subsequent runs
