# 📈 Stock TFT (Temporal Fusion Transformer)

Multi-modal stock prediction using Temporal Fusion Transformers with news sentiment analysis.

## Quick Start

```bash
cd Hayson/TFT
python unified_tft_pipeline.py --epochs 5 --symbols AAPL,MSFT
```

## Features

- **Real TFT Model**: Multi-modal features (stock, news, economic, technical)
- **Trading Strategies**: Kelly criterion, DCA, buy-and-hold
- **Out-of-Sample Validation**: Temporal and symbol-based validation
- **Visualization**: OHLC plots, trading signals, portfolio analytics
- **Performance Metrics**: Sharpe ratio, max drawdown, win rate

## Usage

```bash
# Basic training
python unified_tft_pipeline.py --epochs 5 --symbols AAPL,MSFT

# Out-of-sample validation
python unified_tft_pipeline.py --out-of-sample temporal --test-symbol GOOGL

# Clear cache
python unified_tft_pipeline.py --clear-cache
```

## Structure

```
Hayson/TFT/          # Main implementation
├── unified_tft_pipeline.py    # Complete pipeline
├── train_baseline_tft.py      # Baseline training
└── train_full_pipeline.py     # Full multi-modal training

Code/
├── TFT/             # Core TFT modules
├── Testing/         # Experimental code
└── Data/            # Data processing
```
