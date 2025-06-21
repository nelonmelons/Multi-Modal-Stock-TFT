# 🚀 Quick Start Guide - TFT Financial Data Pipeline

## ⚡ 5-Minute Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure API Keys (Optional but Recommended)

```bash
# Copy environment template
cp .env.example .env

# Edit .env with your API keys:
NEWS_API_KEY=your_newsapi_key          # Get free at: https://newsapi.org/
FRED_API_KEY=your_fred_key             # Get free at: https://fred.stlouisfed.org/
API_NINJAS_KEY=your_ninjas_key         # Get free at: https://api.api-ninjas.com/
```

### 3. Basic Usage

```python
from data import get_data_loader_with_module

# Get TFT-ready DataLoader with 860 features
loader, module = get_data_loader_with_module(
    symbols=['AAPL', 'GOOGL'],
    start='2024-01-01',
    end='2024-12-31',
    encoder_len=60,     # Look back 60 days
    predict_len=5,      # Predict 5 days ahead
    batch_size=32
)

print(f"✅ DataLoader created with {len(loader)} batches")
module.print_tensor_report()  # See detailed feature breakdown
```

### 4. Train TFT Model

```python
from pytorch_forecasting import TemporalFusionTransformer
import pytorch_lightning as pl

# Create model from your data
tft = TemporalFusionTransformer.from_dataset(
    module.train_dataset,
    hidden_size=128,
    attention_head_size=4,
    dropout=0.1
)

# Train with PyTorch Lightning
trainer = pl.Trainer(max_epochs=10)
trainer.fit(tft, loader, module.val_dataloader())

print("🎉 Model training complete!")
```

## 📊 What You Get

### Data Sources (6)

- **📈 Stock Data**: OHLCV + bid/ask from yfinance
- **📅 Corporate Events**: Earnings, splits, dividends
- **📰 News Intelligence**: BERT embeddings + sentiment
- **🏛️ Economic Data**: 8 FRED indicators
- **🔧 Technical Analysis**: 22 indicators
- **🏢 Company Data**: Sector, market cap

### Features (860)

- **Static Features**: 3 (symbol, sector, market_cap)
- **Known Future**: 28 (calendar, economic, events)
- **Past Observed**: 829 (OHLCV, technical, news, etc.)

## 🎯 Key Commands

```bash
# Quick validation
python example.py --validate

# Full example with tensor report
python example.py

# Integration tests
python example.py --test
```

## 💡 Tips for Success

### For Small Datasets

```python
# Use smaller parameters for limited data
loader, module = get_data_loader_with_module(
    symbols=['AAPL'],
    start='2024-12-01',
    end='2024-12-31',
    encoder_len=10,     # Smaller encoder
    predict_len=3,      # Smaller prediction
    batch_size=4        # Smaller batch
)
```

### For Production

```python
# Use all API keys for full feature set
loader, module = get_data_loader_with_module(
    symbols=['AAPL', 'GOOGL', 'MSFT', 'AMZN'],
    start='2023-01-01',
    end='2024-12-31',
    encoder_len=60,
    predict_len=5,
    batch_size=64,
    news_api_key=os.getenv('NEWS_API_KEY'),
    fred_api_key=os.getenv('FRED_API_KEY'),
    api_ninjas_key=os.getenv('API_NINJAS_KEY')
)
```

## 🔧 Common Issues & Quick Fixes

### Issue: "Insufficient data"

```python
# Solution: Use smaller encoder/prediction lengths
encoder_len=10, predict_len=2
```

### Issue: No news data

```python
# Expected for historical dates (NewsAPI limit)
# Recent dates will have news embeddings
```

### Issue: Missing API keys

```python
# System works without keys but with reduced features
# Add keys for full 860-feature experience
```

## 🚀 Next Steps

1. **Scale Up**: Add more symbols and longer time periods
2. **Hyperparameter Tuning**: Optimize TFT architecture
3. **Production**: Deploy with real-time data feeds
4. **Advanced Features**: Add custom technical indicators

**Ready to build your TFT model!** 🎯
