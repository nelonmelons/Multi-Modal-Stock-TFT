# 📚 API Reference - TFT Data Pipeline

## 🚀 Main Interface Functions

### `get_data_loader_with_module()`

The primary entry point for creating TFT-ready DataLoaders.

```python
from data import get_data_loader_with_module

loader, module = get_data_loader_with_module(
    symbols=['AAPL', 'GOOGL'],
    start='2024-01-01',
    end='2024-12-31',
    encoder_len=60,
    predict_len=5,
    batch_size=32,
    news_api_key=None,
    fred_api_key=None,
    api_ninjas_key=None
)
```

**Parameters:**
- `symbols` (List[str]): Stock symbols to fetch data for
- `start` (str): Start date in 'YYYY-MM-DD' format
- `end` (str): End date in 'YYYY-MM-DD' format
- `encoder_len` (int): Historical sequence length for TFT encoder
- `predict_len` (int): Future sequence length for TFT decoder
- `batch_size` (int): Batch size for DataLoader
- `news_api_key` (Optional[str]): NewsAPI key for news embeddings
- `fred_api_key` (Optional[str]): FRED API key for economic data
- `api_ninjas_key` (Optional[str]): API-Ninjas key for earnings calendar

**Returns:**
- `Tuple[DataLoader, TFTDataModule]`: PyTorch DataLoader and analysis module

**Example:**
```python
# Basic usage without API keys
loader, module = get_data_loader_with_module(
    symbols=['AAPL'],
    start='2024-01-01',
    end='2024-01-31',
    encoder_len=20,
    predict_len=5,
    batch_size=16
)

# Full feature set with API keys
loader, module = get_data_loader_with_module(
    symbols=['AAPL', 'GOOGL', 'MSFT'],
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

---

## 📊 TFTDataModule Class

### Methods

#### `setup()`
Prepares the TimeSeriesDataSet and splits data into train/validation.

```python
module.setup()
```

#### `print_tensor_report()`
Generates comprehensive analysis of tensor structure and features.

```python
module.print_tensor_report()
```

**Output includes:**
- Dataset overview (samples, batches, symbols)
- Time alignment details
- Feature categorization breakdown
- Memory usage estimates
- Data source integration status

#### `get_dataset_parameters()`
Returns parameters needed for TFT model initialization.

```python
params = module.get_dataset_parameters()
```

**Returns:**
```python
{
    "max_encoder_length": int,
    "max_prediction_length": int,
    "static_categoricals": List[str],
    "static_reals": List[str],
    "time_varying_known_categoricals": List[str],
    "time_varying_known_reals": List[str],
    "time_varying_unknown_reals": List[str],
    "target": str,
    "group_ids": List[str]
}
```

#### `train_dataloader()` / `val_dataloader()`
Returns training and validation DataLoaders.

```python
train_loader = module.train_dataloader()
val_loader = module.val_dataloader()
```

---

## 🔧 Individual Data Fetching Functions

### Stock Data

```python
from data import fetch_stock_data

stock_df = fetch_stock_data(
    symbols=['AAPL', 'GOOGL'],
    start='2024-01-01',
    end='2024-01-31'
)
```

**Returns:** DataFrame with OHLCV data

### Events Data

```python
from data import fetch_events_data

events = fetch_events_data(
    symbols=['AAPL'],
    start='2024-01-01',
    end='2024-01-31',
    api_ninjas_key='your_key'
)
```

**Returns:** Dictionary with earnings, splits, dividends per symbol

### News Embeddings

```python
from data import fetch_news_embeddings

news_df = fetch_news_embeddings(
    symbols=['AAPL'],
    start='2024-01-01',
    end='2024-01-31',
    api_key='your_newsapi_key'
)
```

**Returns:** DataFrame with 768-dimensional BERT embeddings + sentiment

### Economic Data

```python
from data import fetch_fred_data

fred_df = fetch_fred_data(
    start='2024-01-01',
    end='2024-01-31',
    api_key='your_fred_key'
)
```

**Returns:** DataFrame with 8 economic indicators

### Technical Indicators

```python
from data import compute_technical_indicators

ta_df = compute_technical_indicators(stock_df)
```

**Returns:** DataFrame with 22 technical analysis indicators

---

## 📈 Feature Engineering

### `build_features()`

Combines all data sources into TFT-ready feature matrix.

```python
from data import build_features

feature_df = build_features(
    stock_df=stock_df,
    events=events_data,
    news_df=news_df,
    ta_df=ta_df,
    fred_df=fred_df,
    encoder_len=60,
    predict_len=5
)
```

**Returns:** DataFrame with 860 features ready for TFT

---

## 🎯 Feature Categories

### Static Features (Entity-Level)
- `symbol`: Categorical stock identifier
- `sector`: Industry sector
- `market_cap`: Market capitalization

### Known Future Features (Available at Prediction Time)
- **Categoricals**: `is_earnings_day`, `is_split_day`, `is_dividend_day`, `is_holiday`, `is_weekend`
- **Calendar**: `day_of_week`, `month`, `quarter`, `day_of_month`
- **Economic**: `cpi`, `fedfunds`, `unrate`, `t10y2y`, `gdp`, `vix`, `dxy`, `oil`
- **Events**: `days_to_next_earnings`, `days_since_earnings`, etc.
- **EPS/Revenue**: `eps_estimate`, `eps_actual`, `revenue_estimate`, `revenue_actual`

### Past Observed Features (Unknown Future)
- **OHLCV**: `open`, `high`, `low`, `close`, `volume`, `bid`, `ask`
- **Technical**: 22 indicators (SMA, EMA, RSI, MACD, etc.)
- **News**: 768 BERT embedding dimensions + `sentiment_score`

---

## ⚙️ Configuration Parameters

### Environment Variables
```bash
NEWS_API_KEY=your_newsapi_key
FRED_API_KEY=your_fred_key
API_NINJAS_KEY=your_ninjas_key
```

### Recommended Parameters

**For Development/Testing:**
```python
encoder_len=10-20
predict_len=3-5
batch_size=4-16
symbols=1-2 symbols
date_range=1-3 months
```

**For Production:**
```python
encoder_len=60-120
predict_len=5-10
batch_size=32-128
symbols=5-20 symbols
date_range=1-2 years
```

---

## 🚨 Error Handling

### Common Exceptions

#### `ValueError: Insufficient data`
**Cause:** Not enough samples for encoder/prediction length  
**Solution:** Reduce `encoder_len` or `predict_len`, or increase date range

#### `API Rate Limit Exceeded`
**Cause:** Too many API requests  
**Solution:** Add delays between requests or reduce symbol count

#### `Missing API Key Warning`
**Cause:** API key not provided  
**Solution:** Add to .env file or pass as parameter (system continues with reduced features)

### Validation Methods

```python
# Check data requirements
module._validate_data_requirements()

# Test sample batch loading
module._test_sample_batch()

# Verify feature categorization
params = module.get_dataset_parameters()
```

---

## 📊 Data Structure Examples

### Batch Structure
```python
# DataLoader returns tuples: (x, y)
x, y = next(iter(loader))

# x contains encoder/decoder inputs:
x['encoder_cat']      # [batch_size, encoder_len, n_cat_features]
x['encoder_cont']     # [batch_size, encoder_len, n_cont_features]
x['decoder_cat']      # [batch_size, decoder_len, n_cat_features]
x['decoder_cont']     # [batch_size, decoder_len, n_cont_features]

# y contains targets:
y                     # [batch_size, decoder_len]
```

### Feature DataFrame Structure
```python
# Columns include:
feature_df.columns = [
    'symbol', 'date', 'time_idx',           # Identifiers
    'open', 'high', 'low', 'close', 'volume',  # OHLCV
    'sma_10', 'ema_20', 'rsi_14', ...,     # Technical indicators
    'emb_0', 'emb_1', ..., 'emb_767',      # News embeddings
    'sentiment_score',                      # News sentiment
    'cpi', 'fedfunds', 'unrate', ...,      # Economic indicators
    'is_earnings_day', 'is_holiday', ...,  # Event flags
    'target'                               # Prediction target
]
```

---

## 🎯 Integration with pytorch-forecasting

### Model Creation
```python
from pytorch_forecasting import TemporalFusionTransformer

tft = TemporalFusionTransformer.from_dataset(
    module.train_dataset,
    hidden_size=256,
    attention_head_size=4,
    dropout=0.1,
    hidden_continuous_size=64,
    loss=QuantileLoss()
)
```

### Training
```python
import pytorch_lightning as pl

trainer = pl.Trainer(
    max_epochs=50,
    gpus=1 if torch.cuda.is_available() else 0
)

trainer.fit(tft, loader, module.val_dataloader())
```

### Prediction
```python
predictions = tft.predict(val_loader, return_y=True)
```

---

*Complete API documentation for TFT Financial Data Pipeline*
