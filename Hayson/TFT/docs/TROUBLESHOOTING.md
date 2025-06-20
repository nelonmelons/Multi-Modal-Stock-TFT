# 🔧 Troubleshooting Guide - TFT Data Pipeline

## 🚨 Common Issues & Solutions

### 1. Data Issues

#### ❌ "Insufficient data" Error
```
ValueError: Insufficient data: X samples. Need at least Y samples.
```

**Causes:**
- Not enough historical data for encoder/prediction length
- Symbol has limited trading history
- Date range too short

**Solutions:**
```python
# Option 1: Reduce sequence lengths
loader, module = get_data_loader_with_module(
    symbols=['AAPL'],
    start='2024-01-01',
    end='2024-01-31',
    encoder_len=5,      # Reduced from 60
    predict_len=2,      # Reduced from 5
    batch_size=4
)

# Option 2: Extend date range
loader, module = get_data_loader_with_module(
    symbols=['AAPL'],
    start='2023-01-01',  # Extended range
    end='2024-12-31',
    encoder_len=60,
    predict_len=5,
    batch_size=32
)

# Option 3: Use adaptive mode (automatically adjusts)
# The system automatically tries smaller parameters for limited data
```

#### ❌ "No data returned" Warning
```
Warning: Empty DataFrame returned for symbol X
```

**Causes:**
- Invalid symbol ticker
- Symbol delisted or not traded in date range
- yfinance API issues

**Solutions:**
```python
# Verify symbol exists
import yfinance as yf
ticker = yf.Ticker('AAPL')
info = ticker.info
print(info.get('longName', 'Not found'))

# Use major symbols for testing
symbols = ['AAPL', 'GOOGL', 'MSFT']  # Known to work

# Check date range validity (avoid weekends/holidays)
start = '2024-01-02'  # Tuesday
end = '2024-01-31'
```

---

### 2. API Issues

#### ❌ "No API key provided" Warning
```
⚠️ No API-Ninjas key provided, skipping earnings calendar
```

**Impact:** Reduced features but system continues working
**Solution:**
```bash
# Add to .env file
echo "API_NINJAS_KEY=your_key_here" >> .env
echo "NEWS_API_KEY=your_key_here" >> .env
echo "FRED_API_KEY=your_key_here" >> .env
```

#### ❌ NewsAPI "older than 30 days" Warning
```
📅 Warning: Requested dates older than NewsAPI free tier limit
```

**Expected behavior:** NewsAPI free tier only allows recent news
**Solutions:**
```python
# For historical training: Expected behavior, zero embeddings used
# For recent data: Use dates within last 30 days
from datetime import datetime, timedelta

recent_start = (datetime.now() - timedelta(days=25)).strftime('%Y-%m-%d')
recent_end = datetime.now().strftime('%Y-%m-%d')

loader, module = get_data_loader_with_module(
    symbols=['AAPL'],
    start=recent_start,
    end=recent_end,
    encoder_len=20,
    predict_len=3,
    batch_size=8
)
```

#### ❌ FRED API Rate Limit
```
Error fetching FRED data: rate limit exceeded
```

**Solutions:**
```python
import time

# Add delays between requests (already built-in)
# Or get free FRED API key for higher limits
# https://fred.stlouisfed.org/docs/api/api_key.html
```

---

### 3. Memory Issues

#### ❌ "Out of memory" Error
```
RuntimeError: CUDA out of memory
```

**Solutions:**
```python
# Reduce batch size
batch_size=8  # Instead of 32

# Reduce sequence lengths
encoder_len=30  # Instead of 60
predict_len=3   # Instead of 5

# Use fewer symbols
symbols=['AAPL']  # Instead of multiple symbols

# Use CPU instead of GPU for development
trainer = pl.Trainer(gpus=0)  # Force CPU
```

#### ❌ "Too many features" Warning
```
Warning: Large feature matrix (X MB)
```

**Solutions:**
```python
# Feature selection example (custom implementation)
def create_reduced_dataloader(symbols, start, end):
    # Get full dataloader first
    loader, module = get_data_loader_with_module(symbols, start, end, 10, 3, 4)
    
    # Remove news features if needed
    df = module.feature_df.copy()
    news_cols = [col for col in df.columns if col.startswith('emb_')]
    df_reduced = df.drop(columns=news_cols[:500])  # Keep only 268 news features
    
    # Create new module with reduced features
    from data.datamodule import TFTDataModule
    new_module = TFTDataModule(df_reduced, 10, 3, 4)
    new_module.setup()
    
    return new_module.train_dataloader(), new_module
```

---

### 4. Model Integration Issues

#### ❌ pytorch-forecasting Compatibility
```
AttributeError: 'TimeSeriesDataSet' object has no attribute 'X'
```

**Cause:** Version mismatch or incorrect dataset creation
**Solutions:**
```bash
# Update pytorch-forecasting
pip install pytorch-forecasting --upgrade

# Check versions
pip show pytorch-forecasting torch pytorch-lightning
```

#### ❌ TFT Model Creation Fails
```
ValueError: Feature X not found in dataset
```

**Solutions:**
```python
# Verify dataset parameters
params = module.get_dataset_parameters()
print("Available features:")
for key, features in params.items():
    if isinstance(features, list):
        print(f"  {key}: {len(features)} features")

# Use exact parameter names from dataset
tft = TemporalFusionTransformer.from_dataset(
    module.train_dataset,
    # Don't specify feature names manually, let it auto-detect
    hidden_size=64,
    attention_head_size=2
)
```

#### ❌ "No validation data" Error
```
❌ No validation data available
```

**Solutions:**
```python
# The system automatically handles this with fallback validation
# If you still get errors, manually increase data:

# Option 1: Longer date range
end='2024-12-31'  # More data for validation split

# Option 2: Smaller validation split
# Modify batch_size to ensure minimum samples
batch_size=4      # Smaller batches for limited data
```

---

### 5. Environment Setup Issues

#### ❌ Missing Dependencies
```
ModuleNotFoundError: No module named 'pytorch_forecasting'
```

**Solutions:**
```bash
# Install all dependencies
pip install -r requirements.txt

# If requirements.txt missing, install manually:
pip install torch pytorch-lightning pytorch-forecasting
pip install yfinance pandas numpy transformers requests fredapi
pip install python-dotenv pandas-ta
```

#### ❌ .env File Not Found
```
Could not load .env file
```

**Solutions:**
```bash
# Create .env file
cp .env.example .env

# Or create manually
touch .env
echo "NEWS_API_KEY=your_key" >> .env
echo "FRED_API_KEY=your_key" >> .env
echo "API_NINJAS_KEY=your_key" >> .env
```

#### ❌ Import Errors
```
ImportError: cannot import name 'get_data_loader_with_module'
```

**Solutions:**
```python
# Verify you're in the correct directory
import os
print(os.getcwd())  # Should be .../TFT-mm

# Check Python path
import sys
sys.path.append('.')

# Try explicit import
from data.interface import get_data_loader_with_module
```

---

### 6. Performance Issues

#### ❌ Slow Data Loading
```
Data fetching taking >60 seconds
```

**Causes & Solutions:**
```python
# Too many symbols
symbols = ['AAPL', 'GOOGL']  # Instead of 10+ symbols

# Large date range
start = '2024-01-01'  # Instead of multi-year ranges
end = '2024-03-31'

# API delays
# Built-in rate limiting protects against API bans
# Consider upgrading API plans for faster access
```

#### ❌ BERT Model Download
```
Downloading FinBERT model (400MB)...
```

**Expected behavior on first use:**
```python
# Pre-download to avoid delays during training
python -c "
from transformers import AutoTokenizer, AutoModelForSequenceClassification
model_name = 'yiyanghkust/finbert-tone'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
print('✅ FinBERT model cached successfully')
"
```

---

### 7. Debugging Tools

#### Check Data Pipeline Status
```python
# Run full validation
python main.py --validate

# Check individual components
from data import fetch_stock_data, fetch_events_data

# Test stock data
stock_df = fetch_stock_data(['AAPL'], '2024-01-01', '2024-01-31')
print(f"Stock data: {len(stock_df)} rows")

# Test events data
events = fetch_events_data(['AAPL'], '2024-01-01', '2024-01-31')
print(f"Events data: {len(events)} symbols")
```

#### Tensor Structure Analysis
```python
# Get detailed tensor report
loader, module = get_data_loader_with_module(['AAPL'], '2024-01-01', '2024-01-31', 10, 3, 4)
module.print_tensor_report()

# Examine batch structure
batch = next(iter(loader))
print("Batch structure:")
if isinstance(batch, tuple):
    x, y = batch
    print(f"  x type: {type(x)}")
    print(f"  y type: {type(y)}")
    if hasattr(x, 'keys'):
        print(f"  x keys: {list(x.keys())}")
```

#### Memory Usage Check
```python
import torch
import psutil

# Check system memory
print(f"System RAM: {psutil.virtual_memory().available / 1e9:.1f} GB")

# Check GPU memory (if available)
if torch.cuda.is_available():
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
```

---

### 8. Getting Help

#### Debug Information to Collect
```python
# Version information
import torch, pytorch_lightning, pytorch_forecasting
print(f"PyTorch: {torch.__version__}")
print(f"Lightning: {pytorch_lightning.__version__}")
print(f"Forecasting: {pytorch_forecasting.__version__}")

# System information
import platform
print(f"Python: {platform.python_version()}")
print(f"OS: {platform.system()} {platform.release()}")

# Data pipeline info
loader, module = get_data_loader_with_module(['AAPL'], '2024-01-01', '2024-01-31', 5, 2, 2)
print(f"Feature matrix shape: {module.feature_df.shape}")
print(f"Symbols: {module.feature_df['symbol'].unique()}")
```

#### Still Having Issues?

1. **Check logs carefully** - Error messages contain specific guidance
2. **Reduce complexity** - Start with minimal parameters and scale up
3. **Verify environment** - Ensure all dependencies are correctly installed
4. **Test components individually** - Isolate the failing component
5. **Check API status** - Verify external APIs are accessible

---

## ✅ Success Indicators

When everything is working correctly, you should see:

```
✓ Loaded environment variables from .env file
✅ TFT data pipeline completed successfully!
✅ DataLoader created successfully!
Number of batches: X
✅ Model created with X parameters
✅ Training successful!
✅ Prediction successful!
```

**Happy modeling!** 🚀
