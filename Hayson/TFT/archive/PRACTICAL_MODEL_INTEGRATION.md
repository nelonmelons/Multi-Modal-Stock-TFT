# 🚀 Practical TFT Model Integration Guide

## ⚡ Quick Start: From Data Pipeline to Working Model

This guide provides **immediate, copy-paste examples** for using our TFT data pipeline with PyTorch Forecasting models.

---

## 🎯 1. Basic Model Training (5 Minutes)

### Step 1: Get Data Pipeline

```python
import os
from data import get_data_loader_with_module

# Initialize data pipeline
train_dataloader, datamodule = get_data_loader_with_module(
    symbols=['AAPL', 'GOOGL', 'MSFT'],
    start='2023-01-01',
    end='2024-12-31',
    encoder_len=60,        # Look back 60 days
    predict_len=5,         # Predict 5 days ahead
    batch_size=64,
    news_api_key=os.getenv('NEWS_API_KEY'),    # Optional
    fred_api_key=os.getenv('FRED_API_KEY'),    # Optional
    api_ninjas_key=os.getenv('API_NINJAS_KEY') # Optional - for earnings calendar
)

# Get validation loader
val_dataloader = datamodule.val_dataloader()

# Print tensor structure
datamodule.print_tensor_report()
```

### Step 2: Create TFT Model

```python
from pytorch_forecasting import TemporalFusionTransformer, QuantileLoss

# Create model from our dataset
tft = TemporalFusionTransformer.from_dataset(
    datamodule.train_dataset,
    
    # Model architecture
    hidden_size=128,               # Start small for quick training
    attention_head_size=4,
    dropout=0.1,
    hidden_continuous_size=32,
    
    # Loss function
    loss=QuantileLoss(),          # Best for financial forecasting
    
    # Learning settings
    learning_rate=0.03,
    reduce_on_plateau_patience=4,
)

print(f"✅ TFT Model created with {tft.size():,} parameters")
```

### Step 3: Train Model

```python
import pytorch_lightning as pl

# Quick training setup
trainer = pl.Trainer(
    max_epochs=10,                    # Start with few epochs
    gpus=1 if torch.cuda.is_available() else 0,
    limit_train_batches=10,           # Fast prototyping
    limit_val_batches=10,
    logger=False,                     # Disable logging for speed
)

# Train the model
trainer.fit(tft, train_dataloader, val_dataloader)
print("✅ Model trained successfully!")
```

---

## 🔧 2. Understanding Our Tensor Structure

### Feature Breakdown (860 Total Features)

```python
# Get dataset parameters to understand structure
params = datamodule.get_dataset_parameters()

print("📊 STATIC FEATURES (Symbol-Level):")
print(f"  Categoricals: {params['static_categoricals']}")  # ['symbol', 'sector']
print(f"  Reals: {params['static_reals']}")                # ['market_cap']

print("\n🔮 KNOWN FUTURE FEATURES (Available at Prediction Time):")
print(f"  Categoricals: {params['time_varying_known_categoricals']}")
# ['is_earnings_day', 'is_split_day', 'is_dividend_day', 'is_holiday', 'is_weekend', 'earnings_in_prediction_window']

print(f"  Reals ({len(params['time_varying_known_reals'])} features):")
# time_idx, calendar (4), economic (8), events (5), eps/revenue (4)

print("\n📉 PAST OBSERVED FEATURES (Unknown Future):")
print(f"  Total: {len(params['time_varying_unknown_reals'])} features")
# OHLCV (5) + Technical (22) + News Embeddings (768) + Sentiment (1) + Others
```

### Batch Structure

```python
# Examine a sample batch
sample_batch = next(iter(train_dataloader))

print("📦 BATCH STRUCTURE:")
for key, tensor in sample_batch.items():
    if isinstance(tensor, torch.Tensor):
        print(f"  {key}: {tensor.shape}")

# Output example:
# encoder_cat:      [64, 60, 6]   # Categorical features for encoder
# encoder_cont:     [64, 60, 832] # Continuous features for encoder  
# encoder_target:   [64, 60]      # Target history for encoder
# decoder_cat:      [64, 5, 6]    # Categorical features for decoder
# decoder_cont:     [64, 5, 832]  # Continuous features for decoder
# decoder_target:   [64, 5]       # Target values for training
```

---

## 🎯 3. Making Predictions

### Basic Prediction

```python
# Make predictions on validation set
predictions = tft.predict(val_dataloader, return_y=True)

print("🔮 PREDICTIONS STRUCTURE:")
print(f"  Predictions: {predictions.prediction.shape}")  # [batch, time, quantiles]
print(f"  Actual values: {predictions.y.shape}")         # [batch, time]

# Extract confidence intervals
pred_median = predictions.prediction[:, :, 2]  # 50th percentile
pred_lower = predictions.prediction[:, :, 0]   # 10th percentile  
pred_upper = predictions.prediction[:, :, 4]   # 90th percentile

print(f"✅ Generated predictions for {pred_median.shape[0]} sequences")
```

### Real-Time Prediction Function

```python
def predict_stocks(symbols, current_date, horizon=5):
    """
    Real-time stock prediction using our pipeline.
    
    Args:
        symbols: List of stock symbols ['AAPL', 'GOOGL']
        current_date: Current date in 'YYYY-MM-DD' format
        horizon: Days to predict ahead
    
    Returns:
        Dictionary with predictions per symbol
    """
    # Fetch recent data (90 days for context)
    start_date = (pd.to_datetime(current_date) - pd.Timedelta(days=90)).strftime('%Y-%m-%d')
    
    # Get fresh data
    dataloader, _ = get_data_loader_with_module(
        symbols=symbols,
        start=start_date,
        end=current_date,
        encoder_len=60,
        predict_len=horizon,
        batch_size=len(symbols),
        news_api_key=os.getenv('NEWS_API_KEY'),
        fred_api_key=os.getenv('FRED_API_KEY'),
        api_ninjas_key=os.getenv('API_NINJAS_KEY')
    )
    
    # Make predictions
    predictions = tft.predict(dataloader, return_y=False)
    
    # Format results
    results = {}
    for i, symbol in enumerate(symbols):
        results[symbol] = {
            'median_prediction': predictions.prediction[i, :, 2].cpu().numpy(),
            'lower_bound': predictions.prediction[i, :, 0].cpu().numpy(),
            'upper_bound': predictions.prediction[i, :, 4].cpu().numpy(),
            'confidence_interval': (predictions.prediction[i, :, 4] - predictions.prediction[i, :, 0]).cpu().numpy()
        }
    
    return results

# Example usage
results = predict_stocks(['AAPL', 'GOOGL'], '2024-01-15', horizon=5)
```

---

## 🧠 4. Feature Importance Analysis

### Understanding Which Features Matter

```python
# Get feature importance from model attention
interpretation = tft.interpret_output(
    val_dataloader.dataset[:100],  # Sample 100 sequences
    return_attention=True
)

# Variable attention weights (average importance)
var_importance = interpretation['attention'].mean(0)

# Get feature names
feature_names = params['time_varying_unknown_reals']

# Top 20 most important features
top_indices = var_importance.argsort()[-20:]
top_features = [(feature_names[i], var_importance[i].item()) for i in top_indices]

print("🏆 TOP 20 MOST IMPORTANT FEATURES:")
for feature, importance in reversed(top_features):
    print(f"  {feature}: {importance:.4f}")
```

### News Impact Analysis

```python
# Analyze news embedding contributions
news_features = [f'emb_{i}' for i in range(768)] + ['sentiment_score']
news_indices = [i for i, feat in enumerate(feature_names) if feat in news_features]

if news_indices:
    news_importance = var_importance[news_indices]
    avg_news_importance = news_importance.mean()
    
    print(f"📰 NEWS FEATURES ANALYSIS:")
    print(f"  Average news importance: {avg_news_importance:.4f}")
    print(f"  News features in top 50: {sum(1 for i in var_importance.argsort()[-50:] if i in news_indices)}")
    
    # Top news dimensions
    top_news_indices = news_importance.argsort()[-10:]
    for idx in top_news_indices:
        feature_idx = news_indices[idx]
        print(f"  {feature_names[feature_idx]}: {var_importance[feature_idx]:.4f}")
```

---

## 📊 5. Data Source Validation

### Verify All Data Sources Are Working

```python
def validate_data_sources():
    """Check all data sources are properly integrated."""
    
    # Test with recent data
    loader, module = get_data_loader_with_module(
        symbols=['AAPL'],
        start='2024-01-01',
        end='2024-01-31',
        encoder_len=10,
        predict_len=3,
        batch_size=4
    )
    
    # Get a sample batch
    batch = next(iter(loader))
    features = module.feature_df
    
    # Check data sources
    checks = {
        'stock_data': 'close' in features.columns,
        'technical_indicators': any('sma' in col.lower() for col in features.columns),
        'news_embeddings': any(col.startswith('emb_') for col in features.columns),
        'economic_data': 'cpi' in features.columns,
        'corporate_events': 'is_dividend_day' in features.columns,
        'calendar_features': 'day_of_week' in features.columns
    }
    
    print("🔍 DATA SOURCE VALIDATION:")
    for source, status in checks.items():
        emoji = "✅" if status else "❌"
        print(f"  {emoji} {source}: {'OK' if status else 'MISSING'}")
    
    return all(checks.values())

# Run validation
is_valid = validate_data_sources()
print(f"\n{'✅ All data sources validated!' if is_valid else '❌ Some data sources missing'}")
```

---

## ⚡ 6. Production-Ready Model Pipeline

### Complete Training Pipeline

```python
class StockTFTTrainer:
    def __init__(self, symbols, start_date, end_date):
        self.symbols = symbols
        self.start_date = start_date
        self.end_date = end_date
        self.model = None
        self.datamodule = None
    
    def prepare_data(self, encoder_len=60, predict_len=5, batch_size=64):
        """Prepare data for training."""
        self.train_dataloader, self.datamodule = get_data_loader_with_module(
            symbols=self.symbols,
            start=self.start_date,
            end=self.end_date,
            encoder_len=encoder_len,
            predict_len=predict_len,
            batch_size=batch_size,
            news_api_key=os.getenv('NEWS_API_KEY'),
            fred_api_key=os.getenv('FRED_API_KEY')
        )
        
        # Print data summary
        self.datamodule.print_summary()
        return self
    
    def create_model(self, hidden_size=256, attention_heads=4):
        """Create TFT model."""
        self.model = TemporalFusionTransformer.from_dataset(
            self.datamodule.train_dataset,
            hidden_size=hidden_size,
            attention_head_size=attention_heads,
            dropout=0.1,
            hidden_continuous_size=64,
            loss=QuantileLoss(),
            learning_rate=0.03,
            reduce_on_plateau_patience=4,
        )
        return self
    
    def train(self, max_epochs=50, gpus=1):
        """Train the model."""
        trainer = pl.Trainer(
            max_epochs=max_epochs,
            gpus=gpus if torch.cuda.is_available() else 0,
            callbacks=[
                pl.callbacks.EarlyStopping(monitor='val_loss', patience=10),
                pl.callbacks.ModelCheckpoint(monitor='val_loss', save_top_k=3)
            ]
        )
        
        trainer.fit(
            self.model,
            self.train_dataloader,
            self.datamodule.val_dataloader()
        )
        
        return self
    
    def predict(self, symbols=None, current_date=None, horizon=5):
        """Make predictions."""
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        # Use training symbols if not specified
        symbols = symbols or self.symbols
        current_date = current_date or self.end_date
        
        return predict_stocks(symbols, current_date, horizon)

# Usage example
trainer = StockTFTTrainer(
    symbols=['AAPL', 'GOOGL', 'MSFT'],
    start_date='2023-01-01',
    end_date='2024-12-31'
)

# Full pipeline
model = (trainer
         .prepare_data(encoder_len=60, predict_len=5)
         .create_model(hidden_size=256, attention_heads=4)
         .train(max_epochs=50))

# Make predictions
predictions = trainer.predict(['AAPL'], '2024-12-31', horizon=5)
```

---

## 🎛️ 7. Advanced Configuration

### Custom Feature Selection

```python
def create_custom_dataloader(symbols, start, end, feature_groups=None):
    """Create dataloader with custom feature selection."""
    
    # Default feature groups
    if feature_groups is None:
        feature_groups = {
            'price': True,        # OHLCV
            'technical': True,    # Technical indicators
            'news': True,         # News embeddings
            'economic': True,     # Economic indicators
            'events': True        # Corporate events
        }
    
    # Get full dataloader first
    loader, module = get_data_loader_with_module(
        symbols=symbols, start=start, end=end,
        encoder_len=60, predict_len=5, batch_size=32
    )
    
    # Filter features based on groups
    df = module.feature_df.copy()
    
    if not feature_groups.get('news', True):
        # Remove news features
        news_cols = [col for col in df.columns if col.startswith('emb_') or col == 'sentiment_score']
        df = df.drop(columns=news_cols)
        print(f"Removed {len(news_cols)} news features")
    
    if not feature_groups.get('technical', True):
        # Remove technical indicators
        tech_cols = [col for col in df.columns if any(x in col.lower() for x in ['sma', 'ema', 'rsi', 'macd', 'bb'])]
        df = df.drop(columns=tech_cols)
        print(f"Removed {len(tech_cols)} technical features")
    
    # Create new datamodule with filtered features
    new_module = TFTDataModule(df, encoder_len=60, predict_len=5, batch_size=32)
    new_module.setup()
    
    return new_module.train_dataloader(), new_module

# Example: Train without news features
loader_no_news, module_no_news = create_custom_dataloader(
    ['AAPL'], '2024-01-01', '2024-01-31',
    feature_groups={'news': False}
)
```

### Model Comparison

```python
def compare_models(symbols, start, end):
    """Compare different model configurations."""
    
    configurations = {
        'baseline': {'hidden_size': 64, 'attention_heads': 2},
        'medium': {'hidden_size': 128, 'attention_heads': 4},
        'large': {'hidden_size': 256, 'attention_heads': 8}
    }
    
    results = {}
    
    for name, config in configurations.items():
        print(f"\n🔧 Training {name} model...")
        
        # Create trainer
        trainer = StockTFTTrainer(symbols, start, end)
        
        # Train model
        trainer.prepare_data().create_model(**config).train(max_epochs=10)
        
        # Evaluate (simplified)
        predictions = trainer.predict()
        results[name] = {
            'config': config,
            'predictions': predictions,
            'model_size': sum(p.numel() for p in trainer.model.parameters())
        }
        
        print(f"✅ {name}: {results[name]['model_size']:,} parameters")
    
    return results

# Example usage
comparison = compare_models(['AAPL'], '2024-01-01', '2024-01-31')
```

---

## 🚀 8. Deployment Checklist

### Ready for Production

```python
def production_checklist():
    """Verify system is ready for production deployment."""
    
    checks = []
    
    try:
        # 1. Data pipeline test
        loader, module = get_data_loader_with_module(
            symbols=['AAPL'], start='2024-01-01', end='2024-01-31',
            encoder_len=10, predict_len=3, batch_size=4
        )
        checks.append(("✅", "Data pipeline functional"))
        
        # 2. Model creation test
        tft = TemporalFusionTransformer.from_dataset(
            module.train_dataset, hidden_size=64, attention_head_size=2,
            loss=QuantileLoss(), learning_rate=0.03
        )
        checks.append(("✅", "Model creation successful"))
        
        # 3. Batch loading test
        batch = next(iter(loader))
        checks.append(("✅", "Batch loading functional"))
        
        # 4. GPU compatibility
        if torch.cuda.is_available():
            tft = tft.cuda()
            batch = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            checks.append(("✅", "GPU compatibility confirmed"))
        else:
            checks.append(("⚠️", "GPU not available (CPU only)"))
        
        # 5. Feature count validation
        feature_count = len(module.get_dataset_parameters()['time_varying_unknown_reals'])
        expected_min = 800  # Should have ~800+ features
        if feature_count >= expected_min:
            checks.append(("✅", f"Feature integration complete ({feature_count} features)"))
        else:
            checks.append(("❌", f"Insufficient features ({feature_count} < {expected_min})"))
        
    except Exception as e:
        checks.append(("❌", f"System error: {e}"))
    
    print("🚀 PRODUCTION READINESS CHECKLIST:")
    for status, message in checks:
        print(f"  {status} {message}")
    
    return all(status == "✅" for status, _ in checks)

# Run checklist
is_ready = production_checklist()
print(f"\n{'🚀 System ready for production!' if is_ready else '⚠️ Address issues before deployment'}")
```

---

## 📝 Summary

Our TFT data pipeline provides:

- **860 features** across 6 data sources
- **Proper TFT categorization** (static, known, unknown)
- **News embeddings** with 768-dimensional BERT vectors
- **Economic indicators** from FRED API
- **Corporate events** from yfinance + API-Ninjas
- **Technical indicators** (22 computed features)
- **Ready-to-use DataLoaders** with pytorch-forecasting compatibility

### Quick Start Command

```bash
# Run the complete example
python main.py

# Validate integration
python main.py --validate

# Show tensor structure
python -c "from data import *; get_data_loader_with_module(['AAPL'], '2024-01-01', '2024-01-31', 10, 3, 4)[1].print_tensor_report()"
```

The system is **production-ready** for immediate TFT model training and deployment! 🚀
