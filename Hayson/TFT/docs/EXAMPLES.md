# 💡 Examples - TFT Data Pipeline

## 🚀 Basic Examples

### 1. Minimal Setup (No API Keys)
```python
from data import get_data_loader_with_module

# Basic setup - works without any API keys
loader, module = get_data_loader_with_module(
    symbols=['AAPL'],
    start='2024-01-01',
    end='2024-01-31',
    encoder_len=10,
    predict_len=3,
    batch_size=4
)

print(f"✅ Created DataLoader with {len(loader)} batches")
print(f"Features: {module.feature_df.shape[1]} columns")
```

### 2. Production Setup (All API Keys)
```python
import os
from data import get_data_loader_with_module

# Full feature set with all APIs
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

# Analyze the data
module.print_tensor_report()
```

### 3. Multi-Symbol Analysis
```python
# Analyze multiple symbols across sectors
symbols = ['AAPL', 'GOOGL', 'JPM', 'XOM', 'JNJ']  # Tech, Finance, Energy, Healthcare

loader, module = get_data_loader_with_module(
    symbols=symbols,
    start='2024-01-01',
    end='2024-12-31',
    encoder_len=40,
    predict_len=7,
    batch_size=32
)

print(f"📊 Analyzing {len(symbols)} symbols across sectors")
print(f"🏢 Sectors: {module.feature_df['sector'].unique()}")
print(f"📈 Total samples: {len(module.feature_df)}")
```

---

## 🧠 TFT Model Training Examples

### 1. Quick Training (5 Minutes)
```python
from pytorch_forecasting import TemporalFusionTransformer, QuantileLoss
import pytorch_lightning as pl

# Get data
loader, module = get_data_loader_with_module(
    symbols=['AAPL'],
    start='2024-01-01',
    end='2024-12-31',
    encoder_len=30,
    predict_len=5,
    batch_size=16
)

# Create model
tft = TemporalFusionTransformer.from_dataset(
    module.train_dataset,
    hidden_size=64,          # Small for speed
    attention_head_size=2,
    dropout=0.1,
    hidden_continuous_size=16,
    loss=QuantileLoss(),
    learning_rate=0.03
)

# Quick training
trainer = pl.Trainer(
    max_epochs=5,
    logger=False,
    enable_checkpointing=False
)

trainer.fit(tft, loader, module.val_dataloader())
print("🎉 Quick training complete!")
```

### 2. Production Training
```python
from pytorch_forecasting import TemporalFusionTransformer, QuantileLoss
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
import pytorch_lightning as pl

# Production data setup
loader, module = get_data_loader_with_module(
    symbols=['AAPL', 'GOOGL', 'MSFT', 'AMZN'],
    start='2022-01-01',
    end='2024-12-31',
    encoder_len=60,
    predict_len=5,
    batch_size=64,
    news_api_key=os.getenv('NEWS_API_KEY'),
    fred_api_key=os.getenv('FRED_API_KEY'),
    api_ninjas_key=os.getenv('API_NINJAS_KEY')
)

# Production model
tft = TemporalFusionTransformer.from_dataset(
    module.train_dataset,
    hidden_size=256,
    attention_head_size=4,
    dropout=0.1,
    hidden_continuous_size=64,
    loss=QuantileLoss(),
    learning_rate=0.03
)

# Production training with callbacks
early_stop = EarlyStopping(monitor='val_loss', patience=10, mode='min')
checkpoint = ModelCheckpoint(
    monitor='val_loss',
    dirpath='./models/',
    filename='tft-{epoch:02d}-{val_loss:.2f}',
    save_top_k=3,
    mode='min'
)

trainer = pl.Trainer(
    max_epochs=100,
    callbacks=[early_stop, checkpoint],
    gpus=1 if torch.cuda.is_available() else 0
)

trainer.fit(tft, loader, module.val_dataloader())
print("🚀 Production training complete!")
```

### 3. Hyperparameter Optimization
```python
import optuna
from pytorch_forecasting import TemporalFusionTransformer, QuantileLoss

def objective(trial):
    # Suggest hyperparameters
    hidden_size = trial.suggest_categorical('hidden_size', [64, 128, 256])
    attention_heads = trial.suggest_int('attention_heads', 2, 8)
    dropout = trial.suggest_float('dropout', 0.1, 0.3)
    learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-1, log=True)
    
    # Get data
    loader, module = get_data_loader_with_module(
        symbols=['AAPL'], start='2024-01-01', end='2024-06-30',
        encoder_len=30, predict_len=5, batch_size=32
    )
    
    # Create model
    tft = TemporalFusionTransformer.from_dataset(
        module.train_dataset,
        hidden_size=hidden_size,
        attention_head_size=attention_heads,
        dropout=dropout,
        learning_rate=learning_rate,
        loss=QuantileLoss()
    )
    
    # Train
    trainer = pl.Trainer(max_epochs=10, logger=False, enable_checkpointing=False)
    trainer.fit(tft, loader, module.val_dataloader())
    
    return trainer.callback_metrics['val_loss'].item()

# Run optimization
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=20)

print(f"Best parameters: {study.best_params}")
print(f"Best validation loss: {study.best_value:.4f}")
```

---

## 🔮 Prediction Examples

### 1. Basic Prediction
```python
# Train model (from previous examples)
# ... training code ...

# Make predictions
predictions = tft.predict(module.val_dataloader(), return_y=True)

print(f"Prediction shape: {predictions.prediction.shape}")
print(f"Actual values shape: {predictions.y.shape}")

# Extract confidence intervals
median_pred = predictions.prediction[:, :, 2]  # 50th percentile
lower_bound = predictions.prediction[:, :, 0]  # 10th percentile
upper_bound = predictions.prediction[:, :, 4]  # 90th percentile

print(f"Median predictions: {median_pred[0]}")  # First sequence
```

### 2. Real-Time Prediction Function
```python
def predict_next_days(symbols, model, current_date, horizon=5):
    """
    Real-time prediction for given symbols.
    """
    from datetime import datetime, timedelta
    
    # Get recent data for context
    start_date = (datetime.strptime(current_date, '%Y-%m-%d') - timedelta(days=90)).strftime('%Y-%m-%d')
    
    # Fetch fresh data
    loader, _ = get_data_loader_with_module(
        symbols=symbols,
        start=start_date,
        end=current_date,
        encoder_len=60,
        predict_len=horizon,
        batch_size=len(symbols)
    )
    
    # Make predictions
    predictions = model.predict(loader, return_y=False)
    
    # Format results
    results = {}
    for i, symbol in enumerate(symbols):
        results[symbol] = {
            'median': predictions.prediction[i, :, 2].cpu().numpy(),
            'lower_bound': predictions.prediction[i, :, 0].cpu().numpy(),
            'upper_bound': predictions.prediction[i, :, 4].cpu().numpy(),
            'dates': [(datetime.strptime(current_date, '%Y-%m-%d') + timedelta(days=j+1)).strftime('%Y-%m-%d') 
                     for j in range(horizon)]
        }
    
    return results

# Example usage
predictions = predict_next_days(['AAPL', 'GOOGL'], tft, '2024-12-31', horizon=5)
for symbol, pred in predictions.items():
    print(f"\n{symbol} predictions:")
    for date, median, lower, upper in zip(pred['dates'], pred['median'], pred['lower_bound'], pred['upper_bound']):
        print(f"  {date}: {median:.3f} [{lower:.3f}, {upper:.3f}]")
```

### 3. Feature Importance Analysis
```python
# Get feature importance
interpretation = tft.interpret_output(
    module.val_dataloader().dataset[:50],
    return_attention=True
)

# Variable attention weights
var_importance = interpretation['attention'].mean(0)

# Get feature names
params = module.get_dataset_parameters()
feature_names = params['time_varying_unknown_reals']

# Top features
top_indices = var_importance.argsort()[-20:]
print("🏆 Top 20 Most Important Features:")
for i, idx in enumerate(reversed(top_indices)):
    feature_name = feature_names[idx]
    importance = var_importance[idx].item()
    print(f"{i+1:2d}. {feature_name}: {importance:.4f}")

# Analyze news impact
news_features = [i for i, name in enumerate(feature_names) if name.startswith('emb_') or name == 'sentiment_score']
if news_features:
    news_importance = var_importance[news_features].mean()
    print(f"\n📰 Average news importance: {news_importance:.4f}")
```

---

## 📊 Data Analysis Examples

### 1. Feature Distribution Analysis
```python
import matplotlib.pyplot as plt
import seaborn as sns

# Get data
loader, module = get_data_loader_with_module(
    symbols=['AAPL', 'GOOGL'], start='2024-01-01', end='2024-12-31',
    encoder_len=30, predict_len=5, batch_size=32
)

df = module.feature_df

# Analyze target distribution
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.hist(df['target'], bins=50, alpha=0.7)
plt.title('Target Distribution (Returns)')
plt.xlabel('Daily Return')

plt.subplot(1, 3, 2)
plt.boxplot([df[df['symbol'] == symbol]['target'] for symbol in df['symbol'].unique()])
plt.title('Returns by Symbol')
plt.xticks(range(1, len(df['symbol'].unique())+1), df['symbol'].unique())

plt.subplot(1, 3, 3)
sentiment_col = 'sentiment_score' if 'sentiment_score' in df.columns else None
if sentiment_col:
    plt.scatter(df[sentiment_col], df['target'], alpha=0.5)
    plt.title('Sentiment vs Returns')
    plt.xlabel('News Sentiment')
    plt.ylabel('Return')

plt.tight_layout()
plt.show()
```

### 2. Economic Indicators Analysis
```python
# Analyze economic indicators
economic_features = ['cpi', 'fedfunds', 'unrate', 't10y2y', 'gdp', 'vix', 'dxy', 'oil']
available_econ = [col for col in economic_features if col in df.columns]

if available_econ:
    plt.figure(figsize=(15, 10))
    
    for i, feature in enumerate(available_econ):
        plt.subplot(3, 3, i+1)
        plt.plot(df['date'], df[feature])
        plt.title(f'{feature.upper()}')
        plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.show()
    
    # Correlation with returns
    correlations = {}
    for feature in available_econ:
        corr = df[feature].corr(df['target'])
        correlations[feature] = corr
    
    print("📊 Economic Indicator Correlations with Returns:")
    for feature, corr in sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True):
        print(f"  {feature.upper()}: {corr:.4f}")
```

### 3. Event Impact Analysis
```python
# Analyze event impacts
event_features = ['is_earnings_day', 'is_split_day', 'is_dividend_day']
available_events = [col for col in event_features if col in df.columns]

print("📅 Event Impact Analysis:")
for event in available_events:
    if df[event].sum() > 0:  # If we have any events
        event_returns = df[df[event] == '1']['target']
        normal_returns = df[df[event] == '0']['target']
        
        print(f"\n{event}:")
        print(f"  Event days: {len(event_returns)}")
        print(f"  Avg return on event days: {event_returns.mean():.4f}")
        print(f"  Avg return on normal days: {normal_returns.mean():.4f}")
        print(f"  Difference: {event_returns.mean() - normal_returns.mean():.4f}")
        
        # Statistical test
        from scipy import stats
        t_stat, p_value = stats.ttest_ind(event_returns, normal_returns)
        print(f"  T-test p-value: {p_value:.4f}")
```

---

## 🔧 Custom Configuration Examples

### 1. Custom Feature Selection
```python
def create_custom_features_dataloader(symbols, start, end, include_news=True, include_technical=True):
    """Create dataloader with custom feature selection."""
    
    # Get full dataset first
    loader, module = get_data_loader_with_module(
        symbols=symbols, start=start, end=end,
        encoder_len=30, predict_len=5, batch_size=16
    )
    
    df = module.feature_df.copy()
    
    # Remove features based on selection
    if not include_news:
        news_cols = [col for col in df.columns if col.startswith('emb_') or col == 'sentiment_score']
        df = df.drop(columns=news_cols)
        print(f"Removed {len(news_cols)} news features")
    
    if not include_technical:
        tech_indicators = ['sma', 'ema', 'rsi', 'macd', 'bb', 'stoch', 'atr', 'obv']
        tech_cols = [col for col in df.columns if any(indicator in col.lower() for indicator in tech_indicators)]
        df = df.drop(columns=tech_cols)
        print(f"Removed {len(tech_cols)} technical features")
    
    # Create new datamodule
    from data.datamodule import TFTDataModule
    custom_module = TFTDataModule(df, encoder_len=30, predict_len=5, batch_size=16)
    custom_module.setup()
    
    return custom_module.train_dataloader(), custom_module

# Example usage
loader_basic, module_basic = create_custom_features_dataloader(
    ['AAPL'], '2024-01-01', '2024-06-30',
    include_news=False,
    include_technical=False
)

print(f"Basic features: {module_basic.feature_df.shape[1]} columns")
```

### 2. Different Time Horizons
```python
# Short-term (intraday-style)
short_term_loader, short_term_module = get_data_loader_with_module(
    symbols=['AAPL'],
    start='2024-11-01',
    end='2024-12-31',
    encoder_len=10,    # Look back 10 days
    predict_len=1,     # Predict 1 day ahead
    batch_size=8
)

# Medium-term (weekly)
medium_term_loader, medium_term_module = get_data_loader_with_module(
    symbols=['AAPL'],
    start='2024-01-01',
    end='2024-12-31',
    encoder_len=30,    # Look back 30 days
    predict_len=5,     # Predict 5 days ahead
    batch_size=16
)

# Long-term (monthly)
long_term_loader, long_term_module = get_data_loader_with_module(
    symbols=['AAPL'],
    start='2023-01-01',
    end='2024-12-31',
    encoder_len=60,    # Look back 60 days
    predict_len=20,    # Predict 20 days ahead
    batch_size=32
)

print("Time horizon comparison:")
print(f"Short-term: {short_term_module.feature_df.shape}")
print(f"Medium-term: {medium_term_module.feature_df.shape}")
print(f"Long-term: {long_term_module.feature_df.shape}")
```

### 3. Sector-Specific Analysis
```python
# Define sector groups
sector_groups = {
    'Technology': ['AAPL', 'GOOGL', 'MSFT', 'NVDA'],
    'Finance': ['JPM', 'BAC', 'WFC', 'C'],
    'Healthcare': ['JNJ', 'PFE', 'UNH', 'ABBV'],
    'Energy': ['XOM', 'CVX', 'COP', 'EOG']
}

sector_results = {}

for sector, symbols in sector_groups.items():
    print(f"\n📊 Analyzing {sector} sector...")
    
    try:
        loader, module = get_data_loader_with_module(
            symbols=symbols,
            start='2024-01-01',
            end='2024-12-31',
            encoder_len=40,
            predict_len=5,
            batch_size=16
        )
        
        sector_results[sector] = {
            'symbols': symbols,
            'samples': len(module.feature_df),
            'avg_volatility': module.feature_df['target'].std(),
            'dataloader': loader,
            'module': module
        }
        
        print(f"  ✅ {sector}: {len(symbols)} symbols, {len(module.feature_df)} samples")
        print(f"  📈 Avg volatility: {sector_results[sector]['avg_volatility']:.4f}")
        
    except Exception as e:
        print(f"  ❌ {sector}: Error - {e}")

# Compare sectors
print("\n🏆 Sector Comparison:")
for sector, results in sector_results.items():
    print(f"  {sector}: Vol={results['avg_volatility']:.4f}, Samples={results['samples']}")
```

---

## 🎯 Advanced Use Cases

### 1. Rolling Window Prediction
```python
def rolling_window_prediction(symbol, start_date, end_date, window_size=60, step_size=5):
    """
    Perform rolling window predictions over a time period.
    """
    from datetime import datetime, timedelta
    import pandas as pd
    
    start_dt = datetime.strptime(start_date, '%Y-%m-%d')
    end_dt = datetime.strptime(end_date, '%Y-%m-%d')
    
    predictions_list = []
    current_date = start_dt
    
    while current_date + timedelta(days=window_size) <= end_dt:
        window_start = (current_date - timedelta(days=window_size)).strftime('%Y-%m-%d')
        window_end = current_date.strftime('%Y-%m-%d')
        
        try:
            # Get data for current window
            loader, module = get_data_loader_with_module(
                symbols=[symbol],
                start=window_start,
                end=window_end,
                encoder_len=30,
                predict_len=step_size,
                batch_size=4
            )
            
            # Train quick model
            tft = TemporalFusionTransformer.from_dataset(
                module.train_dataset,
                hidden_size=32,
                attention_head_size=2,
                loss=QuantileLoss()
            )
            
            trainer = pl.Trainer(max_epochs=5, logger=False, enable_checkpointing=False)
            trainer.fit(tft, loader, module.val_dataloader())
            
            # Make prediction
            pred = tft.predict(loader, return_y=False)
            
            predictions_list.append({
                'date': current_date.strftime('%Y-%m-%d'),
                'prediction': pred.prediction[0, 0, 2].item(),  # Median prediction for next day
                'lower': pred.prediction[0, 0, 0].item(),
                'upper': pred.prediction[0, 0, 4].item()
            })
            
            print(f"✅ {current_date.strftime('%Y-%m-%d')}: {pred.prediction[0, 0, 2].item():.4f}")
            
        except Exception as e:
            print(f"❌ {current_date.strftime('%Y-%m-%d')}: {e}")
        
        current_date += timedelta(days=step_size)
    
    return pd.DataFrame(predictions_list)

# Example usage
rolling_predictions = rolling_window_prediction('AAPL', '2024-06-01', '2024-08-01')
print(rolling_predictions)
```

### 2. Multi-Symbol Cross-Validation
```python
def cross_validate_symbols(symbols, start, end, n_folds=3):
    """
    Perform cross-validation across different symbol combinations.
    """
    from sklearn.model_selection import KFold
    import numpy as np
    
    results = []
    
    # Split symbols into folds
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    symbol_indices = np.arange(len(symbols))
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(symbol_indices)):
        train_symbols = [symbols[i] for i in train_idx]
        val_symbols = [symbols[i] for i in val_idx]
        
        print(f"\n🔄 Fold {fold + 1}/{n_folds}")
        print(f"Train symbols: {train_symbols}")
        print(f"Validation symbols: {val_symbols}")
        
        try:
            # Train on subset of symbols
            train_loader, train_module = get_data_loader_with_module(
                symbols=train_symbols,
                start=start,
                end=end,
                encoder_len=30,
                predict_len=5,
                batch_size=16
            )
            
            # Create and train model
            tft = TemporalFusionTransformer.from_dataset(
                train_module.train_dataset,
                hidden_size=64,
                attention_head_size=2,
                loss=QuantileLoss()
            )
            
            trainer = pl.Trainer(max_epochs=10, logger=False, enable_checkpointing=False)
            trainer.fit(tft, train_loader, train_module.val_dataloader())
            
            # Validate on held-out symbols
            val_loader, val_module = get_data_loader_with_module(
                symbols=val_symbols,
                start=start,
                end=end,
                encoder_len=30,
                predict_len=5,
                batch_size=8
            )
            
            predictions = tft.predict(val_loader, return_y=True)
            
            # Calculate metrics
            mse = ((predictions.prediction[:, :, 2] - predictions.y) ** 2).mean().item()
            mae = torch.abs(predictions.prediction[:, :, 2] - predictions.y).mean().item()
            
            results.append({
                'fold': fold + 1,
                'train_symbols': train_symbols,
                'val_symbols': val_symbols,
                'mse': mse,
                'mae': mae
            })
            
            print(f"  MSE: {mse:.6f}, MAE: {mae:.6f}")
            
        except Exception as e:
            print(f"  ❌ Error in fold {fold + 1}: {e}")
    
    return results

# Example usage
cv_results = cross_validate_symbols(
    ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'NVDA'],
    '2024-01-01',
    '2024-06-30',
    n_folds=3
)

print("\n📊 Cross-Validation Results:")
for result in cv_results:
    print(f"Fold {result['fold']}: MSE={result['mse']:.6f}, MAE={result['mae']:.6f}")
```

---

**Complete examples for every use case!** 🚀
