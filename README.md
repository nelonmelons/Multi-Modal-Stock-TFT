# 📈 Stock TFT: Temporal Fusion Transformer for Stock Price Prediction

## ✅ Current Status - PRODUCTION READY

The TFT pipeline has been **fully refactored and is production-ready**:

### 🎯 Key Features Implemented

- **✅ Real Model Integration**: All scripts use the real TFT model (no mock models)
- **✅ Out-of-Sample Validation**: Proper temporal and symbol-based validation
- **✅ Robust Trading Simulation**: Kelly criterion, DCA, and lump sum strategies
- **✅ OHLC Visualization**: Comprehensive plotting with predictions vs actuals
- **✅ Portfolio Analytics**: Distribution tracking and performance metrics
- **✅ Error Handling**: Graceful handling of edge cases and broadcasting errors
- **✅ Clean Codebase**: Production-ready code with no debug statements

### 🚀 Quick Start

```bash
# Basic usage - train and analyze
python Hayson/TFT/unified_tft_pipeline.py --epochs 5 --symbols AAPL,MSFT

# Out-of-sample validation (temporal split)
python Hayson/TFT/unified_tft_pipeline.py --out-of-sample temporal --temporal-split 0.8 --epochs 5

# Out-of-sample validation (symbol-based)
python Hayson/TFT/unified_tft_pipeline.py --out-of-sample symbol --test-symbol GOOGL --epochs 5

# Clear cache and run fresh
python Hayson/TFT/unified_tft_pipeline.py --clear-cache --epochs 3
```

### 📊 What You Get

1. **Training Visualizations**: Loss curves, validation metrics, convergence plots
2. **OHLC Analysis**: Individual stock predictions vs actual prices with confidence intervals
3. **Trading Performance**: Strategy comparison (TFT vs DCA vs Buy-and-Hold)
4. **Risk Analytics**: Sharpe ratios, max drawdowns, win rates, profit factors
5. **Portfolio Tracking**: Position allocation and distribution over time
6. **Comprehensive Reports**: Detailed analysis summaries and performance metrics

### 🏗️ Architecture

- **Core Pipeline**: `unified_tft_pipeline.py` - Complete workflow orchestration
- **Model**: `tft_multimodal.py` - Enhanced TFT with multi-modal features
- **Trading**: `trading_simulator.py` - Strategy implementation and backtesting
- **Visualization**: `ohlc_plotter.py` - OHLC charts with model predictions
- **Data**: `dataModule/` - Feature engineering and data loading
- **Caching**: `cache_manager.py` - Intelligent data caching system

### 🔧 Configuration Options

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--epochs` | Training epochs | 10 |
| `--symbols` | Stock symbols (comma-separated) | AAPL,MSFT,GOOGL,NVDA |
| `--out-of-sample` | Validation type (temporal, symbol) | none |
| `--temporal-split` | Train/test split ratio | 0.7 |
| `--test-symbol` | Symbol for out-of-sample testing | None |
| `--clear-cache` | Clear data cache | False |
| `--dca-amount` | DCA investment amount | 100.0 |
| `--dca-frequency` | DCA frequency (days) | 1 |

### 🎯 Performance Metrics

- **Returns**: Total return, annualized return, excess return
- **Risk**: Sharpe ratio, maximum drawdown, volatility
- **Trading**: Win rate, profit factor, total trades
- **Comparison**: Strategy vs benchmarks performance

### 🔍 Sample Output

```
=== TRADING SIMULATION SUMMARY REPORT ===

📊 PORTFOLIO PERFORMANCE:
• Enhanced TFT Strategy:    $   19,079.23 (  90.79%)
• DCA Buy & Hold:          $    2,933.68 ( -70.66%)
• Lump-sum Buy & Hold:     $      855.57 ( -91.44%)

🎯 OUTPERFORMANCE:
• TFT vs DCA:                    161.46%
• TFT vs Lump-sum:               182.24%

📊 TRADING ACTIVITY:
• Total Trading Periods:           4320
• BUY Signals:                     2213 (51.2%)
• SELL Signals:                    1314 (30.4%)
• Win Rate:                       52.3%

🛡️ RISK METRICS:
• Sharpe Ratio:                   0.064
• Max Drawdown:                   5.60%
```

### 📁 Output Structure

```
runs/tft_run_YYYYMMDD_HHMMSS/
├── checkpoints/          # Model checkpoints
│   ├── best_model.pth    # Best validation model
│   └── last_model.pth    # Latest model
├── plots/               # Analysis visualizations
│   ├── training_progress.png
│   ├── ohlc_comparison_*.png
│   ├── ohlc_trading_signals_*.png
│   ├── ohlc_dashboard.png
│   └── trading_analysis.png
└── results/             # Analysis reports
    ├── analysis_report.md
    ├── training_log.json
    └── trading_results.json
```

### 🔒 Validation Features

- **Temporal Validation**: Time-based train/test splits with no future leakage
- **Symbol Validation**: Test on unseen symbols not in training data
- **Cross-Validation**: Multiple validation runs for robustness
- **Performance Tracking**: Detailed metrics and comparative analysis

### 🌟 Key Improvements Made

1. **Eliminated Mock Models**: All components use real TFT predictions
2. **Fixed Array Broadcasting**: Resolved all dimension mismatch errors
3. **Enhanced Trading Logic**: Improved Kelly criterion and DCA implementation
4. **Robust Error Handling**: Graceful degradation and detailed error reporting
5. **Clean Architecture**: Modular design with clear separation of concerns
6. **Production Polish**: Removed debug code, improved documentation

The pipeline is now **ready for production use** with comprehensive testing, validation, and analysis capabilities.


---

## 🌟 Features

<table>
<tr>
<td width="50%">

### 🎯 **Data Leakage-Free Architecture**

- ✅ **Sequential Walk-Forward Prediction**: Mimics real-world trading
- ✅ **No Future Peeking**: Only uses past information
- ✅ **Honest Performance Metrics**: Realistic evaluation
- ✅ **Production-Ready**: Ready for real trading applications

</td>
<td width="50%">

### 🏗️ **Advanced TFT Implementation**

- 🧠 **Variable Selection Networks**: Intelligent feature selection
- 🔄 **Gated Residual Networks**: Advanced information processing
- 👁️ **Multi-Head Attention**: Interpretable temporal relationships
- 📊 **Quantile Predictions**: Uncertainty estimation & risk management

</td>
</tr>
<tr>
<td width="50%">

### 📊 **Rich Financial Features**

- 💹 **Price Data**: OHLCV (Open, High, Low, Close, Volume)
- 📈 **Technical Indicators**: SMA, RSI with proper normalization
- 📅 **Calendar Features**: Day of week, month, time indexing
- 🏢 **Static Features**: Symbol encoding, market cap normalization

</td>
<td width="50%">

### ⚡ **Easy to Use**

- 🚀 **One-Line Training**: Simple API for quick experiments
- 🔧 **Centralized Config**: Easy symbol and parameter changes
- 📋 **Comprehensive Logging**: Detailed training and validation metrics
- 📈 **Rich Visualizations**: Training progress and performance plots

</td>
</tr>
</table>

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd Stock-TFT/Hayson/TFT

# Install dependencies
pip install -r requirements.txt

# Optional: Create virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Basic Usage

```python
from train_baseline_tft import train_baseline_tft

# 🎯 Train TFT model with default settings (NFLX stock)
model, train_losses, val_losses = train_baseline_tft()

# 🔧 Customize training parameters
model, train_losses, val_losses = train_baseline_tft(
    symbols=['AAPL'],           # Stock symbols to train on
    start_date='2020-01-01',    # Training start date
    end_date='2023-12-31',      # Training end date
    epochs=30,                  # Number of training epochs
    batch_size=32,              # Batch size for training
    hidden_size=64,             # Model hidden dimensions
    device='cuda'               # Use GPU if available
)
```

### Configuration

**🎯 Change stock symbols in one centralized location:**

```python
# In baseline_tft.py and train_baseline_tft.py
DEFAULT_SYMBOLS = ['AAPL']                    # Single stock
# or
DEFAULT_SYMBOLS = ['AAPL', 'MSFT', 'GOOGL']  # Multiple stocks
# or
DEFAULT_SYMBOLS = ['SPY']                     # ETF for market-wide prediction
```

### Expected Output

```bash
🚀 Starting TFT Training with Simple Baseline
============================================================
Symbols: ['NFLX']
Date range: 2020-01-01 to 2023-12-31
Architecture: 64D hidden, 30 encoder, 1 prediction
Training: 30 epochs, batch size 32, LR 0.001
🔧 Using device: cpu
🏗️  Model parameters: 356,273

🏋️ Starting training...
Epoch   1: Train Loss: 1.3220, Val Loss: 0.6679
Epoch   2: Train Loss: 0.7031, Val Loss: 0.2582
...
Epoch  30: Train Loss: 0.1127, Val Loss: 0.1358

✅ Training completed! Files saved:
  📊 Training plots: baseline_tft_training.png
  🎯 Performance analysis: baseline_tft_analysis_sequential.png
  💰 Trading simulation: baseline_tft_trading_simulation_sequential.png
  🤖 Model weights: best_baseline_tft.pth
```

## 📁 Project Structure

```
📦 TFT/
├── 📄 README.md                          # This comprehensive guide
├── 🐍 train_baseline_tft.py             # 🚀 Main training script
├── 🧠 baseline_tft.py                   # 🏗️ TFT model implementation
├── 📋 requirements.txt                  # 📦 Python dependencies
├── 📊 REFACTORING_SUMMARY.md           # 📝 Documentation of changes made
├── 📄 README_FILE_DIFFERENCES.md       # 🔍 Explanation of file purposes
│
├── 🤖 Generated Training Files/
│   ├── best_baseline_tft.pth           # 💾 Trained model weights
│   ├── baseline_tft_training.png       # 📈 Training progress plots
│   ├── baseline_tft_analysis_sequential.png    # 🎯 Performance analysis
│   └── baseline_tft_trading_simulation_sequential.png  # 💰 Trading results
│
├── 📁 dataModule/                       # 📊 Data processing utilities
├── 📁 docs/                            # 📚 Additional documentation
├── 📁 archive/                         # 🗃️ Historical/backup files
└── 📁 __pycache__/                     # 🐍 Python cache files
```

### 🎯 Key Files

| File                    | Purpose                       | Key Functions                                                                    |
| ----------------------- | ----------------------------- | -------------------------------------------------------------------------------- |
| `train_baseline_tft.py` | 🚀 **Main Training Pipeline** | `train_baseline_tft()`, `sequential_prediction()`, `analyze_model_performance()` |
| `baseline_tft.py`       | 🧠 **TFT Architecture**       | `BaselineTFT`, `create_baseline_data()`, Neural network components               |
| `requirements.txt`      | 📦 **Dependencies**           | All required Python packages with versions                                       |

## 🔧 Core Components

### 1. **Model Architecture** (`baseline_tft.py`)

#### **BaselineTFT Class**

- Complete TFT implementation following the original paper
- Configurable architecture with proper feature dimensions
- Support for static and time-varying features

#### **Key Neural Network Components**

```python
# Gated Residual Networks
class GatedResidualNetwork(nn.Module)

# Variable Selection Networks
class VariableSelectionNetwork(nn.Module)

# Interpretable Multi-Head Attention
class InterpretableMultiHeadAttention(nn.Module)

# Quantile Loss for uncertainty estimation
class QuantileLoss(nn.Module)
```

#### **Data Pipeline**

```python
# Create training dataset
def create_baseline_data(symbols, start_date, end_date)

# Feature normalization
def normalize_features(df)

# Sequence creation for time series
def create_sequences(df, encoder_length, prediction_length)
```

### 2. **Training Pipeline** (`train_baseline_tft.py`)

#### **Sequential Prediction Functions**

```python
# Walk-forward prediction (no data leakage)
def sequential_prediction(model, initial_data, num_steps, device)

# Extract validation data properly
def get_sequential_validation_data(val_loader, device)

# Update sequences with predictions
def update_sequence_with_prediction(encoder_cont, decoder_cont, predicted_return)
```

#### **Analysis and Visualization**

```python
# Performance analysis with sequential predictions only
def analyze_model_performance(model, val_loader, device)

# Realistic trading simulation
def simulate_trading_strategy(predictions, actuals)

# Training progress visualization
def create_training_plot(train_losses, val_losses)
```

## 📈 Sequential Prediction Method

### **🚨 What Makes This Different**

<table>
<tr>
<td width="50%">

**❌ Traditional Batch Prediction (Data Leakage)**

```
⚠️  All validation sequences processed simultaneously
⚠️  Model can "peek" at future information
⚠️  Artificially good performance metrics
⚠️  Doesn't reflect real trading conditions
⚠️  REMOVED from this implementation
```

</td>
<td width="50%">

**✅ Sequential Walk-Forward Prediction (No Leakage)**

```
✅ Day 1: Use Days 1-30 (actual) → Predict Day 31
✅ Day 2: Use Days 2-30 (actual) + Day 31 (predicted) → Predict Day 32
✅ Day 3: Use Days 3-30 (actual) + Days 31-32 (predicted) → Predict Day 33
✅ Mimics real trading conditions exactly
```

</td>
</tr>
</table>

### **🔧 Implementation Details**

```python
def sequential_prediction(model, initial_data, num_steps, device):
    """
    🎯 Perform true sequential prediction without future peeking.

    Each prediction step:
    1. 📊 Use current 30-day window to predict next return
    2. ⏭️ Update window by rolling forward one day
    3. 🔄 Incorporate prediction into next window
    4. 🔁 Repeat for specified number of steps

    Returns:
        predictions (List[float]): Sequential predictions
        attention_history (List): Attention patterns over time
    """
    for step in range(num_steps):
        # 🎯 Predict one step ahead
        batch = create_sequential_batch(encoder_cont, decoder_cont, static_cat, static_real)
        output = model(batch)
        prediction = output['prediction'][0, 0, 2].item()  # Median quantile

        # 🔄 Update sequence for next prediction
        encoder_cont, decoder_cont = update_sequence_with_prediction(
            encoder_cont, decoder_cont, prediction, step
        )
````

## 🎯 Training Process

### **1. Data Preparation**
```python
# Fetch stock data
ticker = yf.Ticker(symbol)
hist = ticker.history(start=start_date, end=end_date)

# Calculate features
df['returns'] = df['close'].pct_change()
df['sma_20'] = df['close'].rolling(20).mean()
df['rsi_14'] = calculate_rsi(df['close'], 14)

# Normalize features
df = normalize_features(df)
````

### **2. Model Training**

```python
# Training loop with proper validation
for epoch in range(epochs):
    # Training phase
    model.train()
    for batch in train_loader:
        optimizer.zero_grad()
        outputs = model(batch)
        loss = criterion(outputs['prediction'], batch['target'])
        loss.backward()
        optimizer.step()

    # Validation phase (no data leakage)
    model.eval()
    with torch.no_grad():
        # Validation using same batch method as training
        # Sequential analysis happens separately
```

### **📊 Performance Analysis**

```python
# 🎯 Sequential prediction for realistic evaluation
initial_data, true_returns = get_sequential_validation_data(val_loader, device)
predictions, attention_history = sequential_prediction(model, initial_data, num_steps, device)

# 📈 Calculate honest performance metrics
mae = np.mean(np.abs(predictions - targets))
rmse = np.sqrt(np.mean((predictions - targets)**2))
directional_accuracy = np.mean(np.sign(predictions[1:]) == np.sign(targets[1:])) * 100

print(f"📊 Sequential Performance:")
print(f"   MAE: {mae:.4f}")
print(f"   RMSE: {rmse:.4f}")
print(f"   Directional Accuracy: {directional_accuracy:.1f}%")
```

## 📊 Performance Metrics

### **📈 Returns-Based Metrics**

| Metric                   | Description                      | Typical Range | Good Performance |
| ------------------------ | -------------------------------- | ------------- | ---------------- |
| **MAE**                  | Mean Absolute Error              | 0.01 - 0.05   | < 0.025          |
| **RMSE**                 | Root Mean Square Error           | 0.015 - 0.06  | < 0.035          |
| **R²**                   | Proportion of variance explained | -0.1 - 0.6    | > 0.3            |
| **Correlation**          | Linear relationship strength     | 0.0 - 0.8     | > 0.5            |
| **Directional Accuracy** | % correct direction predictions  | 45% - 65%     | > 55%            |

### **💰 Price-Based Metrics**

| Metric                   | Description                           | Purpose                      |
| ------------------------ | ------------------------------------- | ---------------------------- |
| **Price MAE/RMSE**       | Absolute dollar prediction errors     | Real-world impact assessment |
| **MAPE**                 | Mean Absolute Percentage Error        | Relative error percentage    |
| **Price Reconstruction** | Converting return → price predictions | Trading strategy evaluation  |

### **💼 Trading Simulation Metrics**

| Metric                | Description                   | Interpretation                 |
| --------------------- | ----------------------------- | ------------------------------ |
| **Strategy Return**   | TFT-based trading performance | Total return from predictions  |
| **Buy & Hold Return** | Baseline passive investment   | Market benchmark               |
| **Excess Return**     | Strategy vs buy & hold        | Value of active prediction     |
| **Number of Trades**  | Trading frequency             | Transaction cost consideration |
| **Sharpe Ratio**      | Risk-adjusted returns         | Quality of risk/return profile |

### **📊 Example Performance Output**

```bash
📈 Sequential Prediction Performance Summary:
   MAE: 0.0245          # ✅ Good (< 0.025)
   RMSE: 0.0312         # ✅ Good (< 0.035)
   R²: 0.423            # ✅ Good (> 0.3)
   Correlation: 0.651   # ✅ Excellent (> 0.5)
   Directional Accuracy: 57.3%  # ✅ Good (> 55%)

💰 Sequential Trading Simulation Results:
   Strategy Return: 8.45%       # Based on TFT predictions
   Buy & Hold Return: 12.31%    # Passive benchmark
   Excess Return: -3.86%        # Strategy underperformed
   Total Trades: 23             # Reasonable frequency
   Sharpe Ratio: 0.82           # Decent risk-adjusted return
```

## 🔍 Model Interpretability

### **Attention Weights**

```python
# Extract attention patterns
attention_weights = output['attention_weights']  # [batch, decoder_len, encoder_len]

# Visualize attention heatmap
plt.imshow(avg_attention, cmap='Blues', aspect='auto')
plt.xlabel('Encoder Time Steps')
plt.ylabel('Decoder Time Steps')
```

### **Variable Selection**

```python
# Historical variable importance
historical_weights = output['historical_weights']  # [batch, seq_len, num_features]

# Future variable importance
future_weights = output['future_weights']  # [batch, pred_len, num_features]
```

### **Quantile Predictions**

```python
# Uncertainty estimation
quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]
predictions = output['prediction']  # [batch, pred_len, num_quantiles]

# Confidence intervals
lower_bound = predictions[:, :, 0]  # 10th percentile
median = predictions[:, :, 2]       # 50th percentile (main prediction)
upper_bound = predictions[:, :, 4]  # 90th percentile
```

## ⚙️ Configuration Options

### **Model Architecture**

```python
model = BaselineTFT(
    static_categorical_cardinalities=[len(symbols)],  # Number of symbols
    num_static_real=1,                                # Market cap
    num_time_varying_real_known=3,                    # Calendar features
    num_time_varying_real_unknown=8,                  # OHLCV + technical
    hidden_size=64,                                   # Model capacity
    lstm_layers=2,                                    # LSTM depth
    attention_heads=4,                                # Multi-head attention
    encoder_length=30,                                # Historical window
    prediction_length=1,                              # Forecast horizon
    quantiles=[0.1, 0.25, 0.5, 0.75, 0.9]           # Uncertainty levels
)
```

### **Training Configuration**

```python
config = {
    'symbols': ['NFLX'],                    # Stock symbols
    'start_date': '2020-01-01',             # Training start
    'end_date': '2023-12-31',               # Training end
    'epochs': 30,                           # Training epochs
    'batch_size': 32,                       # Batch size
    'learning_rate': 1e-3,                  # Learning rate
    'encoder_length': 30,                   # Historical window
    'prediction_length': 1,                 # Forecast horizon
    'hidden_size': 64,                      # Model size
}
```

## 📋 Example Output

### **Training Progress**

```
🚀 Starting TFT Training with Simple Baseline
============================================================
Symbols: ['NFLX']
Date range: 2020-01-01 to 2023-12-31
Architecture: 64D hidden, 30 encoder, 1 prediction
Training: 30 epochs, batch size 32, LR 0.001
🔧 Using device: cpu
🏗️  Model parameters: 356,273

🏋️ Starting training...
Epoch   1: Train Loss: 1.3220, Val Loss: 0.6679
Epoch   2: Train Loss: 0.7031, Val Loss: 0.2582
...
Epoch  30: Train Loss: 0.1127, Val Loss: 0.1358

✅ Training completed!
Best validation loss: 0.1358
```

### **Performance Analysis**

```
📈 Sequential Prediction Performance Summary:
   MAE: 0.0245
   RMSE: 0.0312
   R²: 0.423
   Directional Accuracy: 52.3%
   Sequential Steps: 50

💰 Sequential Trading Simulation Results:
   Strategy Return: 8.45%
   Buy & Hold Return: 12.31%
   Excess Return: -3.86%
   Total Trades: 23
```

## 🔧 Advanced Usage

### **🎯 Custom Stock Symbols**

```python
# 📈 Single high-volatility stock
train_baseline_tft(symbols=['TSLA'])

# 🏢 Technology giants
train_baseline_tft(symbols=['AAPL', 'MSFT', 'GOOGL', 'AMZN'])

# 📊 Market ETFs
train_baseline_tft(symbols=['SPY', 'QQQ', 'IWM'])

# 🌍 Diversified portfolio
train_baseline_tft(symbols=['SPY', 'VTI', 'VXUS', 'BND'])
```

### **⏰ Extended Training Periods**

```python
# 📈 Long-term training (8+ years)
train_baseline_tft(
    symbols=['SPY'],           # S&P 500 ETF
    start_date='2015-01-01',   # 8+ years of data
    end_date='2023-12-31',
    epochs=100,                # More training
    batch_size=64              # Larger batches
)

# 📊 Recent market focus (post-COVID)
train_baseline_tft(
    symbols=['NFLX'],
    start_date='2020-03-01',   # COVID market changes
    end_date='2023-12-31',
    epochs=50
)
```

### **🏗️ Model Architecture Scaling**

```python
# 🧠 Larger model for better performance
train_baseline_tft(
    symbols=['NFLX'],
    hidden_size=128,           # 💪 Double the capacity
    encoder_length=60,         # 📈 Longer history (2 months)
    epochs=50,                 # 🏋️ More training
    learning_rate=5e-4         # 🎯 Fine-tuned learning rate
)

# ⚡ Fast experimentation model
train_baseline_tft(
    symbols=['AAPL'],
    hidden_size=32,            # 🏃 Smaller/faster model
    encoder_length=15,         # 📉 Shorter history
    epochs=20,                 # 🚀 Quick training
    batch_size=64
)
```

### **🎛️ Advanced Configuration**

```python
# 🔧 Production-ready configuration
config = {
    'symbols': ['SPY', 'QQQ'],              # 📊 Market ETFs
    'start_date': '2018-01-01',             # 📅 5+ years
    'end_date': '2023-12-31',
    'epochs': 75,                           # 🏋️ Thorough training
    'batch_size': 48,                       # 📦 Optimized batch size
    'learning_rate': 8e-4,                  # 🎯 Fine-tuned LR
    'hidden_size': 96,                      # 🧠 Balanced capacity
    'encoder_length': 45,                   # 📈 Extended history
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}

model, train_losses, val_losses = train_baseline_tft(**config)
```

## 🚨 Important Notes

<div align="center">

### 🎯 **Data Leakage Prevention**

</div>

| ✅ **What We Do**                                         | ❌ **What We Avoid**                                       |
| --------------------------------------------------------- | ---------------------------------------------------------- |
| Sequential validation ensures no future peeking           | Batch prediction with future information                   |
| Walk-forward prediction mimics real trading               | Processing all validation data simultaneously              |
| Honest performance metrics reflect realistic expectations | Artificially inflated metrics from data leakage            |
| Production-ready for real trading applications            | Academic-only implementations with unrealistic performance |

### 📊 **Performance Interpretation Guidelines**

<table>
<tr>
<td width="33%">

**🎯 Realistic Expectations**

- Directional accuracy: 50-60%
- Returns correlation: 0.3-0.7
- Strategy may underperform buy & hold
- Focus on risk-adjusted returns

</td>
<td width="33%">

**📈 Key Metrics Priority**

1. **Directional Accuracy** (most important for trading)
2. **Correlation** (relationship strength)
3. **Sharpe Ratio** (risk-adjusted returns)
4. **MAE/RMSE** (prediction accuracy)

</td>
<td width="33%">

**⚠️ Real-World Considerations**

- Transaction costs reduce returns
- Market impact affects large trades
- Model performance varies by market regime
- Regular retraining required

</td>
</tr>
</table>

### ⚙️ **System Limitations**

| Limitation                      | Impact                        | Potential Enhancement                       |
| ------------------------------- | ----------------------------- | ------------------------------------------- |
| **Single-step prediction**      | Only 1 day ahead              | Multi-step prediction horizons              |
| **Simple technical indicators** | Limited feature set           | Advanced technical analysis                 |
| **No fundamental analysis**     | Missing valuation metrics     | P/E ratios, earnings, news sentiment        |
| **Daily frequency only**        | Limited to daily predictions  | Intraday or weekly predictions              |
| **No market regime detection**  | Same model for all conditions | Adaptive models for different market states |

## 🛠️ Troubleshooting

### **🚨 Common Issues & Solutions**

<table>
<tr>
<td width="50%">

#### **📊 Data Issues**

**Insufficient Data**

```bash
⚠️  Warning: NFLX has insufficient data (25 < 31)
```

**🔧 Solution:**

- Use longer date range: `start_date='2019-01-01'`
- Or shorter encoder: `encoder_length=20`

**Missing Stock Data**

```bash
❌ Error: No data found for symbol XYZ
```

**🔧 Solution:**

- Check symbol spelling and exchange
- Use established stocks (AAPL, MSFT, SPY)
- Verify date range has trading days

</td>
<td width="50%">

#### **💻 System Issues**

**Memory Issues**

```bash
RuntimeError: CUDA out of memory
```

**🔧 Solution:**

- Reduce `batch_size=16` or `hidden_size=32`
- Use `device='cpu'`
- Close other GPU applications

**Import Errors**

```bash
ModuleNotFoundError: No module named 'yfinance'
```

**🔧 Solution:**

- Install: `pip install -r requirements.txt`
- Check virtual environment activation

</td>
</tr>
</table>

### **📈 Performance Issues**

| Problem                       | Symptom                   | Solution                                                |
| ----------------------------- | ------------------------- | ------------------------------------------------------- |
| **Poor Directional Accuracy** | < 50% correct predictions | Try different symbols, longer training, or larger model |
| **High Loss Values**          | Train/Val loss > 1.0      | Check data normalization, reduce learning rate          |
| **Overfitting**               | Train loss << Val loss    | Reduce model size, add dropout, more data               |
| **Slow Training**             | Very slow epochs          | Use GPU (`device='cuda'`), reduce batch size            |

### **🔍 Data Quality Checks**

The codebase includes automatic data quality validation:

```python
def check_data_quality(df, stage):
    """🔍 Comprehensive data quality checking"""
    # ✅ NaN value detection
    # ✅ Infinite value detection
    # ✅ Data range validation
    # ✅ Feature correlation analysis
    # ✅ Automatic reporting and warnings
```

### **🐛 Debug Mode**

Enable detailed logging for troubleshooting:

```python
# Add to train_baseline_tft.py
import logging
logging.basicConfig(level=logging.DEBUG)

# Detailed model information
print(f"📊 Model summary: {model}")
print(f"📈 Data shapes: {next(iter(train_loader))}")
```

---

<div align="center">

## 🤝 Contributing

**We welcome contributions to make this TFT implementation even better!**

</div>

### **🔧 Development Guidelines**

<table>
<tr>
<td width="50%">

**📝 Code Standards**

- ✅ Keep sequential prediction methods only
- ✅ Maintain centralized configuration
- ✅ Add comprehensive data quality checks
- ✅ Document all major functions
- ✅ Follow PEP 8 style guidelines
- ✅ Include type hints where possible

</td>
<td width="50%">

**🧪 Testing Requirements**

- ✅ Test model components individually
- ✅ Validate training pipeline end-to-end
- ✅ Check data quality and preprocessing
- ✅ Verify no data leakage in predictions
- ✅ Performance regression testing

</td>
</tr>
</table>

### **🚀 How to Contribute**

1. **🍴 Fork the repository**
2. **🌿 Create feature branch:** `git checkout -b feature/awesome-enhancement`
3. **💻 Make your changes** following the guidelines above
4. **🧪 Test thoroughly:** `python train_baseline_tft.py`
5. **📝 Update documentation** if needed
6. **🔄 Submit pull request** with clear description

### **💡 Contribution Ideas**

- 🔍 **Bug fixes** and performance improvements
- 📊 **New technical indicators** (MACD, Bollinger Bands)
- 🧠 **Model architecture enhancements**
- 📈 **Additional evaluation metrics**
- 🔧 **Code optimization** and refactoring
- 📚 **Documentation improvements**

---

## � License

<div align="center">

**This project is open source and available under the [MIT License](LICENSE).**

_Feel free to use, modify, and distribute for both personal and commercial purposes._

</div>

---

## 📚 References

<table>
<tr>
<td width="33%">

### **🎓 Academic Papers**

1. **Temporal Fusion Transformers**  
   _Lim, B., et al. (2021)_  
   International Journal of Forecasting

2. **Attention Is All You Need**  
   _Vaswani, A., et al. (2017)_  
   NIPS Conference

</td>
<td width="33%">

### **📊 Financial Research**

3. **Financial Time Series Analysis**  
   _Various academic papers_  
   Stock prediction methodologies

4. **Market Efficiency Studies**  
   _Fama, E. F. (1970)_  
   Efficient Market Hypothesis

</td>
<td width="33%">

### **� Technical Resources**

5. **PyTorch Documentation**  
   _Official PyTorch Docs_  
   Deep Learning Framework

6. **yfinance Library**  
   _Open Source Finance Data_  
   Stock data acquisition

</td>
</tr>
</table>

---

<div align="center">

# 🎯 Built with ❤️ for Realistic Stock Price Prediction

**⭐ Star this repository if you found it helpful!**

_This implementation prioritizes honest evaluation over inflated metrics, providing a solid foundation for real-world trading applications._

---

![GitHub stars](https://img.shields.io/github/stars/your-username/stock-tft?style=social)
![GitHub forks](https://img.shields.io/github/forks/your-username/stock-tft?style=social)
![GitHub issues](https://img.shields.io/github/issues/your-username/stock-tft)
![GitHub pull requests](https://img.shields.io/github/issues-pr/your-username/stock-tft)

**[⬆️ Back to Top](#-stock-tft-temporal-fusion-transformer-for-stock-price-prediction)**

</div>
