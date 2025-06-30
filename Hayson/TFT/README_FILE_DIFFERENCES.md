# File Differences and Usage Guide

## 📁 train_baseline_tft.py vs baseline_tft.py

### **train_baseline_tft.py - Main Training Pipeline**

**Purpose**: This is the **main entry point** for training and evaluating TFT models.

**Key Functions**:

- **End-to-end training pipeline**: Data loading → Model training → Evaluation → Analysis
- **Model training**: Complete training loop with validation, checkpointing, early stopping
- **Sequential prediction**: Implements true walk-forward prediction without future peeking
- **Performance analysis**: Comprehensive metrics, visualizations, and trading simulation
- **Configuration management**: Centralized settings for experiments

**When to use**:

- ✅ **Training new models from scratch**
- ✅ **Running complete experiments**
- ✅ **Evaluating model performance**
- ✅ **Generating training plots and analysis**
- ✅ **Running trading simulations**

**Typical workflow**:

```python
# Run the full training pipeline
python train_baseline_tft.py

# Or import and configure
from train_baseline_tft import train_baseline_tft
model, losses = train_baseline_tft(
    symbols=['AAPL'],
    epochs=50,
    use_enhanced_pipeline=False
)
```

---

### **baseline_tft.py - Model and Data Utilities**

**Purpose**: This is a **utility module** containing the TFT model architecture and data processing functions.

**Key Components**:

- **TFT model architecture**: BaselineTFT class with proper TFT components (VSNs, GRNs, attention)
- **Data creation**: Functions to generate training datasets from stock data
- **Normalization**: Robust feature normalization and data quality checks
- **Model components**: Individual neural network modules (GRN, VSN, attention)

**When to use**:

- ✅ **Quick model testing** (can be run standalone)
- ✅ **Understanding model architecture**
- ✅ **Data preprocessing experiments**
- ✅ **Importing model components for other scripts**

**Typical workflow**:

```python
# Import for use in other scripts
from baseline_tft import BaselineTFT, create_baseline_data

# Or run standalone for quick testing
python baseline_tft.py  # Tests model creation and forward pass
```

---

## 🔧 Stock Symbol Configuration

### **Centralized Configuration**

Both files now use **centralized stock symbol configuration** to eliminate hardcoded values:

```python
# At the top of both files:
DEFAULT_SYMBOLS = ['AMD']  # Primary symbol(s) for training
FALLBACK_SYMBOL = 'AAPL'   # Fallback symbol for evaluation functions
```

### **How to Change Stock Symbols**

**Option 1: Edit the configuration (recommended)**

```python
# In both train_baseline_tft.py and baseline_tft.py
DEFAULT_SYMBOLS = ['AAPL']        # Single stock
# DEFAULT_SYMBOLS = ['SPY']       # Market ETF
# DEFAULT_SYMBOLS = ['MSFT', 'GOOGL']  # Multiple stocks
```

**Option 2: Pass symbols as parameters**

```python
# When calling functions directly
train_baseline_tft(symbols=['TSLA', 'NVDA'])
create_baseline_data(symbols=['SPY'])
```

**Option 3: Update main config (train_baseline_tft.py)**

```python
config = {
    'symbols': ['AAPL', 'MSFT'],  # Override in config dict
    # ... other settings
}
```

---

## 🏃‍♂️ Running the Scripts

### **Run Complete Training Pipeline**

```bash
# Full training with default settings (quantitative features only)
python train_baseline_tft.py

# Output files:
# - best_baseline_tft.pth (trained model)
# - baseline_tft_training.png (training progress)
# - baseline_tft_analysis_sequential.png (performance analysis)
# - baseline_tft_trading_simulation_sequential.png (trading results)
```

### **Quick Model Test**

```bash
# Test model architecture and data creation
python baseline_tft.py

# Output:
# - Verifies model creation
# - Tests forward pass
# - Validates data processing
```

### **Key Differences in Execution**

| Feature                   | train_baseline_tft.py        | baseline_tft.py       |
| ------------------------- | ---------------------------- | --------------------- |
| **Training Loop**         | ✅ Full training with epochs | ❌ No training        |
| **Model Saving**          | ✅ Saves best model          | ❌ No saving          |
| **Validation**            | ✅ Train/validation split    | ❌ No validation      |
| **Sequential Prediction** | ✅ Walk-forward analysis     | ❌ No prediction      |
| **Visualizations**        | ✅ Multiple analysis plots   | ❌ No plots           |
| **Trading Simulation**    | ✅ Performance analysis      | ❌ No simulation      |
| **Configuration**         | ✅ Comprehensive config      | ❌ Basic testing only |

---

## 📊 Output Files Explained

### **Training Pipeline Outputs (train_baseline_tft.py)**

1. **`best_baseline_tft.pth`**: Trained model weights (best validation loss)
2. **`baseline_tft_training.png`**: Training curves and convergence analysis
3. **`baseline_tft_analysis_sequential.png`**:
   - Predictions vs actuals
   - Residual analysis
   - Attention heatmaps
   - Performance metrics
4. **`baseline_tft_trading_simulation_sequential.png`**: Trading strategy results

### **Test Outputs (baseline_tft.py)**

- Console output only (model verification)
- No saved files

---

## 🎯 Use Case Summary

### **For Training New Models**

```bash
python train_baseline_tft.py  # Complete pipeline
```

### **For Model Development/Testing**

```bash
python baseline_tft.py  # Quick architecture test
```

### **For Custom Experiments**

```python
# Import components as needed
from baseline_tft import BaselineTFT, create_baseline_data
from train_baseline_tft import train_baseline_tft, analyze_model_performance
```

### **For Different Stock Symbols**

1. Edit `DEFAULT_SYMBOLS` in both files, OR
2. Pass `symbols=['YOUR_SYMBOL']` to functions, OR
3. Update the config dict in `train_baseline_tft.py`

---

## ⚠️ Important Notes

- **Sequential Prediction**: Only available in `train_baseline_tft.py` - implements true walk-forward prediction without future peeking
- **Quantitative Focus**: Both files use `use_enhanced_pipeline: False` by default for quantitative-only features
- **Consistent Symbols**: Make sure to use the same symbols across both files for consistency
- **Memory Usage**: `train_baseline_tft.py` uses more memory due to full training pipeline

---

This refactoring ensures that stock symbols are defined in one place and can be easily changed for different experiments while maintaining clear separation between the training pipeline and utility modules.
