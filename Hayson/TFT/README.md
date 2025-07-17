# 📈 Stock TFT (Temporal Fusion Transformer)


## ✅ Current Status

The TFT pipeline has been fully refactored and is production-ready:

- **✅ Real Model Integration**: All scripts use the real TFT model (no mock models)
- **✅ Out-of-Sample Validation**: Proper temporal and symbol-based validation
- **✅ Robust Trading Simulation**: Kelly criterion, DCA, and lump sum strategies
- **✅ OHLC Visualization**: Comprehensive plotting with predictions vs actuals
- **✅ Portfolio Analytics**: Distribution tracking and performance metrics
- **✅ Error Handling**: Graceful handling of edge cases and broadcasting errors
- **✅ Clean Codebase**: Removed debug statements and legacy code

### Quick Start

```bash
# Run the complete pipeline
cd Hayson/TFT
python unified_tft_pipeline.py --epochs 5 --symbols AAPL,MSFT

# Run with out-of-sample validation
python unified_tft_pipeline.py --out-of-sample temporal --test-symbol GOOGL --epochs 5

# Clear cache and run fresh
python unified_tft_pipeline.py --clear-cache --epochs 3
```

### Key Features

1. **Enhanced TFT Model**: Uses real multi-modal features (stock, news, economic, technical)
2. **Trading Strategies**: Kelly criterion, DCA, and buy-and-hold comparison
3. **Visualization Suite**: OHLC plots, trading signals, portfolio distribution
4. **Performance Metrics**: Sharpe ratio, max drawdown, win rate, profit factor
5. **Data Caching**: Intelligent caching system for faster iterations
6. **Configurable**: Easy to modify symbols, epochs, and validation methods

### Project Structure

- `Hayson/TFT/` - Main TFT implementation and training scripts
- `Code/TFT/` - Original TFT components and modules
- `Code/Testing/` - Legacy testing and experimental files
- `Code/Data/` - Data processing and raw datasets

### Main Entry Points

- `unified_tft_pipeline.py` - Complete training and analysis pipeline
- `train_baseline_tft.py` - Simple baseline TFT training
- `train_full_pipeline.py` - Full multi-modal TFT training

### Quick Start

```bash
# Run the complete pipeline
cd Hayson/TFT
python unified_tft_pipeline.py --epochs 5 --symbols AAPL,MSFT

# Run with out-of-sample validation
python unified_tft_pipeline.py --out-of-sample temporal --test-symbol GOOGL --epochs 5

# Clear cache and run fresh
python unified_tft_pipeline.py --clear-cache --epochs 3
```

### Key Features

1. **Enhanced TFT Model**: Uses real multi-modal features (stock, news, economic, technical)
2. **Trading Strategies**: Kelly criterion, DCA, and buy-and-hold comparison
3. **Visualization Suite**: OHLC plots, trading signals, portfolio distribution
4. **Performance Metrics**: Sharpe ratio, max drawdown, win rate, profit factor
5. **Data Caching**: Intelligent caching system for faster iterations
6. **Configurable**: Easy to modify symbols, epochs, and validation methods

### Project Structure

- `Hayson/TFT/` - Main TFT implementation and training scripts
- `Code/TFT/` - Original TFT components and modules
- `Code/Testing/` - Legacy testing and experimental files
- `Code/Data/` - Data processing and raw datasets

### Main Entry Points

- `unified_tft_pipeline.py` - Complete training and analysis pipeline
- `train_baseline_tft.py` - Simple baseline TFT training
- `train_full_pipeline.py` - Full multi-modal TFT training



Multi-Modal Stock Prediction using Temporal Fusion Transformers with News Sentiment Analysis.

