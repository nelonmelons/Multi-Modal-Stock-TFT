# 📦 Installation Guide

## ⚡ Quick Install

```bash
# Clone repository
git clone <repository-url>
cd TFT-mm

# Install dependencies
pip install -r requirements.txt

# Setup environment (optional but recommended)
cp .env.example .env
# Edit .env with your API keys
```

## 🔑 API Keys (Optional)

### Get Free API Keys
- **NewsAPI**: https://newsapi.org/ (free tier: 30 days history)
- **FRED**: https://fred.stlouisfed.org/ (free, unlimited)
- **API-Ninjas**: https://api.api-ninjas.com/ (free tier available)

### Configure .env
```bash
NEWS_API_KEY=your_newsapi_key
FRED_API_KEY=your_fred_key
API_NINJAS_KEY=your_ninjas_key
```

## ✅ Verify Installation

```bash
# Quick validation
python main.py --validate

# Full test with sample data
python main.py
```

## 📋 Dependencies

Core packages (automatically installed):
- `torch` - PyTorch framework
- `pytorch-lightning` - Training framework
- `pytorch-forecasting` - TFT implementation
- `yfinance` - Stock data
- `pandas` - Data manipulation
- `numpy` - Numerical computing
- `transformers` - BERT models
- `requests` - API calls
- `fredapi` - Economic data
- `pandas-ta` - Technical analysis
- `python-dotenv` - Environment variables

## 🐍 Python Requirements

- Python 3.8+
- 4GB+ RAM recommended
- GPU optional (CUDA/MPS support)

## 🔧 Troubleshooting

### Common Issues

**Import Error:**
```bash
# Ensure you're in the correct directory
cd TFT-mm
python -c "from data import get_data_loader_with_module; print('✅ Success')"
```

**Missing Dependencies:**
```bash
pip install -r requirements.txt --upgrade
```

**API Issues:**
```bash
# Test without API keys (reduced features)
python -c "
from data import get_data_loader_with_module
loader, module = get_data_loader_with_module(['AAPL'], '2024-01-01', '2024-01-31', 10, 3, 4)
print('✅ Basic functionality working')
"
```

See [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md) for detailed solutions.

---

**Ready to start!** See [docs/QUICK_START.md](docs/QUICK_START.md) for usage examples.
