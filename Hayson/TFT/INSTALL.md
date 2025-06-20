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

## Troubleshooting Common Issues

### 1. NumPy Version Issues

If you see errors about NumPy versions (e.g., "Requires-Python >=3.7,<3.11"), your Python version might be too new. Try:

```bash
# For Python 3.11+
pip install numpy>=1.20.0,<1.25.0

# Then install other requirements
pip install -r requirements.txt
```

### 2. TA-Lib Installation Issues

TA-Lib can be difficult to install. We've included pandas-ta as an alternative:

```bash
# Instead of ta-lib, use pandas-ta
pip install pandas-ta>=0.3.14b0
```

If you really need TA-Lib:

**On macOS:**
```bash
brew install ta-lib
pip install ta-lib
```

**On Ubuntu/Debian:**
```bash
sudo apt-get install libta-lib0-dev
pip install ta-lib
```

**On Windows:**
```bash
# Download the appropriate wheel from:
# https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib
pip install TA_Lib‑0.4.28‑cp311‑cp311‑win_amd64.whl  # Example for Python 3.11
```

### 3. PyTorch Issues

If PyTorch installation fails, install it separately first:

```bash
# CPU version
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# GPU version (if you have CUDA)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 4. Transformers/BERT Dependencies

For news embeddings, you need transformers. If installation fails:

```bash
pip install transformers[torch]>=4.20.0
pip install sentencepiece>=0.1.97
```

## Python Version Compatibility

- **Python 3.8-3.10**: Full compatibility with all packages
- **Python 3.11+**: Some packages may need specific versions
- **Python 3.12+**: Limited compatibility, use minimal requirements

## Step-by-Step Installation

1. **Create a virtual environment (recommended):**
   ```bash
   python -m venv tft-env
   source tft-env/bin/activate  # On Windows: tft-env\Scripts\activate
   ```

2. **Upgrade pip:**
   ```bash
   pip install --upgrade pip
   ```

3. **Install requirements:**
   ```bash
   # Option A: Minimal (recommended for first try)
   pip install -r requirements-minimal.txt
   
   # Option B: Full features
   pip install -r requirements.txt
   ```

4. **Test the installation:**
   ```bash
   python test_package.py
   ```

## Optional Dependencies

You can install these later if needed:

```bash
# For events data
pip install finnhub-python>=2.4.0

# For news embeddings
pip install requests>=2.28.0 transformers>=4.20.0

# For visualization
pip install matplotlib>=3.5.0 seaborn>=0.11.0 plotly>=5.10.0

# For development
pip install pytest>=7.0.0 black>=22.0.0 jupyter>=1.0.0
```

## Alternative Package Versions

If you encounter version conflicts, try these alternatives:

```bash
# Alternative PyTorch versions
pip install torch==1.13.0  # If latest version conflicts

# Alternative lightning versions
pip install pytorch-lightning==1.9.0  # If v2.0+ conflicts

# Alternative transformers
pip install transformers==4.25.0  # If latest version conflicts
```

## Verification

After installation, verify everything works:

```bash
python -c "from data import get_data_loader; print('✅ Import successful!')"
python example_usage.py
```

## Getting Help

If you continue to have issues:

1. Check your Python version: `python --version`
2. Update pip: `pip install --upgrade pip`
3. Try the minimal requirements first
4. Search for the specific error message online
5. Consider using conda instead of pip for scientific packages

## Conda Alternative

If pip installations continue to fail, try conda:

```bash
conda create -n tft-env python=3.10
conda activate tft-env
conda install pytorch pandas numpy scikit-learn matplotlib
pip install pytorch-forecasting yfinance pandas-ta
```
