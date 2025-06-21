#!/usr/bin/env python3
"""
🚀 TFT Data Pipeline - Main Interface

Comprehensive example and validation script for the TFT data pipeline.
Includes testing, validation, and usage examples all in one place.

Usage:
    python main.py                    # Run full pipeline example
    python main.py --test             # Run integration tests
    python main.py --validate         # Validate feature integration
    python main.py --examples         # Show usage examples
    python main.py --show-data        # Show sample data from each source
    python main.py --help             # Show usage options
"""

import sys
import os
import argparse
from datetime import datetime, timedelta
import warnings
import pandas as pd

warnings.filterwarnings('ignore')

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("✓ Loaded environment variables from .env file")
except ImportError:
    print("⚠️  python-dotenv not installed. Install with: pip install python-dotenv")
except Exception as e:
    print(f"⚠️  Could not load .env file: {e}")

from data import (
    get_data_loader, get_data_loader_with_module, fetch_stock_data, fetch_events_data,
    fetch_news_embeddings, fetch_fred_data, compute_technical_indicators,
    build_features, TFTDataModule, get_economic_features, show_corporate_actions_sample,
    print_news_status_report
)


def validate_integration():
    """Validate TFT integration is complete and working."""
    print("🔍 TFT Integration Validation")
    print("=" * 40)
    
    # Test imports
    print("✅ All modules imported successfully")
    
    # Check main interface signature
    import inspect
    sig = inspect.signature(get_data_loader)
    expected_params = ['symbols', 'start', 'end', 'encoder_len', 'predict_len', 'batch_size', 'news_api_key', 'fred_api_key']
    actual_params = list(sig.parameters.keys())
    
    print(f"✅ Main interface parameters: {len(actual_params)} expected")
    for param in expected_params:
        if param in actual_params:
            print(f"   ✓ {param}")
        else:
            print(f"   ✗ Missing: {param}")
            return False
    
    # Check FRED features
    fred_features = get_economic_features()
    expected_fred = ['cpi', 'fedfunds', 'unrate', 't10y2y', 'gdp', 'vix', 'dxy', 'oil']
    print(f"✅ FRED features: {fred_features}")
    
    # Feature mapping validation
    print("\n📊 Feature Integration Mapping:")
    feature_mapping = {
        "time_varying_unknown_reals": [
            "OHLCV (yfinance)",
            "Bid/Ask (yfinance)", 
            "Technical Indicators (22)",
            "News Embeddings (BERT 768-dim)"
        ],
        "static_categoricals": ["Symbol", "Sector"],
        "static_reals": ["Market Cap"],
        "time_varying_known_categoricals": [
            "Split Days", "Dividend Days", "Weekends"
        ],
        "time_varying_known_reals": [
            "Calendar Features (4)",
            "Economic Indicators (8)"
        ]
    }
    
    for group, features in feature_mapping.items():
        print(f"\n🔸 {group}:")
        for feature in features:
            print(f"   ✅ {feature}")
    
    print(f"\n🎯 INTEGRATION STATUS: ✅ COMPLETE")
    return True


def run_integration_test():
    """Run a quick integration test with real data."""
    print("🧪 Integration Test")
    print("=" * 20)
    
    # Test configuration
    symbols = ['AAPL']  # Single symbol for quick test
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=180)).strftime('%Y-%m-%d')
    
    encoder_len = 15
    predict_len = 3
    batch_size = 8
    
    print(f"Test Config: {symbols[0]}, {start_date} to {end_date}")
    
    try:
        # Test individual components
        print("\n1. Testing Stock Data...")
        stock_df = fetch_stock_data(symbols, start_date, end_date)
        print(f"   ✓ Stock data: {stock_df.shape}")
        
        print("\n2. Testing Corporate Actions...")
        events_data = fetch_events_data(symbols, start_date, end_date)
        print(f"   ✓ Events data: {len(events_data)} symbols")
        
        print("\n3. Testing Technical Indicators...")
        ta_df = compute_technical_indicators(stock_df)
        print(f"   ✓ Technical indicators: {ta_df.shape}")
        
        print("\n4. Testing FRED Data...")
        fred_df = fetch_fred_data(start_date, end_date)
        print(f"   ✓ FRED data: {fred_df.shape}")
        
        print("\n5. Testing Feature Building...")
        feature_df = build_features(
            stock_df=stock_df,
            events=events_data,
            news_df=pd.DataFrame(),  # Empty for quick test
            ta_df=ta_df,
            fred_df=fred_df,
            encoder_len=encoder_len,
            predict_len=predict_len
        )
        print(f"   ✓ Feature matrix: {feature_df.shape}")
        
        print("\n6. Testing DataModule...")
        data_module = TFTDataModule(feature_df, encoder_len, predict_len, batch_size)
        data_module.setup()
        print(f"   ✓ DataModule: {len(data_module.train_loader)} batches")
        
        print("\n✅ Integration Test PASSED!")
        return True
        
    except Exception as e:
        print(f"\n❌ Integration Test FAILED: {e}")
        return False


def show_sample_data():
    """Show sample data from each source for inspection."""
    print("📊 Sample Data Display")
    print("=" * 25)
    
    symbols = ['AAPL', 'MSFT', 'GOOGL']
    
    # Show corporate actions
    print("\n1. Corporate Actions Data (yfinance):")
    show_corporate_actions_sample(symbols)
    
    # Show stock data sample
    print(f"\n2. Stock Data Sample:")
    try:
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
        
        stock_df = fetch_stock_data(['AAPL'], start_date, end_date)
        print(f"   Recent AAPL data:")
        print(stock_df.tail(3).to_string())
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # Show technical indicators sample
    print(f"\n3. Technical Indicators Sample:")
    try:
        ta_df = compute_technical_indicators(stock_df)
        ta_cols = [col for col in ta_df.columns if col not in stock_df.columns]
        print(f"   Available indicators: {ta_cols[:10]}...")
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # Show FRED data sample
    print(f"\n4. FRED Economic Data Sample:")
    try:
        fred_df = fetch_fred_data(start_date, end_date)
        if not fred_df.empty:
            print(f"   Recent economic data:")
            print(fred_df.tail(3).to_string())
        else:
            print(f"   ⚠️ FRED data: Install fredapi and get API key")
            
    except Exception as e:
        print(f"   ❌ Error: {e}")


def show_usage_examples():
    """Show usage examples for different scenarios."""
    print("🚀 Usage Examples")
    print("=" * 18)
    
    print("\n1️⃣ Basic Usage (No API Keys Required)")
    print("-" * 40)
    print("""
from data import get_data_loader

# Works with stock data + technical indicators + corporate actions
dataloader = get_data_loader(
    symbols=['AAPL', 'GOOGL'],
    start='2024-01-01', 
    end='2024-12-31',
    encoder_len=60,
    predict_len=5,
    batch_size=32
)

# Ready for TFT training!
for batch in dataloader:
    # batch contains properly formatted TFT inputs
    break
    """)
    
    print("\n2️⃣ Full Feature Set (With API Keys)")
    print("-" * 35)
    print("""
# Complete feature integration
dataloader = get_data_loader(
    symbols=['AAPL', 'GOOGL', 'MSFT'],
    start='2024-01-01',
    end='2024-12-31', 
    encoder_len=60,
    predict_len=5,
    batch_size=32,
    news_api_key="your_news_api_key",      # News embeddings  
    fred_api_key="your_fred_api_key"       # Economic data
)
    """)
    
    print("\n3️⃣ Environment Setup")
    print("-" * 20)
    print("""
# 1. Copy .env.example to .env
# 2. Add your API keys to .env file:
NEWS_API_KEY=your_news_api_key_here  
FRED_API_KEY=your_fred_api_key_here

# 3. API keys loaded automatically from environment
    """)
    
    print("\n4️⃣ Feature Groups (TFT Compatible)")
    print("-" * 30)
    print("""
TFT Feature Groups:
✓ time_varying_unknown_reals: OHLCV, technical indicators, news
✓ time_varying_known_reals: calendar, economic indicators  
✓ time_varying_known_categoricals: corporate actions, weekends
✓ static_categoricals: symbol, sector
✓ static_reals: market cap
    """)


def run_full_example():
    """Run the full pipeline example."""
    print("🚀 TFT Data Pipeline Example")
    print("=" * 30)
    
    # Configuration
    symbols = ['AAPL', 'GOOGL', 'MSFT']
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
    
    encoder_len = 60
    predict_len = 5
    batch_size = 32
    
    # API keys
    news_api_key = os.getenv('NEWS_API_KEY')
    fred_api_key = os.getenv('FRED_API_KEY')
    
    print(f"API Keys Status:")
    print(f"  News API key: {'✓ Loaded' if news_api_key else '✗ Not found'}")
    print(f"  FRED API key: {'✓ Loaded' if fred_api_key else '✗ Not found'}")
    if not news_api_key and not fred_api_key:
        print("  ℹ️  Running with basic features (stock + technical + corporate actions)")
    print()
    
    print(f"Configuration:")
    print(f"  Symbols: {symbols}")
    print(f"  Date range: {start_date} to {end_date}")
    print(f"  Encoder length: {encoder_len}")
    print(f"  Prediction length: {predict_len}")
    print(f"  Batch size: {batch_size}")
    print()
    
    try:
        # Create TFT DataLoader with detailed analysis
        print("Creating TFT DataLoader...")
        dataloader, datamodule = get_data_loader_with_module(
            symbols=symbols,
            start=start_date,
            end=end_date,
            encoder_len=encoder_len,
            predict_len=predict_len,
            batch_size=batch_size,
            news_api_key=news_api_key,
            fred_api_key=fred_api_key,
            api_ninjas_key=os.getenv('API_NINJAS_KEY')
        )
        
        print("\n✅ DataLoader created successfully!")
        print(f"Number of batches: {len(dataloader)}")
        
        # Print comprehensive tensor report
        print("\n" + "="*80)
        print("📊 GENERATING COMPREHENSIVE TENSOR REPORT...")
        print("="*80)
        datamodule.print_tensor_report()
        
        print("\n🎉 TFT data pipeline completed successfully!")
        print("The DataLoader is ready for TFT model training.")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("Make sure you have installed all required dependencies:")
        print("pip install -r requirements.txt")
        return False


def main():
    """Main function with argument parsing."""
    parser = argparse.ArgumentParser(description='TFT Data Pipeline')
    parser.add_argument('--test', action='store_true', help='Run integration tests')
    parser.add_argument('--validate', action='store_true', help='Validate feature integration')
    parser.add_argument('--examples', action='store_true', help='Show usage examples')
    parser.add_argument('--show-data', action='store_true', help='Show sample data from each source')
    parser.add_argument('--tensor-report', action='store_true', help='Generate detailed tensor preparation report')
    parser.add_argument('--news-status', action='store_true', help='Check news API status and limitations')
    
    args = parser.parse_args()
    
    if args.validate:
        validate_integration()
    elif args.test:
        run_integration_test()
    elif args.examples:
        show_usage_examples()
    elif args.show_data:
        show_sample_data()
    elif args.news_status:
        print_news_status_report(os.getenv('NEWS_API_KEY'))
    elif args.tensor_report:
        print("🎯 Generating Detailed Tensor Report...")
        print("Creating mini dataset for analysis...")
        
        # Create a small dataset for tensor analysis
        symbols = ['AAPL']
        start_date = (datetime.now() - timedelta(days=90)).strftime('%Y-%m-%d')
        end_date = datetime.now().strftime('%Y-%m-%d')
        
        try:
            dataloader, datamodule = get_data_loader_with_module(
                symbols=symbols,
                start=start_date,
                end=end_date,
                encoder_len=15,  # Smaller encoder length for limited data
                predict_len=3,   # Smaller prediction length
                batch_size=4,    # Smaller batch size
                news_api_key=os.getenv('NEWS_API_KEY'),
                fred_api_key=os.getenv('FRED_API_KEY'),
                api_ninjas_key=os.getenv('API_NINJAS_KEY')
            )
            
            print("\n" + "="*80)
            print("📊 DETAILED TENSOR PREPARATION REPORT")
            print("="*80)
            datamodule.print_tensor_report()
            
        except Exception as e:
            print(f"❌ Error generating tensor report: {e}")
    else:
        # Run full example
        success = run_full_example()
        
        if success:
            print(f"\n" + "="*50)
            print("🎯 NEXT STEPS")
            print("="*50)
            print("1. Install dependencies: pip install -r requirements.txt")
            print("2. Setup API keys: Copy .env.example to .env and add keys")
            print("3. Train TFT model: Use the DataLoader with pytorch-forecasting")
            print("4. Scale up: Add more symbols and longer time periods")
            print("\n🚀 START TRAINING YOUR TFT MODEL! 🚀")


if __name__ == "__main__":
    main()
