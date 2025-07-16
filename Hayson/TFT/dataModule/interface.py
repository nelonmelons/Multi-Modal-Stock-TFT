"""
Main interface for the TFT data pipeline.
Provides the high-level entry point function `get_data_loader`.
"""

from typing import List, Optional, Tuple, Any
import pandas as pd
import torch
from torch.utils.data import DataLoader
import sys
import os

# Add parent directory to path for cache_manager import
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .fetch_stock import fetch_stock_data
from .fetch_events import fetch_events_data
from .fetch_news import fetch_news_embeddings
from .fetch_fred import fetch_fred_data
from .compute_ta import compute_technical_indicators
from .build_features import build_features
from .datamodule import TFTDataModule
from .adaptive_loader import create_adaptive_dataloader

# Import caching system
try:
    from cache_manager import get_cache_instance, print_cache_info
    CACHING_AVAILABLE = True
except ImportError:
    print("⚠️ Caching system not available - running without cache")
    CACHING_AVAILABLE = False


def get_data_loader(symbols: List[str], start: str, end: str,
        encoder_len: int, predict_len: int,
        batch_size: int,
        news_api_key: Optional[str] = None,
        fred_api_key: Optional[str] = None,
        api_ninjas_key: Optional[str] = None) -> DataLoader:
    """
    High-level entry point for creating a TFT-ready DataLoader.
    
    Args:
        symbols: List of stock symbols (e.g., ['AAPL', 'GOOGL'])
        start: Start date in 'YYYY-MM-DD' format
        end: End date in 'YYYY-MM-DD' format
        encoder_len: Number of historical time steps for encoder
        predict_len: Number of future time steps to predict
        batch_size: Batch size for DataLoader
        news_api_key: News API key for news embeddings (optional)
        fred_api_key: FRED API key for macroeconomic data (optional)
        api_ninjas_key: API-Ninjas key for earnings calendar (optional)
    
    Returns:
        PyTorch DataLoader producing batches ready for TFT training
    
    Process:
        1. Fetch stock OHLCV data (yfinance)
        2. Fetch corporate actions data (yfinance: dividends, splits)
        3. Fetch news embeddings (NewsAPI + BERT)
        4. Fetch FRED macroeconomic data
        5. Compute technical indicators
        6. Build feature matrix
        7. Create DataLoader with TimeSeriesDataSet
    """
    print(f"Starting TFT data pipeline for symbols: {symbols}")
    print(f"Date range: {start} to {end}")
    print(f"Encoder length: {encoder_len}, Predict length: {predict_len}")
    
    # Initialize caching system
    cache = None
    if CACHING_AVAILABLE:
        cache = get_cache_instance()
        print(f"📦 Caching enabled - checking for cached data...")
        print_cache_info()
    else:
        print(f"📦 Caching disabled - fetching fresh data...")
    
    # Step 1: Fetch stock data (with caching)
    print("1. Fetching stock data...")
    if cache:
        stock_df = cache.get_or_fetch_stock_data(symbols, start, end, fetch_stock_data)
    else:
        stock_df = fetch_stock_data(symbols, start, end)
    print(f"   Retrieved {len(stock_df)} stock data points")
    
    # Step 2: Fetch events data (with caching)
    print("2. Fetching corporate actions data...")
    if cache:
        events_data = cache.get_or_fetch_events_data(symbols, start, end, None, api_ninjas_key, fetch_events_data)
    else:
        events_data = fetch_events_data(symbols, start, end, None, api_ninjas_key)
    print(f"   Retrieved events for {len(events_data)} symbols")
    
    # Step 3: Fetch news embeddings (with caching)
    print("3. Fetching news embeddings...")
    news_df = pd.DataFrame()
    if news_api_key:
        print("   Note: BERT model will be cached on first use (~400MB)")
        print("   Pre-load with: python manage_models.py preload")
        if cache:
            news_df = cache.get_or_fetch_news_data(symbols, start, end, news_api_key, fetch_news_embeddings)
        else:
            news_df = fetch_news_embeddings(symbols, start, end, news_api_key)
    else:
        print("   No news API key provided, skipping news embeddings")
    print(f"   Retrieved {len(news_df)} news embeddings")
    
    # Step 4: Fetch FRED macroeconomic data (with caching)
    print("4. Fetching FRED macroeconomic data...")
    if cache:
        fred_df = cache.get_or_fetch_fred_data(start, end, fred_api_key, fetch_fred_data)
    else:
        fred_df = fetch_fred_data(start, end, fred_api_key)
    print(f"   Retrieved {len(fred_df)} economic data points")
    
    # Step 5: Compute technical indicators (with caching)
    print("5. Computing technical indicators...")
    if cache:
        ta_df = cache.get_or_fetch_ta_data(stock_df, compute_technical_indicators)
    else:
        ta_df = compute_technical_indicators(stock_df)
    print(f"   Computed technical indicators for {len(ta_df)} data points")
    
    # Step 6: Build features (with caching)
    print("6. Building feature matrix...")
    if cache:
        feature_df = cache.get_or_build_features(stock_df, events_data, news_df, ta_df, fred_df,
                                               encoder_len, predict_len, build_features)
    else:
        feature_df = build_features(stock_df, events_data, news_df, ta_df, fred_df,
                                  encoder_len, predict_len)
    print(f"   Built feature matrix with shape: {feature_df.shape}")
    
    # Step 7: Create DataLoader with adaptive parameters
    print("7. Creating DataLoader...")
    try:
        dataloader, data_module = create_adaptive_dataloader(
            feature_df, encoder_len, predict_len, batch_size
        )
        print("✅ TFT data pipeline completed successfully!")
        
        # Print final cache info if available
        if cache and CACHING_AVAILABLE:
            print_cache_info()
        
        return dataloader
    except Exception as e:
        print(f"❌ Failed to create DataLoader: {e}")
        print("💡 Try using fewer symbols, longer date range, or smaller encoder/prediction lengths")
        raise


def get_data_loader_with_module(symbols: List[str], start: str, end: str,
        encoder_len: int, predict_len: int,
        batch_size: int,
        news_api_key: Optional[str] = None,
        fred_api_key: Optional[str] = None,
        api_ninjas_key: Optional[str] = None) -> Tuple[DataLoader, TFTDataModule]:
    """
    High-level entry point that returns both DataLoader and DataModule for detailed analysis.
    
    Args:
        symbols: List of stock symbols (e.g., ['AAPL', 'GOOGL'])
        start: Start date in 'YYYY-MM-DD' format
        end: End date in 'YYYY-MM-DD' format
        encoder_len: Number of historical time steps for encoder
        predict_len: Number of future time steps to predict
        batch_size: Batch size for DataLoader
        news_api_key: News API key for news embeddings (optional)
        fred_api_key: FRED API key for macroeconomic data (optional)
        api_ninjas_key: API-Ninjas key for earnings calendar (optional)
    
    Returns:
        Tuple of (DataLoader, TFTDataModule) for training and analysis
    """
    print(f"Starting TFT data pipeline for symbols: {symbols}")
    print(f"Date range: {start} to {end}")
    print(f"Encoder length: {encoder_len}, Predict length: {predict_len}")
    
    # Initialize caching system
    cache = None
    if CACHING_AVAILABLE:
        cache = get_cache_instance()
        print(f"📦 Caching enabled - checking for cached data...")
        print_cache_info()
    else:
        print(f"📦 Caching disabled - fetching fresh data...")
    
    # Step 1: Fetch stock data (with caching)
    print("1. Fetching stock data...")
    if cache:
        stock_df = cache.get_or_fetch_stock_data(symbols, start, end, fetch_stock_data)
    else:
        stock_df = fetch_stock_data(symbols, start, end)
    print(f"   Retrieved {len(stock_df)} stock data points")
    
    # Step 2: Fetch events data (with caching)
    print("2. Fetching corporate actions data...")
    if cache:
        events_data = cache.get_or_fetch_events_data(symbols, start, end, None, api_ninjas_key, fetch_events_data)
    else:
        events_data = fetch_events_data(symbols, start, end, None, api_ninjas_key)
    print(f"   Retrieved events for {len(events_data)} symbols")
    
    # Step 3: Fetch news embeddings (with caching)
    print("3. Fetching news embeddings...")
    print("   Note: BERT model will be cached on first use (~400MB)")
    print("   Pre-load with: python manage_models.py preload")
    if cache:
        news_df = cache.get_or_fetch_news_data(symbols, start, end, news_api_key, fetch_news_embeddings)
    else:
        news_df = fetch_news_embeddings(symbols, start, end, news_api_key)
    print(f"   Retrieved {len(news_df)} news embeddings")
    
    # Step 4: Fetch FRED macroeconomic data (with caching)
    print("4. Fetching FRED macroeconomic data...")
    if cache:
        fred_df = cache.get_or_fetch_fred_data(start, end, fred_api_key, fetch_fred_data)
    else:
        fred_df = fetch_fred_data(start, end, fred_api_key)
    print(f"   Retrieved {len(fred_df)} economic data points")
    
    # Step 5: Compute technical indicators (with caching)
    print("5. Computing technical indicators...")
    if cache:
        ta_df = cache.get_or_fetch_ta_data(stock_df, compute_technical_indicators)
    else:
        ta_df = compute_technical_indicators(stock_df)
    print(f"   Computed technical indicators for {len(ta_df)} data points")
    
    # Step 6: Build features (with caching)
    print("6. Building feature matrix...")
    if cache:
        feature_df = cache.get_or_build_features(stock_df, events_data, news_df, ta_df, fred_df,
                                               encoder_len, predict_len, build_features)
    else:
        feature_df = build_features(stock_df, events_data, news_df, ta_df, fred_df,
                                  encoder_len, predict_len)
    print(f"   Built feature matrix with shape: {feature_df.shape}")
    
    # Step 7: Create DataModule and DataLoader
    print("7. Creating DataLoader...")
    datamodule = TFTDataModule(
        feature_df=feature_df,
        encoder_len=encoder_len,
        predict_len=predict_len,
        batch_size=batch_size
    )
    
    datamodule.setup()
    dataloader = datamodule.train_dataloader()
    
    print("✅ TFT data pipeline completed successfully!")
    
    # Print final cache info if available
    if cache and CACHING_AVAILABLE:
        print_cache_info()
    
    return dataloader, datamodule



