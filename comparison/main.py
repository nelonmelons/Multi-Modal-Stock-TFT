#!/usr/bin/env python3
"""
Leakage-Free Stock Data Loading and Analysis System

This script demonstrates a proper temporal data pipeline for stock prediction that:
1. Prevents data leakage by enforcing strict temporal boundaries
2. Integrates multiple data sources (stock prices, events, FRED economic data)
3. Creates proper train/validation splits with lookahead buffer
4. Generates comprehensive data tables for analysis

Key Anti-Leakage Measures:
- Temporal split with lookahead buffer to prevent future information contamination
- Separate data processing for training and validation periods
- Feature engineering respects temporal constraints
- No future data used in historical feature computation
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Ensure local dataModule package is discoverable
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import directly from specific modules to avoid circular imports
from dataModule.datamodule import NumericDataModule
from dataModule.fetch_stock import fetch_stock_data
from dataModule.fetch_events import fetch_events_data
from dataModule.fetch_fred import fetch_fred_data
from dataModule.compute_ta import compute_technical_indicators
from dataModule.build_features import build_features
from cache_manager import DataCache

# New: universe and fixed windows
try:
    from src.universe import DOW30_2018 as DOW_UNIVERSE
except Exception:
    DOW_UNIVERSE = None

# Fixed experiment windows
TRAIN_START = '2016-01-01'
TRAIN_END = '2019-12-31'
VAL_START = '2020-01-01'
VAL_END = '2020-12-31'
TEST_START = '2021-01-01'
TEST_END = '2024-12-31'
HORIZONS = [1, 5, 21]
EMBARGO_DAYS = 5


class LeakageFreeDataLoader:
    """
    A comprehensive data loading system that prevents temporal data leakage
    for multi-modal stock prediction models.
    """
    
    def __init__(self, config: dict):
        """
        Initialize the data loader with configuration parameters.
        """
        self.config = config
        self.horizons = self.config.get('horizons', HORIZONS)
        self.validate_config()
        self.cache = DataCache(cache_dir="cache")
        self.setup_temporal_boundaries()
        self.raw_data = {}
        self.processed_data = {}
        self.data_module = None
        self.val_df = None
        self.test_df = None
    
    def validate_config(self):
        """Validate configuration against global experiment contract."""
        required_keys = [
            'symbols', 'start_date', 'end_date', 'encoder_len', 'predict_len',
            'batch_size'
        ]
        for key in required_keys:
            if key not in self.config:
                raise ValueError(f"Missing required configuration key: {key}")
        # Dates
        try:
            datetime.strptime(self.config['start_date'], '%Y-%m-%d')
            datetime.strptime(self.config['end_date'], '%Y-%m-%d')
        except ValueError:
            raise ValueError("Dates must be in 'YYYY-MM-DD' format")
        print("✅ Configuration validated successfully")
    
    def setup_temporal_boundaries(self):
        """Use Train/Val/Test windows from config with embargo/purge guard awareness."""
        # Use config values if available, otherwise fall back to hardcoded constants
        self.train_start = self.config.get('train_start', TRAIN_START)
        self.train_end = self.config.get('train_end', TRAIN_END)
        self.val_start = self.config.get('val_start', VAL_START)
        self.val_end = self.config.get('val_end', VAL_END)
        self.test_start = self.config.get('test_start', TEST_START)
        self.test_end = self.config.get('test_end', TEST_END)
        print("🔒 Temporal Boundaries (From Config):")
        print(f"   Train: {self.train_start} → {self.train_end}")
        print(f"   Val:   {self.val_start} → {self.val_end}")
        print(f"   Test:  {self.test_start} → {self.test_end}")
        print(f"   Embargo: {EMBARGO_DAYS} trading days; Purge: up to max(h)={max(self.horizons)} days")
        print()
    
    def fetch_all_data_sources(self):
        """Fetch data from all available sources for the complete time period."""
        print("📊 Fetching Multi-Modal Data Sources...")
        try:
            # 1. Stock prices
            print("1️⃣ Fetching stock price data...")
            self.raw_data['stock'] = self.cache.get_or_fetch_stock_data(
                symbols=self.config['symbols'],
                start=self.config['start_date'],
                end=self.config['end_date'],
                fetch_func=lambda symbols, start, end: fetch_stock_data(symbols, start, end)
            )
            print(f"   Stock data shape: {self.raw_data['stock'].shape}")
            # 2. Events
            print("2️⃣ Fetching corporate events data...")
            self.raw_data['events'] = self.cache.get_or_fetch_events_data(
                symbols=self.config['symbols'],
                start=self.config['start_date'], 
                end=self.config['end_date'],
                earnings_api_key=None,
                api_ninjas_key=self.config.get('api_ninjas_key'),
                fetch_func=lambda symbols, start, end, earnings_key, ninjas_key: fetch_events_data(symbols, start, end, None, ninjas_key)
            )
            print(f"   Events data fetched")
            # 3. FRED
            print("3️⃣ Fetching FRED economic indicators...")
            self.raw_data['fred'] = self.cache.get_or_fetch_fred_data(
                start=self.config['start_date'], 
                end=self.config['end_date'],
                api_key=self.config.get('fred_api_key'),
                fetch_func=lambda start, end, api_key: fetch_fred_data(start, end, api_key)
            )
            print(f"   FRED data shape: {self.raw_data['fred'].shape}")
            # 4. Technicals
            print("4️⃣ Computing technical indicators...")
            self.raw_data['technical'] = self.cache.get_or_fetch_ta_data(
                stock_df=self.raw_data['stock'],
                fetch_func=lambda stock_df: compute_technical_indicators(stock_df)
            )
            print(f"   Technical indicators shape: {self.raw_data['technical'].shape}")
            print("✅ All data sources fetched successfully\n")
        except Exception as e:
            print(f"❌ Error fetching data: {str(e)}")
            raise
    
    def build_leakage_free_features(self):
        """Build features with strict temporal constraints."""
        print("🔧 Building Temporal-Safe Feature Matrix...")
        try:
            # Create empty news DataFrame for compatibility
            empty_news_df = pd.DataFrame()
            
            self.processed_data['features'] = self.cache.get_or_build_features(
                stock_df=self.raw_data['stock'],
                events=self.raw_data['events'],
                news_df=empty_news_df,
                ta_df=self.raw_data['technical'],
                fred_df=self.raw_data['fred'],
                encoder_len=self.config['encoder_len'],
                predict_len=self.config['predict_len'],
                build_func=lambda stock_df, events, news_df, ta_df, fred_df, encoder_len, predict_len: build_features(
                    stock_df=stock_df,
                    events=events,
                    news_df=news_df,
                    ta_df=ta_df,
                    fred_df=fred_df,
                    encoder_len=encoder_len,
                    predict_len=predict_len,
                    split_date=self.train_end
                )
            )
            print(f"✅ Feature matrix built: {self.processed_data['features'].shape}")
            print(f"   Date range: {self.processed_data['features']['date'].min()} → {self.processed_data['features']['date'].max()}")
            self.validate_temporal_integrity()
        except Exception as e:
            print(f"❌ Error building features: {str(e)}")
            raise
    
    def validate_temporal_integrity(self):
        """Basic temporal checks."""
        print("🔍 Validating Temporal Integrity...")
        df = self.processed_data['features']
        # Boundaries check
        expected_start = pd.to_datetime(self.config['start_date'])
        expected_end = pd.to_datetime(self.config['end_date'])
        if pd.to_datetime(df['date'].min()) < expected_start or pd.to_datetime(df['date'].max()) > expected_end:
            print("⚠️  Warning: Dates outside expected range detected")
        print("✅ Temporal integrity validation completed")
    
    def setup_data_module(self):
        """Initialize NumericDataModule with hard date splits, purge, embargo."""
        print("📦 Setting up Data Module with hard splits + leakage guards...")
        try:
            dm = NumericDataModule(
                feature_df=self.processed_data['features'],
                batch_size=self.config['batch_size'],
                date_col='date',
                target_col='target_',
                train_range=(self.train_start, self.train_end),
                val_range=(self.val_start, self.val_end),
                test_range=(self.test_start, self.test_end),
                embargo_days=EMBARGO_DAYS,
                horizons=self.horizons,
            )
            dm.setup()
            self.data_module = dm
            self.val_df = dm.val_df
            self.test_df = dm.test_df
            print("✅ Data module setup completed\n")
        except Exception as e:
            print(f"❌ Error setting up data module: {str(e)}")
            raise
    
    def load_complete_pipeline(self):
        """Execute complete pipeline."""
        print("🚀 Starting Leakage-Free Data Loading Pipeline...\n")
        self.fetch_all_data_sources()
        self.build_leakage_free_features()
        self.setup_data_module()
        print("🎯 Pipeline completed successfully!")
        return self.data_module


def main():
    print("🎯 Multi-Modal Stock Data Pipeline with Leakage Prevention")
    print("=" * 80)
    print()
    config = {
        'symbols': DOW_UNIVERSE or ['AAPL', 'MSFT'],
        'start_date': TRAIN_START,
        'end_date': TEST_END,
        'encoder_len': 60,
        'predict_len': 21,
        'batch_size': 256,
        'fred_api_key': os.getenv('FRED_API_KEY'),
        'api_ninjas_key': os.getenv('API_NINJAS_KEY'),
        'horizons': HORIZONS,
    }
    try:
        data_loader = LeakageFreeDataLoader(config)
        _ = data_loader.load_complete_pipeline()
        print("🎉 Data loading pipeline completed successfully!")
        return _
    except Exception as e:
        print(f"❌ Pipeline failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == '__main__':
    main()