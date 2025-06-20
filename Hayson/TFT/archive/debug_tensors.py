#!/usr/bin/env python3
"""
Debug script to trace tensor creation and identify None values in TFT DataLoader.
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add project root to path
sys.path.append('/Users/haysoncheung/programs/pythonProject/TFT-mm')

from dotenv import load_dotenv
load_dotenv()

def debug_feature_dataframe():
    """Debug the feature DataFrame before it goes into TimeSeriesDataSet."""
    print("🔍 DEBUGGING FEATURE DATAFRAME CREATION")
    print("=" * 50)
    
    # Import pipeline components
    from data import (
        fetch_stock_data, fetch_events_data, fetch_news_embeddings,
        fetch_fred_data, compute_technical_indicators, build_features
    )
    
    # Use same parameters as failing test
    symbols = ['AAPL']
    start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
    end_date = datetime.now().strftime('%Y-%m-%d')
    encoder_len = 8
    predict_len = 3
    
    print(f"Date range: {start_date} to {end_date}")
    print(f"Encoder: {encoder_len}, Predict: {predict_len}")
    
    try:
        # Step 1: Fetch stock data
        print("\n1. Fetching stock data...")
        stock_df = fetch_stock_data(symbols, start_date, end_date)
        print(f"   Stock data shape: {stock_df.shape}")
        print(f"   Stock columns: {list(stock_df.columns)}")
        print(f"   Stock dtypes:\n{stock_df.dtypes}")
        
        # Check for NaN/None values
        nan_cols = stock_df.isnull().sum()
        if nan_cols.sum() > 0:
            print(f"   ⚠️  NaN values found: {nan_cols[nan_cols > 0].to_dict()}")
        
        # Step 2: Events data
        print("\n2. Fetching events data...")
        events_data = fetch_events_data(symbols, start_date, end_date, None)
        print(f"   Events data type: {type(events_data)}")
        
        # Step 3: News embeddings
        print("\n3. Fetching news embeddings...")
        news_df = fetch_news_embeddings(symbols, start_date, end_date, os.getenv('NEWS_API_KEY'))
        print(f"   News data shape: {news_df.shape}")
        print(f"   News columns: {list(news_df.columns)[:10]}...")  # First 10 columns
        
        # Check news embeddings for issues
        if not news_df.empty:
            embedding_cols = [col for col in news_df.columns if col.startswith('emb_')]
            if embedding_cols:
                sample_embedding = news_df[embedding_cols].iloc[0]
                print(f"   Sample embedding stats: min={sample_embedding.min():.3f}, max={sample_embedding.max():.3f}")
                
                # Check for any None/NaN in embeddings
                nan_embeddings = news_df[embedding_cols].isnull().sum().sum()
                inf_embeddings = np.isinf(news_df[embedding_cols]).sum().sum()
                print(f"   NaN embeddings: {nan_embeddings}, Inf embeddings: {inf_embeddings}")
        
        # Step 4: FRED data
        print("\n4. Fetching FRED data...")
        fred_df = fetch_fred_data(start_date, end_date, os.getenv('FRED_API_KEY'))
        print(f"   FRED data shape: {fred_df.shape}")
        
        # Step 5: Technical indicators
        print("\n5. Computing technical indicators...")
        ta_df = compute_technical_indicators(stock_df)
        print(f"   TA data shape: {ta_df.shape}")
        
        # Step 6: Build features
        print("\n6. Building feature matrix...")
        feature_df = build_features(stock_df, events_data, news_df, ta_df, fred_df, encoder_len, predict_len)
        print(f"   Feature matrix shape: {feature_df.shape}")
        
        # DETAILED FEATURE ANALYSIS
        print("\n📊 DETAILED FEATURE ANALYSIS:")
        print(f"   Columns: {len(feature_df.columns)}")
        print(f"   Data types:\n{feature_df.dtypes.value_counts()}")
        
        # Check for problematic values
        print("\n🔍 CHECKING FOR PROBLEMATIC VALUES:")
        
        # Check for None/NaN
        nan_summary = feature_df.isnull().sum()
        nan_columns = nan_summary[nan_summary > 0]
        if len(nan_columns) > 0:
            print(f"   ❌ NaN values found in {len(nan_columns)} columns:")
            for col, count in nan_columns.head(10).items():
                print(f"     {col}: {count} NaN values")
        else:
            print("   ✅ No NaN values found")
        
        # Check for infinite values
        numeric_cols = feature_df.select_dtypes(include=[np.number]).columns
        inf_summary = np.isinf(feature_df[numeric_cols]).sum()
        inf_columns = inf_summary[inf_summary > 0]
        if len(inf_columns) > 0:
            print(f"   ❌ Infinite values found in {len(inf_columns)} columns:")
            for col, count in inf_columns.head(10).items():
                print(f"     {col}: {count} infinite values")
        else:
            print("   ✅ No infinite values found")
        
        # Check required columns for TFT
        required_cols = ['symbol', 'date', 'time_idx', 'target']
        missing_required = [col for col in required_cols if col not in feature_df.columns]
        if missing_required:
            print(f"   ❌ Missing required columns: {missing_required}")
        else:
            print(f"   ✅ All required columns present: {required_cols}")
        
        # Sample data inspection
        print("\n📋 SAMPLE DATA INSPECTION:")
        print(f"   First row sample:")
        sample_row = feature_df.iloc[0]
        for col in ['symbol', 'date', 'time_idx', 'target', 'open', 'close']:
            if col in feature_df.columns:
                print(f"     {col}: {sample_row[col]}")
        
        return feature_df
        
    except Exception as e:
        print(f"❌ Error in feature creation: {e}")
        import traceback
        traceback.print_exc()
        return None

def debug_timeseries_dataset(feature_df):
    """Debug TimeSeriesDataSet creation."""
    print("\n🔍 DEBUGGING TIMESERIESDATASET CREATION")
    print("=" * 50)
    
    try:
        from pytorch_forecasting import TimeSeriesDataSet, GroupNormalizer
        
        # Use same parameters as DataModule
        encoder_len = 8
        predict_len = 3
        
        # Categorize features (simplified version)
        static_categoricals = ['symbol']
        static_reals = ['market_cap'] if 'market_cap' in feature_df.columns else []
        time_varying_known_categoricals = [col for col in ['is_earnings_day', 'is_split_day', 'is_dividend_day'] if col in feature_df.columns]
        time_varying_known_reals = ['time_idx'] + [col for col in ['day_of_week', 'month'] if col in feature_df.columns]
        
        # Get unknown reals (everything else numeric)
        exclude_cols = set(['symbol', 'date', 'target'] + static_categoricals + static_reals + 
                          time_varying_known_categoricals + time_varying_known_reals)
        time_varying_unknown_reals = [
            col for col in feature_df.columns
            if col not in exclude_cols and 
            feature_df[col].dtype in ['float64', 'float32', 'int64', 'int32']
        ]
        
        print(f"Feature categorization:")
        print(f"  Static categoricals: {len(static_categoricals)}")
        print(f"  Static reals: {len(static_reals)}")
        print(f"  Known categoricals: {len(time_varying_known_categoricals)}")
        print(f"  Known reals: {len(time_varying_known_reals)}")
        print(f"  Unknown reals: {len(time_varying_unknown_reals)}")
        
        # Check for issues in categorical columns
        print("\n🔍 CHECKING CATEGORICAL COLUMNS:")
        for col in static_categoricals + time_varying_known_categoricals:
            if col in feature_df.columns:
                unique_vals = feature_df[col].unique()
                null_count = feature_df[col].isnull().sum()
                print(f"  {col}: {len(unique_vals)} unique values, {null_count} nulls")
                if null_count > 0:
                    print(f"    ❌ NULL values in categorical column {col}")
        
        # Create TimeSeriesDataSet
        print("\n📊 CREATING TIMESERIESDATASET:")
        dataset = TimeSeriesDataSet(
            feature_df,
            time_idx="time_idx",
            target="target",
            group_ids=["symbol"],
            min_encoder_length=max(1, encoder_len // 4),
            max_encoder_length=encoder_len,
            min_prediction_length=1,
            max_prediction_length=predict_len,
            static_categoricals=static_categoricals,
            static_reals=static_reals,
            time_varying_known_categoricals=time_varying_known_categoricals,
            time_varying_known_reals=time_varying_known_reals,
            time_varying_unknown_categoricals=[],
            time_varying_unknown_reals=time_varying_unknown_reals,
            target_normalizer=GroupNormalizer(groups=["symbol"], transformation="softplus"),
            add_relative_time_idx=True,
            add_target_scales=True,
            add_encoder_length=True,
            allow_missing_timesteps=True
        )
        
        print(f"✅ TimeSeriesDataSet created successfully")
        print(f"   Dataset length: {len(dataset)}")
        
        # Test single sample
        print("\n🔍 TESTING SINGLE SAMPLE:")
        sample = dataset[0]
        print(f"   Sample type: {type(sample)}")
        print(f"   Sample keys: {list(sample.keys())}")
        
        for key, value in sample.items():
            if value is None:
                print(f"   ❌ {key}: None value found!")
            elif hasattr(value, 'shape'):
                print(f"   ✅ {key}: {value.shape} - {value.dtype}")
                if np.isnan(value).any():
                    print(f"     ⚠️  Contains NaN values")
                if np.isinf(value).any():
                    print(f"     ⚠️  Contains infinite values")
            else:
                print(f"   ? {key}: {type(value)} - {value}")
        
        return dataset
        
    except Exception as e:
        print(f"❌ Error creating TimeSeriesDataSet: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    # Debug feature creation
    feature_df = debug_feature_dataframe()
    
    if feature_df is not None:
        # Debug TimeSeriesDataSet
        dataset = debug_timeseries_dataset(feature_df)
        
        if dataset is not None:
            print("\n🎯 DEBUGGING COMPLETE - Dataset created successfully")
        else:
            print("\n❌ DEBUGGING FAILED - Dataset creation failed")
    else:
        print("\n❌ DEBUGGING FAILED - Feature creation failed")
