#!/usr/bin/env python3
"""
Quick test script to validate feature filtering with realistic feature names
"""
import sys
import os
sys.path.append('/Users/haysoncheung/programs/pythonProject/TFT-b-nelson/Comparision')

from run_all_models import filter_features_by_type
import numpy as np
import pandas as pd

# Create a realistic feature DataFrame based on the actual system
def create_realistic_mock_data():
    """Create mock data that closely matches the real system's feature structure."""
    
    # Based on the output, we know there are ~824 features with 769 being news embeddings
    # News embeddings: emb_0 to emb_767 (768 dimensions) + sentiment_score = 769 features
    # Remaining: 824 - 769 = 55 other features
    
    mock_columns = ['date', 'symbol']  # metadata
    
    # Add news features (768 embedding dimensions + sentiment)
    news_features = [f'emb_{i}' for i in range(768)] + ['sentiment_score']
    mock_columns.extend(news_features)
    
    # Add price features (6 features)
    price_features = ['open', 'high', 'low', 'close', 'volume', 'adjusted_close']
    mock_columns.extend(price_features)
    
    # Add technical indicators (~30 features)
    tech_features = [
        'ta_sma_5', 'ta_sma_10', 'ta_sma_20', 'ta_sma_50', 'ta_sma_200',
        'ta_ema_12', 'ta_ema_26', 'ta_ema_50',
        'ta_rsi_14', 'ta_rsi_21',
        'ta_macd', 'ta_macd_signal', 'ta_macd_histogram',
        'ta_bb_upper', 'ta_bb_middle', 'ta_bb_lower', 'ta_bb_width',
        'ta_atr_14', 'ta_atr_21',
        'ta_obv', 'ta_ad_line',
        'ta_stoch_k', 'ta_stoch_d',
        'ta_williams_r', 'ta_cci',
        'ta_momentum_10', 'ta_momentum_20',
        'ta_roc_10', 'ta_roc_20',
        'ta_trix'
    ]
    mock_columns.extend(tech_features)
    
    # Add economic features (~8 features)
    econ_features = ['cpi', 'fedfunds', 'unrate', 't10y2y', 'gdp', 'vix', 'dxy', 'oil']
    mock_columns.extend(econ_features)
    
    # Add some other features to match the total
    other_features = [
        'market_cap', 'sector', 'day_of_week', 'month', 'quarter', 
        'is_month_end', 'is_quarter_end', 'trading_volume_ratio',
        'price_change_1d', 'price_change_5d', 'volatility_10d'
    ]
    mock_columns.extend(other_features)
    
    # Add target columns
    targets = [f'target_{i}' for i in range(10)]
    mock_columns.extend(targets)
    
    print(f"📊 MOCK DATA STRUCTURE:")
    print(f"   Total columns: {len(mock_columns)}")
    print(f"   News features: {len(news_features)} (emb_0 to emb_767 + sentiment_score)")
    print(f"   Price features: {len(price_features)}")
    print(f"   Technical features: {len(tech_features)}")
    print(f"   Economic features: {len(econ_features)}")
    print(f"   Other features: {len(other_features)}")
    print(f"   Metadata + Targets: {len(['date', 'symbol'] + targets)}")
    
    # Create mock DataFrame
    mock_df = pd.DataFrame(columns=mock_columns)
    
    # Create mock tensor data with same number of features (excluding metadata and targets)
    feature_cols = [col for col in mock_columns 
                   if col not in ['date', 'symbol'] + targets]
    
    expected_tensor_features = len(feature_cols)
    print(f"\n   Expected tensor features: {expected_tensor_features}")
    
    # Create mock tensor data (simulate the discrepancy we see in real data)
    # Real data has 824 features but DataFrame has 830 columns (6 column difference)
    actual_tensor_features = expected_tensor_features - 6  # Simulate the discrepancy
    mock_tensor = np.random.rand(16, actual_tensor_features)
    
    print(f"   Actual tensor shape: {mock_tensor.shape}")
    print(f"   Discrepancy: {expected_tensor_features - actual_tensor_features} features")
    
    return mock_tensor, mock_df, feature_cols

def test_detailed_filtering():
    print("🧪 DETAILED FEATURE FILTERING TEST")
    print("=" * 80)
    
    # Create realistic mock data
    mock_tensor, mock_df, feature_cols = create_realistic_mock_data()
    
    print(f"\n📋 TESTING FILTERS:")
    print("=" * 50)
    
    # Test no_news filtering (this should remove ~769 features)
    print(f"\n1️⃣  TESTING 'no_news' FILTERING:")
    print("-" * 40)
    filtered_data, indices = filter_features_by_type(mock_tensor, mock_df, "no_news", news_dim=0)
    print(f"   Result: {mock_tensor.shape} → {filtered_data.shape}")
    
    # Test price_only filtering (should keep ~6 features)
    print(f"\n2️⃣  TESTING 'price_only' FILTERING:")
    print("-" * 40)
    filtered_data, indices = filter_features_by_type(mock_tensor, mock_df, "price_only", news_dim=0)
    print(f"   Result: {mock_tensor.shape} → {filtered_data.shape}")
    
    # Test technical_only filtering
    print(f"\n3️⃣  TESTING 'technical_only' FILTERING:")
    print("-" * 40)
    filtered_data, indices = filter_features_by_type(mock_tensor, mock_df, "technical_only", news_dim=0)
    print(f"   Result: {mock_tensor.shape} → {filtered_data.shape}")
    
    # Test price_technical filtering
    print(f"\n4️⃣  TESTING 'price_technical' FILTERING:")
    print("-" * 40)
    filtered_data, indices = filter_features_by_type(mock_tensor, mock_df, "price_technical", news_dim=0)
    print(f"   Result: {mock_tensor.shape} → {filtered_data.shape}")
    
    # Test all features
    print(f"\n5️⃣  TESTING 'all' FILTERING:")
    print("-" * 40)
    filtered_data, indices = filter_features_by_type(mock_tensor, mock_df, "all", news_dim=0)
    print(f"   Result: {mock_tensor.shape} → {filtered_data.shape}")
    
    print("\n" + "=" * 80)
    print("✅ FILTERING TEST COMPLETE")
    print("\n💡 INTERPRETATION:")
    print("   • no_news: Removes news embeddings (768) + sentiment (1) = 769 features")
    print("   • price_only: Keeps only OHLCV + adjusted_close = 6 features")
    print("   • technical_only: Keeps only technical indicators (~30 features)")
    print("   • price_technical: Keeps price + technical (~36 features)")
    print("   • all: Keeps all available features")
    print("=" * 80)

if __name__ == '__main__':
    test_detailed_filtering()
