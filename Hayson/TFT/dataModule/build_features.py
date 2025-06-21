"""
Feature building module for TFT.
Merges all data sources and creates TFT-ready feature matrix.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


def build_features(stock_df: pd.DataFrame,
                   events: Dict[str, Dict[str, Any]],
                   news_df: pd.DataFrame,
                   ta_df: pd.DataFrame,
                   fred_df: pd.DataFrame,
                   encoder_len: int,
                   predict_len: int) -> pd.DataFrame:
    """
    Merge all inputs and create TFT-ready feature matrix.
    
    Args:
        stock_df: Stock data with OHLCV
        events: Events data dictionary
        news_df: News embeddings DataFrame
        ta_df: Technical indicators DataFrame
        fred_df: FRED economic data DataFrame
        encoder_len: Encoder sequence length
        predict_len: Prediction sequence length
        
    Returns:
        DataFrame with:
        - time_idx: Sequential time index per symbol group
        - Static features: sector, market_cap, symbol ID
        - Known-future features: day_of_week, is_holiday, days_to_next_earnings, economic indicators
        - Past inputs: OHLCV, TA columns, news embeddings
        - Target: next-day return or price change
    """
    print("Building TFT-ready feature matrix...")
    
    if stock_df.empty:
        print("Warning: Empty stock DataFrame provided")
        return pd.DataFrame()
    
    # Start with technical indicators DataFrame (most complete)
    if not ta_df.empty:
        df = ta_df.copy()
        print(f"Starting with TA DataFrame: {df.shape}")
    else:
        df = stock_df.copy()
        print(f"Starting with stock DataFrame: {df.shape}")
    
    # Ensure date column is datetime
    df['date'] = pd.to_datetime(df['date'])
    
    # Sort by symbol and date
    df = df.sort_values(['symbol', 'date']).reset_index(drop=True)
    
    # Add time index per symbol group
    df = add_time_index(df)
    
    # Add calendar features
    df = add_calendar_features(df)
    
    # Add events-based features
    df = add_events_features(df, events, predict_len)
    
    # Merge news embeddings
    if not news_df.empty:
        df = merge_news_features(df, news_df)
        print(f"After news merge: {df.shape}")
    
    # Merge FRED economic data
    if not fred_df.empty:
        df = merge_fred_features(df, fred_df)
        print(f"After FRED merge: {df.shape}")
    
    # Add target variable (next-day return)
    df = add_target_variable(df)
    
    # Add static categorical and numerical features
    df = add_static_features(df, events)
    
    # Filter for minimum sequence length
    df = filter_minimum_sequence_length(df, encoder_len + predict_len)
    
    # Check if we still have data after filtering
    if df.empty:
        print("❌ No data remaining after filtering. Try:")
        print("   • Longer date range")
        print("   • Smaller encoder/prediction lengths")
        print("   • Different symbols")
        raise ValueError("Insufficient data after filtering for minimum sequence length")
    
    # Forward fill missing values
    df = handle_missing_values(df)
    
    # Final cleanup and validation
    df = final_cleanup(df)
    
    print(f"Final feature matrix shape: {df.shape}")
    print(f"Columns: {len(df.columns)}")
    print(f"Symbols: {df['symbol'].nunique()}")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    
    return df


def add_time_index(df: pd.DataFrame) -> pd.DataFrame:
    """Add sequential time index per symbol group."""
    df_with_time = df.copy()
    
    # Create time index for each symbol separately
    time_idx_data = []
    
    for symbol in df['symbol'].unique():
        symbol_df = df[df['symbol'] == symbol].copy()
        symbol_df = symbol_df.sort_values('date')
        
        # Create sequential time index starting from 0
        symbol_df['time_idx'] = range(len(symbol_df))
        time_idx_data.append(symbol_df)
    
    df_with_time = pd.concat(time_idx_data, ignore_index=True)
    df_with_time = df_with_time.sort_values(['symbol', 'date']).reset_index(drop=True)
    
    print(f"Added time_idx: {df_with_time['time_idx'].min()} to {df_with_time['time_idx'].max()}")
    return df_with_time


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add calendar-based features."""
    df_cal = df.copy()
    
    # Day of week (0=Monday, 6=Sunday)
    df_cal['day_of_week'] = df_cal['date'].dt.dayofweek
    
    # Month
    df_cal['month'] = df_cal['date'].dt.month
    
    # Quarter
    df_cal['quarter'] = df_cal['date'].dt.quarter
    
    # Day of month
    df_cal['day_of_month'] = df_cal['date'].dt.day
    
    # Is weekend
    df_cal['is_weekend'] = (df_cal['day_of_week'] >= 5).astype(int)
    
    # Basic market holidays (US)
    df_cal['is_holiday'] = 0  # Will be updated with events data
    
    print("Added calendar features")
    return df_cal


def add_events_features(df: pd.DataFrame, events: Dict[str, Dict[str, Any]], 
                       predict_len: int = 5) -> pd.DataFrame:
    """
    Add events-based features with proper relative timing for TFT prediction.
    
    Key insight: For time series prediction from i to i+k, we need to know:
    1. Is there an earnings event in the prediction window [i+1, i+k]?
    2. How many days until that event (relative timing)?
    3. EPS estimates/actuals for proper fundamental analysis
    
    Args:
        df: DataFrame with date and symbol columns
        events: Events dictionary from fetch_events_data  
        predict_len: Prediction length to determine prediction window
    """
    df_events = df.copy()
    
    # Initialize events columns with relative timing approach
    df_events['days_to_next_earnings'] = 999     # Relative time to next earnings (key for prediction)
    df_events['days_since_earnings'] = 999       # Relative time since last earnings  
    df_events['is_earnings_day'] = '0'             # String categorical for TFT
    df_events['earnings_in_prediction_window'] = '0'  # NEW: String categorical for TFT
    df_events['days_to_earnings_in_window'] = 999   # NEW: Relative days to earnings within prediction window
    df_events['eps_estimate'] = 0.0              # EPS estimate for upcoming earnings
    df_events['eps_actual'] = 0.0                # Actual EPS (if reported)
    df_events['revenue_estimate'] = 0.0          # Revenue estimate
    df_events['revenue_actual'] = 0.0            # Actual revenue
    df_events['days_to_next_split'] = 999
    df_events['is_split_day'] = '0'             # String categorical for TFT
    df_events['days_to_next_dividend'] = 999
    df_events['is_dividend_day'] = '0'          # String categorical for TFT
    
    if not events:
        print("No events data provided")
        return df_events
    
    for symbol in df_events['symbol'].unique():
        if symbol not in events:
            continue
        
        symbol_mask = df_events['symbol'] == symbol
        symbol_events = events[symbol]
        
        # Process earnings events with enhanced relative timing for prediction
        if 'earnings' in symbol_events and symbol_events['earnings']:
            earnings_dates = [pd.to_datetime(date) for date in symbol_events['earnings']]
            eps_data = symbol_events.get('eps_data', pd.DataFrame())
            
            for idx, row in df_events[symbol_mask].iterrows():
                current_date = pd.to_datetime(row['date'])
                # Normalize timezone for comparison
                if current_date.tz is not None:
                    current_date = current_date.tz_localize(None)
                
                # Define prediction window: [current_date + 1, current_date + predict_len]
                prediction_start = current_date + pd.Timedelta(days=1)
                prediction_end = current_date + pd.Timedelta(days=predict_len)
                
                # 1. Days to next earnings (overall)
                future_earnings = [d for d in earnings_dates if d > current_date]
                if future_earnings:
                    days_to_next = (min(future_earnings) - current_date).days
                    df_events.loc[idx, 'days_to_next_earnings'] = min(days_to_next, 999)
                    
                    # Add EPS estimate for next earnings
                    next_earnings_date = min(future_earnings)
                    if not eps_data.empty:
                        next_eps_row = eps_data[eps_data['date'].dt.date == next_earnings_date.date()]
                        if not next_eps_row.empty:
                            if 'epsEstimated' in next_eps_row.columns:
                                eps_est = next_eps_row['epsEstimated'].iloc[0]
                                if pd.notna(eps_est):
                                    df_events.loc[idx, 'eps_estimate'] = float(eps_est)
                            if 'revenueEstimated' in next_eps_row.columns:
                                rev_est = next_eps_row['revenueEstimated'].iloc[0]
                                if pd.notna(rev_est):
                                    df_events.loc[idx, 'revenue_estimate'] = float(rev_est)
                
                # 2. **KEY FEATURE**: Earnings in prediction window analysis
                earnings_in_window = [d for d in earnings_dates 
                                    if prediction_start <= d <= prediction_end]
                
                if earnings_in_window:
                    # Mark that there's an earnings event in the prediction window
                    df_events.loc[idx, 'earnings_in_prediction_window'] = '1'
                    
                    # Calculate relative days to earnings within prediction window
                    # This tells the model "earnings will happen in X days from prediction start"
                    closest_earnings = min(earnings_in_window)
                    days_to_earnings_in_window = (closest_earnings - prediction_start).days + 1
                    df_events.loc[idx, 'days_to_earnings_in_window'] = days_to_earnings_in_window
                
                # 3. Days since last earnings
                past_earnings = [d for d in earnings_dates if d <= current_date]
                if past_earnings:
                    days_since = (current_date - max(past_earnings)).days
                    df_events.loc[idx, 'days_since_earnings'] = min(days_since, 999)
                
                # 4. Is current day an earnings day - check for actual EPS data
                if current_date.date() in [d.date() for d in earnings_dates]:
                    df_events.loc[idx, 'is_earnings_day'] = '1'
                    
                    # Add actual EPS/revenue if available
                    if not eps_data.empty:
                        actual_eps_row = eps_data[eps_data['date'].dt.date == current_date.date()]
                        if not actual_eps_row.empty:
                            if 'eps' in actual_eps_row.columns:
                                eps_actual = actual_eps_row['eps'].iloc[0]
                                if pd.notna(eps_actual):
                                    df_events.loc[idx, 'eps_actual'] = float(eps_actual)
                            if 'revenue' in actual_eps_row.columns:
                                rev_actual = actual_eps_row['revenue'].iloc[0]
                                if pd.notna(rev_actual):
                                    df_events.loc[idx, 'revenue_actual'] = float(rev_actual)
        
        # Process splits
        if 'splits' in symbol_events and symbol_events['splits']:
            splits_dates = [pd.to_datetime(date) for date in symbol_events['splits']]
            
            for idx, row in df_events[symbol_mask].iterrows():
                current_date = pd.to_datetime(row['date'])
                # Normalize timezone for comparison
                if current_date.tz is not None:
                    current_date = current_date.tz_localize(None)
                
                future_splits = [d for d in splits_dates if d > current_date]
                if future_splits:
                    days_to_next = (min(future_splits) - current_date).days
                    df_events.loc[idx, 'days_to_next_split'] = min(days_to_next, 999)
                
                if current_date.date() in [d.date() for d in splits_dates]:
                    df_events.loc[idx, 'is_split_day'] = '1'
        
        # Process dividends
        if 'dividends' in symbol_events and symbol_events['dividends']:
            dividend_dates = [pd.to_datetime(date) for date in symbol_events['dividends']]
            
            for idx, row in df_events[symbol_mask].iterrows():
                current_date = pd.to_datetime(row['date'])
                # Normalize timezone for comparison
                if current_date.tz is not None:
                    current_date = current_date.tz_localize(None)
                
                future_dividends = [d for d in dividend_dates if d > current_date]
                if future_dividends:
                    days_to_next = (min(future_dividends) - current_date).days
                    df_events.loc[idx, 'days_to_next_dividend'] = min(days_to_next, 999)
                
                if current_date.date() in [d.date() for d in dividend_dates]:
                    df_events.loc[idx, 'is_dividend_day'] = '1'
        
        # Update holidays
        if 'holidays' in symbol_events and symbol_events['holidays']:
            holiday_dates = [pd.to_datetime(date).date() for date in symbol_events['holidays']]
            holiday_mask = df_events['date'].dt.date.isin(holiday_dates) & symbol_mask
            df_events.loc[holiday_mask, 'is_holiday'] = 1
    
    print("Added events features")
    return df_events


def merge_news_features(df: pd.DataFrame, news_df: pd.DataFrame) -> pd.DataFrame:
    """Merge news embeddings with main DataFrame."""
    if news_df.empty:
        return df
    
    # Ensure date columns are datetime and handle timezones
    df_copy = df.copy()
    news_copy = news_df.copy()
    
    # Convert to datetime and normalize timezones
    df_copy['date'] = pd.to_datetime(df_copy['date'])
    news_copy['date'] = pd.to_datetime(news_copy['date'])
    
    # Remove timezone info for consistent merging
    if df_copy['date'].dt.tz is not None:
        df_copy['date'] = df_copy['date'].dt.tz_localize(None)
    if news_copy['date'].dt.tz is not None:
        news_copy['date'] = news_copy['date'].dt.tz_localize(None)
    
    # Merge on symbol and date
    merged_df = pd.merge(
        df_copy, news_copy,
        on=['symbol', 'date'],
        how='left'
    )
    
    # Fill missing news embeddings with zeros
    news_cols = [col for col in news_copy.columns if col.startswith('emb_') or col == 'sentiment_score']
    for col in news_cols:
        if col in merged_df.columns:
            merged_df[col] = merged_df[col].fillna(0)
    
    return merged_df


def merge_fred_features(df: pd.DataFrame, fred_df: pd.DataFrame) -> pd.DataFrame:
    """Merge FRED economic data with main DataFrame."""
    if fred_df.empty:
        return df
    
    # Ensure date columns are datetime and handle timezones
    df_copy = df.copy()
    fred_copy = fred_df.copy()
    
    # Convert to datetime and normalize timezones
    df_copy['date'] = pd.to_datetime(df_copy['date'])
    fred_copy['date'] = pd.to_datetime(fred_copy['date'])
    
    # Remove timezone info for consistent merging
    if df_copy['date'].dt.tz is not None:
        df_copy['date'] = df_copy['date'].dt.tz_localize(None)
    if fred_copy['date'].dt.tz is not None:
        fred_copy['date'] = fred_copy['date'].dt.tz_localize(None)
    
    # Merge on date (economic data is same for all symbols on each date)
    merged_df = pd.merge(
        df_copy, fred_copy,
        on='date',
        how='left'
    )
    
    # Fill missing economic data with forward fill
    from .fetch_fred import get_economic_features
    econ_cols = get_economic_features()
    for col in econ_cols:
        if col in merged_df.columns:
            merged_df[col] = merged_df[col].ffill().fillna(0)
    
    return merged_df


def add_target_variable(df: pd.DataFrame) -> pd.DataFrame:
    """Add target variable (next-day return)."""
    df_target = df.copy()
    
    # Calculate target for each symbol separately
    target_data = []
    
    for symbol in df['symbol'].unique():
        symbol_df = df[df['symbol'] == symbol].copy()
        symbol_df = symbol_df.sort_values('date')
        
        # Next-day return (percentage change)
        symbol_df['target'] = symbol_df['close'].pct_change().shift(-1)
        
        # Alternative targets
        symbol_df['target_price'] = symbol_df['close'].shift(-1)
        symbol_df['target_direction'] = (symbol_df['target'] > 0).astype(int)
        
        target_data.append(symbol_df)
    
    df_target = pd.concat(target_data, ignore_index=True)
    df_target = df_target.sort_values(['symbol', 'date']).reset_index(drop=True)
    
    # Remove rows with NaN targets (last day for each symbol)
    df_target = df_target.dropna(subset=['target'])
    
    print(f"Added target variable. Rows with valid targets: {len(df_target)}")
    return df_target


def add_static_features(df: pd.DataFrame, events: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    """Add static categorical and numerical features."""
    df_static = df.copy()
    
    # Add symbol ID (categorical encoding)
    symbols = sorted(df_static['symbol'].unique())
    symbol_to_id = {symbol: idx for idx, symbol in enumerate(symbols)}
    df_static['symbol_id'] = df_static['symbol'].map(symbol_to_id)
    
    # Add sector and market cap from events data
    df_static['sector'] = 'Unknown'
    df_static['market_cap'] = 0.0
    
    for symbol in symbols:
        if symbol in events:
            symbol_mask = df_static['symbol'] == symbol
            df_static.loc[symbol_mask, 'sector'] = events[symbol].get('sector', 'Unknown')
            df_static.loc[symbol_mask, 'market_cap'] = events[symbol].get('market_cap', 0.0)
    
    # Encode sectors as categorical IDs
    sectors = sorted(df_static['sector'].unique())
    sector_to_id = {sector: idx for idx, sector in enumerate(sectors)}
    df_static['sector_id'] = df_static['sector'].map(sector_to_id)
    
    print(f"Added static features. Sectors: {sectors}")
    return df_static


def filter_minimum_sequence_length(df: pd.DataFrame, min_length: int) -> pd.DataFrame:
    """Filter symbols with insufficient sequence length, with adaptive handling for small datasets."""
    filtered_data = []
    total_symbols = df['symbol'].nunique()
    
    for symbol in df['symbol'].unique():
        symbol_df = df[df['symbol'] == symbol]
        
        if len(symbol_df) >= min_length:
            filtered_data.append(symbol_df)
        else:
            print(f"Filtering out {symbol}: only {len(symbol_df)} data points (need {min_length})")
    
    if filtered_data:
        result_df = pd.concat(filtered_data, ignore_index=True)
        result_df = result_df.sort_values(['symbol', 'date']).reset_index(drop=True)
        return result_df
    else:
        print("⚠️  Warning: No symbols meet minimum sequence length requirement")
        print(f"🔧 Adaptive mode: Relaxing requirements for small dataset")
        
        # Adaptive fallback: use the symbol with the most data
        best_symbol = None
        max_points = 0
        
        for symbol in df['symbol'].unique():
            symbol_df = df[df['symbol'] == symbol]
            if len(symbol_df) > max_points:
                max_points = len(symbol_df)
                best_symbol = symbol
        
        if best_symbol and max_points >= 3:  # Minimum viable sequence
            print(f"   Using {best_symbol} with {max_points} data points")
            result_df = df[df['symbol'] == best_symbol].copy()
            return result_df.sort_values(['symbol', 'date']).reset_index(drop=True)
        else:
            print("❌ Insufficient data for any symbol")
            return pd.DataFrame()


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Handle missing values in the feature matrix."""
    if df.empty:
        print("⚠️  Empty DataFrame passed to handle_missing_values")
        return df
        
    df_filled = df.copy()
    
    # Get numeric columns (excluding categorical and datetime)
    exclude_cols = ['symbol', 'date', 'sector']
    numeric_cols = df_filled.select_dtypes(include=[np.number]).columns
    numeric_cols = [col for col in numeric_cols if col not in exclude_cols]
    
    # Forward fill within each symbol group
    if 'symbol' in df_filled.columns:
        for symbol in df_filled['symbol'].unique():
            symbol_mask = df_filled['symbol'] == symbol
            
            # Forward fill, then backward fill
            df_filled.loc[symbol_mask, numeric_cols] = (
                df_filled.loc[symbol_mask, numeric_cols]
                .ffill()
                .bfill()
            )
            
            # Fill any remaining NaN with 0
            df_filled.loc[symbol_mask, numeric_cols] = (
                df_filled.loc[symbol_mask, numeric_cols].fillna(0)
            )
    else:
        # If no symbol column (shouldn't happen but be safe)
        df_filled[numeric_cols] = df_filled[numeric_cols].ffill().bfill().fillna(0)
    
    return df_filled


def final_cleanup(df: pd.DataFrame) -> pd.DataFrame:
    """Final cleanup and validation of feature matrix."""
    df_clean = df.copy()
    
    # Convert categorical features to string type for TFT
    categorical_features = [
        'is_earnings_day', 'is_split_day', 'is_dividend_day', 
        'is_holiday', 'is_weekend'
    ]
    
    for col in categorical_features:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].astype(str)
    
    # Ensure symbol and sector are strings
    if 'symbol' in df_clean.columns:
        df_clean['symbol'] = df_clean['symbol'].astype(str)
    if 'sector' in df_clean.columns:
        df_clean['sector'] = df_clean['sector'].astype(str)
    
    # Ensure no infinite values
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    df_clean[numeric_cols] = df_clean[numeric_cols].replace([np.inf, -np.inf], np.nan)
    df_clean[numeric_cols] = df_clean[numeric_cols].fillna(0)
    
    # Ensure all required columns exist
    required_cols = ['symbol', 'date', 'time_idx', 'target']
    missing_cols = [col for col in required_cols if col not in df_clean.columns]
    
    if missing_cols:
        print(f"Warning: Missing required columns: {missing_cols}")
    
    # Sort final result
    df_clean = df_clean.sort_values(['symbol', 'time_idx']).reset_index(drop=True)
    
    return df_clean
