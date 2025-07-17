"""
Technical indicators computation module.
Computes various technical analysis indicators for stock data.
Uses pandas-ta as the primary library for technical indicators.
"""

import pandas as pd
import numpy as np
from typing import List
import warnings
warnings.filterwarnings('ignore')

# Try to import pandas_ta for advanced indicators
try:
    import pandas_ta as ta
    HAS_PANDAS_TA = True
except ImportError:
    HAS_PANDAS_TA = False
    print("Warning: pandas_ta not installed. Using basic indicators only.")
    print("Install with: pip install pandas-ta")


def compute_technical_indicators(df: pd.DataFrame, split_date: str = None, 
                                is_training: bool = True) -> pd.DataFrame:
    """
    Compute technical indicators for stock data WITHOUT FUTURE INFORMATION LEAKAGE.
    
    Args:
        df: Stock DataFrame from fetch_stock_data with OHLCV columns
        split_date: Date to split training/validation (ISO format 'YYYY-MM-DD')
        is_training: If True, compute on training data only. If False, use expanding window.
        
    Returns:
        DataFrame with additional technical indicator columns computed with proper temporal constraints
    """
    print(f"Computing technical indicators (temporal-aware, is_training={is_training})...")
    
    if df.empty:
        print("Warning: Empty DataFrame provided")
        return df
    
    # Copy input DataFrame to avoid modifying original
    result_df = df.copy()
    result_df['date'] = pd.to_datetime(result_df['date'])
    
    # Process each symbol separately with proper temporal constraints
    symbols = result_df['symbol'].unique()
    all_symbol_data = []
    
    for symbol in symbols:
        print(f"  Computing indicators for {symbol} (temporal-aware)...")
        
        # Filter data for this symbol
        symbol_df = result_df[result_df['symbol'] == symbol].copy()
        symbol_df = symbol_df.sort_values('date').reset_index(drop=True)
        
        if len(symbol_df) < 50:  # Need sufficient data for indicators
            print(f"    Warning: Insufficient data for {symbol} ({len(symbol_df)} rows)")
        
        # CRITICAL FIX: Compute indicators with expanding window to prevent future leakage
        if split_date and not is_training:
            # For validation/test data, use expanding window from training period
            split_dt = pd.to_datetime(split_date)
            train_mask = symbol_df['date'] <= split_dt
            
            if train_mask.sum() < 10:
                print(f"    Warning: Insufficient training data for {symbol}, using minimal window")
                min_window = max(1, train_mask.sum() // 2)
            else:
                min_window = 10
                
            symbol_df = compute_indicators_expanding_window(symbol_df, train_mask, min_window)
        else:
            # For training data or when no split specified, use standard rolling windows
            # but ensure we don't use future information
            symbol_df = compute_indicators_standard(symbol_df)
        
        all_symbol_data.append(symbol_df)
        
        # Count of computed indicators
        indicator_cols = [col for col in symbol_df.columns 
                         if col not in ['symbol', 'date', 'open', 'high', 'low', 'close', 'volume', 'bid', 'ask']]
        print(f"    ✓ {symbol}: {len(indicator_cols)} indicators computed (temporal-safe)")
    
    # Combine all symbol data
    if all_symbol_data:
        final_df = pd.concat(all_symbol_data, ignore_index=True)
        final_df = final_df.sort_values(['symbol', 'date']).reset_index(drop=True)
    else:
        final_df = result_df
    
    # FIXED: Fill NaN values with proper temporal constraints (no future leakage)
    final_df = fill_missing_temporal_safe(final_df, split_date, is_training)
    
    print(f"Technical indicators computed for {len(symbols)} symbols (LEAKAGE-FREE)")
    print(f"Final DataFrame shape: {final_df.shape}")
    
    return final_df


def compute_indicators_expanding_window(symbol_df: pd.DataFrame, train_mask: pd.Series, 
                                      min_window: int) -> pd.DataFrame:
    """Compute indicators using expanding window to prevent future leakage."""
    # Initialize indicator columns with NaN
    indicator_columns = [
        'sma_10', 'sma_50', 'sma_200', 'ema_12', 'ema_26', 'rsi_14',
        'macd_line', 'macd_signal', 'macd_hist', 'bb_upper', 'bb_middle', 'bb_lower',
        'atr_14', 'volume_sma', 'volume_ratio', 'price_change', 'price_change_5d',
        'price_change_20d', 'volatility_20d', 'high_20d', 'low_20d', 'price_position'
    ]
    
    for col in indicator_columns:
        symbol_df[col] = np.nan
    
    # Compute indicators for each time point using only past data
    for i in range(len(symbol_df)):
        if i < min_window:
            continue  # Skip until we have minimum data
            
        # Use data up to current point (inclusive) - NO FUTURE DATA
        historical_data = symbol_df.iloc[:i+1].copy()
        
        # Compute indicators on historical data only
        if len(historical_data) >= 10:
            symbol_df.loc[i, 'sma_10'] = historical_data['close'].tail(10).mean()
        if len(historical_data) >= 50:
            symbol_df.loc[i, 'sma_50'] = historical_data['close'].tail(50).mean()
        if len(historical_data) >= 200:
            symbol_df.loc[i, 'sma_200'] = historical_data['close'].tail(200).mean()
        
        # EMA calculations
        if len(historical_data) >= 12:
            symbol_df.loc[i, 'ema_12'] = historical_data['close'].ewm(span=12).mean().iloc[-1]
        if len(historical_data) >= 26:
            symbol_df.loc[i, 'ema_26'] = historical_data['close'].ewm(span=26).mean().iloc[-1]
        
        # RSI calculation
        if len(historical_data) >= 14:
            rsi_val = compute_rsi(historical_data['close'], period=14).iloc[-1]
            if not pd.isna(rsi_val):
                symbol_df.loc[i, 'rsi_14'] = rsi_val
        
        # MACD calculation
        if len(historical_data) >= 26:
            macd_line, macd_signal, macd_hist = compute_macd(historical_data['close'])
            if len(macd_line) > 0 and not pd.isna(macd_line.iloc[-1]):
                symbol_df.loc[i, 'macd_line'] = macd_line.iloc[-1]
                symbol_df.loc[i, 'macd_signal'] = macd_signal.iloc[-1]
                symbol_df.loc[i, 'macd_hist'] = macd_hist.iloc[-1]
        
        # Price changes
        if i >= 1:
            symbol_df.loc[i, 'price_change'] = symbol_df.loc[i, 'close'] / symbol_df.loc[i-1, 'close'] - 1
        if i >= 5:
            symbol_df.loc[i, 'price_change_5d'] = symbol_df.loc[i, 'close'] / symbol_df.loc[i-5, 'close'] - 1
        if i >= 20:
            symbol_df.loc[i, 'price_change_20d'] = symbol_df.loc[i, 'close'] / symbol_df.loc[i-20, 'close'] - 1
            
        # Volatility
        if len(historical_data) >= 20:
            returns = historical_data['close'].pct_change().dropna()
            if len(returns) >= 20:
                symbol_df.loc[i, 'volatility_20d'] = returns.tail(20).std()
        
        # High/Low levels
        if len(historical_data) >= 20:
            symbol_df.loc[i, 'high_20d'] = historical_data['high'].tail(20).max()
            symbol_df.loc[i, 'low_20d'] = historical_data['low'].tail(20).min()
            
            # Price position
            high_20 = symbol_df.loc[i, 'high_20d']
            low_20 = symbol_df.loc[i, 'low_20d']
            if high_20 != low_20:
                symbol_df.loc[i, 'price_position'] = (symbol_df.loc[i, 'close'] - low_20) / (high_20 - low_20)
        
        # Volume indicators
        if len(historical_data) >= 20:
            vol_sma = historical_data['volume'].tail(20).mean()
            symbol_df.loc[i, 'volume_sma'] = vol_sma
            if vol_sma > 0:
                symbol_df.loc[i, 'volume_ratio'] = symbol_df.loc[i, 'volume'] / vol_sma
    
    return symbol_df


def compute_indicators_standard(symbol_df: pd.DataFrame) -> pd.DataFrame:
    """Compute indicators using standard rolling windows (for training data)."""
    # Simple Moving Averages
    symbol_df['sma_10'] = symbol_df['close'].rolling(window=10, min_periods=1).mean()
    symbol_df['sma_50'] = symbol_df['close'].rolling(window=50, min_periods=1).mean()
    symbol_df['sma_200'] = symbol_df['close'].rolling(window=200, min_periods=1).mean()
    
    # Exponential Moving Averages
    symbol_df['ema_12'] = symbol_df['close'].ewm(span=12, min_periods=1).mean()
    symbol_df['ema_26'] = symbol_df['close'].ewm(span=26, min_periods=1).mean()
    
    # RSI (Relative Strength Index)
    symbol_df['rsi_14'] = compute_rsi(symbol_df['close'], period=14)
    
    # MACD (Moving Average Convergence Divergence)
    macd_line, macd_signal, macd_hist = compute_macd(symbol_df['close'])
    symbol_df['macd_line'] = macd_line
    symbol_df['macd_signal'] = macd_signal
    symbol_df['macd_hist'] = macd_hist
    
    # Bollinger Bands
    bb_upper, bb_middle, bb_lower = compute_bollinger_bands(symbol_df['close'])
    symbol_df['bb_upper'] = bb_upper
    symbol_df['bb_middle'] = bb_middle
    symbol_df['bb_lower'] = bb_lower
    
    # Average True Range (ATR)
    symbol_df['atr_14'] = compute_atr(symbol_df, period=14)
    
    # Volume indicators
    symbol_df['volume_sma'] = symbol_df['volume'].rolling(window=20, min_periods=1).mean()
    symbol_df['volume_ratio'] = symbol_df['volume'] / symbol_df['volume_sma']
    
    # Price momentum indicators
    symbol_df['price_change'] = symbol_df['close'].pct_change()
    symbol_df['price_change_5d'] = symbol_df['close'].pct_change(periods=5)
    symbol_df['price_change_20d'] = symbol_df['close'].pct_change(periods=20)
    
    # Volatility indicators
    symbol_df['volatility_20d'] = symbol_df['price_change'].rolling(window=20, min_periods=1).std()
    
    # Support/Resistance levels
    symbol_df['high_20d'] = symbol_df['high'].rolling(window=20, min_periods=1).max()
    symbol_df['low_20d'] = symbol_df['low'].rolling(window=20, min_periods=1).min()
    
    # Position relative to range
    symbol_df['price_position'] = (
        (symbol_df['close'] - symbol_df['low_20d']) / 
        (symbol_df['high_20d'] - symbol_df['low_20d'])
    )
    
    return symbol_df


def fill_missing_temporal_safe(df: pd.DataFrame, split_date: str, is_training: bool) -> pd.DataFrame:
    """Fill missing values without future information leakage."""
    if split_date and not is_training:
        # For validation data, only forward-fill within the validation period
        split_dt = pd.to_datetime(split_date)
        train_mask = df['date'] <= split_dt
        val_mask = df['date'] > split_dt
        
        # Fill training data normally
        df.loc[train_mask] = df.loc[train_mask].ffill().bfill()
        
        # Fill validation data without using future information
        # Use last known training values to start validation period
        if train_mask.sum() > 0 and val_mask.sum() > 0:
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            last_train_values = df.loc[train_mask, numeric_columns].iloc[-1]
            
            # Forward fill within validation period only
            val_data = df.loc[val_mask].copy()
            val_data[numeric_columns] = val_data[numeric_columns].ffill()
            
            # Use last training values for any remaining NaN at start of validation
            for col in numeric_columns:
                first_val_idx = val_data.index[0]
                if pd.isna(val_data.loc[first_val_idx, col]):
                    val_data.loc[first_val_idx, col] = last_train_values[col]
            
            # Forward fill again to propagate the initialized values
            val_data[numeric_columns] = val_data[numeric_columns].ffill()
            df.loc[val_mask] = val_data
    else:
        # Standard fill for training data
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        df[numeric_columns] = df[numeric_columns].ffill().bfill()
    
    return df


def compute_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """Compute Relative Strength Index."""
    delta = prices.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi


def compute_macd(prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> tuple:
    """Compute MACD line, signal line, and histogram."""
    ema_fast = prices.ewm(span=fast).mean()
    ema_slow = prices.ewm(span=slow).mean()
    
    macd_line = ema_fast - ema_slow
    macd_signal = macd_line.ewm(span=signal).mean()
    macd_hist = macd_line - macd_signal
    
    return macd_line, macd_signal, macd_hist


def compute_bollinger_bands(prices: pd.Series, period: int = 20, std_dev: float = 2) -> tuple:
    """Compute Bollinger Bands."""
    sma = prices.rolling(window=period).mean()
    std = prices.rolling(window=period).std()
    
    upper_band = sma + (std * std_dev)
    lower_band = sma - (std * std_dev)
    
    return upper_band, sma, lower_band


def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Compute Average True Range."""
    high_low = df['high'] - df['low']
    high_close_prev = np.abs(df['high'] - df['close'].shift(1))
    low_close_prev = np.abs(df['low'] - df['close'].shift(1))
    
    true_range = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
    atr = true_range.rolling(window=period).mean()
    
    return atr


def compute_stochastic(df: pd.DataFrame, k_period: int = 14, d_period: int = 3) -> tuple:
    """Compute Stochastic Oscillator."""
    lowest_low = df['low'].rolling(window=k_period).min()
    highest_high = df['high'].rolling(window=k_period).max()
    
    k_percent = 100 * ((df['close'] - lowest_low) / (highest_high - lowest_low))
    d_percent = k_percent.rolling(window=d_period).mean()
    
    return k_percent, d_percent


def compute_williams_r(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Compute Williams %R."""
    highest_high = df['high'].rolling(window=period).max()
    lowest_low = df['low'].rolling(window=period).min()
    
    williams_r = -100 * ((highest_high - df['close']) / (highest_high - lowest_low))
    
    return williams_r


def validate_technical_indicators(df: pd.DataFrame) -> bool:
    """
    Validate technical indicators DataFrame.
    
    Args:
        df: DataFrame with technical indicators
        
    Returns:
        True if validation passes, False otherwise
    """
    # Check for required base columns
    required_base_cols = ['symbol', 'date', 'open', 'high', 'low', 'close', 'volume']
    missing_base = [col for col in required_base_cols if col not in df.columns]
    
    if missing_base:
        print(f"Missing required base columns: {missing_base}")
        return False
    
    # Check for key technical indicators
    expected_indicators = ['sma_10', 'sma_50', 'rsi_14', 'macd_line', 'bb_upper']
    missing_indicators = [col for col in expected_indicators if col not in df.columns]
    
    if missing_indicators:
        print(f"Missing expected indicators: {missing_indicators}")
        return False
    
    # Check for excessive NaN values in indicators
    indicator_cols = [col for col in df.columns if col not in required_base_cols]
    for col in indicator_cols:
        nan_ratio = df[col].isna().sum() / len(df)
        if nan_ratio > 0.5:  # More than 50% NaN values
            print(f"Warning: High NaN ratio in {col}: {nan_ratio:.2%}")
    
    print("✓ Technical indicators validation completed")
    return True
