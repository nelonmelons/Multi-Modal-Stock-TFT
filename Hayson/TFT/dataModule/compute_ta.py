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


def compute_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute technical indicators for stock data.
    
    Args:
        df: Stock DataFrame from fetch_stock_data with OHLCV columns
        
    Returns:
        DataFrame with additional technical indicator columns:
        sma_10, sma_50, ema_12, ema_26, rsi_14, macd_line, macd_signal, 
        macd_hist, bb_upper, bb_middle, bb_lower, atr_14, volume_sma
    """
    print("Computing technical indicators...")
    
    if df.empty:
        print("Warning: Empty DataFrame provided")
        return df
    
    # Copy input DataFrame to avoid modifying original
    result_df = df.copy()
    
    # Process each symbol separately
    symbols = result_df['symbol'].unique()
    
    all_symbol_data = []
    
    for symbol in symbols:
        print(f"  Computing indicators for {symbol}...")
        
        # Filter data for this symbol
        symbol_df = result_df[result_df['symbol'] == symbol].copy()
        symbol_df = symbol_df.sort_values('date').reset_index(drop=True)
        
        if len(symbol_df) < 50:  # Need sufficient data for indicators
            print(f"    Warning: Insufficient data for {symbol} ({len(symbol_df)} rows)")
        
        # Simple Moving Averages
        symbol_df['sma_10'] = symbol_df['close'].rolling(window=10).mean()
        symbol_df['sma_50'] = symbol_df['close'].rolling(window=50).mean()
        symbol_df['sma_200'] = symbol_df['close'].rolling(window=200).mean()
        
        # Exponential Moving Averages
        symbol_df['ema_12'] = symbol_df['close'].ewm(span=12).mean()
        symbol_df['ema_26'] = symbol_df['close'].ewm(span=26).mean()
        
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
        symbol_df['volume_sma'] = symbol_df['volume'].rolling(window=20).mean()
        symbol_df['volume_ratio'] = symbol_df['volume'] / symbol_df['volume_sma']
        
        # Price momentum indicators
        symbol_df['price_change'] = symbol_df['close'].pct_change()
        symbol_df['price_change_5d'] = symbol_df['close'].pct_change(periods=5)
        symbol_df['price_change_20d'] = symbol_df['close'].pct_change(periods=20)
        
        # Volatility indicators
        symbol_df['volatility_20d'] = symbol_df['price_change'].rolling(window=20).std()
        
        # Support/Resistance levels
        symbol_df['high_20d'] = symbol_df['high'].rolling(window=20).max()
        symbol_df['low_20d'] = symbol_df['low'].rolling(window=20).min()
        
        # Position relative to range
        symbol_df['price_position'] = (
            (symbol_df['close'] - symbol_df['low_20d']) / 
            (symbol_df['high_20d'] - symbol_df['low_20d'])
        )
        
        all_symbol_data.append(symbol_df)
        
        # Count of computed indicators
        indicator_cols = [col for col in symbol_df.columns 
                         if col not in ['symbol', 'date', 'open', 'high', 'low', 'close', 'volume', 'bid', 'ask']]
        print(f"    ✓ {symbol}: {len(indicator_cols)} indicators computed")
    
    # Combine all symbol data
    if all_symbol_data:
        final_df = pd.concat(all_symbol_data, ignore_index=True)
        final_df = final_df.sort_values(['symbol', 'date']).reset_index(drop=True)
    else:
        final_df = result_df
    
    # Fill any remaining NaN values with forward fill, then backward fill
    numeric_columns = final_df.select_dtypes(include=[np.number]).columns
    final_df[numeric_columns] = final_df[numeric_columns].ffill().bfill()
    
    print(f"Technical indicators computed for {len(symbols)} symbols")
    print(f"Final DataFrame shape: {final_df.shape}")
    
    return final_df


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
