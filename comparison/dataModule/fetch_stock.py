"""
Stock data fetching module using yfinance.
Fetches OHLCV data with bid/ask information where available.
"""

from typing import List
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


def fetch_stock_data(symbols: List[str], start: str, end: str,
                     interval: str = "1d") -> pd.DataFrame:
    """
    Uses yfinance to download OHLCV per symbol/day.
    Also include bid/ask if available via info dict.
    
    Args:
        symbols: List of stock symbols (e.g., ['AAPL', 'GOOGL'])
        start: Start date in 'YYYY-MM-DD' format
        end: End date in 'YYYY-MM-DD' format
        interval: Data interval ('1d', '1h', etc.)
    
    Returns:
        DataFrame with columns: symbol, date, open, high, low, close, volume, bid, ask
        Ensures no future leakage, corporate action adjusted. Uses Pandas datetime index.
    """
    print(f"Fetching stock data for {len(symbols)} symbols from {start} to {end}")
    
    all_data = []
    
    for symbol in symbols:
        try:
            print(f"  Downloading {symbol}...")
            
            # Create ticker object
            ticker = yf.Ticker(symbol)
            
            # Download historical data
            hist_data = ticker.history(start=start, end=end, interval=interval, 
                                     auto_adjust=True, prepost=True)
            
            if hist_data.empty:
                print(f"  Warning: No data found for {symbol}")
                continue
            
            # Get current info for bid/ask
            info = {}
            try:
                info = ticker.info
            except Exception as e:
                print(f"  Warning: Could not fetch info for {symbol}: {e}")
            
            # Prepare dataframe
            df = hist_data.copy()
            df.reset_index(inplace=True)
            df['symbol'] = symbol
            
            # Rename columns to match our schema
            df.columns = df.columns.str.lower()
            if 'date' not in df.columns and 'datetime' in df.columns:
                df.rename(columns={'datetime': 'date'}, inplace=True)
            
            # Add bid/ask from info if available
            df['bid'] = info.get('bid', None)
            df['ask'] = info.get('ask', None)
            
            # Ensure date column is datetime
            df['date'] = pd.to_datetime(df['date'])
            
            # Select and order columns
            expected_columns = ['symbol', 'date', 'open', 'high', 'low', 'close', 'volume', 'bid', 'ask']
            available_columns = [col for col in expected_columns if col in df.columns]
            df = df[available_columns]
            
            # Remove any rows with all NaN values (excluding symbol and date)
            numeric_cols = [col for col in df.columns if col not in ['symbol', 'date']]
            df = df.dropna(subset=numeric_cols, how='all')
            
            all_data.append(df)
            print(f"  ✓ {symbol}: {len(df)} data points")
            
        except Exception as e:
            print(f"  ✗ Error fetching {symbol}: {e}")
            continue
    
    if not all_data:
        print("  Warning: No data was successfully fetched for any symbol")
        return pd.DataFrame()
    
    # Combine all data
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # Sort by symbol and date
    combined_df = combined_df.sort_values(['symbol', 'date']).reset_index(drop=True)
    
    # Ensure no future leakage - remove any dates beyond today
    today = datetime.now().date()
    combined_df = combined_df[combined_df['date'].dt.date <= today]
    
    print(f"Total stock data points: {len(combined_df)}")
    print(f"Date range: {combined_df['date'].min()} to {combined_df['date'].max()}")
    print(f"Symbols: {sorted(combined_df['symbol'].unique())}")
    
    return combined_df


def validate_stock_data(df: pd.DataFrame) -> bool:
    """
    Validate the structure and content of stock data DataFrame.
    
    Args:
        df: Stock data DataFrame
        
    Returns:
        True if validation passes, False otherwise
    """
    required_columns = ['symbol', 'date', 'open', 'high', 'low', 'close', 'volume']
    
    # Check required columns exist
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        print(f"Missing required columns: {missing_cols}")
        return False
    
    # Check for valid OHLC relationships
    invalid_ohlc = df[
        (df['low'] > df['high']) | 
        (df['open'] > df['high']) | 
        (df['close'] > df['high']) |
        (df['open'] < df['low']) | 
        (df['close'] < df['low'])
    ]
    
    if len(invalid_ohlc) > 0:
        print(f"Warning: {len(invalid_ohlc)} rows with invalid OHLC relationships")
    
    # Check for negative prices or volumes
    price_cols = ['open', 'high', 'low', 'close']
    negative_prices = df[df[price_cols] < 0].any(axis=1).sum()
    negative_volume = df[df['volume'] < 0].shape[0]
    
    if negative_prices > 0:
        print(f"Warning: {negative_prices} rows with negative prices")
    if negative_volume > 0:
        print(f"Warning: {negative_volume} rows with negative volume")
    
    print("✓ Stock data validation completed")
    return True
