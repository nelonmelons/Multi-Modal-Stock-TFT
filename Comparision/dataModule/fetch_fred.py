"""
FRED (Federal Reserve Economic Data) fetching module.
Fetches macroeconomic indicators like CPI, Federal Funds Rate, Unemployment Rate, etc.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Try to import fredapi
try:
    import fredapi
    HAS_FREDAPI = True
except ImportError:
    HAS_FREDAPI = False
    print("Warning: fredapi not installed. Install with: pip install fredapi")


def fetch_fred_data(start: str, end: str, fred_api_key: Optional[str] = None) -> pd.DataFrame:
    """
    Fetch macroeconomic data from FRED (Federal Reserve Economic Data).
    
    Args:
        start: Start date in 'YYYY-MM-DD' format
        end: End date in 'YYYY-MM-DD' format
        fred_api_key: FRED API key (optional, can use environment variable)
        
    Returns:
        DataFrame with columns: date, cpi, fedfunds, unrate, t10y2y, etc.
        These are known-future features for TFT.
    """
    print("Fetching FRED economic data...")
    
    if not HAS_FREDAPI:
        print("Warning: fredapi not available, returning empty economic data")
        return create_empty_fred_df(start, end)
    
    # Get API key from parameter or environment
    import os
    api_key = fred_api_key or os.getenv('FRED_API_KEY')
    
    if not api_key:
        print("Warning: No FRED API key provided, returning empty economic data")
        print("Get a free API key at: https://fred.stlouisfed.org/docs/api/api_key.html")
        return create_empty_fred_df(start, end)
    
    try:
        # Initialize FRED client with SSL verification handling
        import ssl
        import urllib3
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        
        try:
            fred = fredapi.Fred(api_key=api_key)
        except Exception as ssl_error:
            if "CERTIFICATE_VERIFY_FAILED" in str(ssl_error):
                print("    ⚠️ SSL certificate verification failed, trying with unverified context...")
                # Create unverified SSL context as fallback
                original_context = ssl._create_default_https_context
                ssl._create_default_https_context = ssl._create_unverified_context
                try:
                    fred = fredapi.Fred(api_key=api_key)
                finally:
                    # Restore original context
                    ssl._create_default_https_context = original_context
            else:
                raise ssl_error
        
        # Define economic indicators to fetch
        indicators = {
            'cpi': 'CPIAUCSL',           # Consumer Price Index
            'fedfunds': 'FEDFUNDS',      # Federal Funds Rate
            'unrate': 'UNRATE',          # Unemployment Rate
            't10y2y': 'T10Y2Y',          # 10-Year Treasury minus 2-Year Treasury
            'gdp': 'GDP',                # Gross Domestic Product
            'vix': 'VIXCLS',             # VIX Volatility Index
            'dxy': 'DTWEXBGS',           # US Dollar Index
            'oil': 'DCOILWTICO',         # WTI Oil Price
        }
        
        print(f"Fetching {len(indicators)} economic indicators from FRED...")
        
        all_data = {}
        
        for name, series_id in indicators.items():
            try:
                print(f"  Fetching {name} ({series_id})...")
                data = fred.get_series(series_id, start=start, end=end)
                
                if not data.empty:
                    all_data[name] = data
                    print(f"    ✓ {name}: {len(data)} data points")
                else:
                    print(f"    ⚠️ {name}: No data available")
                    
            except Exception as e:
                print(f"    ✗ Error fetching {name}: {e}")
                continue
        
        if not all_data:
            print("No economic data retrieved")
            return create_empty_fred_df(start, end)
        
        # Combine all series into a single DataFrame
        combined_df = pd.DataFrame(all_data)
        combined_df.index.name = 'date'
        combined_df.reset_index(inplace=True)
        
        # Forward fill missing values (economic data is often monthly/quarterly)
        combined_df = combined_df.ffill()
        
        # Ensure date column is datetime
        combined_df['date'] = pd.to_datetime(combined_df['date'])
        
        # Sort by date
        combined_df = combined_df.sort_values('date').reset_index(drop=True)
        
        print(f"Combined FRED data: {len(combined_df)} rows, {len(combined_df.columns)-1} indicators")
        print(f"Date range: {combined_df['date'].min()} to {combined_df['date'].max()}")
        
        return combined_df
        
    except Exception as e:
        print(f"Error fetching FRED data: {e}")
        return create_empty_fred_df(start, end)


def create_empty_fred_df(start: str, end: str) -> pd.DataFrame:
    """Create empty FRED DataFrame with proper structure."""
    date_range = pd.date_range(start=start, end=end, freq='D')
    
    # Create empty DataFrame with expected columns
    empty_data = {
        'date': date_range,
        'cpi': 0.0,
        'fedfunds': 0.0,
        'unrate': 0.0,
        't10y2y': 0.0,
        'gdp': 0.0,
        'vix': 0.0,
        'dxy': 0.0,
        'oil': 0.0,
    }
    
    return pd.DataFrame(empty_data)


def interpolate_economic_data(df: pd.DataFrame, target_dates: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Interpolate economic data to match daily stock market dates.
    
    Economic data is often monthly/quarterly, but we need daily values
    for TFT training.
    
    Args:
        df: FRED data DataFrame
        target_dates: Daily date range to interpolate to
        
    Returns:
        DataFrame with daily interpolated values
    """
    if df.empty:
        return create_empty_fred_df(
            target_dates.min().strftime('%Y-%m-%d'),
            target_dates.max().strftime('%Y-%m-%d')
        )
    
    # Create target DataFrame
    target_df = pd.DataFrame({'date': target_dates})
    
    # Merge with economic data
    merged_df = pd.merge(target_df, df, on='date', how='left')
    
    # Interpolate missing values
    numeric_cols = merged_df.select_dtypes(include=[np.number]).columns
    merged_df[numeric_cols] = merged_df[numeric_cols].interpolate(method='linear')
    
    # Forward fill any remaining NaNs
    merged_df[numeric_cols] = merged_df[numeric_cols].ffill()
    
    # Backward fill any remaining NaNs
    merged_df[numeric_cols] = merged_df[numeric_cols].bfill()
    
    # Fill any still remaining NaNs with 0
    merged_df[numeric_cols] = merged_df[numeric_cols].fillna(0)
    
    return merged_df


def get_economic_features() -> List[str]:
    """Get list of economic feature column names."""
    return [
        'cpi',       # Consumer Price Index
        'fedfunds',  # Federal Funds Rate  
        'unrate',    # Unemployment Rate
        't10y2y',    # 10Y-2Y Treasury Spread
        'gdp',       # GDP
        'vix',       # VIX Volatility
        'dxy',       # Dollar Index
        'oil',       # Oil Price
    ]


def validate_fred_data(df: pd.DataFrame) -> bool:
    """
    Validate FRED data DataFrame.
    
    Args:
        df: FRED data DataFrame
        
    Returns:
        True if validation passes, False otherwise
    """
    if df.empty:
        print("Warning: Empty FRED DataFrame")
        return False
    
    required_cols = ['date'] + get_economic_features()
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        print(f"Missing FRED columns: {missing_cols}")
        return False
    
    # Check for excessive NaN values
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        nan_ratio = df[col].isna().sum() / len(df)
        if nan_ratio > 0.8:  # More than 80% NaN
            print(f"Warning: High NaN ratio in {col}: {nan_ratio:.2%}")
    
    print("✓ FRED data validation completed")
    return True


def get_fred_api_info():
    """Print information about FRED API setup."""
    print("🏦 FRED (Federal Reserve Economic Data) API Setup")
    print("=" * 50)
    print("FRED provides free access to economic data from the Federal Reserve.")
    print()
    print("To get a free API key:")
    print("1. Go to: https://fred.stlouisfed.org/")
    print("2. Create an account")
    print("3. Request an API key at: https://fred.stlouisfed.org/docs/api/api_key.html")
    print("4. Add to your .env file: FRED_API_KEY=your_api_key")
    print()
    print("Available economic indicators:")
    
    indicators = [
        ("CPI", "Consumer Price Index - inflation measure"),
        ("FEDFUNDS", "Federal Funds Rate - key interest rate"),
        ("UNRATE", "Unemployment Rate - labor market health"),
        ("T10Y2Y", "10Y-2Y Treasury Spread - yield curve"),
        ("GDP", "Gross Domestic Product - economic output"),
        ("VIX", "Volatility Index - market fear gauge"),
        ("DXY", "US Dollar Index - currency strength"),
        ("OIL", "WTI Oil Price - commodity price"),
    ]
    
    for name, description in indicators:
        print(f"  • {name}: {description}")


if __name__ == "__main__":
    get_fred_api_info()
