"""
Events data fetching module using yfinance and API-Ninjas.
Fetches corporate actions (dividends, splits), company info, and earnings calendar.
"""

from typing import List, Dict, Any, Optional
import pandas as pd
import yfinance as yf
import requests
import os
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# API-Ninjas earnings calendar endpoint
API_NINJAS_EARNINGS = "https://api.api-ninjas.com/v1/earningscalendar"


def fetch_earnings_ninjas(symbols: List[str], api_key: Optional[str] = None) -> pd.DataFrame:
    """
    Fetch earnings calendar data from API-Ninjas (free tier with generous limits).
    
    Args:
        symbols: List of symbols to get earnings for
        api_key: API-Ninjas API key (optional, can use environment variable)
        
    Returns:
        DataFrame with columns: ticker, date, estimated_eps, actual_eps, estimated_revenue, actual_revenue
    """
    # Get API key from parameter or environment
    ninjas_api_key = api_key or os.getenv('API_NINJAS_KEY')

    if not ninjas_api_key:
        print("    ⚠️ No API-Ninjas key provided, skipping earnings calendar")
        print("    Get a free API key at: https://api.api-ninjas.com/register")
        return pd.DataFrame()
    
    all_earnings = []
    headers = {"X-Api-Key": ninjas_api_key}
    
    for symbol in symbols:
        try:
            print(f"    Fetching earnings for {symbol} from API-Ninjas...")
            
            # Fetch both upcoming and past earnings
            for upcoming in [True, False]:
                params = {
                    "ticker": symbol,
                    "show_upcoming": str(upcoming).lower()
                }
                
                response = requests.get(API_NINJAS_EARNINGS, headers=headers, params=params, timeout=30)
                response.raise_for_status()
                data = response.json()
                
                if isinstance(data, list) and data:
                    for record in data:
                        all_earnings.append({
                            'symbol': record.get('ticker', symbol),
                            'date': record.get('date'),
                            'epsEstimated': record.get('estimated_eps'),
                            'eps': record.get('actual_eps'),
                            'revenueEstimated': record.get('estimated_revenue'),
                            'revenue': record.get('actual_revenue')
                        })
                    
                    earnings_type = "upcoming" if upcoming else "past"
                    print(f"    ✓ API-Ninjas {earnings_type}: {len(data)} records for {symbol}")
                    
        except requests.exceptions.RequestException as e:
            print(f"    ✗ Error fetching {symbol} from API-Ninjas: {e}")
            continue
        except Exception as e:
            print(f"    ✗ Error processing {symbol} earnings: {e}")
            continue
    
    if all_earnings:
        df = pd.DataFrame(all_earnings)
        # Remove duplicates and convert date
        df = df.drop_duplicates(subset=['symbol', 'date'])
        df['date'] = pd.to_datetime(df['date'])
        print(f"    ✓ Total API-Ninjas earnings: {len(df)} records for {df['symbol'].nunique()} symbols")
        return df
    else:
        return pd.DataFrame()


def fetch_events_data(symbols: List[str], start: str, end: str,
                      token: Optional[str] = None,
                      api_ninjas_key: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
    """
    Uses yfinance for corporate actions and API-Ninjas for earnings calendar.
    
    Args:
        symbols: List of stock symbols
        start: Start date in 'YYYY-MM-DD' format  
        end: End date in 'YYYY-MM-DD' format
        token: Unused parameter (kept for compatibility)
        api_ninjas_key: API-Ninjas API key for earnings calendar (optional)
        
    Returns:
        Dictionary with structure:
        {
            symbol: {
                "earnings": [dates with eps data],
                "splits": [dates], 
                "dividends": [dates],
                "sector": str,
                "market_cap": float,
                "eps_data": DataFrame with eps estimates/actuals
            }, ...
        }
    """
    print(f"Fetching events data for {len(symbols)} symbols using yfinance + API-Ninjas")
    
    # First, try to fetch earnings calendar from API-Ninjas for all symbols
    print("  Fetching earnings calendar from API-Ninjas...")
    ninjas_earnings_df = fetch_earnings_ninjas(symbols, api_ninjas_key)
    
    events_data = {}
    start_date = pd.to_datetime(start)
    end_date = pd.to_datetime(end)
    
    for symbol in symbols:
        print(f"  Processing {symbol}...")
        
        try:
            # Create ticker object
            ticker = yf.Ticker(symbol)
            
            # Initialize data structure
            symbol_data = {
                "earnings": [],
                "splits": [], 
                "dividends": [],
                "sector": "Unknown",
                "market_cap": 0.0,
                "holidays": []  # Market holidays not available from yfinance
            }
            
            # Get company info
            try:
                info = ticker.info
                symbol_data["sector"] = info.get("sector", "Unknown")
                symbol_data["market_cap"] = info.get("marketCap", 0.0)
                print(f"    ✓ Company info: {symbol_data['sector']}")
            except Exception as e:
                print(f"    ⚠️ Could not fetch company info: {e}")
            
            # Get dividends
            try:
                dividends = ticker.dividends
                if not dividends.empty:
                    # Simple date filtering - skip if timezone issues
                    symbol_data["dividends"] = [d.strftime('%Y-%m-%d') for d in dividends.index][-5:]  # Last 5 dividends
                    print(f"    ✓ Dividends: {len(symbol_data['dividends'])} events")
                else:
                    print(f"    ✓ Dividends: 0 events")
            except Exception as e:
                print(f"    ⚠️ Could not fetch dividends: {e}")
            
            # Get stock splits
            try:
                splits = ticker.splits
                if not splits.empty:
                    # Simple date filtering - skip if timezone issues
                    symbol_data["splits"] = [d.strftime('%Y-%m-%d') for d in splits.index][-3:]  # Last 3 splits
                    print(f"    ✓ Splits: {len(symbol_data['splits'])} events")
                else:
                    print(f"    ✓ Splits: 0 events")
            except Exception as e:
                print(f"    ⚠️ Could not fetch splits: {e}")
            
            # Get earnings dates from API-Ninjas data if available
            earnings_dates = []
            eps_data = pd.DataFrame()
            
            if not ninjas_earnings_df.empty:
                symbol_earnings = ninjas_earnings_df[ninjas_earnings_df['symbol'].str.upper() == symbol.upper()]
                if not symbol_earnings.empty:
                    # Filter earnings within our date range and get dates
                    start_dt = pd.to_datetime(start)
                    end_dt = pd.to_datetime(end)
                    symbol_earnings = symbol_earnings[
                        (symbol_earnings['date'] >= start_dt) & 
                        (symbol_earnings['date'] <= end_dt)
                    ]
                    
                    if not symbol_earnings.empty:
                        earnings_dates = symbol_earnings['date'].dt.strftime('%Y-%m-%d').tolist()
                        eps_data = symbol_earnings.copy()
                        print(f"    ✓ API-Ninjas earnings: {len(earnings_dates)} events with EPS data")
                    else:
                        print(f"    ℹ️ No API-Ninjas earnings data in date range for {symbol}")
                else:
                    print(f"    ℹ️ No API-Ninjas earnings data found for {symbol}")
            
            # Fallback to yfinance earnings prediction if no FMP data
            if not earnings_dates:
                try:
                    # Method: Try earnings history to predict future dates
                    earnings_history = ticker.earnings_dates
                    if earnings_history is not None and not earnings_history.empty:
                        # Get recent earnings to establish pattern (quarterly)
                        recent_earnings = earnings_history.dropna().tail(8)  # Last 2 years
                        if len(recent_earnings) >= 2:
                            # Calculate average days between earnings - normalize timezones
                            dates = sorted(recent_earnings.index)
                            
                            # Normalize to timezone-naive for comparison
                            normalized_dates = []
                            for date in dates:
                                if hasattr(date, 'tz') and date.tz is not None:
                                    # Convert to UTC then remove timezone info
                                    normalized_dates.append(date.tz_convert('UTC').tz_localize(None))
                                else:
                                    normalized_dates.append(date)
                            
                            intervals = [(normalized_dates[i] - normalized_dates[i-1]).days for i in range(1, len(normalized_dates))]
                            avg_interval = sum(intervals) / len(intervals) if intervals else 90
                            
                            # Predict next earnings date(s)
                            last_earnings = normalized_dates[-1]
                            next_earnings = last_earnings + pd.Timedelta(days=int(avg_interval))
                            
                            # Compare with normalized start/end dates
                            start_norm = start_date.tz_localize(None) if hasattr(start_date, 'tz') and start_date.tz else start_date
                            end_norm = end_date.tz_localize(None) if hasattr(end_date, 'tz') and end_date.tz else end_date
                            
                            # Only add if it's in our date range
                            if start_norm <= next_earnings <= end_norm:
                                earnings_dates.append(next_earnings.strftime('%Y-%m-%d'))
                                print(f"    📅 Predicted next earnings: {next_earnings.strftime('%Y-%m-%d')}")
                            
                        print(f"    ✓ Analyzed {len(recent_earnings)} historical earnings")
                except Exception as e:
                    print(f"    ⚠️ Could not analyze earnings history: {e}")
            
            symbol_data["earnings"] = earnings_dates
            symbol_data["eps_data"] = eps_data  # Store EPS estimates/actuals
            if earnings_dates:
                print(f"    ✓ Earnings: {len(earnings_dates)} events")
            else:
                print(f"    ℹ️ Earnings: No future earnings found (check API-Ninjas key and data availability)")
                    
            # Store symbol data
            events_data[symbol] = symbol_data
            print(f"    ✓ {symbol}: {len(symbol_data['earnings'])} earnings, {len(symbol_data['splits'])} splits, {len(symbol_data['dividends'])} dividends")
            
        except Exception as e:
            print(f"    ✗ Error processing {symbol}: {e}")
            # Add empty data for failed symbols
            events_data[symbol] = {
                "earnings": [],
                "splits": [], 
                "dividends": [],
                "sector": "Unknown",
                "market_cap": 0.0,
                "holidays": [],
                "eps_data": pd.DataFrame()
            }
    
    print(f"Events data fetched for {len(events_data)} symbols")
    return events_data


def show_corporate_actions_sample(symbols: List[str], period: str = "2y") -> None:
    """
    Show sample of corporate actions data for testing/validation.
    
    Args:
        symbols: List of symbols to show data for
        period: Period to fetch data for (e.g., "1y", "2y", "5y")
    """
    print(f"\n📊 Corporate Actions Sample Data")
    print("=" * 40)
    
    for symbol in symbols[:3]:  # Limit to first 3 symbols
        print(f"\n🏢 {symbol}:")
        try:
            ticker = yf.Ticker(symbol)
            
            # Show dividends
            dividends = ticker.dividends
            if not dividends.empty:
                recent_dividends = dividends.tail(5)
                print(f"  💰 Recent Dividends ({len(recent_dividends)} shown):")
                for date, amount in recent_dividends.items():
                    try:
                        date_str = date.strftime('%Y-%m-%d')
                    except:
                        date_str = str(date)[:10]
                    print(f"    {date_str}: ${amount:.2f}")
            else:
                print(f"  💰 Dividends: None found")
            
            # Show splits
            splits = ticker.splits
            if not splits.empty:
                recent_splits = splits.tail(3)
                print(f"  🔀 Recent Splits ({len(recent_splits)} shown):")
                for date, ratio in recent_splits.items():
                    try:
                        date_str = date.strftime('%Y-%m-%d')
                    except:
                        date_str = str(date)[:10]
                    print(f"    {date_str}: {ratio:.1f}:1")
            else:
                print(f"  🔀 Splits: None found")
            
            # Show company info
            try:
                info = ticker.info
                sector = info.get("sector", "Unknown")
                market_cap = info.get("marketCap", 0)
                print(f"  🏭 Sector: {sector}")
                print(f"  💹 Market Cap: ${market_cap:,.0f}" if market_cap else "  💹 Market Cap: Unknown")
            except:
                print(f"  ℹ️ Company info: Not available")
                
        except Exception as e:
            print(f"  ❌ Error fetching {symbol}: {e}")
    
    print(f"\n💡 Note: This data is used to create features like:")
    print(f"   - is_dividend_day, is_split_day (categorical)")
    print(f"   - days_to_next_dividend, days_since_split (numeric)")
    print(f"   - sector (static categorical)")
    print(f"   - market_cap (static real)")

