"""
Data caching module for TFT pipeline.
Implements caching for stock data, news embeddings, FRED data, and technical indicators.
"""

import os
import pickle
import json
import hashlib
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
import warnings
warnings.filterwarnings('ignore')


class DataCache:
    """
    Unified data cache for all TFT data sources.
    Handles caching of stock data, news embeddings, FRED data, and technical indicators.
    """
    
    def __init__(self, cache_dir: str = "cache"):
        """
        Initialize the data cache.
        
        Args:
            cache_dir: Directory to store cache files
        """
        self.cache_dir = cache_dir
        self.ensure_cache_dir()
        
        # Cache validity periods (in hours)
        self.cache_validity = {
            'stock_data': 1,      # Stock data: 1 hour (market updates frequently)
            'news_data': 6,       # News data: 6 hours (news updates less frequently)
            'fred_data': 24,      # FRED data: 24 hours (economic data updates daily)
            'ta_data': 1,         # Technical indicators: 1 hour (derived from stock data)
            'events_data': 24,    # Events data: 24 hours (corporate events don't change often)
            'features': 1         # Feature matrix: 1 hour (depends on all above)
        }
        
        print(f"📦 DataCache initialized with cache directory: {self.cache_dir}")
        print(f"   Cache validity periods: {self.cache_validity}")
    
    def ensure_cache_dir(self):
        """Create cache directory if it doesn't exist."""
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
            print(f"   Created cache directory: {self.cache_dir}")
    
    def _get_cache_key(self, data_type: str, **kwargs) -> str:
        """
        Generate a unique cache key based on data type and parameters.
        
        Args:
            data_type: Type of data ('stock_data', 'news_data', etc.)
            **kwargs: Parameters used to fetch the data
            
        Returns:
            MD5 hash string to use as cache key
        """
        # Create a string representation of all parameters
        key_parts = [data_type]
        
        # Sort kwargs for consistent key generation
        for key, value in sorted(kwargs.items()):
            if isinstance(value, list):
                key_parts.append(f"{key}={','.join(sorted(map(str, value)))}")
            else:
                key_parts.append(f"{key}={value}")
        
        key_string = "|".join(key_parts)
        cache_key = hashlib.md5(key_string.encode()).hexdigest()
        
        return cache_key
    
    def _get_cache_path(self, cache_key: str, data_type: str) -> str:
        """Get the full path for a cache file."""
        extension = '.pkl' if data_type in ['stock_data', 'ta_data', 'features', 'news_data', 'fred_data'] else '.json'
        return os.path.join(self.cache_dir, f"{cache_key}_{data_type}{extension}")
    
    def _get_metadata_path(self, cache_key: str, data_type: str) -> str:
        """Get the path for cache metadata file."""
        return os.path.join(self.cache_dir, f"{cache_key}_{data_type}_meta.json")
    
    def is_cache_valid(self, cache_key: str, data_type: str) -> bool:
        """
        Check if cached data is still valid based on timestamp.
        
        Args:
            cache_key: Cache key
            data_type: Type of data
            
        Returns:
            True if cache is valid, False otherwise
        """
        metadata_path = self._get_metadata_path(cache_key, data_type)
        
        if not os.path.exists(metadata_path):
            return False
        
        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            cached_time = datetime.fromisoformat(metadata['timestamp'])
            validity_hours = self.cache_validity.get(data_type, 24)
            expiry_time = cached_time + timedelta(hours=validity_hours)
            
            is_valid = datetime.now() < expiry_time
            
            if is_valid:
                time_left = expiry_time - datetime.now()
                print(f"   ✅ Cache hit for {data_type} (expires in {time_left})")
            else:
                print(f"   ⏰ Cache expired for {data_type}")
            
            return is_valid
            
        except Exception as e:
            print(f"   ❌ Error reading cache metadata: {e}")
            return False
    
    def save_to_cache(self, data: Any, cache_key: str, data_type: str, **metadata) -> None:
        """
        Save data to cache with metadata.
        
        Args:
            data: Data to cache
            cache_key: Cache key
            data_type: Type of data
            **metadata: Additional metadata to store
        """
        try:
            cache_path = self._get_cache_path(cache_key, data_type)
            metadata_path = self._get_metadata_path(cache_key, data_type)
            
            # Save data
            if data_type in ['stock_data', 'ta_data', 'features', 'news_data', 'fred_data']:
                # Save pandas DataFrames as pickle
                with open(cache_path, 'wb') as f:
                    pickle.dump(data, f)
            else:
                # Save other data as JSON
                with open(cache_path, 'w') as f:
                    json.dump(data, f, indent=2, default=str)
            
            # Save metadata
            meta_data = {
                'timestamp': datetime.now().isoformat(),
                'data_type': data_type,
                'cache_key': cache_key,
                **metadata
            }
            
            with open(metadata_path, 'w') as f:
                json.dump(meta_data, f, indent=2)
            
            file_size = os.path.getsize(cache_path) / 1024  # KB
            print(f"   💾 Cached {data_type} ({file_size:.1f} KB)")
            
        except Exception as e:
            print(f"   ❌ Error saving to cache: {e}")
    
    def load_from_cache(self, cache_key: str, data_type: str) -> Optional[Any]:
        """
        Load data from cache.
        
        Args:
            cache_key: Cache key
            data_type: Type of data
            
        Returns:
            Cached data or None if not found/invalid
        """
        if not self.is_cache_valid(cache_key, data_type):
            return None
        
        try:
            cache_path = self._get_cache_path(cache_key, data_type)
            
            if data_type in ['stock_data', 'ta_data', 'features', 'news_data', 'fred_data']:
                # Load pandas DataFrames from pickle
                with open(cache_path, 'rb') as f:
                    data = pickle.load(f)
            else:
                # Load other data from JSON
                with open(cache_path, 'r') as f:
                    data = json.load(f)
            
            return data
            
        except Exception as e:
            print(f"   ❌ Error loading from cache: {e}")
            return None
    
    def get_or_fetch_stock_data(self, symbols: List[str], start: str, end: str, 
                               fetch_func) -> pd.DataFrame:
        """
        Get stock data from cache or fetch if not available.
        
        Args:
            symbols: List of stock symbols
            start: Start date
            end: End date
            fetch_func: Function to fetch data if not cached
            
        Returns:
            Stock data DataFrame
        """
        cache_key = self._get_cache_key('stock_data', symbols=symbols, start=start, end=end)
        
        # Try to load from cache
        cached_data = self.load_from_cache(cache_key, 'stock_data')
        if cached_data is not None:
            return cached_data
        
        # Fetch fresh data
        print(f"   🔄 Fetching fresh stock data for {len(symbols)} symbols...")
        data = fetch_func(symbols, start, end)
        
        # Cache the result
        self.save_to_cache(data, cache_key, 'stock_data', 
                          symbols=symbols, start=start, end=end, 
                          num_symbols=len(symbols), num_rows=len(data))
        
        return data
    
    def get_or_fetch_news_data(self, symbols: List[str], start: str, end: str,
                              api_key: Optional[str], fetch_func) -> pd.DataFrame:
        """Get news data from cache or fetch if not available."""
        cache_key = self._get_cache_key('news_data', symbols=symbols, start=start, end=end,
                                      has_api_key=api_key is not None)
        
        # Try to load from cache
        cached_data = self.load_from_cache(cache_key, 'news_data')
        if cached_data is not None:
            return cached_data
        
        # Fetch fresh data
        print(f"   🔄 Fetching fresh news data...")
        data = fetch_func(symbols, start, end, api_key)
        
        # Cache the result
        self.save_to_cache(data, cache_key, 'news_data',
                          symbols=symbols, start=start, end=end,
                          num_symbols=len(symbols), num_rows=len(data))
        
        return data
    
    def get_or_fetch_fred_data(self, start: str, end: str, api_key: Optional[str],
                              fetch_func) -> pd.DataFrame:
        """Get FRED data from cache or fetch if not available."""
        cache_key = self._get_cache_key('fred_data', start=start, end=end,
                                      has_api_key=api_key is not None)
        
        # Try to load from cache
        cached_data = self.load_from_cache(cache_key, 'fred_data')
        if cached_data is not None:
            return cached_data
        
        # Fetch fresh data
        print(f"   🔄 Fetching fresh FRED data...")
        data = fetch_func(start, end, api_key)
        
        # Cache the result
        self.save_to_cache(data, cache_key, 'fred_data',
                          start=start, end=end, num_rows=len(data))
        
        return data
    
    def get_or_fetch_ta_data(self, stock_df: pd.DataFrame, fetch_func) -> pd.DataFrame:
        """Get technical analysis data from cache or compute if not available."""
        # Create a hash of the stock data to use as part of cache key
        stock_hash = hashlib.md5(str(stock_df.shape).encode()).hexdigest()[:8]
        cache_key = self._get_cache_key('ta_data', stock_hash=stock_hash)
        
        # Try to load from cache
        cached_data = self.load_from_cache(cache_key, 'ta_data')
        if cached_data is not None:
            return cached_data
        
        # Compute fresh data
        print(f"   🔄 Computing fresh technical indicators...")
        data = fetch_func(stock_df)
        
        # Cache the result
        self.save_to_cache(data, cache_key, 'ta_data',
                          stock_shape=stock_df.shape, num_rows=len(data))
        
        return data
    
    def get_or_fetch_events_data(self, symbols: List[str], start: str, end: str,
                                earnings_api_key: Optional[str], api_ninjas_key: Optional[str],
                                fetch_func) -> Dict:
        """Get events data from cache or fetch if not available."""
        cache_key = self._get_cache_key('events_data', symbols=symbols, start=start, end=end,
                                      has_earnings_key=earnings_api_key is not None,
                                      has_ninjas_key=api_ninjas_key is not None)
        
        # Try to load from cache
        cached_data = self.load_from_cache(cache_key, 'events_data')
        if cached_data is not None:
            return cached_data
        
        # Fetch fresh data
        print(f"   🔄 Fetching fresh events data...")
        data = fetch_func(symbols, start, end, earnings_api_key, api_ninjas_key)
        
        # Cache the result
        self.save_to_cache(data, cache_key, 'events_data',
                          symbols=symbols, start=start, end=end,
                          num_symbols=len(symbols))
        
        return data
    
    def get_or_build_features(self, stock_df: pd.DataFrame, events: Dict, news_df: pd.DataFrame,
                             ta_df: pd.DataFrame, fred_df: pd.DataFrame, encoder_len: int,
                             predict_len: int, build_func) -> pd.DataFrame:
        """Get feature matrix from cache or build if not available."""
        # Create a hash based on all input data shapes and parameters
        # Handle both DataFrames and other types safely
        try:
            stock_shape = stock_df.shape if hasattr(stock_df, 'shape') else str(stock_df)
            events_len = len(events) if hasattr(events, '__len__') else str(events)
            news_shape = news_df.shape if hasattr(news_df, 'shape') else str(news_df)
            ta_shape = ta_df.shape if hasattr(ta_df, 'shape') else str(ta_df)
            fred_shape = fred_df.shape if hasattr(fred_df, 'shape') else str(fred_df)
            
            inputs_hash = hashlib.md5(
                f"{stock_shape}{events_len}{news_shape}{ta_shape}{fred_shape}{encoder_len}{predict_len}".encode()
            ).hexdigest()[:12]
        except Exception as e:
            # Fallback to simple hash if there are issues
            inputs_hash = hashlib.md5(
                f"{encoder_len}{predict_len}{str(len(events))}".encode()
            ).hexdigest()[:12]
        
        cache_key = self._get_cache_key('features', inputs_hash=inputs_hash,
                                      encoder_len=encoder_len, predict_len=predict_len)
        
        # Try to load from cache
        cached_data = self.load_from_cache(cache_key, 'features')
        if cached_data is not None:
            return cached_data
        
        # Build fresh feature matrix
        print(f"   🔄 Building fresh feature matrix...")
        data = build_func(stock_df, events, news_df, ta_df, fred_df, encoder_len, predict_len)
        
        # Cache the result
        self.save_to_cache(data, cache_key, 'features',
                          encoder_len=encoder_len, predict_len=predict_len,
                          num_rows=len(data), num_cols=len(data.columns) if not data.empty else 0)
        
        return data
    
    def clear_cache(self, data_type: Optional[str] = None) -> None:
        """
        Clear cache files.
        
        Args:
            data_type: Specific data type to clear, or None to clear all
        """
        if not os.path.exists(self.cache_dir):
            return
        
        files_removed = 0
        for filename in os.listdir(self.cache_dir):
            if data_type is None or data_type in filename:
                file_path = os.path.join(self.cache_dir, filename)
                try:
                    os.remove(file_path)
                    files_removed += 1
                except Exception as e:
                    print(f"   ❌ Error removing {filename}: {e}")
        
        print(f"   🗑️ Cleared {files_removed} cache files" + (f" for {data_type}" if data_type else ""))
    
    def get_cache_info(self) -> Dict[str, Any]:
        """Get information about cached data."""
        if not os.path.exists(self.cache_dir):
            return {"total_files": 0, "total_size_mb": 0, "cache_types": {}}
        
        cache_info = {"total_files": 0, "total_size_mb": 0, "cache_types": {}}
        
        for filename in os.listdir(self.cache_dir):
            if filename.endswith(('.pkl', '.json')) and not filename.endswith('_meta.json'):
                file_path = os.path.join(self.cache_dir, filename)
                file_size = os.path.getsize(file_path)
                
                cache_info["total_files"] += 1
                cache_info["total_size_mb"] += file_size / (1024 * 1024)
                
                # Determine cache type
                for data_type in self.cache_validity.keys():
                    if data_type in filename:
                        if data_type not in cache_info["cache_types"]:
                            cache_info["cache_types"][data_type] = {"count": 0, "size_mb": 0}
                        cache_info["cache_types"][data_type]["count"] += 1
                        cache_info["cache_types"][data_type]["size_mb"] += file_size / (1024 * 1024)
                        break
        
        return cache_info


# Global cache instance
_cache_instance = None

def get_cache_instance(cache_dir: str = "cache") -> DataCache:
    """Get the global cache instance."""
    global _cache_instance
    if _cache_instance is None:
        _cache_instance = DataCache(cache_dir)
    return _cache_instance

def clear_all_cache():
    """Clear all cached data."""
    cache = get_cache_instance()
    cache.clear_cache()

def print_cache_info():
    """Print information about cached data."""
    cache = get_cache_instance()
    info = cache.get_cache_info()
    
    print(f"📦 Cache Information:")
    print(f"   Total files: {info['total_files']}")
    print(f"   Total size: {info['total_size_mb']:.2f} MB")
    print(f"   Cache types:")
    for data_type, type_info in info['cache_types'].items():
        print(f"     {data_type}: {type_info['count']} files, {type_info['size_mb']:.2f} MB")
