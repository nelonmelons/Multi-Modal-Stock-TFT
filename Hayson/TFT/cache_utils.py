#!/usr/bin/env python3
"""
Cache management utility for TFT pipeline.
Provides tools to view, clear, and manage cached data.
"""

import argparse
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from cache_manager import get_cache_instance, clear_all_cache, print_cache_info


def main():
    """Main function for cache management CLI."""
    parser = argparse.ArgumentParser(description='Manage TFT pipeline cache')
    parser.add_argument('--info', action='store_true', 
                       help='Show cache information')
    parser.add_argument('--clear', type=str, nargs='?', const='all',
                       help='Clear cache (all, stock_data, news_data, fred_data, ta_data, events_data, features)')
    parser.add_argument('--cache-dir', type=str, default='cache',
                       help='Cache directory path (default: cache)')
    
    args = parser.parse_args()
    
    # Initialize cache with specified directory
    cache = get_cache_instance(args.cache_dir)
    
    if args.info:
        print_cache_info()
        return
    
    if args.clear:
        if args.clear == 'all':
            cache.clear_cache()
            print("✅ All cache cleared")
        else:
            cache.clear_cache(args.clear)
            print(f"✅ Cache cleared for {args.clear}")
        return
    
    # If no specific action, show help
    parser.print_help()


if __name__ == '__main__':
    main()
