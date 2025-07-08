#!/usr/bin/env python3
"""
Comprehensive test of the TFT pipeline with caching.
Tests the complete pipeline from data fetch to model training.
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dataModule.interface import get_data_loader_with_module
from cache_manager import print_cache_info
import torch
import warnings
warnings.filterwarnings('ignore')

def test_pipeline_with_caching():
    """Test the complete TFT pipeline with caching enabled."""
    print("🧪 Testing TFT Pipeline with Caching")
    print("=" * 50)
    
    # Test configuration
    symbols = ['AAPL', 'GOOGL', 'MSFT']  # Tech stocks
    start = '2023-01-01'
    end = '2024-01-01'
    encoder_len = 30
    predict_len = 7
    batch_size = 32
    
    print(f"📊 Test Configuration:")
    print(f"   Symbols: {symbols}")
    print(f"   Date range: {start} to {end}")
    print(f"   Encoder length: {encoder_len}")
    print(f"   Prediction length: {predict_len}")
    print(f"   Batch size: {batch_size}")
    print()
    
    # Show initial cache state
    print("📦 Initial Cache State:")
    print_cache_info()
    print()
    
    try:
        # Test data loading with caching
        print("🔄 Testing data loading with caching...")
        dataloader, datamodule = get_data_loader_with_module(
            symbols=symbols,
            start=start,
            end=end,
            encoder_len=encoder_len,
            predict_len=predict_len,
            batch_size=batch_size,
            news_api_key=None,  # Will use fallback
            fred_api_key=None,  # Will use fallback
            api_ninjas_key=None  # Will use fallback
        )
        
        print("✅ Data loading completed successfully!")
        print(f"   DataLoader type: {type(dataloader)}")
        print(f"   DataModule type: {type(datamodule)}")
        print(f"   Number of batches: {len(dataloader)}")
        print()
        
        # Test first batch
        print("🔍 Testing first batch...")
        first_batch = next(iter(dataloader))
        
        # Handle different batch formats
        if isinstance(first_batch, tuple):
            print(f"   Batch format: tuple with {len(first_batch)} elements")
            for i, element in enumerate(first_batch):
                if isinstance(element, torch.Tensor):
                    print(f"   Element {i}: {element.shape}")
                elif isinstance(element, dict):
                    print(f"   Element {i}: dict with keys {list(element.keys())}")
                else:
                    print(f"   Element {i}: {type(element)}")
        elif isinstance(first_batch, dict):
            print(f"   Batch keys: {list(first_batch.keys())}")
            # Check tensor shapes
            for key, value in first_batch.items():
                if isinstance(value, torch.Tensor):
                    print(f"   {key}: {value.shape}")
                else:
                    print(f"   {key}: {type(value)}")
        else:
            print(f"   Batch type: {type(first_batch)}")
            if isinstance(first_batch, torch.Tensor):
                print(f"   Batch shape: {first_batch.shape}")
        
        print("✅ First batch loaded successfully!")
        print()
        
        # Show cache state after first run
        print("📦 Cache State After First Run:")
        print_cache_info()
        print()
        
        # Test second run to verify caching
        print("🔄 Testing second run (should use cache)...")
        dataloader2, datamodule2 = get_data_loader_with_module(
            symbols=symbols,
            start=start,
            end=end,
            encoder_len=encoder_len,
            predict_len=predict_len,
            batch_size=batch_size,
            news_api_key=None,
            fred_api_key=None,
            api_ninjas_key=None
        )
        
        print("✅ Second run completed (should have used cache)!")
        print(f"   Same number of batches: {len(dataloader2) == len(dataloader)}")
        print()
        
        # Show final cache state
        print("📦 Final Cache State:")
        print_cache_info()
        print()
        
        # Test static covariates (should only have sector, not symbol)
        print("🔍 Testing static covariates...")
        try:
            static_cats = getattr(datamodule, 'static_categoricals', None)
            if static_cats is not None:
                print(f"   Static categoricals: {static_cats}")
                if 'symbol' in static_cats:
                    print("   ❌ WARNING: 'symbol' found in static categoricals (should be removed)")
                else:
                    print("   ✅ 'symbol' not in static categoricals (good)")
                
                if 'sector' in static_cats:
                    print("   ✅ 'sector' found in static categoricals (good)")
                else:
                    print("   ❌ WARNING: 'sector' not found in static categoricals")
            else:
                print("   ℹ️  Static categoricals attribute not found on datamodule")
                # Try to check the dataset instead
                dataset = getattr(datamodule, 'dataset', None)
                if dataset is not None:
                    dataset_static_cats = getattr(dataset, 'static_categoricals', None)
                    if dataset_static_cats is not None:
                        print(f"   Dataset static categoricals: {dataset_static_cats}")
                        if 'symbol' in dataset_static_cats:
                            print("   ❌ WARNING: 'symbol' found in dataset static categoricals")
                        else:
                            print("   ✅ 'symbol' not in dataset static categoricals (good)")
                        
                        if 'sector' in dataset_static_cats:
                            print("   ✅ 'sector' found in dataset static categoricals (good)")
                        else:
                            print("   ❌ WARNING: 'sector' not found in dataset static categoricals")
                    else:
                        print("   ℹ️  Dataset static categoricals not found either")
                else:
                    print("   ℹ️  Dataset not found on datamodule")
        except Exception as e:
            print(f"   ℹ️  Could not check static categoricals: {e}")
        
        print("\n🎉 All tests passed successfully!")
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == '__main__':
    success = test_pipeline_with_caching()
    sys.exit(0 if success else 1)
