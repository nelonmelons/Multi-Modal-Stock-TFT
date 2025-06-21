#!/usr/bin/env python3
"""
🎯 TFT Tensor Shape Analyzer

Simple script to get tensor shapes from the TFT data pipeline.
"""

import sys
import os
from datetime import datetime, timedelta

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data import get_data_loader_with_module


def print_tensor_shapes(batch):
    """Print tensor shapes in a clean format."""
    print("📊 TENSOR SHAPES")
    print("=" * 30)
    
    if isinstance(batch, dict):
        print(f"Batch format: Dictionary with {len(batch)} keys\n")
        for key, value in batch.items():
            if hasattr(value, 'shape'):
                print(f"{key:25} : {str(value.shape):20} | {str(value.dtype)}")
            else:
                print(f"{key:25} : {str(type(value)):20} | {value}")
                
    elif isinstance(batch, (tuple, list)):
        print(f"Batch format: {type(batch).__name__} with {len(batch)} elements\n")
        for i, item in enumerate(batch):
            if hasattr(item, 'shape'):
                print(f"Element_{i:2d}             : {str(item.shape):20} | {str(item.dtype)}")
                
    elif hasattr(batch, 'shape'):
        print(f"Single tensor         : {str(batch.shape):20} | {str(batch.dtype)}")
        
    else:
        print(f"Batch type: {type(batch)}")
        print(f"Content: {batch}")


def main():
    """Get TFT tensors and print their shapes."""
    
    print("🚀 Getting TFT Tensor Shapes")
    print("=" * 35)
    
    # Quick configuration for testing
    symbols = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'META']  # Expanded list of tech stocks
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=90)).strftime('%Y-%m-%d')
    encoder_len = 15
    predict_len = 3
    batch_size = 4
    
    print(f"Symbols: {symbols}")
    print(f"Date range: {start_date} to {end_date}")
    print(f"Encoder length: {encoder_len}, Predict length: {predict_len}")
    print(f"Batch size: {batch_size}\n")
    
    try:
        # Get data loader
        print("Creating TFT DataLoader...")
        dataloader, datamodule = get_data_loader_with_module(
            symbols=symbols,
            start=start_date,
            end=end_date,
            encoder_len=encoder_len,
            predict_len=predict_len,
            batch_size=batch_size,
            news_api_key=os.getenv('NEWS_API_KEY'),
            fred_api_key=os.getenv('FRED_API_KEY'),
            api_ninjas_key=os.getenv('API_NINJAS_KEY')
        )
        
        print(f"✅ DataLoader created with {len(dataloader)} batches\n")
        
        # Get first batch and print shapes
        first_batch = next(iter(dataloader))
        print_tensor_shapes(first_batch)
        
        print(f"\n📈 Dataset Info:")
        print(f"Train samples: {len(datamodule.train_dataset)}")
        print(f"Val samples: {len(datamodule.val_dataset)}")
        
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()
