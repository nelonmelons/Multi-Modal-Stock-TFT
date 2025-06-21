"""
Helper function to create TFT DataLoader with automatic parameter adjustment for small datasets.
"""

from typing import List, Optional, Tuple
import pandas as pd
from torch.utils.data import DataLoader
from .datamodule import TFTDataModule


def create_adaptive_dataloader(feature_df: pd.DataFrame, 
                             encoder_len: int, 
                             predict_len: int, 
                             batch_size: int) -> Tuple[DataLoader, TFTDataModule]:
    """
    Create a TFT DataLoader with automatic parameter adjustment for small datasets.
    
    Args:
        feature_df: Feature DataFrame
        encoder_len: Desired encoder length
        predict_len: Desired prediction length  
        batch_size: Desired batch size
        
    Returns:
        Tuple of (DataLoader, TFTDataModule)
    """
    total_samples = len(feature_df)
    
    # Calculate optimal parameters for small datasets
    if total_samples < 20:
        print(f"📊 Small dataset detected ({total_samples} samples)")
        print("🔧 Auto-adjusting parameters for compatibility...")
        
        # Adjust encoder length
        max_encoder = max(1, (total_samples - predict_len - 3) // 2)
        adjusted_encoder = min(encoder_len, max_encoder)
        
        # Adjust prediction length if necessary
        adjusted_predict = min(predict_len, max(1, total_samples // 4))
        
        # Adjust batch size
        adjusted_batch = min(batch_size, max(1, total_samples // 4))
        
        print(f"   Original: encoder={encoder_len}, predict={predict_len}, batch={batch_size}")
        print(f"   Adjusted: encoder={adjusted_encoder}, predict={adjusted_predict}, batch={adjusted_batch}")
        
        encoder_len = adjusted_encoder
        predict_len = adjusted_predict
        batch_size = adjusted_batch
    
    # Create data module with adjusted parameters
    data_module = TFTDataModule(feature_df, encoder_len, predict_len, batch_size)
    
    try:
        data_module.setup()
        return data_module.train_dataloader(), data_module
    except Exception as e:
        print(f"❌ Failed to create DataLoader even with adjusted parameters: {e}")
        
        # Last resort: very conservative parameters
        print("🔧 Trying minimal parameters...")
        conservative_encoder = max(1, total_samples // 6)
        conservative_predict = 1
        conservative_batch = 1
        
        print(f"   Conservative: encoder={conservative_encoder}, predict={conservative_predict}, batch={conservative_batch}")
        
        data_module = TFTDataModule(feature_df, conservative_encoder, conservative_predict, conservative_batch)
        data_module.setup()
        return data_module.train_dataloader(), data_module
