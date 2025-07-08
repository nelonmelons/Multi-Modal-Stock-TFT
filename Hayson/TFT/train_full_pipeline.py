#!/usr/bin/env python3
"""
Full TFT training pipeline with caching.
Trains a Temporal Fusion Transformer on tech stocks with comprehensive data sources.
"""

import os
import sys
import warnings
warnings.filterwarnings('ignore')

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
import numpy as np
from datetime import datetime, timedelta

from dataModule.interface import get_data_loader_with_module
from tft_multimodal import TFTMultiModal
from cache_manager import print_cache_info, clear_all_cache

def main():
    """Main training function."""
    print("🚀 Starting TFT Full Pipeline Training")
    print("=" * 60)
    
    # Training configuration
    config = {
        # Data configuration
        'symbols': ['AAPL', 'GOOGL', 'MSFT', 'NVDA', 'TSLA'],  # Tech stocks
        'start_date': '2022-01-01',
        'end_date': '2024-01-01',
        'encoder_len': 60,      # 60 trading days of history
        'predict_len': 10,      # Predict 10 days ahead
        'batch_size': 32,
        
        # Model configuration
        'hidden_size': 256,
        'attention_head_size': 8,
        'dropout': 0.1,
        'hidden_continuous_size': 16,
        'output_size': 7,
        'loss': 'QuantileLoss',
        'learning_rate': 0.001,
        
        # Training configuration
        'max_epochs': 100,
        'patience': 15,
        'gradient_clip_val': 1.0,
        'accumulate_grad_batches': 1,
        'val_check_interval': 0.25,
        
        # Hardware configuration
        'accelerator': 'mps' if torch.backends.mps.is_available() else 'cpu',
        'devices': 1,
        'precision': 16 if torch.backends.mps.is_available() else 32
    }
    
    print("📊 Training Configuration:")
    for key, value in config.items():
        print(f"   {key}: {value}")
    print()
    
    # Check cache state
    print("📦 Initial Cache State:")
    print_cache_info()
    print()
    
    # Option to clear cache
    clear_cache = input("Clear cache before training? (y/n): ").lower().strip()
    if clear_cache == 'y':
        clear_all_cache()
        print("✅ Cache cleared")
        print()
    
    try:
        # Step 1: Load data with caching
        print("🔄 Loading data with caching...")
        dataloader, datamodule = get_data_loader_with_module(
            symbols=config['symbols'],
            start=config['start_date'],
            end=config['end_date'],
            encoder_len=config['encoder_len'],
            predict_len=config['predict_len'],
            batch_size=config['batch_size'],
            news_api_key=None,  # Will use fallback
            fred_api_key=None,  # Will use fallback
            api_ninjas_key=None  # Will use fallback
        )
        
        print("✅ Data loaded successfully!")
        print(f"   Training batches: {len(dataloader)}")
        print(f"   Feature matrix shape: {datamodule.feature_df.shape}")
        print()
        
        # Step 2: Initialize model
        print("🧠 Initializing TFT model...")
        
        # Get feature dimensions from datamodule
        num_static_categoricals = len(datamodule.static_categoricals)
        num_static_reals = len(datamodule.static_reals) 
        num_known_categoricals = len(datamodule.time_varying_known_categoricals)
        num_known_reals = len(datamodule.time_varying_known_reals)
        num_unknown_reals = len(datamodule.time_varying_unknown_reals)
        
        print(f"   Static categoricals: {num_static_categoricals}")
        print(f"   Static reals: {num_static_reals}")
        print(f"   Known categoricals: {num_known_categoricals}")
        print(f"   Known reals: {num_known_reals}")
        print(f"   Unknown reals: {num_unknown_reals}")
        
        # Initialize model
        model = TFTMultiModal(
            static_categoricals=datamodule.static_categoricals,
            static_reals=datamodule.static_reals,
            time_varying_known_categoricals=datamodule.time_varying_known_categoricals,
            time_varying_known_reals=datamodule.time_varying_known_reals,
            time_varying_unknown_reals=datamodule.time_varying_unknown_reals,
            target=datamodule.target,
            hidden_size=config['hidden_size'],
            attention_head_size=config['attention_head_size'],
            dropout=config['dropout'],
            hidden_continuous_size=config['hidden_continuous_size'],
            output_size=config['output_size'],
            loss=config['loss'],
            learning_rate=config['learning_rate'],
            reduce_on_plateau_patience=10,
            reduce_on_plateau_reduction=0.5,
            optimizer='AdamW'
        )
        
        print("✅ Model initialized successfully!")
        print(f"   Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        print()
        
        # Step 3: Setup training callbacks
        print("⚙️  Setting up training callbacks...")
        
        # Create run directory with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"tft_multimodal_{timestamp}"
        log_dir = f"lightning_logs/{run_name}"
        checkpoint_dir = f"checkpoints/{run_name}"
        
        # Callbacks
        callbacks = [
            ModelCheckpoint(
                dirpath=checkpoint_dir,
                filename='{epoch}-{val_loss:.4f}',
                monitor='val_loss',
                mode='min',
                save_top_k=3,
                save_last=True,
                verbose=True
            ),
            EarlyStopping(
                monitor='val_loss',
                mode='min',
                patience=config['patience'],
                verbose=True,
                strict=False
            ),
            LearningRateMonitor(logging_interval='step')
        ]
        
        # Logger
        logger = TensorBoardLogger(
            save_dir="lightning_logs",
            name=run_name,
            version=None
        )
        
        print(f"   Log directory: {log_dir}")
        print(f"   Checkpoint directory: {checkpoint_dir}")
        print()
        
        # Step 4: Setup trainer
        print("🏋️ Setting up PyTorch Lightning trainer...")
        
        trainer = pl.Trainer(
            max_epochs=config['max_epochs'],
            accelerator=config['accelerator'],
            devices=config['devices'],
            precision=config['precision'],
            gradient_clip_val=config['gradient_clip_val'],
            accumulate_grad_batches=config['accumulate_grad_batches'],
            val_check_interval=config['val_check_interval'],
            callbacks=callbacks,
            logger=logger,
            enable_progress_bar=True,
            enable_model_summary=True,
            log_every_n_steps=10
        )
        
        print("✅ Trainer configured successfully!")
        print(f"   Max epochs: {config['max_epochs']}")
        print(f"   Accelerator: {config['accelerator']}")
        print(f"   Precision: {config['precision']}")
        print()
        
        # Step 5: Start training
        print("🎯 Starting training...")
        print("=" * 60)
        
        trainer.fit(model, datamodule)
        
        print("=" * 60)
        print("✅ Training completed!")
        
        # Step 6: Save final model
        final_model_path = f"{checkpoint_dir}/final_model.ckpt"
        trainer.save_checkpoint(final_model_path)
        print(f"📁 Final model saved to: {final_model_path}")
        
        # Step 7: Print training summary
        print("\n📊 Training Summary:")
        print(f"   Best model checkpoint: {trainer.checkpoint_callback.best_model_path}")
        print(f"   Best validation loss: {trainer.checkpoint_callback.best_model_score:.4f}")
        print(f"   Total epochs: {trainer.current_epoch + 1}")
        print(f"   Total steps: {trainer.global_step}")
        
        # Step 8: Test the model
        print("\n🧪 Testing the trained model...")
        test_results = trainer.test(model, datamodule)
        print(f"   Test results: {test_results}")
        
        # Step 9: Final cache state
        print("\n📦 Final Cache State:")
        print_cache_info()
        
        print("\n🎉 Full pipeline training completed successfully!")
        print(f"   View logs with: tensorboard --logdir lightning_logs/{run_name}")
        print(f"   Model checkpoints: {checkpoint_dir}")
        
    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted by user")
        print("📦 Cache state preserved for next run")
        print_cache_info()
        
    except Exception as e:
        print(f"\n❌ Training failed with error: {e}")
        import traceback
        traceback.print_exc()
        print("\n📦 Cache state:")
        print_cache_info()
        return False
    
    return True

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
