#!/usr/bin/env python3
"""
Continue Training Script for TFT Model

This script allows loading a pretrained TFT model and continuing training.
"""

import argparse
import torch
import json
from pathlib import Path
from typing import Dict, Any
import warnings
warnings.filterwarnings('ignore')

from unified_tft_pipeline import TFTTrainer
from dataModule.interface import get_data_loader_with_module


def load_pretrained_model(checkpoint_path: str, trainer: TFTTrainer) -> Dict[str, Any]:
    """Load a pretrained model and return its configuration."""
    checkpoint_path = Path(checkpoint_path)
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    print(f"📂 Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=trainer.device)
    
    # Load the configuration from checkpoint
    config = checkpoint.get('config', {})
    
    # Load model state
    if trainer.model is not None:
        trainer.model.load_state_dict(checkpoint['model_state_dict'])
        print("✅ Model weights loaded successfully")
    
    # Load optimizer state
    if trainer.optimizer is not None:
        trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print("✅ Optimizer state loaded successfully")
    
    # Load scheduler state
    if trainer.scheduler is not None:
        trainer.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        print("✅ Scheduler state loaded successfully")
    
    # Load training history
    trainer.train_losses = checkpoint.get('train_losses', [])
    trainer.val_losses = checkpoint.get('val_losses', [])
    
    print(f"📊 Model info:")
    print(f"   - Trained for {checkpoint.get('epoch', 0)} epochs")
    print(f"   - Best training loss: {min(trainer.train_losses) if trainer.train_losses else 'N/A'}")
    print(f"   - Best validation loss: {min(trainer.val_losses) if trainer.val_losses else 'N/A'}")
    
    return config


def update_config_for_continuation(base_config: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    """Update configuration for continued training."""
    # Update training parameters
    if args.epochs:
        base_config['epochs'] = args.epochs
    if args.learning_rate:
        base_config['learning_rate'] = args.learning_rate
    if args.batch_size:
        base_config['batch_size'] = args.batch_size
    if args.symbols:
        base_config['symbols'] = args.symbols.split(',')
    
    # Add continuation-specific parameters
    base_config['auto_continue'] = args.auto_continue
    base_config['enhanced_model'] = args.enhanced_model
    
    return base_config


def main():
    parser = argparse.ArgumentParser(description='Continue Training TFT Model')
    parser.add_argument('--checkpoint', type=str, required=True, 
                       help='Path to the checkpoint file to continue from')
    parser.add_argument('--epochs', type=int, default=10,
                       help='Additional epochs to train (default: 10)')
    parser.add_argument('--learning-rate', type=float, 
                       help='New learning rate (optional, uses checkpoint value if not provided)')
    parser.add_argument('--batch-size', type=int,
                       help='Batch size (optional, uses checkpoint value if not provided)')
    parser.add_argument('--symbols', type=str,
                       help='Comma-separated symbols (optional, uses checkpoint value if not provided)')
    parser.add_argument('--auto-continue', action='store_true',
                       help='Auto-continue training without user prompts')
    parser.add_argument('--enhanced-model', action='store_true',
                       help='Use enhanced TFT model (must match original checkpoint)')
    parser.add_argument('--output-suffix', type=str, default='continued',
                       help='Suffix for output directory (default: continued)')
    
    args = parser.parse_args()
    
    print("🔄 TFT Model Continuation Training")
    print("=" * 50)
    
    # Load checkpoint to get original configuration
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        return
    
    # Load original configuration
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    original_config = checkpoint.get('config', {})
    
    # Update configuration with new parameters
    config = update_config_for_continuation(original_config, args)
    
    print("📊 Updated Configuration:")
    for key, value in config.items():
        print(f"   {key}: {value}")
    print()
    
    # Create trainer with updated config
    trainer = TFTTrainer(config)
    
    # Load data
    print("🔄 Loading data...")
    dataloader, datamodule = trainer.load_data()
    
    # Initialize model with same architecture as checkpoint
    sample_batch = next(iter(dataloader))
    trainer.initialize_model(sample_batch)
    
    # Load pretrained weights
    pretrained_config = load_pretrained_model(args.checkpoint, trainer)
    
    # Verify model compatibility
    if config.get('enhanced_model', False) != pretrained_config.get('enhanced_model', False):
        print("⚠️  Warning: Enhanced model setting doesn't match checkpoint!")
        print("   This may cause compatibility issues.")
    
    # Update output directory to avoid overwriting
    original_output = trainer.output_dir
    new_output = original_output.parent / f"{original_output.name}_{args.output_suffix}"
    trainer.output_dir = new_output
    trainer.checkpoints_dir = new_output / "checkpoints"
    trainer.plots_dir = new_output / "plots"
    trainer.results_dir = new_output / "results"
    
    # Create new directories
    for dir_path in [trainer.output_dir, trainer.checkpoints_dir, trainer.plots_dir, trainer.results_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    print(f"📁 Output directory: {trainer.output_dir}")
    
    # Continue training
    print("🚀 Continuing training...")
    trainer.train(dataloader)
    
    # Generate analysis
    print("📊 Generating analysis...")
    trainer.create_analysis(dataloader, datamodule)
    
    print("🎉 Continued training completed!")
    print(f"📁 Results saved in: {trainer.output_dir}")


if __name__ == "__main__":
    main()
