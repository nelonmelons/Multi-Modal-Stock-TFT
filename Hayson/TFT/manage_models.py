#!/usr/bin/env python3
"""
Model Management Utility for TFT Data Pipeline

This script helps you manage embedding models used in the news processing pipeline.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data.fetch_news import (
    preload_embedding_model, 
    check_model_cache_size, 
    clear_model_cache
)


def main():
    """Main function for model management."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Manage embedding models for TFT pipeline")
    parser.add_argument(
        "action", 
        choices=["preload", "check", "clear", "info"],
        help="Action to perform"
    )
    parser.add_argument(
        "--model", 
        default="bert-base-uncased",
        help="Model name to preload (default: bert-base-uncased)"
    )
    
    args = parser.parse_args()
    
    print("🤖 TFT Pipeline Model Manager")
    print("=" * 40)
    
    if args.action == "preload":
        print(f"Pre-loading model: {args.model}")
        success = preload_embedding_model(args.model)
        if success:
            print("\n✅ Model successfully cached!")
            print("   Future news processing will be much faster.")
        else:
            print("\n❌ Failed to cache model.")
            print("   Check your internet connection and try again.")
    
    elif args.action == "check":
        print("Checking model cache...")
        size_mb = check_model_cache_size()
        if size_mb > 0:
            print(f"\n💾 You have {size_mb:.1f} MB of cached models.")
        else:
            print("\n📭 No models cached yet.")
            print("   Run 'python manage_models.py preload' to cache BERT model.")
    
    elif args.action == "clear":
        print("Clearing in-memory model cache...")
        clear_model_cache()
        print("\n🧹 Memory cache cleared.")
        print("   Note: Disk cache is preserved for faster loading.")
    
    elif args.action == "info":
        print("Model Information:")
        print("\n📦 Default Model: bert-base-uncased")
        print("   - Size: ~400 MB")
        print("   - Purpose: News text embeddings")
        print("   - Cache Location: ~/.cache/huggingface/transformers")
        
        print("\n🔄 Caching Behavior:")
        print("   - First run: Downloads and caches model")
        print("   - Subsequent runs: Loads from cache (much faster)")
        print("   - In-memory cache: Avoids reloading during same session")
        
        print("\n💡 Tips:")
        print("   - Pre-load models with: python manage_models.py preload")
        print("   - Check cache size with: python manage_models.py check")
        print("   - Models are shared across all Hugging Face projects")
        
        # Show current cache status
        print("\n" + "="*40)
        check_model_cache_size()


def quick_setup():
    """Quick setup function for first-time users."""
    print("🚀 Quick Setup: Pre-loading embedding model")
    print("This will download ~400MB and cache it for faster future use.")
    
    response = input("Continue? (y/N): ")
    if response.lower() in ['y', 'yes']:
        success = preload_embedding_model()
        if success:
            print("\n✅ Setup complete! Your pipeline is ready for news processing.")
        else:
            print("\n❌ Setup failed. Check your internet connection.")
    else:
        print("Setup cancelled. You can run this later with:")
        print("python manage_models.py preload")


if __name__ == "__main__":
    if len(sys.argv) == 1:
        # No arguments provided, run interactive setup
        quick_setup()
    else:
        main()
