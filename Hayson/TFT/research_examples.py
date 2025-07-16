#!/usr/bin/env python3
"""
Example Usage: Research Paper Comparison Framework
=================================================

This script demonstrates how to use the comprehensive comparison framework
for generating research paper results comparing multi-modal TFT against baselines.

Usage Examples:
    
    # Run TFT training only
    python unified_tft_pipeline.py --epochs 20 --symbols AAPL,MSFT,GOOGL
    
    # Run TFT + all baselines comparison
    python unified_tft_pipeline.py --epochs 20 --symbols AAPL,MSFT,GOOGL --run-baselines
    
    # Run specific baseline types
    python unified_tft_pipeline.py --epochs 20 --symbols AAPL,MSFT,GOOGL --run-baselines --baseline-types traditional_ml,deep_learning
    
    # Run with feature ablation study
    python unified_tft_pipeline.py --epochs 20 --symbols AAPL,MSFT,GOOGL --run-baselines --run-ablation
    
    # Full research paper evaluation with out-of-sample testing
    python unified_tft_pipeline.py --epochs 50 --symbols AAPL,MSFT,GOOGL,NVDA,AMD --test-symbol TSLA --out-of-sample --run-baselines --run-ablation --comparison-output-dir research_results
    
    # Quick baseline comparison (fewer epochs for testing)
    python unified_tft_pipeline.py --epochs 5 --symbols AAPL --run-baselines --baseline-types traditional_ml

Research Paper Workflow:
    1. Quick testing with limited data and epochs
    2. Full baseline comparison with comprehensive data
    3. Feature ablation study
    4. Market regime analysis
    5. Statistical significance testing
    6. Report generation
"""

import subprocess
import sys
from pathlib import Path

def run_research_experiments():
    """Run a complete set of research experiments."""
    
    print("🔬 Multi-Modal TFT Research Paper Experiment Suite")
    print("=" * 60)
    
    experiments = [
        {
            "name": "Quick Baseline Test",
            "description": "Fast test with minimal data to verify framework",
            "command": [
                "python", "unified_tft_pipeline.py",
                "--epochs", "3",
                "--symbols", "AAPL",
                "--run-baselines",
                "--baseline-types", "traditional_ml",
                "--comparison-output-dir", "quick_test_results"
            ]
        },
        {
            "name": "Traditional ML Baselines",
            "description": "Compare against traditional machine learning methods",
            "command": [
                "python", "unified_tft_pipeline.py", 
                "--epochs", "20",
                "--symbols", "AAPL,MSFT,GOOGL,NVDA",
                "--run-baselines",
                "--baseline-types", "traditional_ml",
                "--comparison-output-dir", "traditional_ml_results"
            ]
        },
        {
            "name": "Deep Learning Baselines", 
            "description": "Compare against deep learning methods",
            "command": [
                "python", "unified_tft_pipeline.py",
                "--epochs", "30", 
                "--symbols", "AAPL,MSFT,GOOGL,NVDA,AMD",
                "--run-baselines",
                "--baseline-types", "deep_learning",
                "--comparison-output-dir", "deep_learning_results"
            ]
        },
        {
            "name": "Complete Research Study",
            "description": "Full comparison with all baselines and ablation study",
            "command": [
                "python", "unified_tft_pipeline.py",
                "--epochs", "50",
                "--symbols", "AAPL,MSFT,GOOGL,NVDA,AMD,TSLA,META,AMZN",
                "--test-symbol", "ORCL",
                "--out-of-sample",
                "--run-baselines",
                "--run-ablation",
                "--comparison-output-dir", "full_research_results"
            ]
        }
    ]
    
    for i, experiment in enumerate(experiments, 1):
        print(f"\n{i}. {experiment['name']}")
        print(f"   Description: {experiment['description']}")
        print(f"   Command: {' '.join(experiment['command'])}")
        
        response = input(f"\n   Run this experiment? (y/n/s=skip all): ").strip().lower()
        
        if response == 's':
            print("   Skipping remaining experiments...")
            break
        elif response == 'y':
            print(f"   🚀 Running {experiment['name']}...")
            try:
                result = subprocess.run(experiment['command'], check=True, capture_output=True, text=True)
                print(f"   ✅ {experiment['name']} completed successfully!")
                if result.stdout:
                    print(f"   Output: {result.stdout[-500:]}")  # Last 500 chars
            except subprocess.CalledProcessError as e:
                print(f"   ❌ {experiment['name']} failed: {e}")
                if e.stderr:
                    print(f"   Error: {e.stderr[-500:]}")  # Last 500 chars
            except KeyboardInterrupt:
                print(f"   ⚠️  {experiment['name']} interrupted by user")
                break
        else:
            print(f"   ⏭️  Skipping {experiment['name']}")
    
    print("\n🎉 Research experiment suite completed!")
    print("\n📁 Results are saved in respective output directories:")
    for experiment in experiments:
        output_dir = None
        for i, arg in enumerate(experiment['command']):
            if arg == '--comparison-output-dir' and i + 1 < len(experiment['command']):
                output_dir = experiment['command'][i + 1]
                break
        if output_dir and Path(output_dir).exists():
            print(f"   📊 {experiment['name']}: {output_dir}/")

def print_usage_examples():
    """Print usage examples for the comparison framework."""
    
    examples = [
        {
            "title": "Basic TFT Training (No Baselines)",
            "command": "python unified_tft_pipeline.py --epochs 20 --symbols AAPL,MSFT",
            "description": "Train TFT model only, no comparison"
        },
        {
            "title": "TFT + Traditional ML Baselines",
            "command": "python unified_tft_pipeline.py --epochs 20 --symbols AAPL,MSFT --run-baselines --baseline-types traditional_ml",
            "description": "Compare TFT against Linear Regression, Random Forest, XGBoost"
        },
        {
            "title": "TFT + Deep Learning Baselines", 
            "command": "python unified_tft_pipeline.py --epochs 30 --symbols AAPL,MSFT,GOOGL --run-baselines --baseline-types deep_learning",
            "description": "Compare TFT against LSTM, GRU, Vanilla Transformer"
        },
        {
            "title": "Complete Research Paper Study",
            "command": "python unified_tft_pipeline.py --epochs 50 --symbols AAPL,MSFT,GOOGL,NVDA --test-symbol TSLA --out-of-sample --run-baselines --run-ablation",
            "description": "Full comparison: all baselines + ablation study + out-of-sample validation"
        },
        {
            "title": "Feature Ablation Study Only",
            "command": "python unified_tft_pipeline.py --epochs 20 --symbols AAPL --run-ablation",
            "description": "Test different feature combinations (OHLCV, technical, news, economic)"
        },
        {
            "title": "Finance-Specific Baselines",
            "command": "python unified_tft_pipeline.py --epochs 15 --symbols AAPL,MSFT --run-baselines --baseline-types finance_specific",
            "description": "Compare against Buy & Hold, Moving Average, ARIMA"
        }
    ]
    
    print("📚 Usage Examples for Multi-Modal TFT Research Framework")
    print("=" * 70)
    
    for i, example in enumerate(examples, 1):
        print(f"\n{i}. {example['title']}")
        print(f"   Command: {example['command']}")
        print(f"   Description: {example['description']}")
    
    print(f"\n💡 Tips:")
    print(f"   • Start with fewer epochs (5-10) for quick testing")
    print(f"   • Use --symbols AAPL for fastest testing with single stock")
    print(f"   • Add --clear-cache to ensure fresh data")
    print(f"   • Results saved in comparison_results/ by default")
    print(f"   • Use --comparison-output-dir to specify custom output location")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--run-experiments":
        run_research_experiments()
    else:
        print_usage_examples()
        print(f"\n🚀 To run the complete experiment suite:")
        print(f"   python {sys.argv[0]} --run-experiments")
