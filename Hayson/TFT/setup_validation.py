#!/usr/bin/env python3
"""
Setup and Validation Script for Research Framework
=================================================

This script validates that all components are properly set up for running
the comprehensive research comparison framework.
"""

import sys
import os
import importlib
from pathlib import Path

def check_python_version():
    """Check Python version compatibility."""
    version = sys.version_info
    print(f"🐍 Python Version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8+ required")
        return False
    else:
        print("✅ Python version compatible")
        return True

def check_required_packages():
    """Check if required packages are installed."""
    required_packages = [
        'numpy', 'pandas', 'matplotlib', 'seaborn', 'sklearn',
        'scipy', 'torch', 'pytorch_lightning', 'pytorch_forecasting'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            importlib.import_module(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} (missing)")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n📦 Install missing packages:")
        print(f"   pip install {' '.join(missing_packages)}")
        return False
    
    return True

def check_framework_files():
    """Check if framework files exist."""
    required_files = [
        'unified_tft_pipeline.py',
        'baseline_models.py', 
        'comparison_framework.py',
        'research_examples.py'
    ]
    
    current_dir = Path.cwd()
    missing_files = []
    
    for file in required_files:
        file_path = current_dir / file
        if file_path.exists():
            print(f"✅ {file}")
        else:
            print(f"❌ {file} (missing)")
            missing_files.append(file)
    
    if missing_files:
        print(f"\n📁 Missing framework files:")
        for file in missing_files:
            print(f"   {file}")
        return False
    
    return True

def test_framework_imports():
    """Test importing framework components."""
    try:
        print("\n🔍 Testing framework imports...")
        
        # Test baseline models
        from baseline_models import BaselineModel, get_all_baselines
        print("✅ baseline_models import successful")
        
        # Test comparison framework
        from comparison_framework import ComprehensiveEvaluator, MetricsCalculator
        print("✅ comparison_framework import successful")
        
        # Test baseline creation
        config = {'max_encoder_length': 60, 'max_prediction_length': 10}
        baselines = get_all_baselines(config)
        print(f"✅ Created {len(baselines)} baseline models")
        
        return True
        
    except Exception as e:
        print(f"❌ Framework import failed: {e}")
        return False

def create_test_directories():
    """Create necessary directories for testing."""
    directories = [
        'comparison_results',
        'comparison_results/plots',
        'comparison_results/results',
        'test_output'
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✅ Created directory: {directory}")

def run_quick_validation():
    """Run a quick validation test."""
    print("\n🚀 Running Quick Validation Test...")
    
    try:
        # Test basic functionality
        from baseline_models import LinearRegressionBaseline
        from comparison_framework import MetricsCalculator
        import numpy as np
        
        # Create sample data
        predictions = np.random.randn(100)
        targets = predictions + np.random.randn(100) * 0.1  # Add some noise
        
        # Test metrics calculation
        calc = MetricsCalculator()
        metrics = calc.prediction_metrics(predictions, targets)
        
        print(f"✅ Sample prediction metrics calculated:")
        print(f"   RMSE: {metrics.get('rmse', 'N/A'):.4f}")
        print(f"   R²: {metrics.get('r2', 'N/A'):.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Quick validation failed: {e}")
        return False

def print_usage_guide():
    """Print usage guide for the research framework."""
    print("\n📚 Research Framework Usage Guide")
    print("=" * 50)
    
    print("\n🎯 Quick Start:")
    print("   1. Test basic setup:")
    print("      python research_examples.py")
    print()
    print("   2. Run quick baseline test:")
    print("      python unified_tft_pipeline.py --epochs 3 --symbols AAPL --run-baselines --baseline-types traditional_ml")
    print()
    print("   3. Full research study:")
    print("      python unified_tft_pipeline.py --epochs 20 --symbols AAPL,MSFT --run-baselines --run-ablation")
    
    print("\n📊 Research Paper Workflow:")
    print("   1. Quick validation (few epochs, single symbol)")
    print("   2. Traditional ML baselines comparison")
    print("   3. Deep learning baselines comparison") 
    print("   4. Finance-specific baselines comparison")
    print("   5. Feature ablation study")
    print("   6. Market regime analysis")
    print("   7. Statistical significance testing")
    print("   8. Report generation")
    
    print("\n📁 Output Structure:")
    print("   comparison_results/")
    print("   ├── plots/")
    print("   │   ├── baseline_comparison.png")
    print("   │   └── feature_importance_analysis.png")
    print("   └── results/")
    print("       ├── comprehensive_comparison_report.json")
    print("       └── research_paper_results.md")

def main():
    """Main setup validation function."""
    print("🔧 Multi-Modal TFT Research Framework Setup")
    print("=" * 60)
    
    checks_passed = 0
    total_checks = 5
    
    # Run all checks
    if check_python_version():
        checks_passed += 1
    
    print(f"\n📦 Checking Required Packages:")
    if check_required_packages():
        checks_passed += 1
    
    print(f"\n📁 Checking Framework Files:")
    if check_framework_files():
        checks_passed += 1
    
    if test_framework_imports():
        checks_passed += 1
    
    print(f"\n📂 Creating Test Directories:")
    create_test_directories()
    
    if run_quick_validation():
        checks_passed += 1
    
    # Summary
    print(f"\n📊 Setup Summary:")
    print(f"   Checks passed: {checks_passed}/{total_checks}")
    
    if checks_passed == total_checks:
        print("🎉 Setup completed successfully!")
        print("\n✅ Your research framework is ready to use!")
        print_usage_guide()
    else:
        print("❌ Setup incomplete. Please address the issues above.")
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
