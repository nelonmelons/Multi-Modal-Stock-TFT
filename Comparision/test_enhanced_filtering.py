#!/usr/bin/env python3
"""
Test script to see detailed feature filtering output
"""
import sys
import os
sys.path.append('/Users/haysoncheung/programs/pythonProject/TFT-b-nelson/Comparision')

# Set environment to prevent full pipeline run
os.environ['TEST_MODE'] = '1'

from run_all_models import run_pipeline

# Temporarily modify the models_to_run to include only one model for testing
def test_feature_filtering():
    print("🧪 TESTING ENHANCED FEATURE FILTERING")
    print("="*80)
    
    # This will run the data loading and show the enhanced filtering output
    # for the first model that uses feature filtering
    try:
        run_pipeline()
    except Exception as e:
        print(f"Pipeline stopped at: {e}")
        print("This is expected for testing - we just wanted to see the filtering output")

if __name__ == '__main__':
    test_feature_filtering()
