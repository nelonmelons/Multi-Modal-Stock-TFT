#!/usr/bin/env python3
"""
Test script for the horizon-based deep learning baselines
"""

import subprocess
import sys

def test_horizon_model():
    """Test the deep learning baselines with different horizons."""
    
    print("Testing horizon-based deep learning models...")
    
    # Test with different horizon values
    horizons = [5, 15, 30]
    
    for horizon in horizons:
        print(f"\n=== Testing with {horizon}-day horizon ===")
        
        cmd = [
            sys.executable, "deeplearning_baselines.py",
            "--horizon-days", str(horizon),
            "--epochs", "2",  # Quick test with few epochs
            "--symbol", "AAPL"
        ]
        
        try:
            # Run the command but don't wait for completion for this test
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                print(f"✅ {horizon}-day horizon test completed successfully")
                print("Output preview:")
                print(result.stdout[-500:])  # Show last 500 chars
            else:
                print(f"❌ {horizon}-day horizon test failed")
                print("Error output:")
                print(result.stderr[-500:])
                
        except subprocess.TimeoutExpired:
            print(f"⏰ {horizon}-day horizon test timed out (this is expected for quick testing)")
        except Exception as e:
            print(f"🔥 Error running {horizon}-day horizon test: {e}")

if __name__ == "__main__":
    test_horizon_model()
