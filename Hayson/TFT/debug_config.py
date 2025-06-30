#!/usr/bin/env python3
"""Quick test to check current configuration values"""

print("🔍 Checking current configuration...")

try:
    from baseline_tft import DEFAULT_SYMBOLS as BASELINE_DEFAULT
    print(f"baseline_tft.py DEFAULT_SYMBOLS: {BASELINE_DEFAULT}")
except Exception as e:
    print(f"Error importing from baseline_tft: {e}")

try:
    from train_baseline_tft import DEFAULT_SYMBOLS as TRAIN_DEFAULT
    print(f"train_baseline_tft.py DEFAULT_SYMBOLS: {TRAIN_DEFAULT}")
except Exception as e:
    print(f"Error importing from train_baseline_tft: {e}")

print("\n🔍 Checking file contents directly...")
with open('baseline_tft.py', 'r') as f:
    lines = f.readlines()
    for i, line in enumerate(lines):
        if 'DEFAULT_SYMBOLS =' in line and not line.strip().startswith('#'):
            print(f"baseline_tft.py line {i+1}: {line.strip()}")

with open('train_baseline_tft.py', 'r') as f:
    lines = f.readlines()
    for i, line in enumerate(lines):
        if 'DEFAULT_SYMBOLS =' in line and not line.strip().startswith('#'):
            print(f"train_baseline_tft.py line {i+1}: {line.strip()}")
