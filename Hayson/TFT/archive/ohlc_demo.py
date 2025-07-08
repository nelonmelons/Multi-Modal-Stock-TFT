#!/usr/bin/env python3
"""
Standalone OHLC Plotting Demo for TFT Trading System

This script demonstrates the OHLC plotting functionality with mock data
and can be easily adapted to work with real TFT models.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import logging

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ohlc_plotter import OHLCPlotter, OHLCData

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_realistic_market_data(symbol: str, days: int = 60) -> pd.DataFrame:
    """
    Generate realistic market data with proper OHLC relationships.
    
    Args:
        symbol: Stock symbol
        days: Number of days to generate
        
    Returns:
        DataFrame with realistic OHLC data
    """
    # Generate timestamps
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    timestamps = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Base price based on symbol
    symbol_prices = {
        'AAPL': 175.0,
        'GOOGL': 140.0,
        'MSFT': 415.0,
        'NVDA': 875.0,
        'TSLA': 240.0,
        'META': 510.0,
        'AMZN': 170.0
    }
    
    base_price = symbol_prices.get(symbol, 150.0)
    
    # Generate price series with realistic trends
    np.random.seed(hash(symbol) % 1000)  # Consistent seed per symbol
    
    # Add some trend based on symbol
    trends = {
        'AAPL': 0.0005,    # Slight positive trend
        'GOOGL': 0.0002,   # Small positive trend
        'MSFT': 0.0008,    # Good positive trend
        'NVDA': 0.001,     # Strong positive trend
        'TSLA': -0.0003,   # Slight negative trend
        'META': 0.0004,    # Moderate positive trend
        'AMZN': 0.0001     # Minimal trend
    }
    
    trend = trends.get(symbol, 0.0002)
    
    prices = []
    for i in range(days):
        if i == 0:
            prices.append(base_price)
        else:
            # Random walk with trend and volatility
            daily_return = np.random.normal(trend, 0.02)  # 2% daily volatility
            new_price = prices[-1] * (1 + daily_return)
            prices.append(max(new_price, base_price * 0.5))  # Floor at 50% of base
    
    # Generate OHLC data
    data = []
    for i, (timestamp, close_price) in enumerate(zip(timestamps, prices)):
        # Intraday volatility
        daily_vol = np.random.uniform(0.01, 0.03)  # 1-3% intraday volatility
        
        # Generate realistic OHLC
        # Open price (based on previous close with gap)
        if i == 0:
            open_price = close_price
        else:
            gap = np.random.normal(0, 0.005)  # 0.5% gap volatility
            open_price = prices[i-1] * (1 + gap)
        
        # High and low based on open and close
        price_range = [open_price, close_price]
        mid_price = np.mean(price_range)
        
        high_price = mid_price * (1 + abs(np.random.normal(0, daily_vol)))
        low_price = mid_price * (1 - abs(np.random.normal(0, daily_vol)))
        
        # Ensure OHLC relationships are correct
        high_price = max(high_price, open_price, close_price)
        low_price = min(low_price, open_price, close_price)
        
        # Volume (realistic patterns)
        base_volume = np.random.randint(1000000, 5000000)
        if i > 0:  # Calculate daily return for volume calculation
            daily_return = (close_price - prices[i-1]) / prices[i-1]
            volume_multiplier = 1 + abs(daily_return) * 10  # Higher volume on big moves
        else:
            volume_multiplier = 1.0
        volume = int(base_volume * volume_multiplier)
        
        data.append({
            'timestamp': timestamp,
            'open': round(open_price, 2),
            'high': round(high_price, 2),
            'low': round(low_price, 2),
            'close': round(close_price, 2),
            'volume': volume
        })
    
    return pd.DataFrame(data)


def generate_tft_predictions(historical_data: pd.DataFrame, 
                           prediction_days: int = 10) -> Dict[str, np.ndarray]:
    """
    Generate mock TFT predictions that look realistic.
    
    Args:
        historical_data: Historical market data
        prediction_days: Number of days to predict
        
    Returns:
        Dictionary with OHLC predictions
    """
    # Use the last few prices to establish trend
    recent_prices = np.array(historical_data['close'].tail(5).values)
    recent_trend = np.mean(np.diff(recent_prices) / recent_prices[:-1])
    
    # Generate prediction sequence
    last_price = historical_data['close'].iloc[-1]
    predicted_prices = []
    
    for i in range(prediction_days):
        # Trend continuation with decreasing confidence
        trend_factor = recent_trend * (0.9 ** i)  # Trend decay
        noise = np.random.normal(0, 0.01 * (1 + i * 0.1))  # Increasing uncertainty
        
        if i == 0:
            predicted_prices.append(last_price * (1 + trend_factor + noise))
        else:
            predicted_prices.append(predicted_prices[-1] * (1 + trend_factor + noise))
    
    # Generate OHLC from predicted close prices
    predicted_ohlc = {
        'open': [],
        'high': [],
        'low': [],
        'close': []
    }
    
    for i, close_price in enumerate(predicted_prices):
        # Open price (gap from previous close)
        if i == 0:
            open_price = last_price * (1 + np.random.normal(0, 0.003))
        else:
            open_price = predicted_prices[i-1] * (1 + np.random.normal(0, 0.003))
        
        # Intraday range
        daily_vol = 0.015 * (1 + i * 0.05)  # Increasing volatility with time
        high_price = close_price * (1 + abs(np.random.normal(0, daily_vol)))
        low_price = close_price * (1 - abs(np.random.normal(0, daily_vol)))
        
        # Ensure OHLC relationships
        high_price = max(high_price, open_price, close_price)
        low_price = min(low_price, open_price, close_price)
        
        predicted_ohlc['open'].append(open_price)
        predicted_ohlc['high'].append(high_price)
        predicted_ohlc['low'].append(low_price)
        predicted_ohlc['close'].append(close_price)
    
    # Convert to numpy arrays
    result = {}
    for key in predicted_ohlc:
        result[key] = np.array(predicted_ohlc[key]).reshape(1, -1)
    
    return result


def generate_trading_signals(predictions: Dict[str, np.ndarray], 
                           current_price: float) -> List[Dict]:
    """
    Generate trading signals based on predictions.
    
    Args:
        predictions: OHLC predictions
        current_price: Current market price
        
    Returns:
        List of trading signals
    """
    signals = []
    close_predictions = predictions['close'][0]
    
    for i, predicted_close in enumerate(close_predictions):
        # Calculate expected return
        if i == 0:
            expected_return = (predicted_close - current_price) / current_price
        else:
            expected_return = (predicted_close - close_predictions[i-1]) / close_predictions[i-1]
        
        # Generate signal
        if expected_return > 0.015:  # 1.5% threshold
            signal_type = 'BUY'
            confidence = min(0.9, 0.5 + abs(expected_return) * 10)
        elif expected_return < -0.015:
            signal_type = 'SELL'
            confidence = min(0.9, 0.5 + abs(expected_return) * 10)
        else:
            signal_type = 'HOLD'
            confidence = 0.3 + np.random.uniform(0, 0.4)
        
        signals.append({
            'signal': signal_type,
            'confidence': confidence,
            'expected_return': expected_return,
            'timestamp': datetime.now() + timedelta(days=i)
        })
    
    return signals


def create_comprehensive_demo():
    """
    Create a comprehensive demo of OHLC plotting functionality.
    """
    print("=== TFT OHLC PLOTTING COMPREHENSIVE DEMO ===")
    
    # Create output directory
    output_dir = "ohlc_demo_plots"
    os.makedirs(output_dir, exist_ok=True)
    
    # Test symbols
    symbols = ['AAPL', 'GOOGL', 'MSFT', 'NVDA', 'TSLA']
    
    # Mock predictor for the plotter
    class MockPredictor:
        def __init__(self):
            self.model_loader = None
            self.model = None
    
    mock_predictor = MockPredictor()
    plotter = OHLCPlotter(mock_predictor)  # type: ignore
    
    print(f"\n1. Generating realistic market data for {len(symbols)} symbols...")
    
    for i, symbol in enumerate(symbols):
        print(f"   Processing {symbol} ({i+1}/{len(symbols)})...")
        
        # Generate historical data
        historical_data = generate_realistic_market_data(symbol, days=60)
        current_price = historical_data['close'].iloc[-1]
        
        # Generate TFT predictions
        predictions = generate_tft_predictions(historical_data, prediction_days=10)
        
        # Generate trading signals
        signals = generate_trading_signals(predictions, current_price)
        
        # Create OHLCData objects
        actual_data = OHLCData(
            timestamps=historical_data['timestamp'].tolist(),
            open=historical_data['open'].tolist(),
            high=historical_data['high'].tolist(),
            low=historical_data['low'].tolist(),
            close=historical_data['close'].tolist(),
            volume=historical_data['volume'].tolist()
        )
        
        # Create predicted data
        pred_start = historical_data['timestamp'].iloc[-1] + timedelta(days=1)
        pred_timestamps = [pred_start + timedelta(days=j) for j in range(10)]
        
        predicted_data = OHLCData(
            timestamps=pred_timestamps,
            open=predictions['open'][0].tolist(),
            high=predictions['high'][0].tolist(),
            low=predictions['low'][0].tolist(),
            close=predictions['close'][0].tolist()
        )
        
        # Create individual plots
        print(f"     Creating OHLC comparison plot...")
        fig1 = plotter.plot_ohlc_vs_predictions(
            actual_data, predicted_data, symbol,
            save_path=os.path.join(output_dir, f"{symbol}_ohlc_comparison.png")
        )
        plt.close(fig1)
        
        print(f"     Creating trading signals plot...")
        fig2 = plotter.plot_trading_signals(
            actual_data.close, actual_data.timestamps, signals, symbol,
            save_path=os.path.join(output_dir, f"{symbol}_trading_signals.png")
        )
        plt.close(fig2)
        
        print(f"     Creating comprehensive dashboard...")
        fig3 = plotter.create_comprehensive_dashboard(
            actual_data, predicted_data, signals, symbol,
            save_path=os.path.join(output_dir, f"{symbol}_dashboard.png")
        )
        plt.close(fig3)
        
        # Print summary for this symbol
        print(f"     📊 {symbol} Summary:")
        print(f"       Current Price: ${current_price:.2f}")
        print(f"       Predicted Next Close: ${predictions['close'][0][0]:.2f}")
        print(f"       Signal Counts: {sum(1 for s in signals if s['signal'] == 'BUY')} BUY, {sum(1 for s in signals if s['signal'] == 'SELL')} SELL, {sum(1 for s in signals if s['signal'] == 'HOLD')} HOLD")
        print()
    
    # Create a multi-symbol comparison
    print("2. Creating multi-symbol comparison dashboard...")
    
    fig, axes = plt.subplots(3, 2, figsize=(20, 15))
    fig.suptitle('TFT Multi-Symbol Trading Analysis', fontsize=18, fontweight='bold')
    
    for i, symbol in enumerate(symbols):
        if i >= 6:  # Only plot first 6 symbols
            break
            
        row = i // 2
        col = i % 2
        ax = axes[row, col]
        
        # Generate data for this symbol
        historical_data = generate_realistic_market_data(symbol, days=30)
        predictions = generate_tft_predictions(historical_data, prediction_days=5)
        
        # Plot price and predictions
        dates = historical_data['timestamp']
        actual_close = historical_data['close']
        
        # Plot historical data
        ax.plot(dates, actual_close, label='Actual', color='blue', linewidth=2)
        
        # Plot predictions
        pred_start = dates.iloc[-1] + timedelta(days=1)
        pred_dates = [pred_start + timedelta(days=j) for j in range(5)]
        pred_close = predictions['close'][0]
        
        ax.plot(pred_dates, pred_close, label='Predicted', color='red', 
               linewidth=2, linestyle='--', alpha=0.8)
        
        # Format and style
        ax.set_title(f'{symbol} - TFT Predictions', fontsize=14, fontweight='bold')
        ax.set_ylabel('Price ($)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Format dates
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    
    # Hide empty subplot
    if len(symbols) % 2 == 1:
        axes[-1, -1].axis('off')
    
    plt.tight_layout()
    multi_symbol_path = os.path.join(output_dir, "multi_symbol_comparison.png")
    plt.savefig(multi_symbol_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print("3. Performance Summary:")
    print(f"   ✅ Generated plots for {len(symbols)} symbols")
    print(f"   📁 All plots saved to: {output_dir}/")
    print(f"   📊 Plot types created:")
    print(f"     - OHLC comparison plots")
    print(f"     - Trading signals plots")
    print(f"     - Comprehensive dashboards")
    print(f"     - Multi-symbol comparison")
    
    # List all generated files
    plot_files = [f for f in os.listdir(output_dir) if f.endswith('.png')]
    print(f"\n4. Generated Files ({len(plot_files)} total):")
    for file in sorted(plot_files):
        print(f"     📄 {file}")
    
    print(f"\n🎉 OHLC Plotting Demo completed successfully!")
    print(f"🔍 View the plots in the '{output_dir}' directory")
    print(f"💡 This demonstrates the full OHLC plotting functionality that can be integrated with real TFT models")


if __name__ == "__main__":
    create_comprehensive_demo()
