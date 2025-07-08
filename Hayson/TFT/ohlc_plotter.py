#!/usr/bin/env python3
"""
OHLC Plotting System for TFT Model Predictions

This module provides comprehensive OHLC plotting functionality that visualizes
TFT model predictions against actual market data.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any, Union
import seaborn as sns
from dataclasses import dataclass

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import our real TFT model manager
from real_tft_integration import RealTFTModelManager

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

@dataclass
class OHLCData:
    """Data class for OHLC information."""
    timestamps: List[Union[datetime, Any]]
    open: List[float]
    high: List[float]
    low: List[float]
    close: List[float]
    volume: Optional[List[float]] = None

class OHLCPlotter:
    """
    Advanced OHLC plotting system with prediction visualization.
    """
    
    def __init__(self, tft_model_manager: Optional[RealTFTModelManager] = None, figsize: Tuple[int, int] = (15, 10)):
        """
        Initialize the OHLC plotter.
        
        Args:
            tft_model_manager: Real TFT model manager instance (optional)
            figsize: Figure size for plots
        """
        self.tft_model_manager = tft_model_manager
        self.figsize = figsize
        self.colors = {
            'bullish': '#2E8B57',      # Sea Green
            'bearish': '#DC143C',      # Crimson
            'prediction': '#4169E1',   # Royal Blue
            'actual': '#2F4F4F',       # Dark Slate Gray
            'volume': '#708090',       # Slate Gray
            'background': '#F5F5F5'    # White Smoke
        }
        
    def plot_candlestick(self, 
                        ohlc_data: OHLCData, 
                        title: str = "OHLC Chart",
                        ax: Optional[Axes] = None) -> Axes:
        """
        Plot candlestick chart.
        
        Args:
            ohlc_data: OHLC data to plot
            title: Chart title
            ax: Matplotlib axes (optional)
            
        Returns:
            Matplotlib axes object
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=self.figsize)
        
        # Convert timestamps to matplotlib format
        dates = mdates.date2num(ohlc_data.timestamps)
        
        # Plot candlesticks
        for i, (date, o, h, l, c) in enumerate(zip(dates, ohlc_data.open, 
                                                  ohlc_data.high, ohlc_data.low, 
                                                  ohlc_data.close)):
            color = self.colors['bullish'] if c >= o else self.colors['bearish']
            
            # Draw the high-low line
            ax.plot([date, date], [l, h], color=color, linewidth=1, alpha=0.8)
            
            # Draw the body
            height = abs(c - o)
            bottom = min(o, c)
            
            rect = Rectangle((date - 0.3, bottom), 0.6, height, 
                           facecolor=color, edgecolor=color, alpha=0.7)
            ax.add_patch(rect)
        
        # Format x-axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=max(1, len(dates)//10)))
        
        # Rotate x-axis labels
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.set_ylabel('Price ($)', fontsize=12)
        ax.grid(True, alpha=0.3)
        
        return ax
    
    def plot_ohlc_vs_predictions(self, 
                                actual_data: OHLCData,
                                predicted_data: OHLCData,
                                symbol: str = "Stock",
                                save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot actual OHLC data vs TFT predictions.
        
        Args:
            actual_data: Actual OHLC data
            predicted_data: Predicted OHLC data
            symbol: Stock symbol for title
            save_path: Optional path to save the plot
            
        Returns:
            Matplotlib figure object
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 12))
        fig.suptitle(f'{symbol} - TFT Model Predictions vs Actual OHLC', 
                    fontsize=18, fontweight='bold')
        
        # Plot actual data
        self.plot_candlestick(actual_data, f'{symbol} - Actual OHLC', ax1)
        
        # Plot predicted data
        self.plot_candlestick(predicted_data, f'{symbol} - Predicted OHLC', ax2)
        
        # Price comparison
        ax3.plot(actual_data.timestamps, actual_data.close, 
                label='Actual Close', color=self.colors['actual'], linewidth=2)
        ax3.plot(predicted_data.timestamps, predicted_data.close, 
                label='Predicted Close', color=self.colors['prediction'], 
                linewidth=2, linestyle='--')
        ax3.set_title('Close Price Comparison', fontsize=14, fontweight='bold')
        ax3.set_ylabel('Price ($)', fontsize=12)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Error analysis
        if len(actual_data.close) == len(predicted_data.close):
            errors = np.array(predicted_data.close) - np.array(actual_data.close)
            ax4.plot(actual_data.timestamps, errors, 
                    color=self.colors['bearish'], linewidth=2)
            ax4.axhline(y=0, color='black', linestyle='-', alpha=0.5)
            ax4.fill_between(actual_data.timestamps, errors, 0, 
                           alpha=0.3, color=self.colors['bearish'])
            ax4.set_title('Prediction Error (Predicted - Actual)', 
                         fontsize=14, fontweight='bold')
            ax4.set_ylabel('Error ($)', fontsize=12)
            ax4.grid(True, alpha=0.3)
        
        # Format all x-axes
        for ax in [ax1, ax2, ax3, ax4]:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to: {save_path}")
        
        return fig
    
    def plot_prediction_confidence(self, 
                                  predictions: Dict[str, np.ndarray],
                                  timestamps: List[datetime],
                                  confidence_intervals: Optional[Dict[str, np.ndarray]] = None,
                                  symbol: str = "Stock",
                                  save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot predictions with confidence intervals.
        
        Args:
            predictions: Dictionary containing OHLC predictions
            timestamps: List of timestamps
            confidence_intervals: Optional confidence interval data
            symbol: Stock symbol
            save_path: Optional save path
            
        Returns:
            Matplotlib figure object
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'{symbol} - TFT Prediction Confidence Analysis', 
                    fontsize=16, fontweight='bold')
        
        ohlc_keys = ['open', 'high', 'low', 'close']
        titles = ['Open Price', 'High Price', 'Low Price', 'Close Price']
        
        for i, (key, title) in enumerate(zip(ohlc_keys, titles)):
            ax = axes[i//2, i%2]
            
            # Main prediction line
            pred_values = predictions[key].flatten() if predictions[key].ndim > 1 else predictions[key]
            ax.plot(timestamps, pred_values, color=self.colors['prediction'], 
                   linewidth=2, label='Prediction')
            
            # Confidence intervals if available
            if confidence_intervals and key in confidence_intervals:
                lower = confidence_intervals[key]['lower']
                upper = confidence_intervals[key]['upper']
                ax.fill_between(timestamps, lower, upper, 
                               alpha=0.2, color=self.colors['prediction'],
                               label='Confidence Interval')
            
            ax.set_title(title, fontsize=12, fontweight='bold')
            ax.set_ylabel('Price ($)', fontsize=10)
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Format x-axis
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Confidence plot saved to: {save_path}")
        
        return fig
    
    def plot_trading_signals(self, 
                           price_data: List[float],
                           timestamps: List[datetime],
                           signals: List[Dict[str, Any]],
                           symbol: str = "Stock",
                           save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot trading signals on price chart.
        
        Args:
            price_data: Price data
            timestamps: Timestamps
            signals: List of trading signals
            symbol: Stock symbol
            save_path: Optional save path
            
        Returns:
            Matplotlib figure object
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
        fig.suptitle(f'{symbol} - Trading Signals from TFT Model', 
                    fontsize=16, fontweight='bold')
        
        # Price chart with signals
        ax1.plot(timestamps, price_data, color=self.colors['actual'], 
                linewidth=2, label='Price')
        
        # Add trading signals
        buy_signals = []
        sell_signals = []
        hold_signals = []
        
        for i, signal in enumerate(signals):
            if i < len(timestamps):
                if signal['signal'] == 'BUY':
                    buy_signals.append((timestamps[i], price_data[i]))
                elif signal['signal'] == 'SELL':
                    sell_signals.append((timestamps[i], price_data[i]))
                else:
                    hold_signals.append((timestamps[i], price_data[i]))
        
        # Plot signal markers
        if buy_signals:
            buy_times, buy_prices = zip(*buy_signals)
            ax1.scatter(buy_times, buy_prices, color='green', marker='^', 
                       s=100, label='Buy Signal', alpha=0.8)
        
        if sell_signals:
            sell_times, sell_prices = zip(*sell_signals)
            ax1.scatter(sell_times, sell_prices, color='red', marker='v', 
                       s=100, label='Sell Signal', alpha=0.8)
        
        ax1.set_title('Price with Trading Signals', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Price ($)', fontsize=12)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Signal confidence over time
        confidences = [s['confidence'] for s in signals]
        ax2.plot(timestamps[:len(confidences)], confidences, 
                color=self.colors['prediction'], linewidth=2)
        ax2.axhline(y=0.5, color='black', linestyle='--', alpha=0.5, label='Neutral')
        ax2.fill_between(timestamps[:len(confidences)], confidences, 0.5, 
                        alpha=0.3, color=self.colors['prediction'])
        
        ax2.set_title('Signal Confidence Over Time', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Confidence', fontsize=12)
        ax2.set_ylim(0, 1)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Format x-axes
        for ax in [ax1, ax2]:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Trading signals plot saved to: {save_path}")
        
        return fig
    
    def create_comprehensive_dashboard(self, 
                                     actual_data: OHLCData,
                                     predicted_data: OHLCData,
                                     signals: List[Dict[str, Any]],
                                     symbol: str = "Stock",
                                     save_path: Optional[str] = None) -> plt.Figure:
        """
        Create a comprehensive trading dashboard.
        
        Args:
            actual_data: Actual OHLC data
            predicted_data: Predicted OHLC data
            signals: Trading signals
            symbol: Stock symbol
            save_path: Optional save path
            
        Returns:
            Matplotlib figure object
        """
        fig = plt.figure(figsize=(20, 16))
        gs = fig.add_gridspec(3, 3, height_ratios=[2, 1, 1], width_ratios=[2, 1, 1])
        
        # Main candlestick chart
        ax_main = fig.add_subplot(gs[0, :])
        self.plot_candlestick(actual_data, f'{symbol} - Actual vs Predicted', ax_main)
        
        # Add predicted close price line
        ax_main.plot(predicted_data.timestamps, predicted_data.close, 
                    color=self.colors['prediction'], linewidth=2, 
                    linestyle='--', alpha=0.8, label='Predicted Close')
        ax_main.legend()
        
        # Price comparison
        ax_comp = fig.add_subplot(gs[1, 0])
        ax_comp.plot(actual_data.timestamps, actual_data.close, 
                    label='Actual', color=self.colors['actual'], linewidth=2)
        ax_comp.plot(predicted_data.timestamps, predicted_data.close, 
                    label='Predicted', color=self.colors['prediction'], 
                    linewidth=2, linestyle='--')
        ax_comp.set_title('Close Price Comparison')
        ax_comp.legend()
        ax_comp.grid(True, alpha=0.3)
        
        # Error distribution
        ax_err = fig.add_subplot(gs[1, 1])
        if len(actual_data.close) == len(predicted_data.close):
            errors = np.array(predicted_data.close) - np.array(actual_data.close)
            ax_err.hist(errors, bins=20, alpha=0.7, color=self.colors['bearish'])
            ax_err.axvline(x=0, color='black', linestyle='--', alpha=0.5)
            ax_err.set_title('Prediction Error Distribution')
            ax_err.set_xlabel('Error ($)')
        
        # Signal distribution
        ax_sig = fig.add_subplot(gs[1, 2])
        signal_counts = {}
        for signal in signals:
            signal_type = signal['signal']
            signal_counts[signal_type] = signal_counts.get(signal_type, 0) + 1
        
        if signal_counts:
            ax_sig.pie(signal_counts.values(), labels=signal_counts.keys(), 
                      autopct='%1.1f%%', startangle=90)
            ax_sig.set_title('Signal Distribution')
        
        # Trading performance metrics
        ax_perf = fig.add_subplot(gs[2, :])
        
        # Calculate simple returns
        returns = []
        if len(actual_data.close) > 1:
            actual_returns = np.diff(actual_data.close) / actual_data.close[:-1]
            predicted_returns = np.diff(predicted_data.close) / predicted_data.close[:-1]
            
            ax_perf.plot(actual_data.timestamps[1:], actual_returns, 
                        label='Actual Returns', color=self.colors['actual'])
            ax_perf.plot(predicted_data.timestamps[1:], predicted_returns, 
                        label='Predicted Returns', color=self.colors['prediction'])
            ax_perf.axhline(y=0, color='black', linestyle='-', alpha=0.5)
            ax_perf.set_title('Returns Comparison')
            ax_perf.set_ylabel('Returns')
            ax_perf.legend()
            ax_perf.grid(True, alpha=0.3)
        
        # Format all x-axes
        for ax in [ax_main, ax_comp, ax_perf]:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        
        plt.suptitle(f'{symbol} - TFT Trading Dashboard', fontsize=20, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Dashboard saved to: {save_path}")
        
        return fig

def generate_sample_data(symbol: str = "AAPL", days: int = 30) -> Tuple[OHLCData, OHLCData]:
    """
    Generate sample OHLC data for demonstration.
    
    Args:
        symbol: Stock symbol
        days: Number of days to generate
        
    Returns:
        Tuple of (actual_data, predicted_data)
    """
    # Generate timestamps - exactly 'days' number of timestamps
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days-1)  # Fix: subtract (days-1) to get exactly 'days' timestamps
    timestamps = pd.date_range(start=start_date, end=end_date, freq='D').tolist()
    
    # Ensure we have exactly 'days' timestamps
    if len(timestamps) != days:
        timestamps = timestamps[:days]
    
    # Generate realistic price data
    base_price = 150.0
    prices = []
    
    for i in range(days):
        # Random walk with slight upward trend
        change = np.random.normal(0.001, 0.02)  # 0.1% daily trend, 2% volatility
        if i == 0:
            prices.append(base_price)
        else:
            prices.append(prices[-1] * (1 + change))
    
    # Generate OHLC from prices
    actual_ohlc = []
    predicted_ohlc = []
    
    for i, price in enumerate(prices):
        # Actual data
        daily_vol = 0.01  # 1% daily volatility
        high = price * (1 + abs(np.random.normal(0, daily_vol)))
        low = price * (1 - abs(np.random.normal(0, daily_vol)))
        open_price = price * (1 + np.random.normal(0, daily_vol/2))
        close_price = price * (1 + np.random.normal(0, daily_vol/2))
        
        actual_ohlc.append([open_price, high, low, close_price])
        
        # Predicted data (with some error)
        error = np.random.normal(0, 0.005)  # 0.5% prediction error
        pred_factor = 1 + error
        predicted_ohlc.append([
            open_price * pred_factor,
            high * pred_factor,
            low * pred_factor,
            close_price * pred_factor
        ])
    
    # Create OHLCData objects
    actual_data = OHLCData(
        timestamps=timestamps,
        open=[ohlc[0] for ohlc in actual_ohlc],
        high=[ohlc[1] for ohlc in actual_ohlc],
        low=[ohlc[2] for ohlc in actual_ohlc],
        close=[ohlc[3] for ohlc in actual_ohlc]
    )
    
    predicted_data = OHLCData(
        timestamps=timestamps,
        open=[ohlc[0] for ohlc in predicted_ohlc],
        high=[ohlc[1] for ohlc in predicted_ohlc],
        low=[ohlc[2] for ohlc in predicted_ohlc],
        close=[ohlc[3] for ohlc in predicted_ohlc]
    )
    
    return actual_data, predicted_data

def demo_ohlc_plotting():
    """
    Demonstrate the OHLC plotting functionality.
    """
    print("=== TFT OHLC Plotting Demo ===")
    
    try:
        # Load the real TFT model
        print("Loading real TFT model...")
        tft_model_manager = RealTFTModelManager()
        print("✅ Real TFT model loaded successfully")
        
        # Generate sample data
        print("Generating sample OHLC data...")
        actual_data, predicted_data = generate_sample_data("AAPL", days=30)
        
        # Generate sample signals
        signals = []
        for i in range(len(actual_data.timestamps)):
            signal_type = np.random.choice(['BUY', 'SELL', 'HOLD'], p=[0.3, 0.3, 0.4])
            signals.append({
                'signal': signal_type,
                'confidence': np.random.uniform(0.3, 0.9),
                'timestamp': actual_data.timestamps[i]
            })
        
        # Create plotter
        plotter = OHLCPlotter(tft_model_manager)
        
        # Create output directory
        output_dir = "plots"
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate plots
        print("Creating OHLC comparison plot...")
        fig1 = plotter.plot_ohlc_vs_predictions(
            actual_data, predicted_data, "AAPL",
            save_path=os.path.join(output_dir, "ohlc_comparison.png")
        )
        
        print("Creating trading signals plot...")
        fig2 = plotter.plot_trading_signals(
            actual_data.close, actual_data.timestamps, signals, "AAPL",
            save_path=os.path.join(output_dir, "trading_signals.png")
        )
        
        print("Creating comprehensive dashboard...")
        fig3 = plotter.create_comprehensive_dashboard(
            actual_data, predicted_data, signals, "AAPL",
            save_path=os.path.join(output_dir, "trading_dashboard.png")
        )
        
        # Show plots
        plt.show()
        
        print(f"\nAll plots saved to '{output_dir}' directory")
        print("Demo completed successfully!")
        
    except Exception as e:

        print("Creating demo with mock data...")
        
        # Fallback demo with mock data
        actual_data, predicted_data = generate_sample_data("AAPL", days=30)
        
        # Try to use real TFT model manager
        try:
            tft_model_manager = RealTFTModelManager()
            plotter = OHLCPlotter(tft_model_manager)
            print("✅ Using real TFT model for demo")
        except Exception as e:
            print(f"⚠️  Could not load real TFT model: {e}")
            print("Creating plot with just sample data...")
            
            # Create plotter with None (will skip model-based predictions)
            plotter = OHLCPlotter(None)
        
        signals = []
        for i in range(len(actual_data.timestamps)):
            signal_type = np.random.choice(['BUY', 'SELL', 'HOLD'], p=[0.3, 0.3, 0.4])
            signals.append({
                'signal': signal_type,
                'confidence': np.random.uniform(0.3, 0.9),
                'timestamp': actual_data.timestamps[i]
            })
        
        output_dir = "plots"
        os.makedirs(output_dir, exist_ok=True)
        
        fig1 = plotter.plot_ohlc_vs_predictions(
            actual_data, predicted_data, "AAPL (Mock Data)",
            save_path=os.path.join(output_dir, "ohlc_comparison_mock.png")
        )
        
        fig2 = plotter.plot_trading_signals(
            actual_data.close, actual_data.timestamps, signals, "AAPL (Mock Data)",
            save_path=os.path.join(output_dir, "trading_signals_mock.png")
        )
        
        fig3 = plotter.create_comprehensive_dashboard(
            actual_data, predicted_data, signals, "AAPL (Mock Data)",
            save_path=os.path.join(output_dir, "trading_dashboard_mock.png")
        )
        
        plt.show()
        print(f"\nMock demo plots saved to '{output_dir}' directory")

if __name__ == "__main__":
    demo_ohlc_plotting()
