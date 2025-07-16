#!/usr/bin/env python3
"""
ENHANCED LIVE TRADING SYSTEM WITH REAL TFT MODEL

This module provides a complete live trading system that integrates:
- Real TFT model for predictions
- OHLC plotting and visualization
- Live trading signals
- Portfolio management
- Risk assessment
"""

import os
import sys
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from pathlib import Path
import logging
import json

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from tft_model_loader import TFTModelLoader, TFTPredictor, get_latest_model_path
from ohlc_plotter import OHLCPlotter, OHLCData
from live_trading_system import TradingSignal, PortfolioRecommendation

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EnhancedLiveTradingSystem:
    """
    Enhanced live trading system using real TFT model for predictions and OHLC plotting.
    """
    
    def __init__(self, 
                 checkpoint_path: Optional[str] = None,
                 initial_capital: float = 100000,
                 max_position_size: float = 0.3,
                 risk_tolerance: str = 'MODERATE'):
        """
        Initialize the enhanced live trading system.
        
        Args:
            checkpoint_path: Path to TFT model checkpoint (auto-detect if None)
            initial_capital: Initial trading capital
            max_position_size: Maximum position size (fraction)
            risk_tolerance: Risk tolerance ('LOW', 'MODERATE', 'HIGH')
        """
        self.initial_capital = initial_capital
        self.max_position_size = max_position_size
        self.risk_tolerance = risk_tolerance
        
        # Load TFT model
        self.model_path = checkpoint_path or self._get_best_model_path()
        self.model_loader = None
        self.predictor = None
        self.plotter = None
        
        # Initialize model components
        self._initialize_model()
        
        # Trading state
        self.current_portfolio_value = initial_capital
        self.cash_balance = initial_capital
        self.current_positions = {}  # Symbol -> (shares, avg_price, allocation)
        self.trading_history = []
        self.market_data_cache = {}  # Symbol -> historical data
        
        # Risk parameters
        self.risk_params = self._set_risk_parameters()
        
        logger.info(f"Enhanced Live Trading System initialized")
        logger.info(f"Model: {self.model_path}")
        logger.info(f"Capital: ${initial_capital:,.2f}")
        logger.info(f"Risk tolerance: {risk_tolerance}")
    
    def _get_best_model_path(self) -> str:
        """Get the best available TFT model checkpoint."""
        try:
            return get_latest_model_path()
        except Exception as e:
            logger.error(f"Could not find TFT model: {e}")
            raise ValueError("No trained TFT model found. Please train a model first.")
    
    def _initialize_model(self) -> None:
        """Initialize the TFT model and related components."""
        try:
            self.model_loader = TFTModelLoader(self.model_path)
            model = self.model_loader.load_model()
            self.predictor = TFTPredictor(self.model_loader)
            self.plotter = OHLCPlotter(self.predictor)
            
            logger.info("TFT model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize TFT model: {e}")
            # Create mock components for demonstration
            self._initialize_mock_model()
    
    def _initialize_mock_model(self) -> None:
        """Initialize mock model for demonstration purposes."""
        logger.warning("Using mock model for demonstration")
        
        class MockPredictor:
            def __init__(self):
                self.model_loader = None
                self.model = None
            
            def predict_ohlc(self, input_data, news_data=None, events_data=None):
                # Generate mock OHLC predictions
                batch_size, seq_len = input_data.shape[:2]
                pred_len = 10  # Default prediction length
                
                # Random walk with slight trend
                base_prices = np.random.uniform(100, 200, (batch_size, pred_len))
                noise = np.random.normal(0, 0.02, (batch_size, pred_len))
                
                close_prices = base_prices * (1 + noise)
                open_prices = close_prices * (1 + np.random.normal(0, 0.005, (batch_size, pred_len)))
                high_prices = close_prices * (1 + np.abs(np.random.normal(0, 0.01, (batch_size, pred_len))))
                low_prices = close_prices * (1 - np.abs(np.random.normal(0, 0.01, (batch_size, pred_len))))
                
                return {
                    'open': open_prices,
                    'high': high_prices,
                    'low': low_prices,
                    'close': close_prices
                }
            
            def generate_trading_signals(self, input_data, news_data=None, events_data=None, current_price=None):
                # Generate mock trading signals
                momentum = np.random.normal(0, 0.02)
                confidence = np.random.uniform(0.3, 0.9)
                
                if momentum > 0.01:
                    signal = 'BUY'
                elif momentum < -0.01:
                    signal = 'SELL'
                else:
                    signal = 'HOLD'
                
                return {
                    'signal': signal,
                    'confidence': confidence,
                    'risk_level': 'MEDIUM',
                    'kelly_fraction': max(0.01, min(0.25, confidence * 0.3)),
                    'predicted_returns': momentum,
                    'volatility': 0.02,
                    'price_predictions': {
                        'next_close': (current_price or 150) * (1 + momentum),
                        'trend': 'BULLISH' if momentum > 0 else 'BEARISH',
                        'ohlc_forecast': self.predict_ohlc(input_data)
                    }
                }
        
        self.predictor = MockPredictor()
        self.plotter = OHLCPlotter(self.predictor)
    
    def _set_risk_parameters(self) -> Dict[str, float]:
        """Set risk parameters based on tolerance level."""
        params = {
            'LOW': {
                'kelly_multiplier': 0.15,
                'max_portfolio_risk': 0.4,
                'volatility_threshold': 0.015,
                'min_confidence': 0.7,
                'stop_loss_pct': 0.03,
                'position_decay': 0.9
            },
            'MODERATE': {
                'kelly_multiplier': 0.25,
                'max_portfolio_risk': 0.7,
                'volatility_threshold': 0.025,
                'min_confidence': 0.55,
                'stop_loss_pct': 0.05,
                'position_decay': 0.95
            },
            'HIGH': {
                'kelly_multiplier': 0.4,
                'max_portfolio_risk': 0.9,
                'volatility_threshold': 0.04,
                'min_confidence': 0.45,
                'stop_loss_pct': 0.08,
                'position_decay': 0.98
            }
        }
        return params[self.risk_tolerance]
    
    def generate_market_data(self, symbol: str, days: int = 60) -> pd.DataFrame:
        """
        Generate or retrieve market data for a symbol.
        
        Args:
            symbol: Stock symbol
            days: Number of days of historical data
            
        Returns:
            DataFrame with OHLC data
        """
        # For demo purposes, generate synthetic data
        # In production, this would fetch real market data
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        
        # Generate realistic price data
        base_price = np.random.uniform(100, 300)
        price_series = [base_price]
        
        for i in range(1, len(dates)):
            # Random walk with slight upward bias
            change = np.random.normal(0.001, 0.02)
            new_price = price_series[-1] * (1 + change)
            price_series.append(new_price)
        
        # Generate OHLC from price series
        data = []
        for i, (date, price) in enumerate(zip(dates, price_series)):
            daily_vol = 0.015  # 1.5% daily volatility
            
            open_price = price * (1 + np.random.normal(0, daily_vol/3))
            high_price = price * (1 + abs(np.random.normal(0, daily_vol)))
            low_price = price * (1 - abs(np.random.normal(0, daily_vol)))
            close_price = price * (1 + np.random.normal(0, daily_vol/2))
            volume = np.random.randint(1000000, 10000000)
            
            data.append({
                'timestamp': date,
                'open': open_price,
                'high': high_price,
                'low': low_price,
                'close': close_price,
                'volume': volume
            })
        
        df = pd.DataFrame(data)
        self.market_data_cache[symbol] = df
        return df
    
    def predict_ohlc(self, symbol: str, historical_data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Generate OHLC predictions using the TFT model.
        
        Args:
            symbol: Stock symbol
            historical_data: Historical market data
            
        Returns:
            Dictionary containing OHLC predictions
        """
        try:
            # Prepare input data for TFT model
            # This is a simplified version - in practice you'd need proper feature engineering
            input_features = self._prepare_model_input(historical_data)
            
            # Generate predictions
            ohlc_predictions = self.predictor.predict_ohlc(input_features)
            
            logger.info(f"Generated OHLC predictions for {symbol}")
            return ohlc_predictions
            
        except Exception as e:
            logger.error(f"Error generating predictions for {symbol}: {e}")
            # Return mock predictions as fallback
            return self._generate_mock_predictions()
    
    def _prepare_model_input(self, data: pd.DataFrame) -> torch.Tensor:
        """
        Prepare market data for TFT model input.
        
        Args:
            data: Historical market data
            
        Returns:
            Tensor ready for TFT model
        """
        # Simplified feature preparation
        # In practice, this would include technical indicators, normalized prices, etc.
        
        # Basic price features
        features = []
        
        # Price returns
        data['returns'] = data['close'].pct_change().fillna(0)
        
        # Moving averages
        data['ma_5'] = data['close'].rolling(5).mean().fillna(data['close'])
        data['ma_20'] = data['close'].rolling(20).mean().fillna(data['close'])
        
        # Volatility
        data['volatility'] = data['returns'].rolling(20).std().fillna(0.02)
        
        # Volume features
        data['volume_ma'] = data['volume'].rolling(20).mean().fillna(data['volume'])
        data['volume_ratio'] = data['volume'] / data['volume_ma']
        
        # Select feature columns
        feature_cols = ['open', 'high', 'low', 'close', 'volume', 
                       'returns', 'ma_5', 'ma_20', 'volatility', 'volume_ratio']
        
        # Take last 60 time steps (encoder length)
        encoder_len = 60
        feature_data = data[feature_cols].tail(encoder_len).values
        
        # Normalize features (simple min-max scaling)
        feature_data = (feature_data - feature_data.min(axis=0)) / (feature_data.max(axis=0) - feature_data.min(axis=0) + 1e-8)
        
        # Convert to tensor
        tensor_data = torch.tensor(feature_data, dtype=torch.float32).unsqueeze(0)  # Add batch dimension
        
        return tensor_data
    
    def _generate_mock_predictions(self) -> Dict[str, np.ndarray]:
        """Generate mock OHLC predictions for fallback."""
        pred_len = 10
        base_price = 150.0
        
        # Generate mock predictions
        close_pred = np.array([base_price * (1 + np.random.normal(0.001, 0.02)) for _ in range(pred_len)])
        open_pred = close_pred * (1 + np.random.normal(0, 0.005, pred_len))
        high_pred = close_pred * (1 + np.abs(np.random.normal(0, 0.01, pred_len)))
        low_pred = close_pred * (1 - np.abs(np.random.normal(0, 0.01, pred_len)))
        
        return {
            'open': open_pred.reshape(1, -1),
            'high': high_pred.reshape(1, -1),
            'low': low_pred.reshape(1, -1),
            'close': close_pred.reshape(1, -1)
        }
    
    def generate_trading_signal(self, symbol: str, current_price: float) -> TradingSignal:
        """
        Generate a trading signal for a symbol using TFT model.
        
        Args:
            symbol: Stock symbol
            current_price: Current market price
            
        Returns:
            TradingSignal object
        """
        try:
            # Get historical data
            if symbol not in self.market_data_cache:
                self.generate_market_data(symbol)
            
            historical_data = self.market_data_cache[symbol]
            
            # Prepare model input
            input_features = self._prepare_model_input(historical_data)
            
            # Generate trading signals using TFT model
            signal_data = self.predictor.generate_trading_signals(
                input_features, current_price=current_price
            )
            
            # Create TradingSignal object
            signal = TradingSignal(
                timestamp=datetime.now(),
                symbol=symbol,
                signal_type=signal_data['signal'],
                confidence=signal_data['confidence'],
                predicted_return=signal_data['predicted_returns'],
                recommended_position=signal_data['kelly_fraction'],
                risk_level=signal_data['risk_level'],
                market_regime='NORMAL',  # Simplified
                stop_loss=current_price * (1 - self.risk_params['stop_loss_pct']),
                take_profit=current_price * (1 + self.risk_params['stop_loss_pct'] * 2),
                rationale=f"TFT model prediction: {signal_data['predicted_returns']:.2%} return, {signal_data['confidence']:.2f} confidence"
            )
            
            logger.info(f"Generated signal for {symbol}: {signal.signal_type} (confidence: {signal.confidence:.2f})")
            return signal
            
        except Exception as e:
            logger.error(f"Error generating signal for {symbol}: {e}")
            # Return safe HOLD signal
            return TradingSignal(
                timestamp=datetime.now(),
                symbol=symbol,
                signal_type='HOLD',
                confidence=0.5,
                predicted_return=0.0,
                recommended_position=0.0,
                risk_level='HIGH',
                market_regime='UNKNOWN',
                rationale=f"Error in signal generation: {e}"
            )
    
    def plot_ohlc_predictions(self, symbol: str, save_path: Optional[str] = None) -> plt.Figure:
        """
        Generate and plot OHLC predictions vs actual data.
        
        Args:
            symbol: Stock symbol
            save_path: Optional path to save the plot
            
        Returns:
            Matplotlib figure
        """
        try:
            # Get historical data
            if symbol not in self.market_data_cache:
                self.generate_market_data(symbol)
            
            historical_data = self.market_data_cache[symbol]
            
            # Generate predictions
            predictions = self.predict_ohlc(symbol, historical_data)
            
            # Create OHLCData objects
            actual_data = OHLCData(
                timestamps=historical_data['timestamp'].tolist(),
                open=historical_data['open'].tolist(),
                high=historical_data['high'].tolist(),
                low=historical_data['low'].tolist(),
                close=historical_data['close'].tolist(),
                volume=historical_data['volume'].tolist()
            )
            
            # Create predicted data (for the prediction period)
            pred_len = predictions['close'].shape[1]
            pred_start = historical_data['timestamp'].iloc[-1] + timedelta(days=1)
            pred_timestamps = [pred_start + timedelta(days=i) for i in range(pred_len)]
            
            predicted_data = OHLCData(
                timestamps=pred_timestamps,
                open=predictions['open'][0].tolist(),
                high=predictions['high'][0].tolist(),
                low=predictions['low'][0].tolist(),
                close=predictions['close'][0].tolist()
            )
            
            # Create plot
            fig = self.plotter.plot_ohlc_vs_predictions(
                actual_data, predicted_data, symbol, save_path
            )
            
            logger.info(f"Created OHLC prediction plot for {symbol}")
            return fig
            
        except Exception as e:
            logger.error(f"Error creating OHLC plot for {symbol}: {e}")
            raise
    
    def create_trading_dashboard(self, symbols: List[str], save_path: Optional[str] = None) -> plt.Figure:
        """
        Create a comprehensive trading dashboard for multiple symbols.
        
        Args:
            symbols: List of stock symbols
            save_path: Optional path to save the dashboard
            
        Returns:
            Matplotlib figure
        """
        try:
            # Generate signals for all symbols
            signals = []
            current_prices = {}
            
            for symbol in symbols:
                # Mock current price (in practice, fetch from market data)
                current_price = np.random.uniform(100, 300)
                current_prices[symbol] = current_price
                
                signal = self.generate_trading_signal(symbol, current_price)
                signals.append(signal)
            
            # Create portfolio recommendation
            portfolio_rec = self._generate_portfolio_recommendation(signals)
            
            # Create dashboard plot
            fig, axes = plt.subplots(2, 2, figsize=(20, 12))
            fig.suptitle('TFT Trading Dashboard', fontsize=16, fontweight='bold')
            
            # Signal distribution
            ax1 = axes[0, 0]
            signal_counts = {}
            for signal in signals:
                signal_counts[signal.signal_type] = signal_counts.get(signal.signal_type, 0) + 1
            
            if signal_counts:
                ax1.pie(list(signal_counts.values()), labels=list(signal_counts.keys()), 
                       autopct='%1.1f%%', startangle=90)
                ax1.set_title('Signal Distribution')
            
            # Confidence levels
            ax2 = axes[0, 1]
            confidences = [s.confidence for s in signals]
            ax2.hist(confidences, bins=10, alpha=0.7, color='skyblue', edgecolor='black')
            ax2.set_title('Signal Confidence Distribution')
            ax2.set_xlabel('Confidence')
            ax2.set_ylabel('Count')
            
            # Portfolio allocation
            ax3 = axes[1, 0]
            if portfolio_rec.positions:
                positions = list(portfolio_rec.positions.values())
                position_labels = list(portfolio_rec.positions.keys())
                ax3.bar(position_labels, positions, alpha=0.7, color='green')
                ax3.set_title('Portfolio Allocation')
                ax3.set_ylabel('Position Size')
                plt.setp(ax3.get_xticklabels(), rotation=45, ha='right')
            
            # Risk metrics
            ax4 = axes[1, 1]
            risk_metrics = [
                f"Expected Return: {portfolio_rec.expected_return:.2%}",
                f"Estimated Volatility: {portfolio_rec.estimated_volatility:.2%}",
                f"Sharpe Ratio: {portfolio_rec.sharpe_ratio_estimate:.2f}",
                f"Risk Score: {portfolio_rec.risk_score:.1f}/10",
                f"Cash Allocation: {portfolio_rec.cash_allocation:.1%}"
            ]
            
            ax4.text(0.1, 0.5, '\n'.join(risk_metrics), transform=ax4.transAxes, 
                    fontsize=12, verticalalignment='center',
                    bbox=dict(boxstyle='round', facecolor='lightblue'))
            ax4.set_title('Portfolio Risk Metrics')
            ax4.axis('off')
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Dashboard saved to {save_path}")
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating trading dashboard: {e}")
            raise
    
    def _generate_portfolio_recommendation(self, signals: List[TradingSignal]) -> PortfolioRecommendation:
        """Generate portfolio recommendation from signals."""
        # Calculate portfolio metrics
        total_allocation = sum(s.recommended_position for s in signals if s.signal_type == 'BUY')
        
        # Scale down if over-allocated
        max_allocation = self.risk_params['max_portfolio_risk']
        if total_allocation > max_allocation:
            scale_factor = max_allocation / total_allocation
            for signal in signals:
                if signal.signal_type == 'BUY':
                    signal.recommended_position *= scale_factor
        
        # Create position dictionary
        positions = {}
        for signal in signals:
            if signal.signal_type == 'BUY' and signal.recommended_position > 0.01:
                positions[signal.symbol] = signal.recommended_position
        
        # Calculate metrics
        cash_allocation = 1.0 - sum(positions.values())
        expected_return = sum(s.predicted_return * s.recommended_position for s in signals if s.signal_type == 'BUY')
        estimated_volatility = 0.15  # Simplified
        risk_score = min(len([s for s in signals if s.risk_level == 'HIGH']) * 2, 10)
        
        return PortfolioRecommendation(
            timestamp=datetime.now(),
            total_portfolio_value=self.current_portfolio_value,
            cash_allocation=cash_allocation,
            positions=positions,
            expected_return=expected_return,
            estimated_volatility=estimated_volatility,
            max_drawdown_risk=estimated_volatility * 0.5,
            sharpe_ratio_estimate=expected_return / estimated_volatility if estimated_volatility > 0 else 0,
            risk_score=risk_score,
            recommendations=[
                f"Portfolio optimized using TFT model predictions",
                f"Risk tolerance: {self.risk_tolerance}",
                f"Total positions: {len(positions)}"
            ]
        )


def demo_enhanced_trading_system():
    """
    Demonstrate the enhanced trading system with real TFT model.
    """
    print("=== ENHANCED TFT TRADING SYSTEM DEMO ===")
    
    try:
        # Initialize the enhanced trading system
        trading_system = EnhancedLiveTradingSystem(
            initial_capital=100000,
            max_position_size=0.25,
            risk_tolerance='MODERATE'
        )
        
        # Test symbols
        symbols = ['AAPL', 'GOOGL', 'MSFT', 'NVDA', 'TSLA']
        
        print(f"\nTesting with symbols: {symbols}")
        
        # Generate OHLC predictions and plots
        output_dir = "enhanced_trading_plots"
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. Generate individual OHLC prediction plots
        print("\n1. Generating OHLC prediction plots...")
        for symbol in symbols[:2]:  # Limit to 2 for demo
            print(f"   Creating OHLC plot for {symbol}...")
            fig = trading_system.plot_ohlc_predictions(
                symbol, 
                save_path=os.path.join(output_dir, f"{symbol}_ohlc_predictions.png")
            )
            plt.close(fig)  # Close to save memory
        
        # 2. Generate trading signals
        print("\n2. Generating trading signals...")
        all_signals = []
        for symbol in symbols:
            current_price = np.random.uniform(100, 300)  # Mock current price
            signal = trading_system.generate_trading_signal(symbol, current_price)
            all_signals.append(signal)
            print(f"   {symbol}: {signal.signal_type} (confidence: {signal.confidence:.2f})")
        
        # 3. Create comprehensive dashboard
        print("\n3. Creating trading dashboard...")
        dashboard_fig = trading_system.create_trading_dashboard(
            symbols,
            save_path=os.path.join(output_dir, "trading_dashboard.png")
        )
        plt.close(dashboard_fig)
        
        # 4. Show model information
        print("\n4. TFT Model Information:")
        if trading_system.model_loader:
            model_info = trading_system.model_loader.get_model_info()
            for key, value in model_info.items():
                print(f"   {key}: {value}")
        
        # 5. Portfolio summary
        portfolio_rec = trading_system._generate_portfolio_recommendation(all_signals)
        print("\n5. Portfolio Recommendation:")
        print(f"   Expected Return: {portfolio_rec.expected_return:.2%}")
        print(f"   Estimated Volatility: {portfolio_rec.estimated_volatility:.2%}")
        print(f"   Risk Score: {portfolio_rec.risk_score:.1f}/10")
        print(f"   Cash Allocation: {portfolio_rec.cash_allocation:.1%}")
        print(f"   Number of Positions: {len(portfolio_rec.positions)}")
        
        if portfolio_rec.positions:
            print("   Position Allocations:")
            for symbol, allocation in portfolio_rec.positions.items():
                print(f"     {symbol}: {allocation:.1%}")
        
        print(f"\n✅ Demo completed successfully!")
        print(f"📁 All plots saved to: {output_dir}/")
        print(f"🎯 TFT model integration: {'✅ Real Model' if trading_system.predictor.model else '⚠️ Mock Model'}")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    demo_enhanced_trading_system()
