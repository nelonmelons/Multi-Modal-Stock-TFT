"""
LIVE TRADING SYSTEM

This module extends the trading simulator for real-time trading with:
- Live market data integration
- Real-time trade signals
- Portfolio recommendations
- Dynamic risk estimation
- Position management
"""

import numpy as np
import pandas as pd
import torch
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from pathlib import Path
import json
import logging

# Import our real TFT model manager
from Hayson.TFT.real_tft_integration import RealTFTModelManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class TradingSignal:
    """Real-time trading signal with all necessary information."""
    timestamp: datetime
    symbol: str
    signal_type: str  # 'BUY', 'SELL', 'HOLD'
    confidence: float  # 0-1
    predicted_return: float  # Expected return
    recommended_position: float  # 0-1 (fraction of portfolio)
    risk_level: str  # 'LOW', 'MEDIUM', 'HIGH'
    market_regime: str  # 'LOW_VOLATILITY', 'NORMAL', 'HIGH_VOLATILITY'
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    rationale: str = ""

@dataclass
class PortfolioRecommendation:
    """Complete portfolio recommendation with risk analysis."""
    timestamp: datetime
    total_portfolio_value: float
    cash_allocation: float  # Fraction to keep in cash
    positions: Dict[str, float]  # Symbol -> allocation fraction
    expected_return: float
    estimated_volatility: float
    max_drawdown_risk: float
    sharpe_ratio_estimate: float
    risk_score: float  # 0-10 scale
    recommendations: List[str]  # Human-readable advice

class LiveTradingSystem:
    """
    Complete live trading system that provides real-time signals and portfolio management.
    """
    
    def __init__(self, 
                 tft_model_manager: RealTFTModelManager,
                 initial_capital: float = 100000,
                 max_position_size: float = 0.3,  # Max 30% per position
                 risk_tolerance: str = 'MODERATE'):  # LOW, MODERATE, HIGH
        
        self.tft_model_manager = tft_model_manager
        self.initial_capital = initial_capital
        self.current_portfolio_value = initial_capital
        self.max_position_size = max_position_size
        self.risk_tolerance = risk_tolerance
        
        # Risk parameters based on tolerance
        self.risk_params = self._set_risk_parameters()
        
        # Portfolio state
        self.current_positions = {}  # Symbol -> (shares, avg_price, allocation)
        self.cash_balance = initial_capital
        self.trading_history = []
        self.performance_metrics = {}
        
        # Market state tracking
        self.market_data_buffer = {}  # Symbol -> recent price data
        self.volatility_buffer = {}   # Symbol -> recent volatility
        self.accuracy_tracker = {}    # Symbol -> recent prediction accuracy
        
        logger.info(f"Live Trading System initialized with ${initial_capital:,.2f}")
        logger.info(f"Risk tolerance: {risk_tolerance}, Max position: {max_position_size:.1%}")
    
    def _set_risk_parameters(self) -> Dict[str, float]:
        """Set risk parameters based on user's risk tolerance."""
        params = {
            'LOW': {
                'kelly_multiplier': 0.15,      # Very conservative Kelly
                'max_portfolio_risk': 0.4,     # Max 40% invested
                'volatility_threshold': 0.015, # Lower vol threshold
                'min_confidence': 0.7,         # Higher confidence requirement
                'stop_loss_pct': 0.03,         # 3% stop loss
                'position_decay': 0.9          # Faster position decay
            },
            'MODERATE': {
                'kelly_multiplier': 0.25,      # Standard Kelly multiplier
                'max_portfolio_risk': 0.7,     # Max 70% invested
                'volatility_threshold': 0.025, # Standard vol threshold
                'min_confidence': 0.55,        # Moderate confidence
                'stop_loss_pct': 0.05,         # 5% stop loss
                'position_decay': 0.95         # Standard decay
            },
            'HIGH': {
                'kelly_multiplier': 0.4,       # Aggressive Kelly
                'max_portfolio_risk': 0.9,     # Max 90% invested
                'volatility_threshold': 0.04,  # Higher vol tolerance
                'min_confidence': 0.45,        # Lower confidence OK
                'stop_loss_pct': 0.08,         # 8% stop loss
                'position_decay': 0.98         # Slower decay
            }
        }
        return params[self.risk_tolerance]
    
    def update_market_data(self, symbol: str, price_data: pd.DataFrame) -> None:
        """
        Update market data buffer for a symbol.
        
        Args:
            symbol: Stock symbol
            price_data: DataFrame with columns ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        """
        # Store recent data (last 100 periods)
        self.market_data_buffer[symbol] = price_data.tail(100).copy()
        
        # Calculate and store volatility
        returns = price_data['close'].pct_change().dropna()
        self.volatility_buffer[symbol] = returns.rolling(20).std().iloc[-1]
        
        logger.debug(f"Updated market data for {symbol}, current volatility: {self.volatility_buffer[symbol]:.4f}")
    
    def generate_live_signal(self, 
                           symbol: str, 
                           current_features: np.ndarray,
                           current_price: float) -> TradingSignal:
        """
        Generate real-time trading signal for a symbol.
        
        Args:
            symbol: Stock symbol
            current_features: Feature array for TFT model
            current_price: Current market price
            
        Returns:
            TradingSignal with recommendation
        """
        timestamp = datetime.now()
        
        try:
            # Step 1: Get TFT model prediction
            with torch.no_grad():
                # Ensure features are in the right format for the model
                if isinstance(current_features, np.ndarray):
                    features_tensor = torch.tensor(current_features, dtype=torch.float32)
                else:
                    features_tensor = current_features
                
                # Add batch dimension if needed
                if features_tensor.dim() == 2:
                    features_tensor = features_tensor.unsqueeze(0)
                
                # Get model prediction
                trading_signals = self.tft_model_manager.generate_trading_signals(features_tensor, current_price=current_price)
                predicted_return = trading_signals['predicted_returns']
                confidence = trading_signals['confidence']
            
            # Step 3: Assess market regime
            market_regime = self._assess_market_regime(symbol)
            
            # Step 4: Calculate recommended position size
            recommended_position = self._calculate_position_size(
                symbol, predicted_return, confidence, market_regime
            )
            
            # Step 5: Determine signal type
            signal_type = self._determine_signal_type(
                symbol, predicted_return, confidence, recommended_position
            )
            
            # Step 6: Risk assessment
            risk_level = self._assess_risk_level(symbol, recommended_position, market_regime)
            
            # Step 7: Calculate stop loss and take profit
            stop_loss, take_profit = self._calculate_stop_loss_take_profit(
                current_price, predicted_return, risk_level
            )
            
            # Step 8: Generate rationale
            rationale = self._generate_rationale(
                predicted_return, confidence, market_regime, recommended_position
            )
            
            signal = TradingSignal(
                timestamp=timestamp,
                symbol=symbol,
                signal_type=signal_type,
                confidence=confidence,
                predicted_return=predicted_return,
                recommended_position=recommended_position,
                risk_level=risk_level,
                market_regime=market_regime,
                stop_loss=stop_loss,
                take_profit=take_profit,
                rationale=rationale
            )
            
            logger.info(f"Generated signal for {symbol}: {signal_type} {recommended_position:.1%} (confidence: {confidence:.2f})")
            
            return signal
            
        except Exception as e:
            logger.error(f"Error generating signal for {symbol}: {e}")
            # Return safe signal
            return TradingSignal(
                timestamp=timestamp,
                symbol=symbol,
                signal_type='HOLD',
                confidence=0.0,
                predicted_return=0.0,
                recommended_position=0.0,
                risk_level='HIGH',
                market_regime='UNKNOWN',
                rationale=f"Error in signal generation: {e}"
            )
    
    def _calculate_confidence(self, symbol: str, predicted_return: float) -> float:
        """Calculate confidence in the prediction."""
        # Base confidence from prediction magnitude
        base_confidence = min(abs(predicted_return) / 0.05, 1.0)  # Normalize by 5% max
        
        # Adjust for recent model accuracy
        if symbol in self.accuracy_tracker:
            accuracy_adjustment = self.accuracy_tracker[symbol]
            confidence = base_confidence * accuracy_adjustment
        else:
            confidence = base_confidence * 0.6  # Default moderate confidence
        
        return np.clip(confidence, 0.0, 1.0)
    
    def _assess_market_regime(self, symbol: str) -> str:
        """Assess current market regime for the symbol."""
        if symbol not in self.volatility_buffer:
            return 'UNKNOWN'
        
        current_vol = self.volatility_buffer[symbol]
        vol_threshold = self.risk_params['volatility_threshold']
        
        if current_vol > vol_threshold * 1.5:
            return 'HIGH_VOLATILITY'
        elif current_vol < vol_threshold * 0.7:
            return 'LOW_VOLATILITY'
        else:
            return 'NORMAL'
    
    def _calculate_position_size(self, symbol: str, predicted_return: float, 
                               confidence: float, market_regime: str) -> float:
        """Calculate optimal position size using Kelly Criterion."""
        
        # Estimate win probability
        base_prob = 0.52
        confidence_boost = confidence * 0.15
        win_prob = np.clip(base_prob + confidence_boost, 0.45, 0.75)
        
        # Kelly fraction
        kelly_fraction = 2 * win_prob - 1
        
        # Apply conservative multiplier based on risk tolerance
        kelly_multiplier = self.risk_params['kelly_multiplier']
        base_position = kelly_fraction * kelly_multiplier
        
        # Market regime adjustment
        regime_multipliers = {
            'HIGH_VOLATILITY': 0.5,
            'NORMAL': 1.0,
            'LOW_VOLATILITY': 1.2,
            'UNKNOWN': 0.3
        }
        
        adjusted_position = base_position * regime_multipliers[market_regime]
        
        # Apply position limits
        final_position = np.clip(adjusted_position, 0.0, self.max_position_size)
        
        # Check if confidence meets minimum threshold
        if confidence < self.risk_params['min_confidence']:
            final_position *= 0.5  # Reduce position for low confidence
        
        return final_position
    
    def _determine_signal_type(self, symbol: str, predicted_return: float, 
                             confidence: float, recommended_position: float) -> str:
        """Determine the signal type based on prediction and position size."""
        
        min_position_threshold = 0.02  # 2% minimum position
        
        if recommended_position < min_position_threshold:
            return 'HOLD'
        elif predicted_return > 0:
            return 'BUY'
        else:
            # In this system, we don't short, so negative predictions = hold cash
            return 'HOLD'
    
    def _assess_risk_level(self, symbol: str, position_size: float, market_regime: str) -> str:
        """Assess the risk level of the trade."""
        
        risk_score = 0
        
        # Position size risk
        if position_size > 0.2:
            risk_score += 2
        elif position_size > 0.1:
            risk_score += 1
        
        # Market regime risk
        if market_regime == 'HIGH_VOLATILITY':
            risk_score += 2
        elif market_regime == 'UNKNOWN':
            risk_score += 1
        
        # Symbol-specific volatility
        if symbol in self.volatility_buffer:
            vol = self.volatility_buffer[symbol]
            if vol > 0.03:  # High volatility
                risk_score += 1
        
        if risk_score >= 3:
            return 'HIGH'
        elif risk_score >= 1:
            return 'MEDIUM'
        else:
            return 'LOW'
    
    def _calculate_stop_loss_take_profit(self, current_price: float, 
                                       predicted_return: float, 
                                       risk_level: str) -> Tuple[Optional[float], Optional[float]]:
        """Calculate stop loss and take profit levels."""
        
        # Stop loss based on risk level
        stop_loss_pct = self.risk_params['stop_loss_pct']
        
        if risk_level == 'HIGH':
            stop_loss_pct *= 0.7  # Tighter stop loss for high risk
        elif risk_level == 'LOW':
            stop_loss_pct *= 1.3  # Wider stop loss for low risk
        
        stop_loss = current_price * (1 - stop_loss_pct)
        
        # Take profit at 2x the risk (risk-reward ratio of 1:2)
        if predicted_return > 0:
            take_profit = current_price * (1 + stop_loss_pct * 2)
        else:
            take_profit = None
        
        return stop_loss, take_profit
    
    def _generate_rationale(self, predicted_return: float, confidence: float, 
                          market_regime: str, position_size: float) -> str:
        """Generate human-readable rationale for the signal."""
        
        rationale_parts = []
        
        # Prediction component
        direction = "bullish" if predicted_return > 0 else "bearish"
        magnitude = "strong" if abs(predicted_return) > 0.02 else "moderate" if abs(predicted_return) > 0.01 else "weak"
        rationale_parts.append(f"Model shows {magnitude} {direction} signal ({predicted_return:.2%})")
        
        # Confidence component
        conf_level = "high" if confidence > 0.7 else "moderate" if confidence > 0.5 else "low"
        rationale_parts.append(f"{conf_level} confidence ({confidence:.2f})")
        
        # Market regime component
        rationale_parts.append(f"market regime: {market_regime.lower().replace('_', ' ')}")
        
        # Position sizing rationale
        if position_size > 0.15:
            rationale_parts.append("significant position recommended")
        elif position_size > 0.05:
            rationale_parts.append("moderate position recommended")
        else:
            rationale_parts.append("small/no position recommended")
        
        return "; ".join(rationale_parts)
    
    def generate_portfolio_recommendation(self, symbols: List[str], 
                                        signals: List[TradingSignal]) -> PortfolioRecommendation:
        """
        Generate comprehensive portfolio recommendation based on multiple signals.
        """
        timestamp = datetime.now()
        
        # Calculate current portfolio value
        total_value = self._calculate_portfolio_value()
        
        # Aggregate all signal recommendations
        total_allocation = sum(signal.recommended_position for signal in signals if signal.signal_type == 'BUY')
        
        # Ensure we don't over-allocate
        max_allocation = self.risk_params['max_portfolio_risk']
        if total_allocation > max_allocation:
            # Scale down proportionally
            scale_factor = max_allocation / total_allocation
            for signal in signals:
                if signal.signal_type == 'BUY':
                    signal.recommended_position *= scale_factor
        
        # Calculate position allocations
        positions = {}
        for signal in signals:
            if signal.signal_type == 'BUY' and signal.recommended_position > 0.01:
                positions[signal.symbol] = signal.recommended_position
        
        # Calculate cash allocation
        cash_allocation = 1.0 - sum(positions.values())
        
        # Portfolio risk assessment
        portfolio_risk = self._assess_portfolio_risk(signals, positions)
        
        # Expected return calculation
        expected_return = sum(
            signal.predicted_return * signal.recommended_position 
            for signal in signals if signal.signal_type == 'BUY'
        )
        
        # Volatility estimation
        estimated_volatility = self._estimate_portfolio_volatility(signals, positions)
        
        # Generate recommendations
        recommendations = self._generate_portfolio_recommendations(signals, positions, portfolio_risk)
        
        portfolio_rec = PortfolioRecommendation(
            timestamp=timestamp,
            total_portfolio_value=total_value,
            cash_allocation=cash_allocation,
            positions=positions,
            expected_return=expected_return,
            estimated_volatility=estimated_volatility,
            max_drawdown_risk=portfolio_risk['max_drawdown_risk'],
            sharpe_ratio_estimate=expected_return / estimated_volatility if estimated_volatility > 0 else 0,
            risk_score=portfolio_risk['risk_score'],
            recommendations=recommendations
        )
        
        logger.info(f"Generated portfolio recommendation: {len(positions)} positions, {cash_allocation:.1%} cash")
        
        return portfolio_rec
    
    def _calculate_portfolio_value(self) -> float:
        """Calculate current total portfolio value."""
        total_value = self.cash_balance
        
        for symbol, (shares, avg_price, allocation) in self.current_positions.items():
            if symbol in self.market_data_buffer:
                current_price = self.market_data_buffer[symbol]['close'].iloc[-1]
                position_value = shares * current_price
                total_value += position_value
        
        return total_value
    
    def _assess_portfolio_risk(self, signals: List[TradingSignal], 
                             positions: Dict[str, float]) -> Dict[str, float]:
        """Assess overall portfolio risk."""
        
        # Risk score calculation
        risk_score = 0
        
        # Concentration risk
        max_position = max(positions.values()) if positions else 0
        if max_position > 0.3:
            risk_score += 3
        elif max_position > 0.2:
            risk_score += 2
        elif max_position > 0.1:
            risk_score += 1
        
        # Number of positions (diversification)
        num_positions = len(positions)
        if num_positions < 3:
            risk_score += 2
        elif num_positions < 5:
            risk_score += 1
        
        # High volatility exposure
        high_vol_exposure = sum(
            positions.get(signal.symbol, 0) 
            for signal in signals 
            if signal.market_regime == 'HIGH_VOLATILITY'
        )
        
        if high_vol_exposure > 0.3:
            risk_score += 2
        elif high_vol_exposure > 0.15:
            risk_score += 1
        
        # Estimate maximum drawdown risk
        max_drawdown_risk = min(sum(positions.values()) * 0.3, 0.25)  # Conservative estimate
        
        return {
            'risk_score': min(risk_score, 10),  # Cap at 10
            'max_drawdown_risk': max_drawdown_risk,
            'concentration_risk': max_position,
            'high_vol_exposure': high_vol_exposure
        }
    
    def _estimate_portfolio_volatility(self, signals: List[TradingSignal], 
                                     positions: Dict[str, float]) -> float:
        """Estimate portfolio volatility."""
        
        # Simple weighted average of individual volatilities
        # In practice, you'd want to account for correlations
        total_vol = 0
        
        for signal in signals:
            if signal.symbol in positions and signal.symbol in self.volatility_buffer:
                weight = positions[signal.symbol]
                symbol_vol = self.volatility_buffer[signal.symbol]
                total_vol += weight * symbol_vol
        
        # Add some base volatility for cash
        cash_weight = 1.0 - sum(positions.values())
        total_vol += cash_weight * 0.001  # Very low volatility for cash
        
        return total_vol
    
    def _generate_portfolio_recommendations(self, signals: List[TradingSignal], 
                                          positions: Dict[str, float], 
                                          risk_assessment: Dict[str, float]) -> List[str]:
        """Generate human-readable portfolio recommendations."""
        
        recommendations = []
        
        # Risk level recommendation
        risk_score = risk_assessment['risk_score']
        if risk_score > 7:
            recommendations.append("⚠️  HIGH RISK: Consider reducing position sizes or waiting for better opportunities")
        elif risk_score > 4:
            recommendations.append("🔶 MODERATE RISK: Portfolio is balanced but monitor closely")
        else:
            recommendations.append("✅ LOW RISK: Portfolio appears well-positioned")
        
        # Diversification recommendation
        num_positions = len(positions)
        if num_positions < 3:
            recommendations.append("📊 Consider adding more positions for better diversification")
        elif num_positions > 10:
            recommendations.append("📊 Consider consolidating positions to reduce complexity")
        
        # Concentration risk
        max_position = risk_assessment['concentration_risk']
        if max_position > 0.25:
            recommendations.append(f"⚠️  High concentration risk: Largest position is {max_position:.1%}")
        
        # Market regime warnings
        high_vol_signals = [s for s in signals if s.market_regime == 'HIGH_VOLATILITY']
        if len(high_vol_signals) > len(signals) * 0.5:
            recommendations.append("🌪️  High market volatility detected: Consider reducing risk exposure")
        
        # Signal quality assessment
        high_confidence_signals = [s for s in signals if s.confidence > 0.7]
        if len(high_confidence_signals) == 0:
            recommendations.append("🤔 Low confidence in current signals: Consider waiting for clearer opportunities")
        
        # Cash level recommendation
        cash_allocation = 1.0 - sum(positions.values())
        if cash_allocation > 0.5:
            recommendations.append(f"💰 High cash allocation ({cash_allocation:.1%}): Look for investment opportunities")
        elif cash_allocation < 0.1:
            recommendations.append("💰 Low cash reserves: Consider keeping some dry powder")
        
        return recommendations
    
    def update_position(self, symbol: str, shares: int, price: float, action: str) -> None:
        """Update position after trade execution."""
        
        if action == 'BUY':
            if symbol in self.current_positions:
                # Add to existing position
                current_shares, current_avg_price, _ = self.current_positions[symbol]
                new_shares = current_shares + shares
                new_avg_price = ((current_shares * current_avg_price) + (shares * price)) / new_shares
                
                # Calculate new allocation
                position_value = new_shares * price
                new_allocation = position_value / self.current_portfolio_value
                
                self.current_positions[symbol] = (new_shares, new_avg_price, new_allocation)
            else:
                # New position
                position_value = shares * price
                allocation = position_value / self.current_portfolio_value
                self.current_positions[symbol] = (shares, price, allocation)
            
            # Update cash
            self.cash_balance -= shares * price
            
        elif action == 'SELL':
            if symbol in self.current_positions:
                current_shares, avg_price, _ = self.current_positions[symbol]
                
                if shares >= current_shares:
                    # Close entire position
                    del self.current_positions[symbol]
                else:
                    # Partial sale
                    new_shares = current_shares - shares
                    position_value = new_shares * price
                    new_allocation = position_value / self.current_portfolio_value
                    self.current_positions[symbol] = (new_shares, avg_price, new_allocation)
                
                # Update cash
                self.cash_balance += shares * price
        
        # Update portfolio value
        self.current_portfolio_value = self._calculate_portfolio_value()
        
        logger.info(f"Position updated: {action} {shares} shares of {symbol} at ${price:.2f}")
    
    def get_live_dashboard_data(self) -> Dict[str, Any]:
        """Get real-time dashboard data for display."""
        
        portfolio_value = self._calculate_portfolio_value()
        
        # Performance metrics
        total_return = (portfolio_value - self.initial_capital) / self.initial_capital
        
        # Position summary
        position_summary = []
        for symbol, (shares, avg_price, allocation) in self.current_positions.items():
            if symbol in self.market_data_buffer:
                current_price = self.market_data_buffer[symbol]['close'].iloc[-1]
                position_value = shares * current_price
                unrealized_pnl = (current_price - avg_price) * shares
                unrealized_pnl_pct = (current_price - avg_price) / avg_price
                
                position_summary.append({
                    'symbol': symbol,
                    'shares': shares,
                    'avg_price': avg_price,
                    'current_price': current_price,
                    'position_value': position_value,
                    'allocation': allocation,
                    'unrealized_pnl': unrealized_pnl,
                    'unrealized_pnl_pct': unrealized_pnl_pct
                })
        
        return {
            'timestamp': datetime.now(),
            'portfolio_value': portfolio_value,
            'cash_balance': self.cash_balance,
            'total_return': total_return,
            'num_positions': len(self.current_positions),
            'positions': position_summary,
            'risk_tolerance': self.risk_tolerance,
            'max_position_size': self.max_position_size
        }

# Example usage for live trading
def create_live_trading_example():
    """Example of how to set up and use the live trading system with real TFT model."""
    
    print("=== LIVE TRADING SYSTEM EXAMPLE ===")
    
    # Initialize real TFT model manager
    try:
        tft_model_manager = RealTFTModelManager()
        print("✅ Real TFT model loaded successfully")
    except Exception as e:
        print(f"❌ Error loading real TFT model: {e}")
        return
    
    # Initialize live trading system
    trading_system = LiveTradingSystem(
        tft_model_manager=tft_model_manager,
        initial_capital=100000,
        max_position_size=0.25,  # Max 25% per position
        risk_tolerance='MODERATE'
    )
    
    # Simulate receiving market data
    mock_price_data = pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=100, freq='D'),
        'open': np.random.randn(100).cumsum() + 100,
        'high': np.random.randn(100).cumsum() + 102,
        'low': np.random.randn(100).cumsum() + 98,
        'close': np.random.randn(100).cumsum() + 100,
        'volume': np.random.randint(1000000, 5000000, 100)
    })
    
    # Update market data
    symbols = ['AAPL', 'MSFT', 'GOOGL']
    for symbol in symbols:
        trading_system.update_market_data(symbol, mock_price_data)
    
    # Generate signals for multiple symbols
    symbols = ['AAPL', 'MSFT', 'GOOGL']
    signals = []
    for symbol in symbols:
        # Generate realistic feature data for the TFT model
        # Features should match the model's expected input format
        seq_len = 60  # 60 days of history
        input_size = 50  # Based on model architecture
        mock_features = torch.randn(1, seq_len, input_size).numpy()  # Convert to numpy
        current_price = mock_price_data['close'].iloc[-1]
        
        signal = trading_system.generate_live_signal(symbol, mock_features, current_price)
        signals.append(signal)
        
        print(f"\n📊 Signal for {symbol}:")
        print(f"   Type: {signal.signal_type}")
        print(f"   Position: {signal.recommended_position:.1%}")
        print(f"   Confidence: {signal.confidence:.2f}")
        print(f"   Risk: {signal.risk_level}")
        print(f"   Rationale: {signal.rationale}")
    
    # Generate portfolio recommendation
    portfolio_rec = trading_system.generate_portfolio_recommendation(symbols, signals)
    
    print(f"\n🎯 PORTFOLIO RECOMMENDATION:")
    print(f"   Total Value: ${portfolio_rec.total_portfolio_value:,.2f}")
    print(f"   Cash Allocation: {portfolio_rec.cash_allocation:.1%}")
    print(f"   Expected Return: {portfolio_rec.expected_return:.2%}")
    print(f"   Risk Score: {portfolio_rec.risk_score}/10")
    print(f"   Positions:")
    
    for symbol, allocation in portfolio_rec.positions.items():
        print(f"     {symbol}: {allocation:.1%}")
    
    print(f"\n💡 RECOMMENDATIONS:")
    for rec in portfolio_rec.recommendations:
        print(f"   • {rec}")
    
    # Show dashboard data
    dashboard = trading_system.get_live_dashboard_data()
    print(f"\n📈 DASHBOARD:")
    print(f"   Portfolio Value: ${dashboard['portfolio_value']:,.2f}")
    print(f"   Cash Balance: ${dashboard['cash_balance']:,.2f}")
    print(f"   Total Return: {dashboard['total_return']:.2%}")
    print(f"   Active Positions: {dashboard['num_positions']}")

if __name__ == "__main__":
    create_live_trading_example()
