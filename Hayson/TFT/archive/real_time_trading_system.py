"""
REAL-TIME TRADING SYSTEM

A comprehensive live trading system that provides:
- Real-time trade signals
- Portfolio recommendations
- Dynamic risk estimation
- Position management
- Live market data integration

This system demonstrates how the TFT model can be used for actual trading.
"""

import numpy as np
import pandas as pd
import torch
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
import json
import logging
from pathlib import Path
import threading
import time
from collections import deque
import warnings
warnings.filterwarnings('ignore')

# Import our real TFT model manager
from Hayson.TFT.real_tft_integration import RealTFTModelManager

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class LiveTradingSignal:
    """Enhanced trading signal with comprehensive information."""
    timestamp: datetime
    symbol: str
    signal_type: str  # 'BUY', 'SELL', 'HOLD'
    confidence: float  # 0-1
    predicted_return: float  # Expected return
    recommended_position: float  # 0-1 (fraction of portfolio)
    kelly_fraction: float  # Raw Kelly fraction before adjustments
    risk_level: str  # 'LOW', 'MEDIUM', 'HIGH'
    market_regime: str  # 'LOW_VOLATILITY', 'NORMAL', 'HIGH_VOLATILITY'
    current_price: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    risk_reward_ratio: float = 0.0
    volatility_estimate: float = 0.0
    rationale: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = asdict(self)
        result['timestamp'] = self.timestamp.isoformat()
        return result

@dataclass
class PortfolioRecommendation:
    """Complete portfolio recommendation with detailed analysis."""
    timestamp: datetime
    total_portfolio_value: float
    cash_allocation: float  # Fraction to keep in cash
    positions: Dict[str, Dict[str, float]]  # Symbol -> {allocation, shares, value}
    expected_return: float
    estimated_volatility: float
    max_drawdown_risk: float
    sharpe_ratio_estimate: float
    risk_score: float  # 0-10 scale
    diversification_score: float  # 0-10 scale
    recommendations: List[str]  # Human-readable advice
    alerts: List[str]  # Important warnings or notifications
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = asdict(self)
        result['timestamp'] = self.timestamp.isoformat()
        return result

class RealTimeTradingSystem:
    """
    Production-ready real-time trading system with comprehensive features.
    """
    
    def __init__(self, 
                 tft_model_manager: RealTFTModelManager,
                 initial_capital: float = 100000,
                 max_position_size: float = 0.25,  # Max 25% per position
                 risk_tolerance: str = 'MODERATE',  # LOW, MODERATE, HIGH
                 symbols: Optional[List[str]] = None,
                 update_frequency: int = 300):  # Update every 5 minutes
        
        self.tft_model_manager = tft_model_manager
        self.initial_capital = initial_capital
        self.current_portfolio_value = initial_capital
        self.max_position_size = max_position_size
        self.risk_tolerance = risk_tolerance
        self.symbols = symbols or ['AAPL', 'SPY', 'MSFT', 'GOOGL']
        self.update_frequency = update_frequency
        
        # Risk parameters
        self.risk_params = self._set_risk_parameters()
        
        # Portfolio state
        self.current_positions = {}  # Symbol -> {shares, avg_price, allocation, last_update}
        self.cash_balance = initial_capital
        self.trading_history = []
        self.performance_metrics = {}
        
        # Market state tracking
        self.market_data_buffer = {}  # Symbol -> DataFrame with recent data
        self.volatility_buffer = {}   # Symbol -> current volatility estimate
        self.accuracy_tracker = {}    # Symbol -> recent prediction accuracy
        self.signal_history = {}      # Symbol -> deque of recent signals
        
        # Live trading state
        self.is_running = False
        self.last_portfolio_update = datetime.now()
        self.emergency_stop = False
        
        # Performance tracking
        self.daily_returns = deque(maxlen=252)  # Store 1 year of daily returns
        self.portfolio_history = []
        
        # Initialize signal history
        for symbol in self.symbols:
            self.signal_history[symbol] = deque(maxlen=50)
        
        logger.info(f"Real-Time Trading System initialized")
        logger.info(f"Capital: ${initial_capital:,.2f}, Risk: {risk_tolerance}")
        logger.info(f"Symbols: {', '.join(self.symbols)}")
    
    def _set_risk_parameters(self) -> Dict[str, float]:
        """Set risk parameters based on user's risk tolerance."""
        params = {
            'LOW': {
                'kelly_multiplier': 0.15,      # Very conservative Kelly
                'max_portfolio_risk': 0.5,     # Max 50% invested
                'volatility_threshold': 0.015, # Lower vol threshold
                'min_confidence': 0.75,        # High confidence requirement
                'stop_loss_pct': 0.03,         # 3% stop loss
                'position_decay': 0.85,        # Fast position decay
                'max_daily_trades': 3,         # Limit trading frequency
                'correlation_threshold': 0.7   # Avoid correlated positions
            },
            'MODERATE': {
                'kelly_multiplier': 0.25,      # Standard Kelly multiplier
                'max_portfolio_risk': 0.75,    # Max 75% invested
                'volatility_threshold': 0.025, # Standard vol threshold
                'min_confidence': 0.60,        # Moderate confidence
                'stop_loss_pct': 0.05,         # 5% stop loss
                'position_decay': 0.90,        # Standard decay
                'max_daily_trades': 5,         # Moderate trading frequency
                'correlation_threshold': 0.8   # Standard correlation limit
            },
            'HIGH': {
                'kelly_multiplier': 0.35,      # Aggressive Kelly
                'max_portfolio_risk': 0.95,    # Max 95% invested
                'volatility_threshold': 0.04,  # Higher vol tolerance
                'min_confidence': 0.45,        # Lower confidence OK
                'stop_loss_pct': 0.08,         # 8% stop loss
                'position_decay': 0.95,        # Slower decay
                'max_daily_trades': 10,        # High trading frequency
                'correlation_threshold': 0.9   # Less strict correlation
            }
        }
        return params[self.risk_tolerance]
    
    def update_market_data(self, symbol: str, price_data: pd.DataFrame) -> None:
        """
        Update market data buffer for a symbol.
        
        Args:
            symbol: Stock symbol
            price_data: DataFrame with OHLCV data
        """
        try:
            # Store recent data (last 100 periods)
            self.market_data_buffer[symbol] = price_data.tail(100).copy()
            
            # Calculate volatility
            returns = price_data['close'].pct_change().dropna()
            if len(returns) >= 20:
                self.volatility_buffer[symbol] = returns.rolling(20).std().iloc[-1]
            else:
                self.volatility_buffer[symbol] = returns.std() if len(returns) > 1 else 0.02
            
            logger.debug(f"Updated market data for {symbol}")
            
        except Exception as e:
            logger.error(f"Error updating market data for {symbol}: {e}")
    
    def generate_real_time_signal(self, 
                                 symbol: str, 
                                 current_features: np.ndarray,
                                 current_price: float,
                                 market_data: Optional[pd.DataFrame] = None) -> LiveTradingSignal:
        """
        Generate comprehensive real-time trading signal.
        
        Args:
            symbol: Stock symbol
            current_features: Feature array for TFT model
            current_price: Current market price
            market_data: Optional recent market data
            
        Returns:
            LiveTradingSignal with complete analysis
        """
        timestamp = datetime.now()
        
        try:
            # Update market data if provided
            if market_data is not None:
                self.update_market_data(symbol, market_data)
            
            # Step 1: Get TFT model prediction
            predicted_return = self._get_model_prediction(current_features)
            
            # Step 2: Calculate prediction confidence
            confidence = self._calculate_confidence(symbol, predicted_return)
            
            # Step 3: Assess market regime
            market_regime = self._assess_market_regime(symbol)
            
            # Step 4: Calculate Kelly fraction
            kelly_fraction = self._calculate_kelly_fraction(predicted_return, confidence)
            
            # Step 5: Calculate recommended position size
            recommended_position = self._calculate_position_size(
                symbol, predicted_return, confidence, market_regime, kelly_fraction
            )
            
            # Step 6: Determine signal type
            signal_type = self._determine_signal_type(
                symbol, predicted_return, confidence, recommended_position
            )
            
            # Step 7: Risk assessment
            risk_level = self._assess_risk_level(symbol, recommended_position, market_regime)
            
            # Step 8: Calculate stop loss and take profit
            stop_loss, take_profit = self._calculate_stop_loss_take_profit(
                current_price, predicted_return, risk_level
            )
            
            # Step 9: Calculate risk-reward ratio
            risk_reward_ratio = self._calculate_risk_reward_ratio(
                current_price, take_profit, stop_loss
            )
            
            # Step 10: Generate rationale
            rationale = self._generate_detailed_rationale(
                symbol, predicted_return, confidence, market_regime, 
                recommended_position, kelly_fraction
            )
            
            # Create signal
            signal = LiveTradingSignal(
                timestamp=timestamp,
                symbol=symbol,
                signal_type=signal_type,
                confidence=confidence,
                predicted_return=predicted_return,
                recommended_position=recommended_position,
                kelly_fraction=kelly_fraction,
                risk_level=risk_level,
                market_regime=market_regime,
                current_price=current_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                risk_reward_ratio=risk_reward_ratio,
                volatility_estimate=self.volatility_buffer.get(symbol, 0.02),
                rationale=rationale
            )
            
            # Store signal in history
            self.signal_history[symbol].append(signal)
            
            # Log signal
            logger.info(f"Signal for {symbol}: {signal_type} {recommended_position:.1%} "
                       f"(confidence: {confidence:.2f}, kelly: {kelly_fraction:.3f})")
            
            return signal
            
        except Exception as e:
            logger.error(f"Error generating signal for {symbol}: {e}")
            return self._create_error_signal(symbol, current_price, str(e))
    
    def _get_model_prediction(self, features: np.ndarray) -> float:
        """Get prediction from TFT model."""
        with torch.no_grad():
            if isinstance(features, np.ndarray):
                features_tensor = torch.tensor(features, dtype=torch.float32)
            else:
                features_tensor = features
            
            if features_tensor.dim() == 2:
                features_tensor = features_tensor.unsqueeze(0)
            
            trading_signals = self.tft_model_manager.generate_trading_signals(features_tensor)
            return trading_signals['predicted_returns']
    
    def _calculate_confidence(self, symbol: str, predicted_return: float) -> float:
        """Calculate prediction confidence with multiple factors."""
        # Base confidence from prediction magnitude
        base_confidence = min(abs(predicted_return) / 0.08, 1.0)
        
        # Model accuracy adjustment
        accuracy_mult = self.accuracy_tracker.get(symbol, 0.7)
        
        # Recent signal consistency
        consistency_mult = 1.0
        if symbol in self.signal_history and len(self.signal_history[symbol]) > 3:
            recent_signals = list(self.signal_history[symbol])[-3:]
            recent_returns = [s.predicted_return for s in recent_signals]
            if len(set(np.sign(recent_returns))) == 1:  # All same direction
                consistency_mult = 1.2
        
        # Market regime adjustment
        regime_mult = {
            'LOW_VOLATILITY': 1.1,
            'NORMAL': 1.0,
            'HIGH_VOLATILITY': 0.8,
            'UNKNOWN': 0.6
        }
        
        market_regime = self._assess_market_regime(symbol)
        final_confidence = base_confidence * accuracy_mult * consistency_mult * regime_mult.get(market_regime, 1.0)
        
        return np.clip(final_confidence, 0.0, 1.0)
    
    def _assess_market_regime(self, symbol: str) -> str:
        """Assess current market regime."""
        if symbol not in self.volatility_buffer:
            return 'UNKNOWN'
        
        current_vol = self.volatility_buffer[symbol]
        vol_threshold = self.risk_params['volatility_threshold']
        
        if current_vol > vol_threshold * 1.8:
            return 'HIGH_VOLATILITY'
        elif current_vol < vol_threshold * 0.6:
            return 'LOW_VOLATILITY'
        else:
            return 'NORMAL'
    
    def _calculate_kelly_fraction(self, predicted_return: float, confidence: float) -> float:
        """Calculate Kelly fraction with enhanced logic."""
        # Estimate win probability
        base_prob = 0.51  # Slightly better than coin flip
        confidence_boost = confidence * 0.20  # Max 20% boost from confidence
        win_prob = np.clip(base_prob + confidence_boost, 0.45, 0.80)
        
        # Kelly fraction: f = 2p - 1 (for equal win/loss amounts)
        kelly_fraction = 2 * win_prob - 1
        
        # Adjust for expected return magnitude
        return_adjustment = min(abs(predicted_return) / 0.05, 2.0)
        kelly_fraction *= return_adjustment
        
        return np.clip(kelly_fraction, 0.0, 0.5)  # Cap at 50%
    
    def _calculate_position_size(self, symbol: str, predicted_return: float, 
                               confidence: float, market_regime: str, 
                               kelly_fraction: float) -> float:
        """Calculate optimal position size."""
        # Start with Kelly fraction
        base_position = kelly_fraction * self.risk_params['kelly_multiplier']
        
        # Market regime adjustment
        regime_multipliers = {
            'HIGH_VOLATILITY': 0.6,
            'NORMAL': 1.0,
            'LOW_VOLATILITY': 1.3,
            'UNKNOWN': 0.4
        }
        
        adjusted_position = base_position * regime_multipliers[market_regime]
        
        # Apply position limits
        final_position = np.clip(adjusted_position, 0.0, self.max_position_size)
        
        # Confidence threshold check
        if confidence < self.risk_params['min_confidence']:
            final_position *= 0.6
        
        # Portfolio diversification check
        total_allocated = sum(pos.get('allocation', 0) for pos in self.current_positions.values())
        if total_allocated > self.risk_params['max_portfolio_risk']:
            final_position *= 0.5
        
        return final_position
    
    def _determine_signal_type(self, symbol: str, predicted_return: float, 
                             confidence: float, recommended_position: float) -> str:
        """Determine signal type with enhanced logic."""
        min_position = 0.02  # 2% minimum
        
        # Check if we already have a position
        current_allocation = 0
        if symbol in self.current_positions:
            current_allocation = self.current_positions[symbol].get('allocation', 0)
        
        if recommended_position < min_position:
            return 'HOLD' if current_allocation == 0 else 'SELL'
        
        if predicted_return > 0:
            if current_allocation < recommended_position * 0.8:
                return 'BUY'
            else:
                return 'HOLD'
        else:
            if current_allocation > 0:
                return 'SELL'
            else:
                return 'HOLD'
    
    def _assess_risk_level(self, symbol: str, position_size: float, 
                          market_regime: str) -> str:
        """Assess overall risk level."""
        risk_score = 0
        
        # Position size risk
        if position_size > 0.15:
            risk_score += 2
        elif position_size > 0.08:
            risk_score += 1
        
        # Market regime risk
        regime_risk = {
            'HIGH_VOLATILITY': 3,
            'NORMAL': 1,
            'LOW_VOLATILITY': 0,
            'UNKNOWN': 2
        }
        risk_score += regime_risk[market_regime]
        
        # Volatility risk
        if symbol in self.volatility_buffer:
            if self.volatility_buffer[symbol] > 0.04:
                risk_score += 2
            elif self.volatility_buffer[symbol] > 0.025:
                risk_score += 1
        
        # Portfolio concentration risk
        total_positions = len([pos for pos in self.current_positions.values() 
                             if pos.get('allocation', 0) > 0.01])
        if total_positions < 3:
            risk_score += 1
        
        if risk_score >= 5:
            return 'HIGH'
        elif risk_score >= 3:
            return 'MEDIUM'
        else:
            return 'LOW'
    
    def _calculate_stop_loss_take_profit(self, current_price: float, 
                                       predicted_return: float, 
                                       risk_level: str) -> Tuple[float, float]:
        """Calculate stop loss and take profit levels."""
        stop_loss_pct = self.risk_params['stop_loss_pct']
        
        # Adjust stop loss for risk level
        risk_multipliers = {'LOW': 0.8, 'MEDIUM': 1.0, 'HIGH': 1.5}
        adjusted_stop_loss_pct = stop_loss_pct * risk_multipliers[risk_level]
        
        if predicted_return > 0:
            stop_loss = current_price * (1 - adjusted_stop_loss_pct)
            take_profit = current_price * (1 + abs(predicted_return) * 0.8)
        else:
            stop_loss = current_price * (1 + adjusted_stop_loss_pct)
            take_profit = current_price * (1 - abs(predicted_return) * 0.8)
        
        return stop_loss, take_profit
    
    def _calculate_risk_reward_ratio(self, current_price: float, 
                                   take_profit: Optional[float], 
                                   stop_loss: Optional[float]) -> float:
        """Calculate risk-reward ratio."""
        if take_profit is None or stop_loss is None:
            return 0.0
        
        potential_gain = abs(take_profit - current_price)
        potential_loss = abs(stop_loss - current_price)
        
        if potential_loss == 0:
            return float('inf')
        
        return potential_gain / potential_loss
    
    def _generate_detailed_rationale(self, symbol: str, predicted_return: float, 
                                   confidence: float, market_regime: str, 
                                   recommended_position: float, kelly_fraction: float) -> str:
        """Generate detailed rationale for the signal."""
        rationale_parts = []
        
        # Model prediction
        return_pct = predicted_return * 100
        rationale_parts.append(f"TFT model predicts {return_pct:+.2f}% return")
        
        # Confidence assessment
        conf_pct = confidence * 100
        rationale_parts.append(f"Confidence: {conf_pct:.1f}%")
        
        # Kelly analysis
        kelly_pct = kelly_fraction * 100
        rationale_parts.append(f"Kelly suggests {kelly_pct:.1f}% position")
        
        # Market regime
        rationale_parts.append(f"Market regime: {market_regime.lower().replace('_', ' ')}")
        
        # Position recommendation
        pos_pct = recommended_position * 100
        rationale_parts.append(f"Recommended position: {pos_pct:.1f}%")
        
        # Risk assessment
        if symbol in self.volatility_buffer:
            vol_pct = self.volatility_buffer[symbol] * 100
            rationale_parts.append(f"Current volatility: {vol_pct:.1f}%")
        
        return " | ".join(rationale_parts)
    
    def _create_error_signal(self, symbol: str, current_price: float, error_msg: str) -> LiveTradingSignal:
        """Create a safe error signal."""
        return LiveTradingSignal(
            timestamp=datetime.now(),
            symbol=symbol,
            signal_type='HOLD',
            confidence=0.0,
            predicted_return=0.0,
            recommended_position=0.0,
            kelly_fraction=0.0,
            risk_level='HIGH',
            market_regime='UNKNOWN',
            current_price=current_price,
            rationale=f"Error: {error_msg}"
        )
    
    def generate_portfolio_recommendation(self, 
                                        signals: List[LiveTradingSignal]) -> PortfolioRecommendation:
        """
        Generate comprehensive portfolio recommendation based on current signals.
        
        Args:
            signals: List of current trading signals
            
        Returns:
            PortfolioRecommendation with complete analysis
        """
        timestamp = datetime.now()
        
        try:
            # Calculate portfolio metrics
            portfolio_metrics = self._calculate_portfolio_metrics(signals)
            
            # Generate position recommendations
            position_recommendations = self._generate_position_recommendations(signals)
            
            # Calculate cash allocation
            total_recommended = sum(pos['allocation'] for pos in position_recommendations.values())
            cash_allocation = max(0.05, 1.0 - total_recommended)  # Min 5% cash
            
            # Risk assessment
            risk_score = self._calculate_portfolio_risk_score(signals, position_recommendations)
            diversification_score = self._calculate_diversification_score(position_recommendations)
            
            # Generate recommendations and alerts
            recommendations = self._generate_portfolio_recommendations(
                portfolio_metrics, risk_score, diversification_score
            )
            alerts = self._generate_portfolio_alerts(signals, risk_score)
            
            return PortfolioRecommendation(
                timestamp=timestamp,
                total_portfolio_value=self.current_portfolio_value,
                cash_allocation=cash_allocation,
                positions=position_recommendations,
                expected_return=portfolio_metrics['expected_return'],
                estimated_volatility=portfolio_metrics['estimated_volatility'],
                max_drawdown_risk=portfolio_metrics['max_drawdown_risk'],
                sharpe_ratio_estimate=portfolio_metrics['sharpe_ratio_estimate'],
                risk_score=risk_score,
                diversification_score=diversification_score,
                recommendations=recommendations,
                alerts=alerts
            )
            
        except Exception as e:
            logger.error(f"Error generating portfolio recommendation: {e}")
            return self._create_error_portfolio_recommendation(str(e))
    
    def _calculate_portfolio_metrics(self, signals: List[LiveTradingSignal]) -> Dict[str, float]:
        """Calculate portfolio-level metrics."""
        if not signals:
            return {
                'expected_return': 0.0,
                'estimated_volatility': 0.02,
                'max_drawdown_risk': 0.15,
                'sharpe_ratio_estimate': 0.0
            }
        
        # Weight by recommended position
        weights = np.array([s.recommended_position for s in signals])
        returns = np.array([s.predicted_return for s in signals])
        volatilities = np.array([s.volatility_estimate for s in signals])
        
        # Normalize weights
        if weights.sum() > 0:
            weights = weights / weights.sum()
        
        # Portfolio expected return
        expected_return = np.sum(weights * returns)
        
        # Portfolio volatility (simplified - assumes some correlation)
        portfolio_volatility = np.sqrt(np.sum((weights * volatilities) ** 2)) * 1.2
        
        # Sharpe ratio estimate (assume 2% risk-free rate)
        sharpe_ratio = (expected_return - 0.02) / portfolio_volatility if portfolio_volatility > 0 else 0
        
        # Max drawdown risk estimate
        max_drawdown_risk = portfolio_volatility * 2.5  # Conservative estimate
        
        return {
            'expected_return': float(expected_return),
            'estimated_volatility': float(portfolio_volatility),
            'max_drawdown_risk': float(max_drawdown_risk),
            'sharpe_ratio_estimate': float(sharpe_ratio)
        }
    
    def _generate_position_recommendations(self, signals: List[LiveTradingSignal]) -> Dict[str, Dict[str, float]]:
        """Generate detailed position recommendations."""
        recommendations = {}
        
        for signal in signals:
            if signal.recommended_position > 0.01:  # Only include meaningful positions
                position_value = self.current_portfolio_value * signal.recommended_position
                shares = position_value / signal.current_price
                
                recommendations[signal.symbol] = {
                    'allocation': signal.recommended_position,
                    'shares': shares,
                    'value': position_value,
                    'price': signal.current_price,
                    'confidence': signal.confidence,
                    'risk_level': signal.risk_level
                }
        
        return recommendations
    
    def _calculate_portfolio_risk_score(self, signals: List[LiveTradingSignal], 
                                      positions: Dict[str, Dict[str, float]]) -> float:
        """Calculate overall portfolio risk score (0-10)."""
        risk_score = 0
        
        # Position concentration risk
        max_position = max([pos['allocation'] for pos in positions.values()], default=0)
        if max_position > 0.3:
            risk_score += 3
        elif max_position > 0.2:
            risk_score += 2
        elif max_position > 0.15:
            risk_score += 1
        
        # High volatility exposure
        high_vol_exposure = sum(pos['allocation'] for signal, pos in zip(signals, positions.values()) 
                              if signal.market_regime == 'HIGH_VOLATILITY')
        if high_vol_exposure > 0.4:
            risk_score += 2
        elif high_vol_exposure > 0.2:
            risk_score += 1
        
        # Low confidence positions
        low_conf_exposure = sum(pos['allocation'] for signal, pos in zip(signals, positions.values()) 
                              if signal.confidence < 0.5)
        if low_conf_exposure > 0.3:
            risk_score += 2
        elif low_conf_exposure > 0.15:
            risk_score += 1
        
        # Diversification
        num_positions = len(positions)
        if num_positions < 3:
            risk_score += 2
        elif num_positions < 5:
            risk_score += 1
        
        return min(risk_score, 10)
    
    def _calculate_diversification_score(self, positions: Dict[str, Dict[str, float]]) -> float:
        """Calculate diversification score (0-10)."""
        if not positions:
            return 0
        
        # Number of positions
        num_positions = len(positions)
        position_score = min(num_positions * 1.5, 6)  # Max 6 points for positions
        
        # Allocation distribution
        allocations = [pos['allocation'] for pos in positions.values()]
        max_allocation = max(allocations)
        
        # Penalize concentration
        if max_allocation > 0.4:
            concentration_penalty = 3
        elif max_allocation > 0.3:
            concentration_penalty = 2
        elif max_allocation > 0.2:
            concentration_penalty = 1
        else:
            concentration_penalty = 0
        
        # Balance bonus
        if num_positions > 1:
            allocation_std = np.std(allocations)
            balance_bonus = max(0, 2 - allocation_std * 10)
        else:
            balance_bonus = 0
        
        total_score = position_score - concentration_penalty + balance_bonus
        return float(max(0, min(total_score, 10)))
    
    def _generate_portfolio_recommendations(self, metrics: Dict[str, float], 
                                          risk_score: float, 
                                          diversification_score: float) -> List[str]:
        """Generate human-readable portfolio recommendations."""
        recommendations = []
        
        # Risk assessment
        if risk_score > 7:
            recommendations.append("⚠️ High risk portfolio - consider reducing position sizes")
        elif risk_score > 5:
            recommendations.append("⚡ Moderate risk - monitor positions closely")
        else:
            recommendations.append("✅ Low risk profile - well-balanced portfolio")
        
        # Diversification
        if diversification_score < 4:
            recommendations.append("📊 Poor diversification - consider adding more positions")
        elif diversification_score < 7:
            recommendations.append("📈 Moderate diversification - could be improved")
        else:
            recommendations.append("🎯 Good diversification - well-spread risk")
        
        # Expected return
        if metrics['expected_return'] > 0.15:
            recommendations.append("🚀 High expected return - verify risk tolerance")
        elif metrics['expected_return'] > 0.08:
            recommendations.append("📈 Solid expected return - good risk-reward balance")
        elif metrics['expected_return'] > 0:
            recommendations.append("💼 Conservative expected return - stable approach")
        else:
            recommendations.append("📉 Negative expected return - consider defensive positions")
        
        # Volatility
        if metrics['estimated_volatility'] > 0.25:
            recommendations.append("🌊 High volatility - expect significant price swings")
        elif metrics['estimated_volatility'] > 0.15:
            recommendations.append("📊 Moderate volatility - normal market fluctuations")
        else:
            recommendations.append("😌 Low volatility - stable price movements expected")
        
        # Sharpe ratio
        if metrics['sharpe_ratio_estimate'] > 1.5:
            recommendations.append("⭐ Excellent risk-adjusted returns")
        elif metrics['sharpe_ratio_estimate'] > 1.0:
            recommendations.append("✨ Good risk-adjusted returns")
        elif metrics['sharpe_ratio_estimate'] > 0.5:
            recommendations.append("📊 Acceptable risk-adjusted returns")
        else:
            recommendations.append("⚠️ Poor risk-adjusted returns - review strategy")
        
        return recommendations
    
    def _generate_portfolio_alerts(self, signals: List[LiveTradingSignal], 
                                 risk_score: float) -> List[str]:
        """Generate important alerts and warnings."""
        alerts = []
        
        # High risk positions
        high_risk_positions = [s.symbol for s in signals if s.risk_level == 'HIGH' and s.recommended_position > 0.1]
        if high_risk_positions:
            alerts.append(f"🚨 High risk exposure in: {', '.join(high_risk_positions)}")
        
        # Low confidence signals
        low_conf_signals = [s.symbol for s in signals if s.confidence < 0.4 and s.recommended_position > 0.05]
        if low_conf_signals:
            alerts.append(f"⚠️ Low confidence signals: {', '.join(low_conf_signals)}")
        
        # High volatility regime
        high_vol_symbols = [s.symbol for s in signals if s.market_regime == 'HIGH_VOLATILITY']
        if len(high_vol_symbols) > len(signals) * 0.5:
            alerts.append("🌪️ High volatility regime detected - increase monitoring")
        
        # Portfolio risk
        if risk_score > 8:
            alerts.append("🚨 CRITICAL: Portfolio risk extremely high - immediate review recommended")
        elif risk_score > 6:
            alerts.append("⚠️ WARNING: Portfolio risk elevated - consider risk reduction")
        
        # Emergency stop
        if self.emergency_stop:
            alerts.append("🛑 EMERGENCY STOP ACTIVATED - All trading suspended")
        
        return alerts
    
    def _create_error_portfolio_recommendation(self, error_msg: str) -> PortfolioRecommendation:
        """Create a safe error portfolio recommendation."""
        return PortfolioRecommendation(
            timestamp=datetime.now(),
            total_portfolio_value=self.current_portfolio_value,
            cash_allocation=1.0,  # 100% cash on error
            positions={},
            expected_return=0.0,
            estimated_volatility=0.0,
            max_drawdown_risk=0.0,
            sharpe_ratio_estimate=0.0,
            risk_score=10.0,  # Maximum risk
            diversification_score=0.0,
            recommendations=[f"❌ Error in analysis: {error_msg}"],
            alerts=["🚨 System Error - Manual review required"]
        )
    
    def update_accuracy_tracker(self, symbol: str, predicted_return: float, 
                              actual_return: float) -> None:
        """Update the accuracy tracker for a symbol."""
        if symbol not in self.accuracy_tracker:
            self.accuracy_tracker[symbol] = 0.7  # Default accuracy
        
        # Simple accuracy calculation
        correct = (predicted_return > 0) == (actual_return > 0)
        current_accuracy = self.accuracy_tracker[symbol]
        
        # Exponential moving average
        self.accuracy_tracker[symbol] = current_accuracy * 0.9 + (1.0 if correct else 0.0) * 0.1
        
        logger.debug(f"Updated accuracy for {symbol}: {self.accuracy_tracker[symbol]:.3f}")
    
    def get_portfolio_status(self) -> Dict[str, Any]:
        """Get current portfolio status."""
        return {
            'timestamp': datetime.now().isoformat(),
            'portfolio_value': self.current_portfolio_value,
            'cash_balance': self.cash_balance,
            'positions': self.current_positions,
            'is_running': self.is_running,
            'emergency_stop': self.emergency_stop,
            'risk_tolerance': self.risk_tolerance,
            'symbols_tracked': self.symbols,
            'last_update': self.last_portfolio_update.isoformat()
        }
    
    def save_trading_log(self, filepath: str) -> None:
        """Save trading history and performance to file."""
        log_data = {
            'system_config': {
                'initial_capital': self.initial_capital,
                'risk_tolerance': self.risk_tolerance,
                'symbols': self.symbols,
                'max_position_size': self.max_position_size
            },
            'portfolio_status': self.get_portfolio_status(),
            'trading_history': self.trading_history,
            'performance_metrics': self.performance_metrics,
            'signal_history': {
                symbol: [s.to_dict() for s in signals] 
                for symbol, signals in self.signal_history.items()
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(log_data, f, indent=2, default=str)
        
        logger.info(f"Trading log saved to {filepath}")

def demonstrate_real_time_system():
    """Demonstrate the real-time trading system with real TFT model."""
    
    print("🚀 Real-Time Trading System Demonstration")
    print("=" * 50)
    
    # Initialize real TFT model manager
    try:
        tft_model_manager = RealTFTModelManager()
        print("✅ Real TFT model loaded successfully")
    except Exception as e:
        print(f"❌ Error loading real TFT model: {e}")
        return
    
    # Initialize system
    system = RealTimeTradingSystem(
        tft_model_manager=tft_model_manager,
        initial_capital=100000,
        risk_tolerance='MODERATE',
        symbols=['AAPL', 'MSFT', 'GOOGL', 'SPY']
    )
    
    print(f"✅ System initialized with ${system.initial_capital:,.2f}")
    print(f"📊 Risk tolerance: {system.risk_tolerance}")
    print(f"📈 Tracking symbols: {', '.join(system.symbols)}")
    print()
    
    # Generate mock signals
    signals = []
    for symbol in system.symbols:
        # Mock market data
        dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
        prices = 100 + np.cumsum(np.random.normal(0, 2, 100))
        market_data = pd.DataFrame({
            'timestamp': dates,
            'open': prices * 0.99,
            'high': prices * 1.02,
            'low': prices * 0.98,
            'close': prices,
            'volume': np.random.randint(1000000, 5000000, 100)
        })
        
        # Update market data
        system.update_market_data(symbol, market_data)
        
        # Generate signal
        # Create realistic features for TFT model (batch_size=1, seq_len=60, features=50)
        mock_features = torch.randn(1, 60, 50).numpy()  # Convert to numpy for interface
        current_price = prices[-1]
        
        signal = system.generate_real_time_signal(
            symbol=symbol,
            current_features=mock_features,
            current_price=current_price
        )
        signals.append(signal)
        
        print(f"📡 {symbol} Signal:")
        print(f"   Signal Type: {signal.signal_type}")
        print(f"   Confidence: {signal.confidence:.2f}")
        print(f"   Predicted Return: {signal.predicted_return:+.2%}")
        print(f"   Recommended Position: {signal.recommended_position:.1%}")
        print(f"   Risk Level: {signal.risk_level}")
        print(f"   Market Regime: {signal.market_regime}")
        print(f"   Kelly Fraction: {signal.kelly_fraction:.3f}")
        print(f"   Rationale: {signal.rationale}")
        print()
    
    # Generate portfolio recommendation
    portfolio_rec = system.generate_portfolio_recommendation(signals)
    
    print("📋 Portfolio Recommendation:")
    print(f"   Total Value: ${portfolio_rec.total_portfolio_value:,.2f}")
    print(f"   Cash Allocation: {portfolio_rec.cash_allocation:.1%}")
    print(f"   Expected Return: {portfolio_rec.expected_return:+.2%}")
    print(f"   Estimated Volatility: {portfolio_rec.estimated_volatility:.2%}")
    print(f"   Risk Score: {portfolio_rec.risk_score:.1f}/10")
    print(f"   Diversification Score: {portfolio_rec.diversification_score:.1f}/10")
    print()
    
    print("💡 Recommendations:")
    for rec in portfolio_rec.recommendations:
        print(f"   {rec}")
    print()
    
    if portfolio_rec.alerts:
        print("⚠️ Alerts:")
        for alert in portfolio_rec.alerts:
            print(f"   {alert}")
        print()
    
    print("📊 Position Breakdown:")
    for symbol, position in portfolio_rec.positions.items():
        print(f"   {symbol}: {position['allocation']:.1%} "
              f"(${position['value']:,.2f}, {position['shares']:.0f} shares)")
    print()
    
    # Save example log
    log_path = "/Users/haysoncheung/programs/pythonProject/Multi-Modal-Stock-TFT/Hayson/TFT/trading_log_example.json"
    system.save_trading_log(log_path)
    print(f"📝 Trading log saved to: {log_path}")

if __name__ == "__main__":
    demonstrate_real_time_system()
