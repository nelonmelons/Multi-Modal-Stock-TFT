"""
Trading Simulation Module

This module contains the trading simulation logic extracted from the TFT pipeline.
It provides classes and functions for backtesting trading strategies against buy-and-hold benchmarks.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, Any
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

class TradingSimulator:
    """Advanced trading strategy simulator with sophisticated risk management."""
    
    def __init__(self, initial_capital: float = 10000, dca_frequency: int = 1):
        """
        Initialize the trading simulator.
        
        Args:
            initial_capital: Initial capital for strategies
            dca_frequency: DCA frequency in days (1 = daily, 7 = weekly, etc.)
        """
        self.initial_capital = initial_capital
        self.dca_frequency = dca_frequency
        self.results = {}
        
    def run_simulation(self, predictions: np.ndarray, actual_returns: np.ndarray) -> Dict[str, Any]:
        """
        Run comprehensive trading simulation.
        
        Args:
            predictions: Model predictions (percentage returns)
            actual_returns: Actual market returns (percentage returns)
            
        Returns:
            Dictionary containing simulation results
        """
        # Flatten arrays
        pred_flat = predictions.flatten()
        target_flat = actual_returns.flatten()
        
        # Ensure we have enough data
        if len(target_flat) < 2:
            raise ValueError("Not enough data for trading simulation")
        
        # Cap returns to realistic daily ranges (±5% per day maximum)
        max_daily_return = 0.05
        pred_returns = np.clip(pred_flat, -max_daily_return, max_daily_return)
        actual_returns_clipped = np.clip(target_flat, -max_daily_return, max_daily_return)
        
        print(f"Simulation Info:")
        print(f"  Data points: {len(actual_returns_clipped)}")
        print(f"  Actual returns - Min: {np.min(actual_returns_clipped):.4f}, Max: {np.max(actual_returns_clipped):.4f}, Mean: {np.mean(actual_returns_clipped):.4f}")
        print(f"  Predicted returns - Min: {np.min(pred_returns):.4f}, Max: {np.max(pred_returns):.4f}, Mean: {np.mean(pred_returns):.4f}")
        
        # Calculate directional accuracy
        pred_direction = np.sign(pred_returns)
        actual_direction = np.sign(actual_returns_clipped)
        direction_correctness = (pred_direction == actual_direction).astype(float)
        
        # Advanced risk assessment
        risk_metrics = self._calculate_risk_metrics(pred_returns, actual_returns_clipped)
        
        # Enhanced portfolio allocation
        position_fractions = self._calculate_position_sizing(
            pred_returns, actual_returns_clipped, risk_metrics
        )
        
        # Calculate strategy returns
        strategy_returns = position_fractions * actual_returns_clipped * np.sign(pred_returns)
        
        # Run portfolio simulations
        portfolio_results = self._run_portfolio_simulations(
            strategy_returns, actual_returns_clipped
        )
        
        # Calculate performance metrics
        performance_metrics = self._calculate_performance_metrics(
            portfolio_results, strategy_returns, actual_returns_clipped
        )
        
        # Store results
        self.results = {
            'pred_returns': pred_returns,
            'actual_returns': actual_returns_clipped,
            'direction_correctness': direction_correctness,
            'risk_metrics': risk_metrics,
            'position_fractions': position_fractions,
            'strategy_returns': strategy_returns,
            'portfolio_results': portfolio_results,
            'performance_metrics': performance_metrics,
            'trading_signals': self._calculate_trading_signals(pred_returns, position_fractions)
        }
        
        return self.results
    
    def _calculate_risk_metrics(self, pred_returns: np.ndarray, actual_returns: np.ndarray) -> Dict[str, np.ndarray]:
        """Calculate advanced risk assessment metrics."""
        
        # 1. Prediction confidence based on magnitude
        prediction_confidence = np.abs(pred_returns)
        
        # 2. Rolling volatility for dynamic risk adjustment
        window = min(20, len(actual_returns) // 4)  # 20-day or 1/4 of data
        rolling_volatility = np.array([
            np.std(actual_returns[max(0, i-window):i+1]) if i >= window//2 
            else np.std(actual_returns[:window]) 
            for i in range(len(actual_returns))
        ])
        
        # 3. Market regime detection using volatility
        median_vol = np.median(rolling_volatility)
        high_vol_threshold = median_vol * 1.5
        low_vol_threshold = median_vol * 0.7
        
        market_regime = np.where(rolling_volatility > high_vol_threshold, 'high_vol',
                                np.where(rolling_volatility < low_vol_threshold, 'low_vol', 'normal'))
        
        # 4. Prediction accuracy analysis
        if len(pred_returns) > 10:
            prediction_errors = np.abs(pred_returns - actual_returns)
            rolling_pred_accuracy = np.array([
                1 - np.mean(prediction_errors[max(0, i-10):i+1]) if i >= 5
                else 1 - np.mean(prediction_errors[:10])
                for i in range(len(prediction_errors))
            ])
            rolling_pred_accuracy = np.clip(rolling_pred_accuracy, 0.1, 0.9)
        else:
            rolling_pred_accuracy = np.full(len(pred_returns), 0.55)
        
        # Normalize confidence to create risk scores (0 to 1)
        if np.max(prediction_confidence) > 0:
            risk_scores = prediction_confidence / np.max(prediction_confidence)
        else:
            risk_scores = np.zeros_like(prediction_confidence)
        
        return {
            'prediction_confidence': prediction_confidence,
            'rolling_volatility': rolling_volatility,
            'market_regime': market_regime,
            'rolling_pred_accuracy': rolling_pred_accuracy,
            'risk_scores': risk_scores
        }
    
    def _calculate_position_sizing(self, pred_returns: np.ndarray, actual_returns: np.ndarray, 
                                 risk_metrics: Dict[str, np.ndarray]) -> np.ndarray:
        """Calculate enhanced portfolio allocation using Kelly Criterion with risk adjustments."""
        
        # Make Kelly more responsive to actual market conditions
        # Estimate win probability from multiple factors with higher variability
        base_win_prob = 0.52  # Slightly better than random
        
        # More responsive confidence boost based on prediction magnitude
        prediction_magnitude = np.abs(pred_returns)
        confidence_boost = np.clip(prediction_magnitude * 5, 0, 0.25)  # Up to 25% boost for strong predictions
        
        # Historical accuracy adjustment with more impact
        accuracy_boost = (risk_metrics['rolling_pred_accuracy'] - 0.5) * 0.3  # Increased from 0.2 to 0.3
        
        # Market momentum factor - if market is trending, increase confidence
        market_momentum = np.sign(pred_returns) * prediction_magnitude
        momentum_boost = np.clip(market_momentum * 2, -0.15, 0.15)  # Can reduce or increase confidence
        
        estimated_win_prob = base_win_prob + confidence_boost + accuracy_boost + momentum_boost
        estimated_win_prob = np.clip(estimated_win_prob, 0.35, 0.80)  # Wider bounds for more variability
        
        # Enhanced Kelly Criterion that's more responsive
        kelly_fractions = 2 * estimated_win_prob - 1
        
        # Apply directional bias - if prediction is strong, increase allocation
        directional_multiplier = 1 + prediction_magnitude * 2  # Can go up to 3x for very strong predictions
        kelly_fractions = kelly_fractions * directional_multiplier
        
        kelly_fractions = np.clip(kelly_fractions, 0, 2)  # Allow higher leverage in strong predictions
        
        # Market regime adjustment with more impact
        regime_multiplier = np.where(risk_metrics['market_regime'] == 'high_vol', 0.3,      # More conservative in high vol
                                   np.where(risk_metrics['market_regime'] == 'low_vol', 1.5,  # More aggressive in low vol
                                           1.0))                               # Normal in normal volatility
        
        # More responsive conservative factor based on rolling performance
        base_conservative_factor = 0.4  # Increased base from 0.25
        performance_factor = np.clip(risk_metrics['rolling_pred_accuracy'], 0.3, 1.2)  # Scale based on accuracy
        conservative_factor = base_conservative_factor * performance_factor
        
        # Volatility adjustment that's more responsive
        volatility_adjustment = np.clip(1 / (1 + risk_metrics['rolling_volatility'] * 5), 0.1, 1.5)
        
        position_fractions = kelly_fractions * conservative_factor * regime_multiplier * volatility_adjustment
        
        # Dynamic position limits based on market conditions
        min_position = 0.0   # Can hold cash
        max_position = np.where(risk_metrics['market_regime'] == 'low_vol', 0.8,  # Higher max in low vol
                               np.where(risk_metrics['market_regime'] == 'high_vol', 0.3,  # Lower max in high vol
                                       0.6))                                        # Normal max
        
        position_fractions = np.clip(position_fractions, min_position, max_position)
        
        return position_fractions
    
    def _run_portfolio_simulations(self, strategy_returns: np.ndarray, 
                                 actual_returns: np.ndarray) -> Dict[str, Any]:
        """Run portfolio simulations for different strategies."""
        
        # Portfolio value limits
        max_portfolio_value = self.initial_capital * 500  # Emergency brake
        min_portfolio_value = self.initial_capital * 0.01  # Emergency brake
        
        # 1. Enhanced TFT Strategy
        portfolio_values = [self.initial_capital]
        for i in range(len(strategy_returns)):
            new_value = portfolio_values[-1] * (1 + strategy_returns[i])
            new_value = np.clip(new_value, min_portfolio_value, max_portfolio_value)
            portfolio_values.append(new_value)
        portfolio_values = np.array(portfolio_values)
        
        # 2. Traditional lump-sum buy & hold
        lump_sum_values = [self.initial_capital]
        for i in range(len(actual_returns)):
            new_value = lump_sum_values[-1] * (1 + actual_returns[i])
            new_value = np.clip(new_value, min_portfolio_value, max_portfolio_value)
            lump_sum_values.append(new_value)
        lump_sum_values = np.array(lump_sum_values)
        
        # 3. CORRECTED DCA Buy & Hold
        dca_values, dca_stats = self._calculate_dca_portfolio(actual_returns)
        
        print(f"Portfolio Simulation Results:")
        print(f"  Strategy final value: ${portfolio_values[-1]:,.2f}")
        print(f"  Lump-sum final value: ${lump_sum_values[-1]:,.2f}")
        print(f"  DCA final value: ${dca_values[-1]:,.2f}")
        print(f"  DCA total invested: ${dca_stats['total_invested']:,.2f}")
        print(f"  DCA shares owned: {dca_stats['total_shares']:.4f}")
        print(f"  DCA final stock price: ${dca_stats['final_stock_price']:.2f}")
        
        return {
            'strategy_values': portfolio_values,
            'lump_sum_values': lump_sum_values,
            'dca_values': dca_values,
            'dca_stats': dca_stats
        }
    
    def _calculate_dca_portfolio(self, actual_returns: np.ndarray) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Calculate DCA portfolio with CORRECTED logic.
        
        The key insight: For fair comparison, we need to invest the same total amount
        but spread it over time for DCA vs all at once for lump-sum.
        """
        
        # Calculate DCA schedule
        total_periods = len(actual_returns)
        dca_periods = list(range(0, total_periods, self.dca_frequency))
        
        # CORRECTED: Total amount to invest should be the same as lump-sum
        # But spread over DCA periods
        total_amount_to_invest = self.initial_capital
        amount_per_dca = total_amount_to_invest / len(dca_periods)
        
        print(f"DCA Setup:")
        print(f"  Total periods: {total_periods}")
        print(f"  DCA frequency: {self.dca_frequency} days")
        print(f"  Number of DCA investments: {len(dca_periods)}")
        print(f"  Amount per DCA: ${amount_per_dca:.2f}")
        print(f"  Total to invest: ${total_amount_to_invest:.2f}")
        
        # Simulate stock price starting at $100
        # CRITICAL FIX: Create price array with same length as actual_returns + 1
        stock_prices = [100.0]
        for ret in actual_returns:
            stock_prices.append(stock_prices[-1] * (1 + ret))
        stock_prices = np.array(stock_prices)
        
        print(f"Array length check:")
        print(f"  actual_returns length: {len(actual_returns)}")
        print(f"  stock_prices length: {len(stock_prices)}")
        
        # DCA simulation - iterate only over the same length as actual_returns
        dca_values = []
        cash_remaining = 0
        shares_owned = 0
        total_invested = 0
        
        # CRITICAL FIX: Only iterate through actual_returns length to avoid size mismatch
        for i in range(len(actual_returns)):
            current_stock_price = stock_prices[i]
            
            # Check if it's a DCA investment day
            if i in dca_periods:
                # Add cash for investment
                cash_remaining += amount_per_dca
                total_invested += amount_per_dca
            
            # Invest all available cash immediately (typical DCA behavior)
            if cash_remaining > 0:
                shares_to_buy = cash_remaining / current_stock_price
                shares_owned += shares_to_buy
                cash_remaining = 0  # All cash invested
            
            # Calculate portfolio value (shares * current price + remaining cash)
            portfolio_value = shares_owned * current_stock_price + cash_remaining
            dca_values.append(portfolio_value)
        
        # Add final value using the last stock price
        final_stock_price = stock_prices[-1]
        final_portfolio_value = shares_owned * final_stock_price + cash_remaining
        dca_values.append(final_portfolio_value)
        
        dca_values = np.array(dca_values)
        
        # Debug info
        stats = {
            'total_shares': shares_owned,
            'total_invested': total_invested,
            'final_stock_price': final_stock_price,
            'dca_periods': dca_periods,
            'amount_per_dca': amount_per_dca
        }
        
        return dca_values, stats
    
    def _calculate_trading_signals(self, pred_returns: np.ndarray, 
                                 position_fractions: np.ndarray) -> Dict[str, Any]:
        """Calculate trading signals for analysis."""
        
        pred_direction = np.sign(pred_returns)
        position_threshold = 0.1  # Show decisions only when position > 10%
        significant_positions = position_fractions > position_threshold
        
        buy_signals = (pred_direction > 0) & significant_positions
        sell_signals = (pred_direction < 0) & significant_positions
        hold_signals = ~significant_positions
        
        return {
            'buy_signals': buy_signals,
            'sell_signals': sell_signals,
            'hold_signals': hold_signals,
            'position_threshold': position_threshold
        }
    
    def _calculate_performance_metrics(self, portfolio_results: Dict[str, np.ndarray],
                                     strategy_returns: np.ndarray,
                                     actual_returns: np.ndarray) -> Dict[str, Any]:
        """Calculate comprehensive performance metrics."""
        
        # Extract values
        strategy_values = portfolio_results['strategy_values']
        lump_sum_values = portfolio_results['lump_sum_values']
        dca_values = portfolio_results['dca_values']
        dca_stats = portfolio_results['dca_stats']
        
        # Calculate returns
        strategy_total_return = ((strategy_values[-1] - self.initial_capital) / self.initial_capital) * 100
        lump_sum_total_return = ((lump_sum_values[-1] - self.initial_capital) / self.initial_capital) * 100
        
        # CORRECTED DCA return calculation
        dca_total_return = ((dca_values[-1] - dca_stats['total_invested']) / dca_stats['total_invested']) * 100
        
        # Calculate Sharpe ratios
        strategy_sharpe = np.mean(strategy_returns) / np.std(strategy_returns) if np.std(strategy_returns) > 0 else 0
        lump_sum_sharpe = np.mean(actual_returns) / np.std(actual_returns) if np.std(actual_returns) > 0 else 0
        
        # For DCA Sharpe, we need to calculate period returns

        print(f"  strategy_values: {len(strategy_values)}")
        print(f"  lump_sum_values: {len(lump_sum_values)}")
        print(f"  dca_values: {len(dca_values)}")
        print(f"  strategy_returns: {len(strategy_returns)}")
        print(f"  actual_returns: {len(actual_returns)}")
        
        # Ensure all arrays have the same length for consistent calculation
        # All values arrays should have length N+1 (one more than returns)
        min_values_length = min(len(strategy_values), len(lump_sum_values), len(dca_values))
        strategy_values = strategy_values[:min_values_length]
        lump_sum_values = lump_sum_values[:min_values_length]
        dca_values = dca_values[:min_values_length]
        
        # Returns arrays should have length N (one less than values)
        # Ensure returns arrays match the expected length
        expected_returns_length = min_values_length - 1
        if len(strategy_returns) != expected_returns_length:
            strategy_returns = strategy_returns[:expected_returns_length]
        if len(actual_returns) != expected_returns_length:
            actual_returns = actual_returns[:expected_returns_length]

        print(f"  strategy_values: {len(strategy_values)}")
        print(f"  lump_sum_values: {len(lump_sum_values)}")
        print(f"  dca_values: {len(dca_values)}")
        print(f"  strategy_returns: {len(strategy_returns)}")
        print(f"  actual_returns: {len(actual_returns)}")
        print(f"  Expected relationship: values_length = returns_length + 1")
        
        try:
            # Calculate DCA returns from portfolio values
            if len(dca_values) > 1:
                dca_returns = np.diff(dca_values) / dca_values[:-1]
                # Remove zero returns from non-investment periods to avoid division issues
                dca_returns = dca_returns[dca_returns != 0]
                
                if len(dca_returns) > 1 and np.std(dca_returns) > 0:
                    dca_sharpe = np.mean(dca_returns) / np.std(dca_returns)
                else:
                    dca_sharpe = 0.0
                    
                print(f"  DCA returns calculated: {len(dca_returns)} non-zero returns")
                print(f"  DCA Sharpe ratio: {dca_sharpe:.4f}")
            else:
                dca_sharpe = 0.0
                print(f"  DCA Sharpe ratio: 0.0 (insufficient data)")
                
        except Exception as e:

            print(f"dca_values length: {len(dca_values)}")
            print(f"dca_values dtype: {dca_values.dtype}")
            dca_sharpe = 0
        
        # Maximum drawdowns
        def calculate_max_drawdown(values):
            try:
                peak = np.maximum.accumulate(values)
                drawdown = (peak - values) / peak
                return np.max(drawdown) * 100
            except Exception as e:

                print(f"values shape: {values.shape}, dtype: {values.dtype}")
                return 0.0
        
        try:
            max_drawdown_strategy = calculate_max_drawdown(strategy_values)
        except Exception as e:

            max_drawdown_strategy = 0.0
            
        try:
            max_drawdown_lump_sum = calculate_max_drawdown(lump_sum_values)
        except Exception as e:

            max_drawdown_lump_sum = 0.0
            
        try:
            max_drawdown_dca = calculate_max_drawdown(dca_values)
        except Exception as e:

            max_drawdown_dca = 0.0
        
        # Trading statistics
        try:

            positive_trades = strategy_returns[strategy_returns > 0]
            negative_trades = strategy_returns[strategy_returns < 0]
            win_rate = len(positive_trades) / len(strategy_returns) if len(strategy_returns) > 0 else 0
            profit_factor = np.sum(positive_trades) / abs(np.sum(negative_trades)) if len(negative_trades) > 0 and np.sum(negative_trades) != 0 else np.inf
        except Exception as e:

            print(f"strategy_returns shape: {strategy_returns.shape}, dtype: {strategy_returns.dtype}")
            positive_trades = np.array([])
            negative_trades = np.array([])
            win_rate = 0.0
            profit_factor = 1.0
        
        return {
            'returns': {
                'strategy': strategy_total_return,
                'lump_sum': lump_sum_total_return,
                'dca': dca_total_return
            },
            'sharpe_ratios': {
                'strategy': strategy_sharpe,
                'lump_sum': lump_sum_sharpe,
                'dca': dca_sharpe
            },
            'max_drawdowns': {
                'strategy': max_drawdown_strategy,
                'lump_sum': max_drawdown_lump_sum,
                'dca': max_drawdown_dca
            },
            'trading_stats': {
                'win_rate': win_rate,
                'profit_factor': profit_factor,
                'num_trades': len(strategy_returns),
                'avg_return': np.mean(strategy_returns)
            },
            'final_values': {
                'strategy': strategy_values[-1],
                'lump_sum': lump_sum_values[-1],
                'dca': dca_values[-1]
            }
        }
    
    def plot_results(self, output_dir: Path) -> None:
        """Create comprehensive visualization of simulation results."""
        if not self.results:
            raise ValueError("No simulation results to plot. Run simulation first.")
        
        self._plot_trading_overview(output_dir)
        self._plot_portfolio_comparison(output_dir)
        self._plot_portfolio_distribution(output_dir)
    
    def _plot_trading_overview(self, output_dir: Path) -> None:
        """Plot trading overview with returns and decisions."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 14))
        
        pred_returns = self.results['pred_returns']
        actual_returns = self.results['actual_returns']
        position_fractions = self.results['position_fractions']
        risk_metrics = self.results['risk_metrics']
        trading_signals = self.results['trading_signals']
        
        # CRITICAL FIX: Ensure all arrays have the same length before plotting
        min_length = min(len(pred_returns), len(actual_returns), len(position_fractions))
        pred_returns = pred_returns[:min_length]
        actual_returns = actual_returns[:min_length]
        position_fractions = position_fractions[:min_length]
        
        time_indices = range(min_length)

        print(f"  pred_returns: {len(pred_returns)}")
        print(f"  actual_returns: {len(actual_returns)}")
        print(f"  position_fractions: {len(position_fractions)}")
        print(f"  time_indices: {len(time_indices)}")
        
        # 1. Returns comparison with market regime
        ax1.plot(time_indices, actual_returns * 100, 'b-', label='Actual Returns (%)', linewidth=1, alpha=0.7)
        ax1.plot(time_indices, pred_returns * 100, 'r--', label='Predicted Returns (%)', linewidth=1, alpha=0.7)
        
        # Color-code background by market regime
        regime_colors = {'high_vol': 'red', 'low_vol': 'green', 'normal': 'gray'}
        market_regime = risk_metrics['market_regime'][:min_length]
        
        start_idx = 0
        for i, regime in enumerate(market_regime):
            if i == 0 or market_regime[i-1] != regime:
                start_idx = i
            if i == len(market_regime)-1 or market_regime[i+1] != regime:
                end_idx = i
                ax1.axvspan(start_idx, end_idx, alpha=0.1, color=regime_colors[regime])
        
        # Add trading signals (align to same length)
        buy_signals = trading_signals['buy_signals'][:min_length]
        sell_signals = trading_signals['sell_signals'][:min_length]
        hold_signals = trading_signals['hold_signals'][:min_length]
        
        if np.any(buy_signals):
            ax1.scatter(np.where(buy_signals)[0], pred_returns[buy_signals] * 100, 
                       color='green', marker='^', s=60, alpha=0.9, label='BUY Decision', zorder=5)
        
        if np.any(sell_signals):
            ax1.scatter(np.where(sell_signals)[0], pred_returns[sell_signals] * 100, 
                       color='red', marker='v', s=60, alpha=0.9, label='SELL Decision', zorder=5)
        
        ax1.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax1.set_title('Daily Returns & Trading Decisions\\n(Red: High Vol, Green: Low Vol, Gray: Normal)', 
                     fontsize=14, fontweight='bold')
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Returns (%)')
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # 2. Position sizing over time
        ax2.plot(time_indices, position_fractions * 100, 'purple', linewidth=2, label='Position Size (%)')
        ax2.fill_between(time_indices, 0, position_fractions * 100, alpha=0.3, color='purple')
        ax2.axhline(y=trading_signals['position_threshold'] * 100, color='red', linestyle='--', 
                   alpha=0.7, label=f'Threshold ({trading_signals["position_threshold"]*100:.0f}%)')
        
        ax2.set_title('Dynamic Position Sizing', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Position Size (%)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 70)
        
        # 3. Market regime and volatility
        rolling_vol = risk_metrics['rolling_volatility'][:min_length]
        ax3.plot(time_indices, rolling_vol, 'orange', linewidth=1, label='Rolling Volatility')
        ax3.axhline(y=np.median(rolling_vol), color='blue', linestyle='--', alpha=0.7, label='Median Vol')
        ax3.axhline(y=np.median(rolling_vol) * 1.5, color='red', linestyle='--', alpha=0.7, label='High Vol Threshold')
        ax3.axhline(y=np.median(rolling_vol) * 0.7, color='green', linestyle='--', alpha=0.7, label='Low Vol Threshold')
        
        ax3.set_title('Market Volatility Analysis', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Time')
        ax3.set_ylabel('Volatility')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Prediction accuracy over time
        rolling_accuracy = risk_metrics['rolling_pred_accuracy'][:min_length]
        ax4.plot(time_indices, rolling_accuracy * 100, 'green', linewidth=2, label='Rolling Accuracy (%)')
        ax4.axhline(y=50, color='black', linestyle='--', alpha=0.5, label='Random (50%)')
        ax4.fill_between(time_indices, 45, rolling_accuracy * 100, alpha=0.3, color='green')
        
        ax4.set_title('Prediction Accuracy Over Time', fontsize=14, fontweight='bold')
        ax4.set_xlabel('Time')
        ax4.set_ylabel('Accuracy (%)')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.set_ylim(40, 80)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'trading_overview.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_portfolio_comparison(self, output_dir: Path) -> None:
        """Plot portfolio value comparison."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 14))
        
        portfolio_results = self.results['portfolio_results']
        performance_metrics = self.results['performance_metrics']
        
        strategy_values = portfolio_results['strategy_values']
        lump_sum_values = portfolio_results['lump_sum_values']
        dca_values = portfolio_results['dca_values']
        
        # CRITICAL FIX: Ensure all arrays have the same length before plotting
        min_length = min(len(strategy_values), len(lump_sum_values), len(dca_values))
        strategy_values = strategy_values[:min_length]
        lump_sum_values = lump_sum_values[:min_length]
        dca_values = dca_values[:min_length]
        
        time_indices = range(min_length)

        print(f"  strategy_values: {len(strategy_values)}")
        print(f"  lump_sum_values: {len(lump_sum_values)}")
        print(f"  dca_values: {len(dca_values)}")
        print(f"  time_indices: {len(time_indices)}")
        
        # 1. Portfolio values over time
        ax1.plot(time_indices, [self.initial_capital] * len(time_indices), 'g--', 
                label='Initial Capital', linewidth=1)
        ax1.plot(time_indices, strategy_values, 'purple', label='Enhanced TFT Strategy', linewidth=2)
        ax1.plot(time_indices, dca_values, 'orange', label='DCA Buy & Hold', linewidth=2)
        ax1.plot(time_indices, lump_sum_values, 'blue', label='Lump-sum Buy & Hold', linewidth=2, alpha=0.7)
        
        # Add drawdown shading for strategy
        strategy_peak = np.maximum.accumulate(strategy_values)
        strategy_drawdown = (strategy_peak - strategy_values) / strategy_peak
        ax1.fill_between(time_indices, strategy_values, strategy_peak, 
                        where=(strategy_drawdown > 0), alpha=0.3, color='red', label='Strategy Drawdown')
        
        ax1.set_title('Portfolio Value Comparison Over Time', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Portfolio Value ($)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Returns comparison
        returns = performance_metrics['returns']
        strategies = ['Enhanced TFT', 'DCA Buy & Hold', 'Lump-sum B&H']
        return_values = [returns['strategy'], returns['dca'], returns['lump_sum']]
        colors = ['red' if r < 0 else 'green' for r in return_values]
        
        bars = ax2.bar(strategies, return_values, color=colors, alpha=0.7)
        ax2.set_title('Strategy Returns Comparison', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Return (%)')
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        
        # Add value labels
        for bar, ret in zip(bars, return_values):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.5 if height >= 0 else height - 1,
                    f'{ret:.2f}%', ha='center', va='bottom' if height >= 0 else 'top')
        
        # 3. Risk metrics comparison
        sharpe_ratios = performance_metrics['sharpe_ratios']
        max_drawdowns = performance_metrics['max_drawdowns']
        
        x = np.arange(len(strategies))
        width = 0.35
        
        bars1 = ax3.bar(x - width/2, [sharpe_ratios['strategy'], sharpe_ratios['dca'], sharpe_ratios['lump_sum']], 
                       width, label='Sharpe Ratio', alpha=0.7)
        # Convert max drawdown from percentage to decimal format to match Sharpe ratio scale
        bars2 = ax3.bar(x + width/2, [-max_drawdowns['strategy']/100, -max_drawdowns['dca']/100, -max_drawdowns['lump_sum']/100], 
                       width, label='Max Drawdown (as decimal)', alpha=0.7)
        
        ax3.set_title('Risk-Adjusted Performance', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Strategy')
        ax3.set_ylabel('Ratio/Decimal Value')
        ax3.set_xticks(x)
        ax3.set_xticklabels(strategies)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Add value labels on bars for clarity
        for bar in bars1:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.02 if height >= 0 else height - 0.02,
                    f'{height:.3f}', ha='center', va='bottom' if height >= 0 else 'top')
        
        for bar in bars2:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height - 0.02 if height <= 0 else height + 0.02,
                    f'{height:.3f}', ha='center', va='top' if height <= 0 else 'bottom')
        ax3.grid(True, alpha=0.3)
        ax3.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        
        # 4. Performance summary
        final_values = performance_metrics['final_values']
        trading_stats = performance_metrics['trading_stats']
        dca_stats = portfolio_results['dca_stats']
        
        summary_text = f"""CORRECTED Trading Performance Summary:

📊 FINAL VALUES:
Enhanced TFT Strategy: ${final_values['strategy']:,.2f}
DCA Buy & Hold: ${final_values['dca']:,.2f}
Lump-sum Buy & Hold: ${final_values['lump_sum']:,.2f}

💰 INVESTMENT DETAILS:
Initial Capital: ${self.initial_capital:,}
DCA Total Invested: ${dca_stats['total_invested']:,.2f}
DCA Frequency: Every {self.dca_frequency} day(s)
DCA Amount per Investment: ${dca_stats['amount_per_dca']:.2f}

📈 RETURNS:
TFT Strategy: {returns['strategy']:.2f}%
DCA Buy & Hold: {returns['dca']:.2f}%
Lump-sum B&H: {returns['lump_sum']:.2f}%

🎯 RISK METRICS:
Sharpe Ratios: {sharpe_ratios['strategy']:.3f} | {sharpe_ratios['dca']:.3f} | {sharpe_ratios['lump_sum']:.3f}
Max Drawdowns: {max_drawdowns['strategy']:.2f}% | {max_drawdowns['dca']:.2f}% | {max_drawdowns['lump_sum']:.2f}%

📊 TRADING STATS:
Win Rate: {trading_stats['win_rate']:.2%}
Profit Factor: {trading_stats['profit_factor']:.2f}
Number of Trades: {trading_stats['num_trades']}
Avg Daily Return: {trading_stats['avg_return']*100:.3f}%

Note: DCA and Lump-sum now use same total investment amount for fair comparison."""
        
        ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightcyan'))
        ax4.set_title('CORRECTED Performance Summary', fontsize=14, fontweight='bold')
        ax4.axis('off')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'portfolio_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_portfolio_distribution(self, output_dir: Path) -> None:
        """Plot portfolio allocation distribution over time."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 14))
        
        position_fractions = self.results['position_fractions']
        portfolio_results = self.results['portfolio_results']
        strategy_values = portfolio_results['strategy_values']
        risk_metrics = self.results['risk_metrics']
        
        # CRITICAL FIX: Ensure all arrays have the same length
        min_length = min(len(position_fractions), len(strategy_values))
        position_fractions = position_fractions[:min_length]
        strategy_values = strategy_values[:min_length]
        
        time_indices = range(min_length)

        print(f"  position_fractions: {len(position_fractions)}")
        print(f"  strategy_values: {len(strategy_values)}")
        print(f"  time_indices: {len(time_indices)}")
        
        # 1. Position allocation over time
        cash_allocation = 1 - position_fractions
        ax1.fill_between(time_indices, 0, position_fractions * 100, alpha=0.7, color='blue', label='Stock Allocation')
        ax1.fill_between(time_indices, position_fractions * 100, 100, alpha=0.7, color='green', label='Cash Allocation')
        
        ax1.set_title('Portfolio Allocation Over Time', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Allocation (%)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 100)
        
        # 2. Dollar amounts in each allocation
        dollar_in_stocks = position_fractions * strategy_values
        dollar_in_cash = (1 - position_fractions) * strategy_values
        
        ax2.fill_between(time_indices, 0, dollar_in_stocks, alpha=0.7, color='blue', label='Stocks ($)')
        ax2.fill_between(time_indices, dollar_in_stocks, dollar_in_stocks + dollar_in_cash, 
                        alpha=0.7, color='green', label='Cash ($)')
        
        ax2.set_title('Portfolio Allocation in Dollars', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Value ($)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Risk-adjusted position sizing
        # Ensure risk metrics arrays are also aligned
        rolling_vol = risk_metrics['rolling_volatility'][:min_length]
        rolling_accuracy = risk_metrics['rolling_pred_accuracy'][:min_length]
        
        ax3.plot(time_indices, position_fractions * 100, 'purple', label='Position Size', linewidth=2)
        ax3.plot(time_indices, rolling_vol * 1000, 'red', 
                label='Volatility (x1000)', linewidth=1, alpha=0.7)
        ax3.plot(time_indices, rolling_accuracy * 100, 'orange', 
                label='Prediction Accuracy', linewidth=1, alpha=0.7)
        
        ax3.set_title('Position Sizing vs Risk Metrics', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Time')
        ax3.set_ylabel('Value')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Market regime and allocation
        regime_colors = {'low_vol': 'green', 'normal': 'blue', 'high_vol': 'red'}
        market_regime = risk_metrics['market_regime'][:min_length]
        regime_numeric = np.array([0 if r == 'low_vol' else 1 if r == 'normal' else 2 
                                  for r in market_regime])
        
        # Create a colored background based on market regime
        for i in range(len(time_indices)-1):
            regime = market_regime[i]
            color = regime_colors.get(regime, 'gray')
            ax4.axvspan(i, i+1, alpha=0.2, color=color)
        
        ax4.plot(time_indices, position_fractions * 100, 'black', label='Position Size (%)', linewidth=2)
        
        # Add regime legend
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor='green', alpha=0.5, label='Low Volatility'),
                          Patch(facecolor='blue', alpha=0.5, label='Normal'),
                          Patch(facecolor='red', alpha=0.5, label='High Volatility')]
        
        ax4.set_title('Position Sizing by Market Regime', fontsize=14, fontweight='bold')
        ax4.set_xlabel('Time')
        ax4.set_ylabel('Position Size (%)')
        ax4.legend(handles=legend_elements, loc='upper left')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'portfolio_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def get_summary_report(self) -> str:
        """Generate a text summary report of the simulation results."""
        if not self.results:
            return "No simulation results available. Run simulation first."
        
        performance_metrics = self.results['performance_metrics']
        portfolio_results = self.results['portfolio_results']
        trading_signals = self.results['trading_signals']
        
        returns = performance_metrics['returns']
        final_values = performance_metrics['final_values']
        trading_stats = performance_metrics['trading_stats']
        dca_stats = portfolio_results['dca_stats']
        
        # Calculate signal statistics
        buy_count = np.sum(trading_signals['buy_signals'])
        sell_count = np.sum(trading_signals['sell_signals'])
        hold_count = np.sum(trading_signals['hold_signals'])
        total_signals = buy_count + sell_count + hold_count
        
        report = f"""
=== TRADING SIMULATION SUMMARY REPORT ===

📊 PORTFOLIO PERFORMANCE:
• Enhanced TFT Strategy:    ${final_values['strategy']:>12,.2f} ({returns['strategy']:>7.2f}%)
• DCA Buy & Hold:          ${final_values['dca']:>12,.2f} ({returns['dca']:>7.2f}%)
• Lump-sum Buy & Hold:     ${final_values['lump_sum']:>12,.2f} ({returns['lump_sum']:>7.2f}%)

💰 INVESTMENT COMPARISON:
• Initial Capital:         ${self.initial_capital:>12,.2f}
• DCA Total Invested:      ${dca_stats['total_invested']:>12,.2f}
• DCA Frequency:           Every {self.dca_frequency} day(s)
• DCA Amount per Period:   ${dca_stats['amount_per_dca']:>12.2f}

🎯 OUTPERFORMANCE:
• TFT vs DCA:              {returns['strategy'] - returns['dca']:>12.2f}%
• TFT vs Lump-sum:         {returns['strategy'] - returns['lump_sum']:>12.2f}%
• DCA vs Lump-sum:         {returns['dca'] - returns['lump_sum']:>12.2f}%

📊 TRADING ACTIVITY:
• Total Trading Periods:   {trading_stats['num_trades']:>12}
• BUY Signals:             {buy_count:>12} ({buy_count/total_signals:.1%})
• SELL Signals:            {sell_count:>12} ({sell_count/total_signals:.1%})
• HOLD Signals:            {hold_count:>12} ({hold_count/total_signals:.1%})

• Win Rate:                {trading_stats['win_rate']:>12.1%}
• Profit Factor:           {trading_stats['profit_factor']:>12.2f}
• Avg Daily Return:        {trading_stats['avg_return']*100:>12.3f}%

🛡️ RISK METRICS:
• Sharpe Ratios:           {performance_metrics['sharpe_ratios']['strategy']:>6.3f} | {performance_metrics['sharpe_ratios']['dca']:>6.3f} | {performance_metrics['sharpe_ratios']['lump_sum']:>6.3f}
• Max Drawdowns:           {performance_metrics['max_drawdowns']['strategy']:>6.2f}% | {performance_metrics['max_drawdowns']['dca']:>6.2f}% | {performance_metrics['max_drawdowns']['lump_sum']:>6.2f}%

📝 KEY INSIGHTS:
• The corrected DCA calculation now provides fair comparison by using the same total investment amount
• {'DCA outperformed lump-sum' if returns['dca'] > returns['lump_sum'] else 'Lump-sum outperformed DCA'} by {abs(returns['dca'] - returns['lump_sum']):.2f}%
• {'TFT strategy was profitable' if returns['strategy'] > 0 else 'TFT strategy was unprofitable'} with {returns['strategy']:.2f}% return
• Trading activity: {(buy_count + sell_count)/total_signals:.1%} active trading vs {hold_count/total_signals:.1%} holding cash

=== END REPORT ===
"""
        return report
