"""
Example demonstration of the corrected trading simulator.

This script shows how the DCA and lump-sum buy & hold calculations are now correctly implemented.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from trading_simulator import TradingSimulator

def generate_sample_data(num_days=252, volatility=0.02, trend=0.0005):
    """Generate sample stock return data for testing."""
    # Generate realistic stock returns with some autocorrelation
    np.random.seed(42)  # For reproducible results
    
    # Base random returns
    random_returns = np.random.normal(trend, volatility, num_days)
    
    # Add some autocorrelation to make it more realistic
    returns = np.zeros(num_days)
    returns[0] = random_returns[0]
    
    for i in range(1, num_days):
        # 70% random, 30% influenced by previous return
        returns[i] = 0.7 * random_returns[i] + 0.3 * returns[i-1] * 0.1
    
    # Generate some "predictions" that are somewhat correlated with actual returns
    # but with noise (simulating a model that's better than random but not perfect)
    predictions = returns * 0.6 + np.random.normal(0, volatility * 0.8, num_days)
    
    return predictions, returns

def demonstrate_neural_network_to_kelly_flow():
    """Demonstrate the step-by-step conversion from NN predictions to Kelly positions."""
    print("\n" + "="*80)
    print("🧠 NEURAL NETWORK TO KELLY CRITERION FLOW DEMONSTRATION")
    print("="*80)
    
    # Example: Single day prediction flow
    print("\n1️⃣ NEURAL NETWORK OUTPUT (TFT Model)")
    print("-" * 50)
    nn_prediction = 0.025  # 2.5% predicted return
    print(f"   TFT Model Prediction: {nn_prediction:+.3f} ({nn_prediction*100:+.1f}%)")
    print(f"   Interpretation: Model expects {nn_prediction*100:+.1f}% return tomorrow")
    
    print("\n2️⃣ RISK ASSESSMENT")
    print("-" * 50)
    # Simulated risk metrics
    max_prediction = 0.032  # Maximum prediction seen recently
    prediction_confidence = abs(nn_prediction) / max_prediction
    current_volatility = 0.018
    recent_accuracy = 0.63
    
    print(f"   Prediction Confidence: {prediction_confidence:.3f} ({prediction_confidence*100:.1f}%)")
    print(f"   Current Market Volatility: {current_volatility:.3f} ({current_volatility*100:.1f}%)")
    print(f"   Recent Model Accuracy: {recent_accuracy:.3f} ({recent_accuracy*100:.1f}%)")
    
    print("\n3️⃣ WIN PROBABILITY ESTIMATION")
    print("-" * 50)
    base_win_prob = 0.52
    confidence_boost = prediction_confidence * 0.15
    accuracy_boost = (recent_accuracy - 0.5) * 0.2
    final_win_prob = base_win_prob + confidence_boost + accuracy_boost
    final_win_prob = max(0.45, min(0.75, final_win_prob))  # Clamp
    
    print(f"   Base Win Probability: {base_win_prob:.3f} ({base_win_prob*100:.1f}%)")
    print(f"   + Confidence Boost: {confidence_boost:.3f} ({confidence_boost*100:.1f}%)")
    print(f"   + Accuracy Boost: {accuracy_boost:.3f} ({accuracy_boost*100:.1f}%)")
    print(f"   = Final Win Probability: {final_win_prob:.3f} ({final_win_prob*100:.1f}%)")
    
    print("\n4️⃣ KELLY CRITERION POSITION SIZING")
    print("-" * 50)
    kelly_fraction = 2 * final_win_prob - 1
    conservative_factor = 0.25
    volatility_penalty = 1 / (1 + current_volatility * 10)
    base_position = kelly_fraction * conservative_factor * volatility_penalty
    
    print(f"   Kelly Fraction: 2 × {final_win_prob:.3f} - 1 = {kelly_fraction:.3f}")
    print(f"   × Conservative Factor: {conservative_factor:.3f}")
    print(f"   × Volatility Penalty: {volatility_penalty:.3f}")
    print(f"   = Base Position Size: {base_position:.3f} ({base_position*100:.1f}%)")
    
    print("\n5️⃣ MARKET REGIME ADJUSTMENT")
    print("-" * 50)
    # Simulate market regime
    if current_volatility > 0.025:
        regime = "HIGH_VOLATILITY"
        regime_multiplier = 0.5
    elif current_volatility < 0.012:
        regime = "LOW_VOLATILITY" 
        regime_multiplier = 1.2
    else:
        regime = "NORMAL"
        regime_multiplier = 1.0
    
    final_position = base_position * regime_multiplier
    final_position = max(0.0, min(0.6, final_position))  # 0-60% limit
    
    print(f"   Market Regime: {regime}")
    print(f"   Regime Multiplier: {regime_multiplier:.1f}")
    print(f"   = Final Position Size: {final_position:.3f} ({final_position*100:.1f}%)")
    
    print("\n6️⃣ DIRECTION APPLICATION")
    print("-" * 50)
    direction = "LONG" if nn_prediction > 0 else "SHORT"
    directional_position = final_position * (1 if nn_prediction > 0 else -1)
    
    print(f"   Prediction Direction: {direction} (prediction = {nn_prediction:+.3f})")
    print(f"   Final Trade Signal: {direction} {abs(directional_position)*100:.1f}% of portfolio")
    
    print("\n7️⃣ EXPECTED OUTCOME")
    print("-" * 50)
    # Simulate actual return
    actual_return = 0.018  # What actually happened
    strategy_return = directional_position * actual_return
    buy_hold_return = actual_return
    
    print(f"   Actual Market Return: {actual_return:+.3f} ({actual_return*100:+.1f}%)")
    print(f"   Strategy Return: {strategy_return:+.4f} ({strategy_return*100:+.2f}%)")
    print(f"   Buy & Hold Return: {buy_hold_return:+.3f} ({buy_hold_return*100:+.1f}%)")
    print(f"   Strategy Advantage: {(strategy_return - buy_hold_return)*100:+.2f} percentage points")
    
    print(f"\n💡 KEY INSIGHT: The model predicted {nn_prediction*100:+.1f}%, market delivered {actual_return*100:+.1f}%")
    print(f"   The Kelly system sized the position at {final_position*100:.1f}% to optimize risk-adjusted returns")
    
    return final_position, strategy_return

def main():
    """Run a demonstration of the trading simulator."""
    print("=== TRADING SIMULATOR DEMONSTRATION ===")
    print("How Neural Network Predictions Become Kelly Criterion Trading Decisions\n")
    
    # First show the step-by-step conversion
    demonstrate_neural_network_to_kelly_flow()
    
    print("\n" + "="*80)
    print("📊 FULL SIMULATION: CORRECTED DCA vs Lump-sum calculations")
    print("="*80)
    
    # Generate sample data
    print("\n📊 Generating sample data...")
    predictions, actual_returns = generate_sample_data(num_days=252, volatility=0.02, trend=0.0008)
    
    print(f"   Generated {len(actual_returns)} days of data")
    print(f"   Actual returns: Mean={np.mean(actual_returns)*100:.3f}%, Std={np.std(actual_returns)*100:.2f}%")
    print(f"   Predictions: Mean={np.mean(predictions)*100:.3f}%, Std={np.std(predictions)*100:.2f}%")
    
    # Test different DCA frequencies
    dca_frequencies = [1, 7, 30]  # Daily, weekly, monthly
    
    for dca_freq in dca_frequencies:
        print(f"\n{'='*60}")
        print(f"🧪 TESTING DCA FREQUENCY: Every {dca_freq} day(s)")
        print(f"{'='*60}")
        
        # Initialize simulator
        simulator = TradingSimulator(
            initial_capital=10000,
            dca_frequency=dca_freq
        )
        
        # Run simulation
        try:
            results = simulator.run_simulation(predictions, actual_returns)
            
            # Print summary
            print(simulator.get_summary_report())
            
            # Create output directory
            output_dir = Path(f"simulation_results_dca_{dca_freq}")
            output_dir.mkdir(exist_ok=True)
            
            # Generate plots
            simulator.plot_results(output_dir)
            print(f"📊 Results saved to: {output_dir}")
            
        except Exception as e:
            print(f"❌ Error running simulation: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*60)
    print("🎯 KEY INSIGHTS FROM NEURAL NETWORK → KELLY CONVERSION:")
    print("="*60)
    print("1. TFT Model outputs percentage return predictions (e.g., +2.5%)")
    print("2. Risk Assessment converts predictions into confidence scores")
    print("3. Kelly Criterion calculates mathematically optimal position sizes")
    print("4. Market Regime detection adjusts for volatility conditions")
    print("5. Conservative factors prevent over-leveraging (max 60% position)")
    print("6. DCA and Lump-sum now use SAME total investment for fair comparison")
    print("7. Strategy performance reflects MODEL SKILL + RISK MANAGEMENT")
    
    print("\n🧠 NEURAL NETWORK INTEGRATION:")
    print("   • Prediction Magnitude → Confidence Level → Position Size")
    print("   • Model Accuracy Tracking → Dynamic Win Probability")
    print("   • Volatility Detection → Risk Adjustment")
    print("   • Kelly Optimization → Maximum Long-term Growth")
    
    print("\n✅ This demonstrates how sophisticated ML predictions get converted")
    print("    into mathematically sound, risk-managed trading decisions!")
    
    print("\n✅ Demonstration completed!")

if __name__ == "__main__":
    main()
