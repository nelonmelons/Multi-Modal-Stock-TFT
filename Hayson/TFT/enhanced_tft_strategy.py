#!/usr/bin/env python3
"""
Enhanced TFT Strategy with Full Feature Verification, Interpretation, and Profit Visualization

This enhanced implementation provides:
1. Complete future context verification
2. Feature importance analysis  
3. Attention visualization
4. Trading strategy simulation
5. Profit/loss visualization
6. Model interpretability
"""

import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# PyTorch Lightning and TFT imports
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_forecasting import TemporalFusionTransformer
from pytorch_forecasting.metrics import MAE, RMSE, MAPE
from dataModule.interface import get_data_loader_with_module

import dotenv
dotenv.load_dotenv()


class EnhancedTFTAnalyzer:
    """Enhanced TFT analyzer with full feature verification and profit tracking."""
    
    def __init__(self, symbols: List[str], start_date: str, end_date: str, 
                 encoder_len: int = 90, predict_len: int = 21, batch_size: int = 64):
        self.symbols = symbols
        self.start_date = start_date
        self.end_date = end_date
        self.encoder_len = encoder_len
        self.predict_len = predict_len
        self.batch_size = batch_size
        
        # Model components
        self.dataloader = None
        self.datamodule = None
        self.model = None
        self.trainer = None
        
        # Analysis results
        self.feature_importance = None
        self.attention_weights = None
        self.trading_results = None
        
    def load_data_and_verify_features(self):
        """Load data and verify all future context features are implemented."""
        print("🔍 FEATURE VERIFICATION AND DATA LOADING")
        print("=" * 60)
        
        # Load data with full feature pipeline
        self.dataloader, self.datamodule = get_data_loader_with_module(
            symbols=self.symbols,
            start=self.start_date,
            end=self.end_date,
            encoder_len=self.encoder_len,
            predict_len=self.predict_len,
            batch_size=self.batch_size,
            news_api_key=os.getenv('NEWS_API_KEY'),
            fred_api_key=os.getenv('FRED_API_KEY'),
            api_ninjas_key=os.getenv('API_NINJAS_KEY')
        )
        
        # Verify future context implementation
        self._verify_future_context_features()
        
        # Print detailed feature analysis
        self._analyze_feature_distribution()
        
    def _verify_future_context_features(self):
        """Verify that all future context features are properly implemented."""
        print("\n📊 FUTURE CONTEXT FEATURE VERIFICATION")
        print("-" * 50)
        
        # Get dataset parameters
        params = self.datamodule.get_dataset_parameters()
        
        # Expected future context features
        expected_future_features = {
            # Calendar features (always known)
            'calendar': ['day_of_week', 'month', 'quarter', 'day_of_month', 'is_weekend', 'is_holiday'],
            
            # Economic indicators (released on schedule)
            'economic': ['cpi', 'fedfunds', 'unrate', 't10y2y', 'gdp', 'vix', 'dxy', 'oil'],
            
            # Corporate events (scheduled in advance)
            'events_categorical': ['is_earnings_day', 'is_split_day', 'is_dividend_day', 
                                 'earnings_in_prediction_window'],
            
            # Event timing (known future dates)
            'events_timing': ['days_to_next_earnings', 'days_to_next_split', 'days_to_next_dividend',
                            'days_since_earnings', 'days_to_earnings_in_window'],
            
            # Fundamental estimates (known from earnings calendar)
            'fundamentals': ['eps_estimate', 'eps_actual', 'revenue_estimate', 'revenue_actual']
        }
        
        # Verify implementation
        known_categoricals = params.get('time_varying_known_categoricals', [])
        known_reals = params.get('time_varying_known_reals', [])
        
        print("✅ KNOWN FUTURE FEATURES (Fed to TFT for Prediction Window):")
        
        for category, features in expected_future_features.items():
            print(f"\n📅 {category.upper()}:")
            implemented = []
            missing = []
            
            for feature in features:
                if feature in known_categoricals or feature in known_reals:
                    implemented.append(feature)
                else:
                    missing.append(feature)
            
            if implemented:
                print(f"   ✅ Implemented: {implemented}")
            if missing:
                print(f"   ❌ Missing: {missing}")
        
        # Verify data format and coverage
        print(f"\n📊 FEATURE COVERAGE ANALYSIS:")
        print(f"   📈 Total Known Categoricals: {len(known_categoricals)}")
        print(f"   📊 Total Known Reals: {len(known_reals)}")
        print(f"   📉 Unknown Reals (Historical): {len(params.get('time_varying_unknown_reals', []))}")
        
        # Sample data format verification
        if self.dataloader is not None:
            sample_batch = next(iter(self.dataloader))
        self._verify_data_format(sample_batch, params)
        
    def _verify_data_format(self, batch, params):
        """Verify the exact format of data fed into TFT model."""
        print(f"\n🔧 DATA FORMAT VERIFICATION:")
        print("-" * 40)
        
        x, y = batch
        
        print(f"INPUT TENSORS TO TFT MODEL:")
        print(f"   📥 Encoder inputs shape: {x['encoder_cont'].shape}")  # [batch, encoder_len, features]
        print(f"   📤 Decoder inputs shape: {x['decoder_cont'].shape}")  # [batch, predict_len, known_features]
        print(f"   🎯 Targets shape: {y[0].shape}")  # [batch, predict_len]
        
        # Verify known future features are in decoder
        print(f"\n🔮 FUTURE CONTEXT IN DECODER:")
        known_features = params.get('time_varying_known_reals', [])
        print(f"   📊 Known future features: {len(known_features)}")
        print(f"   📈 Decoder feature dim: {x['decoder_cont'].shape[-1]}")
        print(f"   ✅ Features match: {len(known_features) == x['decoder_cont'].shape[-1]}")
        
        # Show sample future context values
        if len(known_features) > 0:
            print(f"\n📋 SAMPLE FUTURE CONTEXT VALUES (First Sample, First 5 Days):")
            sample_decoder = x['decoder_cont'][0, :5, :].numpy()  # First sample, first 5 prediction days
            
            for i, feature in enumerate(known_features[:10]):  # Show first 10 features
                values = sample_decoder[:, i]
                print(f"   {feature:25}: {values}")
        
    def _analyze_feature_distribution(self):
        """Analyze the distribution and quality of features."""
        print(f"\n📈 FEATURE DISTRIBUTION ANALYSIS:")
        print("-" * 45)
        
        # Get feature dataframe
        feature_df = self.datamodule.feature_df
        
        # Analyze temporal coverage
        print(f"📅 TEMPORAL COVERAGE:")
        print(f"   Date range: {feature_df['date'].min()} to {feature_df['date'].max()}")
        print(f"   Total days: {(feature_df['date'].max() - feature_df['date'].min()).days}")
        print(f"   Symbols: {feature_df['symbol'].unique().tolist()}")
        print(f"   Rows per symbol: {feature_df.groupby('symbol').size().to_dict()}")
        
        # Analyze future events coverage
        events_features = ['earnings_in_prediction_window', 'days_to_next_earnings']
        for feature in events_features:
            if feature in feature_df.columns:
                if feature_df[feature].dtype == 'object':  # Categorical
                    coverage = feature_df[feature].value_counts()
                    print(f"   {feature}: {coverage.to_dict()}")
                else:  # Numerical
                    stats = feature_df[feature].describe()
                    print(f"   {feature}: mean={stats['mean']:.2f}, std={stats['std']:.2f}")
        
    def build_model(self):
        """Build TFT model with enhanced configuration."""
        print("\n🏗️  BUILDING ENHANCED TFT MODEL")
        print("=" * 40)
        
        # Create model with interpretability features
        self.model = TemporalFusionTransformer.from_dataset(
            self.datamodule.train_dataset,
            learning_rate=0.01,  # Slightly lower for stability
            hidden_size=32,      # Increased for better capacity
            attention_head_size=4,  # More attention heads
            dropout=0.15,
            hidden_continuous_size=16,
            output_size=7,  # Quantile regression
            loss=MAE(),
            log_interval=10,
            reduce_on_plateau_patience=4,
            # Enhanced interpretability
            logging_metrics=["mae", "rmse", "mape"],
        )
        
        print(f"✅ Model created with {self.model.hparams.hidden_size} hidden units")
        print(f"✅ Attention heads: {self.model.hparams.attention_head_size}")
        print(f"✅ Loss function: {type(self.model.loss).__name__}")
        
    def train_model(self, max_epochs: int = 50):
        """Train the model with enhanced callbacks."""
        print(f"\n🚀 TRAINING TFT MODEL ({max_epochs} epochs)")
        print("=" * 45)
        
        # Enhanced callbacks
        early_stop_callback = EarlyStopping(
            monitor="val_loss",
            min_delta=1e-4,
            patience=10,
            verbose=True,
            mode="min"
        )
        
        checkpoint_callback = ModelCheckpoint(
            dirpath="enhanced_tft_checkpoints",
            filename="enhanced-tft-{epoch:02d}-{val_loss:.3f}",
            monitor="val_loss",
            mode="min",
            save_top_k=3,
            verbose=True
        )
        
        # Trainer with enhanced features
        self.trainer = Trainer(
            max_epochs=max_epochs,
            accelerator="auto",
            callbacks=[early_stop_callback, checkpoint_callback],
            log_every_n_steps=10,
            enable_progress_bar=True
        )
        
        # Train model
        self.trainer.fit(
            self.model,
            train_dataloaders=self.datamodule.train_dataloader(),
            val_dataloaders=self.datamodule.val_dataloader(),
        )
        
        # Load best model
        if checkpoint_callback.best_model_path:
            print(f"📂 Loading best model: {checkpoint_callback.best_model_path}")
            self.model = TemporalFusionTransformer.load_from_checkpoint(
                checkpoint_callback.best_model_path
            )
        
    def analyze_model_interpretation(self):
        """Analyze model interpretability including attention and feature importance."""
        print(f"\n🔍 MODEL INTERPRETABILITY ANALYSIS")
        print("=" * 45)
        
        # Feature importance analysis
        interpretation = self.model.interpret_output(
            self.datamodule.val_dataloader(),
            reduce_on_plateau_patience=1
        )
        
        # Variable importance
        self.feature_importance = interpretation["feature_importances"]
        
        print("📊 TOP 10 FEATURE IMPORTANCES:")
        for i, (feature, importance) in enumerate(self.feature_importance.items()):
            if i < 10:
                print(f"   {i+1:2d}. {feature:25}: {importance:.4f}")
        
        # Attention analysis
        if "attention" in interpretation:
            self.attention_weights = interpretation["attention"]
            print(f"✅ Attention weights extracted: {self.attention_weights.shape}")
        
        return interpretation
    
    def create_trading_strategy(self, threshold: float = 0.02):
        """Create and backtest trading strategy based on TFT predictions."""
        print(f"\n💰 TRADING STRATEGY SIMULATION")
        print("=" * 40)
        
        # Get predictions
        predictions = self.model.predict(self.datamodule.val_dataloader())
        
        # Get actual values  
        actuals = torch.cat([y[0] for x, y in iter(self.datamodule.val_dataloader())])
        
        # Convert to numpy
        pred_values = predictions.numpy() if hasattr(predictions, 'numpy') else predictions
        actual_values = actuals.numpy()
        
        # Create trading signals
        trading_results = []
        
        for i in range(len(pred_values)):
            predicted_change = pred_values[i]
            actual_change = actual_values[i]
            
            # Simple trading signal based on prediction confidence
            if predicted_change > threshold:
                signal = "BUY"
                position = 1
            elif predicted_change < -threshold:
                signal = "SELL"  
                position = -1
            else:
                signal = "HOLD"
                position = 0
            
            # Calculate profit (assuming we trade based on the signal)
            if position != 0:
                profit = position * actual_change
            else:
                profit = 0
            
            trading_results.append({
                'prediction': predicted_change,
                'actual': actual_change,
                'signal': signal,
                'position': position,
                'profit': profit,
                'accuracy': 1 if np.sign(predicted_change) == np.sign(actual_change) else 0
            })
        
        self.trading_results = pd.DataFrame(trading_results)
        
        # Calculate performance metrics
        total_profit = self.trading_results['profit'].sum()
        accuracy = self.trading_results['accuracy'].mean()
        num_trades = len(self.trading_results[self.trading_results['position'] != 0])
        
        print(f"📈 TRADING PERFORMANCE:")
        print(f"   Total Profit: {total_profit:.4f}")
        print(f"   Prediction Accuracy: {accuracy:.2%}")
        print(f"   Number of Trades: {num_trades}")
        print(f"   Average Profit per Trade: {total_profit/max(num_trades,1):.4f}")
        
        return self.trading_results
    
    def visualize_results(self, save_plots: bool = True):
        """Create comprehensive visualizations of results and profits."""
        print(f"\n📊 CREATING VISUALIZATIONS")
        print("=" * 35)
        
        # Set up the plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Create figure with subplots
        fig = plt.figure(figsize=(20, 16))
        
        # 1. Feature Importance Plot
        ax1 = plt.subplot(3, 3, 1)
        if self.feature_importance:
            features = list(self.feature_importance.keys())[:15]
            importances = [self.feature_importance[f] for f in features]
            
            ax1.barh(range(len(features)), importances)
            ax1.set_yticks(range(len(features)))
            ax1.set_yticklabels(features, fontsize=8)
            ax1.set_xlabel('Importance')
            ax1.set_title('🎯 Top 15 Feature Importances')
            ax1.grid(True, alpha=0.3)
        
        # 2. Prediction vs Actual Scatter
        ax2 = plt.subplot(3, 3, 2)
        if self.trading_results is not None:
            ax2.scatter(self.trading_results['actual'], self.trading_results['prediction'], 
                       alpha=0.6, s=20)
            ax2.plot([-0.1, 0.1], [-0.1, 0.1], 'r--', alpha=0.8)
            ax2.set_xlabel('Actual Change')
            ax2.set_ylabel('Predicted Change')
            ax2.set_title('🎯 Predictions vs Actuals')
            ax2.grid(True, alpha=0.3)
            
            # Add correlation coefficient
            corr = np.corrcoef(self.trading_results['actual'], self.trading_results['prediction'])[0,1]
            ax2.text(0.05, 0.95, f'Correlation: {corr:.3f}', 
                    transform=ax2.transAxes, fontsize=10,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
        # 3. Cumulative Profit Over Time
        ax3 = plt.subplot(3, 3, 3)
        if self.trading_results is not None:
            cumulative_profit = self.trading_results['profit'].cumsum()
            ax3.plot(cumulative_profit.index, cumulative_profit.values, 'g-', linewidth=2)
            ax3.fill_between(cumulative_profit.index, 0, cumulative_profit.values, alpha=0.3)
            ax3.set_xlabel('Trade Number')
            ax3.set_ylabel('Cumulative Profit')
            ax3.set_title('💰 Cumulative Trading Profit')
            ax3.grid(True, alpha=0.3)
            
            # Add final profit text
            final_profit = cumulative_profit.iloc[-1]
            ax3.text(0.05, 0.95, f'Final P&L: {final_profit:.4f}', 
                    transform=ax3.transAxes, fontsize=12, weight='bold',
                    bbox=dict(boxstyle="round,pad=0.3", 
                             facecolor="green" if final_profit > 0 else "red", 
                             alpha=0.8, color="white"))
        
        # 4. Trading Signal Distribution
        ax4 = plt.subplot(3, 3, 4)
        if self.trading_results is not None:
            signal_counts = self.trading_results['signal'].value_counts()
            colors = ['red' if s == 'SELL' else 'green' if s == 'BUY' else 'gray' 
                     for s in signal_counts.index]
            ax4.pie(signal_counts.values, labels=signal_counts.index, autopct='%1.1f%%',
                   colors=colors)
            ax4.set_title('📊 Trading Signal Distribution')
        
        # 5. Prediction Change Distribution
        ax5 = plt.subplot(3, 3, 5)
        if self.trading_results is not None:
            ax5.hist(self.trading_results['prediction'], bins=30, alpha=0.7, 
                    color='blue', label='Predicted')
            ax5.hist(self.trading_results['actual'], bins=30, alpha=0.7, 
                    color='orange', label='Actual')
            ax5.set_xlabel('Price Change')
            ax5.set_ylabel('Frequency')
            ax5.set_title('📈 Price Change Distribution')
            ax5.legend()
            ax5.grid(True, alpha=0.3)
        
        # 6. Profit per Signal Type
        ax6 = plt.subplot(3, 3, 6)
        if self.trading_results is not None:
            profit_by_signal = self.trading_results.groupby('signal')['profit'].agg(['mean', 'std', 'count'])
            
            x_pos = range(len(profit_by_signal))
            ax6.bar(x_pos, profit_by_signal['mean'], 
                   yerr=profit_by_signal['std'], capsize=5,
                   color=['red' if s == 'SELL' else 'green' if s == 'BUY' else 'gray' 
                         for s in profit_by_signal.index])
            ax6.set_xticks(x_pos)
            ax6.set_xticklabels(profit_by_signal.index)
            ax6.set_ylabel('Average Profit')
            ax6.set_title('💵 Profit by Signal Type')
            ax6.grid(True, alpha=0.3)
            
            # Add count annotations
            for i, count in enumerate(profit_by_signal['count']):
                ax6.text(i, ax6.get_ylim()[1]*0.8, f'n={count}', 
                        ha='center', fontsize=9)
        
        # 7. Time Series Visualization (if we have date info)
        ax7 = plt.subplot(3, 1, 3)
        if self.trading_results is not None and len(self.trading_results) > 0:
            # Create synthetic price series from changes
            initial_price = 100  # Assume starting price of $100
            actual_prices = [initial_price]
            predicted_prices = [initial_price]
            
            for i in range(len(self.trading_results)):
                # Calculate next price from change
                actual_change = self.trading_results.iloc[i]['actual']
                pred_change = self.trading_results.iloc[i]['prediction']
                
                # Convert change to price (assuming change is percentage)
                next_actual_price = actual_prices[-1] * (1 + actual_change)
                next_pred_price = predicted_prices[-1] * (1 + pred_change)
                
                actual_prices.append(next_actual_price)
                predicted_prices.append(next_pred_price)
            
            # Plot price series
            x_range = range(len(actual_prices))
            ax7.plot(x_range, actual_prices, 'b-', linewidth=2, label='Actual Price', alpha=0.8)
            ax7.plot(x_range, predicted_prices, 'r--', linewidth=2, label='Predicted Price', alpha=0.8)
            
            # Add changes as bar chart on secondary y-axis
            ax7_twin = ax7.twinx()
            changes_x = range(1, len(actual_prices))  # Changes start from index 1
            actual_changes = [self.trading_results.iloc[i]['actual'] for i in range(len(self.trading_results))]
            
            # Plot changes as bars
            ax7_twin.bar(changes_x, actual_changes, alpha=0.3, color='green', 
                        label='Actual Changes', width=0.8)
            
            ax7.set_xlabel('Time Steps')
            ax7.set_ylabel('Price ($)', color='b')
            ax7_twin.set_ylabel('Price Change', color='g')
            ax7.set_title('📈 Price Evolution: Actual vs Predicted (with Changes)')
            ax7.legend(loc='upper left')
            ax7_twin.legend(loc='upper right')
            ax7.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_plots:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'tft_analysis_{timestamp}.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"📁 Plots saved to: {filename}")
        
        plt.show()
        
        # Print summary statistics
        self._print_summary_statistics()
    
    def _print_summary_statistics(self):
        """Print comprehensive summary statistics."""
        print(f"\n📊 COMPREHENSIVE ANALYSIS SUMMARY")
        print("=" * 50)
        
        if self.trading_results is not None:
            results = self.trading_results
            
            print(f"📈 PREDICTION PERFORMANCE:")
            mae = np.mean(np.abs(results['prediction'] - results['actual']))
            rmse = np.sqrt(np.mean((results['prediction'] - results['actual'])**2))
            correlation = np.corrcoef(results['prediction'], results['actual'])[0,1]
            
            print(f"   MAE: {mae:.6f}")
            print(f"   RMSE: {rmse:.6f}")
            print(f"   Correlation: {correlation:.4f}")
            print(f"   Direction Accuracy: {results['accuracy'].mean():.2%}")
            
            print(f"\n💰 TRADING PERFORMANCE:")
            total_profit = results['profit'].sum()
            profitable_trades = len(results[results['profit'] > 0])
            total_trades = len(results[results['position'] != 0])
            
            print(f"   Total Profit: {total_profit:.6f}")
            print(f"   Profitable Trades: {profitable_trades}/{total_trades} ({profitable_trades/max(total_trades,1):.1%})")
            print(f"   Max Single Profit: {results['profit'].max():.6f}")
            print(f"   Max Single Loss: {results['profit'].min():.6f}")
            print(f"   Sharpe Ratio: {results['profit'].mean()/results['profit'].std():.4f}")
            
        if self.feature_importance:
            print(f"\n🎯 KEY INSIGHTS:")
            top_features = list(self.feature_importance.keys())[:5]
            print(f"   Top predictive features: {top_features}")
            
            # Analyze future vs historical feature importance
            future_features = ['earnings_in_prediction_window', 'days_to_next_earnings', 
                             'day_of_week', 'eps_estimate']
            future_importance = sum([self.feature_importance.get(f, 0) for f in future_features])
            total_importance = sum(self.feature_importance.values())
            
            print(f"   Future context importance: {future_importance/total_importance:.1%}")


def main():
    """Main execution function for enhanced TFT strategy."""
    print("🚀 ENHANCED TFT STRATEGY WITH PROFIT VISUALIZATION")
    print("=" * 60)
    
    # Configuration
    symbols = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'META']
    start_date = '2024-01-01'  # More recent for news data
    end_date = '2024-06-01'    # More recent for news data
    encoder_len = 60           # Shorter for faster training
    predict_len = 10           # Shorter prediction window
    batch_size = 32            # Smaller batch for stability
    
    # Initialize analyzer
    analyzer = EnhancedTFTAnalyzer(
        symbols=symbols,
        start_date=start_date,
        end_date=end_date,
        encoder_len=encoder_len,
        predict_len=predict_len,
        batch_size=batch_size
    )
    
    try:
        # Step 1: Load data and verify features
        analyzer.load_data_and_verify_features()
        
        # Step 2: Build enhanced model
        analyzer.build_model()
        
        # Step 3: Train with monitoring
        analyzer.train_model(max_epochs=30)
        
        # Step 4: Analyze interpretability
        analyzer.analyze_model_interpretation()
        
        # Step 5: Create trading strategy
        analyzer.create_trading_strategy(threshold=0.01)
        
        # Step 6: Visualize comprehensive results
        analyzer.visualize_results(save_plots=True)
        
        print("\n✅ ENHANCED TFT ANALYSIS COMPLETE!")
        print("   Check the generated plots for detailed insights.")
        
    except Exception as e:
        print(f"\n❌ Error in enhanced TFT analysis: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
