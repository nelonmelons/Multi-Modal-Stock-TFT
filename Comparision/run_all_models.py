#!/usr/bin/env python3
"""
Main pipeline to run all model comparisons.
"""
import os
import pandas as pd
import numpy as np
from datetime import datetime
from prettytable import PrettyTable
from dateutil.relativedelta import relativedelta

from main import LeakageFreeDataLoader
from models import LSTMModel, GRUModel, TransformerModel, TFT
from train import train_model, get_predictions, train_tft_model
from portfolio import simulate_portfolio
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import xgboost as xgb
import lightgbm as lgb
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tools.sm_exceptions import ConvergenceWarning
import warnings
import xgboost as xgb
import lightgbm as lgb
from statsmodels.tsa.arima.model import ARIMA
import warnings
warnings.filterwarnings('ignore')

def get_model_summary(model_name, model_instance):
    """Returns a brief summary of the model."""
    summaries = {
        "LSTM": "A standard Long Short-Term Memory network, good for capturing sequential patterns.",
        "GRU": "A Gated Recurrent Unit network, similar to LSTM but with a simpler architecture.",
        "Transformer": "A model using "
        "self-attention mechanisms to weigh the importance of different past data points.",
        "TFT": "A complex Temporal Fusion Transformer designed to handle diverse features and temporal hierarchies.",
        "Ridge": "A classical linear regression model with L2 regularization to prevent overfitting."
    }
    summary = summaries.get(model_name, "A machine learning model.")
    
    if hasattr(model_instance, 'parameters'):
        total_params = sum(p.numel() for p in model_instance.parameters() if p.requires_grad)
        summary += f"\n  - Trainable Parameters: {total_params:,}"
    
    return summary

def run_pipeline():
    """
    Executes the full model comparison pipeline.
    """
    print("🚀 Starting Full Model Comparison Pipeline...")
    
    # --- 1. Configuration ---
    # Set dates to most recent 6 months
    from dateutil.relativedelta import relativedelta
    
    end_date = datetime.now()
    start_date = end_date - relativedelta(months=6)
    
    config = {
        'symbols': ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA', 'AMD', 'QCOM', 'INTC'],
        'start_date': start_date.strftime('%Y-%m-%d'),
        'end_date': end_date.strftime('%Y-%m-%d'),
        'encoder_len': 90,
        'predict_len': 10,
        'batch_size': 32,
        'validation_split': 0.8,
        'lookahead_buffer': 10,
        'epochs': 10,
        'learning_rate': 0.001,
        'news_api_key': os.getenv('NEWS_API_KEY'),
        'fred_api_key': os.getenv('FRED_API_KEY'),
        'api_ninjas_key': os.getenv('API_NINJAS_KEY'),
    }
    print("\n" + "="*50)
    print("CONFIGURATION")
    print("="*50)
    for key, val in config.items():
        print(f"{key:20}: {val}")
    print("="*50 + "\n")

    # --- 2. Data Loading ---
    print("🔄 Loading and Processing Data...")
    data_loader = LeakageFreeDataLoader(config)
    data_module = data_loader.load_complete_pipeline()
    
    if not data_module:
        print("❌ Data loading failed. Aborting pipeline.")
        return

    val_df = data_loader.val_df
    
    # Prepare data for sklearn models
    train_loader = data_module.train_loader
    val_loader = data_module.val_loader
    
    X_train_list, y_train_list = [], []
    for features, targets in train_loader:
        X_train_list.append(features.numpy())
        y_train_list.append(targets.numpy())

    X_val_list, y_val_list = [], []
    for features, targets in val_loader:
        X_val_list.append(features.numpy())
        y_val_list.append(targets.numpy())

    X_train = np.vstack(X_train_list)
    y_train = np.vstack(y_train_list)
    X_val = np.vstack(X_val_list)
    y_val = np.vstack(y_val_list)

    # Flatten for sklearn
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    y_train_flat = y_train[:, 0] # Predict first step
    X_val_flat = X_val.reshape(X_val.shape[0], -1)

    # --- 3. Model Definitions ---
    # Get input dimension from the first batch
    sample_batch = next(iter(train_loader))
    input_dim = sample_batch[0].shape[-1]  # Last dimension is feature dimension
    
    # Determine news dimension from actual data
    news_dim = 0
    if hasattr(data_loader, 'raw_data') and 'news' in data_loader.raw_data:
        news_data = data_loader.raw_data['news']
        if isinstance(news_data, pd.DataFrame):
            numeric_cols = news_data.select_dtypes(include=[np.number]).columns
            news_dim = len(numeric_cols)
        elif hasattr(news_data, 'shape'):
            news_dim = news_data.shape[1]
    
    print(f"   Input dimension: {input_dim}, News dimension: {news_dim}")
    
    models_to_run = {
        "LSTM": {
            "type": "pytorch",
            "model": LSTMModel(input_dim=input_dim, hidden_dim=64, num_layers=2, output_dim=config['predict_len']),
            "description": "A standard Long Short-Term Memory network, good for capturing sequential patterns."
        },
        "GRU": {
            "type": "pytorch",
            "model": GRUModel(input_dim=input_dim, hidden_dim=64, num_layers=2, output_dim=config['predict_len']),
            "description": "A Gated Recurrent Unit network, similar to LSTM but with a simpler architecture."
        },
        "Transformer": {
            "type": "pytorch",
            "model": TransformerModel(input_dim=input_dim, model_dim=64, num_heads=4, num_layers=2, output_dim=config['predict_len']),
            "description": "A model using self-attention mechanisms to weigh the importance of different past data points."
        },
        "TFT_with_News": {
            "type": "pytorch_news",
            "model": TFT(input_size=input_dim, news_dim=news_dim, hidden_size=64, num_heads=4, dropout=0.1, prediction_len=config['predict_len']),
            "description": "A Temporal Fusion Transformer that incorporates news sentiment data for enhanced stock prediction."
        },
        "TFT_without_News": {
            "type": "pytorch",
            "model": TFT(input_size=input_dim, news_dim=0, hidden_size=64, num_heads=4, dropout=0.1, prediction_len=config['predict_len']),
            "description": "A Temporal Fusion Transformer without news data for baseline comparison."
        },
        "Ridge": {
            "type": "sklearn",
            "model": Pipeline([('scaler', StandardScaler()), ('ridge', Ridge(alpha=1.0))]),
            "description": "A classical linear regression model with L2 regularization to prevent overfitting."
        },
        "XGBoost": {
            "type": "sklearn",
            "model": xgb.XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42),
            "description": "Extreme Gradient Boosting - an optimized gradient boosting framework designed for speed and performance."
        },
        "LightGBM": {
            "type": "sklearn", 
            "model": lgb.LGBMRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42, verbose=-1),
            "description": "Light Gradient Boosting Machine - a fast, distributed, high performance gradient boosting framework."
        },
        "ARIMA": {
            "type": "arima",
            "model": None,  # ARIMA models will be created per symbol
            "description": "AutoRegressive Integrated Moving Average - a classical time series forecasting method."
        }
    }

    # --- 4. Training and Evaluation Loop ---
    all_results = {}

    for model_name, model_info in models_to_run.items():
        print("\n" + "="*80)
        print(f" M O D E L :   {model_name.upper()}")
        print("="*80)
        print(model_info['description'])
        
        model_type = model_info['type']
        model = model_info['model']
        predictions_df = None  # Initialize predictions_df
        
        if model_type == "pytorch":
            print(f"\n--- Training {model_name} (PyTorch) ---")
            trained_model, history = train_model(
                model, 
                data_module, 
                epochs=config['epochs'], 
                lr=config['learning_rate']
            )
            
            print(f"\n--- Generating Predictions for {model_name} ---")
            predictions_df = get_predictions(trained_model, data_module.val_loader)

        elif model_type == "pytorch_news":
            print(f"\n--- Training {model_name} (PyTorch with News Data) ---")
            # For TFT model, we need to create a special training function that handles news data
            # Try to get news data from the data loader
            news_data = None
            if hasattr(data_loader, 'raw_data') and 'news' in data_loader.raw_data:
                news_data = data_loader.raw_data['news']
                print(f"   Found news data with shape: {news_data.shape}")
            else:
                print("   No separate news data found, using embedded features")
            
            # Use specialized TFT training function with news data
            trained_model, history = train_tft_model(
                model, 
                data_module,
                news_data=news_data,
                epochs=config['epochs'], 
                lr=config['learning_rate']
            )
            
            print(f"\n--- Generating Predictions for {model_name} ---")
            predictions_df = get_predictions(trained_model, data_module.val_loader)

        elif model_type == "sklearn":
            print(f"\n--- Training {model_name} (Scikit-learn) ---")
            model.fit(X_train_flat, y_train_flat)
            
            print(f"\n--- Generating Predictions for {model_name} ---")
            predictions = model.predict(X_val_flat)
            
            # Reshape predictions to match portfolio simulation expectations
            # We only predict the first step, so we'll repeat it for the horizon
            predictions_multi_step = np.tile(predictions[:, np.newaxis], (1, config['predict_len']))
            
            # Create predictions dataframe using validation data structure
            # Since we flattened the data, we need to reconstruct the structure
            # For now, create a simple structure assuming sequential validation data
            val_size = len(predictions)
            predictions_df = pd.DataFrame({
                'prediction': list(predictions_multi_step)
            })
            # Add basic indexing - this is a simplified approach
            predictions_df['date'] = pd.date_range(start=config['start_date'], periods=val_size, freq='D')
            predictions_df['symbol'] = config['symbols'][0]  # Default to first symbol for now

        elif model_type == "arima":
            print(f"\n--- Training {model_name} (ARIMA) ---")
            # ARIMA requires time series data, we'll use the first target column
            predictions_list = []
            
            # Get validation data for ARIMA
            val_df_for_arima = data_loader.val_df if hasattr(data_loader, 'val_df') else None
            
            if val_df_for_arima is not None:
                for symbol in config['symbols']:
                    symbol_data = val_df_for_arima[val_df_for_arima['symbol'] == symbol]
                    if len(symbol_data) > 0:
                        # Use closing price for ARIMA
                        ts_data = symbol_data['close'].values
                        
                        if len(ts_data) >= 10:  # Need minimum data for ARIMA
                            try:
                                # Fit ARIMA model
                                arima_model = ARIMA(ts_data[:len(ts_data)//2], order=(1,1,1))
                                fitted_model = arima_model.fit()
                                
                                # Generate predictions
                                forecast = fitted_model.forecast(steps=len(ts_data)//2)
                                
                                # Create predictions in required format
                                for i, pred in enumerate(forecast):
                                    pred_array = np.full(config['predict_len'], pred)
                                    predictions_list.append({
                                        'symbol': symbol,
                                        'prediction': pred_array.tolist()
                                    })
                            except Exception as e:
                                print(f"   ARIMA failed for {symbol}: {e}")
                                # Add dummy predictions if ARIMA fails
                                dummy_pred = np.zeros(config['predict_len'])
                                predictions_list.append({
                                    'symbol': symbol,
                                    'prediction': dummy_pred.tolist()
                                })
            
            if predictions_list:
                predictions_df = pd.DataFrame(predictions_list)
                predictions_df['date'] = pd.date_range(start=config['start_date'], periods=len(predictions_df), freq='D')
            else:
                # Create empty predictions if no data
                predictions_df = pd.DataFrame({
                    'symbol': [config['symbols'][0]], 
                    'prediction': [np.zeros(config['predict_len']).tolist()],
                    'date': [pd.to_datetime(config['start_date'])]
                })
                
            print(f"   ARIMA predictions generated for {len(predictions_list)} data points")
        
        # Ensure predictions_df is always defined
        if 'predictions_df' not in locals():
            print(f"   Warning: No predictions generated for {model_name}")
            predictions_df = pd.DataFrame({
                'symbol': [config['symbols'][0]], 
                'prediction': [np.zeros(config['predict_len']).tolist()],
                'date': [pd.to_datetime(config['start_date'])]
            })

        print(f"\n--- Simulating Portfolio for {model_name} ---")
        if predictions_df is not None:
            portfolio_results = simulate_portfolio(predictions_df, val_df)
            all_results[model_name] = portfolio_results
            print(f"✅ {model_name} evaluation complete.")
        else:
            print(f"❌ No predictions generated for {model_name}")
            all_results[model_name] = {
                'final_capital': 10000,
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0,
                'win_rate': 0.0,
                'avg_gain': 0.0,
                'avg_loss': 0.0
            }

    # --- 5. Results Summary ---
    print("\n\n" + "="*80)
    print("🏆 FINAL MODEL COMPARISON RESULTS")
    print("="*80)

    table = PrettyTable()
    table.field_names = [
        "Model", "Final Capital", "Sharpe Ratio", "Max Drawdown", 
        "Win Rate (%)", "Avg Gain (%)", "Avg Loss (%)"
    ]
    # Set alignment for all columns to right
    for field in table.field_names:
        table.align[field] = "r"
    # Set model column to left alignment
    table.align["Model"] = "l"
    table.float_format = ".4"

    for model_name, results in all_results.items():
        table.add_row([
            model_name,
            f"${results['final_capital']:,.2f}",
            results['sharpe_ratio'],
            f"{results['max_drawdown']:.2%}",
            f"{results['win_rate'] * 100:.2f}",
            f"{results['avg_gain'] * 100:.2f}",
            f"{results['avg_loss'] * 100:.2f}"
        ])
    
    print(table)
    print("\nPipeline finished successfully!")

if __name__ == '__main__':
    # Ensure you have a .env file with your API keys, e.g.:
    # NEWS_API_KEY="YOUR_KEY"
    # FRED_API_KEY="YOUR_KEY"
    # API_NINJAS_KEY="YOUR_KEY"
    from dotenv import load_dotenv
    load_dotenv()
    
    run_pipeline()
