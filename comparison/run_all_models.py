#!/usr/bin/env python3
"""
Main pipeline to run all model comparisons.
"""
import os
import pandas as pd
import numpy as np
import torch
from datetime import datetime
from prettytable import PrettyTable
from dateutil.relativedelta import relativedelta

from main import LeakageFreeDataLoader
from models import LSTMModel, GRUModel, TransformerModel, TFT
from train import train_model, get_predictions, train_tft_model
from model.tft_model import setup_device
from evaluation import evaluate_multi_horizon_predictions, create_horizon_comparison_table
from plotting import (create_results_directory, plot_training_curves, plot_prediction_samples, 
                     plot_horizon_comparison_heatmap, plot_error_distribution,
                     save_all_artifacts, create_comprehensive_plots, save_evaluation_artifacts)
from baseline_models import DeepLearningExperimentRunner
from enhanced_plotting import EnhancedPlottingManager
import warnings
import torch
warnings.filterwarnings('ignore')

def create_empty_detailed_predictions_df():
    """Create an empty detailed predictions DataFrame with proper column structure."""
    return pd.DataFrame(columns=['symbol', 'date', 'actual', 'prediction', 'horizon', 'squared_error', 'absolute_error'])

def get_model_summary(model_name, model_instance):
    """Returns a brief summary of the deep learning model."""
    summaries = {
        "LSTM": "A standard Long Short-Term Memory network, good for capturing sequential patterns.",
        "GRU": "A Gated Recurrent Unit network, similar to LSTM but with a simpler architecture.",
        "Transformer": "A model using self-attention mechanisms to weigh the importance of different past data points.",
        "TFT": "A complex Temporal Fusion Transformer designed to handle diverse features and temporal hierarchies."
    }
    summary = summaries.get(model_name, "A deep learning model for time series prediction.")
    
    if hasattr(model_instance, 'parameters'):
        total_params = sum(p.numel() for p in model_instance.parameters() if p.requires_grad)
        summary += f"\n  - Trainable Parameters: {total_params:,}"
    
    return summary

def filter_features_by_type(X_data, feature_df, filter_type, news_dim=0, debug=False):
    """
    Filter features based on the specified filter type.
    
    Args:
        X_data: Input data to filter
        feature_df: Feature dataframe with column information
        filter_type: Type of filtering to apply
        news_dim: Number of news features
        debug: Whether to print debug information
    
    Returns:
        Filtered X_data and corresponding feature indices
    """
    if feature_df is None:
        if debug:
            print(f"⚠️  No feature information available for {filter_type}, using all features")
        return X_data, list(range(X_data.shape[-1]))
    
    # Get feature column names (excluding target and metadata columns)
    feature_cols = [col for col in feature_df.columns 
                   if col not in ['date', 'symbol', 'target_0', 'target_1', 'target_2', 'target_3', 'target_4',
                                  'target_5', 'target_6', 'target_7', 'target_8', 'target_9']]
    
    # Ensure we don't exceed the actual data dimensions
    actual_feature_count = X_data.shape[-1]
    if len(feature_cols) > actual_feature_count:
        if debug:
            print(f"⚠️  Feature DataFrame has {len(feature_cols)} columns but data has {actual_feature_count} features")
        feature_cols = feature_cols[:actual_feature_count]
    
    if debug:
        print(f"   Total available features in data: {actual_feature_count}")
        print(f"   Feature columns to consider: {len(feature_cols)}")
        
        print(f"\n   📋 FEATURE FILTERING EXPLANATION:")
        print(f"      • Total tensor features: {actual_feature_count}")
        print(f"      • Feature DataFrame columns: {len(feature_cols)} (may include metadata)")
        print(f"      • Filter type: '{filter_type}'")
        print(f"      • Goal: Select subset of features based on data type")
        print()
    
    if filter_type == "no_news":
        # Remove news features (embeddings and sentiment)
        news_features = [col for col in feature_cols 
                        if col.startswith('emb_') or col == 'sentiment_score']
        selected_cols = [col for col in feature_cols 
                        if not (col.startswith('emb_') or col == 'sentiment_score')]
        
        if debug:
            print(f"   📰 NEWS FEATURES IDENTIFIED ({len(news_features)} total):")
            if len(news_features) <= 10:
                print(f"      {news_features}")
            else:
                print(f"      First 5: {news_features[:5]}")
                print(f"      Last 5: {news_features[-5:]}")
                print(f"      (and {len(news_features)-10} more embedding dimensions)")
            
            print(f"   ✅ REMAINING FEATURES ({len(selected_cols)} total):")
            remaining_by_type = {
                'price': [col for col in selected_cols if any(pf in col.lower() for pf in ['open', 'high', 'low', 'close', 'volume', 'adjusted'])],
                'technical': [col for col in selected_cols if col.startswith('ta_') or any(ind in col.lower() for ind in ['sma', 'ema', 'rsi', 'macd', 'bb'])],
                'economic': [col for col in selected_cols if any(econ in col.lower() for econ in ['cpi', 'fedfunds', 'unrate', 't10y2y', 'gdp', 'vix', 'dxy', 'oil'])],
                'other': []
            }
            
            # Categorize remaining features
            categorized = set()
            for category_features in remaining_by_type.values():
                categorized.update(category_features)
            remaining_by_type['other'] = [col for col in selected_cols if col not in categorized]
            
            for category, features in remaining_by_type.items():
                if features:
                    print(f"      {category.capitalize()}: {len(features)} features")
                    if len(features) <= 5:
                        print(f"        → {features}")
                    else:
                        print(f"        → {features[:3]} ... {features[-2:]}")
            
            print(f"   ❌ FILTERED OUT: {len(news_features)} news features removed")
        
    elif filter_type == "no_economic":
        # Remove economic (FRED) features
        economic_features = ['cpi', 'fedfunds', 'unrate', 't10y2y', 'gdp', 'vix', 'dxy', 'oil']
        econ_cols = [col for col in feature_cols 
                    if any(econ_feat in col.lower() for econ_feat in economic_features)]
        selected_cols = [col for col in feature_cols 
                        if not any(econ_feat in col.lower() for econ_feat in economic_features)]
        
        if debug:
            print(f"   🏦 ECONOMIC FEATURES REMOVED ({len(econ_cols)} total): {econ_cols}")
            print(f"   ✅ REMAINING FEATURES: {len(selected_cols)} features")
            print(f"   ❌ FILTERED OUT: {len(econ_cols)} economic features removed")
        
    elif filter_type == "price_only":
        # Only basic price features
        price_features = ['open', 'high', 'low', 'close', 'volume', 'adjusted_close']
        selected_cols = [col for col in feature_cols if any(pf in col.lower() for pf in price_features)]
        excluded_cols = [col for col in feature_cols if col not in selected_cols]
        
        if debug:
            print(f"   💰 PRICE FEATURES SELECTED ({len(selected_cols)} total): {selected_cols}")
            print(f"   ❌ EXCLUDED FEATURES ({len(excluded_cols)} total):")
            excluded_by_type = {
                'news': [col for col in excluded_cols if col.startswith('emb_') or col == 'sentiment_score'],
                'technical': [col for col in excluded_cols if col.startswith('ta_') or any(ind in col.lower() for ind in ['sma', 'ema', 'rsi', 'macd', 'bb'])],
                'economic': [col for col in excluded_cols if any(econ in col.lower() for econ in ['cpi', 'fedfunds', 'unrate', 't10y2y', 'gdp', 'vix', 'dxy', 'oil'])],
            }
            for category, features in excluded_by_type.items():
                if features:
                    print(f"      {category}: {len(features)} features excluded")
        
    elif filter_type == "technical_only":
        # Only technical indicators (these start with 'ta_' based on the codebase)
        selected_cols = [col for col in feature_cols 
                        if col.startswith('ta_') or any(indicator in col.lower() 
                        for indicator in ['sma', 'ema', 'rsi', 'macd', 'bb', 'atr', 'obv'])]
        excluded_cols = [col for col in feature_cols if col not in selected_cols]
        
        if debug:
            print(f"   📊 TECHNICAL FEATURES SELECTED ({len(selected_cols)} total):")
            if len(selected_cols) <= 10:
                print(f"      {selected_cols}")
            else:
                print(f"      {selected_cols[:5]} ... {selected_cols[-5:]}")
            print(f"   ❌ EXCLUDED: {len(excluded_cols)} features (price, news, economic)")
        
    elif filter_type == "price_technical":
        # Price + technical indicators (exclude news and economic features)
        price_features = ['open', 'high', 'low', 'close', 'volume', 'adjusted_close']
        technical_indicators = ['sma', 'ema', 'rsi', 'macd', 'bb', 'atr', 'obv']
        
        selected_cols = [col for col in feature_cols 
                        if (any(pf in col.lower() for pf in price_features) or 
                            col.startswith('ta_') or
                            any(indicator in col.lower() for indicator in technical_indicators)) and
                           not (col.startswith('emb_') or col == 'sentiment_score')]
        
        price_cols = [col for col in selected_cols if any(pf in col.lower() for pf in price_features)]
        tech_cols = [col for col in selected_cols if col not in price_cols]
        excluded_cols = [col for col in feature_cols if col not in selected_cols]
        news_excluded = [col for col in excluded_cols if col.startswith('emb_') or col == 'sentiment_score']
        econ_excluded = [col for col in excluded_cols if any(econ in col.lower() for econ in ['cpi', 'fedfunds', 'unrate', 't10y2y', 'gdp', 'vix', 'dxy', 'oil'])]
        
        if debug:
            print(f"   💰 PRICE FEATURES INCLUDED ({len(price_cols)}): {price_cols}")
            print(f"   📊 TECHNICAL FEATURES INCLUDED ({len(tech_cols)}): {tech_cols[:5]}{'...' if len(tech_cols) > 5 else ''}")
            print(f"   ❌ EXCLUDED - News: {len(news_excluded)}, Economic: {len(econ_excluded)}")
            print(f"   ✅ TOTAL SELECTED: {len(selected_cols)} features")
        
    else:
        # Use all features
        selected_cols = feature_cols
        
        if debug:
            print(f"   ✅ USING ALL FEATURES: {len(selected_cols)} features")
            
            # Show breakdown of all features by type
            all_by_type = {
                'news': [col for col in feature_cols if col.startswith('emb_') or col == 'sentiment_score'],
                'price': [col for col in feature_cols if any(pf in col.lower() for pf in ['open', 'high', 'low', 'close', 'volume', 'adjusted'])],
                'technical': [col for col in feature_cols if col.startswith('ta_') or any(ind in col.lower() for ind in ['sma', 'ema', 'rsi', 'macd', 'bb'])],
                'economic': [col for col in feature_cols if any(econ in col.lower() for econ in ['cpi', 'fedfunds', 'unrate', 't10y2y', 'gdp', 'vix', 'dxy', 'oil'])],
            }
            
            categorized_all = set()
            for category_features in all_by_type.values():
                categorized_all.update(category_features)
            all_by_type['other'] = [col for col in feature_cols if col not in categorized_all]
            
            for category, features in all_by_type.items():
                if features:
                    print(f"      {category.capitalize()}: {len(features)} features")
    
    # Get indices of selected columns, ensuring they're within bounds
    selected_indices = []
    for i, col in enumerate(feature_cols):
        if col in selected_cols and i < actual_feature_count:
            selected_indices.append(i)
    
    # Ensure we have valid indices
    if not selected_indices:
        if debug:
            print(f"⚠️  No valid feature indices found, using first {min(6, actual_feature_count)} features as fallback")
        selected_indices = list(range(min(6, actual_feature_count)))
    
    if debug:
        print(f"   Selected feature indices: {len(selected_indices)} indices")
    
    # Filter the data
    try:
        if len(X_data.shape) == 3:  # (batch, sequence, features)
            filtered_X = X_data[:, :, selected_indices]
        elif len(X_data.shape) == 2:  # (batch, features)
            filtered_X = X_data[:, selected_indices]
        else:
            if debug:
                print(f"⚠️  Unexpected data shape: {X_data.shape}, using original data")
            filtered_X = X_data
            selected_indices = list(range(X_data.shape[-1]))
        
        if debug:
            print(f"   Data shape changed from {X_data.shape} to {filtered_X.shape}")
        return filtered_X, selected_indices
        
    except IndexError as e:
        print(f"⚠️  IndexError during filtering: {e}")
        print(f"   Max index attempted: {max(selected_indices) if selected_indices else 'None'}")
        print(f"   Data shape: {X_data.shape}")
        print(f"   Using original data as fallback")
        return X_data, list(range(X_data.shape[-1]))

def run_pipeline():
    """
    Executes the full model comparison pipeline.
    """
    print("🚀 Starting Full Model Comparison Pipeline...")
    
    # --- 0. Device Setup ---
    print("\n" + "="*50)
    print("DEVICE SETUP")
    print("="*50)
    device = setup_device()
    print(f"Selected device: {device}")
    
    # Enable CUDA optimizations if using CUDA
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True  # Optimize for consistent input sizes
        torch.backends.cudnn.deterministic = False  # Allow non-deterministic for speed
        print("🔧 CUDA optimizations enabled")
        print(f"   - cuDNN benchmark: {torch.backends.cudnn.benchmark}")
        print(f"   - Available GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        # Clear any existing cache
        torch.cuda.empty_cache()
        print("   - GPU cache cleared")
    
    print("="*50 + "\n")
    
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
        'horizons': [1, 5, 10, 15, 20],
        'news_api_key': os.getenv('NEWS_API_KEY'),
        'fred_api_key': os.getenv('FRED_API_KEY'),
        'api_ninjas_key': os.getenv('API_NINJAS_KEY'),
        'device': device,  # Add device to config
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
        # === Neural Network Models (PyTorch) ===
        "LSTM_Full": {
            "type": "pytorch",
            "model": LSTMModel(input_dim=input_dim, hidden_dim=64, num_layers=2, output_dim=config['predict_len']).to(config['device']),
            "description": "LSTM with all available features including news, economic, and technical indicators."
        },
        "LSTM_No_News": {
            "type": "pytorch_no_news",
            "model": None,  # Will be created dynamically with correct input_dim
            "model_class": LSTMModel,
            "model_params": {"hidden_dim": 64, "num_layers": 2, "output_dim": config['predict_len']},
            "description": "LSTM without news features - price, technical, and economic data only."
        },
        "LSTM_Price_Only": {
            "type": "pytorch_price_only",
            "model": None,  # Will be created dynamically with correct input_dim
            "model_class": LSTMModel,
            "model_params": {"hidden_dim": 64, "num_layers": 2, "output_dim": config['predict_len']},
            "description": "LSTM with only basic price features (OHLCV)."
        },
        
        "GRU_Full": {
            "type": "pytorch",
            "model": GRUModel(input_dim=input_dim, hidden_dim=64, num_layers=2, output_dim=config['predict_len']).to(config['device']),
            "description": "GRU with all available features including news, economic, and technical indicators."
        },
        "GRU_No_News": {
            "type": "pytorch_no_news",
            "model": None,  # Will be created dynamically with correct input_dim
            "model_class": GRUModel,
            "model_params": {"hidden_dim": 64, "num_layers": 2, "output_dim": config['predict_len']},
            "description": "GRU without news features - price, technical, and economic data only."
        },
        
        "Transformer_Full": {
            "type": "pytorch",
            "model": TransformerModel(input_dim=input_dim, model_dim=64, num_heads=4, num_layers=2, output_dim=config['predict_len']).to(config['device']),
            "description": "Transformer with all available features for comprehensive time series modeling."
        },
        "Transformer_No_News": {
            "type": "pytorch_no_news",
            "model": None,  # Will be created dynamically with correct input_dim
            "model_class": TransformerModel,
            "model_params": {"model_dim": 64, "num_heads": 4, "num_layers": 2, "output_dim": config['predict_len']},
            "description": "Transformer without news features for baseline comparison."
        },
        
        # === TFT Models with Different Data Combinations ===
        "TFT_with_News": {
            "type": "pytorch_news",
            "model": TFT(input_size=input_dim, news_dim=news_dim, hidden_size=64, num_heads=4, dropout=0.1, prediction_len=config['predict_len']).to(config['device']),
            "description": "Temporal Fusion Transformer with news sentiment data integration."
        },
        "TFT_without_News": {
            "type": "pytorch",
            "model": TFT(input_size=input_dim, news_dim=0, hidden_size=64, num_heads=4, dropout=0.1, prediction_len=config['predict_len']).to(config['device']),
            "description": "TFT baseline without news data for ablation study."
        },
        "TFT_Price_Technical": {
            "type": "pytorch_price_technical",
            "model": None,  # Will be created dynamically with correct input_dim
            "model_class": TFT,
            "model_params": {"news_dim": 0, "hidden_size": 64, "num_heads": 4, "dropout": 0.1, "prediction_len": config['predict_len']},
            "description": "TFT with only price and technical indicators (no news or economic data)."
        }
    }

    # --- 4. Training and Evaluation Loop ---
    # Create results directory for this experiment
    results_dir = create_results_directory()
    
    all_evaluation_results = {}
    all_training_histories = {}
    horizons_to_evaluate = [1, 5, 10, 15, 20]

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
                lr=config['learning_rate'],
                device=config['device']
            )
            all_training_histories[model_name] = history
            
            print(f"\n--- Generating Predictions for {model_name} ---")
            predictions_df = get_predictions(trained_model, data_module.val_loader, val_df)
            
            print(f"\n--- Evaluating Multi-Horizon Performance for {model_name} ---")
            eval_results = evaluate_multi_horizon_predictions(trained_model, data_module, horizons_to_evaluate)
            all_evaluation_results[model_name] = eval_results

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
                lr=config['learning_rate'],
                device=config['device']
            )
            all_training_histories[model_name] = history
            
            print(f"\n--- Generating Predictions for {model_name} ---")
            predictions_df = get_predictions(trained_model, data_module.val_loader, val_df)
            
            print(f"\n--- Evaluating Multi-Horizon Performance for {model_name} ---")
            eval_results = evaluate_multi_horizon_predictions(trained_model, data_module, horizons_to_evaluate)
            all_evaluation_results[model_name] = eval_results

        # === New PyTorch model types with feature filtering ===
        elif model_type in ["pytorch_no_news", "pytorch_price_only", "pytorch_price_technical"]:
            print(f"\n--- Training {model_name} (PyTorch with filtered features) ---")
            
            # Get feature information for filtering
            feature_df = data_loader.processed_data.get('features', None)
            
            # Determine filter type
            if "no_news" in model_type:
                filter_type = "no_news"
            elif "price_only" in model_type:
                filter_type = "price_only"
            elif "price_technical" in model_type:
                filter_type = "price_technical"
            else:
                filter_type = "all"
            
            print(f"   Applying feature filter: {filter_type}")
            
            # Create filtered data loaders
            from torch.utils.data import DataLoader, TensorDataset
            
            # Get a sample batch to determine filtered dimensions
            sample_batch = next(iter(data_module.train_loader))
            sample_features, _ = filter_features_by_type(sample_batch[0].numpy(), feature_df, filter_type, news_dim, debug=True)
            filtered_input_dim = sample_features.shape[-1]
            
            print(f"   Original input dim: {input_dim}, Filtered input dim: {filtered_input_dim}")
            
            # Create the model with correct input dimension
            if model_info['model'] is None:
                model_class = model_info['model_class']
                model_params = model_info['model_params'].copy()
                model_params['input_dim'] = filtered_input_dim
                
                # Special handling for TFT models
                if model_class == TFT:
                    model_params['input_size'] = filtered_input_dim
                    if 'input_dim' in model_params:
                        del model_params['input_dim']
                
                model = model_class(**model_params).to(config['device'])
                print(f"   Created {model_class.__name__} with input_dim={filtered_input_dim} on {config['device']}")
            else:
                model = model_info['model']
            
            # Get raw data and apply filtering
            train_features_list, train_targets_list = [], []
            for features, targets in data_module.train_loader:
                filtered_features, _ = filter_features_by_type(features.numpy(), feature_df, filter_type, news_dim, debug=False)
                train_features_list.append(torch.FloatTensor(filtered_features).to(config['device']))
                train_targets_list.append(targets.to(config['device']))
            
            val_features_list, val_targets_list = [], []
            for features, targets in data_module.val_loader:
                filtered_features, _ = filter_features_by_type(features.numpy(), feature_df, filter_type, news_dim, debug=False)
                val_features_list.append(torch.FloatTensor(filtered_features).to(config['device']))
                val_targets_list.append(targets.to(config['device']))
            
            # Create new data loaders with filtered features
            train_features_tensor = torch.cat(train_features_list, dim=0)
            train_targets_tensor = torch.cat(train_targets_list, dim=0)
            val_features_tensor = torch.cat(val_features_list, dim=0)
            val_targets_tensor = torch.cat(val_targets_list, dim=0)
            
            train_dataset = TensorDataset(train_features_tensor, train_targets_tensor)
            val_dataset = TensorDataset(val_features_tensor, val_targets_tensor)
            
            filtered_train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
            filtered_val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)
            
            # Create a filtered data module
            class FilteredDataModule:
                def __init__(self, train_loader, val_loader):
                    self.train_loader = train_loader
                    self.val_loader = val_loader
            
            filtered_data_module = FilteredDataModule(filtered_train_loader, filtered_val_loader)
            
            # Train model with filtered data
            trained_model, history = train_model(
                model, 
                filtered_data_module, 
                epochs=config['epochs'], 
                lr=config['learning_rate'],
                device=config['device']
            )
            all_training_histories[model_name] = history
            
            print(f"\n--- Generating Predictions for {model_name} ---")
            predictions_df = get_predictions(trained_model, filtered_data_module.val_loader, val_df)
            
            print(f"\n--- Evaluating Multi-Horizon Performance for {model_name} ---")
            eval_results = evaluate_multi_horizon_predictions(trained_model, filtered_data_module, horizons_to_evaluate)
            all_evaluation_results[model_name] = eval_results



        # === Unknown model type ===
        else:
            print(f"❌ Unknown model type: {model_type}")
            continue
        
        # Ensure predictions_df is always defined
        if 'predictions_df' not in locals():
            print(f"   Warning: No predictions generated for {model_name}")
            predictions_df = pd.DataFrame({
                'symbol': [config['symbols'][0]], 
                'prediction': [np.zeros(config['predict_len']).tolist()],
                'date': [pd.to_datetime(config['start_date'])]
            })

        print(f"✅ {model_name} evaluation complete.")
        
        # Clean up GPU memory after each model if using CUDA
        if config['device'].type == 'cuda':
            torch.cuda.empty_cache()
            print(f"   🧹 GPU cache cleared after {model_name}")

    # --- 5. Results Summary and Visualization ---
    print("\n\n" + "="*80)
    print("📊 MULTI-HORIZON PREDICTION EVALUATION")
    print("="*80)
    
    # Print horizon comparison table
    horizon_table_str = ""
    if all_evaluation_results:
        horizon_table_str = create_horizon_comparison_table(all_evaluation_results, horizons_to_evaluate)
        print(horizon_table_str)
    else:
        print("No evaluation results available for horizon analysis.")
    
    # --- 6. Generate All Plots and Save Artifacts ---
    print("\n\n" + "="*80)
    print("📈 GENERATING COMPREHENSIVE VISUALIZATIONS AND SAVING ARTIFACTS")
    print("="*80)
    
    # Generate comprehensive plots with enhanced visualizations
    create_comprehensive_plots({}, all_evaluation_results, all_training_histories, results_dir)
    
    # Generate legacy plots for compatibility
    plot_training_curves(all_training_histories, results_dir)
    plot_prediction_samples(all_evaluation_results, results_dir)
    plot_horizon_comparison_heatmap(all_evaluation_results, horizons_to_evaluate, results_dir)
    plot_error_distribution(all_evaluation_results, results_dir)
    
    # Save evaluation artifacts with detailed tables
    save_evaluation_artifacts(all_evaluation_results, {}, horizons_to_evaluate, results_dir)
    
    # Save all artifacts
    save_all_artifacts(all_evaluation_results, {}, horizon_table_str, 
                      "", config, results_dir)
    
    print(f"\n🎉 Pipeline completed successfully!")
    print(f"📁 All results saved to: {results_dir}")

if __name__ == '__main__':
    # Ensure you have a .env file with your API keys, e.g.:
    # NEWS_API_KEY="YOUR_KEY"
    # FRED_API_KEY="YOUR_KEY"
    # API_NINJAS_KEY="YOUR_KEY"
    from dotenv import load_dotenv
    load_dotenv()
    
    run_pipeline()
