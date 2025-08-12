#!/usr/bin/env python3
"""
Model Parameter and Complexity Analysis
Counts parameters for PyTorch models and complexity metrics for classical ML models.
"""
import os
import sys
import torch
import numpy as np
import pandas as pd
from prettytable import PrettyTable
import warnings
warnings.filterwarnings('ignore')

# Add the current directory to the path to import local modules
sys.path.append('/Users/haysoncheung/programs/pythonProject/TFT-b-nelson/Comparision')

# Import PyTorch models
from models import LSTMModel, GRUModel, TransformerModel, TFT

def count_pytorch_parameters(model):
    """Count trainable parameters in a PyTorch model."""
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    
    return {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'non_trainable_params': non_trainable_params,
        'memory_mb': (total_params * 4) / (1024 * 1024)  # Assuming float32, 4 bytes per parameter
    }

def get_sklearn_complexity(model, model_name, input_features=818):
    """Get complexity metrics for scikit-learn models."""
    complexity_info = {
        'complexity_metric': 'Unknown',
        'complexity_value': 0,
        'memory_estimate_mb': 0,
        'description': ''
    }
    
    if 'Ridge' in model_name or 'Lasso' in model_name or 'ElasticNet' in model_name:
        # Linear models: complexity is proportional to number of features
        complexity_info.update({
            'complexity_metric': 'Features × Outputs',
            'complexity_value': input_features * 1,  # Single output
            'memory_estimate_mb': (input_features * 8) / (1024 * 1024),  # 8 bytes per coefficient (float64)
            'description': f'Linear model with {input_features} coefficients'
        })
    
    elif 'RandomForest' in model_name:
        # Random Forest: complexity based on number of trees and max depth
        if hasattr(model, 'n_estimators'):
            n_trees = model.n_estimators
            max_depth = getattr(model, 'max_depth', 10)
            # Rough estimate: each tree can have up to 2^max_depth nodes
            nodes_per_tree = min(2**max_depth, input_features * 2)  # Cap at reasonable size
            total_nodes = n_trees * nodes_per_tree
            
            complexity_info.update({
                'complexity_metric': 'Trees × Nodes',
                'complexity_value': total_nodes,
                'memory_estimate_mb': (total_nodes * 32) / (1024 * 1024),  # ~32 bytes per node
                'description': f'{n_trees} trees, max depth {max_depth}'
            })
        else:
            complexity_info.update({
                'complexity_metric': 'Trees × Nodes',
                'complexity_value': 100 * 1024,  # Default estimate
                'memory_estimate_mb': 10,
                'description': '~100 trees (default config)'
            })
    
    elif 'XGBoost' in model_name or 'LightGBM' in model_name or 'GradientBoosting' in model_name:
        # Gradient boosting: similar to random forest but sequential
        if hasattr(model, 'n_estimators'):
            n_estimators = model.n_estimators
            max_depth = getattr(model, 'max_depth', 6)
            nodes_per_tree = min(2**max_depth, input_features)
            total_nodes = n_estimators * nodes_per_tree
            
            complexity_info.update({
                'complexity_metric': 'Boosting Rounds × Nodes',
                'complexity_value': total_nodes,
                'memory_estimate_mb': (total_nodes * 24) / (1024 * 1024),  # ~24 bytes per node
                'description': f'{n_estimators} rounds, max depth {max_depth}'
            })
        else:
            complexity_info.update({
                'complexity_metric': 'Boosting Rounds × Nodes',
                'complexity_value': 100 * 64,  # Default estimate
                'memory_estimate_mb': 5,
                'description': '~100 rounds (default config)'
            })
    
    elif 'SVR' in model_name:
        # SVR: complexity related to number of support vectors (roughly proportional to training samples)
        # For simplicity, assume ~10% of training data becomes support vectors
        estimated_support_vectors = max(100, input_features // 10)  # Rough estimate
        
        complexity_info.update({
            'complexity_metric': 'Support Vectors',
            'complexity_value': estimated_support_vectors,
            'memory_estimate_mb': (estimated_support_vectors * input_features * 8) / (1024 * 1024),
            'description': f'~{estimated_support_vectors} support vectors (estimated)'
        })
    
    elif 'MLP' in model_name:
        # MLP: similar to neural networks, count weights
        if hasattr(model, 'named_steps') and 'mlp' in model.named_steps:
            mlp = model.named_steps['mlp']
            hidden_layers = getattr(mlp, 'hidden_layer_sizes', (100, 50))
            if isinstance(hidden_layers, int):
                hidden_layers = (hidden_layers,)
            
            # Calculate total weights: input -> hidden1 -> hidden2 -> ... -> output
            layer_sizes = [input_features] + list(hidden_layers) + [1]  # 1 output
            total_weights = sum(layer_sizes[i] * layer_sizes[i+1] for i in range(len(layer_sizes)-1))
            total_biases = sum(layer_sizes[1:])  # One bias per neuron (except input)
            total_params = total_weights + total_biases
            
            complexity_info.update({
                'complexity_metric': 'Weights + Biases',
                'complexity_value': total_params,
                'memory_estimate_mb': (total_params * 8) / (1024 * 1024),  # 8 bytes per parameter
                'description': f'Hidden layers: {hidden_layers}'
            })
        else:
            complexity_info.update({
                'complexity_metric': 'Weights + Biases',
                'complexity_value': input_features * 100 + 100 * 50 + 50,  # Default (100, 50) architecture
                'memory_estimate_mb': 5,
                'description': 'Hidden layers: (100, 50) (estimated)'
            })
    

    
    return complexity_info

def analyze_model_complexity():
    """Analyze complexity of all models used in the comparison pipeline."""
    
    print("🔍 MODEL PARAMETER & COMPLEXITY ANALYSIS")
    print("=" * 80)
    print("Analyzing all models from the comparison pipeline...")
    print()
    
    # Configuration matching the main pipeline
    input_dim_full = 818  # Full feature set
    input_dim_no_news = 49  # No news features
    input_dim_price_only = 7  # Price only
    input_dim_price_technical = 36  # Price + technical
    input_dim_technical_only = 30  # Technical only
    
    predict_len = 10
    news_dim = 769
    
    results = []
    
    print("🧠 PYTORCH NEURAL NETWORK MODELS")
    print("-" * 50)
    
    # === PyTorch Models ===
    pytorch_models = [
        # Full feature models
        ("LSTM_Full", LSTMModel(input_dim=input_dim_full, hidden_dim=64, num_layers=2, output_dim=predict_len)),
        ("GRU_Full", GRUModel(input_dim=input_dim_full, hidden_dim=64, num_layers=2, output_dim=predict_len)),
        ("Transformer_Full", TransformerModel(input_dim=input_dim_full, model_dim=64, num_heads=4, num_layers=2, output_dim=predict_len)),
        
        # No news models
        ("LSTM_No_News", LSTMModel(input_dim=input_dim_no_news, hidden_dim=64, num_layers=2, output_dim=predict_len)),
        ("GRU_No_News", GRUModel(input_dim=input_dim_no_news, hidden_dim=64, num_layers=2, output_dim=predict_len)),
        ("Transformer_No_News", TransformerModel(input_dim=input_dim_no_news, model_dim=64, num_heads=4, num_layers=2, output_dim=predict_len)),
        
        # Price only models
        ("LSTM_Price_Only", LSTMModel(input_dim=input_dim_price_only, hidden_dim=64, num_layers=2, output_dim=predict_len)),
        
        # TFT models
        ("TFT_with_News", TFT(input_size=input_dim_full, news_dim=news_dim, hidden_size=64, num_heads=4, dropout=0.1, prediction_len=predict_len)),
        ("TFT_without_News", TFT(input_size=input_dim_full, news_dim=0, hidden_size=64, num_heads=4, dropout=0.1, prediction_len=predict_len)),
        ("TFT_Price_Technical", TFT(input_size=input_dim_price_technical, news_dim=0, hidden_size=64, num_heads=4, dropout=0.1, prediction_len=predict_len)),
    ]
    
    for model_name, model in pytorch_models:
        try:
            param_info = count_pytorch_parameters(model)
            
            # Determine input features for this model
            if "Full" in model_name:
                input_features = input_dim_full
            elif "No_News" in model_name:
                input_features = input_dim_no_news
            elif "Price_Only" in model_name:
                input_features = input_dim_price_only
            elif "Price_Technical" in model_name:
                input_features = input_dim_price_technical
            else:
                input_features = input_dim_full
            
            results.append({
                'Model': model_name,
                'Type': 'PyTorch Neural Network',
                'Input Features': input_features,
                'Complexity Metric': 'Trainable Parameters',
                'Complexity Value': param_info['trainable_params'],
                'Memory (MB)': param_info['memory_mb'],
                'Description': f"{param_info['trainable_params']:,} trainable parameters"
            })
            
            print(f"✅ {model_name:25} | {param_info['trainable_params']:>12,} params | {param_info['memory_mb']:>8.2f} MB")
            
        except Exception as e:
            print(f"❌ {model_name:25} | Error: {e}")
            results.append({
                'Model': model_name,
                'Type': 'PyTorch Neural Network',
                'Input Features': 'Error',
                'Complexity Metric': 'Error',
                'Complexity Value': 0,
                'Memory (MB)': 0,
                'Description': f"Error: {e}"
            })
    
    print("\n COMPREHENSIVE MODEL COMPLEXITY SUMMARY")
    print("=" * 100)
    
    # Create detailed table
    table = PrettyTable()
    table.field_names = ["Model", "Type", "Input Features", "Complexity Metric", "Complexity Value", "Memory (MB)", "Description"]
    table.align["Model"] = "l"
    table.align["Type"] = "l" 
    table.align["Complexity Metric"] = "l"
    table.align["Description"] = "l"
    table.align["Input Features"] = "r"
    table.align["Complexity Value"] = "r"
    table.align["Memory (MB)"] = "r"
    
    for result in results:
        table.add_row([
            result['Model'],
            result['Type'],
            result['Input Features'],
            result['Complexity Metric'],
            f"{result['Complexity Value']:,}",
            f"{result['Memory (MB)']:.3f}",
            result['Description'][:50] + "..." if len(result['Description']) > 50 else result['Description']
        ])
    
    print(table)
    
    # === Analysis by Model Type ===
    print("\n\n" + "=" * 80)
    print("📊 ANALYSIS BY MODEL TYPE")
    print("=" * 80)
    
    df = pd.DataFrame(results)
    
    # Group by model type
    for model_type in df['Type'].unique():
        type_data = df[df['Type'] == model_type]
        print(f"\n{model_type.upper()}:")
        print(f"   Models: {len(type_data)}")
        print(f"   Avg Complexity: {type_data['Complexity Value'].mean():,.0f}")
        print(f"   Avg Memory: {type_data['Memory (MB)'].mean():.2f} MB")
        print(f"   Total Memory: {type_data['Memory (MB)'].sum():.2f} MB")
        
        # Show complexity range
        min_complexity = type_data['Complexity Value'].min()
        max_complexity = type_data['Complexity Value'].max()
        print(f"   Complexity Range: {min_complexity:,} to {max_complexity:,}")
    
    # === Feature Impact Analysis ===
    print("\n\n" + "=" * 80)
    print("🎯 FEATURE IMPACT ON MODEL COMPLEXITY")
    print("=" * 80)
    
    # Group models by feature set
    feature_groups = {
        'Full Features (818)': df[df['Input Features'] == input_dim_full],
        'No News (49)': df[df['Input Features'] == input_dim_no_news],
        'Price Only (7)': df[df['Input Features'] == input_dim_price_only],
        'Price + Technical (36)': df[df['Input Features'] == input_dim_price_technical],
        'Technical Only (30)': df[df['Input Features'] == input_dim_technical_only],
    }
    
    for group_name, group_data in feature_groups.items():
        if len(group_data) > 0:
            print(f"\n{group_name}:")
            pytorch_models = group_data[group_data['Type'] == 'PyTorch Neural Network']
            if len(pytorch_models) > 0:
                print(f"   PyTorch Models: {len(pytorch_models)} | Avg Params: {pytorch_models['Complexity Value'].mean():,.0f}")
            
            classical_models = group_data[group_data['Type'] == 'Classical ML']
            if len(classical_models) > 0:
                print(f"   Classical Models: {len(classical_models)} | Avg Complexity: {classical_models['Complexity Value'].mean():,.0f}")
    
    # === Save Results ===
    print("\n\n" + "=" * 80)
    print("💾 PROCESSING RESULTS")
    print("=" * 80)
    
    # CSV output removed - complexity analysis DataFrame created but not saved to CSV
    print(f"✅ Model complexity analysis DataFrame created with {len(df)} rows")
    
    # Save summary statistics
    summary_stats = {
        'total_models': len(results),
        'pytorch_models': len(df[df['Type'] == 'PyTorch Neural Network']),
        'classical_models': len(df[df['Type'] == 'Classical ML']),
        'time_series_models': len(df[df['Type'] == 'Time Series']),
        'total_memory_mb': df['Memory (MB)'].sum(),
        'avg_complexity': df['Complexity Value'].mean(),
        'max_complexity': df['Complexity Value'].max(),
        'min_complexity': df['Complexity Value'].min()
    }
    
    print(f"\n📈 SUMMARY STATISTICS:")
    for key, value in summary_stats.items():
        if isinstance(value, float):
            print(f"   {key.replace('_', ' ').title()}: {value:,.2f}")
        else:
            print(f"   {key.replace('_', ' ').title()}: {value:,}")
    
    print(f"\n🎉 Model complexity analysis complete!")
    print(f"   Total models analyzed: {len(results)}")
    print(f"   Memory footprint range: {df['Memory (MB)'].min():.3f} MB to {df['Memory (MB)'].max():.1f} MB")
    print(f"   Complexity range: {df['Complexity Value'].min():,} to {df['Complexity Value'].max():,}")

if __name__ == '__main__':
    analyze_model_complexity()
