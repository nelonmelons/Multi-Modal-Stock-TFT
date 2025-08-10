#!/usr/bin/env python3
"""
Deep learning models for multi-modal stock prediction.

This module focuses on deep learning approaches for stock prediction
using various neural network architectures and data combinations.
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

class DeepLearningModelFactory:
    """Factory for creating deep learning models - traditional ML models removed."""
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        
    def create_model(self, model_name: str, **kwargs):
        """Create a model instance based on the model name. Traditional ML models removed."""
        # Traditional ML models have been removed from this pipeline
        # Only deep learning models (LSTM, GRU, Transformer, TFT) are supported
        # These are defined in models.py and train.py
        
        available_models = [
            'LSTM', 'GRU', 'Transformer', 'TFT'
        ]
        
        raise ValueError(f"Traditional ML models removed. Use deep learning models: {available_models}. "
                        f"These are implemented in models.py and trained via train.py")

class DataCombinationManager:
    """Manages different feature combinations for ablation studies."""
    
    def __init__(self):
        self.feature_groups = {
            'price': ['open', 'high', 'low', 'close', 'volume', 'adjusted_close'],
            'technical': [col for col in [] if 'ta_' in col],  # Will be populated dynamically
            'news': [col for col in [] if 'news_' in col],     # Will be populated dynamically
            'economic': [col for col in [] if 'fred_' in col], # Will be populated dynamically
            'events': [col for col in [] if 'event_' in col],  # Will be populated dynamically
            'temporal': ['day_of_week', 'month', 'quarter', 'is_month_end', 'is_quarter_end']
        }
        
    def identify_feature_groups(self, feature_df: pd.DataFrame):
        """Identify feature groups from the dataframe columns."""
        columns = feature_df.columns.tolist()
        
        # Update feature groups based on actual columns
        self.feature_groups['price'] = [col for col in columns if col in ['open', 'high', 'low', 'close', 'volume', 'adjusted_close']]
        self.feature_groups['technical'] = [col for col in columns if col.startswith('ta_')]
        self.feature_groups['news'] = [col for col in columns if col.startswith('news_')]
        self.feature_groups['economic'] = [col for col in columns if col.startswith('fred_')]
        self.feature_groups['events'] = [col for col in columns if col.startswith('event_')]
        self.feature_groups['temporal'] = [col for col in columns if col in ['day_of_week', 'month', 'quarter', 'is_month_end', 'is_quarter_end']]
        
        # Add any return-based features
        return_cols = [col for col in columns if 'return' in col.lower() and col != 'target_0']
        if return_cols:
            self.feature_groups['returns'] = return_cols
        
        # Filter out empty groups
        self.feature_groups = {k: v for k, v in self.feature_groups.items() if v}
        
        print(f"📊 Identified feature groups:")
        for group, features in self.feature_groups.items():
            print(f"   {group}: {len(features)} features")
        
        return self.feature_groups
    
    def get_feature_combinations(self):
        """Generate different feature combinations for ablation study."""
        combinations = []
        
        # Individual groups
        for group_name, features in self.feature_groups.items():
            if features:  # Only include non-empty groups
                combinations.append({
                    'name': f'{group_name}_only',
                    'description': f'Only {group_name} features',
                    'features': features
                })
        
        # Common combinations
        base_features = self.feature_groups.get('price', []) + self.feature_groups.get('temporal', [])
        
        if base_features:
            combinations.append({
                'name': 'baseline',
                'description': 'Price + Temporal features only',
                'features': base_features
            })
            
            # Add each additional data source to baseline
            for group_name, features in self.feature_groups.items():
                if group_name not in ['price', 'temporal'] and features:
                    combinations.append({
                        'name': f'baseline_plus_{group_name}',
                        'description': f'Baseline + {group_name} features',
                        'features': base_features + features
                    })
            
            # All features
            all_features = []
            for features in self.feature_groups.values():
                all_features.extend(features)
            
            combinations.append({
                'name': 'all_features',
                'description': 'All available features',
                'features': list(set(all_features))  # Remove duplicates
            })
        
        return combinations

class DeepLearningExperimentRunner:
    """Runs experiments focused on deep learning models only."""
    
    def __init__(self, random_state=42):
        self.data_manager = DataCombinationManager()
        self.random_state = random_state
        self.results = {}
        
    def prepare_data(self, feature_df: pd.DataFrame, target_col: str = 'target_0'):
        """Prepare data for deep learning experiments."""
        # Identify feature groups
        self.data_manager.identify_feature_groups(feature_df)
        
        # Get feature combinations
        self.feature_combinations = self.data_manager.get_feature_combinations()
        
        # Store the full dataset
        self.feature_df = feature_df
        self.target_col = target_col
        
        print(f"📋 Prepared {len(self.feature_combinations)} feature combinations for deep learning models")
        return self.feature_combinations
    
    def get_results_summary(self):
        """Get a summary of all experimental results."""
        if not self.results:
            return pd.DataFrame()
        
        summary_data = []
        for key, result in self.results.items():
            summary_data.append({
                'Experiment': key,
                'Model': result['model'],
                'Data_Combination': result['combination'],
                'Description': result['description'],
                'Features_Used': result['features_used'],
                'Train_MSE': result.get('train_mse', np.nan),
                'Train_MAE': result.get('train_mae', np.nan),
                'Train_R2': result.get('train_r2', np.nan),
                'Val_MSE': result.get('val_mse', np.nan),
                'Val_MAE': result.get('val_mae', np.nan),
                'Val_R2': result.get('val_r2', np.nan)
            })
        
        return pd.DataFrame(summary_data)
    
    def get_best_models_by_metric(self, metric='val_r2', top_k=10):
        """Get the best performing models by a specific metric."""
        summary_df = self.get_results_summary()
        if summary_df.empty:
            return summary_df
        
        # Sort by metric (ascending for MSE/MAE, descending for R2)
        ascending = True if 'mse' in metric.lower() or 'mae' in metric.lower() else False
        
        return summary_df.sort_values(metric.replace('_', '_'), ascending=ascending).head(top_k)
