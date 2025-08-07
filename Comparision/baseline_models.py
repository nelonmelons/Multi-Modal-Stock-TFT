#!/usr/bin/env python3
"""
Comprehensive baseline models with different data combination experiments.

This module implements various baseline models tested across different feature combinations
to understand the impact of different data sources on prediction performance.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import lightgbm as lgb
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import warnings
warnings.filterwarnings('ignore')

class BaselineModelFactory:
    """Factory for creating different baseline models with various configurations."""
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        
    def create_model(self, model_name: str, **kwargs):
        """Create a model instance based on the model name."""
        models = {
            # Linear Models
            'Ridge': lambda: Pipeline([
                ('scaler', StandardScaler()),
                ('model', Ridge(alpha=1.0, random_state=self.random_state))
            ]),
            'Lasso': lambda: Pipeline([
                ('scaler', StandardScaler()),
                ('model', Lasso(alpha=0.1, random_state=self.random_state))
            ]),
            'ElasticNet': lambda: Pipeline([
                ('scaler', StandardScaler()),
                ('model', ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=self.random_state))
            ]),
            
            # Tree-based Models
            'Random Forest': lambda: RandomForestRegressor(
                n_estimators=100, 
                max_depth=10, 
                random_state=self.random_state,
                n_jobs=-1
            ),
            'XGBoost': lambda: xgb.XGBRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=self.random_state,
                n_jobs=-1
            ),
            'LightGBM': lambda: lgb.LGBMRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=self.random_state,
                n_jobs=-1,
                verbose=-1
            ),
            
            # Support Vector Machine
            'SVR': lambda: Pipeline([
                ('scaler', StandardScaler()),
                ('model', SVR(kernel='rbf', C=1.0, gamma='scale'))
            ]),
            
            # Neural Network
            'MLP': lambda: Pipeline([
                ('scaler', StandardScaler()),
                ('model', MLPRegressor(
                    hidden_layer_sizes=(100, 50),
                    max_iter=500,
                    random_state=self.random_state
                ))
            ]),
        }
        
        if model_name not in models:
            raise ValueError(f"Unknown model: {model_name}. Available: {list(models.keys())}")
        
        return models[model_name]()


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


class BaselineExperimentRunner:
    """Runs comprehensive baseline experiments across models and feature combinations."""
    
    def __init__(self, random_state=42):
        self.model_factory = BaselineModelFactory(random_state)
        self.data_manager = DataCombinationManager()
        self.random_state = random_state
        self.results = {}
        
    def prepare_data(self, feature_df: pd.DataFrame, target_col: str = 'target_0'):
        """Prepare data for baseline experiments."""
        # Identify feature groups
        self.data_manager.identify_feature_groups(feature_df)
        
        # Get feature combinations
        self.feature_combinations = self.data_manager.get_feature_combinations()
        
        # Store the full dataset
        self.feature_df = feature_df
        self.target_col = target_col
        
        print(f"📋 Prepared {len(self.feature_combinations)} feature combinations")
        return self.feature_combinations
    
    def train_and_evaluate_model(self, model_name: str, combination: dict, 
                                train_data: pd.DataFrame, val_data: pd.DataFrame):
        """Train and evaluate a single model with specific feature combination."""
        try:
            # Extract features and target
            feature_cols = [col for col in combination['features'] if col in train_data.columns]
            if not feature_cols:
                return None
            
            X_train = train_data[feature_cols].values
            y_train = train_data[self.target_col].values
            X_val = val_data[feature_cols].values
            y_val = val_data[self.target_col].values
            
            # Handle missing values
            X_train = np.nan_to_num(X_train, nan=0.0)
            X_val = np.nan_to_num(X_val, nan=0.0)
            y_train = np.nan_to_num(y_train, nan=0.0)
            y_val = np.nan_to_num(y_val, nan=0.0)
            
            # Create and train model
            model = self.model_factory.create_model(model_name)
            
            # Special handling for time series models
            if model_name == 'ARIMA':
                return self._train_arima_model(y_train, y_val)
            
            # Train the model
            model.fit(X_train, y_train)
            
            # Make predictions
            train_pred = model.predict(X_train)
            val_pred = model.predict(X_val)
            
            # Calculate metrics
            results = {
                'model': model_name,
                'combination': combination['name'],
                'description': combination['description'],
                'features_used': len(feature_cols),
                'train_mse': mean_squared_error(y_train, train_pred),
                'train_mae': mean_absolute_error(y_train, train_pred),
                'train_r2': max(0.0, min(1.0, r2_score(y_train, train_pred))),
                'val_mse': mean_squared_error(y_val, val_pred),
                'val_mae': mean_absolute_error(y_val, val_pred),
                'val_r2': max(0.0, min(1.0, r2_score(y_val, val_pred))),
                'predictions': val_pred,
                'actuals': y_val
            }
            
            return results
            
        except Exception as e:
            print(f"⚠️  Error training {model_name} with {combination['name']}: {str(e)}")
            return None
    
    def _train_arima_model(self, y_train, y_val):
        """Special handling for ARIMA model."""
        try:
            # Simple ARIMA(1,1,1) model
            model = ARIMA(y_train, order=(1, 1, 1))
            fitted_model = model.fit()
            
            # Forecast
            forecast = fitted_model.forecast(steps=len(y_val))
            
            # Calculate metrics (only validation since ARIMA doesn't have traditional training predictions)
            results = {
                'model': 'ARIMA',
                'combination': 'time_series_only',
                'description': 'ARIMA time series model',
                'features_used': 1,
                'train_mse': np.nan,
                'train_mae': np.nan,
                'train_r2': np.nan,
                'val_mse': mean_squared_error(y_val, forecast),
                'val_mae': mean_absolute_error(y_val, forecast),
                'val_r2': max(0.0, min(1.0, r2_score(y_val, forecast))),
                'predictions': forecast,
                'actuals': y_val
            }
            
            return results
            
        except Exception as e:
            print(f"⚠️  Error training ARIMA: {str(e)}")
            return None
    
    def run_comprehensive_experiment(self, train_data: pd.DataFrame, val_data: pd.DataFrame):
        """Run comprehensive baseline experiments."""
        print("🚀 Starting Comprehensive Baseline Experiments...\n")
        
        # Prepare data
        self.prepare_data(pd.concat([train_data, val_data], ignore_index=True))
        
        # Define models to test
        models_to_test = [
            'Ridge', 'Lasso', 'ElasticNet', 'Random Forest', 
            'XGBoost', 'LightGBM', 'SVR', 'MLP'
        ]
        
        # Add ARIMA separately since it doesn't use features
        arima_result = self._train_arima_model(
            train_data[self.target_col].values,
            val_data[self.target_col].values
        )
        if arima_result:
            key = f"ARIMA_time_series_only"
            self.results[key] = arima_result
        
        # Run experiments for each model and feature combination
        total_experiments = len(models_to_test) * len(self.feature_combinations)
        current_experiment = 0
        
        for model_name in models_to_test:
            print(f"🔧 Training {model_name} models...")
            
            for combination in self.feature_combinations:
                current_experiment += 1
                print(f"   [{current_experiment}/{total_experiments}] {combination['name']} - {combination['description']}")
                
                result = self.train_and_evaluate_model(
                    model_name, combination, train_data, val_data
                )
                
                if result:
                    key = f"{model_name}_{combination['name']}"
                    self.results[key] = result
        
        print(f"\n✅ Completed {len(self.results)} successful experiments")
        return self.results
    
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
                'Train_MSE': result['train_mse'],
                'Train_MAE': result['train_mae'],
                'Train_R2': result['train_r2'],
                'Val_MSE': result['val_mse'],
                'Val_MAE': result['val_mae'],
                'Val_R2': result['val_r2']
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
