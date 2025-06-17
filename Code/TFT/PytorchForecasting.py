# ---------------------------------------------------#
#
#   File       : PytorchForecasting.py
#   Author     : Soham Deshpande
#   Date       : January 2022
#   Description: Assembling and training the model
#                using Pytorch
#
#
# ----------------------------------------------------#


#Imports
#############################

#General
import datetime
import time
import pandas as pd
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
import warnings
from typing import override

#Pytorch
from pytorch_forecasting import Baseline, TemporalFusionTransformer, TimeSeriesDataSet
import pytorch_lightning as pl
from pytorch_forecasting.metrics import SMAPE, PoissonLoss, QuantileLoss
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_forecasting.data.encoders import NaNLabelEncoder
import torch
import torch.utils.data as data_utils
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# Import Hayson's data classes - we'll define simplified versions inline
# from HaysonData import Data_Day_Hourly_StocksPrice, Valid_Data_Class

##############################

# Constants
TRIM = 0.1

warnings.filterwarnings("ignore")  #to avoid printing out absolute paths


class SimplifiedDataValidator:
    """Simplified version of Hayson's data validation"""
    
    @staticmethod
    def is_valid_dataframe(df: pd.DataFrame) -> bool:
        """Check if the DataFrame is valid for stock data"""
        if not isinstance(df, pd.DataFrame) or df.empty:
            return False
        
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in df.columns for col in required_columns):
            return False
        
        # Check if numeric columns are actually numeric
        for col in required_columns:
            if not pd.api.types.is_numeric_dtype(df[col]):
                try:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                except:
                    return False
        
        # Check for null values
        if df[required_columns].isnull().any().any():
            return False
            
        return True
    
    @staticmethod
    def load_and_validate_csv(file_path: str) -> pd.DataFrame:
        """Load and validate a CSV file"""
        if not os.path.isfile(file_path) or not file_path.endswith('.csv'):
            raise FileNotFoundError(f"File {file_path} does not exist or is not a CSV")
        
        df = pd.read_csv(file_path)
        
        # Ensure required columns exist
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        if 'timestamp' not in df.columns:
            # Try to use index as timestamp
            df['timestamp'] = df.index
        
        if not all(col in df.columns for col in required_columns):
            raise ValueError(f"DataFrame missing required columns: {required_columns}")
        
        # Convert to proper types
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        for col in required_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Remove rows with NaN values
        df = df.dropna()
        
        if not SimplifiedDataValidator.is_valid_dataframe(df):
            raise ValueError("DataFrame failed validation")
        
        return df


class EnhancedStockDataLoader:
    """Enhanced data loader combining original FTSE functionality with Hayson's approach"""
    
    def __init__(self, data_dir: str = None, window_size: int = 60, prediction_length: int = 30):
        self.data_dir = data_dir
        self.window_size = window_size
        self.prediction_length = prediction_length
        self.stocks_data = {}
        self.combined_data = None
        self.validator = SimplifiedDataValidator()
    
    def load_directory_data(self) -> dict:
        """Load all stock data from directory"""
        if not self.data_dir or not os.path.isdir(self.data_dir):
            raise FileNotFoundError(f"Data directory {self.data_dir} does not exist")
        
        # Find all CSV files
        csv_files = [f for f in os.listdir(self.data_dir) if f.endswith('.csv')]
        
        # Group by stock symbol
        stock_groups = {}
        for file in csv_files:
            if '_train.csv' in file:
                symbol = file.replace('_train.csv', '')
                if symbol not in stock_groups:
                    stock_groups[symbol] = {}
                stock_groups[symbol]['train'] = os.path.join(self.data_dir, file)
            elif '_test.csv' in file:
                symbol = file.replace('_test.csv', '')
                if symbol not in stock_groups:
                    stock_groups[symbol] = {}
                stock_groups[symbol]['test'] = os.path.join(self.data_dir, file)
        
        # Load and validate data for each stock
        for symbol, files in stock_groups.items():
            try:
                stock_data = {}
                if 'train' in files:
                    train_df = self.validator.load_and_validate_csv(files['train'])
                    train_df['data_type'] = 'train'
                    train_df['symbol'] = symbol
                    stock_data['train'] = train_df
                
                if 'test' in files:
                    test_df = self.validator.load_and_validate_csv(files['test'])
                    test_df['data_type'] = 'test' 
                    test_df['symbol'] = symbol
                    stock_data['test'] = test_df
                
                if stock_data:
                    self.stocks_data[symbol] = stock_data
                    print(f"Loaded data for {symbol}: {list(stock_data.keys())}")
                    
            except Exception as e:
                print(f"Failed to load data for {symbol}: {e}")
        
        return self.stocks_data
    
    def create_sliding_windows(self, df: pd.DataFrame, symbol: str, data_type: str) -> list:
        """Create sliding windows from a DataFrame"""
        if len(df) < self.window_size + self.prediction_length:
            print(f"Warning: {symbol} {data_type} has insufficient data for sliding windows")
            return []
        
        windows = []
        df_sorted = df.sort_values('timestamp').reset_index(drop=True)
        
        for i in range(len(df_sorted) - self.window_size - self.prediction_length + 1):
            window_df = df_sorted.iloc[i:i + self.window_size + self.prediction_length].copy()
            window_df['window_id'] = f"{symbol}_{data_type}_{i}"
            window_df['sequence_position'] = range(len(window_df))
            windows.append(window_df)
        
        return windows
    
    def prepare_for_tft(self) -> pd.DataFrame:
        """Prepare all loaded data for TFT TimeSeriesDataSet"""
        all_windows = []
        
        for symbol, stock_data in self.stocks_data.items():
            for data_type, df in stock_data.items():
                windows = self.create_sliding_windows(df, symbol, data_type)
                all_windows.extend(windows)
        
        if not all_windows:
            raise ValueError("No valid sliding windows created from data")
        
        # Combine all windows
        combined_df = pd.concat(all_windows, ignore_index=True)
        
        # Create TFT-compatible columns
        combined_df['time_idx'] = combined_df.groupby('window_id')['sequence_position'].transform(lambda x: x)
        combined_df['group_id'] = combined_df['symbol'] + '_' + combined_df['data_type']
        
        # Ensure numeric columns
        numeric_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_cols:
            combined_df[col] = pd.to_numeric(combined_df[col], errors='coerce')
        
        # Remove any remaining NaN values
        combined_df = combined_df.dropna()
        
        self.combined_data = combined_df
        print(f"Created TFT dataset with {len(combined_df)} rows and {combined_df['group_id'].nunique()} groups")
        
        return combined_df


class FTSEDataSet:
    """
    Enhanced FTSE Dataset with integrated data processing capabilities
    
    Features:
    - Sliding window data extraction  
    - Data validation and cleaning
    - Multiple stock support
    - Integration with TFT TimeSeriesDataSet
    - Backward compatibility with original FTSE processing
    """

    def __init__(self, 
                 data_dir: str = None,
                 stocks_file_name: str = None,
                 start=datetime.datetime(2010, 1, 1), 
                 stop=datetime.datetime.now(),
                 window_size: int = 60,
                 prediction_length: int = 30,
                 use_directory_structure: bool = True):
        
        self.df_returns = None
        self.data_dir = data_dir
        self.stocks_file_name = stocks_file_name or "/home/soham/Documents/PycharmProjects/NEA/Code/Data/NEAFTSE2010-21.csv"
        self.start = start
        self.stop = stop
        self.window_size = window_size
        self.prediction_length = prediction_length
        self.use_directory_structure = use_directory_structure
        
        # Enhanced data loader
        self.data_loader = EnhancedStockDataLoader(data_dir, window_size, prediction_length)

    def load(self, binary=True, use_enhanced=True):
        """
        Load data with option to use enhanced processing or original FTSE processing
        
        Parameters:
        binary (bool): Whether to convert target to binary classification  
        use_enhanced (bool): Whether to use enhanced directory-based loading
        """
        if use_enhanced and self.data_dir:
            try:
                # Use enhanced data processing
                print("Loading data using enhanced directory structure...")
                self.data_loader.load_directory_data()
                df_returns = self.data_loader.prepare_for_tft()
                self.df_returns = df_returns
                print(f"Successfully loaded enhanced data: {df_returns.shape}")
                return df_returns
            except Exception as e:
                print(f"Enhanced loading failed: {e}")
                print("Falling back to original FTSE processing...")
                return self._load_original_ftse(binary)
        else:
            # Use original FTSE processing
            return self._load_original_ftse(binary)

    def _load_original_ftse(self, binary=True):
        """Original FTSE data loading method with improvements"""
        try:
            df0 = pd.read_csv(self.stocks_file_name, index_col=0, parse_dates=True)
            print("Original FTSE data shape:", df0.shape)

            # Clean data
            df0.dropna(axis=1, how='all', inplace=True)
            df0.dropna(axis=0, how='all', inplace=True)
            
            # Drop columns with >50% NaN values
            nan_threshold = 0.5
            cols_to_drop = df0.loc[:, (df0.isnull().sum() / len(df0.index)) > nan_threshold].columns
            print(f"Dropping columns due to nans > {nan_threshold*100}%:", list(cols_to_drop))
            df0 = df0.drop(cols_to_drop, axis=1)
            
            # Forward and backward fill
            df0 = df0.ffill().bfill()
            print("Any columns still contain nans:", df0.isnull().values.any())

            # Calculate log returns
            df_returns = pd.DataFrame()
            for name in df0.columns:
                df_returns[name] = np.log(df0[name]).diff()

            # Clean and prepare
            df_returns.dropna(axis=0, how='any', inplace=True)
            
            # Binary classification for FTSE if requested
            if binary and 'FTSE' in df_returns.columns:
                df_returns['FTSE'] = [1 if ftse > 0 else 0 for ftse in df_returns.FTSE]
            
            # Add time index and metadata for TFT compatibility
            df_returns.reset_index(inplace=True)
            df_returns['timestamp'] = pd.to_datetime(df_returns['Date'])
            df_returns['time_idx'] = range(len(df_returns))
            df_returns['group_id'] = 'FTSE_group'
            df_returns['symbol'] = 'FTSE'
            
            # Ensure we have required columns for TFT
            if 'Open' in df_returns.columns:
                df_returns['open'] = df_returns['Open']
            if 'close' not in df_returns.columns and 'FTSE' in df_returns.columns:
                df_returns['close'] = df_returns['FTSE']
            
            self.df_returns = df_returns
            return df_returns
            
        except Exception as e:
            print(f"Error loading original FTSE data: {e}")
            # Create minimal dummy data for testing
            dates = pd.date_range(start='2020-01-01', periods=1000, freq='D')
            dummy_data = pd.DataFrame({
                'timestamp': dates,
                'time_idx': range(1000),
                'group_id': 'dummy_group',
                'symbol': 'DUMMY',
                'open': np.random.randn(1000).cumsum() + 100,
                'high': np.random.randn(1000).cumsum() + 105,
                'low': np.random.randn(1000).cumsum() + 95,
                'close': np.random.randn(1000).cumsum() + 100,
                'volume': np.random.randint(1000, 10000, 1000)
            })
            print("Created dummy data for testing")
            self.df_returns = dummy_data
            return dummy_data

    def get_tft_dataset_params(self, target_column='close'):
        """Get parameters for TFT TimeSeriesDataSet creation"""
        if self.df_returns is None:
            raise ValueError("Data not loaded. Call load() first.")
        
        # Determine features based on available columns
        available_cols = self.df_returns.columns.tolist()
        exclude_cols = ['timestamp', 'time_idx', 'group_id', 'symbol', 'window_id', 
                       'sequence_position', 'data_type', target_column, 'Date']
        
        time_varying_unknown_reals = [col for col in available_cols if col not in exclude_cols]
        
        # Ensure we have some features
        if not time_varying_unknown_reals:
            # Fallback to OHLCV if available
            potential_features = ['open', 'high', 'low', 'volume', 'Open', 'High', 'Low', 'Volume']
            time_varying_unknown_reals = [col for col in potential_features 
                                        if col in available_cols and col != target_column]
        
        print(f"Using features: {time_varying_unknown_reals}")
        print(f"Target column: {target_column}")
        
        params = {
            'time_idx': 'time_idx',
            'target': target_column,
            'group_ids': ['group_id'],
            'time_varying_unknown_reals': time_varying_unknown_reals,
            'max_encoder_length': min(self.window_size, 60),  # Reasonable default
            'max_prediction_length': min(self.prediction_length, 30),  # Reasonable default
            'min_encoder_length': min(self.window_size // 2, 30),
            'min_prediction_length': 1,
            'add_relative_time_idx': True,
            'add_target_scales': True,
            'add_encoder_length': True,
            'allow_missing_timesteps': True
        }
        
        return params

    def get_loaders(self, batch_size=16, n_test=1000, device='cpu'):
        """Enhanced data loaders with validation"""
        if self.df_returns is None:
            self.load()

        # For backward compatibility, return tensor loaders if not using enhanced structure
        if not self.use_directory_structure or 'window_id' not in self.df_returns.columns:
            try:
                # Original tensor-based loaders for compatibility
                exclude_cols = ['timestamp', 'time_idx', 'group_id', 'symbol', 'Date']
                feature_cols = [col for col in self.df_returns.columns if col not in exclude_cols]
                
                if 'close' in self.df_returns.columns:
                    target_col = 'close'
                elif 'FTSE' in self.df_returns.columns:
                    target_col = 'FTSE'
                else:
                    target_col = feature_cols[0]
                
                feature_cols = [col for col in feature_cols if col != target_col]
                
                features = self.df_returns[feature_cols].values
                labels = self.df_returns[target_col].values
                
                training_data = data_utils.TensorDataset(
                    torch.tensor(features[:-n_test]).float().to(device),
                    torch.tensor(labels[:-n_test]).float().to(device)
                )
                test_data = data_utils.TensorDataset(
                    torch.tensor(features[-n_test:]).float().to(device),
                    torch.tensor(labels[-n_test:]).float().to(device)
                )
                
                train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=False)
                test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
                return train_dataloader, test_dataloader
            except Exception as e:
                print(f"Error creating tensor loaders: {e}")
                # Return the DataFrame for TFT processing
                return self.df_returns
        else:
            # Return the processed DataFrame for TFT TimeSeriesDataSet
            return self.df_returns

    def validate_data(self):
        """Validate loaded data"""
        if self.df_returns is not None:
            return not self.df_returns.empty and not self.df_returns.isnull().all().all()
        return False

    def get_stock_symbols(self):
        """Get list of available stock symbols"""
        if self.data_loader and self.data_loader.stocks_data:
            return list(self.data_loader.stocks_data.keys())
        elif self.df_returns is not None and 'symbol' in self.df_returns.columns:
            return self.df_returns['symbol'].unique().tolist()
        return []
    """
    Enhanced FTSE Dataset with Hayson's data processing capabilities
    
    Features:
    - Sliding window data extraction
    - Data validation and cleaning
    - Multiple stock support
    - Normalized preprocessing
    - Integration with TFT TimeSeriesDataSet
    """

    def __init__(self, 
                 data_dir: str = None,
                 stocks_file_name: str = None,
                 start=datetime.datetime(2010, 1, 1), 
                 stop=datetime.datetime.now(),
                 window_size: int = 60,
                 prediction_length: int = 30,
                 ignore_leakage: bool = False,
                 use_directory_structure: bool = True):
        
        super().__init__()
        self.df_returns = None
        self.data_dir = data_dir
        self.stocks_file_name = stocks_file_name or "/home/soham/Documents/PycharmProjects/NEA/Code/Data/NEAFTSE2010-21.csv"
        self.start = start
        self.stop = stop
        self.window_size = window_size
        self.prediction_length = prediction_length
        self.ignore_leakage = ignore_leakage
        self.use_directory_structure = use_directory_structure
        
        # Initialize data structures from Valid_Data_Class
        self.test = []
        self.train = []
        self.validation = []
        self.symbols_test_train = {}
        self.symbols_validation = {}
        self.num_test_train = 0
        self.num_validation = 0
        self.num_stocks = 0
        self.trim = TRIM
        
        # Hayson's data handler
        self.hayson_data = None

    def load_from_hayson_directory(self, data_dir: str) -> pd.DataFrame:
        """Load data using Hayson's directory structure and processing"""
        if not os.path.isdir(data_dir):
            raise FileNotFoundError(f"Data directory {data_dir} does not exist")
            
        # Use Hayson's Data_Day_Hourly_StocksPrice class
        self.hayson_data = Data_Day_Hourly_StocksPrice.from_dir(
            dir_path=data_dir,
            ignore_leakage=self.ignore_leakage
        )
        
        # Copy data structures from Hayson's class
        self.test = self.hayson_data.test
        self.train = self.hayson_data.train
        self.validation = self.hayson_data.validation
        self.symbols_test_train = self.hayson_data.symbols_test_train
        self.symbols_validation = self.hayson_data.symbols_validation
        self.num_test_train = self.hayson_data.num_test_train
        self.num_validation = self.hayson_data.num_validation
        self.num_stocks = self.hayson_data.num_stocks
        
        # Create sliding windows and combine data for TFT
        combined_df = self._create_sliding_windows_for_tft()
        return combined_df

    def _create_sliding_windows_for_tft(self) -> pd.DataFrame:
        """Create sliding windows from the loaded data and format for TFT"""
        all_windows = []
        
        # Process test/train data
        for symbol, idx in self.symbols_test_train.items():
            train_df = self.train[idx].copy()
            test_df = self.test[idx].copy()
            
            # Add metadata columns
            train_df['stock_symbol'] = symbol
            test_df['stock_symbol'] = symbol
            train_df['data_type'] = 'train'
            test_df['data_type'] = 'test'
            
            # Create sliding windows for train data
            train_windows = self._create_windows_from_df(train_df, symbol, 'train')
            test_windows = self._create_windows_from_df(test_df, symbol, 'test')
            
            all_windows.extend(train_windows)
            all_windows.extend(test_windows)
        
        # Process validation data
        for symbol, idx in self.symbols_validation.items():
            val_df = self.validation[idx].copy()
            val_df['stock_symbol'] = symbol
            val_df['data_type'] = 'validation'
            
            val_windows = self._create_windows_from_df(val_df, symbol, 'validation')
            all_windows.extend(val_windows)
        
        # Convert to single DataFrame for TFT
        if all_windows:
            combined_df = pd.concat(all_windows, ignore_index=True)
            combined_df = self._prepare_for_tft(combined_df)
            return combined_df
        else:
            raise ValueError("No valid data windows created")

    def _create_windows_from_df(self, df: pd.DataFrame, symbol: str, data_type: str) -> list:
        """Create sliding windows from a single DataFrame"""
        windows = []
        
        # Ensure timestamp column exists and is properly formatted
        if 'timestamp' not in df.columns:
            raise ValueError(f"DataFrame for {symbol} missing timestamp column")
        
        # Convert timestamp to datetime if needed
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        # Create sliding windows
        for i in range(len(df) - self.window_size - self.prediction_length + 1):
            window_df = df.iloc[i:i + self.window_size + self.prediction_length].copy()
            window_df['window_id'] = f"{symbol}_{data_type}_{i}"
            window_df['sequence_position'] = range(len(window_df))
            windows.append(window_df)
        
        return windows

    def _prepare_for_tft(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare the combined DataFrame for TFT TimeSeriesDataSet"""
        # Create time index for TFT
        df['time_idx'] = df.groupby('window_id')['sequence_position'].transform(lambda x: x)
        
        # Create group IDs for TFT
        df['group_id'] = df['stock_symbol'] + '_' + df['data_type']
        
        # Ensure all required columns are present and properly typed
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Remove any rows with NaN values
        df = df.dropna()
        
        return df

    def load(self, binary=True, use_hayson=True):
        """
        Load data with option to use Hayson's processing or original FTSE processing
        
        Parameters:
        binary (bool): Whether to convert target to binary classification
        use_hayson (bool): Whether to use Hayson's directory structure
        """
        if use_hayson and self.data_dir:
            # Use Hayson's data processing
            df_returns = self.load_from_hayson_directory(self.data_dir)
            self.df_returns = df_returns
            return df_returns
        else:
            # Use original FTSE processing
            return self._load_original_ftse(binary)

    def _load_original_ftse(self, binary=True):
        """Original FTSE data loading method with improvements"""
        df0 = pd.read_csv(self.stocks_file_name, index_col=0, parse_dates=True)
        print("Original FTSE data shape:", df0.shape)

        # Clean data
        df0.dropna(axis=1, how='all', inplace=True)
        df0.dropna(axis=0, how='all', inplace=True)
        
        # Drop columns with >50% NaN values
        nan_threshold = 0.5
        cols_to_drop = df0.loc[:, (df0.isnull().sum() / len(df0.index)) > nan_threshold].columns
        print(f"Dropping columns due to nans > {nan_threshold*100}%:", cols_to_drop)
        df0 = df0.drop(cols_to_drop, axis=1)
        
        # Forward and backward fill
        df0 = df0.ffill().bfill()
        print("Any columns still contain nans:", df0.isnull().values.any())

        # Calculate log returns
        df_returns = pd.DataFrame()
        for name in df0.columns:
            df_returns[name] = np.log(df0[name]).diff()

        # Clean and prepare
        df_returns.dropna(axis=0, how='any', inplace=True)
        
        # Binary classification for FTSE if requested
        if binary and 'FTSE' in df_returns.columns:
            df_returns['FTSE'] = [1 if ftse > 0 else 0 for ftse in df_returns.FTSE]
        
        # Add time index and metadata for TFT compatibility
        df_returns['timestamp'] = pd.to_datetime(df_returns.index)
        df_returns['time_idx'] = range(len(df_returns))
        df_returns['group_id'] = 'FTSE_group'
        df_returns['stock_symbol'] = 'FTSE'
        
        self.df_returns = df_returns
        return df_returns

    def get_tft_dataset_params(self, target_column='close'):
        """Get parameters for TFT TimeSeriesDataSet creation"""
        if self.df_returns is None:
            raise ValueError("Data not loaded. Call load() first.")
        
        # Determine features based on data structure
        if self.use_directory_structure and self.hayson_data:
            # Use OHLCV features for stock data
            time_varying_unknown_reals = ['open', 'high', 'low', 'volume']
            if target_column in time_varying_unknown_reals:
                time_varying_unknown_reals.remove(target_column)
        else:
            # Use all available features except target
            available_cols = [col for col in self.df_returns.columns 
                            if col not in ['timestamp', 'time_idx', 'group_id', 'stock_symbol', target_column]]
            time_varying_unknown_reals = available_cols
        
        params = {
            'time_idx': 'time_idx',
            'target': target_column,
            'group_ids': ['group_id'],
            'time_varying_unknown_reals': time_varying_unknown_reals,
            'max_encoder_length': self.window_size,
            'max_prediction_length': self.prediction_length,
            'min_encoder_length': self.window_size // 2,
            'min_prediction_length': 1,
            'add_relative_time_idx': True,
            'add_target_scales': True,
            'add_encoder_length': True,
            'allow_missing_timesteps': True
        }
        
        return params

    def get_loaders(self, batch_size=16, n_test=1000, device='cpu'):
        """Enhanced data loaders with Hayson's validation"""
        if self.df_returns is None:
            self.load()

        # Original tensor-based loaders for compatibility
        if not self.use_directory_structure:
            features = self.df_returns.drop(['timestamp', 'time_idx', 'group_id', 'stock_symbol'], 
                                          axis=1, errors='ignore').values
            labels = self.df_returns['close'] if 'close' in self.df_returns.columns else self.df_returns.iloc[:, 0]
            
            training_data = data_utils.TensorDataset(
                torch.tensor(features[:-n_test]).float().to(device),
                torch.tensor(labels[:-n_test]).float().to(device)
            )
            test_data = data_utils.TensorDataset(
                torch.tensor(features[-n_test:]).float().to(device),
                torch.tensor(labels[-n_test:]).float().to(device)
            )
            
            train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=False)
            test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
            return train_dataloader, test_dataloader
        else:
            # Return the processed DataFrame for TFT TimeSeriesDataSet
            return self.df_returns

    def validate_data(self):
        """Validate loaded data using Hayson's validation methods"""
        if self.hayson_data:
            return self.hayson_data.is_valid_class()
        elif self.df_returns is not None:
            # Basic validation for original FTSE data
            return not self.df_returns.empty and not self.df_returns.isnull().all().all()
        else:
            return False

    def get_stock_data(self, symbol: str):
        """Get data for a specific stock symbol using Hayson's method"""
        if self.hayson_data:
            return self.hayson_data.get_from_symbol(symbol)
        else:
            raise ValueError("Hayson data not loaded. Use load(use_hayson=True) first.")

    def visualize_stock(self, symbol: str, **kwargs):
        """Visualize stock data using Hayson's visualization method"""
        if self.hayson_data:
            return self.hayson_data.visualize(symbol, **kwargs)
        else:
            raise ValueError("Hayson data not loaded. Use load(use_hayson=True) first.")



class TFT:

    """
    Temporal Fusion Transformer

    Setting up the model using PyTorch lighting.
    The class determines the main key features of the model, listed below:

    Tuneable Hyperparameters:
        int prediction length
        str   features
        int   max encoder length
        int   training cutoff
        str   time index
        str   group ids
        int   min encoder length
        int   min prediction length
        str   target
        int   max epochs
        int   gpus
        int   learning rate
        int   hidden layer size
        int   drop out
        int   hidden continous size
        int   output size
        int   attention head size
        float loss function

    """

    def __init__(self, prediction_length = 2000):
        self.prediction_length = prediction_length
        self.training = None
        self.validation = None
        self.trainer = None
        self.model = None
        self.batch_size =16

    def load_data(self):
        """
        Load data using the enhanced FTSEDataSet class with integrated processing
        Set prediction and encoder lengths
        Set up training data using TimeSeriesDataSet function
        """
        # Use enhanced dataset with directory structure
        dataset = FTSEDataSet(
            data_dir="/Users/haysoncheung/programs/pythonProject/Stock-TFT/Code/datafinal",
            window_size=60,
            prediction_length=self.prediction_length,
            use_directory_structure=True
        )
        
        print("Dataset:", dataset)
        
        try:
            # Load data using enhanced processing
            ftse_df = dataset.load(binary=False, use_enhanced=True)
            print("Data loaded successfully")
            print("Data shape:", ftse_df.shape)
            print("Columns:", ftse_df.columns.tolist())
            print("Available symbols:", dataset.get_stock_symbols())
            
            # Determine target column
            target_column = 'close' if 'close' in ftse_df.columns else 'Open'
            if target_column not in ftse_df.columns:
                # Use first numeric column as target
                numeric_cols = ftse_df.select_dtypes(include=[np.number]).columns
                target_column = numeric_cols[0] if len(numeric_cols) > 0 else ftse_df.columns[0]
            
            print(f"Using target column: {target_column}")
            
            # Get TFT dataset parameters from the enhanced dataset
            tft_params = dataset.get_tft_dataset_params(target_column=target_column)
            
            # Set up training cutoff
            max_time_idx = ftse_df['time_idx'].max()
            training_cutoff = max_time_idx - self.prediction_length
            
            print("Training cutoff:", training_cutoff)
            print("Max time index:", max_time_idx)
            
            # Filter training data
            training_data = ftse_df[ftse_df['time_idx'] <= training_cutoff]
            print("Training data shape:", training_data.shape)
            
            if training_data.empty:
                raise ValueError("No training data after applying cutoff")
            
            # Create TimeSeriesDataSet with enhanced parameters
            self.training = TimeSeriesDataSet(
                training_data,
                **tft_params
            )
            
            print("Training dataset created successfully")
            print("Training dataset parameters:", self.training.get_parameters())

            # create validation set (predict=True) which means to predict the last max_prediction_length points in time
            # for each series
            self.validation = TimeSeriesDataSet.from_dataset(self.training, ftse_df, predict=True, stop_randomization=True)
            print("Validation dataset created successfully")
            
        except Exception as e:
            print(f"Enhanced data loading failed: {e}")
            print("Attempting fallback to original processing...")
            
            # Fallback to original processing with better error handling
            try:
                dataset.use_directory_structure = False
                ftse_df = dataset.load(binary=False, use_enhanced=False)
                
                # Original TFT setup for backward compatibility
                if 'Date' not in ftse_df.columns and 'timestamp' in ftse_df.columns:
                    ftse_df['Date'] = ftse_df['timestamp']
                
                time_index = "time_idx"
                target = "close" if "close" in ftse_df.columns else "Open"
                
                # Get available features
                exclude_cols = ['timestamp', 'time_idx', 'group_id', 'symbol', 'Date', target]
                features = [col for col in ftse_df.columns if col not in exclude_cols]
                
                print("Fallback - Features:", features)
                print("Fallback - Target:", target)
                
                # Ensure we have a reasonable encoder length
                max_encoder_length = min(len(ftse_df) // 4, 100)
                training_cutoff = ftse_df[time_index].max() - self.prediction_length
                
                # Add categorical encoder if needed
                categorical_encoders = {}
                if 'group_id' in ftse_df.columns:
                    categorical_encoders = {"group_id": NaNLabelEncoder().fit(ftse_df.group_id)}
                
                self.training = TimeSeriesDataSet(
                    ftse_df[ftse_df[time_index] <= training_cutoff],
                    time_idx=time_index,
                    target=target,
                    group_ids=["group_id"],
                    min_encoder_length=max_encoder_length // 2,
                    max_encoder_length=max_encoder_length,
                    min_prediction_length=1,
                    max_prediction_length=self.prediction_length,
                    time_varying_unknown_reals=features,
                    add_relative_time_idx=True,
                    add_target_scales=True,
                    add_encoder_length=True,
                    allow_missing_timesteps=True,
                    categorical_encoders=categorical_encoders
                )
                
                print("Fallback training dataset created")
                self.validation = TimeSeriesDataSet.from_dataset(self.training, ftse_df, predict=True, stop_randomization=True)
                
            except Exception as fallback_error:
                print(f"Fallback also failed: {fallback_error}")
                raise RuntimeError("Both enhanced and fallback data loading failed") from fallback_error

    def create_tft_model(self):
        """
        Create the model
        Define hyperparameters
        Declare input, hidden, drop out, attention head and output size
        Declare epochs


        TFT Design
            1. Variable Selection Network
            2. LSTM Encoder
            3. Normalisation
            4. GRN
            5. MutiHead Attention
            6. Normalisation
            7. GRN
            8. Normalisation
            9. Dense network
            10.Quantile outputs


        """
        # configure network and trainer
        early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=10, verbose=False, mode="min")
        lr_logger = LearningRateMonitor()  # log the learning rate
        logger = TensorBoardLogger("lightning_logs")  # logging results to a tensorboard

        self.trainer = pl.Trainer(
            max_epochs=10,
            gpus=0,
            weights_summary="top",
            gradient_clip_val=0.1,
            limit_train_batches=30,
            callbacks=[lr_logger, early_stop_callback],
            logger=logger,
        )

        self.model = TemporalFusionTransformer.from_dataset(
            self.training,
            # not meaningful for finding the learning rate but otherwise very important
            learning_rate=0.05,
            hidden_size= 4,  # most important hyperparameter apart from learning rate
            # number of attention heads. Set to up to 4 for large datasets
            attention_head_size=1,
            dropout=0.1,  # between 0.1 and 0.3 are good values
            hidden_continuous_size=4,  # set to <= hidden_size
            output_size=7,  # 7 quantiles by default
            loss=QuantileLoss(),
            # reduce learning rate if no improvement in validation loss after x epochs
            reduce_on_plateau_patience=4,
        )
        print(f"Number of parameters in network: {self.model.size() / 1e3:.1f}k")

    def train(self):
        # create dataloaders for model
        train_dataloader = self.training.to_dataloader(train=True, batch_size=self.batch_size, num_workers=0)
        val_dataloader = self.validation.to_dataloader(train=False, batch_size=self.batch_size * 10, num_workers=0)

        # fit network
        self.trainer.fit(
            self.model,
            train_dataloader=train_dataloader,
            val_dataloaders=val_dataloader,
        )

    def evaluate(self, number_of_examples = 15):
        """
        Evaluate the model
        Load the saved model from the last saved epoch
        Compare predictions against real values
        Create graphs to visualise performance
        """
        # load the best model according to the validation loss
        # (given that we use early stopping, this is not necessarily the last epoch)
        best_model_path = self.trainer.checkpoint_callback.best_model_path
        best_tft = TemporalFusionTransformer.load_from_checkpoint(best_model_path)

        # raw predictions are a dictionary from which all kind of information including quantiles can be extracted
        val_dataloader = self.validation.to_dataloader(train=False, batch_size=self.batch_size * 10, num_workers=0)
        raw_predictions, x = best_tft.predict(val_dataloader, mode="raw", return_x=True)
        #print('raw_predictions', raw_predictions)
        for idx in range(number_of_examples):  # plot 10 examples
            best_tft.plot_prediction(x, raw_predictions, idx=idx, add_loss_to_title=True);

        predictions, x = best_tft.predict(val_dataloader, return_x=True)
        #print('predictions2', predictions)
        #print('x values', x)
        predictions_vs_actuals = best_tft.calculate_prediction_actual_by_variable(x, predictions)
        #print('predictions_vs_actuals', predictions_vs_actuals)
        best_tft.plot_prediction_actual_by_variable(predictions_vs_actuals);
        #best_tft.plot(predictions,x)
        # print(best_tft)



def tft():
    tft = TFT()
    tft.load_data()
    tft.create_tft_model()
    tft.train()
    #torch.save(tft,"Model.pickle")
    tft.evaluate(number_of_examples=1)
    plt.show()

if __name__ == "__main__":
    tft()
    #tft = TFT()
    #tft.load_data()
    #tft.create_tft_model()
    #tft.train()
    #tft.evaluate(number_of_examples=1)
    #plt.show()
    #torch.save(tft, "Model.pickle")





