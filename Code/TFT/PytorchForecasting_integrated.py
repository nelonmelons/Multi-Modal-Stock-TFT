import datetime
import numpy as np
import pandas as pd
import torch
from torch.utils import data as data_utils
from torch.utils.data import DataLoader
import os
from typing import List, Dict, Tuple # For type hints

# Define TRIM constant
TRIM = 0.1

# Valid_Data_Class: Ensure this is defined before FTSEDataSet
class Valid_Data_Class(object):
    test: List[pd.DataFrame]
    train: List[pd.DataFrame]
    validation: List[pd.DataFrame]
    symbols_test_train: Dict[str, int]
    symbols_validation: Dict[str, int]
    num_test_train: int
    num_validation: int
    num_stocks: int
    ignore_leakage: bool
    # trim = TRIM # Subclasses like FTSEDataSet will handle their own trim attribute

    def __init__(self):
        pass

    @staticmethod
    def is_valid_dataframe(df: pd.DataFrame) -> bool:
        c1 = isinstance(df, pd.DataFrame)
        c2 = not df.empty

        # Standardize column names to lowercase for checks
        df_cols_lower = {col.lower() for col in df.columns}
        required_cols_lower = {'open', 'high', 'low', 'close', 'volume'}
        c3 = required_cols_lower.issubset(df_cols_lower)

        Ca = c1 and c2 and c3
        if not Ca:
            missing = required_cols_lower - df_cols_lower
            assert Ca, f"DataFrame is not valid, is empty, or missing required columns (expected {required_cols_lower}, found {df_cols_lower}, missing {missing})."

        # Create a copy with lowercase column names for checks
        df_check = df.copy()
        df_check.columns = df_check.columns.str.lower()

        numeric_cols_to_check = list(required_cols_lower)
        try:
            for col in numeric_cols_to_check:
                df_check[col] = pd.to_numeric(df_check[col], errors='raise')
        except (ValueError, TypeError) as e:
            assert False, f"Failed to convert required columns to numeric: {e}"

        r1 = all(pd.api.types.is_numeric_dtype(df_check[col]) for col in numeric_cols_to_check)
        assert r1, "Required columns are not all numeric even after attempted conversion."

        c4 = all(df_check[col].notnull().all() for col in numeric_cols_to_check)
        assert c4, "DataFrame contains null values in the required columns after conversion."

        return True

    @staticmethod
    def is_valid_train_test_data(test: pd.DataFrame, train: pd.DataFrame) -> bool:
        c1 = isinstance(test, pd.DataFrame) and isinstance(train, pd.DataFrame)
        c2 = not test.empty and not train.empty
        Ca = c1 and c2
        assert Ca, "Test or train DataFrame is not a valid DataFrame or is empty."

        for df_name, df_item in [('test', test), ('train', train)]:
            assert 'timestamp' in df_item.columns, f"{df_name} DataFrame missing 'timestamp' column."
            # Ensure timestamp is datetime64
            if not pd.api.types.is_datetime64_any_dtype(df_item['timestamp']):
                try:
                    df_item['timestamp'] = pd.to_datetime(df_item['timestamp'])
                except Exception as e:
                    assert False, f"Failed to convert 'timestamp' in {df_name} to datetime: {e}"

        # Ensure timestamps are sorted before min/max comparison if not guaranteed
        # For is_monotonic_increasing, they must be sorted.
        # If DataFrames are large, consider checking a sample or relying on is_monotonic.
        assert train['timestamp'].is_monotonic_increasing, "Train DataFrame timestamps are not monotonic increasing."
        assert test['timestamp'].is_monotonic_increasing, "Test DataFrame timestamps are not monotonic increasing."
        assert test['timestamp'].min() > train['timestamp'].max(), "Test DataFrame is not strictly after the train DataFrame."

        return True

    def is_valid_class(self) -> bool:
        # This base method should be overridden by subclasses with meaningful checks.
        # The original checks were too specific for a generic mixin.
        return True

# Original comment for FTSEDataSet
# FTSEDataSet: old code, the normizlaiztion is bad, but we would like to keep the interfaces and the functions that we can invoke.

class FTSEDataSet(Valid_Data_Class):
    """
    Stock Dataset - Works with any stock data (FTSE, SPY, etc.)
    Enhanced with new normalization, trimming, and directory loading capabilities.
    """
    trim = TRIM # Class attribute for default trim ratio

    def __init__(self, path: str = "../datafinal/AAPL_train.csv",
                 source_type: str = 'file',
                 trim_ratio: float = None, # If None, uses class default FTSEDataSet.trim
                 start=datetime.datetime(2010, 1, 1), # Kept for interface compatibility
                 stop=datetime.datetime.now()):      # Kept for interface compatibility
        super().__init__()
        self.path = path
        self.source_type = source_type.lower()
        self.trim_ratio = trim_ratio if trim_ratio is not None else FTSEDataSet.trim

        self.df_returns = None # Stores the final processed DataFrame (or combined from train/test)
        self.processed_train_df = None
        self.processed_test_df = None
        # self.stocks_file_name = path # Old attribute, self.path is used now.

    @staticmethod
    def _load_check_csv(file_path: str) -> pd.DataFrame:
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"File {file_path} does not exist.")

        df = pd.read_csv(file_path)
        df.columns = df.columns.str.lower() # Standardize to lowercase

        required_columns = ['open', 'high', 'low', 'close', 'volume', 'timestamp']
        missing_cols = [col for col in required_columns if col not in df.columns]

        if missing_cols:
            # Attempt to map from TitleCase if lowercase is missing (common in older data)
            renamed_in_load_check = False
            temp_df_cols = df.columns.tolist() # Operate on a copy for renaming checks
            current_df_col_map = {c.lower(): c for c in temp_df_cols}

            for col_lower in missing_cols:
                col_title = col_lower.title()
                if col_title in temp_df_cols and col_lower not in current_df_col_map : # Check if TitleCase exists and not already lowercase
                    df.rename(columns={col_title: col_lower}, inplace=True)
                    renamed_in_load_check = True
            if renamed_in_load_check: # Recheck missing after potential renames
                 missing_cols = [col for col in required_columns if col not in df.columns]

        if missing_cols:
            raise ValueError(f"DataFrame {file_path} is missing required columns (even after TitleCase check): {missing_cols}. Found: {df.columns.tolist()}")

        # Validate DataFrame structure and basic types using the inherited method
        # is_valid_dataframe expects specific columns, so pass the relevant part or ensure df has them.
        # For this method, we assume df contains at least the required_columns.
        if not FTSEDataSet.is_valid_dataframe(df[required_columns]): # Call as static method
            raise ValueError(f"DataFrame from {file_path} is not valid according to is_valid_dataframe checks.")

        # Ensure timestamp is datetime
        if 'timestamp' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.sort_values(by='timestamp', inplace=True) # Ensure chronological order
        df.reset_index(drop=True, inplace=True)
        return df

    @staticmethod
    def _smart_trim_split(
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        trim_ratio: float = TRIM,
        jump_threshold: int = 60 * 60 * 4 # 4 hours in seconds
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:

        for df_name, df_item in [('train_df', train_df), ('test_df', test_df)]:
            if 'timestamp' not in df_item.columns:
                raise ValueError(f"{df_name} must contain a 'timestamp' column.")
            if not pd.api.types.is_datetime64_any_dtype(df_item['timestamp']):
                 df_item['timestamp'] = pd.to_datetime(df_item['timestamp'])

        # Convert to Unix timestamps (seconds) for numeric difference calculation
        train_ts_unix = train_df['timestamp'].astype('int64') // 10**9
        test_ts_unix = test_df['timestamp'].astype('int64') // 10**9

        # Create temporary DataFrames with Unix timestamps for manipulation
        _train_df = train_df.assign(timestamp_unix=train_ts_unix)
        _test_df = test_df.assign(timestamp_unix=test_ts_unix)

        full_df = pd.concat([_train_df, _test_df], ignore_index=True)
        n_total = len(full_df)
        i1a = int(n_total * trim_ratio) # Initial trim point index
        i2a = len(_train_df) # Original split point index (end of train data in full_df)

        timestamps_unix_all = full_df['timestamp_unix'].values
        jumps = [i for i in range(1, n_total) if timestamps_unix_all[i] - timestamps_unix_all[i-1] > jump_threshold]

        if not jumps:
            print("Warning: No trading day boundaries (jumps) detected in _smart_trim_split. Applying simple percentage trim and split.")
            trimmed_train_head = full_df.iloc[:i2a].iloc[i1a:].reset_index(drop=True)
            trimmed_test_head = full_df.iloc[i2a:].reset_index(drop=True) # Test data is not trimmed at head here
            return trimmed_train_head.drop(columns=['timestamp_unix', 'timestamp_unix'], errors='ignore'), \
                   trimmed_test_head.drop(columns=['timestamp_unix', 'timestamp_unix'], errors='ignore')

        # Find new start for train data (i1b): first jump after initial trim point i1a
        i1b_candidates = [j for j in jumps if j >= i1a] # Changed > to >= to include jump at i1a
        i1b = i1b_candidates[0] if i1b_candidates else i1a # Fallback to i1a if no jump after

        # Find new end for train data (i2b): last jump before or at original split point i2a, but after new start i1b
        jumps_for_train_end = [j for j in jumps if i1b <= j < i2a] # Strictly before i2a
        i2b = jumps_for_train_end[-1] if jumps_for_train_end else i1b # Fallback: train ends where it starts (empty if i1b=i2a) or use i2a?
                                                                    # If no jump in train part, train ends at original i2a or i1b.
                                                                    # Let's make train end at a jump or be empty.
        if not jumps_for_train_end : # if no jump found after i1b within original train part
            if i1b < i2a: # if there was space for train data
                i2b = i2a # train data is from i1b to original end of train
            else: # i1b >= i2a, means trim point is beyond original train data
                i2b = i1b # train data is empty

        # Test data starts at i2b.
        # The original logic for i3b (end of test) was complex.
        # Let's simplify: test data is from i2b to the end of full_df, or to the last jump in that segment.

        final_train_df = full_df.iloc[i1b:i2b].reset_index(drop=True)
        final_test_df = full_df.iloc[i2b:].reset_index(drop=True) # Test data from new split point to end

        return final_train_df.drop(columns=['timestamp_unix'], errors='ignore'), \
               final_test_df.drop(columns=['timestamp_unix'], errors='ignore')

    @staticmethod
    def _mask_normalize_dataframe(df: pd.DataFrame, *dfs: pd.DataFrame, trim: float = 0.0) -> pd.DataFrame | Tuple[pd.DataFrame, ...]:
        price_cols = ['open', 'high', 'low', 'close']
        volume_cols = ['volume']

        all_dfs_list = [df] + list(dfs)

        for i, d_item in enumerate(all_dfs_list):
            if d_item.empty: # Skip empty dataframes
                print(f"Warning: DataFrame at index {i} is empty in _mask_normalize_dataframe. Skipping.")
                continue
            for col_group in [price_cols, volume_cols]:
                for col in col_group:
                    if col not in d_item.columns:
                        raise ValueError(f"Column '{col}' not found in DataFrame {i} for normalization. Columns: {d_item.columns.tolist()}")
            if 'timestamp' not in d_item.columns and trim > 0 and len(all_dfs_list) >= 2:
                 raise ValueError(f"'timestamp' column required in DataFrame {i} for trimming with smart_trim_split.")

        # Filter out empty DataFrames before concatenation
        valid_dfs_list = [d for d in all_dfs_list if not d.empty]
        if not valid_dfs_list:
            return df if not dfs else (df,) + tuple(pd.DataFrame() for _ in dfs) # Return original or empty structures

        lens = [len(d_item) for d_item in valid_dfs_list]
        combined = pd.concat(valid_dfs_list, ignore_index=True)
        normalized = combined.copy()

        for col in price_cols:
            pct_change = combined[col].pct_change().fillna(0)
            expanding_volatility = pct_change.expanding().std(ddof=0).replace(0, 1) # ddof=0 for population std
            mean = combined[col].expanding().mean()
            std = combined[col].expanding().std(ddof=0).replace(0, 1)
            z_normalized = (combined[col] - mean) / std
            normalized[col] = z_normalized * expanding_volatility

        for col in volume_cols:
            log_volume = np.log1p(combined[col]) # log1p for log(1+x), handles zero volumes
            log_mean = log_volume.expanding().mean()
            log_std = log_volume.expanding().std(ddof=0).replace(0, 1)
            z_log_normalized = (log_volume - log_mean) / log_std
            normalized[col] = z_log_normalized

        split_points = [0] + list(pd.Series(lens).cumsum())
        normalized_dfs_out = []
        original_dfs_iter = iter(all_dfs_list) # To map results back to original number of DFs

        processed_idx = 0
        for _ in all_dfs_list: # Iterate through original number of DFs
            current_original_df = next(original_dfs_iter)
            if current_original_df.empty:
                normalized_dfs_out.append(pd.DataFrame(columns=current_original_df.columns)) # Keep structure for empty
            else:
                start_idx = split_points[processed_idx]
                end_idx = split_points[processed_idx+1]
                normalized_dfs_out.append(normalized.iloc[start_idx:end_idx].reset_index(drop=True))
                processed_idx += 1


        if trim > 0 and len(normalized_dfs_out) >= 2:
            # Apply smart trim only if both DFs for trimming are non-empty
            if not normalized_dfs_out[0].empty and not normalized_dfs_out[1].empty:
                normalized_dfs_out[0], normalized_dfs_out[1] = FTSEDataSet._smart_trim_split(
                    normalized_dfs_out[0].copy(), normalized_dfs_out[1].copy(), trim_ratio=trim
                )
            elif not normalized_dfs_out[0].empty : # Only first DF exists, apply head trim
                 n_total = len(normalized_dfs_out[0])
                 i1a = int(n_total * trim)
                 normalized_dfs_out[0] = normalized_dfs_out[0].iloc[i1a:].reset_index(drop=True)

        elif trim > 0 and len(normalized_dfs_out) == 1 and not normalized_dfs_out[0].empty:
            df_to_trim = normalized_dfs_out[0]
            n_total = len(df_to_trim)
            i1a = int(n_total * trim)
            normalized_dfs_out[0] = df_to_trim.iloc[i1a:].reset_index(drop=True)

        return normalized_dfs_out[0] if len(normalized_dfs_out) == 1 else tuple(normalized_dfs_out)

    @staticmethod
    def _combine_dataframes(*dfs: pd.DataFrame) -> pd.DataFrame:
        if not dfs: return pd.DataFrame()
        valid_dfs = [df for df in dfs if df is not None and not df.empty]
        if not valid_dfs: return pd.DataFrame()
        try:
            combined_df = pd.concat(valid_dfs, ignore_index=True)
            if 'timestamp' in combined_df.columns:
                combined_df.sort_values(by='timestamp', inplace=True)
                combined_df.reset_index(drop=True, inplace=True)
            return combined_df
        except Exception as e:
            raise Exception(f"Error combining DataFrames: {e}")

    @staticmethod
    def _extract_dir_data_internal(dir_path: str, trim_ratio_for_norm: float) -> Tuple[List[pd.DataFrame], List[pd.DataFrame], Dict[str, int]]:
        if not os.path.isdir(dir_path):
            raise FileNotFoundError(f"Directory {dir_path} does not exist.")

        symbols = set()
        file_map = {} # stores symbol -> (train_path, test_path)

        for file_name in os.listdir(dir_path):
            if '_' not in file_name or not file_name.endswith('.csv'):
                continue

            parts = file_name.rsplit('_', 1)
            symbol_name = parts[0]
            file_type = parts[1].replace('.csv', '')

            if symbol_name not in file_map:
                file_map[symbol_name] = {'train': None, 'test': None}

            if file_type == 'train':
                file_map[symbol_name]['train'] = os.path.join(dir_path, file_name)
            elif file_type == 'test':
                file_map[symbol_name]['test'] = os.path.join(dir_path, file_name)

        train_dfs_list = []
        test_dfs_list = []
        processed_symbols_map = {}
        symbol_idx = 0

        for symbol, paths in file_map.items():
            if paths['train'] and paths['test']:
                try:
                    raw_train_df = FTSEDataSet._load_check_csv(paths['train'])
                    raw_test_df = FTSEDataSet._load_check_csv(paths['test'])

                    # Validate train/test temporal order before normalization/trimming
                    # FTSEDataSet.is_valid_train_test_data(raw_test_df, raw_train_df) # Raw check

                    # Normalize and trim. Applied per stock pair.
                    norm_train_df, norm_test_df = FTSEDataSet._mask_normalize_dataframe(
                        raw_train_df.copy(), raw_test_df.copy(), trim=trim_ratio_for_norm
                    )

                    # Post-normalization validation (optional, but good)
                    if not norm_train_df.empty and not norm_test_df.empty:
                         FTSEDataSet.is_valid_train_test_data(norm_test_df, norm_train_df) # Check after processing
                         train_dfs_list.append(norm_train_df)
                         test_dfs_list.append(norm_test_df)
                         processed_symbols_map[symbol] = symbol_idx
                         symbol_idx += 1
                    else:
                        print(f"Warning: Normalization/trimming resulted in empty DataFrame for {symbol}. Skipping.")

                except AssertionError as ae:
                     print(f"AssertionError processing symbol {symbol}: {ae}. Skipping.")
                except Exception as e:
                    print(f"Error loading/processing data for stock {symbol}: {e}. Skipping.")
            else:
                print(f"Warning: Missing train or test file for symbol {symbol}. Skipping.")

        if not train_dfs_list and not test_dfs_list:
            print(f"Warning: No valid train/test pairs processed from directory {dir_path}.")

        return train_dfs_list, test_dfs_list, processed_symbols_map


    def load(self, binary: bool = True): # binary default from old interface
        print(f"Loading data from: {self.path}, type: {self.source_type}, trim: {self.trim_ratio}")

        if self.source_type == 'dir':
            all_train_dfs, all_test_dfs, _ = FTSEDataSet._extract_dir_data_internal(self.path, self.trim_ratio)

            self.processed_train_df = FTSEDataSet._combine_dataframes(*all_train_dfs)
            self.processed_test_df = FTSEDataSet._combine_dataframes(*all_test_dfs)

            if self.processed_train_df.empty and self.processed_test_df.empty:
                print("Warning: No data loaded from directory after processing.")
                self.df_returns = pd.DataFrame()
            elif self.processed_train_df.empty:
                 self.df_returns = self.processed_test_df.copy()
            elif self.processed_test_df.empty:
                 self.df_returns = self.processed_train_df.copy()
            else: # Both have data
                # For df_returns, we might want to combine them if some old code expects a single df.
                # However, for get_loaders, it's better to keep them separate.
                # Let df_returns be the train part for now, or a combination if needed elsewhere.
                self.df_returns = self.processed_train_df.copy()

            print(f"Loaded from directory. Train shape: {self.processed_train_df.shape}, Test shape: {self.processed_test_df.shape}")

        elif self.source_type == 'file':
            raw_df = FTSEDataSet._load_check_csv(self.path)

            # Original NaN dropping logic (pre-normalization)
            raw_df.dropna(axis=1, how='all', inplace=True) # Drop cols with all NaN
            raw_df.dropna(axis=0, how='all', inplace=True) # Drop rows with all NaN
            # Consider ffill/bfill for remaining NaNs before normalization if norm function is sensitive
            # The _mask_normalize_dataframe is robust to some NaNs due to expanding functions, but clean data is better.
            raw_df.ffill(inplace=True).bfill(inplace=True)


            self.df_returns = FTSEDataSet._mask_normalize_dataframe(raw_df.copy(), trim=self.trim_ratio)
            self.processed_train_df = self.df_returns # For file source, all data is initially train
            self.processed_test_df = pd.DataFrame() # Test will be split in get_loaders

            print(f"Processed data from file. Shape: {self.df_returns.shape}")
        else:
            raise ValueError(f"Unsupported source_type: {self.source_type}. Must be 'file' or 'dir'.")

        if self.df_returns is None or self.df_returns.empty:
            if not (self.source_type == 'dir' and (not self.processed_train_df.empty or not self.processed_test_df.empty)):
                 print("Warning: df_returns is empty after loading and not a dir load with split data.")
                 # return None # Or let it proceed and fail in get_loaders if data truly missing

        # Target binarization (applies to 'close' column)
        target_col = 'close' # Standardized to lowercase
        if binary:
            dfs_to_binarize = []
            if self.processed_train_df is not None and not self.processed_train_df.empty:
                dfs_to_binarize.append(self.processed_train_df)
            if self.processed_test_df is not None and not self.processed_test_df.empty:
                dfs_to_binarize.append(self.processed_test_df)

            # Also binarize df_returns if it's distinct and used
            if self.df_returns is not None and not self.df_returns.empty and \
               not any(self.df_returns is df for df in dfs_to_binarize): # Avoid double binarization if same object
                dfs_to_binarize.append(self.df_returns)

            binarized_count = 0
            for df_to_bin in dfs_to_binarize:
                if target_col in df_to_bin.columns:
                    if pd.api.types.is_numeric_dtype(df_to_bin[target_col]):
                        df_to_bin[target_col] = df_to_bin[target_col].apply(lambda x: 1 if x > 0 else 0)
                        binarized_count +=1
                    else:
                        print(f"Warning: Target column '{target_col}' in a DataFrame is not numeric. Skipping binarization for it.")
                else:
                    print(f"Warning: Target column '{target_col}' not found in a DataFrame. Skipping binarization for it.")
            if binarized_count > 0:
                 print(f"Converted '{target_col}' to binary for {binarized_count} DataFrame(s).")

        return self.df_returns


    def get_loaders(self, batch_size=16, n_test=1000, device='cpu'):
        train_df_for_loader = None
        test_df_for_loader = None

        if self.source_type == 'file':
            if self.processed_train_df is None or self.processed_train_df.empty:
                print("File data not loaded or empty. Calling load() first.")
                self.load()
            if self.processed_train_df is None or self.processed_train_df.empty:
                 raise ValueError("Data loading failed or resulted in empty dataframe for file source.")

            full_data_df = self.processed_train_df
            if len(full_data_df) <= n_test:
                raise ValueError(f"n_test ({n_test}) is too large for the dataset size ({len(full_data_df)}) from file.")
            train_df_for_loader = full_data_df[:-n_test]
            test_df_for_loader = full_data_df[-n_test:]

        elif self.source_type == 'dir':
            if (self.processed_train_df is None or self.processed_train_df.empty) and \
               (self.processed_test_df is None or self.processed_test_df.empty):
                print("Directory data not loaded or empty. Calling load() first.")
                self.load()

            # After load, one or both might still be empty if dir had no valid data
            if (self.processed_train_df is None or self.processed_train_df.empty) and \
               (self.processed_test_df is None or self.processed_test_df.empty):
                raise ValueError("Data loading from directory failed or resulted in all empty dataframes.")

            train_df_for_loader = self.processed_train_df if self.processed_train_df is not None else pd.DataFrame()
            test_df_for_loader = self.processed_test_df if self.processed_test_df is not None else pd.DataFrame()

            if train_df_for_loader.empty and test_df_for_loader.empty:
                 raise ValueError("Both processed train and test DataFrames are empty for dir source.")


        else: # Should have been caught by load()
            raise ValueError(f"Unsupported source_type in get_loaders: {self.source_type}")

        target_col = 'close'
        feature_cols = ['open', 'high', 'low', 'volume']

        # Prepare train_features, train_labels
        if not train_df_for_loader.empty:
            missing_train_cols = [col for col in feature_cols + [target_col] if col not in train_df_for_loader.columns]
            if missing_train_cols:
                raise ValueError(f"Missing required columns in training data for get_loaders: {missing_train_cols}")
            train_features_np = train_df_for_loader[feature_cols].values
            train_labels_np = train_df_for_loader[target_col].values
        else: # Handle empty train_df case
            print("Warning: Training data is empty for get_loaders.")
            train_features_np = np.array([]).reshape(0, len(feature_cols)) # Empty array with correct feature dimension
            train_labels_np = np.array([])

        # Prepare test_features, test_labels
        if not test_df_for_loader.empty:
            missing_test_cols = [col for col in feature_cols + [target_col] if col not in test_df_for_loader.columns]
            if missing_test_cols:
                raise ValueError(f"Missing required columns in testing data for get_loaders: {missing_test_cols}")
            test_features_np = test_df_for_loader[feature_cols].values
            test_labels_np = test_df_for_loader[target_col].values
        else: # Handle empty test_df case
            print("Warning: Testing data is empty for get_loaders.")
            test_features_np = np.array([]).reshape(0, len(feature_cols))
            test_labels_np = np.array([])

        # Create TensorDatasets
        # Ensure tensors are created even if data is empty, DataLoader handles empty datasets.
        train_data = data_utils.TensorDataset(
            torch.tensor(train_features_np).float().to(device),
            torch.tensor(train_labels_np).float().to(device)
        )
        test_data = data_utils.TensorDataset(
            torch.tensor(test_features_np).float().to(device),
            torch.tensor(test_labels_np).float().to(device)
        )

        train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=False) # shuffle=False for time series
        test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

        return train_dataloader, test_dataloader

    def is_valid_class(self) -> bool: # Override from Valid_Data_Class
        """Basic validation for FTSEDataSet instance."""
        if not self.path or not self.source_type:
            print("FTSEDataSet Validation failed: path or source_type not set.")
            return False
        if self.source_type not in ['file', 'dir']:
            print(f"FTSEDataSet Validation failed: invalid source_type '{self.source_type}'.")
            return False
        # After load, one might check:
        # if self.df_returns is None: return False
        return True

# The original print statement after FTSEDataSet class definition
# print("FTSEDataSet class updated to work with any stock data!")
# Can be updated or removed. Let's update it.
print("FTSEDataSet class has been updated and integrated with new normalization and loading features.")

# The user\'s other classes (like Data_Day_Hourly_StocksPrice) would follow here.
# If Data_Day_Hourly_StocksPrice is now largely redundant because its static methods
# for loading/normalization/trimming are ported into FTSEDataSet,
# it might be refactored or removed if its instance methods (evaluate, visualize)
# are not independently needed or can also be adapted/moved.
# For now, this edit focuses on FTSEDataSet.
# The original TRIM = 0.1 definition from user\'s new code block is now at the top.
# The original Valid_Data_Class from user\'s new code block is now at the top.

# ... (rest of the original file, including Data_Day_Hourly_StocksPrice if kept)
