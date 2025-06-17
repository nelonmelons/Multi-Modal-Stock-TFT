import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from typing import override

TRIM = 0.1

class Valid_Data_Class(object): # Mixin
    test: list[pd.DataFrame]
    train: list[pd.DataFrame]
    validation: list[pd.DataFrame]
    symbols_test_train: dict[str, int]
    symbols_validation: dict[str, int]
    num_test_train: int
    num_validation: int
    num_stocks: int
    ignore_leakage: bool
    trim = TRIM

    def __init__(self):
        pass

    @staticmethod
    def is_valid_dataframe(df: pd.DataFrame) -> bool:
        """
        Check if the DataFrame is valid.

        Parameters:
        df (pd.DataFrame): The DataFrame to check.

        Returns:
        bool: True if the DataFrame is valid, False otherwise.

        Raises:
        AssertionError: If the DataFrame is not valid or does not contain the required columns.
        """
        c1 = isinstance(df, pd.DataFrame)
        c2 = not df.empty
        c3 = all(col in df.columns for col in ['open', 'high', 'low', 'close', 'volume'])

        Ca = c1 and c2 and c3

        assert Ca, "DataFrame is not a valid DataFrame or does not contain the required columns."

        # Corrected r1 check for numeric dtypes
        r1 = all(pd.api.types.is_numeric_dtype(df[col]) for col in ['open', 'high', 'low', 'close', 'volume'])

        if not r1:
            # we try to convert the columns to float
            try:
                df['open'] = df['open'].astype(float)
                df['high'] = df['high'].astype(float)
                df['low'] = df['low'].astype(float)
                df['close'] = df['close'].astype(float)
                df['volume'] = df['volume'].astype(float)
            except ValueError:
                return False # Return False if conversion fails

        # Re-check numeric types after attempted conversion, as astype might not change dtype if it fails for some elements
        # or if the column was already of a type that astype considers compatible but isn't strictly float/int.
        # For simplicity, we rely on the subsequent checks (c4, c5) to catch issues if conversion wasn't fully successful
        # in making data purely numeric and non-null. A more robust approach might re-evaluate numeric types here.

        c4 = all(df[col].notnull().all() for col in ['open', 'high', 'low', 'close', 'volume'])
        # Corrected c5: added .all() to the Series returned by apply()
        c5 = all(df[col].apply(lambda x: isinstance(x, (int, float))).all() for col in ['open', 'high', 'low', 'close', 'volume'])

        Cb = c4 and c5
        assert Cb, "DataFrame contains null values or non-numeric values in the required columns."

        return True


    @staticmethod
    def is_valid_train_test_data(test: pd.DataFrame, train: pd.DataFrame):
        """
        Check if the test and train DataFrames are valid.

        Parameters:
        test (pd.DataFrame): The test DataFrame.
        train (pd.DataFrame): The train DataFrame.

        Returns:
        bool: True if both DataFrames are valid, False otherwise.
        """
        # check if the timestamps is in the same order, and thaat the test DataFrame is after the train DataFrame
        c1 = isinstance(test, pd.DataFrame) and isinstance(train, pd.DataFrame)
        c2 = not test.empty and not train.empty

        Ca = c1 and c2
        assert Ca, "Test or train DataFrame is not a valid DataFrame or is empty."

        c5 = pd.to_datetime(test['timestamp']).min() > pd.to_datetime(train['timestamp']).max()
        c6 = pd.to_datetime(test['timestamp']).is_monotonic_increasing
        c7 = pd.to_datetime(train['timestamp']).is_monotonic_increasing

        Cb = c5 and c6 and c7
        assert Cb, "Test DataFrame is not after the train DataFrame or the timestamps are not in the same order."

    def is_valid_class(self) -> bool:
        """
        Check if the instance is valid.

        Returns:
        bool: True if the instance is valid, False otherwise.
        """
        # check if the instance is a subclass of dict and has the name 'Data'
        c1 =  isinstance(self, dict) # Use type(self) and isinstance(self, ...)

        Ca = c1
        assert Ca, "Instance is not a subclass of dict"

        # check if the test and train attributes are lists of DataFrames and have the same length
        c2 = isinstance(self.test, list) and all(isinstance(df, pd.DataFrame) for df in self.test) # Use self.test
        c3 = isinstance(self.train, list) and all(isinstance(df, pd.DataFrame) for df in self.train) # Use self.train
        c4 = isinstance(self.validation, list) and all(isinstance(df, pd.DataFrame) for df in self.validation) # Use self.validation
        c5 = len(self.test) == len(self.train)  # test and train should have the same length

        Cb = c2 and c3 and c4 and c5
        assert Cb, "Test, train, or validation attributes are not lists of DataFrames or have different lengths."

        # check if ignore_leakage is in the instance and is a boolean
        ignore_leakage = getattr(self, 'ignore_leakage', False) # Use self
        if not ignore_leakage:
            # Check if there is any leakage by checking symbols in test/train and validation DataFrames
            temp_test_symbols = set(self.symbols_test_train.keys()) # Use self.symbols_test_train
            temp_validation_symbols = set(self.symbols_validation.keys()) # Use self.symbols_validation
            c6 = temp_test_symbols.isdisjoint(temp_validation_symbols)
        else:
            c6 = True

        Cc = c6
        assert Cc, "There is leakage between test/train and validation DataFrames."


        return True




class Data_Day_Hourly_StocksPrice(dict, Valid_Data_Class):
    """
    A class to handle data loading and processing for stock market data.
    """
    def __init__(self,
                 test: list[pd.DataFrame],
                 train: list[pd.DataFrame],
                 validation: list[pd.DataFrame],
                 symbols_test_train: dict[str, int],
                 symbols_validation: dict[str, int],
                 ignore_leakage: bool = False,
                 ) -> None:
        """
        Initialize the Data class with a file path. Do not use this method directly.

        Parameters:
        test (list[pd.DataFrame]): List of test DataFrames.
        train (list[pd.DataFrame]): List of train DataFrames. Train and Test are to be expected to be lists of DataFrames, their order must match.
        validation (list[pd.DataFrame]): List of validation DataFrames.
        symbols_test_train (dict[str, int]): Dictionary mapping stock symbols to their indices for test and train data.
        symbols_validation (dict[str, int]): Dictionary mapping stock symbols to their indices for validation data.
        ignore_leakage (bool): If True, ignore leakage checks. Default is False.

        Raises:
        ValueError: If the provided data is not valid.
        ValueError: If the lengths of test and train DataFrames do not match.
        ValueError: If the lengths of validation DataFrames do not match.
        ValueError: If the provided data is not a list of DataFrames.
        """
        super().__init__({
            'test' : test, # list of test DataFrames
            'train' : train,  # list of train DataFrames
            # Train and Test are to be expected to be lists of DataFrames, their order must match
            'validation' : validation  # list of validation DataFrames
        })
        self.test = test
        self.train = train
        self.validation = validation

        self.symbols_test_train = symbols_test_train
        self.symbols_validation = symbols_validation

        self.num_test_train = len(test)
        self.num_validation = len(validation)
        self.num_stocks = self.num_test_train + self.num_validation

        self.ignore_leakage = ignore_leakage

    @staticmethod
    def from_dir(dir_path: str,
                 ignore_leakage: bool = False,
                 ) -> "Data_Day_Hourly_StocksPrice":
        """
        Load training and testing data from a directory containing CSV files.

        Parameters:
        dir_path (str): The path to the directory containing the CSV files.

        Returns:
        list: A list of tuples, each containing a test DataFrame and a train DataFrame.
        """
        if not os.path.isdir(dir_path):
            raise FileExistsError(f"Directory {dir_path} does not exist.")

        dir_path_train_test = os.path.join(dir_path, 'train-test')
        dir_path_validation = os.path.join(dir_path, 'validation')

        test, train, validation, symbols_test_train, symbols_validation = Data_Day_Hourly_StocksPrice.extract_dir_data(
            test_train_dir=dir_path_train_test,
            validation_dir=dir_path_validation
        )

        if not test or not train or not validation:
            raise ValueError("No data found in the specified directories.")

        ret = Data_Day_Hourly_StocksPrice(test=test, train=train, validation=validation,
                                          symbols_test_train=symbols_test_train,
                                          symbols_validation=symbols_validation,
                                          ignore_leakage=ignore_leakage)
        if ignore_leakage:
            print("WARNING: Leakage checks are ignored. This is not recommended for production use.")

        assert ret.is_valid_class()

        return ret

    @staticmethod
    def extract_dir_data(test_train_dir: str, validation_dir: str) -> tuple[list[pd.DataFrame], list[pd.DataFrame], list[pd.DataFrame], dict[str, int], dict[str, int]]:
        """
        Load training, testing, and validation data from directories containing CSV files.

        Parameters:
        test_train_dir (str): The path to the directory containing test and train CSV files.
        validation_dir (str): The path to the directory containing validation CSV files.

        Returns:
        tuple: A tuple containing lists of test DataFrames, train DataFrames,
               validation DataFrames, and dictionaries mapping stock symbols to their indices.
        """
        test_train_data, symbols_test_train = Data_Day_Hourly_StocksPrice.extract_dir_train_test_data(test_train_dir)
        validation_data, symbols_validation = Data_Day_Hourly_StocksPrice.extract_dir_validation_data(validation_dir)

        if not test_train_data or not validation_data:
            raise ValueError("No data found in the specified directories.")

        return [test[0] for test in test_train_data], [train[1] for train in test_train_data], [val for val in validation_data], \
                symbols_test_train, symbols_validation

    @staticmethod
    def combine_dataframes(df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
        """
        Combine two DataFrames by concatenating by stacking them vertically.

        Parameters:
        df1 (pd.DataFrame): The first DataFrame.
        df2 (pd.DataFrame): The second DataFrame.

        Returns:
        pd.DataFrame: The combined DataFrame.
        """
        if df1 is None or df2 is None:
            raise ValueError("One or both DataFrames are None.")
        try:
            combined_df = pd.concat([df1, df2], ignore_index=True)
            return combined_df
        except Exception as e:
            raise Exception(f"Error combining DataFrames: {e}")

    @staticmethod
    def smart_trim_split(
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        trim_ratio: float = 0.1,
        jump_threshold: int = 60 * 60 * 4
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Trims and splits combined data into train and test aligned to trading day boundaries.
        It trims the head of the combined data by trim_ratio (snapped to next day start),
        then adjusts the train/test split so that train ends at a day boundary and test starts at the next.

        Parameters:
        train_df (pd.DataFrame): Training DataFrame with 'timestamp' column.
        test_df (pd.DataFrame): Testing DataFrame with 'timestamp' column.
        trim_ratio (float): Fraction of total data to trim from the start.
        jump_threshold (int): Time gap (in seconds) that signals a new trading session.

        Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Cleanly split train/test DataFrames.
        """
        if 'timestamp' not in train_df.columns or 'timestamp' not in test_df.columns:
            raise ValueError("Both DataFrames must contain a 'timestamp' column.")


        full_df = pd.concat([train_df, test_df], ignore_index=True)
        n_total = len(full_df)
        i1a = int(n_total * trim_ratio)
        i2a = len(train_df)
        i3a = n_total


        timestamps = full_df['timestamp'].values
        jumps = [i for i in range(1, n_total) if timestamps[i] - timestamps[i - 1] > jump_threshold]
        if not jumps:
            raise ValueError("No trading day boundaries detected.")


        i1b_candidates = [j for j in jumps if j > i1a]
        if not i1b_candidates:
            raise ValueError("No valid trading day start found after trim point.")
        i1b = i1b_candidates[0]


        jumps_after_trim = [j for j in jumps if i1b <= j <= i2a]
        if not jumps_after_trim:
            raise ValueError("No valid day end before test split.")
        i2b = jumps_after_trim[-1]


        jumps_before_end = [j for j in jumps if i1b < j <= i3a]
        if not jumps_before_end:
            raise ValueError("No trading day end found in test range.")
        i3b = jumps_before_end[-1]


        trimmed_train = full_df.iloc[i1b:i2b].reset_index(drop=True)
        trimmed_test = full_df.iloc[i2b:i3b].reset_index(drop=True)

        return trimmed_train, trimmed_test

    @staticmethod
    def mask_normalize_dataframe(df: pd.DataFrame, *dfs, trim: float = TRIM):
        """
        Normalize stock time series with causal stats while preserving the volatility
        differences across stocks. Works for 'open', 'high', 'low', 'close', 'volume' columns.
        Optionally trims the first two DataFrames at day boundaries.
        """
        price_cols = ['open', 'high', 'low', 'close']
        volume_cols = ['volume']
        all_cols = price_cols + volume_cols

        all_dfs = [df] + list(dfs)
        lens = [len(d) for d in all_dfs]
        combined = pd.concat(all_dfs, ignore_index=True)

        normalized = combined.copy()

        # Normalize price columns (existing logic)
        for col in price_cols:
            # Causal percentage change
            pct_change = combined[col].pct_change().fillna(0)
            expanding_volatility = pct_change.expanding().std(ddof=0).replace(0, 1)

            # Causal z-normalization
            mean = combined[col].expanding().mean()
            std = combined[col].expanding().std(ddof=0).replace(0, 1)
            z_normalized = (combined[col] - mean) / std

            # Volatility-scaled z
            normalized[col] = z_normalized * expanding_volatility

        # Normalize volume columns (causal log-normalization)
        for col in volume_cols:
            # Log transform to handle volume's typical distribution
            log_volume = np.log1p(combined[col])  # log1p handles zero volumes

            # Causal z-normalization on log-transformed volume
            log_mean = log_volume.expanding().mean()
            log_std = log_volume.expanding().std(ddof=0).replace(0, 1)
            z_log_normalized = (log_volume - log_mean) / log_std

            # Keep in log-normalized space for better stability
            normalized[col] = z_log_normalized

        # Split back into original DataFrames
        split_points = [0] + list(pd.Series(lens).cumsum())
        normalized_dfs = [
            normalized.iloc[split_points[i]:split_points[i + 1]].reset_index(drop=True)
            for i in range(len(all_dfs))
        ]

        # Apply smart trim if enabled and at least 2 DataFrames
        if trim > 0 and len(normalized_dfs) >= 2:
            normalized_dfs[0], normalized_dfs[1] = Data_Day_Hourly_StocksPrice.smart_trim_split(
                normalized_dfs[0], normalized_dfs[1], trim_ratio=trim
            )

        return normalized_dfs[0] if len(normalized_dfs) == 1 else tuple(normalized_dfs)

    @staticmethod
    def load_check_csv(file_path: str) -> pd.DataFrame:
        """
        Load a CSV file and check if it contains the required columns.

        Parameters:
        file_path (str): The path to the CSV file.

        Returns:
        pd.DataFrame: The loaded DataFrame.
        Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the DataFrame does not contain the required columns.
        ValueError: If the DataFrame is not valid.
        """
        if not (file_path.endswith('.csv')):
            file_path += '.csv'

        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"File {file_path} does not exist.")

        dataframes = pd.read_csv(file_path)

        # make sure it has open, high, low, close, volume columns
        required_columns = ['open', 'high', 'low', 'close', 'volume', 'timestamp']
        assert all(col in dataframes.columns for col in required_columns), \
            f"DataFrame does not contain the required columns: {required_columns}"

        # check if the DataFrame is valid
        Data_Day_Hourly_StocksPrice.is_valid_dataframe(dataframes)

        return dataframes

    @staticmethod
    def extract_dir_train_test_data(dir_path: str) -> tuple[list[tuple[pd.DataFrame, pd.DataFrame]], dict[str, int]]:
        """
        Extract training and testing data from a directory containing CSV files.
        The files are {stock}_test.csv and {stock}_train.csv for each stock.

        Parameters:
        dir_path (str): The path to the directory containing the CSV files.

        Returns:
        tuple: A tuple containing a list of tuples (test DataFrame, train DataFrame) and a dictionary mapping stock symbols to their indices.
        """
        if not os.path.isdir(dir_path):
            raise FileNotFoundError(f"Directory {dir_path} does not exist.")

        symbols = [file_name.replace('_test.csv', '') for file_name in os.listdir(dir_path) if file_name.endswith('_test.csv')]

        # make sure that there are both test and train files for each stock
        if not symbols:
            raise ValueError("No test files found in the specified directory.")

        for symbol in symbols:
            train_file = os.path.join(dir_path, f"{symbol}_train.csv")
            test_file = os.path.join(dir_path, f"{symbol}_test.csv")
            if not (os.path.isfile(train_file) and os.path.isfile(test_file)):
                raise FileNotFoundError(f"Missing train or test file for stock {symbol}. Expected files: {train_file}, {test_file}")

        dataframes = []
        symbols_test_train = { symbol: i for i, symbol in enumerate(symbols) }

        for symbol in symbols:
            train_file = os.path.join(dir_path, f"{symbol}_train.csv")
            test_file = os.path.join(dir_path, f"{symbol}_test.csv")

            try:
                train_df = Data_Day_Hourly_StocksPrice.load_check_csv(train_file)
                test_df = Data_Day_Hourly_StocksPrice.load_check_csv(test_file)

                train_df, test_df= Data_Day_Hourly_StocksPrice.mask_normalize_dataframe(train_df, test_df)

                Data_Day_Hourly_StocksPrice.is_valid_train_test_data(test_df, train_df)

                if Data_Day_Hourly_StocksPrice.is_valid_dataframe(train_df) and Data_Day_Hourly_StocksPrice.is_valid_dataframe(test_df):
                    dataframes.append((test_df, train_df))
                else:
                    raise ValueError(f"Invalid DataFrame for stock {symbol}.")
            except Exception as e:
                print(f"Error loading data for stock {symbol}: {e}")
                continue
        if not dataframes:
            raise ValueError("No valid data found in the specified directory.")

        return dataframes, symbols_test_train

    @staticmethod
    def extract_dir_validation_data(dir_path: str) -> tuple[list[pd.DataFrame], dict[str, int]]:
        """
        Extract validation data from a directory containing CSV files.
        The files are also {stock}_test.csv and {stock}_train.csv for each stock.
        We would be concat'ing the data because we assume that we dont need to distinguish between test and train data for validation.

        Parameters:
        dir_path (str): The path to the directory containing the CSV files.

        Returns:
        tuple: A tuple containing a list of validation DataFrames and a dictionary mapping stock symbols to their indices.
        """
        if not os.path.isdir(dir_path):
            raise FileNotFoundError(f"Directory {dir_path} does not exist.")

        symbols = [file_name.replace('_train.csv', '') for file_name in os.listdir(dir_path) if file_name.endswith('_train.csv')]

        if not symbols:
            raise ValueError("No train files found in the specified directory.")

        for symbol in symbols:
            train_file = os.path.join(dir_path, f"{symbol}_train.csv")
            test_file = os.path.join(dir_path, f"{symbol}_test.csv")
            if not (os.path.isfile(train_file) and os.path.isfile(test_file)):
                raise FileNotFoundError(f"Missing train or test file for stock {symbol}. Expected files: {train_file}, {test_file}")

        dataframes = []
        symbols_validation = { symbol: i for i, symbol in enumerate(symbols) }

        for symbol in symbols:
            train_file = os.path.join(dir_path, f"{symbol}_train.csv")
            test_file = os.path.join(dir_path, f"{symbol}_test.csv")

            try:
                train_df = Data_Day_Hourly_StocksPrice.load_check_csv(train_file)
                test_df = Data_Day_Hourly_StocksPrice.load_check_csv(test_file)

                # concatenate the train and test DataFrames for validation
                validation_df = Data_Day_Hourly_StocksPrice.combine_dataframes(train_df, test_df)
                validation_df = Data_Day_Hourly_StocksPrice.mask_normalize_dataframe(validation_df)
                if Data_Day_Hourly_StocksPrice.is_valid_dataframe(validation_df):
                    dataframes.append(validation_df)
                else:
                    raise ValueError(f"Invalid DataFrame for stock {symbol}.")
            except Exception as e:
                print(f"Error loading data for stock {symbol}: {e}")
                continue
        if not dataframes:
            raise ValueError("No valid data found in the specified directory.")
        return dataframes, symbols_validation

    def get_from_symbol(self, stock_symbol: str, default=None) -> tuple[pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame], str] | None:
        """
        Get the DataFrame for the given stock symbol from the Data dictionary.

        Parameters:
        stock_symbol (str): The stock symbol to get the DataFrame for.
        default: The default value to return if the stock symbol is not found.

        Returns:
        tuple[pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame], str] | None: The DataFrame or DataFrames for the given stock symbol, and whether it is test/train or validation data.
        """
        if stock_symbol in self.symbols_test_train:
            index = self.symbols_test_train[stock_symbol]
            test_df = self.test[index]
            train_df = self.train[index]
            return (train_df, test_df), 'test/train'
        elif stock_symbol in self.symbols_validation:
            index = self.symbols_validation[stock_symbol]
            validation_df = self.validation[index]
            return validation_df, 'validation'
        else:
            if default is not None:
                return default
            else:
                raise KeyError(f"Stock symbol {stock_symbol} not found in the Data dictionary.")


    @override
    def get(self, key: str, default=None):
        """
        Get the value for the given key from the Data dictionary.

        we can now us data['symbols_test_train'] to get the test/train symbols, and data['symbols_validation'] to get the validation symbols.
        we can also use data['symbols_{stock_symbol}'] to get the test/train or validation DataFrame for a specific stock symbol.

        Parameters:
        key (str): The key to get the value for.
        default: The default value to return if the key is not found.

        Returns:
        pd.DataFrame | list[pd.DataFrame]: The value for the given key or the default value.
        or
        tuple[pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame], str] | None: The DataFrame or DataFrames for the given stock symbol, and whether it is test/train or validation data.
        """
        if not key.startswith('symbols_'):
            return super().get(key, default)
        # if the key starts with 'symbols_', return the corresponding symbols dictionary
        if key == 'symbols_test_train':
            return self.symbols_test_train
        elif key == 'symbols_validation':
            return self.symbols_validation
        else:
            # see if the symbols is in one of the dictionaries
            stock_symbol = key.replace('symbols_', '')
            return self.get_from_symbol(stock_symbol, default)


    def evaluate(self,
                 stock_symbol: str,
                 pred: dict[int, float],
                 error_types: list[str] = None) -> dict[str, float] | None:
        """
        Evaluate the stock data for a given stock symbol and predictions.
        If error_types is provided, it should be a list of error types to calculate. (default would be ['mse', 'mae', 'rmse'])

        Parameters:
        stock_symbol (str): The stock symbol to evaluate.
        pred (dict[int, float]): A dictionary mapping timestamps to predicted values.
        error_types (list[str]): A list of error types to calculate. Default is None, which means no errors will be calculated.
        Returns:
        dict[str, float] | None: A dictionary containing the error metrics and the profit estimate, or None if the stock symbol is not found.

        Raises:
        KeyError: If the stock symbol is not found in the Data dictionary.
        ValueError: If the stock symbol is not valid or if the DataFrame is not valid.
        """
        if error_types is None:
            error_types = ['mse', 'mae', 'rmse']

        data_s, type_of_data = self.get_from_symbol(stock_symbol)
        if data_s is None:
            raise KeyError(f"Stock symbol {stock_symbol} not found in the Data dictionary.")
        if isinstance(data_s, tuple):
            train_df, test_df = data_s
            data_df = Data_Day_Hourly_StocksPrice.combine_dataframes(test_df, train_df)
        else:
            data_df = data_s
        if not Data_Day_Hourly_StocksPrice.is_valid_dataframe(data_df):
            raise ValueError(f"DataFrame for stock symbol {stock_symbol} is not valid.")
        # we assume that the data_df is already normalized, so we can use it directly
        if not isinstance(pred, dict):
            raise ValueError("Predictions must be a dictionary mapping timestamps to predicted values.")

        # we assume that the pred dictionary is already normalized, so we can use it directly
        if not all(isinstance(k, int) and isinstance(v, (int, float)) for k, v in pred.items()):
            raise ValueError("Predictions must be a dictionary mapping timestamps to predicted values, with timestamps as integers and values as floats or integers.")
        if not all(col in data_df.columns for col in ['open', 'high', 'low', 'close', 'volume']):
            raise ValueError("DataFrame does not contain the required columns: ['open', 'high', 'low', 'close', 'volume']")
        if not all(k in data_df['timestamp'].values for k in pred.keys()):
            raise ValueError("Predictions must contain timestamps that are present in the DataFrame.")

        # we assume that the data_df is already normalized, so we can use it directly
        # we will calculate the error metrics based on the predictions and the actual values in the DataFrame
        actual_values = data_df.set_index('timestamp')['close'].to_dict()
        if not all(k in actual_values for k in pred.keys()):
            raise ValueError("Predictions must contain timestamps that are present in the DataFrame.")

        errors = {}
        for error_type in error_types:
            if error_type == 'mse':
                errors['mse'] = np.mean([(actual_values[k] - v) ** 2 for k, v in pred.items()])
            elif error_type == 'mae':
                errors['mae'] = np.mean([abs(actual_values[k] - v) for k, v in pred.items()])
            elif error_type == 'rmse':
                errors['rmse'] = np.sqrt(errors['mse'])
            else:
                raise ValueError(f"Unknown error type: {error_type}")
        # we will also calculate the profit estimate based on the predictions
        profit_estimate = sum(pred[k] - actual_values[k] for k in pred.keys() if k in actual_values)
        print("WARNING: PROFIT IS NOT CORRECT")
        errors['profit_estimate'] = profit_estimate
        return errors

    def visualize(self,
                  stock_symbol: str,
                  pred: dict[int, float] | None = None,
                  error_metrics: callable = None,
                  figsize: tuple[int, int] = (15, 10),
                  title_pre: str = None,
                  title: str = None,
                  title_post: str = None,
                  save_path: str = None,
                  show: bool = True,
                  ) -> None:

        if error_metrics is None:
            error_metrics = self.evaluate

        data_s, _ = self.get_from_symbol(stock_symbol)

        if isinstance(data_s, tuple):
            train_df, test_df = data_s
            data_df = Data_Day_Hourly_StocksPrice.combine_dataframes(test_df, train_df)
        else:
            data_df = data_s


        timestamps = pd.to_datetime(data_df['timestamp'])
        actual = data_df['close']

        plt.figure(figsize=figsize)
        plt.plot(timestamps, actual, label='Actual')

        if pred:
            pred_ts = pd.to_datetime(list(pred.keys()))
            pred_vals = list(pred.values())
            plt.plot(pred_ts, pred_vals, label='Predicted')
            errs = error_metrics(stock_symbol, pred)
            metrics_text = ", ".join(f"{k}: {v:.4f}" for k, v in errs.items())
            full_title = f"{title_pre or ''}{title or stock_symbol}{title_post or ''}\n{metrics_text}"
        else:
            full_title = f"{title_pre or ''}{title or stock_symbol}{title_post or ''}"

        plt.title(full_title)
        plt.xlabel('Timestamp')
        plt.ylabel('Normalized Close Price')
        plt.legend()

        if save_path:
            plt.savefig(save_path)
        if show:
            plt.show()
        plt.close()

    @override
    def __str__(self) -> str:
        """
        Return a string representation of the Data object.

        Returns:
        str: A string representation of the Data object.
        """
        return f"Data(test={len(self.test)}, train={len(self.train)}, validation={len(self.validation)}, num_stocks={self.num_stocks})"