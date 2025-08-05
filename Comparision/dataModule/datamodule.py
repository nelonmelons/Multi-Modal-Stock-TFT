import pandas as pd
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader


class NumericDataModule:
    """
    A simple DataModule to load time series data as numeric tensors without PyTorch-Forecasting.
    Splits data by date, selects numeric features, and returns DataLoaders.
    Handles single or multi-target predictions based on column names.
    """
    def __init__(
        self,
        feature_df: pd.DataFrame,
        split_date: str,
        batch_size: int,
        date_col: str = "date",
        target_col: str = "target",  # Can be a single column or a prefix like "target_"
        shuffle: bool = True,
        num_workers: int = 0,
    ):
        # Prepare DataFrame
        self.df = feature_df.copy()
        if date_col not in self.df.columns:
            raise ValueError(f"Date column '{date_col}' not found in DataFrame")
        # Convert date column to datetime
        self.date_col = date_col
        self.df[self.date_col] = pd.to_datetime(self.df[self.date_col], errors='coerce')
        if self.df[self.date_col].isna().all():
            raise ValueError(f"Column '{date_col}' could not be parsed as dates")
        # Store parameters
        self.split_date = pd.to_datetime(split_date)
        self.batch_size = batch_size
        self.target_identifier = target_col  # Use this to find target columns
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.target_cols = []

    def setup(self):
        # Split into train and validation sets
        train_df = self.df[self.df[self.date_col] <= self.split_date].copy()
        val_df = self.df[self.df[self.date_col] > self.split_date].copy()
        self.val_df = val_df
        
        print(f"Original data: {len(self.df)} rows")
        print(f"Train data (before cleaning): {len(train_df)} rows")
        print(f"Val data (before cleaning): {len(val_df)} rows")

        # Identify target columns based on the identifier (prefix or full name)
        if self.df.columns.str.startswith(self.target_identifier).any():
            self.target_cols = sorted([c for c in self.df.columns if c.startswith(self.target_identifier)])
            print(f"Found {len(self.target_cols)} target columns with prefix '{self.target_identifier}': {self.target_cols}")
        elif self.target_identifier in self.df.columns:
            self.target_cols = [self.target_identifier]
            print(f"Found single target column: '{self.target_identifier}'")
        else:
            raise ValueError(f"No target columns found with identifier '{self.target_identifier}'")

        # Drop missing targets
        train_df = train_df.dropna(subset=self.target_cols)
        val_df = val_df.dropna(subset=self.target_cols)
        
        print(f"Train data (after target cleaning): {len(train_df)} rows")
        print(f"Val data (after target cleaning): {len(val_df)} rows")
        
        # Select numeric features only
        train_num = train_df.select_dtypes(include=[np.number])
        val_num = val_df.select_dtypes(include=[np.number])
        
        # Check targets are present in numeric data
        if not all(c in train_num.columns for c in self.target_cols):
            missing = [c for c in self.target_cols if c not in train_num.columns]
            raise ValueError(f"Target columns {missing} not found in numeric data.")
        
        # Separate features and target
        feat_cols = [c for c in train_num.columns if c not in self.target_cols]
        print(f"Feature columns ({len(feat_cols)}): {feat_cols[:10]}...")  # Show first 10
        
        X_train = torch.tensor(train_num[feat_cols].values, dtype=torch.float32)
        y_train = torch.tensor(train_num[self.target_cols].values, dtype=torch.float32)
        X_val = torch.tensor(val_num[feat_cols].values, dtype=torch.float32)
        y_val = torch.tensor(val_num[self.target_cols].values, dtype=torch.float32)

        # Ensure y has 2 dimensions
        if y_train.ndim == 1:
            y_train = y_train.unsqueeze(-1)
        if y_val.ndim == 1:
            y_val = y_val.unsqueeze(-1)
        
        print(f"Train tensors: X={X_train.shape}, y={y_train.shape}")
        print(f"Val tensors: X={X_val.shape}, y={y_val.shape}")
        
        # Create DataLoaders
        self.train_loader = DataLoader(
            TensorDataset(X_train, y_train),
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
        )
        self.val_loader = DataLoader(
            TensorDataset(X_val, y_val),
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
        print(f"✓ NumericDataModule set up: {len(self.train_loader)} train batches, {len(self.val_loader)} val batches")
