import pandas as pd
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from datetime import timedelta
from typing import List, Optional, Tuple


class NumericDataModule:
    """
    A simple DataModule to load time series data as numeric tensors without PyTorch-Forecasting.
    Now supports explicit Train/Val/Test splits by date, purge/embargo guards, and separate loaders.
    Handles single or multi-target predictions based on column names.
    """
    def __init__(
        self,
        feature_df: pd.DataFrame,
        split_date: Optional[str] = None,
        batch_size: int = 32,
        date_col: str = "date",
        target_col: str = "target",  # Can be a single column or a prefix like "target_"
        shuffle: bool = True,
        num_workers: int = 0,
        # New explicit splits and leakage guards
        train_range: Optional[Tuple[str, str]] = None,
        val_range: Optional[Tuple[str, str]] = None,
        test_range: Optional[Tuple[str, str]] = None,
        embargo_days: int = 0,
        horizons: Optional[List[int]] = None,
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
        self.split_date = pd.to_datetime(split_date) if split_date else None
        self.batch_size = batch_size
        self.target_identifier = target_col  # Use this to find target columns
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.target_cols = []
        # New params
        self.train_range = tuple(pd.to_datetime(d) for d in train_range) if train_range else None
        self.val_range = tuple(pd.to_datetime(d) for d in val_range) if val_range else None
        self.test_range = tuple(pd.to_datetime(d) for d in test_range) if test_range else None
        self.embargo_days = int(embargo_days) if embargo_days else 0
        self.horizons = horizons or []

        # Placeholders exposed after setup()
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        self.trainval_loader = None
        self.train_df = None
        self.val_df = None
        self.test_df = None

    def _find_target_cols(self):
        # Identify target columns based on the identifier (prefix or full name)
        if self.df.columns.str.startswith(self.target_identifier).any():
            self.target_cols = sorted([c for c in self.df.columns if c.startswith(self.target_identifier)])
            print(f"Found {len(self.target_cols)} target columns with prefix '{self.target_identifier}': {self.target_cols}")
        elif self.target_identifier in self.df.columns:
            self.target_cols = [self.target_identifier]
            print(f"Found single target column: '{self.target_identifier}'")
        else:
            raise ValueError(f"No target columns found with identifier '{self.target_identifier}'")

    def _apply_purge_embargo(self, df: pd.DataFrame, split_start: pd.Timestamp) -> pd.DataFrame:
        """Apply purge and embargo to the training portion relative to a split_start date.
        Purge: drop any training rows whose label window (date -> date+h) overlaps validation/test start.
        Embargo: drop last N days before split_start from training.
        """
        if df.empty:
            return df
        train_mask = df[self.date_col] < split_start
        train_df = df[train_mask].copy()
        keep_mask = pd.Series(True, index=train_df.index)

        # Purge using max horizon if multiple horizons are given; else no-op
        if self.horizons:
            max_h = max(self.horizons)
            # Label window ends at date + max_h days; drop if >= split_start
            keep_mask &= (train_df[self.date_col] + pd.to_timedelta(max_h, unit='D') < split_start)
        # Embargo: drop last embargo_days prior to split_start
        if self.embargo_days and self.embargo_days > 0:
            embargo_start = split_start - pd.to_timedelta(self.embargo_days, unit='D')
            keep_mask &= (train_df[self.date_col] < embargo_start)

        cleaned_train = train_df[keep_mask].copy()
        # Combine back with non-train rows unchanged
        remainder_df = df[~train_mask].copy()
        result = pd.concat([cleaned_train, remainder_df], axis=0).sort_values(self.date_col)

        # Assertions/logs
        overlap_cnt = ((train_df[self.date_col] + pd.to_timedelta(max(self.horizons) if self.horizons else 0, unit='D')) >= split_start).sum()
        if overlap_cnt > 0:
            print(f"[leakage-guard] Purged {overlap_cnt} training rows due to label-window overlap with split {split_start.date()}")
        if self.embargo_days:
            print(f"[leakage-guard] Applied {self.embargo_days} day embargo before {split_start.date()}")
        return result

    def _to_loaders(self, train_df: pd.DataFrame, val_df: pd.DataFrame) -> Tuple[DataLoader, DataLoader]:
        # Drop missing targets
        train_df = train_df.dropna(subset=self.target_cols)
        val_df = val_df.dropna(subset=self.target_cols)

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
        print(f"Val/Test tensors: X={X_val.shape}, y={y_val.shape}")

        train_loader = DataLoader(
            TensorDataset(X_train, y_train),
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
        )
        val_loader = DataLoader(
            TensorDataset(X_val, y_val),
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
        return train_loader, val_loader

    def setup(self):
        # Backward compatibility: simple split if explicit ranges not provided
        if not (self.train_range and self.val_range and self.test_range):
            train_df = self.df[self.df[self.date_col] <= self.split_date].copy()
            val_df = self.df[self.df[self.date_col] > self.split_date].copy()
            self.val_df = val_df
            self.train_df = train_df

            print(f"Original data: {len(self.df)} rows")
            print(f"Train data (before cleaning): {len(train_df)} rows")
            print(f"Val data (before cleaning): {len(val_df)} rows")

            self._find_target_cols()
            self.train_loader, self.val_loader = self._to_loaders(train_df, val_df)
            print(f"✓ NumericDataModule set up: {len(self.train_loader)} train batches, {len(self.val_loader)} val batches")
            return

        # Explicit split path
        self._find_target_cols()

        # Apply purge/embargo against Val start and Test start
        df_guarded = self._apply_purge_embargo(self.df, self.val_range[0])
        df_guarded = self._apply_purge_embargo(df_guarded, self.test_range[0])

        # Build splits
        tr_mask = (df_guarded[self.date_col] >= self.train_range[0]) & (df_guarded[self.date_col] <= self.train_range[1])
        va_mask = (df_guarded[self.date_col] >= self.val_range[0]) & (df_guarded[self.date_col] <= self.val_range[1])
        te_mask = (df_guarded[self.date_col] >= self.test_range[0]) & (df_guarded[self.date_col] <= self.test_range[1])

        train_df = df_guarded[tr_mask].copy()
        val_df = df_guarded[va_mask].copy()
        test_df = df_guarded[te_mask].copy()

        print(f"Original data: {len(self.df)} rows")
        print(f"Train data window: {self.train_range[0].date()} → {self.train_range[1].date()} ({len(train_df)} rows)")
        print(f"Val data window:   {self.val_range[0].date()} → {self.val_range[1].date()} ({len(val_df)} rows)")
        print(f"Test data window:  {self.test_range[0].date()} → {self.test_range[1].date()} ({len(test_df)} rows)")

        # Store dataframes
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df

        # Primary loaders
        self.train_loader, self.val_loader = self._to_loaders(train_df, val_df)
        # Test loader (features based on train_df columns)
        _, self.test_loader = self._to_loaders(train_df, test_df)

        # Combined Train+Val for final training
        trainval_df = pd.concat([train_df, val_df], ignore_index=True).sort_values(self.date_col)
        self.trainval_loader, _ = self._to_loaders(trainval_df, test_df)

        # Assertions: no overlap
        assert (train_df[self.date_col].max() < self.val_range[0] - pd.to_timedelta(self.embargo_days, unit='D')), "Leakage: Train extends into embargo window before Val"
        assert (train_df[self.date_col].max() < self.test_range[0] - pd.to_timedelta(self.embargo_days, unit='D')), "Leakage: Train extends into embargo window before Test"
        print("✓ Explicit splits with purge/embargo configured")
