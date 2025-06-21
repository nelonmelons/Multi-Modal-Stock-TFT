"""
TFT Data Pipeline Module

A comprehensive data processing package for Temporal Fusion Transformer (TFT) models.
Provides end-to-end data fetching, feature engineering, and DataLoader creation.
"""

from .interface import get_data_loader, get_data_loader_with_module
from .fetch_stock import fetch_stock_data
from .fetch_events import fetch_events_data, show_corporate_actions_sample
from .fetch_news import fetch_news_embeddings, preload_embedding_model, check_model_cache_size, print_news_status_report
from .fetch_fred import fetch_fred_data, get_economic_features
from .compute_ta import compute_technical_indicators
from .build_features import build_features
from .datamodule import TFTDataModule
from .adaptive_loader import create_adaptive_dataloader

__version__ = "1.0.0"
__all__ = [
    "get_data_loader",
    "get_data_loader_with_module",
    "fetch_stock_data",
    "fetch_events_data", 
    "show_corporate_actions_sample",
    "fetch_news_embeddings",
    "print_news_status_report",
    "fetch_fred_data",
    "get_economic_features",
    "preload_embedding_model",
    "check_model_cache_size",
    "compute_technical_indicators",
    "build_features",
    "TFTDataModule",
    "create_adaptive_dataloader"
]
