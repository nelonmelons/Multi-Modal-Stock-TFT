#!/usr/bin/env python3
"""
Baseline TFT Implementation with Simple Features

This implements a proper TFT architecture following the original paper:
- Variable Selection Networks (VSNs)
- Gated Residual Networks (GRNs) 
- Static covariate encoders
- LSTM encoder-decoder
- Interpretable multi-head attention
- Quantile prediction outputs

Uses simple baseline features:
- Static: symbol, sector
- Known future: day_of_week, month, time_idx
- Unknown past: OHLCV, simple technical indicators
"""

# =============================================================================
# CONFIGURATION - Define default stock symbols
# =============================================================================
# Default stock symbols to use when none are specified
DEFAULT_SYMBOLS = ['NFLX']  # Primary symbol(s) for data creation - UPDATED
# =============================================================================

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Add path for data module
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


class GatedResidualNetwork(nn.Module):
    """Gated Residual Network (GRN) - core TFT component."""
    
    def __init__(self, input_size: int, hidden_size: int, output_size: int, 
                 dropout: float = 0.1, context_size: Optional[int] = None):
        super().__init__()
        
        self.input_size = input_size
        self.output_size = output_size
        self.context_size = context_size
        self.hidden_size = hidden_size
        
        # Primary layers
        if context_size is not None:
            self.linear1 = nn.Linear(input_size + context_size, hidden_size)
        else:
            self.linear1 = nn.Linear(input_size, hidden_size)
            
        self.linear2 = nn.Linear(hidden_size, output_size)
        
        # Gating mechanism
        self.gate = nn.Linear(hidden_size, output_size)
        
        # Skip connection
        if input_size != output_size:
            self.skip = nn.Linear(input_size, output_size)
        else:
            self.skip = None
            
        # Normalization and dropout
        self.layer_norm = nn.LayerNorm(output_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, context: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Concatenate context if provided
        if context is not None:
            x = torch.cat([x, context], dim=-1)
            
        # Primary path
        hidden = F.relu(self.linear1(x))
        hidden = self.dropout(hidden)
        output = self.linear2(hidden)
        
        # Gating
        gate = torch.sigmoid(self.gate(hidden))
        gated_output = output * gate
        
        # Skip connection
        if self.skip is not None:
            residual = self.skip(x if context is None else x[..., :self.input_size])
        else:
            residual = x if context is None else x[..., :self.input_size]
            
        # Final output with layer norm
        return self.layer_norm(gated_output + residual)


class VariableSelectionNetwork(nn.Module):
    """Variable Selection Network (VSN) - core TFT component."""
    
    def __init__(self, input_size: int, num_inputs: int, hidden_size: int, 
                 dropout: float = 0.1, context_size: Optional[int] = None):
        super().__init__()
        
        self.input_size = input_size
        self.num_inputs = num_inputs
        self.hidden_size = hidden_size
        
        # Individual variable GRNs
        self.variable_grns = nn.ModuleList([
            GatedResidualNetwork(input_size, hidden_size, hidden_size, dropout, context_size)
            for _ in range(num_inputs)
        ])
        
        # Selection weights GRN
        self.selection_grn = GatedResidualNetwork(
            input_size * num_inputs, hidden_size, num_inputs, dropout, context_size
        )
        
        # Output GRN
        self.output_grn = GatedResidualNetwork(
            hidden_size, hidden_size, hidden_size, dropout
        )
        
    def forward(self, variables: torch.Tensor, 
                context: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            variables: [batch_size, seq_len, num_inputs, input_size] or [batch_size, num_inputs, input_size]
            context: [batch_size, seq_len, context_size] or [batch_size, context_size]
            
        Returns:
            selected_variables: [batch_size, seq_len, hidden_size] or [batch_size, hidden_size]
            selection_weights: [batch_size, seq_len, num_inputs] or [batch_size, num_inputs]
        """
        batch_size = variables.shape[0]
        has_time_dim = len(variables.shape) == 4
        
        if has_time_dim:
            seq_len = variables.shape[1]
            variables = variables.view(batch_size * seq_len, self.num_inputs, self.input_size)
            if context is not None:
                context = context.reshape(batch_size * seq_len, -1)
        
        # Process each variable through its GRN
        processed_vars = []
        for i, grn in enumerate(self.variable_grns):
            var = variables[:, i, :]  # [batch*seq, input_size]
            processed = grn(var, context)  # [batch*seq, hidden_size]
            processed_vars.append(processed)
        
        processed_vars = torch.stack(processed_vars, dim=1)  # [batch*seq, num_inputs, hidden_size]
        
        # Compute selection weights
        flattened = variables.view(variables.shape[0], -1)  # [batch*seq, num_inputs * input_size]
        selection_weights = self.selection_grn(flattened, context)  # [batch*seq, num_inputs]
        selection_weights = F.softmax(selection_weights, dim=-1)
        
        # Apply selection weights
        selected = torch.einsum('bnh,bn->bh', processed_vars, selection_weights)  # [batch*seq, hidden_size]
        
        # Final processing
        selected = self.output_grn(selected)
        
        # Reshape back if needed
        if has_time_dim:
            selected = selected.view(batch_size, seq_len, -1)
            selection_weights = selection_weights.view(batch_size, seq_len, -1)
            
        return selected, selection_weights


class InterpretableMultiHeadAttention(nn.Module):
    """Interpretable Multi-Head Attention for TFT."""
    
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        assert self.head_dim * num_heads == embed_dim
        
        # Single head attention for interpretability
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            query: [batch_size, seq_len, embed_dim]
            key: [batch_size, seq_len, embed_dim] 
            value: [batch_size, seq_len, embed_dim]
            mask: Optional attention mask
            
        Returns:
            output: [batch_size, seq_len, embed_dim]
            attention_weights: [batch_size, seq_len, seq_len]
        """
        batch_size, seq_len, _ = query.shape
        
        # Compute Q, K, V
        Q = self.query(query)  # [batch, seq_len, embed_dim]
        K = self.key(key)      # [batch, seq_len, embed_dim]
        V = self.value(value)  # [batch, seq_len, embed_dim]
        
        # Reshape for multi-head attention
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.head_dim)
        
        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Softmax attention weights
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        attended = torch.matmul(attention_weights, V)
        
        # Reshape and project output
        attended = attended.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.embed_dim
        )
        output = self.out_proj(attended)
        
        # Average attention weights across heads for interpretability
        avg_attention = attention_weights.mean(dim=1)  # [batch, seq_len, seq_len]
        
        return output, avg_attention


class BaselineTFT(nn.Module):
    """
    Baseline TFT implementation with proper architecture and simple features.
    
    Features used:
    - Static categoricals: symbol (encoded as ID)
    - Static reals: market_cap_normalized
    - Known future: day_of_week, month, time_idx
    - Unknown past: open, high, low, close, volume, returns, sma_20, rsi_14
    """
    
    def __init__(self, 
                 # Feature dimensions
                 num_static_categorical: int = 1,  # symbol
                 static_categorical_cardinalities: List[int] = [10],  # max 10 symbols
                 num_static_real: int = 1,  # market_cap
                 num_time_varying_categorical: int = 0,  # none for baseline
                 num_time_varying_real_known: int = 3,  # day_of_week, month, time_idx
                 num_time_varying_real_unknown: int = 8,  # OHLCV + returns + sma_20 + rsi_14
                 
                 # Architecture parameters
                 hidden_size: int = 64,
                 lstm_layers: int = 2,
                 attention_heads: int = 4,
                 dropout: float = 0.1,
                 
                 # Sequence parameters
                 encoder_length: int = 30,
                 prediction_length: int = 1,
                 
                 # Output parameters
                 quantiles: List[float] = [0.1, 0.25, 0.5, 0.75, 0.9]):
        
        super().__init__()
        
        self.hidden_size = hidden_size
        self.encoder_length = encoder_length
        self.prediction_length = prediction_length
        self.quantiles = quantiles
        self.num_quantiles = len(quantiles)
        
        # Static categorical embeddings
        self.static_categorical_embeddings = nn.ModuleList([
            nn.Embedding(cardinality, hidden_size)
            for cardinality in static_categorical_cardinalities
        ])
        
        # Static covariate encoders
        static_input_size = (num_static_categorical * hidden_size + num_static_real)
        self.static_encoder = GatedResidualNetwork(
            static_input_size, hidden_size, hidden_size, dropout
        )
        
        # Variable selection networks
        # Historical inputs (encoder)
        self.historical_vsn = VariableSelectionNetwork(
            input_size=1,  # Each variable is 1D after normalization
            num_inputs=num_time_varying_real_unknown,
            hidden_size=hidden_size,
            dropout=dropout,
            context_size=hidden_size  # Static context
        )
        
        # Future inputs (decoder) 
        self.future_vsn = VariableSelectionNetwork(
            input_size=1,
            num_inputs=num_time_varying_real_known,
            hidden_size=hidden_size,
            dropout=dropout,
            context_size=hidden_size
        )
        
        # LSTM encoder-decoder
        self.encoder_lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=lstm_layers,
            dropout=dropout if lstm_layers > 1 else 0,
            batch_first=True
        )
        
        self.decoder_lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=lstm_layers,
            dropout=dropout if lstm_layers > 1 else 0,
            batch_first=True
        )
        
        # Self-attention
        self.self_attention = InterpretableMultiHeadAttention(
            embed_dim=hidden_size,
            num_heads=attention_heads,
            dropout=dropout
        )
        
        # Post-attention processing
        self.post_attention_grn = GatedResidualNetwork(
            hidden_size, hidden_size, hidden_size, dropout
        )
        
        # Quantile prediction heads
        self.quantile_projections = nn.ModuleList([
            nn.Linear(hidden_size, 1) for _ in range(self.num_quantiles)
        ])
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(hidden_size)
        
    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Forward pass through baseline TFT.
        
        Args:
            batch: Dictionary containing:
                - static_categorical: [batch_size, num_static_categorical]
                - static_real: [batch_size, num_static_real] 
                - encoder_cont: [batch_size, encoder_length, num_time_varying_real_unknown]
                - decoder_cont: [batch_size, prediction_length, num_time_varying_real_known]
                
        Returns:
            Dictionary containing:
                - prediction: [batch_size, prediction_length, num_quantiles]
                - attention_weights: [batch_size, prediction_length, encoder_length]
                - static_weights: [batch_size, num_static_features]
                - historical_weights: [batch_size, encoder_length, num_time_varying_real_unknown]
                - future_weights: [batch_size, prediction_length, num_time_varying_real_known]
        """
        batch_size = batch['encoder_cont'].shape[0]
        
        # 1. Process static covariates
        static_embeddings = []
        
        # Static categorical embeddings
        if 'static_categorical' in batch:
            for i, embedding_layer in enumerate(self.static_categorical_embeddings):
                emb = embedding_layer(batch['static_categorical'][:, i])
                static_embeddings.append(emb)
        
        # Static real features
        if 'static_real' in batch:
            static_embeddings.append(batch['static_real'])
            
        # Combine static features
        if static_embeddings:
            static_input = torch.cat(static_embeddings, dim=-1)
            static_context = self.static_encoder(static_input)  # [batch_size, hidden_size]
        else:
            static_context = torch.zeros(batch_size, self.hidden_size, device=batch['encoder_cont'].device)
        
        # 2. Historical variable selection (encoder inputs)
        historical_inputs = batch['encoder_cont']  # [batch_size, encoder_length, num_historical]
        
        # Reshape for VSN: [batch_size, encoder_length, num_inputs, 1]
        historical_reshaped = historical_inputs.unsqueeze(-1)
        
        # Expand static context for sequence
        static_context_expanded = static_context.unsqueeze(1).expand(
            batch_size, self.encoder_length, -1
        )
        
        # Apply historical VSN
        historical_selected, historical_weights = self.historical_vsn(
            historical_reshaped, static_context_expanded
        )  # [batch_size, encoder_length, hidden_size]
        
        # 3. Future variable selection (decoder inputs)
        future_inputs = batch['decoder_cont']  # [batch_size, prediction_length, num_future]
        future_reshaped = future_inputs.unsqueeze(-1)
        
        static_context_future = static_context.unsqueeze(1).expand(
            batch_size, self.prediction_length, -1
        )
        
        future_selected, future_weights = self.future_vsn(
            future_reshaped, static_context_future
        )  # [batch_size, prediction_length, hidden_size]
        
        # 4. LSTM encoder-decoder
        # Encoder
        encoder_output, encoder_state = self.encoder_lstm(historical_selected)
        
        # Decoder with future inputs
        decoder_output, _ = self.decoder_lstm(future_selected, encoder_state)
        
        # 5. Self-attention
        # Concatenate encoder and decoder for attention
        full_sequence = torch.cat([encoder_output, decoder_output], dim=1)
        
        # Apply self-attention
        attended_output, attention_weights = self.self_attention(
            full_sequence, full_sequence, full_sequence
        )
        
        # Extract decoder part
        decoder_attended = attended_output[:, self.encoder_length:, :]
        
        # Get attention weights for decoder to encoder
        decoder_to_encoder_attention = attention_weights[:, self.encoder_length:, :self.encoder_length]
        
        # 6. Post-attention processing
        processed_output = self.post_attention_grn(decoder_attended)
        processed_output = self.layer_norm(processed_output)
        
        # 7. Quantile predictions
        quantile_outputs = []
        for quantile_projection in self.quantile_projections:
            quantile_pred = quantile_projection(processed_output)  # [batch_size, prediction_length, 1]
            quantile_outputs.append(quantile_pred)
        
        predictions = torch.cat(quantile_outputs, dim=-1)  # [batch_size, prediction_length, num_quantiles]
        
        return {
            'prediction': predictions,
            'attention_weights': decoder_to_encoder_attention,
            'static_weights': torch.ones(batch_size, 1),  # Placeholder
            'historical_weights': historical_weights,
            'future_weights': future_weights
        }


class QuantileLoss(nn.Module):
    """Quantile loss for TFT training."""
    
    def __init__(self, quantiles: List[float]):
        super().__init__()
        self.quantiles = quantiles
        
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            predictions: [batch_size, seq_len, num_quantiles]
            targets: [batch_size, seq_len]
            
        Returns:
            loss: scalar tensor
        """
        targets = targets.unsqueeze(-1)  # [batch_size, seq_len, 1]
        
        errors = targets - predictions  # [batch_size, seq_len, num_quantiles]
        
        losses = []
        for i, quantile in enumerate(self.quantiles):
            error = errors[:, :, i]
            loss = torch.where(
                error >= 0,
                quantile * error,
                (quantile - 1) * error
            )
            losses.append(loss)
        
        total_loss = torch.stack(losses, dim=-1).sum(dim=-1)  # [batch_size, seq_len]
        return total_loss.mean()


def create_baseline_data(symbols: List[str] = None, 
                        start_date: str = '2020-01-01',
                        end_date: str = '2023-12-31',
                        encoder_length: int = 30,
                        prediction_length: int = 1) -> Dict[str, torch.Tensor]:
    """
    Create simple baseline dataset for TFT training.
    
    Uses only basic features:
    - Static: symbol_id, market_cap
    - Known future: day_of_week, month, time_idx  
    - Unknown past: OHLCV, returns, sma_20, rsi_14
    """
    import yfinance as yf
    
    # Use default symbols if none provided
    if symbols is None:
        symbols = DEFAULT_SYMBOLS.copy()
    
    print(f"Creating baseline dataset for {symbols}")
    print(f"Date range: {start_date} to {end_date}")
    
    all_data = []
    symbol_to_id = {symbol: i for i, symbol in enumerate(symbols)}
    
    for symbol in symbols:
        print(f"Fetching data for {symbol}...")
        
        # Fetch stock data
        ticker = yf.Ticker(symbol)
        hist = ticker.history(start=start_date, end=end_date)
        
        if hist.empty:
            print(f"No data for {symbol}, skipping...")
            continue
            
        # Basic price features
        df = pd.DataFrame({
            'date': hist.index,
            'open': hist['Open'].values,
            'high': hist['High'].values, 
            'low': hist['Low'].values,
            'close': hist['Close'].values,
            'volume': hist['Volume'].values,
        })
        
        # Calculate returns
        df['returns'] = df['close'].pct_change()
        
        # Simple technical indicators
        df['sma_20'] = df['close'].rolling(20).mean()
        df['rsi_14'] = calculate_rsi(df['close'], 14)
        
        # Calendar features
        df['day_of_week'] = df['date'].dt.dayofweek
        df['month'] = df['date'].dt.month
        
        # Static features
        df['symbol'] = symbol
        df['symbol_id'] = symbol_to_id[symbol]
        
        # Mock market cap (in practice, get from API)
        # Use symbol position in list to assign different market caps
        symbol_index = symbol_to_id.get(symbol, 0)
        df['market_cap'] = 1.0 if symbol_index == 0 else 0.5 + (symbol_index * 0.1)
        
        # Time index
        df['time_idx'] = range(len(df))
        
        # Target (next day return)
        df['target'] = df['returns'].shift(-1)
        
        # Drop initial NaN values
        df = df.dropna().reset_index(drop=True)
        
        all_data.append(df)
    
    if not all_data:
        raise ValueError("No data loaded for any symbols")
    
    # Combine all symbols
    combined_df = pd.concat(all_data, ignore_index=True)
    print(f"\n📊 Combined dataset: {len(combined_df)} total rows across {len(all_data)} symbols")
    
    # Check data quality before normalization
    check_data_quality(combined_df, "After data loading")
    
    # Normalize features
    combined_df = normalize_features(combined_df)
    
    # Create sequences
    return create_sequences(combined_df, encoder_length, prediction_length)


def calculate_rsi(prices: pd.Series, window: int = 14) -> pd.Series:
    """Calculate RSI indicator."""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def check_data_quality(df: pd.DataFrame, stage: str) -> None:
    """Check for NaN and infinite values in the dataframe."""
    print(f"\n🔍 Data quality check at stage: {stage}")
    
    # Check for NaN values
    nan_counts = df.isnull().sum()
    if nan_counts.any():
        print(f"⚠️  Found NaN values:")
        for col, count in nan_counts[nan_counts > 0].items():
            print(f"   {col}: {count} NaN values")
    
    # Check for infinite values
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    inf_counts = {}
    for col in numeric_cols:
        inf_count = np.isinf(df[col]).sum()
        if inf_count > 0:
            inf_counts[col] = inf_count
    
    if inf_counts:
        print(f"⚠️  Found infinite values:")
        for col, count in inf_counts.items():
            print(f"   {col}: {count} infinite values")
    
    # Check data ranges
    print(f"📊 Data ranges for numeric columns:")
    for col in numeric_cols:
        if df[col].notna().any():
            min_val = df[col].min()
            max_val = df[col].max()
            mean_val = df[col].mean()
            print(f"   {col}: [{min_val:.4f}, {max_val:.4f}], mean: {mean_val:.4f}")


def normalize_features(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize numerical features with robust handling of NaN values."""
    df = df.copy()
    
    print("\n🔧 Starting feature normalization...")
    check_data_quality(df, "Before normalization")
    
    # Price features - use percentage change normalization
    price_features = ['open', 'high', 'low', 'close']
    for feature in price_features:
        df[f'{feature}_norm'] = df.groupby('symbol')[feature].pct_change()
        # Fill the first NaN value with 0 for each symbol
        df[f'{feature}_norm'] = df.groupby('symbol')[f'{feature}_norm'].transform(
            lambda x: x.fillna(0)
        )
    
    # Volume - log transform and normalize
    # Ensure volume is positive before log transform
    df['volume'] = df['volume'].clip(lower=1)  # Avoid log(0)
    df['volume_norm'] = np.log1p(df['volume'])
    df['volume_norm'] = df.groupby('symbol')['volume_norm'].transform(
        lambda x: (x - x.mean()) / (x.std() + 1e-8)
    )
    
    # Technical indicators - normalize with robust scaling
    df['sma_20_norm'] = df.groupby('symbol')['sma_20'].transform(
        lambda x: (x - x.mean()) / (x.std() + 1e-8)
    )
    
    # RSI is already 0-100, normalize to 0-1
    # Fill NaN RSI values with 50 (neutral)
    df['rsi_14'] = df['rsi_14'].fillna(50)
    df['rsi_14_norm'] = df['rsi_14'] / 100.0
    
    # Returns - use as is but fill NaN
    df['returns_norm'] = df['returns'].fillna(0)
    
    # Calendar features - normalize to 0-1
    df['day_of_week_norm'] = df['day_of_week'] / 6.0
    df['month_norm'] = (df['month'] - 1) / 11.0
    
    # Time index - normalize per symbol
    df['time_idx_norm'] = df.groupby('symbol')['time_idx'].transform(
        lambda x: (x - x.min()) / (x.max() - x.min() + 1e-8)
    )
    
    # Market cap - normalize
    df['market_cap_norm'] = (df['market_cap'] - df['market_cap'].mean()) / (df['market_cap'].std() + 1e-8)
    
    # Final cleanup - replace any remaining NaN/inf with 0
    normalized_cols = [col for col in df.columns if col.endswith('_norm')]
    for col in normalized_cols:
        df[col] = df[col].replace([np.inf, -np.inf], 0).fillna(0)
    
    check_data_quality(df, "After normalization")
    print("✅ Feature normalization completed")
    
    return df


def create_sequences(df: pd.DataFrame, encoder_length: int, prediction_length: int) -> Dict[str, torch.Tensor]:
    """Create sequences for TFT training."""
    print(f"\n🔧 Creating sequences with encoder_length={encoder_length}, prediction_length={prediction_length}")
    
    sequences = []
    
    for symbol in df['symbol'].unique():
        symbol_df = df[df['symbol'] == symbol].sort_values('time_idx').reset_index(drop=True)
        
        # Minimum sequence length
        min_length = encoder_length + prediction_length
        if len(symbol_df) < min_length:
            print(f"⚠️  Warning: {symbol} has insufficient data ({len(symbol_df)} < {min_length})")
            continue
        
        print(f"📊 Processing {symbol}: {len(symbol_df)} data points")
        
        # Check data quality for this symbol
        print(f"   Data quality for {symbol}:")
        encoder_features = ['open_norm', 'high_norm', 'low_norm', 'close_norm', 
                           'volume_norm', 'returns_norm', 'sma_20_norm', 'rsi_14_norm']
        decoder_features = ['day_of_week_norm', 'month_norm', 'time_idx_norm']
        
        for feature in encoder_features + decoder_features + ['target']:
            if feature in symbol_df.columns:
                nan_count = symbol_df[feature].isnull().sum()
                inf_count = np.isinf(symbol_df[feature]).sum()
                if nan_count > 0 or inf_count > 0:
                    print(f"     ⚠️  {feature}: {nan_count} NaN, {inf_count} inf values")
        
        # Create sliding windows
        valid_sequences = 0
        for i in range(len(symbol_df) - min_length + 1):
            encoder_data = symbol_df.iloc[i:i+encoder_length][encoder_features]
            decoder_data = symbol_df.iloc[i+encoder_length:i+encoder_length+prediction_length][decoder_features]
            target_data = symbol_df.iloc[i+encoder_length:i+encoder_length+prediction_length]['target']
            
            # Check for any NaN or inf values in this sequence
            if (encoder_data.isnull().any().any() or np.isinf(encoder_data.values).any() or
                decoder_data.isnull().any().any() or np.isinf(decoder_data.values).any() or
                target_data.isnull().any() or np.isinf(target_data.values).any()):
                continue  # Skip sequences with bad data
            
            sequence = {
                # Static features
                'static_categorical': [symbol_df.iloc[i]['symbol_id']],
                'static_real': [symbol_df.iloc[i]['market_cap_norm']],
                
                # Historical features (encoder)
                'encoder_cont': encoder_data.values,
                
                # Future features (decoder)
                'decoder_cont': decoder_data.values,
                
                # Target
                'target': target_data.values
            }
            
            sequences.append(sequence)
            valid_sequences += 1
        
        print(f"   ✅ Created {valid_sequences} valid sequences for {symbol}")
    
    if not sequences:
        raise ValueError("No valid sequences created - all data contains NaN or inf values")
    
    print(f"\n📊 Total sequences created: {len(sequences)}")
    
    # Convert to tensors
    batch = {}
    for key in sequences[0].keys():
        if key in ['static_categorical', 'static_real']:
            batch[key] = torch.tensor([seq[key] for seq in sequences], dtype=torch.float32)
        elif key == 'target':
            batch[key] = torch.tensor([seq[key] for seq in sequences], dtype=torch.float32)
        else:
            batch[key] = torch.tensor([seq[key] for seq in sequences], dtype=torch.float32)
    
    # Handle categorical as long
    if 'static_categorical' in batch:
        batch['static_categorical'] = batch['static_categorical'].long()
    
    # Final data quality check on tensors
    print("\n🔍 Final tensor quality check:")
    for key, tensor in batch.items():
        nan_count = torch.isnan(tensor).sum().item()
        inf_count = torch.isinf(tensor).sum().item()
        if nan_count > 0 or inf_count > 0:
            print(f"   ⚠️  {key}: {nan_count} NaN, {inf_count} inf values")
        else:
            print(f"   ✅ {key}: clean data")
    
    return batch


if __name__ == "__main__":
    # Test baseline TFT
    print("🔄 Testing Baseline TFT Implementation...")
    
    # Create test data
    try:
        batch = create_baseline_data(
            symbols=DEFAULT_SYMBOLS,  # Use centralized configuration
            start_date='2023-01-01',
            end_date='2023-06-30',
            encoder_length=30,
            prediction_length=1
        )
        print(f"✅ Created test dataset with {batch['encoder_cont'].shape[0]} samples")
    except Exception as e:
        print(f"❌ Failed to create test data: {e}")
        exit(1)
    
    # Initialize model
    model = BaselineTFT(
        static_categorical_cardinalities=[1],  # 1 symbol
        hidden_size=32,  # Smaller for testing
        encoder_length=30,
        prediction_length=1
    )
    
    print(f"✅ Created baseline TFT model with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Test forward pass
    try:
        with torch.no_grad():
            output = model(batch)
        
        print("✅ Forward pass successful!")
        print(f"   Prediction shape: {output['prediction'].shape}")
        print(f"   Attention weights shape: {output['attention_weights'].shape}")
        print(f"   Historical weights shape: {output['historical_weights'].shape}")
        print(f"   Future weights shape: {output['future_weights'].shape}")
        
    except Exception as e:
        print(f"❌ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
    
    # Test loss calculation
    try:
        criterion = QuantileLoss([0.1, 0.25, 0.5, 0.75, 0.9])
        loss = criterion(output['prediction'], batch['target'])
        print(f"✅ Loss calculation successful: {loss.item():.4f}")
    except Exception as e:
        print(f"❌ Loss calculation failed: {e}")
        exit(1)
    
    print("\n🎯 Baseline TFT implementation ready!")
    print("Key features:")
    print("  ✓ Proper TFT architecture with VSNs, GRNs, LSTM, attention")
    print("  ✓ Simple baseline features (OHLCV + calendar)")
    print("  ✓ Quantile loss for uncertainty estimation")
    print("  ✓ Interpretable attention weights")
    print("  ✓ Ready for training and evaluation")
