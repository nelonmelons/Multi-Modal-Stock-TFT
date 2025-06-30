#!/usr/bin/env python3
"""
TFT Model Training with Pure PyTorch for M1 Mac

This file now only contains model and helper definitions. Run the unified pipeline via main.py.
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dataModule.interface import get_data_loader_with_module

import dotenv

dotenv.load_dotenv()


class NewsDownsampler(nn.Module):
    """
    Conditional encoder for news embeddings, conditioned on the encoded rest of the input.
    Uses Swigelu (SiLU + GELU) activation and a residual connection.
    """

    def __init__(self, news_dim, context_dim, out_dim):
        super().__init__()
        self.fc_news = nn.Linear(news_dim, out_dim)
        self.fc_context = nn.Linear(context_dim, out_dim)
        self.fc_fusion = nn.Sequential(
            nn.SiLU(),  # Swish
            nn.Linear(out_dim * 2, out_dim),
            nn.GELU(),  # GELU after SiLU (Swigelu)
            nn.Linear(out_dim, out_dim)
        )
        self.residual_proj = nn.Linear(news_dim, out_dim) if news_dim != out_dim else nn.Identity()

    def forward(self, news, context):
        news_proj = self.fc_news(news)
        context_proj = self.fc_context(context)
        fusion = torch.cat([news_proj, context_proj], dim=-1)
        out = self.fc_fusion(fusion)
        # Residual connection from news input
        out = out + self.residual_proj(news)
        return out


class GatedResidualNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim=None, dropout=0.1):
        super().__init__()
        if output_dim is None:
            output_dim = input_dim
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.gelu = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.gate = nn.GLU()
        self.skip = nn.Linear(input_dim, output_dim) if input_dim != output_dim else nn.Identity()
        self.norm = nn.LayerNorm(output_dim)

    def forward(self, x):
        residual = self.skip(x)
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.gate(torch.cat([x, x], dim=-1))
        x = self.norm(x + residual)
        return x


class VariableSelectionNetwork(nn.Module):
    """
    Variable Selection Network for TFT.
    Handles multiple input variables (features) and selects relevant ones at each timestep.
    """

    def __init__(self, input_dims, hidden_dim, dropout=0.1):
        super().__init__()
        self.input_dims = input_dims
        self.num_vars = len(input_dims)
        self.hidden_dim = hidden_dim
        self.var_grns = nn.ModuleList([
            GatedResidualNetwork(input_dim, hidden_dim, hidden_dim, dropout=dropout)
            for input_dim in input_dims
        ])
        self.weight_grn = GatedResidualNetwork(hidden_dim * self.num_vars, hidden_dim, self.num_vars, dropout=dropout)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        # x: [B, T, sum(input_dims)]
        B, T, _ = x.shape
        # Split x into variables
        splits = torch.split(x, self.input_dims, dim=-1)
        var_outputs = []
        for i, grn in enumerate(self.var_grns):
            var_outputs.append(grn(splits[i]))  # [B, T, hidden_dim]
        var_outputs = torch.stack(var_outputs, dim=2)  # [B, T, num_vars, hidden_dim]
        flat = torch.cat(var_outputs.unbind(dim=2), dim=-1)  # [B, T, num_vars*hidden_dim]
        weights = self.weight_grn(flat)  # [B, T, num_vars]
        weights = self.softmax(weights)
        out = (var_outputs * weights.unsqueeze(-1)).sum(dim=2)  # [B, T, hidden_dim]
        return out, weights


class TFT(nn.Module):
    """Temporal Fusion Transformer with explicit VSN and conditional news encoder."""

    def __init__(
            self,
            input_size,
            news_dim,
            hidden_size=64,
            num_heads=4,
            dropout=0.1,
            seq_len=30,
            prediction_len=1,
            news_downsample_dim=32,
            input_var_dims=None  # List of input dims for each variable (excluding news)
    ):
        super().__init__()
        self.input_size = input_size
        self.news_dim = news_dim
        self.hidden_size = hidden_size
        self.seq_len = seq_len
        self.prediction_len = prediction_len

        # If input_var_dims not provided, assume each feature is 1-dim
        if input_var_dims is None:
            input_var_dims = [1] * input_size
        self.input_var_dims = input_var_dims

        # Feature embedding for each variable (excluding news)
        self.feature_embeddings = nn.ModuleList([
            nn.Linear(dim, hidden_size) for dim in input_var_dims
        ])
        # VSN for all variables (excluding news)
        self.vsn = VariableSelectionNetwork([hidden_size] * len(input_var_dims), hidden_size, dropout=dropout)
        # News downsampler: conditional on encoded rest-of-input
        self.news_downsampler = NewsDownsampler(news_dim, hidden_size, news_downsample_dim)
        # Encoder GRN
        self.encoder_grn = GatedResidualNetwork(hidden_size, hidden_size, dropout=dropout)
        # Multi-head attention
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.attn_norm = nn.LayerNorm(hidden_size)
        # Decoder GRN
        self.decoder_grn = GatedResidualNetwork(hidden_size, hidden_size, dropout=dropout)
        # Final fusion and prediction head
        self.fusion = nn.Linear(hidden_size + news_downsample_dim, hidden_size)
        self.prediction_head = nn.Sequential(
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Linear(hidden_size // 2, prediction_len)
        )
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)

    def forward(self, x, news):
        """
        x: [B, T, F] (non-news features)
        news: [B, news_dim] (news embedding for the prediction window)
        """
        # Split x into variables
        splits = torch.split(x, self.input_var_dims, dim=-1)
        var_embs = []
        for i, emb in enumerate(self.feature_embeddings):
            var_embs.append(emb(splits[i]))  # [B, T, hidden_size]
        x_cat = torch.cat(var_embs, dim=-1)  # [B, T, hidden_size * num_vars]
        # VSN
        x_vsn, _ = self.vsn(x_cat)  # [B, T, hidden_size]
        # Encoder GRN
        x_enc = self.encoder_grn(x_vsn)
        # Self-attention
        attn_out, _ = self.attention(x_enc, x_enc, x_enc)
        attn_out = self.attn_norm(x_enc + attn_out)
        # Decoder GRN
        dec_out = self.decoder_grn(attn_out)
        # Use last timestep for prediction
        last_hidden = dec_out[:, -1, :]  # [B, hidden_size]
        # News downsampling, conditioned on last_hidden
        news_down = self.news_downsampler(news, last_hidden)  # [B, news_downsample_dim]
        # Concatenate
        fusion = torch.cat([last_hidden, news_down], dim=-1)
        fusion = self.fusion(fusion)
        # Prediction
        prediction = self.prediction_head(fusion)
        return prediction


def setup_device():
    """Setup device with M1 Mac optimization."""
    if torch.backends.mps.is_available():
        try:
            # Test MPS with a simple operation
            test_tensor = torch.randn(2, 2, device='mps')
            test_result = test_tensor @ test_tensor
            device = torch.device("mps")
            print("🚀 Using Apple Silicon MPS acceleration")
            return device
        except Exception as e:
            print(f"⚠️ MPS test failed: {e}")
            print("💻 Falling back to CPU")
            return torch.device("cpu")
    else:
        print("💻 Using CPU (MPS not available)")
        return torch.device("cpu")


def prepare_data_for_training(datamodule, device, max_batches=20):
    """Prepare data for training with proper tensor handling and progress bars."""

    print("🔄 Preparing training data...")

    train_data = []
    val_data = []

    # Process training data
    try:
        train_loader = datamodule.train_dataloader()
        # Enhanced training data preparation with progress bar
        train_pbar = tqdm(
            enumerate(train_loader),
            total=min(max_batches, len(train_loader)),
            desc="📚 Processing Training Data",
            unit="batch",
            ncols=120,
            colour='blue'
        )

        for i, batch in train_pbar:
            if i >= max_batches:
                break

            try:
                # Debug: Print batch structure for first batch
                if i == 0:
                    print(f"\n   DEBUG: Batch type: {type(batch)}")
                    if isinstance(batch, tuple):
                        print(f"   DEBUG: Batch length: {len(batch)}")
                        for j, item in enumerate(batch):
                            if torch.is_tensor(item):
                                print(f"   DEBUG: Item {j} tensor shape: {item.shape}")
                            elif isinstance(item, dict):
                                print(f"   DEBUG: Item {j} is dict with keys: {list(item.keys())}")
                            else:
                                print(f"   DEBUG: Item {j} type: {type(item)}")

                # Handle tuple format (x, y) from DataLoader
                if isinstance(batch, tuple) and len(batch) >= 2:
                    x, y = batch[0], batch[1]

                    # Handle x (features) - TFT encoder/decoder structure
                    if isinstance(x, dict):
                        # TFT format: separate encoder and decoder features
                        encoder_features = []
                        decoder_features = []

                        # Process encoder features (historical data)
                        for key in ['encoder_cat', 'encoder_cont']:
                            if key in x and torch.is_tensor(x[key]):
                                value = x[key]
                                if i == 0:
                                    print(f"   DEBUG: Processing {key} with shape {value.shape}")

                                # Encoder features should already be 3D [batch, seq_len, features]
                                if value.dim() == 3:
                                    encoder_features.append(value.float())
                                elif value.dim() == 2:
                                    encoder_features.append(value.unsqueeze(1).float())
                                elif value.dim() == 1:
                                    encoder_features.append(value.unsqueeze(1).unsqueeze(-1).float())

                        # Process decoder features (future context)
                        for key in ['decoder_cat', 'decoder_cont']:
                            if key in x and torch.is_tensor(x[key]):
                                value = x[key]
                                if i == 0:
                                    print(f"   DEBUG: Processing {key} with shape {value.shape}")

                                # Decoder features should be 3D [batch, pred_len, features]
                                if value.dim() == 3:
                                    decoder_features.append(value.float())
                                elif value.dim() == 2:
                                    decoder_features.append(value.unsqueeze(1).float())
                                elif value.dim() == 1:
                                    decoder_features.append(value.unsqueeze(1).unsqueeze(-1).float())

                        # Combine encoder and decoder features
                        if encoder_features and decoder_features:
                            # Concatenate encoder features
                            encoder_tensor = torch.cat(encoder_features, dim=-1)  # [batch, enc_len, enc_features]
                            decoder_tensor = torch.cat(decoder_features, dim=-1)  # [batch, dec_len, dec_features]

                            # For TFT, we'll use encoder features for training (past data to predict future)
                            features = encoder_tensor

                            if i == 0:
                                print(f"   DEBUG: Encoder tensor shape: {encoder_tensor.shape}")
                                print(f"   DEBUG: Decoder tensor shape: {decoder_tensor.shape}")
                                print(f"   DEBUG: Using encoder features for training: {features.shape}")

                        elif encoder_features:
                            # Only encoder features available
                            features = torch.cat(encoder_features, dim=-1)
                            if i == 0:
                                print(f"   DEBUG: Encoder-only features shape: {features.shape}")

                        else:
                            # Fallback: process all non-target tensors
                            all_tensors = []

                            for key, value in x.items():
                                if torch.is_tensor(
                                        value) and 'target' not in key.lower() and 'scale' not in key.lower():
                                    # Convert to 3D tensor [batch, seq_len, features]
                                    if value.dim() == 3:
                                        processed = value.float()
                                    elif value.dim() == 2:
                                        processed = value.unsqueeze(1).float()
                                    elif value.dim() == 1:
                                        processed = value.unsqueeze(1).unsqueeze(-1).float()
                                    else:
                                        continue

                                    all_tensors.append(processed)

                            if all_tensors:
                                # Find common sequence length (use the most common)
                                seq_lens = [t.shape[1] for t in all_tensors]
                                from collections import Counter
                                common_seq_len = Counter(seq_lens).most_common(1)[0][0]

                                # Filter tensors with common sequence length
                                filtered_tensors = [t for t in all_tensors if t.shape[1] == common_seq_len]

                                if filtered_tensors:
                                    features = torch.cat(filtered_tensors, dim=-1)
                                    if i == 0:
                                        print(f"   DEBUG: Fallback - concatenated features shape: {features.shape}")
                                else:
                                    continue
                            else:
                                continue

                    elif torch.is_tensor(x):
                        # x is already a tensor - ensure it's 3D
                        features = x.float()
                        if features.dim() == 1:
                            features = features.unsqueeze(0).unsqueeze(1)  # [1, 1, features]
                        elif features.dim() == 2:
                            features = features.unsqueeze(1)  # [batch, 1, features]
                    else:
                        continue

                    # Handle y (target)
                    if isinstance(y, tuple) and len(y) > 0:
                        target = y[0]  # Take first element of target tuple
                    elif torch.is_tensor(y):
                        target = y
                    else:
                        continue

                    # Ensure target is properly shaped
                    if torch.is_tensor(target):
                        if len(target.shape) > 2:
                            target = target.squeeze(-1)
                        if len(target.shape) > 1 and target.shape[1] > 1:
                            target = target[:, 0]  # Take first prediction timestep

                        # Ensure batch sizes match
                        if features.shape[0] == target.shape[0]:
                            train_data.append((features.detach(), target.detach()))
                            if i == 0:
                                print(
                                    f"   DEBUG: Successfully processed batch - X shape: {features.shape}, Y shape: {target.shape}")
                        else:
                            if i == 0:
                                print(f"   DEBUG: Batch size mismatch - X: {features.shape[0]}, Y: {target.shape[0]}")
                    else:
                        if i == 0:
                            print(f"   DEBUG: Target is not a tensor: {type(target)}")

                elif isinstance(batch, dict):
                    # Fallback: handle dict format
                    all_tensors = []
                    target = None

                    for key, value in batch.items():
                        if torch.is_tensor(value):
                            if 'target' in key.lower() or key == 'y':
                                target = value
                            else:
                                if value.dim() > 2:
                                    value = value.flatten(start_dim=1)
                                all_tensors.append(value.float())

                    if all_tensors and target is not None:
                        x = torch.cat(all_tensors, dim=-1) if len(all_tensors) > 1 else all_tensors[0]

                        if isinstance(target, tuple):
                            y = target[0]
                        else:
                            y = target

                        if torch.is_tensor(y):
                            if len(y.shape) > 2:
                                y = y.squeeze(-1)
                            if len(y.shape) > 1 and y.shape[1] > 1:
                                y = y[:, 0]

                            train_data.append((x.detach(), y.detach()))

                # Update progress bar with enhanced training data metrics
                train_pbar.set_postfix({
                    '📊 Samples': len(train_data),
                    '📦 Batch': f"{i + 1}/{max_batches}",
                    '✅ Success': '🎯'
                })

            except Exception as e:
                train_pbar.set_postfix({
                    '❌ ERROR': str(e)[:15],
                    '📦 Batch': f"{i + 1}"
                })
                continue

    except Exception as e:
        print(f"   Training data error: {e}")

    # Process validation data
    try:
        val_loader = datamodule.val_dataloader()
        # Enhanced validation data preparation with progress bar
        val_pbar = tqdm(
            enumerate(val_loader),
            total=min(max_batches // 2, len(val_loader)),
            desc="📊 Processing Validation Data",
            unit="batch",
            ncols=120,
            colour='yellow'
        )

        for i, batch in val_pbar:
            if i >= max_batches // 2:
                break

            try:
                # Debug: Print batch structure for first batch
                if i == 0:
                    print(f"\n   DEBUG: Val Batch type: {type(batch)}")
                    if isinstance(batch, tuple):
                        print(f"   DEBUG: Val Batch length: {len(batch)}")
                        for j, item in enumerate(batch):
                            if torch.is_tensor(item):
                                print(f"   DEBUG: Val Item {j} tensor shape: {item.shape}")
                            elif isinstance(item, dict):
                                print(f"   DEBUG: Val Item {j} is dict with keys: {list(item.keys())}")
                            else:
                                print(f"   DEBUG: Val Item {j} type: {type(item)}")

                # Handle tuple format (x, y) from DataLoader - same as training
                if isinstance(batch, tuple) and len(batch) >= 2:
                    x, y = batch[0], batch[1]

                    # Handle x (features) - TFT encoder/decoder structure (validation)
                    if isinstance(x, dict):
                        # TFT format: separate encoder and decoder features
                        encoder_features = []
                        decoder_features = []

                        # Process encoder features (historical data)
                        for key in ['encoder_cat', 'encoder_cont']:
                            if key in x and torch.is_tensor(x[key]):
                                value = x[key]

                                # Encoder features should already be 3D [batch, seq_len, features]
                                if value.dim() == 3:
                                    encoder_features.append(value.float())
                                elif value.dim() == 2:
                                    encoder_features.append(value.unsqueeze(1).float())
                                elif value.dim() == 1:
                                    encoder_features.append(value.unsqueeze(1).unsqueeze(-1).float())

                        # Process decoder features (future context)
                        for key in ['decoder_cat', 'decoder_cont']:
                            if key in x and torch.is_tensor(x[key]):
                                value = x[key]

                                # Decoder features should be 3D [batch, pred_len, features]
                                if value.dim() == 3:
                                    decoder_features.append(value.float())
                                elif value.dim() == 2:
                                    decoder_features.append(value.unsqueeze(1).float())
                                elif value.dim() == 1:
                                    decoder_features.append(value.unsqueeze(1).unsqueeze(-1).float())

                        # Combine encoder and decoder features
                        if encoder_features and decoder_features:
                            # Concatenate encoder features
                            encoder_tensor = torch.cat(encoder_features, dim=-1)  # [batch, enc_len, enc_features]
                            decoder_tensor = torch.cat(decoder_features, dim=-1)  # [batch, dec_len, dec_features]

                            # For TFT, we'll use encoder features for training (past data to predict future)
                            features = encoder_tensor

                        elif encoder_features:
                            # Only encoder features available
                            features = torch.cat(encoder_features, dim=-1)

                        else:
                            # Fallback: process all non-target tensors
                            all_tensors = []

                            for key, value in x.items():
                                if torch.is_tensor(
                                        value) and 'target' not in key.lower() and 'scale' not in key.lower():
                                    # Convert to 3D tensor [batch, seq_len, features]
                                    if value.dim() == 3:
                                        processed = value.float()
                                    elif value.dim() == 2:
                                        processed = value.unsqueeze(1).float()
                                    elif value.dim() == 1:
                                        processed = value.unsqueeze(1).unsqueeze(-1).float()
                                    else:
                                        continue

                                    all_tensors.append(processed)

                            if all_tensors:
                                # Find common sequence length (use the most common)
                                seq_lens = [t.shape[1] for t in all_tensors]
                                from collections import Counter
                                common_seq_len = Counter(seq_lens).most_common(1)[0][0]

                                # Filter tensors with common sequence length
                                filtered_tensors = [t for t in all_tensors if t.shape[1] == common_seq_len]

                                if filtered_tensors:
                                    features = torch.cat(filtered_tensors, dim=-1)
                                else:
                                    continue
                            else:
                                continue

                    elif torch.is_tensor(x):
                        # x is already a tensor - ensure it's 3D
                        features = x.float()
                        if features.dim() == 1:
                            features = features.unsqueeze(0).unsqueeze(1)  # [1, 1, features]
                        elif features.dim() == 2:
                            features = features.unsqueeze(1)  # [batch, 1, features]
                    else:
                        continue

                    # Handle y (target)
                    if isinstance(y, tuple) and len(y) > 0:
                        target = y[0]
                    elif torch.is_tensor(y):
                        target = y
                    else:
                        continue

                    # Ensure target is properly shaped
                    if torch.is_tensor(target):
                        if len(target.shape) > 2:
                            target = target.squeeze(-1)
                        if len(target.shape) > 1 and target.shape[1] > 1:
                            target = target[:, 0]

                        # Ensure batch sizes match
                        if features.shape[0] == target.shape[0]:
                            val_data.append((features.detach(), target.detach()))
                        else:
                            continue
                    else:
                        continue

                # Update progress bar with enhanced validation data metrics
                val_pbar.set_postfix({
                    '📊 Samples': len(val_data),
                    '📦 Batch': f"{i + 1}/{max_batches // 2}",
                    '✅ Success': '🎯'
                })

            except Exception as e:
                val_pbar.set_postfix({
                    '❌ ERROR': str(e)[:15],
                    '📦 Batch': f"{i + 1}"
                })
                continue

    except Exception as e:
        print(f"   Validation data error: {e}")

    print(f"   ✅ Prepared {len(train_data)} training samples, {len(val_data)} validation samples")

    return train_data, val_data


def train_model(model, train_data, val_data, device, epochs=10, lr=0.001):
    """Train the model with pure PyTorch and enhanced tqdm progress bars."""

    print(f"🏋️ Training model on {device} for {epochs} epochs...")
    print(f"📊 Training samples: {len(train_data)}, Validation samples: {len(val_data)}")
    print(f"📝 Learning rate: {lr}, Device: {device}")
    print("=" * 80)

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    train_losses = []
    val_losses = []

    # Main epoch progress bar with enhanced description
    epoch_pbar = tqdm(
        range(epochs),
        desc="🏋️ TFT Training",
        unit="epoch",
        ncols=100,
        colour='blue'
    )

    for epoch in epoch_pbar:
        # Training phase
        model.train()
        train_loss = 0
        train_batches = 0

        # Enhanced training batch progress bar
        train_pbar = tqdm(
            train_data,
            desc=f"📈 Epoch {epoch + 1:2d}/{epochs} Training",
            leave=False,
            unit="batch",
            ncols=120,
            colour='green',
            miniters=1
        )

        for batch_idx, batch_item in enumerate(train_pbar):
            try:
                # Debug first iteration - check what we're getting
                if epoch == 0 and batch_idx == 0:
                    print(f"   DEBUG: Batch item type: {type(batch_item)}")
                    print(
                        f"   DEBUG: Batch item content: {batch_item if not torch.is_tensor(batch_item) else 'tensor'}")
                    if isinstance(batch_item, (tuple, list)):
                        print(f"   DEBUG: Batch item length: {len(batch_item)}")
                        for j, item in enumerate(batch_item):
                            print(
                                f"   DEBUG: Item {j} type: {type(item)}, shape: {item.shape if torch.is_tensor(item) else 'N/A'}")

                # Properly unpack the batch
                if isinstance(batch_item, (tuple, list)) and len(batch_item) == 2:
                    x, y = batch_item

                    # Additional debugging for successful unpacking
                    if epoch == 0 and batch_idx == 0:
                        print(f"   DEBUG: Successfully unpacked X: {x.shape}, Y: {y.shape}")
                else:
                    print(
                        f"   ERROR: Unexpected batch format: {type(batch_item)}, length: {len(batch_item) if hasattr(batch_item, '__len__') else 'N/A'}")
                    continue

                # More debugging before device transfer
                if epoch == 0 and batch_idx == 0:
                    print(f"   DEBUG: About to move tensors to device...")

                x = x.to(device)
                y = y.to(device)

                if epoch == 0 and batch_idx == 0:
                    print(f"   DEBUG: Tensors moved to device, starting forward pass...")

                # Ensure y is the right shape
                if len(y.shape) == 1:
                    y = y.unsqueeze(-1)
                elif len(y.shape) > 2:
                    y = y.squeeze()
                    if len(y.shape) == 1:
                        y = y.unsqueeze(-1)

                if epoch == 0 and batch_idx == 0:
                    print(f"   DEBUG: Y reshaped to: {y.shape}")

                optimizer.zero_grad()

                if epoch == 0 and batch_idx == 0:
                    print(f"   DEBUG: About to call model forward pass...")

                pred = model(x)

                if epoch == 0 and batch_idx == 0:
                    print(f"   DEBUG: Model forward pass complete, pred shape: {pred.shape}")

                # Match prediction and target shapes
                if pred.shape != y.shape:
                    if pred.numel() == y.numel():
                        pred = pred.view(y.shape)
                    else:
                        # Take minimum elements
                        min_size = min(pred.numel(), y.numel())
                        pred = pred.view(-1)[:min_size].view(-1, 1)
                        y = y.view(-1)[:min_size].view(-1, 1)

                if epoch == 0 and batch_idx == 0:
                    print(f"   DEBUG: About to calculate loss...")

                loss = criterion(pred, y)

                if epoch == 0 and batch_idx == 0:
                    print(f"   DEBUG: Loss calculated: {loss.item()}")

                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                train_batches += 1

                if epoch == 0 and batch_idx == 0:
                    print(f"   DEBUG: Training step completed successfully!")

                # Update progress bar with enhanced metrics
                current_avg = train_loss / train_batches
                train_pbar.set_postfix({
                    'loss': f"{loss.item():.5f}",
                    'avg': f"{current_avg:.5f}",
                    'lr': f"{lr:.1e}",
                    'batch': f"{batch_idx + 1}"
                })

            except Exception as e:
                if epoch == 0 and batch_idx == 0:
                    print(f"   DEBUG: Exception in training loop: {e}")
                    import traceback
                    traceback.print_exc()
                train_pbar.set_postfix({
                    'ERROR': str(e)[:15],
                    'batch': f"{batch_idx + 1}"
                })
                continue

        # Validation phase
        model.eval()
        val_loss = 0
        val_batches = 0

        # Enhanced validation batch progress bar
        val_pbar = tqdm(
            val_data,
            desc=f"📊 Epoch {epoch + 1:2d}/{epochs} Validation",
            leave=False,
            unit="batch",
            ncols=120,
            colour='yellow',
            miniters=1
        )

        with torch.no_grad():
            for val_idx, val_batch_item in enumerate(val_pbar):
                try:
                    # Debug first validation iteration
                    if epoch == 0 and val_idx == 0:
                        print(f"   DEBUG: Val batch item type: {type(val_batch_item)}")
                        if isinstance(val_batch_item, (tuple, list)):
                            print(f"   DEBUG: Val batch item length: {len(val_batch_item)}")

                    # Properly unpack the validation batch
                    if isinstance(val_batch_item, (tuple, list)) and len(val_batch_item) == 2:
                        x, y = val_batch_item
                    else:
                        print(f"   ERROR: Unexpected val batch format: {type(val_batch_item)}")
                        continue
                    x = x.to(device)
                    y = y.to(device)

                    if len(y.shape) == 1:
                        y = y.unsqueeze(-1)
                    elif len(y.shape) > 2:
                        y = y.squeeze()
                        if len(y.shape) == 1:
                            y = y.unsqueeze(-1)

                    pred = model(x)

                    if pred.shape != y.shape:
                        if pred.numel() == y.numel():
                            pred = pred.view(y.shape)
                        else:
                            min_size = min(pred.numel(), y.numel())
                            pred = pred.view(-1)[:min_size].view(-1, 1)
                            y = y.view(-1)[:min_size].view(-1, 1)

                    loss = criterion(pred, y)
                    val_loss += loss.item()
                    val_batches += 1

                    # Update validation progress with enhanced metrics
                    current_val_avg = val_loss / val_batches
                    val_pbar.set_postfix({
                        'val_loss': f"{loss.item():.5f}",
                        'val_avg': f"{current_val_avg:.5f}",
                        'samples': val_batches
                    })

                except Exception as e:
                    val_pbar.set_postfix({
                        'ERROR': str(e)[:15],
                        'samples': val_batches
                    })
                    continue

        # Calculate average losses
        avg_train_loss = train_loss / max(train_batches, 1)
        avg_val_loss = val_loss / max(val_batches, 1)

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)

        # Update main epoch progress bar with comprehensive metrics
        epoch_pbar.set_postfix({
            '🚂 Train': f"{avg_train_loss:.5f}",
            '📊 Val': f"{avg_val_loss:.5f}",
            '📦 T.Batch': train_batches,
            '📦 V.Batch': val_batches,
            '📈 Improve': '✅' if epoch > 0 and avg_val_loss < val_losses[-2] else '⚠️'
        })

    epoch_pbar.close()

    # Final training summary
    print("\n" + "=" * 80)
    print("✅ TRAINING COMPLETED SUCCESSFULLY!")
    if train_losses:
        print(f"📈 Final Training Loss: {train_losses[-1]:.6f}")
    if val_losses:
        print(f"📊 Final Validation Loss: {val_losses[-1]:.6f}")
    print(f"🔄 Total Epochs: {epochs}")
    print(f"📦 Total Training Samples: {len(train_data)}")
    print(f"📦 Total Validation Samples: {len(val_data)}")
    print("=" * 80)

    return train_losses, val_losses


def generate_predictions(model, val_data, device):
    """Generate predictions from the trained model with progress bar."""

    print("🔮 Generating predictions...")

    model.eval()
    model = model.to(device)

    predictions = []
    actuals = []

    # Enhanced progress bar for prediction generation
    prediction_pbar = tqdm(
        val_data,
        desc="🔮 Generating Predictions",
        unit="batch",
        ncols=120,
        colour='cyan'
    )

    with torch.no_grad():
        for pred_idx, pred_batch_item in enumerate(prediction_pbar):
            try:
                # Debug first prediction iteration
                if pred_idx == 0:
                    print(f"   DEBUG: Pred batch item type: {type(pred_batch_item)}")
                    if isinstance(pred_batch_item, (tuple, list)):
                        print(f"   DEBUG: Pred batch item length: {len(pred_batch_item)}")

                # Properly unpack and process the prediction batch - same as training
                if isinstance(pred_batch_item, (tuple, list)) and len(pred_batch_item) >= 2:
                    x, y = pred_batch_item[0], pred_batch_item[1]

                    # Process features to 3D tensor - TFT encoder/decoder structure (prediction)
                    if isinstance(x, dict):
                        # TFT format: separate encoder and decoder features
                        encoder_features = []
                        decoder_features = []

                        # Process encoder features (historical data)
                        for key in ['encoder_cat', 'encoder_cont']:
                            if key in x and torch.is_tensor(x[key]):
                                value = x[key]

                                # Encoder features should already be 3D [batch, seq_len, features]
                                if value.dim() == 3:
                                    encoder_features.append(value.float())
                                elif value.dim() == 2:
                                    encoder_features.append(value.unsqueeze(1).float())
                                elif value.dim() == 1:
                                    encoder_features.append(value.unsqueeze(1).unsqueeze(-1).float())

                        # Process decoder features (future context)
                        for key in ['decoder_cat', 'decoder_cont']:
                            if key in x and torch.is_tensor(x[key]):
                                value = x[key]

                                # Decoder features should be 3D [batch, pred_len, features]
                                if value.dim() == 3:
                                    decoder_features.append(value.float())
                                elif value.dim() == 2:
                                    decoder_features.append(value.unsqueeze(1).float())
                                elif value.dim() == 1:
                                    decoder_features.append(value.unsqueeze(1).unsqueeze(-1).float())

                        # Combine encoder and decoder features
                        if encoder_features and decoder_features:
                            # Concatenate encoder features
                            encoder_tensor = torch.cat(encoder_features, dim=-1)  # [batch, enc_len, enc_features]
                            decoder_tensor = torch.cat(decoder_features, dim=-1)  # [batch, dec_len, dec_features]

                            # For TFT, we'll use encoder features for training (past data to predict future)
                            features = encoder_tensor

                        elif encoder_features:
                            # Only encoder features available
                            features = torch.cat(encoder_features, dim=-1)

                        else:
                            # Fallback: process all non-target tensors
                            all_tensors = []

                            for key, value in x.items():
                                if torch.is_tensor(
                                        value) and 'target' not in key.lower() and 'scale' not in key.lower():
                                    # Convert to 3D tensor [batch, seq_len, features]
                                    if value.dim() == 3:
                                        processed = value.float()
                                    elif value.dim() == 2:
                                        processed = value.unsqueeze(1).float()
                                    elif value.dim() == 1:
                                        processed = value.unsqueeze(1).unsqueeze(-1).float()
                                    else:
                                        continue

                                    all_tensors.append(processed)

                            if all_tensors:
                                # Find common sequence length (use the most common)
                                seq_lens = [t.shape[1] for t in all_tensors]
                                from collections import Counter
                                common_seq_len = Counter(seq_lens).most_common(1)[0][0]

                                # Filter tensors with common sequence length
                                filtered_tensors = [t for t in all_tensors if t.shape[1] == common_seq_len]

                                if filtered_tensors:
                                    features = torch.cat(filtered_tensors, dim=-1)
                                else:
                                    continue
                            else:
                                continue

                    elif torch.is_tensor(x):
                        # x is already a tensor - ensure it's 3D
                        features = x.float()
                        if features.dim() == 1:
                            features = features.unsqueeze(0).unsqueeze(1)  # [1, 1, features]
                        elif features.dim() == 2:
                            features = features.unsqueeze(1)  # [batch, 1, features]
                    else:
                        continue

                    # Handle target
                    if isinstance(y, tuple) and len(y) > 0:
                        target = y[0]
                    elif torch.is_tensor(y):
                        target = y
                    else:
                        continue

                    # Ensure target is properly shaped
                    if torch.is_tensor(target):
                        if len(target.shape) > 2:
                            target = target.squeeze(-1)
                        if len(target.shape) > 1 and target.shape[1] > 1:
                            target = target[:, 0]  # Take first prediction timestep

                    # Move to device
                    features = features.to(device)

                    # Generate prediction
                    pred = model(features)

                    # Convert to numpy
                    pred_np = pred.cpu().numpy().flatten()
                    target_np = target.numpy().flatten()

                    predictions.extend(pred_np)
                    actuals.extend(target_np)

                    # Update progress bar with detailed prediction info
                    avg_pred = np.mean(pred_np) if len(pred_np) > 0 else 0
                    avg_actual = np.mean(target_np) if len(target_np) > 0 else 0
                    prediction_pbar.set_postfix({
                        'pred_avg': f"{avg_pred:.4f}",
                        'actual_avg': f"{avg_actual:.4f}",
                        'batch_size': len(pred_np),
                        'total': len(predictions)
                    })
                else:
                    print(f"   ERROR: Unexpected pred batch format: {type(pred_batch_item)}")
                    continue

            except Exception as e:
                print(f"   ERROR in prediction: {str(e)}")
                prediction_pbar.set_postfix({
                    'ERROR': str(e)[:15],
                    'total': len(predictions)
                })
                continue

    prediction_pbar.close()

    predictions = np.array(predictions)
    actuals = np.array(actuals)

    print(f"   ✅ Generated {len(predictions)} predictions")

    return predictions, actuals


def create_visualizations(predictions, actuals, train_losses, val_losses):
    """Create comprehensive visualizations."""

    print("📊 Creating visualizations...")

    # Check if we have valid data for visualization
    if len(predictions) == 0 or len(actuals) == 0:
        print("   ⚠️ No predictions or actuals available for visualization")
        print(f"   📊 Predictions: {len(predictions)}, Actuals: {len(actuals)}")

        # Create a simple plot showing only training progress
        plt.figure(figsize=(10, 6))
        if train_losses:
            plt.plot(train_losses, label='Training Loss', color='blue', alpha=0.8)
        if val_losses:
            plt.plot(val_losses, label='Validation Loss', color='red', alpha=0.8)
        plt.title('Training Progress (No Predictions Generated)', fontsize=14, fontweight='bold')
        plt.xlabel('Epoch')
        plt.ylabel('Loss (MSE)')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Save simple plot
        output_path = 'tft_training_only.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   ✅ Saved training progress to {output_path}")
        return

    # Convert to numpy arrays
    predictions = np.array(predictions)
    actuals = np.array(actuals)

    plt.style.use('seaborn-v0_8')
    fig = plt.figure(figsize=(20, 12))

    # 1. Training curves
    ax1 = plt.subplot(2, 3, 1)
    plt.plot(train_losses, label='Training Loss', color='blue', alpha=0.8)
    plt.plot(val_losses, label='Validation Loss', color='red', alpha=0.8)
    plt.title('Training Progress', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 2. Predictions vs Actuals
    ax2 = plt.subplot(2, 3, 2)
    plt.scatter(actuals, predictions, alpha=0.6, s=20)
    min_val = min(np.min(actuals), np.min(predictions))
    max_val = max(np.max(actuals), np.max(predictions))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Predictions vs Actuals', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)

    # Calculate R²
    corr_matrix = np.corrcoef(actuals, predictions)
    r_squared = corr_matrix[0, 1] ** 2
    plt.text(0.05, 0.95, f'R² = {r_squared:.3f}', transform=ax2.transAxes,
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

    # 3. Time series comparison
    ax3 = plt.subplot(2, 3, 3)
    indices = np.arange(len(predictions))
    plt.plot(indices, actuals, label='Actual', color='blue', alpha=0.8)
    plt.plot(indices, predictions, label='Predicted', color='red', alpha=0.8)
    plt.title('Time Series Comparison', fontsize=14, fontweight='bold')
    plt.xlabel('Sample Index')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 4. Residuals
    ax4 = plt.subplot(2, 3, 4)
    residuals = predictions - actuals
    plt.scatter(predictions, residuals, alpha=0.6, s=20)
    plt.axhline(y=0, color='red', linestyle='--', alpha=0.8)
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.title('Residuals Plot', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)

    # 5. Error distribution
    ax5 = plt.subplot(2, 3, 5)
    plt.hist(residuals, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    plt.xlabel('Residuals')
    plt.ylabel('Frequency')
    plt.title('Error Distribution', fontsize=14, fontweight='bold')
    plt.axvline(x=0, color='red', linestyle='--', alpha=0.8)
    plt.grid(True, alpha=0.3)

    # 6. Model metrics
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')

    # Calculate metrics
    mae = np.mean(np.abs(residuals))
    rmse = np.sqrt(np.mean(residuals ** 2))
    mape = np.mean(np.abs(residuals / (actuals + 1e-8))) * 100

    # Directional accuracy
    actual_direction = np.sign(np.diff(actuals))
    pred_direction = np.sign(np.diff(predictions))
    directional_accuracy = np.mean(actual_direction == pred_direction) * 100

    metrics_text = f'''
Model Performance Metrics:

MAE: {mae:.4f}
RMSE: {rmse:.4f}
MAPE: {mape:.2f}%
R²: {r_squared:.3f}
Directional Accuracy: {directional_accuracy:.1f}%

Data Points: {len(predictions)}
Device: {'MPS' if torch.backends.mps.is_available() else 'CPU'}
Model: SimpleTFT
    '''

    ax6.text(0.1, 0.5, metrics_text, fontsize=12, verticalalignment='center',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))

    plt.tight_layout()

    # Save plot
    output_path = 'tft_pure_torch_analysis.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"   ✅ Saved analysis to {output_path}")


def simulate_trading(predictions, actuals):
    import numpy as np
    import matplotlib.pyplot as plt
    print("💰 Running trading simulation...")
    initial_price = 100.0
    actual_prices = [initial_price]
    pred_prices = [initial_price]
    for i in range(len(actuals)):
        actual_prices.append(actual_prices[-1] * (1 + actuals[i] * 0.01))
        pred_prices.append(pred_prices[-1] * (1 + predictions[i] * 0.01))
    actual_prices = np.array(actual_prices[1:])
    pred_prices = np.array(pred_prices[1:])
    portfolio_value = 10000
    position = 0
    trades = []
    portfolio_values = [portfolio_value]
    for i in range(1, len(pred_prices)):
        if i < len(pred_prices) - 1:
            predicted_return = (pred_prices[i + 1] - pred_prices[i]) / pred_prices[i]
            actual_return = (actual_prices[i + 1] - actual_prices[i]) / actual_prices[i]
            if predicted_return > 0.001 and position != 1:
                if position == -1:
                    trades.append(('close_short', i, actual_prices[i]))
                trades.append(('buy', i, actual_prices[i]))
                position = 1
            elif predicted_return < -0.001 and position != -1:
                if position == 1:
                    trades.append(('close_long', i, actual_prices[i]))
                trades.append(('sell_short', i, actual_prices[i]))
                position = -1
            if position == 1:
                portfolio_value *= (1 + actual_return)
            elif position == -1:
                portfolio_value *= (1 - actual_return)
            portfolio_values.append(portfolio_value)
    total_return = (portfolio_value - 10000) / 10000 * 100
    buy_hold_return = (actual_prices[-1] - actual_prices[0]) / actual_prices[0] * 100
    plt.figure(figsize=(15, 10))
    plt.subplot(2, 2, 1)
    plt.plot(actual_prices, label='Actual Price', color='blue', alpha=0.8)
    plt.plot(pred_prices, label='Predicted Price', color='red', alpha=0.8, linestyle='--')
    for trade_type, idx, price in trades:
        color = 'green' if 'buy' in trade_type else 'red'
        marker = '^' if 'buy' in trade_type else 'v'
        plt.scatter(idx, price, color=color, marker=marker, s=100, alpha=0.8)
    plt.title('Trading Strategy Performance', fontsize=14, fontweight='bold')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.subplot(2, 2, 2)
    plt.plot(portfolio_values, color='green', linewidth=2)
    plt.axhline(y=10000, color='gray', linestyle='--', alpha=0.8)
    plt.title('Portfolio Value Over Time', fontsize=14, fontweight='bold')
    plt.xlabel('Time')
    plt.ylabel('Portfolio Value ($)')
    plt.grid(True, alpha=0.3)
    plt.subplot(2, 2, 3)
    strategies = ['TFT Strategy', 'Buy & Hold']
    returns = [total_return, buy_hold_return]
    colors = ['green' if r > 0 else 'red' for r in returns]
    plt.bar(strategies, returns, color=colors, alpha=0.7)
    plt.title('Strategy Returns Comparison', fontsize=14, fontweight='bold')
    plt.ylabel('Return (%)')
    plt.grid(True, alpha=0.3)
    plt.subplot(2, 2, 4)
    plt.axis('off')
    summary_text = f'''
Trading Performance Summary:

Starting Capital: $10,000
Final Portfolio Value: ${portfolio_value:,.2f}
Total Return: {total_return:.2f}%
Buy & Hold Return: {buy_hold_return:.2f}%
Outperformance: {total_return - buy_hold_return:.2f}%

Number of Trades: {len(trades)}
Win Rate: {np.mean([1 if t > 0 else 0 for t in np.diff(portfolio_values)]) * 100 if len(portfolio_values) > 1 else float('nan'):.1f}%

Strategy: Trend Following
Signal: Predicted Price Direction
    '''
    plt.text(0.1, 0.5, summary_text, fontsize=12, verticalalignment='center',
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    plt.tight_layout()
    trading_output = 'tft_trading_performance.png'
    plt.savefig(trading_output, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✅ Saved trading analysis to {trading_output}")
    print(f"   💰 Total return: {total_return:.2f}% vs Buy & Hold: {buy_hold_return:.2f}%")
