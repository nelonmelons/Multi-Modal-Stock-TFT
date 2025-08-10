#!/usr/bin/env python3
import math
import torch
import torch.nn as nn

import math

class LSTMModel(nn.Module):
    """A simple LSTM model for baseline comparison."""
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, dropout=0.1):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # If input is 2D (batch, features), reshape to 3D (batch, 1, features)
        if x.dim() == 2:
            x = x.unsqueeze(1)
        
        batch_size = x.size(0)
        h0 = torch.zeros(self.lstm.num_layers, batch_size, self.lstm.hidden_size).to(x.device)
        c0 = torch.zeros(self.lstm.num_layers, batch_size, self.lstm.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

class GRUModel(nn.Module):
    """A simple GRU model for baseline comparison."""
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, dropout=0.1):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # If input is 2D (batch, features), reshape to 3D (batch, 1, features)
        if x.dim() == 2:
            x = x.unsqueeze(1)
        
        batch_size = x.size(0)
        h0 = torch.zeros(self.gru.num_layers, batch_size, self.gru.hidden_size).to(x.device)
        out, _ = self.gru(x, h0)
        out = self.fc(out[:, -1, :])
        return out

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :].to(x.device)
        return self.dropout(x)

class TransformerModel(nn.Module):
    """
    Transformer model for time series forecasting.
    This model uses a transformer encoder to capture temporal dependencies in the data.
    """
    def __init__(self, input_dim, model_dim, num_heads, num_layers, output_dim, dropout=0.1):
        super(TransformerModel, self).__init__()
        self.model_dim = model_dim
        self.encoder = nn.Linear(input_dim, model_dim)
        self.pos_encoder = PositionalEncoding(model_dim, dropout)
        encoder_layers = nn.TransformerEncoderLayer(d_model=model_dim, nhead=num_heads, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        self.decoder = nn.Linear(model_dim, output_dim)

    def forward(self, src):
        # If input is 2D (batch, features), reshape to 3D (batch, 1, features)
        if src.dim() == 2:
            src = src.unsqueeze(1)
            
        src = self.encoder(src) * math.sqrt(self.model_dim)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        output = self.decoder(output[:, -1, :])
        return output

class GatedResidualNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim=None, dropout=0.1):
        super().__init__()
        if output_dim is None:
            output_dim = input_dim
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.swish = nn.SiLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.gate = nn.GLU()
        self.skip = nn.Linear(input_dim, output_dim) if input_dim != output_dim else nn.Identity()
        self.norm = nn.LayerNorm(output_dim)

    def forward(self, x):
        residual = self.skip(x)
        x = self.fc1(x)
        x = self.swish(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.gelu(x)
        x = self.gate(torch.cat([x, x], dim=-1))
        x = self.norm(x + residual)
        return x

class VariableSelectionNetwork(nn.Module):
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
        B, T, _ = x.shape
        splits = torch.split(x, self.input_dims, dim=-1)
        var_outputs = []
        for i, grn in enumerate(self.var_grns):
            var_outputs.append(grn(splits[i]))
        var_outputs = torch.stack(var_outputs, dim=2)
        flat = torch.cat(var_outputs.unbind(dim=2), dim=-1)
        weights = self.weight_grn(flat)
        weights = self.softmax(weights)
        out = (var_outputs * weights.unsqueeze(-1)).sum(dim=2)
        return out, weights

class NewsDownsampler(nn.Module):
    def __init__(self, news_dim, context_dim, out_dim):
        super().__init__()
        self.fc_news = nn.Linear(news_dim, out_dim)
        self.fc_context = nn.Linear(context_dim, out_dim)
        self.fc_fusion = nn.Sequential(
            nn.SiLU(),
            nn.Linear(out_dim * 2, out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim)
        )
        self.residual_proj = nn.Linear(news_dim, out_dim) if news_dim != out_dim else nn.Identity()

    def forward(self, news, context):
        news_proj = self.fc_news(news)
        context_proj = self.fc_context(context)

        if news_proj.dim() == 2 and context_proj.dim() == 3:
            news_proj = news_proj.unsqueeze(1).expand(-1, context_proj.size(1), -1)
        elif context_proj.dim() == 2 and news_proj.dim() == 3:
            context_proj = context_proj.unsqueeze(1).expand(-1, news_proj.size(1), -1)

        fusion = torch.cat([news_proj, context_proj], dim=-1)
        out = self.fc_fusion(fusion)

        residual = news if isinstance(self.residual_proj, nn.Linear) else news_proj
        out = out + self.residual_proj(residual)

        if out.dim() == 3:
            out = out.mean(dim=1)

        return out

class TFT(nn.Module):
    """Temporal Fusion Transformer with explicit VSN and conditional news encoder."""

    def __init__(
            self,
            input_size,
            news_dim,
            hidden_size=64,
            num_heads=4,
            dropout=0.1,
            prediction_len=1,
            news_downsample_dim=32,
            input_var_dims=None,
            quantiles=None
    ):
        super().__init__()
        self.input_size = input_size
        self.news_dim = news_dim
        self.hidden_size = hidden_size
        self.prediction_len = prediction_len
        self.news_downsample_dim = news_downsample_dim
        self.quantiles = quantiles

        if input_var_dims is None:
            non_news_input_size = input_size
            input_var_dims = [1] * non_news_input_size if non_news_input_size > 0 else []
        self.input_var_dims = input_var_dims

        if self.input_var_dims:
            self.feature_embeddings = nn.ModuleList([
                nn.Linear(dim, hidden_size) for dim in input_var_dims
            ])
            self.vsn = VariableSelectionNetwork([hidden_size] * len(input_var_dims), hidden_size, dropout=dropout)
        
        # Calculate fusion input dimension correctly
        fusion_input_dim = hidden_size
        if self.news_dim > 0:
            self.news_downsampler = NewsDownsampler(news_dim, hidden_size, news_downsample_dim)
            fusion_input_dim += news_downsample_dim

        self.encoder_grn = GatedResidualNetwork(hidden_size, hidden_size, dropout=dropout)
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.attn_norm = nn.LayerNorm(hidden_size)
        self.decoder_grn = GatedResidualNetwork(hidden_size, hidden_size, dropout=dropout)
        
        # Store fusion_input_dim for debugging
        self.fusion_input_dim = fusion_input_dim
        self.fusion = nn.Linear(fusion_input_dim, hidden_size)
        
        output_size = prediction_len
        if self.quantiles:
            output_size *= len(self.quantiles)

        self.prediction_head = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.SiLU(),
            nn.Linear(hidden_size // 4, output_size)
        )
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)

    def forward(self, x, news=None):
        # Handle 2D input by adding sequence dimension
        if x.dim() == 2:
            x = x.unsqueeze(1)
        
        if self.input_var_dims:
            # Split non-news features and process through VSN
            non_news_features = x
            splits = torch.split(non_news_features, self.input_var_dims, dim=-1)
            var_embs = []
            for i, emb in enumerate(self.feature_embeddings):
                var_embs.append(emb(splits[i]))
            x_cat = torch.cat(var_embs, dim=-1)
            x_vsn, vsn_weights = self.vsn(x_cat)
        else:
            x_vsn = torch.zeros(x.size(0), x.size(1), self.hidden_size, device=x.device)
            vsn_weights = None

        # Process through encoder GRN
        x_enc = self.encoder_grn(x_vsn)
        
        # Apply self-attention
        attn_out, attn_weights = self.attention(x_enc, x_enc, x_enc)
        attn_out = self.attn_norm(x_enc + attn_out)
        
        # Process through decoder GRN
        dec_out = self.decoder_grn(attn_out)
        last_hidden = dec_out[:, -1, :]  # Take last timestep

        # Prepare fusion inputs
        fusion_inputs = [last_hidden]
        
        # Add news information if available and news_dim > 0
        if self.news_dim > 0 and news is not None:
            # Handle news data - ensure it's 2D (batch, news_features)
            if news.dim() == 3:
                news = news.mean(dim=1)  # Average over time dimension if 3D
            elif news.dim() == 1:
                news = news.unsqueeze(0)  # Add batch dimension if 1D
            
            news_down = self.news_downsampler(news, last_hidden)
            fusion_inputs.append(news_down)
        elif self.news_dim > 0:
            # If model expects news but none provided, add zero padding
            batch_size = last_hidden.size(0)
            zero_news = torch.zeros(batch_size, self.news_downsample_dim, device=last_hidden.device)
            fusion_inputs.append(zero_news)

        # Fuse all inputs
        fusion_cat = torch.cat(fusion_inputs, dim=-1)
        fusion = self.fusion(fusion_cat)
        
        # Generate prediction
        prediction = self.prediction_head(fusion)

        # Handle quantile predictions if specified
        if self.quantiles:
            prediction = prediction.view(-1, self.prediction_len, len(self.quantiles))

        return prediction
