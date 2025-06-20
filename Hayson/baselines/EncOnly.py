from basics import *


class CausalTransformer(nn.Module):
    """
    Encoder-only Transformer with causal masking that supports:
      - teacher forcing (training with a ground-truth target sequence)
      - autoregressive inference
    """
    def __init__(self, input_dim=35, embed_dim=128, n_heads=4, num_layers=3):
        super().__init__()
        self.embedding = nn.Linear(input_dim, embed_dim)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=n_heads, batch_first=True),
            num_layers=num_layers
        )
        self.grn = GRN(input_dim=embed_dim, hidden_dim=embed_dim)
        self.output_proj = nn.Linear(embed_dim, input_dim)

    def forward(self,
                x: torch.Tensor,              # (B, S, input_dim)  source
                tgt: torch.Tensor = None,     # (B, T, input_dim)  ground-truth future (for teacher forcing)
                src_padding_mask: torch.Tensor = None,
                max_len: int = 1              # #steps to generate in inference
               ) -> torch.Tensor:
        B, S, _ = x.size()
        x_emb = self.embedding(x)         # (B, S, D)

        if self.training and tgt is not None:
            # — Teacher forcing —
            T = tgt.size(1)
            tgt_emb = self.embedding(tgt)   # (B, T, D)

            # Build one long sequence [x_emb | tgt_emb]
            seq = torch.cat([x_emb, tgt_emb], dim=1)  # (B, S+T, D)

            # Causal mask over S+T
            L = S + T
            causal_mask = torch.triu(torch.full((L, L), float('-inf')), diagonal=1, device=x.device)

            # Padding mask: assume no padding in tgt, so just use src mask padded with zeros
            if src_padding_mask is not None:
                # (B, S) → (B, S+T)
                tgt_pad = torch.zeros(B, T, dtype=torch.bool, device=x.device)
                pad_mask = torch.cat([src_padding_mask, tgt_pad], dim=1)
            else:
                pad_mask = None

            # Single forward pass
            out = self.encoder(seq, mask=causal_mask, src_key_padding_mask=pad_mask)
            out = self.grn(out)  # (B, S+T, D)

            # Project only the T “future” positions
            future = out[:, S:, :]           # (B, T, D)
            return self.output_proj(future)  # (B, T, input_dim)

        else:
            # — Autoregressive inference —
            preds = []
            seq = x_emb                    # start with (B, S, D)
            pad_mask = src_padding_mask

            for _ in range(max_len):
                L = seq.size(1)
                causal_mask = torch.triu(torch.full((L, L), float('-inf')), diagonal=1, device=x.device)

                out = self.encoder(seq, mask=causal_mask, src_key_padding_mask=pad_mask)
                out = self.grn(out)                # (B, L, D)
                nxt = self.output_proj(out[:, -1:, :])  # predict next word (B, 1, input_dim)

                preds.append(nxt)

                # embed and append to the sequence
                emb_nxt = self.embedding(nxt)   # (B, 1, D)
                seq = torch.cat([seq, emb_nxt], dim=1)

                # pad_mask: new step is not padding, so append False
                if pad_mask is not None:
                    pad_mask = torch.cat([pad_mask, torch.zeros(B, 1, dtype=torch.bool, device=x.device)], dim=1)

            return torch.cat(preds, dim=1)    # (B, max_len, input_dim)
