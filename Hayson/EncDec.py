from basics import *

# --- for using mask in mps fallback ---
import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
# --- DO NOT REMOVE ABOVE CODE , ALTERNATIVELY USE
#    ```bash
#    export PYTORCH_ENABLE_MPS_FALLBACK=1
#    ```

class EncDecTransformer(nn.Module):
    def __init__(self, embed_dim=128, n_heads=4, num_layers=2):
        super().__init__()
        assert embed_dim % n_heads == 0, "embed_dim must be divisible by n_heads"

        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(embed_dim, n_heads, batch_first=True),
            num_layers=num_layers
        )
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(embed_dim, n_heads, batch_first=True),
            num_layers=num_layers
        )
        self.query_proj = nn.Linear(embed_dim, embed_dim)
        self.grn_enc = GRN(input_dim=embed_dim, hidden_dim=embed_dim)
        self.grn_dec = GRN(input_dim=embed_dim, hidden_dim=embed_dim)
        self.output_proj = nn.Identity()  # Final projection handled outside

    def forward(self,
                x: torch.Tensor,
                tgt: torch.Tensor = None,
                src_mask: torch.Tensor = None,
                src_padding_mask: torch.Tensor = None,
                max_len: int = 1
               ) -> torch.Tensor:
        """
        Args:
            x: (B, S, E) — encoder input
            tgt: (B, T, E) or None
            src_mask: (S, S) or None
            src_padding_mask: (B, S) or None
            max_len: number of autoregressive steps at inference

        Returns:
            (B, T, E) — decoder output
        """
        memory = self.encoder(x,
                              mask=src_mask,
                              src_key_padding_mask=src_padding_mask)
        memory = self.grn_enc(memory)

        if self.training and tgt is not None:
            T = tgt.size(1)
            causal_mask = torch.triu(torch.full((T, T), float('-inf'), device=x.device), diagonal=1)
            dec_out = self.decoder(tgt,
                                   memory,
                                   tgt_mask=causal_mask,
                                   memory_key_padding_mask=src_padding_mask,
                                   tgt_key_padding_mask=None)  # Add if needed
            dec_out = self.grn_dec(dec_out)
            return self.output_proj(dec_out)

        # Inference autoregressive generation
        preds = []
        step = self.query_proj(x[:, -1:, :])  # (B, 1, E)

        for _ in range(max_len):
            L = step.size(1)
            causal_mask = torch.triu(torch.full((L, L), float('-inf'), device=x.device), diagonal=1)

            dec_out = self.decoder(step,
                                   memory,
                                   tgt_mask=causal_mask,
                                   memory_key_padding_mask=src_padding_mask)
            dec_out = self.grn_dec(dec_out)
            next_tok = self.output_proj(dec_out[:, -1:, :])  # (B, 1, E)
            preds.append(next_tok)
            step = torch.cat([step, next_tok], dim=1)

        return torch.cat(preds, dim=1)