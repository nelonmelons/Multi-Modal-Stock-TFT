from basics import *
#instead of using Transformer, we use this ?? it has attention mechanism 

class SoftAlignModel(nn.Module):
    """
    Soft Attention-based Encoder-Decoder model based on Bahdanau et al. (2015).
    """
    def __init__(self, embed_dim=128, hidden_dim=128, input_dim=128):
        super().__init__()
        self.encoder_rnn = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.decoder_rnn = nn.GRU(input_dim + hidden_dim, hidden_dim, batch_first=True)

        # Attention mechanism
        self.W_h = nn.Linear(hidden_dim, hidden_dim, bias=False)  # for encoder h_i
        self.W_s = nn.Linear(hidden_dim, hidden_dim, bias=False)  # for decoder s_{t-1}
        self.v = nn.Linear(hidden_dim, 1, bias=False)             # energy vector

        # Output projection
        self.output_proj = nn.Linear(hidden_dim, input_dim)
        self.grn = GRN(hidden_dim, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (B, T, input_dim) — the embedded input sequence

        Returns:
            Tensor of shape (B, 1, input_dim) — prediction for the next token
        """
        B, T, D = x.shape

        # -------- Encoder --------
        encoder_outputs, _ = self.encoder_rnn(x)   # (B, T, H)

        # -------- Decoder (1 step) --------
        # Initialize decoder hidden state with zeros
        s_t = torch.zeros(1, B, encoder_outputs.size(2), device=x.device)  # (1, B, H)

        # Last input token as y_{t-1}
        y_prev = x[:, -1:, :]  # (B, 1, D)

        # Compute attention scores
        Wh = self.W_h(encoder_outputs)           # (B, T, H)
        Ws = self.W_s(s_t.transpose(0, 1))       # (B, 1, H)
        e = self.v(torch.tanh(Wh + Ws))          # (B, T, 1)
        alpha = torch.softmax(e, dim=1)          # (B, T, 1)

        # Compute context vector
        context = torch.sum(alpha * encoder_outputs, dim=1, keepdim=True)  # (B, 1, H)

        # Concatenate y_{t-1} and context
        decoder_input = torch.cat([y_prev, context], dim=-1)  # (B, 1, D + H)

        # Decode one step
        output, _ = self.decoder_rnn(decoder_input, s_t)  # output: (B, 1, H)

        output = self.grn(output)                         # (B, 1, H)
        return self.output_proj(output)                   # (B, 1, input_dim)
