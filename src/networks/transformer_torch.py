import torch
import math
from torch import nn
import torch.nn.functional as F


class Transformer_torch(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        hidden_channels,
        increase_mlp,
        resid_pdrop,
        attn_pdrop,
        num_heads,
        num_blocks,
        is_causal,
    ):
        super().__init__()
        if resid_pdrop != attn_pdrop:
            # warning: torch transformer uses same dropout everywhere
            pass

        self.is_causal = is_causal

        self.linear_in = nn.Linear(in_channels, hidden_channels)
        self.transformer = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=hidden_channels,
                    nhead=num_heads,
                    dim_feedforward=hidden_channels * increase_mlp,
                    dropout=resid_pdrop,
                )
                for _ in range(num_blocks)
            ]
        )
        self.linear_out = nn.Linear(hidden_channels, out_channels)

    def forward(self, x, attn_mask=None):
        # linear in
        x = self.linear_in(x)

        # attn_mask and is_causal is weird in torch Transformer
        # (do not fully understand how to properly use this)
        if self.is_causal:
            # quick fix... this should be required...
            n_tokens = x.shape[1]
            # generate causal mask of shape (n_tokens, n_tokens)
            attn_mask = torch.nn.Transformer.generate_square_subsequent_mask(
                n_tokens, device=x.device
            )

        # transformer
        for block in self.transformer:
            x = block(src=x, src_mask=attn_mask)

        # linear out
        x = self.linear_out(x)

        return x
