import torch
import math
from torch import nn
import torch.nn.functional as F

"""
Code mostly taken from https://github.com/karpathy/minGPT
and extended by Bayesian Linear layers
"""


class NewGELU(nn.Module):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
    Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
    """

    def forward(self, x):
        return (
            0.5
            * x
            * (
                1.0
                + torch.tanh(
                    math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))
                )
            )
        )


class CausalSelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    """

    def __init__(
        self, is_causal, num_heads, hidden_channels, attn_pdrop=0.0, resid_pdrop=0.1
    ):
        super().__init__()
        self.n_head = num_heads
        self.hidden_channels = hidden_channels
        self.attn_pdrop = attn_pdrop
        self.resid_pdrop = resid_pdrop
        self.is_causal = is_causal

        n_learnable = 3
        self.c_attn = nn.Linear(self.hidden_channels, 3 * self.hidden_channels)
        self.c_proj = nn.Linear(self.hidden_channels, self.hidden_channels)

        self.resid_dropout = nn.Dropout(self.resid_pdrop)

    def forward(self, x, attn_mask=None):
        x_shape = (
            x.size()
        )  # keep first indices (use label B) general, can be multi-dimensional (start counting from the end!)
        B = x_shape[:-2]  # batch index (can be multi-dimensional)
        T = x_shape[-2]  # feature index
        C = x_shape[-1]  # latent space index

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v = self.c_attn(x).split(self.hidden_channels, dim=-1)

        k = k.view(*B, T, self.n_head, C // self.n_head).transpose(
            -3, -2
        )  # (B, nh, T, hs)
        q = q.view(*B, T, self.n_head, C // self.n_head).transpose(
            -3, -2
        )  # (B, nh, T, hs)
        v = v.view(*B, T, self.n_head, C // self.n_head).transpose(
            -3, -2
        )  # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=self.attn_pdrop,
            is_causal=self.is_causal,
        )
        y = (
            y.transpose(-3, -2).contiguous().view(*B, T, C)
        )  # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y


class TransformerBlock(nn.Module):
    """A transformer block, consisting of a CausalSelfAttention block and a multilayer perceptron"""

    def __init__(
        self,
        hidden_channels,
        increase_mlp,
        resid_pdrop,
        attn_pdrop,
        num_heads,
        is_causal,
    ):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.increase_mlp = increase_mlp
        self.resid_pdrop = resid_pdrop

        self.ln_1 = nn.LayerNorm(self.hidden_channels)
        self.attn = CausalSelfAttention(
            num_heads=num_heads,
            hidden_channels=hidden_channels,
            attn_pdrop=attn_pdrop,
            resid_pdrop=resid_pdrop,
            is_causal=is_causal,
        )
        self.ln_2 = nn.LayerNorm(self.hidden_channels)
        self.mlp = nn.ModuleDict(
            dict(
                c_fc=nn.Linear(
                    self.hidden_channels, self.increase_mlp * self.hidden_channels
                ),
                c_proj=nn.Linear(
                    self.hidden_channels * self.increase_mlp, self.hidden_channels
                ),
                act=NewGELU(),
                dropout=nn.Dropout(self.resid_pdrop),
            )
        )
        m = self.mlp
        self.mlpf = lambda x: m.dropout(m.c_proj(m.act(m.c_fc(x))))

    # forward using the ResNet concept
    def forward(self, x, attn_mask=None):
        x = x + self.attn(self.ln_1(x), attn_mask=attn_mask)
        x = x + self.mlpf(self.ln_2(x))
        return x


class Transformer_mingpt(nn.Module):
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

        self.linear_in = nn.Linear(in_channels, hidden_channels)
        self.transformer = nn.ModuleList(
            [
                TransformerBlock(
                    hidden_channels=hidden_channels,
                    increase_mlp=increase_mlp,
                    resid_pdrop=resid_pdrop,
                    attn_pdrop=attn_pdrop,
                    num_heads=num_heads,
                    is_causal=is_causal,
                )
                for _ in range(num_blocks)
            ]
        )
        self.linear_out = nn.Linear(hidden_channels, out_channels)

    def forward(self, x, attn_mask=None):
        # linear in
        x = self.linear_in(x)

        # transformer
        for block in self.transformer:
            x = block(x, attn_mask=attn_mask)

        # linear out
        x = self.linear_out(x)

        return x
