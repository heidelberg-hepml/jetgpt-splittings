from functools import partial
from typing import Optional, Tuple

import torch
from einops import rearrange
from torch import nn

from torch.nn.functional import scaled_dot_product_attention

"""
Adopted from https://github.com/Qualcomm-AI-research/geometric-algebra-transformer
Removed rotary positional embeddings, added increase_mlp parameter
"""


def to_nd(tensor, d):
    """Make tensor n-dimensional, group extra dimensions in first."""
    return tensor.view(
        -1, *(1,) * (max(0, d - 1 - tensor.dim())), *tensor.shape[-(d - 1) :]
    )


class BaselineLayerNorm(nn.Module):
    """Baseline layer norm over all dimensions except the first."""

    @staticmethod
    def forward(inputs: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        inputs : Tensor
            Input data

        Returns
        -------
        outputs : Tensor
            Normalized inputs.
        """
        return torch.nn.functional.layer_norm(inputs, normalized_shape=inputs.shape[1:])


class MultiHeadQKVLinear(nn.Module):
    """Compute queries, keys, and values via multi-head attention.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    hidden_channels : int
        Number of hidden channels = size of query, key, and value.
    num_heads : int
        Number of attention heads.
    """

    def __init__(self, in_channels, hidden_channels, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.linear = nn.Linear(in_channels, 3 * hidden_channels * num_heads)

    def forward(self, inputs):
        """Forward pass.

        Returns
        -------
        q : Tensor
            Queries
        k : Tensor
            Keys
        v : Tensor
            Values
        """
        qkv = self.linear(inputs)  # (..., num_items, 3 * hidden_channels * num_heads)
        q, k, v = rearrange(
            qkv,
            "... items (qkv hidden_channels num_heads) -> qkv ... num_heads items hidden_channels",
            num_heads=self.num_heads,
            qkv=3,
        )
        return q, k, v


class MultiQueryQKVLinear(nn.Module):
    """Compute queries, keys, and values via multi-query attention.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    hidden_channels : int
        Number of hidden channels = size of query, key, and value.
    num_heads : int
        Number of attention heads.
    """

    def __init__(self, in_channels, hidden_channels, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.q_linear = nn.Linear(in_channels, hidden_channels * num_heads)
        self.k_linear = nn.Linear(in_channels, hidden_channels)
        self.v_linear = nn.Linear(in_channels, hidden_channels)

    def forward(self, inputs):
        """Forward pass.

        Parameters
        ----------
        inputs : Tensor
            Input data

        Returns
        -------
        q : Tensor
            Queries
        k : Tensor
            Keys
        v : Tensor
            Values
        """
        q = rearrange(
            self.q_linear(inputs),
            "... items (hidden_channels num_heads) -> ... num_heads items hidden_channels",
            num_heads=self.num_heads,
        )
        k = self.k_linear(inputs)[
            ..., None, :, :
        ]  # (..., head=1, item, hidden_channels)
        v = self.v_linear(inputs)[..., None, :, :]
        return q, k, v


class BaselineSelfAttention(nn.Module):
    """Baseline self-attention layer.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of input channels.
    hidden_channels : int
        Number of hidden channels = size of query, key, and value.
    num_heads : int
        Number of attention heads.
    pos_encoding : bool
        Whether to apply rotary positional embeddings along the item dimension to the scalar keys
        and queries.
    pos_enc_base : int
        Maximum frequency used in positional encodings. (The minimum frequency is always 1.)
    multi_query : bool
        Use multi-query attention instead of multi-head attention.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int,
        num_heads: int = 8,
        multi_query: bool = True,
    ) -> None:
        super().__init__()

        # Store settings
        self.num_heads = num_heads
        self.hidden_channels = hidden_channels

        # Linear maps
        qkv_class = MultiQueryQKVLinear if multi_query else MultiHeadQKVLinear
        self.qkv_linear = qkv_class(in_channels, hidden_channels, num_heads)
        self.out_linear = nn.Linear(hidden_channels * num_heads, out_channels)

    def forward(
        self, inputs: torch.Tensor, attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        inputs : Tensor
            Input data
        attention_mask : None or Tensor or xformers.ops.AttentionBias
            Optional attention mask

        Returns
        -------
        outputs : Tensor
            Outputs
        """
        q, k, v = self.qkv_linear(
            inputs
        )  # each: (..., num_heads, num_items, num_channels, 16)

        # Attention layer
        h = self._attend(q, k, v, attention_mask)

        # Concatenate heads and transform linearly
        h = rearrange(
            h,
            "... num_heads num_items hidden_channels -> ... num_items (num_heads hidden_channels)",
        )
        outputs = self.out_linear(h)  # (..., num_items, out_channels)

        return outputs

    @staticmethod
    def _attend(q, k, v, attention_mask=None):
        """Scaled dot-product attention."""

        # Add batch dimension if needed
        bh_shape = q.shape[:-2]
        q = to_nd(q, 4)
        k = to_nd(k, 4)
        v = to_nd(v, 4)

        # SDPA
        outputs = scaled_dot_product_attention(
            q.contiguous(),
            k.expand_as(q).contiguous(),
            v.expand_as(q),
            attn_mask=attention_mask,
        )

        # Return batch dimensions to inputs
        outputs = outputs.view(*bh_shape, *outputs.shape[-2:])

        return outputs


class BaselineTransformerBlock(nn.Module):
    """Baseline transformer block.

    Inputs are first processed by a block consisting of LayerNorm, multi-head self-attention, and
    residual connection. Then the data is processed by a block consisting of another LayerNorm, an
    item-wise two-layer MLP with GeLU activations, and another residual connection.

    Parameters
    ----------
    channels : int
        Number of input and output channels.
    num_heads : int
        Number of attention heads.
    pos_encoding : bool
        Whether to apply rotary positional embeddings along the item dimension to the scalar keys
        and queries.
    pos_encoding_base : int
        Maximum frequency used in positional encodings. (The minimum frequency is always 1.)
    increase_hidden_channels : int
        Factor by which the key, query, and value size is increased over the default value of
        hidden_channels / num_heads.
    multi_query : bool
        Use multi-query attention instead of multi-head attention.
    """

    def __init__(
        self,
        channels,
        num_heads: int = 8,
        increase_hidden_channels=1,
        increase_mlp: int = 2,
        multi_query: bool = True,
    ) -> None:
        super().__init__()

        self.norm = BaselineLayerNorm()

        # When using positional encoding, the number of scalar hidden channels needs to be even.
        # It also should not be too small.
        hidden_channels = channels // num_heads * increase_hidden_channels

        self.attention = BaselineSelfAttention(
            channels,
            channels,
            hidden_channels,
            num_heads=num_heads,
            multi_query=multi_query,
        )

        self.mlp = nn.Sequential(
            nn.Linear(channels, increase_mlp * channels),
            nn.GELU(),
            nn.Linear(increase_mlp * channels, channels),
        )

    def forward(self, inputs: torch.Tensor, attention_mask=None) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        inputs : Tensor
            Input data
        attention_mask : None or Tensor or xformers.ops.AttentionBias
            Optional attention mask

        Returns
        -------
        outputs : Tensor
            Outputs
        """

        # Residual attention
        h = self.norm(inputs)
        h = self.attention(h, attention_mask)
        outputs = inputs + h

        # Residual MLP
        h = self.norm(outputs)
        h = self.mlp(h)
        outputs = outputs + h

        return outputs


class Transformer_amsterdam(nn.Module):
    """Baseline transformer.

    Combines num_blocks transformer blocks, each consisting of multi-head self-attention layers, an
    MLP, residual connections, and normalization layers.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    hidden_channels : int
        Number of hidden channels.
    num_blocks : int
        Number of transformer blocks.
    num_heads : int
        Number of attention heads.
    pos_encoding : bool
        Whether to apply rotary positional embeddings along the item dimension to the scalar keys
        and queries.
    pos_encoding_base : int
        Maximum frequency used in positional encodings. (The minimum frequency is always 1.)
    increase_hidden_channels : int
        Factor by which the key, query, and value size is increased over the default value of
        hidden_channels / num_heads.
    multi_query : bool
        Use multi-query attention instead of multi-head attention.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int,
        num_blocks: int,
        num_heads: int,
        is_causal: bool,
        increase_hidden_channels: int = 1,
        resid_pdrop: float = 0.0,
        attn_pdrop: float = 0.0,
        increase_mlp: int = 2,
        multi_query: bool = False,
    ) -> None:
        super().__init__()
        self.is_causal = is_causal

        self.linear_in = nn.Linear(in_channels, hidden_channels)
        self.blocks = nn.ModuleList(
            [
                BaselineTransformerBlock(
                    hidden_channels,
                    num_heads=num_heads,
                    increase_hidden_channels=increase_hidden_channels,
                    increase_mlp=increase_mlp,
                    multi_query=multi_query,
                )
                for _ in range(num_blocks)
            ]
        )
        self.linear_out = nn.Linear(hidden_channels, out_channels)

    def forward(self, inputs: torch.Tensor, attention_mask=None) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        inputs : Tensor with shape (..., num_items, num_channels)
            Input data
        attention_mask : None or Tensor or xformers.ops.AttentionBias
            Optional attention mask

        Returns
        -------
        outputs : Tensor with shape (..., num_items, num_channels)
            Outputs
        """
        if self.is_causal:
            n_tokens = inputs.shape[1]
            attention_mask = torch.nn.Transformer.generate_square_subsequent_mask(
                n_tokens, device=inputs.device
            )

        h = self.linear_in(inputs)
        for block in self.blocks:
            h = block(h, attention_mask=attention_mask)
        outputs = self.linear_out(h)
        return outputs
