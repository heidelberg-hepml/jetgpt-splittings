from typing import Optional

from torch import Tensor
from torch.nn.functional import scaled_dot_product_attention as torch_sdpa


def scaled_dot_product_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    attn_mask: Optional[Tensor] = None,
    is_causal=False,
) -> Tensor:
    """Execute (vanilla) scaled dot-product attention.

    Parameters
    ----------
    query : Tensor
        of shape [batch, head, item, d]
    key : Tensor
        of shape [batch, head, item, d]
    value : Tensor
        of shape [batch, head, item, d]
    attn_mask : Optional[Tensor]
        Attention mask
    is_causal: bool

    Returns
    -------
    Tensor
        of shape [batch, head, item, d]
    """
    return torch_sdpa(query, key, value, attn_mask=attn_mask, is_causal=is_causal)
