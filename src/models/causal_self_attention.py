import math

import torch
import torch.nn as nn
from torch.nn import functional as F


# Causal Self-Attention
class CausalSelfAttention(nn.Module):
    """Classic dense self attention.
    Heavily inspired from Karpathy's: https://github.com/karpathy/minGPT/blob/37baab71b9abea1b76ab957409a1cc2fbfba8a26/mingpt/model.py#L29
    The causal mask is stored in a buffer, which makes this class less flexible since
    it'll work with sequences up until context_size. This will be modified later on
    for inference.

    Args:
        config: Requires the following keys:
            n_embed (int): the embedding dimension
            n_head (int): the number of heads
            context_size (int): the maximum context size during training
    """

    def __init__(self, config):
        super().__init__()

        self.config = config

        # Combining key, query, and value in a single matrix
        self.key_query_value = nn.Linear(config.n_embed, 3 * config.n_embed)
        self.proj = nn.Linear(config.n_embed, config.n_embed)
        self.dropout = nn.Dropout(0.1)

        # Make the causal mask a part of the module's state through register_buffer
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(config.context_size, config.context_size)).view(
                1, 1, config.context_size, config.context_size
            ),
        )

    def forward(self, x):
        batch_size, sequence_length, n_embed = x.size()

        # Project to key, query, value in a single pass
        kqv = self.key_query_value(x)
        k, q, v = kqv.split(n_embed, dim=-1)

        # Reshape for multi-head attention
        k, q, v = map(
            lambda t: t.view(
                batch_size,
                sequence_length,
                self.config.n_head,
                n_embed // self.config.n_head,
            ).transpose(1, 2),
            [k, q, v],
        )

        # Scaled dot product attention
        attn = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        attn = attn.masked_fill(
            self.mask[:, :, :sequence_length, :sequence_length] == 0, float("-inf")
        )
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        # Apply attention to the value and combine the heads back to the original tensor shape
        y = (
            (attn @ v)
            .transpose(1, 2)
            .contiguous()
            .view(batch_size, sequence_length, n_embed)
        )

        return self.proj(y)


# Causal Self-Attention with ALiBi
class CausalSelfAttentionWithALiBi(nn.Module):
    """Dense self attention with Attention Linear Biases
    Press, Ofir et al. “Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation.” ArXiv abs/2108.12409 (2021): n. pag.
    https://arxiv.org/abs/2108.12409

    The mask here contains both the causal mask and the linear biases.

    Args:
        config: Requires the following keys:
            n_embed (int): the embedding dimension
            n_head (int): the number of heads
            context_size (int): the maximum context size during training
        slopes: A tensor of size [n_head]. We choose to initialize the slopes in the
            model and pass them by reference to the attention module.
    """

    def __init__(self, config, slopes):
        super().__init__()

        self.config = config
        self.slopes = slopes

        # Combining key, query, and value in a single matrix
        self.key_query_value = nn.Linear(config.n_embed, 3 * config.n_embed)
        self.proj = nn.Linear(config.n_embed, config.n_embed)
        self.dropout = nn.Dropout(0.1)

        # Make the causal mask a part of the module's state through register_buffer
        # Idea from the official repo ofirpress: https://github.com/ofirpress/attention_with_linear_biases/blob/master/fairseq/models/transformer.py#L1011
        self.register_buffer(
            "mask", self.create_combined_mask(config.context_size, slopes)
        )

    def create_combined_mask(self, context_size, slopes):
        # Create the base linear mask
        linear_mask = torch.tril(
            torch.arange(0, -context_size, -1).unsqueeze(1) + torch.arange(context_size)
        )

        # Multiply by the slope for each head
        linear_mask = slopes.unsqueeze(1).unsqueeze(2) * linear_mask

        # Fill the upper diagonal part by -inf for the causal mask
        causal = torch.triu(torch.ones_like(linear_mask), diagonal=1)
        linear_mask[causal.bool()] = float("-inf")

        return linear_mask

    def forward(self, x):
        batch_size, sequence_length, n_embed = x.size()

        # Project to key, query, value in a single pass
        kqv = self.key_query_value(x)
        k, q, v = kqv.split(n_embed, dim=-1)

        # Reshape for multi-head attention
        k, q, v = map(
            lambda t: t.view(
                batch_size,
                sequence_length,
                self.config.n_head,
                n_embed // self.config.n_head,
            ).transpose(1, 2),
            [k, q, v],
        )

        # Scaled dot product attention
        attn = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        attn += self.mask.unsqueeze(0)[
            :, :, :sequence_length, :sequence_length
        ]  # Broadcast the mask depending on batch size
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        # Apply attention to the value and combine the heads back to the original tensor shape
        y = (
            (attn @ v)
            .transpose(1, 2)
            .contiguous()
            .view(batch_size, sequence_length, n_embed)
        )

        return self.proj(y)
