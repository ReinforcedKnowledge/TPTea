import math
from dataclasses import dataclass
from typing import Callable, Optional

import torch
import torch.nn as nn
from torch.nn import functional as F

# TODO: KV-caching
# TODO: Flash attention
# TODO: Grouped Multi-Query Attention


def gpt2_initialization(model):
    # Each block contains two residual layers, the multi-head attention and the MLP
    n_res_layers = 2 * 12  # n_layer
    scale = 1 / math.sqrt(n_res_layers**0.5)

    for name, param in model.named_parameters():
        if "weight" in name and "ln" not in name:  # Issue with nn.Parameter
            if (
                "attn" in name or "mlp" in name
            ):  # Scale weights for attention and MLP layers
                nn.init.normal_(param.data, mean=0.0, std=0.02 * scale)
            else:  # Do not scale embeddings or other layers, which are not residual layers
                nn.init.normal_(param.data, mean=0.0, std=0.02)
        elif "bias" in name:
            nn.init.constant_(param.data, 0)
        elif isinstance(param, nn.LayerNorm):  # LayerNorm layers initialization
            param.bias.data.zero_()
            param.weight.data.fill_(1.0)
    return model


# Configuration class for model parameters
@dataclass
class GPT2Config:
    vocab_size: int = 50257
    block_size: int = 1024
    n_layer: int = 12
    n_head: int = 12
    n_embed: int = 768
    initialization: Optional[Callable] = gpt2_initialization


# Causal Self-Attention
class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        # Combining key, query, and value in a single matrix
        self.key_query_value = nn.Linear(config.n_embed, 3 * config.n_embed)
        self.proj = nn.Linear(config.n_embed, config.n_embed)
        self.dropout = nn.Dropout(0.1)

        # Make the causal mask a part of the module's state through register_buffer ensures
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(config.block_size, config.block_size)).view(
                1, 1, config.block_size, config.block_size
            ),
        )

    def forward(self, x):
        batch_size, sequence_length, embed_size = x.size()

        # Project to key, query, value in a single pass
        kqv = self.key_query_value(x)
        k, q, v = kqv.split(embed_size, dim=-1)

        # Reshape for multi-head attention
        def reshape_to_heads(tensor):
            return tensor.view(
                batch_size,
                sequence_length,
                self.config.n_head,
                embed_size // self.config.n_head,
            ).transpose(1, 2)

        k, q, v = map(reshape_to_heads, [k, q, v])

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
            .view(batch_size, sequence_length, embed_size)
        )

        return self.proj(y)


# MLP
class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.fc1 = nn.Linear(config.n_embed, 4 * config.n_embed)
        self.fc2 = nn.Linear(4 * config.n_embed, config.n_embed)
        self.act = nn.GELU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        return self.fc2(x)


# Transformer Block
class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embed)
        self.ln2 = nn.LayerNorm(config.n_embed)
        self.attn = CausalSelfAttention(config)
        self.mlp = MLP(config)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = x + self.dropout(self.attn(self.ln1(x)))
        x = x + self.dropout(self.mlp(self.ln2(x)))
        return x


# GPT-2 Model
class GPT2(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embed)
        self.pos_emb = nn.Parameter(torch.zeros(1, config.block_size, config.n_embed))
        self.drop = nn.Dropout(0.1)
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embed)

        self.head = nn.Linear(config.n_embed, config.vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight

        self.apply(self.config.initialization)

    def forward(self, idx):
        _, sequence_length = idx.size()
        token_embeddings = self.tok_emb(idx)
        position_embeddings = self.pos_emb[:, :sequence_length, :]
        x = self.drop(token_embeddings + position_embeddings)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.head(x)
        return logits


gpt2_config = GPT2Config()
gpt2 = GPT2(gpt2_config)
print(gpt2)
