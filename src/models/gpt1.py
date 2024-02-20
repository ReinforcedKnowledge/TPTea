"""
We're trying to closely follow the descriptions provided in this paper: Radford, Alec and Karthik Narasimhan. “Improving Language Understanding by Generative Pre-Training.” (2018).
The paper also talks about training and optimization. This will be implemented later as a training scheme. More details can be found on my blog: https://reinforcedknowledge.com
"""

import math
from dataclasses import dataclass
from typing import Callable, Optional

import torch
import torch.nn as nn
from torch.nn import functional as F


def gpt1_initialization(model):
    for name, param in model.named_parameters():
        # This will handle position embeddings as well
        if "weight" in name and "ln" not in name:
            nn.init.normal_(param.data, mean=0.0, std=0.02)
        elif "bias" in name:
            nn.init.constant_(param.data, 0)
        elif isinstance(param, nn.LayerNorm):
            param.bias.data.zero_()
            param.weight.data.fill_(1.0)
    return model


# Configuration class for model parameters
@dataclass
class GPT1Config:
    vocab_size: int = 40478
    context_size: int = 512
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    initialization: Optional[Callable] = gpt1_initialization


# Causal Self-Attention
class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        # Combining key, query, and value in a single matrix
        self.key_query_value = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(0.1)

        # Make the causal mask a part of the module's state through register_buffer ensures
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(config.context_size, config.context_size)).view(
                1, 1, config.context_size, config.context_size
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
        self.fc1 = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.fc2 = nn.Linear(4 * config.n_embd, config.n_embd)
        self.act = nn.GELU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        return self.fc2(x)


# Transformer Block
class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attn = CausalSelfAttention(config)
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = self.ln1(self.dropout(self.attn(x)) + x)
        x = self.ln2(self.dropout(self.mlp(x)) + x)
        return x


# GPT-1 Model
class GPT1(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_emb = nn.Parameter(torch.zeros(1, config.context_size, config.n_embd))
        self.drop = nn.Dropout(0.1)
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])

        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight

        self.apply(self.config.initialization)

    def forward(self, idx):
        _, T = idx.size()
        token_embeddings = self.tok_emb(idx)
        position_embeddings = self.pos_emb[:, :T, :]
        x = self.drop(token_embeddings + position_embeddings)
        x = self.blocks(x)
        logits = self.head(x)
        return logits
