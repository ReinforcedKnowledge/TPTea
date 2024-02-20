"""
Trying to find a way to dynamically create a GPT-like model from a config file.
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn

from .causal_self_attention import CausalSelfAttentionWithALiBi


# Configuration class for model parameters. This config is GPT2's.
@dataclass
class ExpConfig:
    vocab_size: int = 50257
    context_size: int = 1024
    n_layer: int = 12
    n_head: int = 12
    n_embed: int = 768


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
    def __init__(self, config, slopes):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embed)
        self.ln2 = nn.LayerNorm(config.n_embed)
        self.attn = CausalSelfAttentionWithALiBi(config, slopes)
        self.mlp = MLP(config)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = x + self.dropout(self.attn(self.ln1(x)))
        x = x + self.dropout(self.mlp(self.ln2(x)))
        return x


# GPT Model with ALiBi
class ExpGPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embed)
        self.drop = nn.Dropout(0.1)

        self.slopes = self.calculate_slopes(config.nhead)
        self.blocks = nn.Sequential(
            *[Block(config, self.slopes) for _ in range(config.n_layer)]
        )

        self.ln_f = nn.LayerNorm(config.n_embed)

        self.head = nn.Linear(config.n_embed, config.vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight

    def calculate_slopes(n_head):
        """
        Idea from the official repo ofirpress: https://github.com/ofirpress/attention_with_linear_biases/blob/4b92f28a005ead2567abe2359f633e73e08f3833/fairseq/models/transformer.py#L742
        """

        def get_slopes_power_of_2(n):
            start = 2 ** (-(2 ** -(math.log2(n) - 3)))
            ratio = start
            return torch.tensor(
                [start * ratio**i for i in range(n)], dtype=torch.float32
            )

        if math.log2(n_head).is_integer():
            return get_slopes_power_of_2(n_head)
        else:
            closest_power_of_2 = 2 ** math.floor(math.log2(n_head))
            slopes = get_slopes_power_of_2(closest_power_of_2)
            return torch.cat((slopes, slopes[::2][: n_head - closest_power_of_2]))

    def forward(self, idx):
        token_embeddings = self.tok_emb(idx)
        x = self.drop(token_embeddings)  # TODO check if people use dropout here
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.head(x)
        return logits
