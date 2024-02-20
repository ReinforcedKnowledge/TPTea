"""
We're trying to closely follow the descriptions provided in this paper: Radford, Alec and Karthik Narasimhan. “Improving Language Understanding by Generative Pre-Training.” (2018).
The paper also talks about training and optimization. This will be implemented later as a training scheme. More details can be found on my blog: https://reinforcedknowledge.com
"""

from dataclasses import dataclass
from typing import Callable, Optional

import torch
import torch.nn as nn

from .causal_self_attention import CausalSelfAttention


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
