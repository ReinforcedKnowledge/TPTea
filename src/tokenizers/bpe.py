from os.path import exists
from .base import BaseTokenizer
from dataclasses import dataclass
from typing import Dict
from collections import defaultdict, Counter
import re
import json

VOCAB_SIZE = 13


@dataclass()
class BPEConfig:
    vocab_size: int = VOCAB_SIZE
    special_tokens = ["<unk>", "<pad>", "<beo>", "<eos>"]
    base_vocab: str = "bghnpsu"


class BPE(BaseTokenizer):
    def __init__(self, config):
        self.vocab_size = config.vocab_size
        self.special_tokens = config.special_tokens
        self.base_vocab = config.base_vocab
        self.base_vocab_size = len(self.base_vocab)
        self.num_merges = self.vocab_size - self.base_vocab_size
        self.vocab = {char: index for index, char in enumerate(self.base_vocab)}

    def tokenize(self, text: str):
        raise NotImplementedError()

    def get_vocab(self) -> Dict[str, int]:
        return self.vocab

    def __len__(self) -> int:
        return len(self.vocab)

    def train(self, corpus: str):
        # normalize the corpus
        corpus = self.normalize(corpus)

        # remove white spaces
        corpus = self.pre_tokenize(corpus)

        train_dict = Counter(corpus.split())
        train_dict = {" ".join(list(k)): v for k, v in train_dict.items()}
        print(train_dict)

        print(f"{self.num_merges=}")
        for i in range(self.num_merges):
            pairs = self.get_stats(train_dict)
            best = max(pairs, key=pairs.get)
            train_dict = self.merge_vocab(best, train_dict)
            self.vocab["".join(best)] = self.base_vocab_size + i
            print("----------------------")
            print(train_dict)
            print("----------------------")

        print(self.vocab)

    def get_stats(self, train_dict):
        pairs = defaultdict(int)
        for word, freq in train_dict.items():
            symbols = word.split()
            for i in range(len(symbols) - 1):
                pairs[symbols[i], symbols[i + 1]] += freq
        return pairs

    def merge_vocab(self, pair, v_in):
        v_out = {}
        bigram = re.escape(" ".join(pair))
        p = re.compile(r"(?<!\S)" + bigram + r"(?!\S)")
        for word in v_in:
            w_out = p.sub("".join(pair), word)
            v_out[w_out] = v_in[word]
        return v_out

    def save(self, filename, overwrite=False):
        if exists(filename) and not overwrite:
            raise ValueError(f"File {filename} already exists!")

        try:
            with open(filename, "w") as fp:
                json.dump(self.get_vocab(), fp)
        except Exception as e:
            raise IOError(f"An I/O error occured while writing {filename} : {e}")

    def load(self, filename):
        if not exists(filename):
            raise (f"File {filename} does not exist!")

        try:
            with open(filename, "r") as fp:
                self.vocab = json.load(fp)
        except Exception as e:
            raise IOError(f"An I/O error occured while reading {filename} : {e}")


if __name__ == "__main__":
    cfg = BPEConfig()
    bpe = BPE(cfg)
    corpus = "hug " * 10 + "pug " * 5 + "pun " * 12 + "bun " * 4 + "hugs " * 5

    bpe.train(corpus)
