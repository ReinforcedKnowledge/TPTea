import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from os.path import exists
from typing import Any, List, Optional

from .base import BaseTokenizer
from .utils import InvertibleDict, InvertibleDictEncoder

VOCAB_SIZE = 13


@dataclass()
class BPEConfig:
    vocab_size: int = VOCAB_SIZE
    special_tokens = ["<unk>", "<pad>", "<bos>", "<eos>", "Ġ"]
    base_vocab: str = "bghnpsu"


class BPE(BaseTokenizer):
    def __init__(self, config):
        self.vocab_size = config.vocab_size
        self.special_tokens = config.special_tokens
        self.base_vocab = config.base_vocab
        self.base_vocab_size = len(self.base_vocab) + len(self.special_tokens)
        self.num_merges = self.vocab_size - self.base_vocab_size
        self.create_vocab()

    def tokenize(self, text: str) -> List[Optional[int]]:
        clean_text = self.pre_tokenize(self.normalize(text)).split()
        tokens = []

        # prepend a bos token at the start
        tokens.append(self.vocab.get("<bos>"))

        while clean_text:
            word = clean_text.pop(0)
            if word in self.vocab:
                tokens.append(self.vocab[word])
            else:
                # handle unknown words by breaking them down into subwords
                subwords = self.break_into_subwords(word)
                for subword in subwords:
                    if subword in self.vocab:
                        tokens.append(self.vocab[subword])
                    else:
                        # handle subwords that are still unknown
                        unknown_token_index = self.vocab.get("<unk>")
                        if unknown_token_index is not None:
                            tokens.append(unknown_token_index)

        # append a eos token at the end
        tokens.append(self.vocab.get("<eos>"))

        return tokens

    def detokenize(self, inputs: List[int]) -> str:
        BOS_INDEX = self.vocab["<bos>"]
        EOS_INDEX = self.vocab["<eos>"]
        SPACE_INDEX = self.vocab["Ġ"]

        detokenized_string = ""

        for index in inputs:
            if index in (BOS_INDEX, EOS_INDEX):
                continue
            if index == SPACE_INDEX:
                detokenized_string += " "
            else:
                detokenized_string += self.vocab.inv[index]

        return detokenized_string.lstrip()

    def break_into_subwords(self, word: str) -> List[str]:
        """Break unknown words into subwords by finding the longest subword that is in the vocab
        and then recursively processing the rest of the word.

        Args:
            word (str): a word that does not belong in the vocabulary

        Returns:
            List[str]: a list of subwords
        """
        subwords = []
        while word:
            # find the longest subword
            for i in range(len(word), 0, -1):
                subword = word[:i]
                if subword in self.vocab or i == 1:
                    subwords.append(subword)
                    word = word[i:]
                    break
        return subwords

    def create_vocab(self) -> None:
        # combine special tokens and base vocabulary
        combined_vocab = self.special_tokens + list(self.base_vocab)

        # create the vocabulary as an invertible mapping: char <=> index
        self.vocab = InvertibleDict(
            {char: index for index, char in enumerate(combined_vocab)}
        )

    @property
    def get_vocab(self) -> InvertibleDict[Any, int]:
        return self.vocab

    def __len__(self) -> int:
        return len(self.vocab)

    def train(self, corpus: str):
        # normalize the corpus
        corpus = self.normalize(corpus)

        # remove white spaces
        corpus = self.pre_tokenize(corpus)

        # create a dictionary of counts
        train_dict = Counter(corpus.split())
        train_dict = Counter({" ".join(list(k)): v for k, v in train_dict.items()})
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
                json.dump(self.get_vocab, fp, cls=InvertibleDictEncoder)
        except Exception as e:
            raise IOError(f"An I/O error occured while writing {filename} : {e}")

    def load(self, filename):
        if not exists(filename):
            raise ValueError(f"File {filename} does not exist!")

        try:
            with open(filename, "r") as fp:
                forward_dict = json.load(fp)
                self.vocab = InvertibleDict(forward_dict)
        except Exception as e:
            raise IOError(f"An I/O error occurred while reading {filename} : {str(e)}")


if __name__ == "__main__":
    cfg = BPEConfig()
    bpe = BPE(cfg)
    corpus = "hug " * 10 + "pug " * 5 + "pun " * 12 + "bun " * 4 + "hugs " * 5

    bpe.train(corpus)
