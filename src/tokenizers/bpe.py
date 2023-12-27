import json
import logging
import re
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from os.path import exists
from typing import Any, List, Optional

from .base import BaseTokenizer
from .constants import BOS, EOS, PAD, SPACE, UNK
from .utils import InvertibleDict, InvertibleDictEncoder, setup_logger

logger = setup_logger("logger", logging.ERROR)


@dataclass
class BPEConfig:
    vocab_size: int
    special_tokens = [UNK, PAD, BOS, EOS, SPACE]
    base_vocab: str


class BPE(BaseTokenizer):
    def __init__(self, config):
        self.vocab_size = config.vocab_size
        self.special_tokens = config.special_tokens
        self.base_vocab = config.base_vocab
        self.base_vocab_size = len(self.base_vocab) + len(self.special_tokens)
        self.num_merges = self.vocab_size - self.base_vocab_size
        self.create_vocab()

    def __len__(self) -> int:
        return len(self.vocab)

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

    def train(self, corpus: str, debug: bool = False):
        if debug:
            logger.setLevel(logging.DEBUG)

        # normalize the corpus
        corpus = self.normalize(corpus)
        logger.debug(f"Normalized corpus: {corpus}")

        # remove white spaces
        corpus = self.pre_tokenize(corpus)
        logger.debug(f"Pretokenized corpus: {corpus}")

        # create a dictionary of counts
        train_dict = Counter(corpus.split())
        train_dict = Counter({" ".join(list(k)): v for k, v in train_dict.items()})
        logger.debug(f"Train dict: {train_dict}")
        logger.debug(f"Starting vocab: {self.vocab}\n")

        for i in range(self.num_merges):
            logger.debug(f"Merge num: {i}")
            pairs = self.get_stats(train_dict)
            logger.debug(f"\tUpdated pairs frequencies: {pairs}")
            best = max(pairs, key=pairs.get)
            logger.debug(f"\tMerge rule: {best}\n")
            train_dict = self.merge_vocab(best, train_dict)
            self.vocab["".join(best)] = self.base_vocab_size + i

        logger.debug(f"End vocab: {self.vocab}")

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

    def tokenize(
        self, text: str
    ) -> List[Optional[int]]:  # TODO: Check why optional is needed with mypy
        clean_text = self.pre_tokenize(self.normalize(text)).split()
        print(clean_text)
        tokens = []

        # prepend a bos token at the start
        tokens.append(self.vocab.get(BOS))

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
                        unknown_token_index = self.vocab.get(UNK)
                        if unknown_token_index is not None:
                            tokens.append(unknown_token_index)

        # append a eos token at the end
        tokens.append(self.vocab.get(EOS))

        return tokens

    def tokenize_batch(
        self, texts: List[str], num_threads: int = 4
    ) -> List[List[Optional[int]]]:
        with ThreadPoolExecutor(num_threads) as e:
            return list(e.map(self.tokenize, texts))

    def detokenize(self, inputs: List[int]) -> str:
        BOS_EOS = (self.vocab[BOS], self.vocab[EOS])
        SPACE_INDEX = self.vocab[SPACE]

        detokenized_string = ""

        for index in inputs:
            if index in BOS_EOS:
                continue
            if index == SPACE_INDEX:
                detokenized_string += " "
            else:
                detokenized_string += self.vocab.inv[index]

        return detokenized_string.lstrip()

    def detokenize_batch(self, inputs: List[int], num_threads: int = 4) -> List[str]:
        with ThreadPoolExecutor(num_threads) as e:
            return list(e.map(self.detokenize, inputs))

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
