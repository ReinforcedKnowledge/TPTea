import json
from collections import Counter
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from os.path import exists
from typing import Any, Dict, List, Optional, Tuple

from .base import BaseTokenizer
from .constants import BOS, EOS, PAD, SPACE, UNK
from .utils import InvertibleDict, InvertibleDictEncoder


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
        self._create_vocab()

    def __len__(self) -> int:
        return len(self.vocab)

    def _create_vocab(self) -> None:
        # combine special tokens and base vocabulary
        combined_vocab = self.special_tokens + list(self.base_vocab)

        # create the vocabulary as an invertible mapping: char <=> index
        self.vocab = InvertibleDict(
            {char: index for index, char in enumerate(combined_vocab)}
        )

    @property
    def get_vocab(self) -> InvertibleDict[Any, int]:
        return self.vocab

    def _get_corpus_words_frequencies(self, corpus: str) -> Counter[str, int]:
        corpus = self.normalize(corpus)
        corpus = self.pre_tokenize(corpus)
        return Counter(
            {" ".join(list(k)): v for k, v in Counter(corpus.split()).items()}
        )

    def _create_pairs_from_symbols(self, symbols):
        symbols = symbols.split()
        return [(symbols[i], symbols[i + 1]) for i in range(len(symbols) - 1)]

    def _create_words_dict(self, train_dict):
        return {
            id: [freq, self._create_pairs_from_symbols(word)]
            for id, (word, freq) in enumerate(train_dict.items())
        }

    def _create_pairs_dicts(self, words_dict):
        pairs_dict = {}
        for id, (word_freq, linked_pairs) in words_dict.items():
            for pair in linked_pairs:
                pairs_dict.setdefault(pair, [0, set()])[0] += word_freq
                pairs_dict[pair][1].add(id)
        return pairs_dict

    def _create_words_and_pairs_dicts(self, train_dict):
        words_dict = self._create_words_dict(train_dict)
        pairs_dict = self._create_pairs_dicts(words_dict)
        return words_dict, pairs_dict

    def _update_pairs_dict(self, pairs_dict, pair, freq_change, word_id):
        if pair in pairs_dict:
            pairs_dict[pair][0] += freq_change
            if freq_change > 0:  # If we are adding frequency, add the word ID
                pairs_dict[pair][1].add(word_id)
            if (
                pairs_dict[pair][0] <= 0
            ):  # If frequency is zero or less, delete the pair
                del pairs_dict[pair]
        else:
            pairs_dict[pair] = (
                [freq_change, {word_id}] if freq_change > 0 else [freq_change, set()]
            )

    def _merge_pairs(self, words_dict, pairs_dict, max_freq_pair):
        max_freq_pair_merged = "".join(max_freq_pair)
        for word_id in words_dict:
            word_freq = words_dict[word_id][0]
            pairs = words_dict[word_id][1]
            new_pairs = []
            i = 0
            while i < len(pairs):
                if pairs[i] == max_freq_pair:
                    # Check for preceding pair
                    if i > 0 and new_pairs:
                        prev_pair = new_pairs[-1]
                        new_pairs[-1] = (prev_pair[0], max_freq_pair_merged)
                        self._update_pairs_dict(
                            pairs_dict, prev_pair, -word_freq, word_id
                        )
                        self._update_pairs_dict(
                            pairs_dict, new_pairs[-1], word_freq, word_id
                        )
                    # Check for following pair
                    if i < len(pairs) - 1:
                        next_pair = (max_freq_pair_merged, pairs[i + 1][1])
                        new_pairs.append(next_pair)
                        self._update_pairs_dict(
                            pairs_dict, pairs[i + 1], -word_freq, word_id
                        )
                        self._update_pairs_dict(
                            pairs_dict, next_pair, word_freq, word_id
                        )
                        i += 1  # Skip the next pair as it's now merged
                else:
                    new_pairs.append(pairs[i])
                i += 1
            words_dict[word_id][1] = new_pairs

        # Delete max_freq_pair from pairs_dict
        del pairs_dict[max_freq_pair]

    def _preprocess_corpus(self, corpus: Iterable, num_threads: int) -> Counter:
        if isinstance(corpus, str):
            return self._get_corpus_words_frequencies(corpus)
        else:
            train_dict = Counter()
            with ThreadPoolExecutor(max_workers=num_threads) as e:
                corpora = e.map(self._get_corpus_words_frequencies, corpus)
                for processed_corpus in corpora:
                    train_dict.update(processed_corpus)
            return train_dict

    def _initialize_training(self, train_dict: Counter) -> Tuple[Dict, Dict]:
        return self._create_words_and_pairs_dicts(train_dict)

    def _train_loop(self, train_dict: Dict, pairs_dict: Dict, num_merges: int):
        for i in range(num_merges):
            best = max(pairs_dict, key=lambda pair: pairs_dict[pair][0])
            self.vocab["".join(best)] = self.base_vocab_size + i
            self._merge_pairs(train_dict, pairs_dict, best)

    def train(self, corpus: Iterable, num_threads: int = 4):
        train_dict = self._preprocess_corpus(corpus, num_threads)
        train_dict, pairs_dict = self._initialize_training(train_dict)
        self._train_loop(train_dict, pairs_dict, self.num_merges)

    def _break_into_subwords(self, word: str) -> List[str]:
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
        tokens = []

        # prepend a bos token at the start
        tokens.append(self.vocab.get(BOS))

        while clean_text:
            word = clean_text.pop(0)
            if word in self.vocab:
                tokens.append(self.vocab[word])
            else:
                # handle unknown words by breaking them down into subwords
                subwords = self._break_into_subwords(word)
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
            print(f"Vocab: {self.vocab}")
            print(f"Index: {index}")
            print(f"Inv vocab: {self.vocab.get_backward_dict()}")
            if index in BOS_EOS:
                continue
            elif index == SPACE_INDEX:
                detokenized_string += " "
            elif self.vocab.inv(index).endswith(f"{SPACE}"):
                detokenized_string += self.vocab.inv(index)[:-1] + " "
            else:
                detokenized_string += self.vocab.inv(index)

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
