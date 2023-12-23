import unicodedata
from abc import ABC, abstractmethod
from typing import Dict


class BaseTokenizer(ABC):
    def __init__(self):
        raise NotImplementedError()

    @abstractmethod
    def tokenize(self, text: str):
        raise NotImplementedError()

    @property
    def get_vocab(self) -> Dict[str, int]:
        raise NotImplementedError()

    def __len__(self) -> int:
        raise NotImplementedError()

    def pre_tokenize(self, text: str) -> str:
        return text.replace(r"\s+", "")

    def normalize(self, text: str) -> str:
        return unicodedata.normalize("NFKD", text)
