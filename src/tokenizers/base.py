import re
from abc import ABC, abstractmethod
from typing import Any, List, Optional

from .constants import SPACE
from .utils import InvertibleDict


class BaseTokenizer(ABC):
    """
    A base class for tokenizers. All tokenizers should subclass this class.

    Subclassers should always implement the `tokenize()` method, which will also
    be the default when calling the layer directly on inputs.
    """

    def __init__(self):
        raise NotImplementedError()

    def __call__(self, text: str):
        return self.tokenize(text)

    @abstractmethod
    def tokenize(self, text: str) -> List[Optional[int]]:
        """Transform strings into tokens.

        Args:
            text (str): input string
        """
        raise NotImplementedError(
            "No implementation of `tokenize()` was found for "
            f"{self.__class__.__name__}. All tokenizers should implement "
            "`tokenize()`."
        )

    @abstractmethod
    def detokenize(self, inputs: List[int]) -> str:
        """Transform tokens back into strings;

        Args:
            inputs (List[int]): list of token ids
        """
        raise NotImplementedError(
            "No implementation of `detokenize()` was found for "
            f"{self.__class__.__name__}."
        )

    @property
    def get_vocab(self) -> InvertibleDict[Any, int]:
        raise NotImplementedError()

    def __len__(self) -> int:
        raise NotImplementedError()

    def pre_tokenize(self, text: str) -> str:
        text = re.sub(r"\s+", " ", text)
        text = text.replace(" ", f"{SPACE} ")
        return text

    # TODO: implement later
    def normalize(self, text: str) -> str:
        # return unicodedata.normalize("NFKD", text)
        return text

    def save(self, filename: str):
        raise NotImplementedError()

    def load(self, filename: str):
        raise NotImplementedError()
