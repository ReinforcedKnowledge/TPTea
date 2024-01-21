from __future__ import annotations

import json
import logging
from collections.abc import MutableMapping
from typing import Any, Dict, Optional, TypeVar

KT = TypeVar("KT")
VT = TypeVar("VT")


class InvertibleDict(MutableMapping[KT, VT]):
    """An invertible (one-to-one) mapping.

    Attempting to set multiple keys to the same value will raise a ValueError.

    invariant: _forward and _backward are mathematical inverses
        i.e. _forward[a] == b if and only if _backward[b] == a
    """

    __slots__ = ("_forward", "_backward")

    _forward: Dict[KT, VT]
    _backward: Dict[VT, KT]

    def __init__(
        self,
        forward: Optional[Dict[KT, VT]] = None,
        /,
        *,
        _backward: Optional[Dict[VT, KT]] = None,
    ):
        if (forward is None) and (_backward is None):
            self._forward = {}
            self._backward = {}
        elif (forward is not None) and (_backward is not None):
            self._check_non_invertible()
            self._forward = forward
            self._backward = _backward
        else:
            if forward is not None:
                self._forward = forward
                self._backward = {value: key for key, value in self._forward.items()}
            else:
                self._backward = _backward
                self._forward = {value: key for key, value in self._backward.items()}

    def _check_non_invertible(self):
        # Check if the sizes match, accounting for None values
        forward_none_keys = {k for k, v in self._forward.items() if v is None}
        backward_none_keys = {k for k, v in self._backward.items() if v is None}

        if len(self._forward) - len(forward_none_keys) != len(self._backward) - len(
            backward_none_keys
        ):
            raise ValueError("The dictionaries do not form a perfect 1-to-1 mapping.")

        # Check each item in _forward
        for key, value in self._forward.items():
            if value is not None:
                if self._backward.get(value, None) != key:
                    self._raise_non_invertible(key, value)
            else:
                # If the value is None, it's okay for _backward to either not have the key or have it with a None value
                if key not in backward_none_keys and key in self._backward:
                    self._raise_non_invertible(key, value)

        # Check each item in _backward
        for key, value in self._backward.items():
            if value is not None:
                if self._forward.get(value, None) != key:
                    self._raise_non_invertible(value, key)
            else:
                # If the value is None, it's okay for _forward to either not have the key or have it with a None value
                if key not in forward_none_keys and key in self._forward:
                    self._raise_non_invertible(value, key)

    def _raise_non_invertible(self, key1: KT, key2: KT, value: VT):
        raise ValueError(f"non-invertible: {key1}, {key2} both map to: {value}")

    @property
    def inv(self, value: VT) -> KT:
        return self._backward[value]

    def __getitem__(self, item: KT) -> VT:
        return self._forward[item]

    def __setitem__(self, key: KT, value: VT):
        try:
            old_key = self._backward[value]
            if old_key != key:
                self._raise_non_invertible(old_key, key, value)
        except KeyError:
            pass

        try:
            old_value = self._forward[key]
            del self._backward[old_value]
        except KeyError:
            pass

        self._forward[key] = value
        self._backward[value] = key

    def __delitem__(self, key: KT):
        old_value = self._forward.pop(key)
        del self._backward[old_value]

    def __iter__(self):
        return iter(self._forward)

    def __len__(self) -> int:
        return len(self._forward)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self._forward!r})"

    def clear(self) -> None:
        self._forward.clear()
        self._backward.clear()

    def get_forward_dict(self) -> Dict[KT, VT]:
        return dict(self._forward)

    def get_backward_dict(self) -> Dict[KT, VT]:
        return dict(self._backward)


# for saving Invertible dict with JSON
class InvertibleDictEncoder(json.JSONEncoder):
    def default(self, o: Any) -> Any:
        if isinstance(o, InvertibleDict):
            return o.get_forward_dict()
        return json.JSONEncoder.default(self, o)


def setup_logger(name, level=logging.INFO):
    # Create a logger
    logger = logging.getLogger(name)

    # Set the logging level based on the provided parameter
    logger.setLevel(level)

    # Create a console handler
    handler = logging.StreamHandler()

    # Create a formatter and set it for the handler
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)

    # Add the handler to the logger
    logger.addHandler(handler)

    return logger
