from __future__ import annotations

import json
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
        if forward is None:
            self._forward = {}
            self._backward = {}
        elif _backward is not None:
            self._forward = forward
            self._backward = _backward
        else:
            self._forward = forward
            self._backward = {value: key for key, value in self._forward.items()}
            self._check_non_invertible()

    def _check_non_invertible(self):
        if len(self._backward) != len(self._forward):
            for key, value in self._forward.items():
                other_key = self._backward.get(value, None)
                if other_key is not None and other_key != key:
                    self._raise_non_invertible(key, other_key, value)

    def _raise_non_invertible(self, key1: KT, key2: KT, value: VT):
        raise ValueError(f"non-invertible: {key1}, {key2} both map to: {value}")

    @property
    def inv(self) -> InvertibleDict[VT, KT]:
        """A mutable view of the inverse dict."""
        return InvertibleDict(self._backward, _backward=self._forward)

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


# for saving Invertible dict with JSON
class InvertibleDictEncoder(json.JSONEncoder):
    def default(self, o: Any) -> Any:
        if isinstance(o, InvertibleDict):
            return o.get_forward_dict()
        return json.JSONEncoder.default(self, o)
