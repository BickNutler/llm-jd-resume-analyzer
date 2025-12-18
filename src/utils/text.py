from __future__ import annotations
import re
from typing import Iterable

_WORD_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9+.#/-]*")

def normalize_token(t: str) -> str:
    return re.sub(r"\s+", " ", t.strip().lower())

def tokens(text: str) -> list[str]:
    return [m.group(0).lower() for m in _WORD_RE.finditer(text or "")]

def unique_preserve_order(items: Iterable[str]) -> list[str]:
    seen=set()
    out=[]
    for x in items:
        nx=normalize_token(x)
        if nx and nx not in seen:
            seen.add(nx)
            out.append(nx)
    return out
