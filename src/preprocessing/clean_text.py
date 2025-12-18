from __future__ import annotations
import re

_WHITESPACE_RE = re.compile(r"\s+")
_BULLET_RE = re.compile(r"[•\u2022\u25CF\u25AA]+")

def clean_text(text: str) -> str:
    if not text:
        return ""
    t = text.replace("\r", "\n")
    t = _BULLET_RE.sub(" - ", t)
    t = _WHITESPACE_RE.sub(" ", t)
    return t.strip()
