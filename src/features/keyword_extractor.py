from __future__ import annotations
from rapidfuzz import fuzz
from ..utils.text import normalize_token, tokens, unique_preserve_order

STOPWORDS = set([
    "and","or","the","a","an","to","of","in","for","with","on","as","at","by","from",
    "is","are","be","this","that","it","we","you","your","our","their","they",
    "experience","years","year","role","responsibilities","required","preferred",
])

def extract_keywords(text: str, min_len: int = 3) -> list[str]:
    toks=[t for t in tokens(text) if len(t) >= min_len]
    toks=[t for t in toks if t not in STOPWORDS]
    return unique_preserve_order(toks)

def fuzzy_contains(haystack: str, needle: str, threshold: int = 90) -> bool:
    h = normalize_token(haystack)
    n = normalize_token(needle)
    if not h or not n:
        return False
    # quick exact substring
    if n in h:
        return True
    # fuzzy match against sliding windows of tokens
    # (simple and fast enough for portfolio-scale inputs)
    return fuzz.partial_ratio(h, n) >= threshold
