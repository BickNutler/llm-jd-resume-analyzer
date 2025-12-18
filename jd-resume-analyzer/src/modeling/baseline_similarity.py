from __future__ import annotations
from dataclasses import dataclass
from ..features.tfidf_features import default_tfidf
from ..features.keyword_extractor import extract_keywords, fuzzy_contains
from ..utils.text import unique_preserve_order

@dataclass
class BaselineResult:
    tfidf_score: float
    keyword_coverage: float
    missing_keywords: list[str]
    matched_keywords: list[str]

def baseline_compare(resume_text: str, job_text: str, top_k: int = 40) -> BaselineResult:
    tfidf = default_tfidf()
    tfidf_score = tfidf.score(resume_text, job_text)

    job_kw = extract_keywords(job_text)
    # keep top_k earliest keywords to avoid huge lists
    job_kw = job_kw[:top_k] if top_k else job_kw

    matched=[]
    missing=[]
    for kw in job_kw:
        if fuzzy_contains(resume_text, kw, threshold=90):
            matched.append(kw)
        else:
            missing.append(kw)

    denom = max(len(job_kw), 1)
    coverage = len(matched) / denom

    return BaselineResult(
        tfidf_score=tfidf_score,
        keyword_coverage=float(coverage),
        missing_keywords=unique_preserve_order(missing),
        matched_keywords=unique_preserve_order(matched),
    )
