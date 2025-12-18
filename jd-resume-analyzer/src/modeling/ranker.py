from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict, Any

from .baseline_similarity import baseline_compare, BaselineResult
from .llm_extract import extract_with_llm, LlmExtraction
from ..utils.text import unique_preserve_order

@dataclass
class AnalyzeResult:
    match_score: float
    baseline: BaselineResult
    skills_resume: Dict[str, list[str]]
    skills_job: Dict[str, list[str]]
    missing_keywords: list[str]
    recommended_keywords: list[str]
    explanations: Dict[str, Any]

def _merge_skill_dict(llm: Optional[LlmExtraction]) -> Dict[str, list[str]]:
    if not llm:
        return {"skills": [], "tools": [], "requirements": []}
    return {
        "skills": llm.skills,
        "tools": llm.tools,
        "requirements": llm.requirements,
    }

def analyze(resume_text: str, job_text: str) -> AnalyzeResult:
    base = baseline_compare(resume_text, job_text)

    llm_resume = extract_with_llm(resume_text)
    llm_job = extract_with_llm(job_text)

    s_resume = _merge_skill_dict(llm_resume)
    s_job = _merge_skill_dict(llm_job)

    # recommended keywords: LLM job tools/skills not present in resume LLM outputs
    resume_all = set(unique_preserve_order(s_resume["skills"] + s_resume["tools"] + s_resume["requirements"]))
    job_all = unique_preserve_order(s_job["skills"] + s_job["tools"] + s_job["requirements"])
    recommended = [k for k in job_all if k not in resume_all]

    # blended score: baseline similarity + keyword coverage
    match_score = 0.55 * base.tfidf_score + 0.45 * base.keyword_coverage
    match_score = max(0.0, min(1.0, float(match_score)))

    explanations = {
        "score_blend": {"tfidf_weight": 0.55, "coverage_weight": 0.45},
        "tfidf_score": base.tfidf_score,
        "keyword_coverage": base.keyword_coverage,
        "llm_enabled": bool(llm_job or llm_resume),
    }

    return AnalyzeResult(
        match_score=match_score,
        baseline=base,
        skills_resume=s_resume,
        skills_job=s_job,
        missing_keywords=base.missing_keywords[:25],
        recommended_keywords=recommended[:25],
        explanations=explanations,
    )
