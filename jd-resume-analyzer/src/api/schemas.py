from __future__ import annotations
from pydantic import BaseModel, Field, HttpUrl
from typing import Optional, Dict, Any, List

class AnalyzeRequest(BaseModel):
    resume_text: Optional[str] = Field(default=None, description="Raw resume text.")
    resume_path: Optional[str] = Field(default=None, description="Optional path to a local resume file (txt/pdf).")
    job_text: Optional[str] = Field(default=None, description="Raw job description text.")
    job_url: Optional[HttpUrl] = Field(default=None, description="Optional URL to fetch job posting text.")

class BaselineOut(BaseModel):
    tfidf_score: float
    keyword_coverage: float
    missing_keywords: List[str]
    matched_keywords: List[str]

class AnalyzeResponse(BaseModel):
    match_score: float = Field(..., ge=0.0, le=1.0)
    baseline: BaselineOut
    skills_resume: Dict[str, List[str]]
    skills_job: Dict[str, List[str]]
    missing_keywords: List[str]
    recommended_keywords: List[str]
    explanations: Dict[str, Any]
