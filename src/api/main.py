from __future__ import annotations
import logging
from fastapi import FastAPI, HTTPException
from ..utils.logging import configure_logging
from ..utils.config import settings
from .schemas import AnalyzeRequest, AnalyzeResponse, BaselineOut
from ..ingestion.fetch_job_posting import fetch_job_text
from ..ingestion.load_resume import load_text_from_file
from ..modeling.ranker import analyze

configure_logging()
log = logging.getLogger("api")

app = FastAPI(
    title="JD–Resume Analyzer",
    version="1.0.0",
    description="ATS-style analyzer with baseline NLP + optional LLM extraction"
)

@app.get("/health")
def health() -> dict:
    return {"status":"ok", "llm_provider": settings.llm_provider}

@app.post("/analyze", response_model=AnalyzeResponse)
def analyze_endpoint(req: AnalyzeRequest) -> AnalyzeResponse:
    # Resolve resume text
    resume_text = req.resume_text
    if not resume_text and req.resume_path:
        try:
            resume_text = load_text_from_file(req.resume_path)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to load resume_path: {e}")

    # Resolve job text
    job_text = req.job_text
    if not job_text and req.job_url:
        try:
            job_text = fetch_job_text(str(req.job_url))
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to fetch job_url: {e}")

    if not resume_text:
        raise HTTPException(status_code=400, detail="Provide resume_text or resume_path")
    if not job_text:
        raise HTTPException(status_code=400, detail="Provide job_text or job_url")

    try:
        result = analyze(resume_text=resume_text, job_text=job_text)
    except Exception as e:
        log.exception("Analyze failed")
        raise HTTPException(status_code=500, detail=str(e))

    return AnalyzeResponse(
        match_score=result.match_score,
        baseline=BaselineOut(**result.baseline.__dict__),
        skills_resume=result.skills_resume,
        skills_job=result.skills_job,
        missing_keywords=result.missing_keywords,
        recommended_keywords=result.recommended_keywords,
        explanations=result.explanations,
    )
