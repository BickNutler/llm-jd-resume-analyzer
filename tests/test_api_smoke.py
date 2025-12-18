import pytest
from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)

def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert "status" in r.json()

def test_analyze_baseline_only():
    payload = {
        "resume_text": "Python SQL scikit-learn FastAPI Docker",
        "job_text": "Looking for Python, scikit-learn, MLOps, Docker and APIs"
    }
    r = client.post("/analyze", json=payload)
    assert r.status_code == 200
    data = r.json()
    assert 0.0 <= data["match_score"] <= 1.0
    assert "baseline" in data
    assert "missing_keywords" in data
