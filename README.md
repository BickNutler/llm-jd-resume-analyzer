# JD–Resume Analyzer (LLM + NLP + Baseline Similarity) — Portfolio Mini Project

This project provides a **GitHub-ready, reproducible** NLP pipeline and a small AI product:
a service that compares a resume to a job description (JD) and returns:

- ATS-style **match score**
- **missing keywords** and **recommended keywords**
- extracted **skills / tools / requirements** (schema-based)
- reproducible baseline scoring (TF-IDF + fuzzy matching)
- optional **LLM-based structured extraction** (OpenAI-compatible or local Ollama)

## Why this exists (portfolio positioning)

This single repo demonstrates several “nice-to-have” skills for Junior AI / ML Engineer roles:

- **NLP + LLM tooling**: structured information extraction (JSON schema), prompt versioning
- **APIs**: URL ingestion + a FastAPI service for inference
- **MLOps fundamentals**: reproducible pipeline, artifact versioning hooks, tests, CI, Docker
- **Evaluation**: small labeled set + extraction metrics

---

## Quick start (local)

### 1) Create a venv and install dependencies
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
source .venv/bin/activate
pip install -r requirements.txt
```

### 2) Configure environment variables
Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```

You can run the project **without** an LLM (baseline-only).  
If you want LLM extraction, set `LLM_PROVIDER=openai` and `OPENAI_API_KEY=...`
(or `LLM_PROVIDER=ollama` for a local model).

### 3) Run the API
```bash
uvicorn src.api.main:app --reload
```

Open docs:
- http://127.0.0.1:8000/docs

---

## API usage

### POST `/analyze`
Request body (minimal):
```json
{
  "resume_text": "Python, SQL, scikit-learn...",
  "job_text": "We need Python, ML, MLOps..."
}
```

Response (example fields):
```json
{
  "match_score": 0.74,
  "missing_keywords": ["pytorch", "docker"],
  "recommended_keywords": ["fastapi", "mlops"],
  "skills_resume": {...},
  "skills_job": {...},
  "explanations": {...}
}
```

You can also provide a `job_url` instead of `job_text`.

---

## Evaluation

This repo includes a small evaluation harness for skill extraction.

1) Put labeled samples in `data/labeled/` (JSON format shown in `data/labeled/example_label.json`)
2) Run:
```bash
python -m src.evaluation.evaluate_extraction --labeled_dir data/labeled
```

Metrics reported:
- per-field precision / recall (skills, tools, requirements)
- micro-average scores
- consistency checks

---

## Repository layout

```
jd-resume-analyzer/
  README.md
  requirements.txt
  .env.example
  data/
    raw/
      sample_resume.txt
      sample_jd.txt
    labeled/
      example_label.json
  notebooks/
    01_exploration.ipynb  # optional, placeholder
  src/
    api/
      main.py
      schemas.py
    ingestion/
      fetch_job_posting.py
      load_resume.py
    preprocessing/
      clean_text.py
      section_parser.py
    features/
      keyword_extractor.py
      tfidf_features.py
    modeling/
      baseline_similarity.py
      llm_extract.py
      ranker.py
    evaluation/
      evaluate_extraction.py
    utils/
      config.py
      logging.py
      text.py
  tests/
    test_api_smoke.py
    test_schema_validation.py
  docker/
    Dockerfile
  .github/workflows/ci.yml
```

---

## Notes on “production” considerations

This project is intentionally lightweight but structured as a real service. In a production environment you would add:
- request authentication + rate limiting
- caching (e.g., Redis) for repeated analyses
- tracing (OpenTelemetry) and dashboards
- robust document parsing (DOCX, better PDF parsing, etc.)
- prompt monitoring & human review for LLM outputs

---

## License

MIT (add if you want; currently not included).
