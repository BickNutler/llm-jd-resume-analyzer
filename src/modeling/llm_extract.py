from __future__ import annotations
import json
from dataclasses import dataclass
from typing import Any, Dict, Optional
import httpx

from ..utils.config import settings
from ..utils.text import unique_preserve_order

DEFAULT_SCHEMA_HINT = {
    "skills": ["python", "machine learning", "statistics"],
    "tools": ["scikit-learn", "pytorch", "aws"],
    "requirements": ["deploy models", "work with APIs"]
}

SYSTEM_PROMPT = """You extract structured information for an ATS-style analyzer.
Return ONLY valid JSON that matches the required schema.
Do not invent items not supported by the input text.
If unsure, omit the item.
"""

USER_PROMPT_TEMPLATE = """Extract a JSON object with keys:
- skills: list of general skills (e.g., machine learning, NLP, MLOps)
- tools: list of tools/technologies (e.g., scikit-learn, FastAPI, AWS)
- requirements: list of requirements/expectations (e.g., deploy models, work with APIs)

Output must be JSON only. Example shape:
{example}

TEXT:
"""{text}"""
"""

@dataclass
class LlmExtraction:
    skills: list[str]
    tools: list[str]
    requirements: list[str]

def _normalize_list(x: Any) -> list[str]:
    if not isinstance(x, list):
        return []
    return unique_preserve_order([str(i) for i in x if str(i).strip()])

def extract_with_llm(text: str) -> Optional[LlmExtraction]:
    """Optional LLM extraction.

    Returns None if no provider is configured.
    Supports:
    - OpenAI-compatible Chat Completions endpoint
    - Ollama chat endpoint
    """
    provider = settings.llm_provider
    if not provider:
        return None

    payload_text = USER_PROMPT_TEMPLATE.format(example=json.dumps(DEFAULT_SCHEMA_HINT), text=text[:12000])

    if provider.lower() == "openai":
        if not settings.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY not set but LLM_PROVIDER=openai")

        # OpenAI-compatible Chat Completions API
        url = settings.openai_base_url.rstrip("/") + "/chat/completions"
        headers = {"Authorization": f"Bearer {settings.openai_api_key}"}
        body = {
            "model": settings.openai_model,
            "temperature": 0,
            "messages": [
                {"role":"system","content": SYSTEM_PROMPT},
                {"role":"user","content": payload_text},
            ],
            "response_format": {"type":"json_object"},
        }
        with httpx.Client(timeout=45.0) as client:
            r = client.post(url, headers=headers, json=body)
            r.raise_for_status()
            data = r.json()
        content = data["choices"][0]["message"]["content"]

    elif provider.lower() == "ollama":
        url = settings.ollama_base_url.rstrip("/") + "/api/chat"
        body = {
            "model": settings.ollama_model,
            "messages": [
                {"role":"system","content": SYSTEM_PROMPT},
                {"role":"user","content": payload_text},
            ],
            "stream": False,
            "format": "json"
        }
        with httpx.Client(timeout=60.0) as client:
            r = client.post(url, json=body)
            r.raise_for_status()
            data = r.json()
        content = data.get("message", {}).get("content", "")

    else:
        raise ValueError(f"Unsupported LLM_PROVIDER: {provider}")

    # Parse JSON safely
    try:
        obj = json.loads(content)
    except json.JSONDecodeError:
        # best-effort attempt to locate JSON
        start = content.find("{")
        end = content.rfind("}")
        if start >= 0 and end > start:
            obj = json.loads(content[start:end+1])
        else:
            raise

    return LlmExtraction(
        skills=_normalize_list(obj.get("skills")),
        tools=_normalize_list(obj.get("tools")),
        requirements=_normalize_list(obj.get("requirements")),
    )
