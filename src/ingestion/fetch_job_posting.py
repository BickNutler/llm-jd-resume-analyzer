from __future__ import annotations
import httpx
from bs4 import BeautifulSoup

DEFAULT_HEADERS = {
    "User-Agent": "jd-resume-analyzer/1.0 (+portfolio project)"
}

def fetch_job_text(job_url: str, timeout_s: float = 20.0) -> str:
    """Fetch a job posting from a URL and return best-effort visible text.

    This is intentionally simple (portfolio-grade). In production, you'd add:
    - robots.txt compliance
    - stronger boilerplate removal
    - retries/backoff and caching
    """
    if not job_url:
        raise ValueError("job_url is required")

    with httpx.Client(headers=DEFAULT_HEADERS, timeout=timeout_s, follow_redirects=True) as client:
        resp = client.get(job_url)
        resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "lxml")
    # remove scripts/styles
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    text = soup.get_text("\n")
    # collapse excessive blank lines
    lines = [ln.strip() for ln in text.splitlines()]
    lines = [ln for ln in lines if ln]
    return "\n".join(lines)
