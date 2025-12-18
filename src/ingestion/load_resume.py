from __future__ import annotations
from pathlib import Path
from PyPDF2 import PdfReader

def load_text_from_file(path: str) -> str:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)

    if p.suffix.lower() == ".pdf":
        reader = PdfReader(str(p))
        parts=[]
        for page in reader.pages:
            parts.append(page.extract_text() or "")
        return "\n".join(parts)

    # txt / md fallback
    return p.read_text(encoding="utf-8", errors="ignore")
