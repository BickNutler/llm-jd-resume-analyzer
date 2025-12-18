from __future__ import annotations
import re
from dataclasses import dataclass
from typing import Dict

SECTION_HEADERS = [
    "summary","professional summary","about",
    "skills","technical skills",
    "experience","work experience","employment",
    "projects","project experience",
    "education","certifications"
]

_HEADER_RE = re.compile(r"^\s*(" + "|".join([re.escape(h) for h in SECTION_HEADERS]) + r")\s*$", re.I)

@dataclass(frozen=True)
class Sections:
    sections: Dict[str, str]

def parse_sections(raw_text: str) -> Sections:
    """Best-effort section parsing for plaintext resumes/JDs."""
    if not raw_text:
        return Sections(sections={})

    lines=[ln.strip() for ln in raw_text.splitlines()]
    sections={}
    current="body"
    buf=[]
    for ln in lines:
        if _HEADER_RE.match(ln.lower()):
            # flush
            if buf:
                sections[current] = "\n".join(buf).strip()
            current = ln.lower()
            buf=[]
        else:
            buf.append(ln)
    if buf:
        sections[current] = "\n".join(buf).strip()
    return Sections(sections=sections)
