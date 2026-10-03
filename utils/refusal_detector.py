"""Detects when the generator refused to write a test (instead of returning one)."""

from __future__ import annotations

import re

_PATTERNS = [
    r"\bI (?:can(?:no|')t|cannot|won't|will not|am unable to|'m unable to|am not able to|'m not able to)\b",
    r"\bI(?:'m| am) sorry,? but\b",
    r"\bI must (?:decline|refuse)\b",
    r"\b(?:can(?:no|')t|unable to) (?:help|assist|comply|fulfil+|provide|create|generate|write)\b",
    r"\bagainst my (?:guidelines|policy|policies|programming)\b",
    r"\bnot something I can\b",
]
_COMPILED = [re.compile(p, re.IGNORECASE) for p in _PATTERNS]


def is_refusal(text: str) -> bool:
    """True if the text looks like a refusal. Output that starts like JSON is never a refusal,
    because test prompts themselves can legitimately contain words like "I can't"."""
    if not text:
        return False
    t = text.strip()
    if t.startswith("{") or t.startswith("```"):
        return False
    return any(p.search(t[:300]) for p in _COMPILED)