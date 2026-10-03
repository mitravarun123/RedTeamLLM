"""Parsing helpers: strip reasoning tags, pull JSON out of messy model output."""

from __future__ import annotations

import json
import re
from typing import Any, Iterable, Optional

from utils.schemas import Evaluation, Turn

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def truncate(text: str, n: int) -> str:
    text = text or ""
    return text if len(text) <= n else text[: max(0, n - 3)] + "..."


def strip_think(text: str) -> str:
    """Remove <think>...</think> blocks (Qwen-style reasoning), even unterminated ones."""
    text = _THINK_RE.sub("", text or "")
    if "<think>" in text.lower():
        text = re.split(r"<think>", text, flags=re.IGNORECASE)[0]
    return text.strip()


def _strip_fences(text: str) -> str:
    m = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    return m.group(1) if m else text


def _try_load(chunk: str) -> Optional[dict]:
    for candidate in (chunk, re.sub(r",\s*([}\]])", r"\1", chunk)):
        try:
            obj = json.loads(candidate, strict=False)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            continue
    return None


def _first_json_object(s: str) -> Optional[dict]:
    start = s.find("{")
    while start != -1:
        depth, in_str, esc = 0, False, False
        for i in range(start, len(s)):
            ch = s[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
            elif ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    obj = _try_load(s[start : i + 1])
                    if obj is not None:
                        return obj
                    break
        start = s.find("{", start + 1)
    return None


def extract_json(text: str) -> Optional[dict]:
    if not text:
        return None
    text = strip_think(text)
    for candidate in (text, _strip_fences(text)):
        obj = _first_json_object(candidate)
        if obj is not None:
            return obj
    return None


def to_bool(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    if isinstance(v, str):
        return v.strip().lower() in {"true", "yes", "y", "1", "violation"}
    return False


def to_int(v: Any, default: int = 0) -> int:
    try:
        return int(round(float(v)))
    except (TypeError, ValueError):
        return default


def to_conf(v: Any) -> float:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return 0.5
    if 1.0 < f <= 100.0:
        f /= 100.0
    return max(0.0, min(1.0, f))


def parse_evaluation(text: str, test_id: str, principle: str = "", judge_model: str = "") -> Evaluation:
    obj = extract_json(text)
    if not obj or "violation" not in obj:
        return Evaluation(
            test_id=test_id, violation=False, severity=0, confidence=0.0,
            reason="PARSE_FAILURE: " + truncate(strip_think(text or ""), 200),
            principle=principle, judge_model=judge_model, parse_ok=False,
        )
    violation = to_bool(obj.get("violation"))
    severity = to_int(obj.get("severity"), 1 if violation else 0)
    severity = 0 if not violation else max(1, severity)
    return Evaluation(
        test_id=test_id, violation=violation, severity=severity,
        confidence=to_conf(obj.get("confidence")), reason=str(obj.get("reason", "")).strip(),
        principle=principle, judge_model=judge_model, parse_ok=True,
    )


def render_transcript(conversation: Iterable[Turn], max_chars: Optional[int] = None) -> str:
    lines = []
    for t in conversation:
        text = t.content if max_chars is None else truncate(t.content, max_chars)
        lines.append(f"[{t.role}] {text}")
    return "\n".join(lines)