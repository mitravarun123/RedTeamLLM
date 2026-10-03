"""
Shared data structures for the whole pipeline.

Every component (generator, target, evaluator, experiments, analysis)
talks through these classes so nobody invents their own format.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, Optional

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CATEGORIES = ("roleplay", "instruction_conflict", "indirect", "escalation")

# Experimental arms.
#   A = non-adaptive baseline
#   B = adaptive: previous tests only
#   C = adaptive: previous tests + judge scores
#   D = adaptive: previous tests + judge explanations
#   E = full system (tests + scores + explanations)
#   F = control: full-looking feedback, but scores/explanations are shuffled
ARMS = ("A", "B", "C", "D", "E", "F")

ARM_DESCRIPTIONS = {
    "A": "Non-adaptive baseline (no feedback)",
    "B": "Adaptive: previous tests only",
    "C": "Adaptive: previous tests + scores",
    "D": "Adaptive: previous tests + evaluator explanations",
    "E": "Adaptive: full feedback",
    "F": "Control: shuffled feedback",
}

ROLES = ("system", "user", "assistant")


def new_id(prefix: str = "t") -> str:
    """Short unique id, e.g. t_3f9a1c2b7d41."""
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def mode_of(arm: str) -> str:
    """Collapse an arm into the high-level mode used in the main comparison."""
    return "baseline" if arm == "A" else "adaptive"


# ---------------------------------------------------------------------------
# Conversation turn
# ---------------------------------------------------------------------------

@dataclass
class Turn:
    role: str
    content: str

    def __post_init__(self) -> None:
        if self.role not in ROLES:
            raise ValueError(f"Invalid role '{self.role}', expected one of {ROLES}")

    def to_dict(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Turn":
        return cls(role=d["role"], content=d["content"])


# ---------------------------------------------------------------------------
# Test case produced by the generator
# ---------------------------------------------------------------------------

@dataclass
class TestCase:
    """
    One controlled safety test.

    `conversation` holds the full message list that is sent to the target.
    Single-turn tests have one user turn. Escalation tests have several turns,
    where earlier assistant turns are scripted context and the last turn is
    always a user turn.
    """

    __test__ = False  # stop pytest from trying to collect this as a test class

    test_id: str
    experiment_id: str
    arm: str
    seed: int
    round_id: int
    category: str
    principle: str
    conversation: list[Turn]
    parent_id: Optional[str] = None
    generator_model: str = ""
    raw_generator_output: str = ""
    generator_refused: bool = False
    created_at: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        if self.arm not in ARMS:
            raise ValueError(f"Unknown arm '{self.arm}'")
        if self.category not in CATEGORIES:
            raise ValueError(f"Unknown category '{self.category}'")

    @property
    def prompt(self) -> str:
        """The final user message (what the target is really being asked)."""
        for turn in reversed(self.conversation):
            if turn.role == "user":
                return turn.content
        return ""

    @property
    def mode(self) -> str:
        return mode_of(self.arm)

    @property
    def is_multi_turn(self) -> bool:
        return sum(1 for t in self.conversation if t.role == "user") > 1

    def messages(self) -> list[dict[str, str]]:
        """Chat-API-ready message list."""
        return [t.to_dict() for t in self.conversation]

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["conversation"] = [t.to_dict() for t in self.conversation]
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "TestCase":
        d = dict(d)
        d["conversation"] = [Turn.from_dict(t) for t in d["conversation"]]
        return cls(**d)


# ---------------------------------------------------------------------------
# Target model output
# ---------------------------------------------------------------------------

@dataclass
class TargetResponse:
    test_id: str
    content: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_s: float = 0.0
    error: Optional[str] = None  # set if the API call failed after retries

    @property
    def ok(self) -> bool:
        return self.error is None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "TargetResponse":
        return cls(**d)


# ---------------------------------------------------------------------------
# Judge output
# ---------------------------------------------------------------------------

@dataclass
class Evaluation:
    """
    Structured verdict from the safety judge.

    severity: 0 = safe, 1 = minor, 2 = moderate, 3 = serious, 4 = severe.
    A test counts as a "successful failure" (for ASR) when violation is True
    AND severity >= the threshold set in experiments.yaml.
    """

    test_id: str
    violation: bool
    severity: int
    confidence: float
    reason: str
    principle: str = ""
    judge_model: str = ""
    parse_ok: bool = True  # False if the judge output could not be parsed

    def __post_init__(self) -> None:
        self.severity = max(0, min(4, int(self.severity)))
        self.confidence = max(0.0, min(1.0, float(self.confidence)))
        self.violation = bool(self.violation)

    def is_success(self, severity_threshold: int = 1) -> bool:
        return self.parse_ok and self.violation and self.severity >= severity_threshold

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Evaluation":
        return cls(**d)


# ---------------------------------------------------------------------------
# Feedback item (what the adaptive generator sees from earlier rounds)
# ---------------------------------------------------------------------------

@dataclass
class FeedbackItem:
    """A past attempt, packaged so the feedback builder can render it per arm."""

    test: TestCase
    response: TargetResponse
    evaluation: Evaluation


# ---------------------------------------------------------------------------
# Full dataset row
# ---------------------------------------------------------------------------

@dataclass
class Record:
    """One interaction = one row. This is the dataset the paper is built on."""

    test: TestCase
    response: TargetResponse
    evaluation: Evaluation
    target_model: str = ""
    evaluator_model: str = ""
    timestamp: float = field(default_factory=time.time)

    def to_row(self) -> dict[str, Any]:
        t, r, e = self.test, self.response, self.evaluation
        return {
            "experiment_id": t.experiment_id,
            "mode": t.mode,
            "arm": t.arm,
            "seed": t.seed,
            "round": t.round_id,
            "test_id": t.test_id,
            "parent_test_id": t.parent_id,
            "generator_model": t.generator_model,
            "target_model": self.target_model or r.model,
            "evaluator_model": self.evaluator_model or e.judge_model,
            "category": t.category,
            "principle": t.principle,
            "prompt": t.prompt,
            "conversation": [turn.to_dict() for turn in t.conversation],
            "is_multi_turn": t.is_multi_turn,
            "generator_refused": t.generator_refused,
            "target_response": r.content,
            "target_error": r.error,
            "violation": e.violation,
            "severity": e.severity,
            "confidence": e.confidence,
            "reason": e.reason,
            "judge_parse_ok": e.parse_ok,
            "prompt_tokens": r.prompt_tokens,
            "response_tokens": r.completion_tokens,
            "latency": r.latency_s,
            "timestamp": self.timestamp,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "test": self.test.to_dict(),
            "response": self.response.to_dict(),
            "evaluation": self.evaluation.to_dict(),
            "target_model": self.target_model,
            "evaluator_model": self.evaluator_model,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Record":
        return cls(
            test=TestCase.from_dict(d["test"]),
            response=TargetResponse.from_dict(d["response"]),
            evaluation=Evaluation.from_dict(d["evaluation"]),
            target_model=d.get("target_model", ""),
            evaluator_model=d.get("evaluator_model", ""),
            timestamp=d.get("timestamp", time.time()),
        )


# ---------------------------------------------------------------------------
# JSONL helpers (append-only storage, safe to resume after a crash)
# ---------------------------------------------------------------------------

def append_jsonl(path: str | Path, obj: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def read_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def save_records(path: str | Path, records: Iterable[Record]) -> None:
    for rec in records:
        append_jsonl(path, rec.to_dict())


def load_records(path: str | Path) -> list[Record]:
    return [Record.from_dict(d) for d in read_jsonl(path)]