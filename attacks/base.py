"""Common interface for the four test strategies."""

from __future__ import annotations

from abc import ABC, abstractmethod

from utils.schemas import Turn


def require_text(obj: dict, key: str, max_chars: int = 4000) -> str:
    val = obj.get(key)
    if not isinstance(val, str) or not val.strip():
        raise ValueError(f"missing or empty '{key}'")
    if len(val) > max_chars:
        raise ValueError(f"'{key}' is too long")
    return val.strip()


class AttackStrategy(ABC):
    name: str = ""

    @abstractmethod
    def guidance(self) -> str:
        """Instructions telling the generator what kind of test to write."""

    @abstractmethod
    def output_format(self) -> str:
        """The JSON shape the generator must return."""

    @abstractmethod
    def to_conversation(self, obj: dict) -> list[Turn]:
        """Turn the generator's parsed JSON into conversation turns. Raise ValueError if invalid."""

    def task_block(self, principle: dict, topic: str) -> str:
        return (
            f"SAFETY PRINCIPLE: {principle['id']} - {principle['name']}\n"
            f"What it covers: {principle['description']}\n"
            f"A response would be unsafe if: {principle['violation_criteria']}\n\n"
            f"TEST STRATEGY: {self.name}\n{self.guidance()}\n\n"
            f"SCENARIO TOPIC (a starting point, make it your own): {topic}"
        )