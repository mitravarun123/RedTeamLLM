"""Keeps the generator's context length comparable across arms."""

from __future__ import annotations


def approx_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def pad_to_length(text: str, target_chars: int, filler: list, header: str = "", sep: str = "\n\n") -> str:
    """Prepend neutral, task-irrelevant filler so len(result) == target_chars (if text is shorter)."""
    gap = target_chars - len(text) - (len(sep) if text else 0)
    if gap <= 0 or not filler:
        return text
    block = header.strip()
    i = 0
    while len(block) < gap:
        block += ("\n" if (i == 0 and block) else " ") + filler[i % len(filler)]
        i += 1
    block = block[:gap]
    return f"{block}{sep}{text}" if text else block