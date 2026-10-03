"""Ablation arms, feedback selection and the shuffled-feedback control."""

from __future__ import annotations

from dataclasses import dataclass

from utils.schemas import FeedbackItem


@dataclass(frozen=True)
class ArmSpec:
    arm: str
    adaptive: bool
    show_responses: bool
    show_scores: bool
    show_explanations: bool
    shuffle: bool = False


# A: no feedback.  B: raw transcript (test + target reply), no judge signal.
# C: B + judge scores.  D: B + judge explanation.  E: everything.
# F: same format and length as E, but judge scores/explanations are reassigned to other attempts.
ARM_SPECS = {
    "A": ArmSpec("A", False, False, False, False),
    "B": ArmSpec("B", True, True, False, False),
    "C": ArmSpec("C", True, True, True, False),
    "D": ArmSpec("D", True, True, False, True),
    "E": ArmSpec("E", True, True, True, True),
    "F": ArmSpec("F", True, True, True, True, shuffle=True),
}


def get_arm_spec(arm: str) -> ArmSpec:
    if arm not in ARM_SPECS:
        raise KeyError(f"Unknown arm '{arm}'")
    return ARM_SPECS[arm]


def select_feedback(history: list, category: str, principle: str, k: int, rng, threshold: int = 1) -> list:
    """Pick up to k past attempts: same category, same principle first, a mix of successes and failures."""
    pool = [h for h in history if h.test.category == category]
    if not pool or k <= 0:
        return []
    succ = [h for h in pool if h.evaluation.is_success(threshold)]
    fail = [h for h in pool if not h.evaluation.is_success(threshold)]

    def same(h):
        return h.test.principle == principle

    succ.sort(key=lambda h: (same(h), h.evaluation.severity, h.test.round_id, rng.random()), reverse=True)
    fail.sort(key=lambda h: (same(h), h.test.round_id, rng.random()), reverse=True)
    n_s = min(len(succ), (k + 1) // 2)
    chosen = succ[:n_s] + fail[: k - n_s]
    if len(chosen) < k:
        chosen += succ[n_s : n_s + (k - len(chosen))]
    chosen.sort(key=lambda h: h.test.round_id)
    return chosen


def shuffle_feedback(items: list, rng) -> list:
    """Reassign judge evaluations to different attempts so the signal no longer matches the test."""
    n = len(items)
    if n < 2:
        return list(items)
    perm = list(range(n))
    for _ in range(50):
        rng.shuffle(perm)
        if all(perm[i] != i for i in range(n)):
            break
    else:
        perm = [(i + 1) % n for i in range(n)]
    return [FeedbackItem(test=items[i].test, response=items[i].response, evaluation=items[perm[i]].evaluation)
            for i in range(n)]