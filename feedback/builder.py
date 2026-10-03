"""Builds the in-context feedback block the generator sees, per ablation arm."""

from __future__ import annotations

from typing import Optional

from feedback.arms import ARM_SPECS, ArmSpec, get_arm_spec, select_feedback, shuffle_feedback
from feedback.length_matching import pad_to_length
from utils.config import Config
from utils.parser import render_transcript, truncate


class FeedbackBuilder:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.t = cfg.templates

    def render(self, items: list, spec: ArmSpec) -> str:
        if not spec.adaptive or not items:
            return ""
        exp = self.cfg.exp
        parts = [self.t["header"].rstrip()]
        for n, it in enumerate(items, 1):
            block = self.t["item_test"].format(
                n=n, round=it.test.round_id,
                transcript=render_transcript(it.test.conversation, exp.max_transcript_chars),
            )
            if spec.show_responses:
                block += self.t["item_response"].format(response=truncate(it.response.content, exp.max_response_chars))
            if spec.show_scores:
                e = it.evaluation
                block += self.t["item_scores"].format(violation=str(e.violation).lower(),
                                                      severity=e.severity, confidence=e.confidence)
            if spec.show_explanations:
                block += self.t["item_explanation"].format(reason=truncate(it.evaluation.reason, 400))
            parts.append(block.rstrip())
        parts.append(self.t["footer"].rstrip())
        return "\n".join(parts)

    def build(self, arm: str, history: list, category: str, principle: str, rng) -> tuple[str, Optional[str]]:
        """Returns (context_text, parent_test_id)."""
        exp = self.cfg.exp
        spec = get_arm_spec(arm)
        items = select_feedback(history, category, principle, exp.max_feedback_items, rng, exp.severity_threshold)
        if not items:
            return "", None
        full_len = len(self.render(items, ARM_SPECS["E"]))
        shown = shuffle_feedback(items, rng) if spec.shuffle else items
        text = self.render(shown, spec)
        if exp.match_prompt_length:
            text = pad_to_length(text, full_len, self.t["neutral_filler"], self.t["filler_header"])
        parent = None
        if spec.adaptive:
            best = max(items, key=lambda it: (it.evaluation.is_success(exp.severity_threshold),
                                              it.evaluation.severity, it.test.round_id))
            parent = best.test.test_id
        return text, parent