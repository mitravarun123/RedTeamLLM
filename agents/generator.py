"""Agent 1: the red-team test generator."""

from __future__ import annotations

from typing import Optional

from attacks import get_strategy
from utils.config import Config
from utils.parser import extract_json, strip_think
from utils.refusal_detector import is_refusal
from utils.schemas import TestCase
from utils.seeding import derive_seed


def is_valid_test(test: TestCase) -> bool:
    return bool(test.conversation) and not test.generator_refused


class RedTeamGenerator:
    def __init__(self, client, cfg: Config):
        self.client = client
        self.cfg = cfg
        self.spec = cfg.model(cfg.roles["generator"])
        self.system_prompt = cfg.prompts["generator_system"].strip()

    def build_user_prompt(self, strategy, principle: dict, topic: str, feedback_context: str) -> str:
        blocks = [strategy.task_block(principle, topic)]
        if feedback_context.strip():
            blocks.append(feedback_context.strip())
        blocks.append("OUTPUT FORMAT (JSON only, no other text):\n" + strategy.output_format())
        return "\n\n".join(blocks)

    def generate(self, *, test_id: str, experiment_id: str, arm: str, seed: int, round_id: int,
                 category: str, principle_id: str, topic: str, feedback_context: str = "",
                 parent_id: Optional[str] = None, attempt: int = 0) -> TestCase:
        exp = self.cfg.exp
        strategy = get_strategy(category, n_turns=exp.escalation_turns)
        principle = self.cfg.principle(principle_id)
        test = TestCase(
            test_id=test_id, experiment_id=experiment_id, arm=arm, seed=seed, round_id=round_id,
            category=category, principle=principle_id, conversation=[], parent_id=parent_id,
            generator_model=self.spec.id,
        )
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": self.build_user_prompt(strategy, principle, topic, feedback_context)},
        ]
        try:
            res = self.client.chat(
                self.spec, messages,
                temperature=min(1.2, exp.generator_temperature + 0.1 * attempt),
                max_tokens=exp.generator_max_tokens,
                seed=derive_seed(experiment_id, arm, seed, test_id, attempt),
            )
        except Exception as exc:  # noqa: BLE001
            test.raw_generator_output = f"API_ERROR: {type(exc).__name__}: {exc}"
            return test

        raw = strip_think(res.content)
        test.raw_generator_output = raw
        obj = extract_json(raw)
        if obj is None:
            test.generator_refused = is_refusal(raw)
            return test
        try:
            test.conversation = strategy.to_conversation(obj)
        except ValueError as exc:
            test.raw_generator_output = f"{raw}\n[PARSE_ERROR: {exc}]"
        return test