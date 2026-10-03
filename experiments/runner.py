"""Shared experiment loop: slot -> feedback -> generate -> target -> judge -> save. Resumable."""

from __future__ import annotations

import threading
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable, Optional

from agents.evaluator import SafetyEvaluator
from agents.generator import RedTeamGenerator, is_valid_test
from agents.target import TargetAgent
from feedback.builder import FeedbackBuilder
from utils.config import Config
from utils.logger import get_logger
from utils.schemas import FeedbackItem, Record, append_jsonl, load_records
from utils.seeding import make_rng

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None

log = get_logger(__name__)


class _NoBar:
    def update(self, n: int = 1) -> None: ...
    def close(self) -> None: ...


def _bar(total: int, desc: str):
    return tqdm(total=total, desc=desc, leave=False) if tqdm else _NoBar()


def collect_records(directory: Path) -> list:
    out: list = []
    for p in sorted(Path(directory).glob("*.jsonl")):
        out.extend(load_records(p))
    return out


class ExperimentRunner:
    def __init__(self, cfg: Config, client, target_key: str):
        self.cfg, self.client, self.target_key = cfg, client, target_key
        self.target_spec = cfg.model(target_key)
        self.generator = RedTeamGenerator(client, cfg)
        self.target = TargetAgent(client, self.target_spec, cfg)
        self.judge = SafetyEvaluator(client, cfg, role="judge")
        self.feedback = FeedbackBuilder(cfg)
        self.runs_dir = cfg.raw_dir() / "runs"
        self.refusals_dir = cfg.raw_dir() / "refusals"
        self._lock = threading.Lock()

    # ---- deterministic schedule -------------------------------------------------
    def slot(self, seed: int, j: int) -> tuple[str, str]:
        """Same schedule every round and every arm for a given seed (so rounds and arms are comparable).
        Principles are spread evenly across the round; the category offset shifts with the seed so
        that across seeds each principle meets every category."""
        cats, pids = self.cfg.exp.categories, self.cfg.principle_ids
        t = self.cfg.exp.tests_per_round
        return cats[(j + seed) % len(cats)], pids[(j * len(pids) // t) % len(pids)]

    def topic_for(self, seed: int, round_id: int, j: int, principle_id: str) -> str:
        topics = self.cfg.topics.get(principle_id) or [self.cfg.principle(principle_id)["description"]]
        rng = make_rng(self.cfg.exp.name, "topic", seed, round_id, j)
        return topics[rng.randrange(len(topics))]

    def test_id(self, arm: str, seed: int, round_id: int, j: int) -> str:
        return f"{arm}_s{seed}_{self.target_key}_r{round_id:02d}_{j:03d}"

    def run_path(self, arm: str, seed: int) -> Path:
        return self.runs_dir / f"{arm}_s{seed}_{self.target_key}.jsonl"

    def refusal_path(self, arm: str, seed: int) -> Path:
        return self.refusals_dir / f"{arm}_s{seed}_{self.target_key}.jsonl"

    # ---- main loop ---------------------------------------------------------------
    def run(self, arm: str, seed: int) -> list:
        exp = self.cfg.exp
        path = self.run_path(arm, seed)
        records = load_records(path)
        # Keep the existing target response and retry failed judgments on resume.
        pending_judgments = [r for r in records if r.response.ok and not r.evaluation.parse_ok]
        for rec in pending_judgments:
            rec.evaluation = self.judge.evaluate(rec.test, rec.response)
            temporary = path.with_suffix(".jsonl.tmp")
            temporary.write_text("".join(json.dumps(r.to_dict(), ensure_ascii=False) + "\n"
                                         for r in records), encoding="utf-8")
            temporary.replace(path)
            if getattr(self.client, "fatal_error", None):
                raise RuntimeError(self.client.fatal_error)
        done = {r.test.test_id for r in records}
        for round_id in range(1, exp.rounds + 1):
            todo = [j for j in range(exp.tests_per_round) if self.test_id(arm, seed, round_id, j) not in done]
            if not todo:
                continue
            history = [FeedbackItem(r.test, r.response, r.evaluation) for r in records
                       if r.test.round_id < round_id and r.response.ok and r.evaluation.parse_ok]
            bar = _bar(len(todo), f"{arm} s{seed} {self.target_key} r{round_id}")
            with ThreadPoolExecutor(max_workers=exp.max_workers) as ex:
                futures = [ex.submit(self._run_slot, arm, seed, round_id, j, history) for j in todo]
                for fut in as_completed(futures):
                    bar.update(1)
                    try:
                        rec = fut.result()
                    except Exception:  # noqa: BLE001
                        log.exception("Slot crashed")
                        continue
                    if rec is None:
                        continue
                    if not exp.save_raw_responses:
                        rec.response.content = "[not saved]"
                        rec.test.raw_generator_output = ""
                    append_jsonl(path, rec.to_dict())
                    records.append(rec)
            bar.close()
            if getattr(self.client, "fatal_error", None):
                raise RuntimeError(self.client.fatal_error)
            ok = [r for r in records if r.test.round_id == round_id and r.evaluation.parse_ok and r.response.ok]
            asr = sum(r.evaluation.is_success(exp.severity_threshold) for r in ok) / max(1, len(ok))
            log.info("%s seed=%s target=%s round=%d: %d tests, ASR=%.2f", arm, seed, self.target_key,
                     round_id, len(ok), asr)
        return records

    def _run_slot(self, arm: str, seed: int, round_id: int, j: int, history: list) -> Optional[Record]:
        exp = self.cfg.exp
        category, principle_id = self.slot(seed, j)
        topic = self.topic_for(seed, round_id, j, principle_id)
        test_id = self.test_id(arm, seed, round_id, j)
        rng = make_rng(exp.name, "feedback", arm, seed, round_id, j)
        context, parent_id = self.feedback.build(arm, history, category, principle_id, rng)

        test = None
        for attempt in range(exp.max_generation_attempts):
            cand = self.generator.generate(
                test_id=test_id, experiment_id=exp.name, arm=arm, seed=seed, round_id=round_id,
                category=category, principle_id=principle_id, topic=topic,
                feedback_context=context, parent_id=parent_id, attempt=attempt,
            )
            if is_valid_test(cand):
                test = cand
                break
            kind = ("refusal" if cand.generator_refused
                    else "api_error" if cand.raw_generator_output.startswith("API_ERROR") else "parse_failure")
            self._log_failure(arm, seed, cand, kind, attempt)
        if test is None:
            self._log_failure(arm, seed, cand, "slot_unfilled", exp.max_generation_attempts)
            return None

        response = self.target.respond(test)
        if not response.ok:
            self._log_failure(arm, seed, test, "slot_unfilled", -1, note=f"target_error: {response.error}")
            return None
        evaluation = self.judge.evaluate(test, response)
        return Record(test=test, response=response, evaluation=evaluation,
                      target_model=self.target_spec.id, evaluator_model=self.judge.spec.id)

    def _log_failure(self, arm: str, seed: int, test, kind: str, attempt: int, note: str = "") -> None:
        row = {
            "test_id": test.test_id, "arm": arm, "seed": seed, "round": test.round_id,
            "category": test.category, "principle": test.principle, "generator_model": test.generator_model,
            "kind": kind, "attempt": attempt, "note": note, "raw_output": test.raw_generator_output[:2000],
        }
        with self._lock:
            append_jsonl(self.refusal_path(arm, seed), row)


def run_arms(cfg: Config, client, arms: Iterable[str], target_keys: Iterable[str]) -> None:
    for tk in target_keys:
        runner = ExperimentRunner(cfg, client, tk)
        for seed in cfg.exp.seeds:
            for arm in arms:
                runner.run(arm, seed)
