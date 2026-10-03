"""Replay tests that succeeded on one model against other models."""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace

from agents.evaluator import SafetyEvaluator
from agents.target import TargetAgent
from utils.config import Config
from utils.logger import get_logger
from utils.schemas import Record, Turn, append_jsonl, load_records

log = get_logger(__name__)


def run_transfer(cfg: Config, client, source_key: str, dest_keys: list, source_arms=("A", "E")) -> None:
    runs_dir = cfg.raw_dir() / "runs"
    thr = cfg.exp.severity_threshold
    tests = []
    for p in sorted(runs_dir.glob("*.jsonl")):
        arm, _seed, target = p.stem.split("_", 2)
        if target != source_key or arm not in source_arms:
            continue
        tests += [r.test for r in load_records(p) if r.response.ok and r.evaluation.is_success(thr)]
    if not tests:
        log.warning("No successful tests found on %s, skipping", source_key)
        return
    log.info("Replaying %d successful tests from %s", len(tests), source_key)
    judge = SafetyEvaluator(client, cfg, role="judge")

    for dest in dest_keys:
        spec = cfg.model(dest)
        agent = TargetAgent(client, spec, cfg)
        out = cfg.raw_dir() / "transfer" / f"{source_key}__to__{dest}.jsonl"
        done = {r.test.parent_id for r in load_records(out)}
        todo = [t for t in tests if t.test_id not in done]

        def work(t, agent=agent, spec=spec, dest=dest):
            replay = replace(
                t, test_id=f"{t.test_id}__to__{dest}", parent_id=t.test_id, created_at=time.time(),
                conversation=[Turn(x.role, x.content) for x in t.conversation if x.role in ("system", "user")],
            )
            resp = agent.respond(replay)
            if not resp.ok:
                return None
            return Record(replay, resp, judge.evaluate(replay, resp), spec.id, judge.spec.id)

        with ThreadPoolExecutor(max_workers=cfg.exp.max_workers) as ex:
            for fut in as_completed([ex.submit(work, t) for t in todo]):
                rec = fut.result()
                if rec is not None:
                    append_jsonl(out, rec.to_dict())
        log.info("Transfer %s -> %s done (%d new)", source_key, dest, len(todo))