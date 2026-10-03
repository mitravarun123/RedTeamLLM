"""A second judge from a different model, used to check the primary judge."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from agents.evaluator import SafetyEvaluator
from utils.config import Config
from utils.logger import get_logger
from utils.schemas import append_jsonl, read_jsonl

log = get_logger(__name__)


class CrossJudge(SafetyEvaluator):
    def __init__(self, client, cfg: Config):
        super().__init__(client, cfg, role="cross_judge")


def rescore(cross_judge: CrossJudge, records: list, out_path: Path, max_workers: int = 4) -> int:
    """Re-score records with the cross-judge. Resumable: already scored test_ids are skipped."""
    done = {d["test_id"] for d in read_jsonl(out_path)}
    todo = [r for r in records if r.test.test_id not in done and r.response.ok]
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(cross_judge.evaluate, r.test, r.response) for r in todo]
        for fut in as_completed(futures):
            append_jsonl(out_path, fut.result().to_dict())
    log.info("Cross-judge scored %d new records", len(todo))
    return len(todo)