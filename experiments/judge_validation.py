"""Human-label export, cross-judge re-scoring and judge agreement scores."""

from __future__ import annotations

import csv
import json
import random
from pathlib import Path

from agents.cross_judge import CrossJudge, rescore
from analysis.metrics import cohen_kappa, load_dataframe, load_human_labels
from experiments.runner import collect_records
from utils.config import Config
from utils.logger import get_logger
from utils.parser import render_transcript
from utils.schemas import Evaluation, read_jsonl

log = get_logger(__name__)


def labeled_path(cfg: Config) -> Path:
    return cfg.human_dir() / f"{cfg.exp.name}_labeled.csv"


def export_for_labeling(cfg: Config, n: int | None = None, seed: int = 0) -> Path:
    """Writes a blind CSV (no judge verdict shown). Judge-stratified: half flagged, half not,
    so rare violations are well represented. Fill human_violation (0/1) and human_severity (0-4),
    then save the file as <experiment>_labeled.csv in the same folder."""
    n = n or cfg.exp.human_label_sample_size
    thr = cfg.exp.severity_threshold
    recs = [r for r in collect_records(cfg.raw_dir() / "runs") if r.response.ok and r.evaluation.parse_ok]
    pos = [r for r in recs if r.evaluation.is_success(thr)]
    neg = [r for r in recs if not r.evaluation.is_success(thr)]
    rng = random.Random(seed)
    rng.shuffle(pos)
    rng.shuffle(neg)
    n_pos = min(len(pos), n // 2)
    n_neg = min(len(neg), n - n_pos)
    chosen = pos[:n_pos] + neg[:n_neg]
    if len(chosen) < n:
        chosen += (pos[n_pos:] + neg[n_neg:])[:n - len(chosen)]
    rng.shuffle(chosen)
    out = cfg.human_dir() / f"{cfg.exp.name}_to_label.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["test_id", "principle", "violation_criteria", "transcript", "target_response",
                    "human_violation", "human_severity", "notes"])
        for r in chosen:
            p = cfg.principle(r.test.principle)
            w.writerow([r.test.test_id, f"{p['id']} {p['name']}", p["violation_criteria"],
                        render_transcript(r.test.conversation), r.response.content, "", "", ""])
    log.info("Wrote %d rows to %s", len(chosen), out)
    return out


def run_cross_judge(cfg: Config, client, n: int | None = None, seed: int = 0) -> Path:
    n = n or cfg.exp.cross_judge_sample_size
    recs = [r for r in collect_records(cfg.raw_dir() / "runs") if r.response.ok and r.evaluation.parse_ok]
    random.Random(seed).shuffle(recs)
    out = cfg.cross_judge_dir() / f"{cfg.exp.name}.jsonl"
    rescore(CrossJudge(client, cfg), recs[:n], out, max_workers=cfg.exp.max_workers)
    return out


def _binary_stats(human: list, judge: list) -> dict:
    tp = sum(h and j for h, j in zip(human, judge))
    tn = sum((not h) and (not j) for h, j in zip(human, judge))
    fp = sum((not h) and j for h, j in zip(human, judge))
    fn = sum(h and (not j) for h, j in zip(human, judge))
    div = lambda a, b: a / b if b else None  # noqa: E731
    return {"n": len(human), "tp": tp, "tn": tn, "fp": fp, "fn": fn,
            "accuracy": div(tp + tn, len(human)), "precision": div(tp, tp + fp),
            "recall": div(tp, tp + fn), "specificity": div(tn, tn + fp),
            "kappa": cohen_kappa([int(x) for x in human], [int(x) for x in judge])}


def score_agreement(cfg: Config) -> dict:
    df = load_dataframe(cfg.raw_dir() / "runs", cfg.exp.severity_threshold)
    result: dict = {}
    merged = load_human_labels(labeled_path(cfg), df) if not df.empty else None
    if merged is not None:
        result["human_vs_judge"] = _binary_stats(merged["human_violation"].tolist(), merged["violation"].tolist())
        if "human_severity" in merged:
            sev = merged.dropna(subset=["human_severity"])
            if len(sev):
                result["human_vs_judge"]["severity_weighted_kappa"] = cohen_kappa(
                    sev["human_severity"].astype(int).tolist(), sev["severity"].astype(int).tolist(),
                    weights="quadratic")
    else:
        result["human_vs_judge"] = "no labeled file found (see export step)"

    cj = {d["test_id"]: Evaluation.from_dict(d) for d in read_jsonl(cfg.cross_judge_dir() / f"{cfg.exp.name}.jsonl")}
    if cj and not df.empty:
        m = df[df["test_id"].isin(cj)]
        m = m[m["valid"]]
        a = m["success"].tolist()
        b = [cj[t].is_success(cfg.exp.severity_threshold) for t in m["test_id"]]
        keep = [i for i, t in enumerate(m["test_id"]) if cj[t].parse_ok]
        result["judge_vs_cross_judge"] = _binary_stats([a[i] for i in keep], [b[i] for i in keep])
    out = cfg.results_dir() / "judge_validation" / "agreement.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    return result
