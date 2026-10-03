"""Run the complete configured HF study, with isolated outputs and quota-stop status.

Usage: python run_pipeline.py
Rerun to resume completed slots. Human labels must be filled manually.
"""
from __future__ import annotations

import json
import threading
from dataclasses import asdict
from datetime import datetime, timezone

from dotenv import load_dotenv

from utils.config import PROJECT_ROOT, load_config
from utils.hf_client import MultiClient
from utils.logger import setup_logging


class PipelineClient(MultiClient):
    """Stop further calls when account access, quota, or a model is unavailable."""

    def __init__(self):
        super().__init__(max_attempts=2, timeout=90)
        self.fatal_error = None
        self._fatal_lock = threading.Lock()

    def chat(self, spec, messages, **kwargs):
        if self.fatal_error:
            raise RuntimeError(self.fatal_error)
        try:
            return super().chat(spec, messages, **kwargs)
        except Exception as exc:
            status = getattr(exc, "status_code", None)
            if status in (401, 402, 403, 404, 429):
                with self._fatal_lock:
                    self.fatal_error = f"HF {type(exc).__name__} (HTTP {status}) for {spec.id}"
            raise


def main():
    load_dotenv(PROJECT_ROOT / ".env")
    cfg = load_config()
    cfg = cfg.with_exp(name=cfg.exp.name + "_hf_full")
    out = cfg.results_dir()
    out.mkdir(parents=True, exist_ok=True)
    setup_logging(log_file=out / "logs" / "pipeline.log")
    for directory in (cfg.raw_dir() / "runs", cfg.raw_dir() / "refusals",
                      cfg.raw_dir() / "transfer", cfg.processed_dir(),
                      cfg.human_dir(), cfg.cross_judge_dir(), cfg.figures_dir()):
        directory.mkdir(parents=True, exist_ok=True)
    state = {"experiment": cfg.exp.name, "status": "running", "stage": "preflight",
             "expected_run_slots": len(cfg.targets) * len(cfg.exp.seeds) * len(cfg.exp.arms)
             * cfg.exp.rounds * cfg.exp.tests_per_round,
             "completed_stages": [], "human_labels": "manual labeling required"}

    def status(stage=None, **updates):
        if stage:
            state["stage"] = stage
        state.update(updates)
        state["updated_at"] = datetime.now(timezone.utc).isoformat()
        (out / "pipeline_status.json").write_text(json.dumps(state, indent=2))
        print(json.dumps(state), flush=True)

    snapshot = {"experiment": asdict(cfg.exp), "models": {k: asdict(v) for k, v in cfg.models.items()},
                "roles": cfg.roles, "principles": cfg.principles}
    snapshot_path = out / "config_snapshot.json"
    if snapshot_path.exists() and json.loads(snapshot_path.read_text()) != snapshot:
        raise RuntimeError("Config differs from the existing run; use a new experiment name.")
    snapshot_path.write_text(json.dumps(snapshot, indent=2))
    status()
    client = PipelineClient()
    from experiments.runner import run_arms, collect_records
    from experiments.transfer import run_transfer
    from experiments.judge_validation import export_for_labeling, run_cross_judge, score_agreement
    from analysis.report import analyze

    try:
        keys = list(dict.fromkeys([cfg.roles["generator"], cfg.roles["judge"],
                                  cfg.roles["cross_judge"], *cfg.targets]))
        for key in keys:
            response = client.chat(cfg.model(key), [{"role": "user", "content": "Reply with only OK. /no_think"}],
                                   max_tokens=1024, temperature=0)
            if not response.content.strip():
                raise RuntimeError(f"Preflight empty response: {key}")
            print(f"Preflight OK: {key}", flush=True)
        state["completed_stages"].append("preflight")
        status("experiments")
        run_arms(cfg, client, cfg.exp.arms, cfg.targets)
        records = collect_records(cfg.raw_dir() / "runs")
        state["records_saved"] = len(records)
        state["valid_records"] = sum(r.response.ok and r.evaluation.parse_ok for r in records)
        state["completed_stages"].append("experiments")
        status("transfer")
        for source in cfg.targets:
            run_transfer(cfg, client, source, cfg.targets)
            if client.fatal_error:
                raise RuntimeError(client.fatal_error)
        state["completed_stages"].append("transfer")
        status("validation")
        export_for_labeling(cfg)
        run_cross_judge(cfg, client)
        if client.fatal_error:
            raise RuntimeError(client.fatal_error)
        score_agreement(cfg)
        state["completed_stages"].append("validation")
        status("analysis")
        analyze(cfg)
        state["completed_stages"].append("analysis")
        complete = state["valid_records"] == state["expected_run_slots"]
        status(status="complete" if complete else "incomplete", stage="finished")
    except Exception as exc:
        records = collect_records(cfg.raw_dir() / "runs")
        status(status="interrupted", error=client.fatal_error or f"{type(exc).__name__}: {exc}",
               records_saved=len(records), valid_records=sum(r.response.ok and r.evaluation.parse_ok for r in records))
        if records:
            export_for_labeling(cfg)
            score_agreement(cfg)
            try:
                analyze(cfg)
                state["partial_analysis_saved"] = True
            except Exception as analysis_exc:
                state["analysis_error"] = f"{type(analysis_exc).__name__}: {analysis_exc}"
            status()
        raise SystemExit(1)


if __name__ == "__main__":
    main()
