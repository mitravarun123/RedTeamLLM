"""240 local tests, transfer, cross-judge, human export, analysis, then model cleanup."""
from __future__ import annotations

import json
from dataclasses import asdict, replace
from datetime import datetime, timezone

from agents.cross_judge import CrossJudge, rescore
from analysis.report import analyze
from experiments.judge_validation import export_for_labeling, score_agreement
from experiments.runner import ExperimentRunner, collect_records
from experiments.transfer import run_transfer
from prepare_local_models import MODEL_DIR, file_hash
from utils.config import PROJECT_ROOT, ModelSpec, load_config, load_yaml
from utils.local_client import LocalClient
from utils.logger import setup_logging
from utils.schemas import Evaluation, read_jsonl


def local_config(data_root=None):
    cfg = load_config(data_root=data_root)
    models = load_yaml(PROJECT_ROOT / "config" / "local_models.yaml")
    cfg = replace(cfg, models={k: ModelSpec(key=k, **v) for k, v in models["registry"].items()}, roles=models["roles"])
    return cfg.with_exp(name="adaptive_redteam_local_240", rounds=2, tests_per_round=10, seeds=[0],
                        arms=list("ABCDEF"), max_workers=1, max_feedback_items=3,
                        max_response_chars=300, max_transcript_chars=600,
                        generator_max_tokens=1024, target_max_tokens=256, judge_max_tokens=256,
                        human_label_sample_size=240, cross_judge_sample_size=240), models["weights"]


def cleanup_weights(model_dir, weights):
    """Delete only our known files, after validating ownership by size and hash."""
    expected = []
    for entry in weights.values():
        path = model_dir / entry["filename"]
        if path.is_symlink() or path.parent.resolve() != model_dir.resolve():
            raise RuntimeError(f"Unsafe model path: {path}")
        if path.exists():
            if path.stat().st_size != entry["size"] or file_hash(path) != entry["sha256"]:
                raise RuntimeError(f"Model file changed; preserving it: {path}")
            expected.append(path)
    reclaimed = sum(path.stat().st_size for path in expected)
    for path in expected:
        path.unlink()
    return reclaimed


def main():
    cfg, weights = local_config()
    out = cfg.results_dir()
    out.mkdir(parents=True, exist_ok=True)
    setup_logging(log_file=out / "logs" / "pipeline.log")
    state_path = out / "pipeline_status.json"
    if state_path.exists():
        previous = json.loads(state_path.read_text())
        if previous.get("status") == "complete" and previous.get("weights_deleted"):
            print("Local pipeline already completed; model weights already deleted.")
            return
    state = {"experiment": cfg.exp.name, "status": "running", "stage": "starting", "expected_run_slots": 240,
             "completed_stages": [], "human_validation": "pending manual review", "weights_deleted": False}

    def status(stage=None, **updates):
        if stage:
            state["stage"] = stage
        state.update(updates)
        state["updated_at"] = datetime.now(timezone.utc).isoformat()
        temporary = state_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(state, indent=2))
        temporary.replace(state_path)
        print(json.dumps(state), flush=True)

    snapshot = {"experiment": asdict(cfg.exp), "models": {k: asdict(v) for k, v in cfg.models.items()},
                "roles": cfg.roles, "weights": weights, "runtime": "llama.cpp b4514; Metal; context 8192; threads 4",
                "structured_outputs": "grammar constrained generator and judges; unconstrained targets"}
    snapshot_path = out / "config_snapshot.json"
    if snapshot_path.exists() and json.loads(snapshot_path.read_text()) != snapshot:
        raise RuntimeError("Configuration changed; use a new experiment directory")
    snapshot_path.write_text(json.dumps(snapshot, indent=2))
    for directory in (cfg.raw_dir() / "runs", cfg.raw_dir() / "refusals", cfg.raw_dir() / "transfer",
                      cfg.processed_dir(), cfg.human_dir(), cfg.cross_judge_dir(), cfg.figures_dir()):
        directory.mkdir(parents=True, exist_ok=True)
    status("verify_weights")
    client = LocalClient(cfg, weights, MODEL_DIR, out)
    success = False
    try:
        for entry in weights.values():
            path = MODEL_DIR / entry["filename"]
            if not path.is_file() or path.stat().st_size != entry["size"] or file_hash(path) != entry["sha256"]:
                raise RuntimeError("Missing or invalid weights; run prepare_local_models.py")
        status("start_local_models")
        client.start()
        state["completed_stages"].append("local_models")
        for target in cfg.targets:
            client.set_active([cfg.roles["generator"], cfg.roles["judge"], target])
            runner = ExperimentRunner(cfg, client, target)
            for arm in cfg.exp.arms:
                status("experiments", current_target=target, current_arm=arm)
                runner.run(arm, 0)
                records = collect_records(cfg.raw_dir() / "runs")
                status(records_saved=len(records), valid_records=sum(r.response.ok and r.evaluation.parse_ok for r in records))
        records = collect_records(cfg.raw_dir() / "runs")
        # A bounded second pass repairs failed slots or judgments, without duplicating valid slots.
        if len(records) != 240 or any(not r.evaluation.parse_ok for r in records):
            status("repair_failed_slots")
            for target in cfg.targets:
                client.set_active([cfg.roles["generator"], cfg.roles["judge"], target])
                runner = ExperimentRunner(cfg, client, target)
                for arm in cfg.exp.arms:
                    runner.run(arm, 0)
            records = collect_records(cfg.raw_dir() / "runs")
        state["records_saved"] = len(records)
        state["valid_records"] = sum(r.response.ok and r.evaluation.parse_ok for r in records)
        state["completed_stages"].append("experiments")
        status("transfer")
        for source in cfg.targets:
            for destination in cfg.targets:
                client.set_active([cfg.roles["judge"], destination])
                run_transfer(cfg, client, source, [destination])
        transferred = collect_records(cfg.raw_dir() / "transfer")
        state["expected_transfer_records"] = len(cfg.targets) * sum(
            r.test.arm in ("A", "E") and r.response.ok and r.evaluation.is_success(cfg.exp.severity_threshold)
            for r in records)
        state["transfer_records"] = len(transferred)
        state["transfer_valid"] = sum(r.response.ok and r.evaluation.parse_ok for r in transferred)
        state["completed_stages"].append("transfer")
        status("human_label_export")
        export_for_labeling(cfg)
        state["completed_stages"].append("human_label_export")
        status("cross_judge")
        client.set_active([cfg.roles["cross_judge"]])
        cross_path = cfg.cross_judge_dir() / f"{cfg.exp.name}.jsonl"
        rescore(CrossJudge(client, cfg), records, cross_path, max_workers=1)
        cross = {d["test_id"]: Evaluation.from_dict(d) for d in read_jsonl(cross_path)}
        state["cross_judge_records"] = len(cross)
        state["cross_judge_valid"] = sum(e.parse_ok for e in cross.values())
        score_agreement(cfg)
        state["completed_stages"].append("cross_judge")
        status("analysis")
        analyze(cfg)
        state["completed_stages"].append("analysis")
        required = [out / "summary.json", cfg.processed_dir() / f"{cfg.exp.name}_rows.csv",
                    cfg.human_dir() / f"{cfg.exp.name}_to_label.csv", out / "ablations" / "asr_by_arm.csv"]
        from PIL import Image
        images = list(cfg.figures_dir().rglob("*.png"))
        for path in images:
            with Image.open(path) as image:
                image.verify()
        state["figures_verified"] = len(images)
        success = (state["valid_records"] == 240 and state["cross_judge_valid"] == 240
                   and state["expected_transfer_records"] == state["transfer_records"] == state["transfer_valid"]
                   and len(images) >= 10 and all(p.is_file() for p in required))
        status("verify_outputs", automated_stages_successful=success)
    except Exception as exc:
        records = collect_records(cfg.raw_dir() / "runs")
        status(status="interrupted", error=f"{type(exc).__name__}: {exc}", records_saved=len(records),
               valid_records=sum(r.response.ok and r.evaluation.parse_ok for r in records))
        if records:
            export_for_labeling(cfg)
        raise
    finally:
        client.close()
    if success:
        status("cleanup_weights")
        reclaimed = cleanup_weights(MODEL_DIR, weights)
        status("finished", status="complete", weights_deleted=True, reclaimed_bytes=reclaimed)
    else:
        status("finished", status="incomplete", weights_deleted=False,
               note="Weights preserved for resume; inspect failed judgments and missing slots")


if __name__ == "__main__":
    main()
