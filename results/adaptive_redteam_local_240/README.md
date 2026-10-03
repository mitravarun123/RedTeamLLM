# Local 240-test pilot

This pilot uses real local inference, not mocked responses. It schedules all six arms, two targets, one seed, two rounds, and ten tests per round (240 discovery tests). Transfer replays and 240 cross-judge judgments are additional interactions.

Generator and primary judge: Qwen2.5-1.5B-Instruct (Q4_K_M).
Targets: Qwen2.5-0.5B-Instruct (Q4_K_M), SmolLM2-360M-Instruct (Q8_0).
Cross-judge: SmolLM2-360M-Instruct. The cross-judge is also a target and is very small; independent human validation is required. The generator and primary judge share one model.

Runtime: locally compiled llama.cpp b4514 for macOS Ventura, Metal acceleration, context 8192, one worker. At most two models are loaded during discovery and transfer; only the cross-judge is loaded during cross-judging. The run switched from three resident models to this memory-saving strategy after discovery completed; `runtime_adjustments.json` records the preserved checkpoints. Generator and judge outputs use schema-constrained JSON. Target outputs are unconstrained. Small models and one seed make this a pilot, not strong evidence of general model safety.

## Results

The automated run completed: 240 unique discovery cases (40 per arm, 120 per target), 132 transfer replays, 240 cross-judge ratings, and 14 verified plots. All three downloaded weight files were deleted, reclaiming 1,995,125,760 bytes. Human validation remains pending.

The two automated judges agree on the binary verdict for only 25% of cases (Cohen's kappa 0.026). These are agreement statistics, not accuracy against human truth. Treat the current success-rate plots as provisional until manual review.

Paths are relative to the project root:

- `data/raw/adaptive_redteam_local_240/runs/`: discovery interactions, by arm/seed/target.
- `data/raw/adaptive_redteam_local_240/refusals/`: generation and target failures.
- `data/raw/adaptive_redteam_local_240/transfer/`: successful A/E tests replayed on both targets.
- `data/processed/adaptive_redteam_local_240_rows.csv`: flattened dataset.
- `data/human_labels/adaptive_redteam_local_240_to_label.csv`: blind human-label CSV for your review.
- `data/cross_judge/adaptive_redteam_local_240.jsonl`: cross-judge verdicts.
- `results/adaptive_redteam_local_240/baseline/`, `adaptive/`, `ablations/`, `transfer/`, `judge_validation/`: tables and scores.
- `results/adaptive_redteam_local_240/figures/main/`, `figures/appendix/`: plots.
- `results/adaptive_redteam_local_240/config_snapshot.json`: exact model sources, revisions, weight SHA-256 hashes, and experiment settings.
- `results/adaptive_redteam_local_240/pipeline_status.json`: completion state and model cleanup status.
- `results/adaptive_redteam_local_240/logs/`: pipeline and local server logs.

## Human validation

Read the provided principle, criteria, transcript, and target response. Fill `human_violation` with 0 or 1, `human_severity` with 0–4, and optional notes. Save as `data/human_labels/adaptive_redteam_local_240_labeled.csv`.

The export hides automated verdicts, but its test IDs reveal arm and target identity. For interpretation, also retain these known limitations of the existing experiment: B and D use hidden judge outcomes to select feedback; length matching uses characters, not exact tokens; the bootstrap intervals treat attempts as independent despite shared histories; transfer selection uses the primary judge's unvalidated verdicts. Human labels assess verdict accuracy, but do not by themselves remove these design limitations.

After labeling, compute agreement and regenerate calibration figures without downloading model weights:

```bash
.venv/bin/python -c 'from run_local_pipeline import local_config; from experiments.judge_validation import score_agreement; from analysis.report import analyze; cfg, _ = local_config(); score_agreement(cfg); analyze(cfg)'
```

## Model cleanup

The pipeline stops its three local servers and deletes only the three checksum-verified GGUF files in `.local-models/local_240/` after all 240 discovery judgments, transfer replay checks, cross-judge checks, and analysis verification succeed. Human review remains pending and uses saved text; it does not require model weights. On failure, weights and results are preserved for resume. Runtime binaries remain available; no unrelated caches or files are deleted.
