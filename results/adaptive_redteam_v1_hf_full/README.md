# Full Hugging Face pipeline

Run or resume from the project root:

```bash
.venv/bin/python -u run_pipeline.py
```

Configuration: 2 targets, arms A–F, seeds 0–2, 10 rounds, 20 tests per round (7,200 scheduled slots).

Outputs:

- `data/raw/adaptive_redteam_v1_hf_full/runs/`: baseline, adaptive, and ablation interactions; each filename identifies arm, seed, and target.
- `data/raw/adaptive_redteam_v1_hf_full/refusals/`: failed generation and target attempts.
- `data/raw/adaptive_redteam_v1_hf_full/transfer/`: cross-model and same-model replay records.
- `data/processed/adaptive_redteam_v1_hf_full_rows.csv`: flattened run dataset.
- `data/human_labels/adaptive_redteam_v1_hf_full_to_label.csv`: blind human-label sample. Fill manually and save as `adaptive_redteam_v1_hf_full_labeled.csv`.
- `data/cross_judge/adaptive_redteam_v1_hf_full.jsonl`: independent cross-judge evaluations.
- `results/adaptive_redteam_v1_hf_full/baseline/`: baseline tables.
- `results/adaptive_redteam_v1_hf_full/adaptive/`: adaptive tables and cumulative unique failures.
- `results/adaptive_redteam_v1_hf_full/ablations/`: arm comparisons, diversity, usage, and refusal tables.
- `results/adaptive_redteam_v1_hf_full/transfer/`: transfer matrix.
- `results/adaptive_redteam_v1_hf_full/judge_validation/`: agreement scores.
- `results/adaptive_redteam_v1_hf_full/figures/`: main and appendix plots.
- `results/adaptive_redteam_v1_hf_full/logs/pipeline.log`: progress log.
- `results/adaptive_redteam_v1_hf_full/config_snapshot.json`: configuration used.
- `results/adaptive_redteam_v1_hf_full/pipeline_status.json`: running, interrupted, incomplete, or complete status.

Downstream outputs are written when their stages execute. If quota or provider access interrupts the pipeline, stored interactions are preserved and partial analysis is attempted. Partial results are not a completed study. Human agreement and calibration figures require actual human labels.
