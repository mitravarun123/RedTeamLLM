"""Reads stored results, writes tables and figures. Triggered by: python main.py --analyze"""

from __future__ import annotations

import json

import pandas as pd

from analysis.diversity import cumulative_unique_failures, diversity_table, embed_texts
from analysis.metrics import (asr_table, load_dataframe, load_human_labels, load_refusals,
                              load_transfer_dataframe, refusal_rate, transfer_matrix, usage_table)
from analysis.plots import PlotData, generate_all
from analysis.statistics import compare_arms, holm_bonferroni
from utils.config import Config
from utils.logger import get_logger

log = get_logger(__name__)


def analyze(cfg: Config) -> None:
    exp = cfg.exp
    df = load_dataframe(cfg.raw_dir() / "runs", exp.severity_threshold)
    if df.empty:
        log.error("No run data found in %s. Run an experiment first.", cfg.raw_dir() / "runs")
        return
    df = df.reset_index(drop=True)
    res = cfg.results_dir()

    def save(table: pd.DataFrame, sub: str, name: str) -> None:
        if table is not None and not table.empty:
            p = res / sub / name
            p.parent.mkdir(parents=True, exist_ok=True)
            table.to_csv(p, index=False)

    cfg.processed_dir().mkdir(parents=True, exist_ok=True)
    flat = df.copy()
    flat["conversation"] = flat["conversation"].map(json.dumps)
    flat.to_csv(cfg.processed_dir() / f"{exp.name}_rows.csv", index=False)

    for arm, sub in (("A", "baseline"), ("E", "adaptive")):
        part = df[df["arm"] == arm]
        save(asr_table(part, ["round"]), sub, "asr_by_round.csv")
        save(asr_table(part, ["category"]), sub, "asr_by_category.csv")
        save(asr_table(part, ["principle"]), sub, "asr_by_principle.csv")
    save(asr_table(df, ["arm"]), "ablations", "asr_by_arm.csv")
    save(asr_table(df, ["arm", "target_key", "seed"]), "ablations", "asr_per_seed.csv")
    save(usage_table(df), "ablations", "usage_by_arm.csv")

    pairs = [("A", "B"), ("A", "C"), ("A", "D"), ("A", "E"), ("A", "F"), ("F", "E")]
    comps = [c for c in (compare_arms(df, a, b) for a, b in pairs) if c]
    if comps:
        ps = [c["p_paired_perm"] for c in comps]
        if not any(pd.isna(p) for p in ps):
            for c, adj in zip(comps, holm_bonferroni(ps)):
                c["p_holm"] = adj
        save(pd.DataFrame(comps), "ablations", "arm_comparisons.csv")

    emb = embed_texts(df["test_text"].tolist())
    cum = cumulative_unique_failures(df, emb, exp.dedup_similarity)
    div = diversity_table(df, emb)
    save(cum, "adaptive", "cumulative_unique_failures.csv")
    save(div, "ablations", "diversity.csv")

    tdf = load_transfer_dataframe(cfg.raw_dir() / "transfer", exp.severity_threshold)
    tmat = transfer_matrix(tdf) if not tdf.empty else None
    if tmat is not None and not tmat.empty:
        p = res / "transfer" / "transfer_matrix.csv"
        p.parent.mkdir(parents=True, exist_ok=True)
        tmat.to_csv(p)

    human = load_human_labels(cfg.human_dir() / f"{exp.name}_labeled.csv", df)
    refusals = refusal_rate(df, load_refusals(cfg.raw_dir() / "refusals"))
    save(refusals, "ablations", "generator_refusals.csv")

    generate_all(PlotData(df=df, emb=emb, cum=cum, div=div, transfer=tmat, human=human,
                          refusal=refusals, thr=exp.severity_threshold), cfg.figures_dir())

    valid = df[df["valid"]]
    summary = {
        "experiment": exp.name, "tests_total": int(len(df)), "tests_valid": int(len(valid)),
        "asr_by_arm": {a: float(g["success"].mean()) for a, g in valid.groupby("arm")},
        "main_comparison_A_vs_E": next((c for c in comps if c["arm_a"] == "A" and c["arm_b"] == "E"), None),
    }
    (res / "summary.json").parent.mkdir(parents=True, exist_ok=True)
    (res / "summary.json").write_text(json.dumps(summary, indent=2, default=float), encoding="utf-8")
    print(json.dumps(summary, indent=2, default=float))