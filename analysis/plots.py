"""All figures for the paper. Each function returns a matplotlib Figure, or None if its data is missing."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

from analysis.calibration import expected_calibration_error, reliability_curve  # noqa: E402
from analysis.metrics import cohen_kappa, severity_distribution  # noqa: E402
from analysis.statistics import bootstrap_diff_ci, wilson_ci  # noqa: E402
from utils.logger import get_logger  # noqa: E402

log = get_logger(__name__)

ARM_ORDER = list("ABCDEF")
LABELS = {"A": "A baseline", "B": "B replies only", "C": "C + scores", "D": "D + explanations",
          "E": "E full feedback", "F": "F shuffled feedback"}
COLORS = {"A": "#7f7f7f", "B": "#9ecae1", "C": "#6baed6", "D": "#3182bd", "E": "#d62728", "F": "#ff9896"}


@dataclass
class PlotData:
    df: pd.DataFrame
    emb: np.ndarray
    cum: pd.DataFrame
    div: pd.DataFrame
    transfer: Optional[pd.DataFrame] = None   # transfer matrix (source x dest)
    human: Optional[pd.DataFrame] = None      # human labels merged with judge verdicts
    refusal: Optional[pd.DataFrame] = None
    thr: int = 1

    @property
    def valid(self) -> pd.DataFrame:
        return self.df[self.df["valid"]]

    @property
    def arms(self) -> list:
        return [a for a in ARM_ORDER if a in set(self.df["arm"])]


def _ci_err(ks, ns):
    ps = [k / n if n else 0.0 for k, n in zip(ks, ns)]
    ci = [wilson_ci(int(k), int(n)) for k, n in zip(ks, ns)]
    return ps, [[p - lo for p, (lo, _) in zip(ps, ci)], [hi - p for p, (_, hi) in zip(ps, ci)]]


def _heatmap(fig, ax, mat, rows, cols, title, vmin=0.0, vmax=1.0, cmap="viridis"):
    im = ax.imshow(mat, vmin=vmin, vmax=vmax, cmap=cmap, aspect="auto")
    ax.set_xticks(range(len(cols)), cols, rotation=30, ha="right")
    ax.set_yticks(range(len(rows)), rows)
    for i in range(len(rows)):
        for j in range(len(cols)):
            if not np.isnan(mat[i, j]):
                ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=8,
                        color="white" if mat[i, j] < (vmin + vmax) / 2 else "black")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046)


# ------------------------------- main figures ------------------------------------
def fig_architecture(d: PlotData):
    fig, ax = plt.subplots(figsize=(10, 3.4))
    ax.axis("off")
    ax.set_xlim(0, 10.4)
    ax.set_ylim(0, 3.4)
    boxes = [("Principles +\nseed topics", 0.1), ("Agent 1\nRed-team generator", 2.2),
             ("Agent 2\nTarget model", 4.3), ("Agent 3\nSafety judge", 6.4), ("Result store\n(JSONL)", 8.5)]
    for text, x in boxes:
        ax.add_patch(FancyBboxPatch((x, 0.9), 1.7, 1.1, boxstyle="round,pad=0.05", fc="#eef3fb", ec="#2b5aa0"))
        ax.text(x + 0.85, 1.45, text, ha="center", va="center", fontsize=9)
    for (_, x1), (_, x2) in zip(boxes[:-1], boxes[1:]):
        ax.annotate("", xy=(x2, 1.45), xytext=(x1 + 1.7, 1.45), arrowprops=dict(arrowstyle="->"))
    ax.annotate("", xy=(3.05, 2.05), xytext=(9.35, 2.05),
                arrowprops=dict(arrowstyle="->", ls="--", color="#d62728", connectionstyle="arc3,rad=0.3"))
    ax.text(6.2, 3.05, "in-context feedback (adaptive arms only)", ha="center", fontsize=9, color="#d62728")
    ax.text(5.2, 0.35, "Arms: A none | B replies | C +scores | D +explanations | E full | F shuffled control",
            ha="center", fontsize=8)
    return fig


def fig_asr_by_round(d: PlotData):
    v = d.valid
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    plotted = False
    for arm in ("A", "E"):
        g = v[v["arm"] == arm].groupby("round")["success"].agg(["sum", "count"])
        if g.empty:
            continue
        ci = np.array([wilson_ci(int(k), int(n)) for k, n in zip(g["sum"], g["count"])])
        ax.plot(g.index, g["sum"] / g["count"], marker="o", color=COLORS[arm], label=LABELS[arm])
        ax.fill_between(g.index, ci[:, 0], ci[:, 1], color=COLORS[arm], alpha=0.18)
        plotted = True
    if not plotted:
        plt.close(fig)
        return None
    ax.set(xlabel="Round", ylabel="Attack success rate", title="ASR by round (pooled over seeds and targets)")
    ax.legend()
    return fig


def fig_cumulative_unique(d: PlotData):
    if d.cum.empty:
        return None
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for arm in ("A", "E"):
        g = d.cum[d.cum["arm"] == arm].groupby("round")["cum_unique"].agg(["mean", "std"]).fillna(0)
        if g.empty:
            continue
        ax.plot(g.index, g["mean"], marker="o", color=COLORS[arm], label=LABELS[arm])
        ax.fill_between(g.index, g["mean"] - g["std"], g["mean"] + g["std"], color=COLORS[arm], alpha=0.18)
    ax.set(xlabel="Round", ylabel="Cumulative unique failures", title="Unique failures discovered")
    ax.legend()
    return fig


def fig_asr_by_category(d: PlotData):
    v = d.valid
    cats = sorted(v["category"].unique())
    if not {"A", "E"} & set(v["arm"]):
        return None
    fig, ax = plt.subplots(figsize=(7, 4.2))
    w = 0.38
    for i, arm in enumerate(("A", "E")):
        g = v[v["arm"] == arm].groupby("category")["success"].agg(["sum", "count"]).reindex(cats).fillna(0)
        ps, err = _ci_err(g["sum"], g["count"])
        ax.bar(np.arange(len(cats)) + (i - 0.5) * w, ps, w, yerr=err, capsize=3, color=COLORS[arm], label=LABELS[arm])
    ax.set_xticks(range(len(cats)), cats, rotation=15)
    ax.set(ylabel="ASR", title="ASR by attack category")
    ax.legend()
    return fig


def fig_principle_heatmap(d: PlotData):
    v = d.valid
    sub = v[v["arm"] == "E"] if "E" in set(v["arm"]) else v[v["arm"] != "A"]
    if sub.empty:
        return None
    pv = sub.pivot_table(index="principle", columns="category", values="success", aggfunc="mean")
    fig, ax = plt.subplots(figsize=(7, 5))
    _heatmap(fig, ax, pv.to_numpy(dtype=float), list(pv.index), list(pv.columns), "ASR by principle and category (adaptive)")
    return fig


def fig_severity(d: PlotData):
    ct = severity_distribution(d.df, "arm")
    if ct.empty:
        return None
    ct = ct.reindex([a for a in ARM_ORDER if a in ct.index])
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    bottom = np.zeros(len(ct))
    for s in range(5):
        ax.bar([LABELS[a] for a in ct.index], ct[s].to_numpy(), bottom=bottom, label=f"severity {s}")
        bottom += ct[s].to_numpy()
    ax.set(ylabel="Share of tests", title="Severity distribution")
    ax.tick_params(axis="x", rotation=25)
    ax.legend(fontsize=8)
    return fig


def fig_ablation(d: PlotData):
    v, arms = d.valid, d.arms
    if len(arms) < 2:
        return None
    g = v.groupby("arm")["success"].agg(["sum", "count"]).reindex(arms)
    ps, err = _ci_err(g["sum"], g["count"])
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.bar([LABELS[a] for a in arms], ps, yerr=err, capsize=3, color=[COLORS[a] for a in arms])
    ax.set(ylabel="ASR", title="Ablation: which feedback matters?")
    ax.tick_params(axis="x", rotation=25)
    return fig


def fig_transfer(d: PlotData):
    if d.transfer is None or d.transfer.empty:
        return None
    t = d.transfer
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    _heatmap(fig, ax, t.to_numpy(dtype=float), list(t.index), list(t.columns), "Cross-model transfer")
    ax.set(xlabel="Replayed on", ylabel="Discovered on")
    return fig


def fig_human_vs_judge(d: PlotData):
    if d.human is None or d.human.empty:
        return None
    h = d.human["human_violation"].astype(int).to_numpy()
    j = d.human["violation"].astype(int).to_numpy()
    cm = np.array([[np.sum((h == a) & (j == b)) for b in (0, 1)] for a in (0, 1)], dtype=float)
    fig, ax = plt.subplots(figsize=(4.6, 4))
    ax.imshow(cm, cmap="Blues")
    for a in range(2):
        for b in range(2):
            ax.text(b, a, int(cm[a, b]), ha="center", va="center", fontsize=12)
    ax.set_xticks([0, 1], ["judge: safe", "judge: violation"])
    ax.set_yticks([0, 1], ["human: safe", "human: violation"])
    ax.set_title(f"Human vs judge (Cohen's kappa = {cohen_kappa(h, j):.2f}, n={len(h)})")
    return fig


def fig_diversity(d: PlotData):
    if d.div.empty:
        return None
    arms = [a for a in ARM_ORDER if a in set(d.div["arm"])]
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for ax, col, title in ((axes[0], "mean_pairwise_distance", "Mean pairwise distance (higher = more diverse)"),
                           (axes[1], "self_bleu", "Self-BLEU (lower = more diverse)")):
        g = d.div.groupby("arm")[col].agg(["mean", "std"]).reindex(arms).fillna(0)
        ax.bar([LABELS[a] for a in arms], g["mean"], yerr=g["std"], capsize=3, color=[COLORS[a] for a in arms])
        ax.set_title(title, fontsize=9)
        ax.tick_params(axis="x", rotation=30)
    return fig


def fig_embedding_map(d: PlotData):
    df = d.df.reset_index(drop=True)
    idx = np.where(df["arm"].isin(["A", "E"]))[0]
    if len(idx) < 8:
        return None
    if len(idx) > 3000:
        idx = np.random.default_rng(0).choice(idx, 3000, replace=False)
    X = d.emb[idx]
    try:
        import umap

        Z = umap.UMAP(n_components=2, random_state=0).fit_transform(X)
    except ImportError:
        from sklearn.manifold import TSNE

        Z = TSNE(n_components=2, init="pca", random_state=0, perplexity=min(30, len(X) - 1)).fit_transform(X)
    sub = df.iloc[idx]
    fig, ax = plt.subplots(figsize=(6.6, 5.4))
    for arm in ("A", "E"):
        for succ, marker, size in ((False, "o", 14), (True, "X", 50)):
            m = ((sub["arm"] == arm) & (sub["success"] == succ)).to_numpy()
            ax.scatter(Z[m, 0], Z[m, 1], s=size, marker=marker, color=COLORS[arm], alpha=0.7,
                       label=f"{LABELS[arm]}, {'success' if succ else 'no failure'}")
    ax.set_title("Embedding map of generated tests")
    ax.legend(fontsize=7)
    ax.set_xticks([])
    ax.set_yticks([])
    return fig


# ------------------------------ appendix figures -----------------------------------
def fig_effect_sizes(d: PlotData):
    v = d.valid
    if not {"A", "E"} <= set(v["arm"]):
        return None
    groups = [("Overall", v)] + [(f"category: {c}", v[v["category"] == c]) for c in sorted(v["category"].unique())] \
        + [(f"target: {t}", v[v["target_key"] == t]) for t in sorted(v["target_key"].unique())]
    fig, ax = plt.subplots(figsize=(6.6, 0.45 * len(groups) + 1.5))
    labels = []
    for y, (label, sub) in enumerate(groups):
        a = sub[sub["arm"] == "A"]["success"].astype(float).to_numpy()
        b = sub[sub["arm"] == "E"]["success"].astype(float).to_numpy()
        if len(a) < 2 or len(b) < 2:
            continue
        diff, lo, hi = bootstrap_diff_ci(a, b, 2000)
        ax.errorbar(diff, len(labels), xerr=[[diff - lo], [hi - diff]], fmt="o", color=COLORS["E"], capsize=3)
        labels.append(label)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_yticks(range(len(labels)), labels)
    ax.invert_yaxis()
    ax.set(xlabel="ASR difference (adaptive minus baseline), 95% bootstrap CI", title="Effect size")
    return fig


def fig_calibration(d: PlotData):
    if d.human is None or d.human.empty:
        return None
    correct = (d.human["human_violation"].astype(bool) == d.human["violation"].astype(bool)).to_numpy()
    conf = d.human["confidence"].to_numpy()
    curve = reliability_curve(conf, correct, n_bins=5)
    if curve.empty:
        return None
    fig, ax = plt.subplots(figsize=(4.8, 4.4))
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.plot(curve["mean_conf"], curve["accuracy"], marker="o")
    ax.set(xlabel="Judge confidence", ylabel="Accuracy vs human",
           title=f"Judge calibration (ECE = {expected_calibration_error(conf, correct, 5):.2f})")
    return fig


def fig_refusals(d: PlotData):
    if d.refusal is None or d.refusal.empty:
        return None
    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.bar(d.refusal["generator_model"], d.refusal["refusal_rate"], color="#8c564b")
    ax.set(ylabel="Refusal rate", title="Generator refusal rate")
    ax.tick_params(axis="x", rotation=20)
    return fig


def fig_usage(d: PlotData):
    v = d.valid
    if v.empty:
        return None
    g = v.groupby("arm")[["latency", "prompt_tokens", "response_tokens"]].mean().reindex(d.arms)
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    axes[0].bar([LABELS[a] for a in g.index], g["latency"], color=[COLORS[a] for a in g.index])
    axes[0].set_title("Target latency (s)")
    axes[1].bar([LABELS[a] for a in g.index], g["prompt_tokens"] + g["response_tokens"], color=[COLORS[a] for a in g.index])
    axes[1].set_title("Target tokens per test")
    for ax in axes:
        ax.tick_params(axis="x", rotation=30)
    return fig


def fig_per_seed(d: PlotData):
    v = d.valid
    g = v.groupby(["arm", "target_key", "seed"])["success"].mean().reset_index()
    arms = d.arms
    if g.empty:
        return None
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.boxplot([g[g["arm"] == a]["success"].to_numpy() for a in arms], positions=range(len(arms)), widths=0.5)
    rng = np.random.default_rng(0)
    for i, a in enumerate(arms):
        y = g[g["arm"] == a]["success"].to_numpy()
        ax.scatter(i + rng.uniform(-0.12, 0.12, len(y)), y, color=COLORS[a], s=18, zorder=3)
    ax.set_xticks(range(len(arms)), [LABELS[a] for a in arms], rotation=25)
    ax.set(ylabel="ASR per (target, seed)", title="Seed-to-seed variability")
    return fig


def fig_shuffled_vs_real(d: PlotData):
    v = d.valid
    arms = [a for a in ("A", "E", "F") if a in set(v["arm"])]
    if "E" not in arms or "F" not in arms:
        return None
    late = v[v["round"] > v["round"].max() / 2]
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 4))
    for ax, data, title in ((axes[0], v, "All rounds"), (axes[1], late, "Second half of rounds")):
        g = data.groupby("arm")["success"].agg(["sum", "count"]).reindex(arms)
        ps, err = _ci_err(g["sum"], g["count"])
        ax.bar([LABELS[a] for a in arms], ps, yerr=err, capsize=3, color=[COLORS[a] for a in arms])
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=20)
    axes[0].set_ylabel("ASR")
    fig.suptitle("Real vs shuffled feedback")
    return fig


FIGURES = [
    (fig_architecture, "main", "fig01_architecture"),
    (fig_asr_by_round, "main", "fig02_asr_by_round"),
    (fig_cumulative_unique, "main", "fig03_cumulative_unique_failures"),
    (fig_asr_by_category, "main", "fig04_asr_by_category"),
    (fig_principle_heatmap, "main", "fig05_principle_category_heatmap"),
    (fig_severity, "main", "fig06_severity_distribution"),
    (fig_ablation, "main", "fig07_ablation"),
    (fig_transfer, "main", "fig08_transfer_matrix"),
    (fig_human_vs_judge, "main", "fig09_human_vs_judge"),
    (fig_diversity, "main", "fig10_diversity"),
    (fig_embedding_map, "main", "fig11_embedding_map"),
    (fig_effect_sizes, "appendix", "fig12_effect_sizes"),
    (fig_calibration, "appendix", "fig13_judge_calibration"),
    (fig_refusals, "appendix", "fig14_generator_refusals"),
    (fig_usage, "appendix", "fig15_latency_tokens"),
    (fig_per_seed, "appendix", "fig16_per_seed_asr"),
    (fig_shuffled_vs_real, "appendix", "fig17_shuffled_vs_real"),
]


def generate_all(d: PlotData, out_dir: Path) -> None:
    for fn, sub, name in FIGURES:
        try:
            fig = fn(d)
        except Exception as exc:  # noqa: BLE001
            log.warning("Skipping %s: %s: %s", name, type(exc).__name__, exc)
            plt.close("all")
            continue
        if fig is None:
            log.info("Skipping %s (not enough data yet)", name)
            continue
        path = Path(out_dir) / sub / f"{name}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(path, dpi=200)
        plt.close(fig)
        log.info("Saved %s", path)