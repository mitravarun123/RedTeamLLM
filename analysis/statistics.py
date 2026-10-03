"""Confidence intervals and significance tests."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def bootstrap_ci(values, stat=np.mean, n_boot: int = 5000, alpha: float = 0.05, seed: int = 0):
    x = np.asarray(values, float)
    if len(x) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    samples = x[rng.integers(0, len(x), (n_boot, len(x)))]
    stats = stat(samples, axis=1)
    return float(stat(x)), float(np.quantile(stats, alpha / 2)), float(np.quantile(stats, 1 - alpha / 2))


def bootstrap_diff_ci(a, b, n_boot: int = 5000, alpha: float = 0.05, seed: int = 0):
    """CI for mean(b) - mean(a), resampling each group independently."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    rng = np.random.default_rng(seed)
    da = a[rng.integers(0, len(a), (n_boot, len(a)))].mean(1)
    db = b[rng.integers(0, len(b), (n_boot, len(b)))].mean(1)
    d = db - da
    return float(b.mean() - a.mean()), float(np.quantile(d, alpha / 2)), float(np.quantile(d, 1 - alpha / 2))


def paired_permutation_test(a, b, n_perm: int = 20000, seed: int = 0) -> float:
    """Sign-flip test on matched differences (b - a). Two-sided."""
    d = np.asarray(b, float) - np.asarray(a, float)
    if len(d) == 0:
        return float("nan")
    rng = np.random.default_rng(seed)
    obs = abs(d.mean())
    signs = rng.choice([-1.0, 1.0], size=(n_perm, len(d)))
    perm = np.abs((signs * d).mean(1))
    return float((np.sum(perm >= obs - 1e-12) + 1) / (n_perm + 1))


def cohens_h(p1: float, p2: float) -> float:
    return 2 * math.asin(math.sqrt(p2)) - 2 * math.asin(math.sqrt(p1))


def holm_bonferroni(pvals: list) -> list:
    m = len(pvals)
    order = np.argsort(pvals)
    adj, running = [0.0] * m, 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * pvals[i])
        adj[i] = min(1.0, running)
    return adj


def compare_arms(df: pd.DataFrame, arm_a: str, arm_b: str, n_boot: int = 5000, seed: int = 0) -> dict:
    """Effect of arm_b relative to arm_a (positive diff means b has the higher ASR).
    The paired test matches ASR per (target, seed, round)."""
    v = df[df["valid"]]
    a, b = v[v["arm"] == arm_a], v[v["arm"] == arm_b]
    if a.empty or b.empty:
        return {}
    xa, xb = a["success"].astype(float).to_numpy(), b["success"].astype(float).to_numpy()
    diff, lo, hi = bootstrap_diff_ci(xa, xb, n_boot, seed=seed)
    g = (v[v["arm"].isin([arm_a, arm_b])].groupby(["arm", "target_key", "seed", "round"])["success"]
         .mean().unstack("arm").dropna())
    p = paired_permutation_test(g[arm_a].to_numpy(), g[arm_b].to_numpy(), seed=seed) if len(g) else float("nan")
    ka, kb = int(xa.sum()), int(xb.sum())
    return {
        "arm_a": arm_a, "arm_b": arm_b, "n_a": len(xa), "n_b": len(xb),
        "asr_a": xa.mean(), "asr_a_lo": wilson_ci(ka, len(xa))[0], "asr_a_hi": wilson_ci(ka, len(xa))[1],
        "asr_b": xb.mean(), "asr_b_lo": wilson_ci(kb, len(xb))[0], "asr_b_hi": wilson_ci(kb, len(xb))[1],
        "diff": diff, "diff_lo": lo, "diff_hi": hi,
        "cohens_h": cohens_h(xa.mean(), xb.mean()), "p_paired_perm": p, "n_matched_units": len(g),
    }