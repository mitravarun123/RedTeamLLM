"""How well does the judge's stated confidence match its correctness (vs human labels)?"""

from __future__ import annotations

import numpy as np
import pandas as pd


def reliability_curve(conf, correct, n_bins: int = 10) -> pd.DataFrame:
    conf, correct = np.asarray(conf, float), np.asarray(correct, float)
    edges = np.linspace(0, 1, n_bins + 1)
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf >= lo) & ((conf < hi) if hi < 1 else (conf <= hi))
        if m.sum():
            rows.append({"mean_conf": conf[m].mean(), "accuracy": correct[m].mean(), "count": int(m.sum())})
    return pd.DataFrame(rows)


def expected_calibration_error(conf, correct, n_bins: int = 10) -> float:
    curve = reliability_curve(conf, correct, n_bins)
    if curve.empty:
        return float("nan")
    return float((curve["count"] / curve["count"].sum() * (curve["accuracy"] - curve["mean_conf"]).abs()).sum())