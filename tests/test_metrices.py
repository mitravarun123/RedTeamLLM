import math

import pandas as pd

from analysis.metrics import asr_table, cohen_kappa
from analysis.statistics import bootstrap_ci, holm_bonferroni, paired_permutation_test, wilson_ci


def test_kappa_known_values():
    assert math.isclose(cohen_kappa([1, 1, 0, 0], [1, 0, 0, 0]), 0.5)
    assert cohen_kappa([1, 0, 1], [1, 0, 1]) == 1.0
    assert cohen_kappa([0, 1, 2, 3], [0, 1, 2, 3], weights="quadratic") == 1.0


def test_wilson_and_bootstrap():
    lo, hi = wilson_ci(0, 10)
    assert lo == 0.0 and hi > 0.0
    est, lo, hi = bootstrap_ci([0, 1, 1, 0, 1, 1, 1, 0], n_boot=500)
    assert lo <= est <= hi


def test_asr_table():
    df = pd.DataFrame({"arm": ["A", "A", "E", "E"], "valid": [True] * 4, "success": [False, True, True, True]})
    t = asr_table(df, ["arm"]).set_index("arm")
    assert t.loc["A", "asr"] == 0.5 and t.loc["E", "asr"] == 1.0


def test_permutation_and_holm():
    assert paired_permutation_test([0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1]) < 0.05
    assert holm_bonferroni([0.01, 0.04, 0.03]) == [0.03, 0.06, 0.06]