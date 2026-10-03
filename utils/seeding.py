"""Deterministic seeds and RNGs derived from arbitrary keys."""

from __future__ import annotations

import hashlib
import random

import numpy as np


def derive_seed(*parts) -> int:
    digest = hashlib.sha256("|".join(str(p) for p in parts).encode()).hexdigest()
    return int(digest[:8], 16)


def make_rng(*parts) -> random.Random:
    return random.Random(derive_seed(*parts))


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))