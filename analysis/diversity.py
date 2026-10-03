"""Test diversity: embeddings, pairwise distance, self-BLEU, unique failures."""

from __future__ import annotations

import math
import random
from collections import Counter

import numpy as np
import pandas as pd

_MODEL = None


def embed_texts(texts: list, model_name: str = "all-MiniLM-L6-v2") -> np.ndarray:
    """Sentence embeddings if sentence-transformers is installed, otherwise TF-IDF + SVD. L2-normalised."""
    global _MODEL
    texts = [t if t.strip() else "." for t in texts]
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        SentenceTransformer = None
    if SentenceTransformer is not None:
        if _MODEL is None:
            _MODEL = SentenceTransformer(model_name)
        return np.asarray(_MODEL.encode(texts, normalize_embeddings=True, show_progress_bar=False))
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer

    X = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), sublinear_tf=True).fit_transform(texts)
    k = max(2, min(128, X.shape[0] - 1, X.shape[1] - 1))
    Z = TruncatedSVD(n_components=k, random_state=0).fit_transform(X)
    return Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-9)


def mean_pairwise_distance(emb: np.ndarray) -> float:
    n = len(emb)
    if n < 2:
        return float("nan")
    sim = emb @ emb.T
    return float(np.mean(1 - sim[np.triu_indices(n, 1)]))


def _ngrams(tokens: list, n: int) -> Counter:
    return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


def self_bleu(texts: list, max_n: int = 4, max_texts: int = 150, seed: int = 0) -> float:
    """Average BLEU of each text against all the others. Higher means less diverse."""
    texts = list(texts)
    if len(texts) > max_texts:
        texts = random.Random(seed).sample(texts, max_texts)
    toks = [t.lower().split() for t in texts]
    grams = [[_ngrams(tk, n) for n in range(1, max_n + 1)] for tk in toks]
    scores = []
    for i, tk in enumerate(toks):
        if len(tk) < max_n:
            continue
        others = [j for j in range(len(toks)) if j != i]
        logs = []
        for n in range(1, max_n + 1):
            cand = grams[i][n - 1]
            total = sum(cand.values())
            clipped = sum(min(c, max((grams[j][n - 1].get(g, 0) for j in others), default=0))
                          for g, c in cand.items())
            logs.append(math.log(max(clipped / total if total else 0.0, 1e-9)))
        scores.append(math.exp(sum(logs) / max_n))
    return float(np.mean(scores)) if scores else float("nan")


def cumulative_unique_failures(df: pd.DataFrame, emb: np.ndarray, thr: float = 0.85) -> pd.DataFrame:
    """Per (arm, seed, target): running count of successful tests that are not near-duplicates
    (cosine similarity >= thr) of an earlier successful test. df index must match rows of emb."""
    rows = []
    max_round = int(df["round"].max())
    for (arm, seed, target), g in df.groupby(["arm", "seed", "target_key"]):
        kept: list = []
        for r in range(1, max_round + 1):
            for i in g.index[(g["round"] == r) & g["success"]]:
                v = emb[i]
                if not kept or max(float(k @ v) for k in kept) < thr:
                    kept.append(v)
            rows.append({"arm": arm, "seed": seed, "target_key": target, "round": r, "cum_unique": len(kept)})
    return pd.DataFrame(rows)


def diversity_table(df: pd.DataFrame, emb: np.ndarray) -> pd.DataFrame:
    rows = []
    for (arm, seed, target), g in df.groupby(["arm", "seed", "target_key"]):
        rows.append({"arm": arm, "seed": seed, "target_key": target,
                     "mean_pairwise_distance": mean_pairwise_distance(emb[g.index.to_numpy()]),
                     "self_bleu": self_bleu(g["test_text"].tolist())})
    return pd.DataFrame(rows)