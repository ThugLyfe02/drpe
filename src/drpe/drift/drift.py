from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np


@dataclass
class EmbeddingDriftResult:
    mean_cosine_shift: float
    p95_cosine_shift: float
    n: int


def cosine_shift(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute per-vector cosine distance between aligned embedding matrices.

    a, b: shape (N, D)
    returns: shape (N,) cosine distance = 1 - cosine_similarity
    """
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} vs {b.shape}")

    a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-8)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-8)
    sim = np.sum(a_norm * b_norm, axis=1)
    dist = 1.0 - sim
    return dist


def embedding_drift(a: np.ndarray, b: np.ndarray) -> EmbeddingDriftResult:
    dist = cosine_shift(a, b)
    return EmbeddingDriftResult(
        mean_cosine_shift=float(np.mean(dist)),
        p95_cosine_shift=float(np.quantile(dist, 0.95)),
        n=int(dist.shape[0]),
    )


def kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """KL(p||q) for discrete distributions."""
    eps = 1e-12
    p = p.astype(np.float64) + eps
    q = q.astype(np.float64) + eps
    p = p / p.sum()
    q = q / q.sum()
    return float(np.sum(p * np.log(p / q)))


def histogram_kl(a: np.ndarray, b: np.ndarray, bins: int = 30) -> float:
    """Approximate drift between two scalar distributions via histogram KL."""
    lo = float(min(a.min(), b.min()))
    hi = float(max(a.max(), b.max()))
    if lo == hi:
        return 0.0
    p, _ = np.histogram(a, bins=bins, range=(lo, hi))
    q, _ = np.histogram(b, bins=bins, range=(lo, hi))
    return kl_divergence(p, q)


def cohort_variance(values_by_cohort: Dict[str, List[float]]) -> float:
    """Variance of cohort means (simple stability signal)."""
    means = [np.mean(v) for v in values_by_cohort.values() if len(v) > 0]
    if len(means) <= 1:
        return 0.0
    return float(np.var(means))
