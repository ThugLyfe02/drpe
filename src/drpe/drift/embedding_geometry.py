from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np

from drpe.drift.drift import EmbeddingDriftResult, embedding_drift


@dataclass
class GeometryDriftReport:
    users: EmbeddingDriftResult
    items: EmbeddingDriftResult
    cohort_user_mean_shift: Dict[str, float]


def cohort_user_drift(users_v1: np.ndarray, users_v2: np.ndarray, user_cohorts: Dict[int, str]) -> Dict[str, float]:
    """Compute mean cosine shift per cohort for users."""
    # aligned matrices => drift per row
    dist = 1.0 - (users_v1 / (np.linalg.norm(users_v1, axis=1, keepdims=True) + 1e-8)) * (
        users_v2 / (np.linalg.norm(users_v2, axis=1, keepdims=True) + 1e-8)
    )
    per_row = np.sum(dist, axis=1)

    buckets: Dict[str, list[float]] = {}
    for uid, c in user_cohorts.items():
        buckets.setdefault(c, []).append(float(per_row[uid]))

    return {c: float(np.mean(v)) for c, v in buckets.items() if len(v) > 0}


def build_geometry_drift_report(
    *,
    users_v1: np.ndarray,
    users_v2: np.ndarray,
    items_v1: np.ndarray,
    items_v2: np.ndarray,
    user_cohorts: Dict[int, str],
) -> GeometryDriftReport:
    users_res = embedding_drift(users_v1, users_v2)
    items_res = embedding_drift(items_v1, items_v2)

    cohort = cohort_user_drift(users_v1, users_v2, user_cohorts)

    return GeometryDriftReport(users=users_res, items=items_res, cohort_user_mean_shift=cohort)
