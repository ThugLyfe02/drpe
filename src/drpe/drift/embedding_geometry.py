from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np

from drpe.drift.drift import EmbeddingDriftResult, cosine_shift, embedding_drift


@dataclass
class GeometryDriftReport:
    users: EmbeddingDriftResult
    items: EmbeddingDriftResult
    cohort_user_mean_shift: Dict[str, float]


def cohort_user_drift(users_v1: np.ndarray, users_v2: np.ndarray, user_cohorts: Dict[int, str]) -> Dict[str, float]:
    """Compute mean cosine *distance* per cohort for users.

    Returns values in [0, 2] (typically small, e.g., 0.0x–0.1x for incremental updates).
    """
    per_row = cosine_shift(users_v1, users_v2)  # shape (N,)

    buckets: Dict[str, list[float]] = {}
    for uid, c in user_cohorts.items():
        if uid < 0 or uid >= per_row.shape[0]:
            continue
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
