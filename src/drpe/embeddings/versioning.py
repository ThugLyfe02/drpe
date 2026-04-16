from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np


@dataclass
class EmbeddingVersions:
    users_v1: np.ndarray
    users_v2: np.ndarray
    items_v1: np.ndarray
    items_v2: np.ndarray
    user_cohorts: Dict[int, str]


def _assign_cohort(rng: np.random.Generator) -> str:
    r = rng.random()
    if r < 0.25:
        return "new"
    if r < 0.75:
        return "core"
    return "power"


def _normalize_rows(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)


def generate_embedding_versions(
    *,
    seed: int,
    embedding_dim: int,
    num_users: int,
    num_items: int,
    drift_strength: float = 0.03,
) -> EmbeddingVersions:
    """Generate aligned v1/v2 embedding matrices for users/items.

    v2 is a controlled perturbation of v1. This is intentionally simple for DRPE v0:
    it lets us demonstrate *embedding geometry drift* and rollout gating.
    """

    rng = np.random.default_rng(seed)

    users_v1 = _normalize_rows(rng.normal(0, 1, (num_users, embedding_dim)).astype(np.float32))
    items_v1 = _normalize_rows(rng.normal(0, 1, (num_items, embedding_dim)).astype(np.float32))

    users_v2 = _normalize_rows(users_v1 + rng.normal(0, drift_strength, users_v1.shape).astype(np.float32))
    items_v2 = _normalize_rows(items_v1 + rng.normal(0, drift_strength, items_v1.shape).astype(np.float32))

    cohorts: Dict[int, str] = {uid: _assign_cohort(rng) for uid in range(num_users)}

    return EmbeddingVersions(
        users_v1=users_v1,
        users_v2=users_v2,
        items_v1=items_v1,
        items_v2=items_v2,
        user_cohorts=cohorts,
    )
