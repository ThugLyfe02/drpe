from __future__ import annotations

import numpy as np
import torch
from torch import nn


class MultiObjectiveRanker(nn.Module):
    """Lightweight MLP ranker with two heads.

    Head A: engagement logit (optimize engagement depth)
    Head B: retention proxy prediction (durability signal)

    This is intentionally compact: the project’s core value is *systems maturity*
    (guardrails, drift, rollout), not overbuilt model complexity.
    """

    def __init__(self, in_dim: int, hidden: int = 64, dropout: float = 0.10):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.engagement_head = nn.Linear(hidden, 1)
        self.retention_head = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.backbone(x)
        e = self.engagement_head(h).squeeze(-1)  # logits
        r = self.retention_head(h).squeeze(-1)  # regression
        return e, r


def build_features(
    *,
    score: float,
    affinity: float,
    rank: int,
    item_quality: float,
    item_popularity: float,
    cohort: str,
) -> np.ndarray:
    """Small, stable feature set (human-explainable).

    Note: score is from retrieval stage (dot + pop bias).
    affinity is simulator-side "truth" proxy used only for training labels/features here.
    """
    # cohort one-hot (new/core/power)
    c_new = 1.0 if cohort == "new" else 0.0
    c_core = 1.0 if cohort == "core" else 0.0
    c_power = 1.0 if cohort == "power" else 0.0

    return np.array(
        [
            float(score),
            float(affinity),
            float(rank),
            float(item_quality),
            float(item_popularity),
            c_new,
            c_core,
            c_power,
        ],
        dtype=np.float32,
    )
