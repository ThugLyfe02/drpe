from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class GuardrailDecision:
    allow_rollout: bool
    reason: str


@dataclass
class GuardrailConfig:
    # Maximum allowed relative drop in retention proxy (e.g. 0.01 = 1% drop)
    max_retention_drop: float = 0.01

    # Optional: maximum allowed mean cosine shift for user/item embeddings between versions.
    # This treats embedding updates like schema migrations: measurable drift gates rollout.
    max_embedding_mean_cosine_shift: Optional[float] = None

    # Optional: maximum allowed mean cosine shift per cohort (users).
    # If provided, we block rollout if any cohort exceeds this threshold.
    max_embedding_mean_cosine_shift_per_cohort: Optional[float] = None


def decide_rollout(
    *,
    baseline_retention: float,
    candidate_retention: float,
    cohort_retention_baseline: Dict[str, float] | None = None,
    cohort_retention_candidate: Dict[str, float] | None = None,
    embedding_mean_cosine_shift_users: Optional[float] = None,
    embedding_mean_cosine_shift_items: Optional[float] = None,
    cohort_user_mean_shift: Dict[str, float] | None = None,
    cfg: GuardrailConfig = GuardrailConfig(),
) -> GuardrailDecision:
    """Block rollout if durability guardrails are breached.

    Guardrails:
    - retention proxy relative drop
    - worst-cohort retention proxy relative drop
    - embedding geometry drift (mean cosine shift for users/items)
    - embedding cohort drift (any cohort user mean shift exceeds threshold)
    """

    if baseline_retention <= 0:
        return GuardrailDecision(False, "invalid baseline retention")

    rel_drop = (baseline_retention - candidate_retention) / baseline_retention

    if rel_drop > cfg.max_retention_drop:
        return GuardrailDecision(
            False,
            f"blocked: retention proxy drop {rel_drop:.3%} exceeds {cfg.max_retention_drop:.3%}",
        )

    if cohort_retention_baseline and cohort_retention_candidate:
        worst = 0.0
        worst_cohort = None
        for c, base in cohort_retention_baseline.items():
            cand = cohort_retention_candidate.get(c)
            if cand is None or base <= 0:
                continue
            d = (base - cand) / base
            if d > worst:
                worst = d
                worst_cohort = c
        if worst_cohort is not None and worst > cfg.max_retention_drop:
            return GuardrailDecision(
                False,
                f"blocked: cohort {worst_cohort} retention drop {worst:.3%} exceeds {cfg.max_retention_drop:.3%}",
            )

    if cfg.max_embedding_mean_cosine_shift is not None:
        if embedding_mean_cosine_shift_users is not None and embedding_mean_cosine_shift_users > cfg.max_embedding_mean_cosine_shift:
            return GuardrailDecision(
                False,
                f"blocked: user embedding drift mean {embedding_mean_cosine_shift_users:.4f} exceeds {cfg.max_embedding_mean_cosine_shift:.4f}",
            )
        if embedding_mean_cosine_shift_items is not None and embedding_mean_cosine_shift_items > cfg.max_embedding_mean_cosine_shift:
            return GuardrailDecision(
                False,
                f"blocked: item embedding drift mean {embedding_mean_cosine_shift_items:.4f} exceeds {cfg.max_embedding_mean_cosine_shift:.4f}",
            )

    if cfg.max_embedding_mean_cosine_shift_per_cohort is not None and cohort_user_mean_shift:
        # Gate if any cohort exceeds threshold
        for c, v in cohort_user_mean_shift.items():
            if v > cfg.max_embedding_mean_cosine_shift_per_cohort:
                return GuardrailDecision(
                    False,
                    f"blocked: cohort {c} user embedding drift mean {v:.4f} exceeds {cfg.max_embedding_mean_cosine_shift_per_cohort:.4f}",
                )

    return GuardrailDecision(True, "allowed: guardrails satisfied")
