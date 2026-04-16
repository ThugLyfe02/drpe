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
    # Optional: max allowed increase in cohort variance before we pause
    max_cohort_variance: Optional[float] = None


def decide_rollout(
    *,
    baseline_retention: float,
    candidate_retention: float,
    cohort_retention_baseline: Dict[str, float] | None = None,
    cohort_retention_candidate: Dict[str, float] | None = None,
    cfg: GuardrailConfig = GuardrailConfig(),
) -> GuardrailDecision:
    """Block rollout if durability guardrail is breached.

    This intentionally prioritizes durability over short-term lift.
    """
    if baseline_retention <= 0:
        return GuardrailDecision(False, "invalid baseline retention")

    rel_drop = (baseline_retention - candidate_retention) / baseline_retention

    if rel_drop > cfg.max_retention_drop:
        return GuardrailDecision(
            False,
            f"blocked: retention proxy drop {rel_drop:.3%} exceeds {cfg.max_retention_drop:.3%}",
        )

    # Optional cohort-level guardrail (simple worst-cohort drop check)
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

    return GuardrailDecision(True, "allowed: guardrails satisfied")
