from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

from drpe.data.simulator import SimConfig, run_simulation
from drpe.evaluation.metrics import cohort_retention_means, engagement_depth_mean, retention_proxy_mean
from drpe.rollout.guardrails import GuardrailConfig, GuardrailDecision, decide_rollout


@dataclass
class VariantSummary:
    engagement_depth: float
    retention_proxy: float
    cohort_retention: Dict[str, float]


@dataclass
class RolloutComparison:
    baseline: VariantSummary
    candidate: VariantSummary
    decision: GuardrailDecision


def summarize_variant(cfg: SimConfig, *, embedding_version: str, model_version: str) -> VariantSummary:
    _, summaries = run_simulation(cfg, embedding_version=embedding_version, model_version=model_version)
    return VariantSummary(
        engagement_depth=engagement_depth_mean(summaries),
        retention_proxy=retention_proxy_mean(summaries),
        cohort_retention=cohort_retention_means(summaries),
    )


def compare_for_rollout(
    *,
    baseline_cfg: SimConfig,
    candidate_cfg: SimConfig,
    guardrails: GuardrailConfig = GuardrailConfig(),
    embedding_version: str = "emb_v1",
) -> RolloutComparison:
    """Run two simulated variants and decide whether the candidate can roll out.

    This is intentionally systems-first: we evaluate durability (retention proxy) as a guardrail
    rather than celebrating short-term engagement lift in isolation.
    """
    base = summarize_variant(baseline_cfg, embedding_version=embedding_version, model_version="rank_v1")
    cand = summarize_variant(candidate_cfg, embedding_version=embedding_version, model_version="rank_v2")

    decision = decide_rollout(
        baseline_retention=base.retention_proxy,
        candidate_retention=cand.retention_proxy,
        cohort_retention_baseline=base.cohort_retention,
        cohort_retention_candidate=cand.cohort_retention,
        cfg=guardrails,
    )

    return RolloutComparison(baseline=base, candidate=cand, decision=decision)


def default_candidate_cfg(baseline: SimConfig) -> SimConfig:
    """A convenience helper to create a candidate config that is "different" for demos.

    In real systems, the candidate differs by model weights/code; in DRPE we can also
    simulate behavioral shifts by changing retention dynamics.
    """
    return SimConfig(
        **{
            **baseline.__dict__,
            # simulate a worse durability outcome
            "retention_gain_per_depth": baseline.retention_gain_per_depth * 0.85,
            "retention_decay_per_day": baseline.retention_decay_per_day * 1.25,
        }
    )
