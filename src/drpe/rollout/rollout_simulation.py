from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np

from drpe.data.simulator import (
    SimConfig,
    run_simulation_with_embeddings,
)
from drpe.embeddings.versioning import generate_embedding_versions
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


def summarize_variant_with_embeddings(
    cfg: SimConfig,
    *,
    users_embed: np.ndarray,
    items_vec: np.ndarray,
    items_quality: np.ndarray,
    items_popularity: np.ndarray,
    embedding_version: str,
    model_version: str,
) -> VariantSummary:
    _, summaries = run_simulation_with_embeddings(
        cfg,
        users_embed=users_embed,
        items_vec=items_vec,
        items_quality=items_quality,
        items_popularity=items_popularity,
        embedding_version=embedding_version,
        model_version=model_version,
    )
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
) -> RolloutComparison:
    """Compare baseline vs candidate using explicit embedding versions.

    Baseline uses v1 embeddings; candidate uses v2 embeddings. This makes the
    embedding geometry drift meaningful within the simulator.
    """

    emb = generate_embedding_versions(
        seed=baseline_cfg.seed,
        embedding_dim=baseline_cfg.embedding_dim,
        num_users=baseline_cfg.num_users,
        num_items=baseline_cfg.num_items,
        drift_strength=baseline_cfg.drift_strength,
    )

    # Use simulator item properties (quality/popularity) as additional fixed arrays.
    # For v0 we reuse simple random draws that are consistent across both variants.
    rng = np.random.default_rng(baseline_cfg.seed)
    items_quality = rng.uniform(0.3, 1.0, baseline_cfg.num_items).astype(np.float32)
    items_pop = rng.beta(2, 8, baseline_cfg.num_items).astype(np.float32)

    base = summarize_variant_with_embeddings(
        baseline_cfg,
        users_embed=emb.users_v1,
        items_vec=emb.items_v1,
        items_quality=items_quality,
        items_popularity=items_pop,
        embedding_version="emb_v1",
        model_version="rank_v1",
    )

    cand = summarize_variant_with_embeddings(
        candidate_cfg,
        users_embed=emb.users_v2,
        items_vec=emb.items_v2,
        items_quality=items_quality,
        items_popularity=items_pop,
        embedding_version="emb_v2",
        model_version="rank_v2",
    )

    decision = decide_rollout(
        baseline_retention=base.retention_proxy,
        candidate_retention=cand.retention_proxy,
        cohort_retention_baseline=base.cohort_retention,
        cohort_retention_candidate=cand.cohort_retention,
        cfg=guardrails,
    )

    return RolloutComparison(baseline=base, candidate=cand, decision=decision)


def default_candidate_cfg(baseline: SimConfig) -> SimConfig:
    """A convenience helper to create a candidate config that is "different" for demos."""
    return SimConfig(
        **{
            **baseline.__dict__,
            "retention_gain_per_depth": baseline.retention_gain_per_depth * 0.85,
            "retention_decay_per_day": baseline.retention_decay_per_day * 1.25,
        }
    )
