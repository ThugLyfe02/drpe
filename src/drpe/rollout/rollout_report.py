from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np

from drpe.data.simulator import SimConfig, run_simulation
from drpe.drift.drift import cohort_variance, histogram_kl
from drpe.drift.embedding_geometry import GeometryDriftReport, build_geometry_drift_report
from drpe.embeddings.versioning import generate_embedding_versions
from drpe.evaluation.metrics import cohort_retention_means, engagement_depth_mean, retention_proxy_mean
from drpe.rollout.guardrails import GuardrailConfig, GuardrailDecision, decide_rollout


@dataclass
class VariantStats:
    engagement_depth: float
    retention_proxy: float
    cohort_retention: Dict[str, float]


@dataclass
class DriftStats:
    engagement_depth_kl: float
    retention_proxy_kl: float
    cohort_retention_variance_baseline: float
    cohort_retention_variance_candidate: float
    embedding_geometry: GeometryDriftReport


@dataclass
class RolloutReport:
    baseline: VariantStats
    candidate: VariantStats
    drift: DriftStats
    decision: GuardrailDecision


def _summarize(cfg: SimConfig, *, embedding_version: str, model_version: str) -> tuple[VariantStats, np.ndarray, np.ndarray]:
    _, summaries = run_simulation(cfg, embedding_version=embedding_version, model_version=model_version)
    ed = np.array([s.engagement_depth for s in summaries], dtype=np.float64)
    rp = np.array([s.retention_proxy for s in summaries], dtype=np.float64)

    cohort_ret = cohort_retention_means(summaries)

    return (
        VariantStats(
            engagement_depth=engagement_depth_mean(summaries),
            retention_proxy=retention_proxy_mean(summaries),
            cohort_retention=cohort_ret,
        ),
        ed,
        rp,
    )


def build_rollout_report(
    *,
    baseline_cfg: SimConfig,
    candidate_cfg: SimConfig,
    guardrails: GuardrailConfig = GuardrailConfig(),
    embedding_version: str = "emb_v1",
) -> RolloutReport:
    """Compare baseline vs candidate with durability guardrails + drift signals.

    Drift signals include:
    - distribution shift (KL) for engagement depth and retention proxy
    - cohort stability variance
    - embedding geometry drift (mean/p95 cosine shift for users/items + cohort breakdown)

    Embedding geometry drift is modeled via aligned v1/v2 embedding matrices.
    """

    base_stats, base_ed, base_rp = _summarize(baseline_cfg, embedding_version=embedding_version, model_version="rank_v1")
    cand_stats, cand_ed, cand_rp = _summarize(candidate_cfg, embedding_version=embedding_version, model_version="rank_v2")

    # Embedding geometry drift (users + items). This is intentionally synthetic in v0.
    emb = generate_embedding_versions(
        seed=baseline_cfg.seed,
        embedding_dim=baseline_cfg.embedding_dim,
        num_users=baseline_cfg.num_users,
        num_items=baseline_cfg.num_items,
        drift_strength=baseline_cfg.drift_strength,
    )
    geom = build_geometry_drift_report(
        users_v1=emb.users_v1,
        users_v2=emb.users_v2,
        items_v1=emb.items_v1,
        items_v2=emb.items_v2,
        user_cohorts=emb.user_cohorts,
    )

    drift = DriftStats(
        engagement_depth_kl=histogram_kl(base_ed, cand_ed, bins=40),
        retention_proxy_kl=histogram_kl(base_rp, cand_rp, bins=40),
        cohort_retention_variance_baseline=cohort_variance({k: [v] for k, v in base_stats.cohort_retention.items()}),
        cohort_retention_variance_candidate=cohort_variance({k: [v] for k, v in cand_stats.cohort_retention.items()}),
        embedding_geometry=geom,
    )

    decision = decide_rollout(
        baseline_retention=base_stats.retention_proxy,
        candidate_retention=cand_stats.retention_proxy,
        cohort_retention_baseline=base_stats.cohort_retention,
        cohort_retention_candidate=cand_stats.cohort_retention,
        embedding_mean_cosine_shift_users=geom.users.mean_cosine_shift,
        embedding_mean_cosine_shift_items=geom.items.mean_cosine_shift,
        cfg=guardrails,
    )

    return RolloutReport(baseline=base_stats, candidate=cand_stats, drift=drift, decision=decision)
