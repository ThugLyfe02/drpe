from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np

from drpe.data.simulator import SimConfig, run_simulation_with_embeddings
from drpe.drift.embedding_geometry import build_geometry_drift_report
from drpe.drift.drift import histogram_kl
from drpe.embeddings.io import load_embeddings
from drpe.evaluation.metrics import cohort_retention_means, engagement_depth_mean, retention_proxy_mean
from drpe.rollout.guardrails import GuardrailConfig, GuardrailDecision, decide_rollout


@dataclass
class VariantStats:
    engagement_depth: float
    retention_proxy: float
    cohort_retention: Dict[str, float]


@dataclass
class ArtifactRolloutReport:
    baseline: VariantStats
    candidate: VariantStats
    depth_kl: float
    retention_kl: float
    geom_users_mean: float
    geom_items_mean: float
    decision: GuardrailDecision


def _summarize(cfg: SimConfig, *, users: np.ndarray, items: np.ndarray, embedding_version: str, model_version: str) -> tuple[VariantStats, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(cfg.seed)
    items_quality = rng.uniform(0.3, 1.0, cfg.num_items).astype(np.float32)
    items_pop = rng.beta(2, 8, cfg.num_items).astype(np.float32)

    _, summaries = run_simulation_with_embeddings(
        cfg,
        users_embed=users,
        items_vec=items,
        items_quality=items_quality,
        items_popularity=items_pop,
        embedding_version=embedding_version,
        model_version=model_version,
    )

    ed = np.array([s.engagement_depth for s in summaries], dtype=np.float64)
    rp = np.array([s.retention_proxy for s in summaries], dtype=np.float64)

    return (
        VariantStats(
            engagement_depth=engagement_depth_mean(summaries),
            retention_proxy=retention_proxy_mean(summaries),
            cohort_retention=cohort_retention_means(summaries),
        ),
        ed,
        rp,
    )


def compare_embedding_artifacts(
    *,
    baseline_path: str,
    candidate_path: str,
    cfg: SimConfig,
    guardrails: GuardrailConfig = GuardrailConfig(),
) -> ArtifactRolloutReport:
    base_users, base_items = load_embeddings(baseline_path)
    cand_users, cand_items = load_embeddings(candidate_path)

    base_stats, base_ed, base_rp = _summarize(cfg, users=base_users, items=base_items, embedding_version="emb_v1", model_version="rank_v1")
    cand_stats, cand_ed, cand_rp = _summarize(cfg, users=cand_users, items=cand_items, embedding_version="emb_v2", model_version="rank_v2")

    depth_kl = histogram_kl(base_ed, cand_ed, bins=40)
    ret_kl = histogram_kl(base_rp, cand_rp, bins=40)

    # geometry drift uses aligned embeddings
    # cohorts are unknown here, so omit cohort breakdown (use overall mean drift)
    geom = build_geometry_drift_report(
        users_v1=base_users,
        users_v2=cand_users,
        items_v1=base_items,
        items_v2=cand_items,
        user_cohorts={i: "all" for i in range(cfg.num_users)},
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

    return ArtifactRolloutReport(
        baseline=base_stats,
        candidate=cand_stats,
        depth_kl=depth_kl,
        retention_kl=ret_kl,
        geom_users_mean=geom.users.mean_cosine_shift,
        geom_items_mean=geom.items.mean_cosine_shift,
        decision=decision,
    )
