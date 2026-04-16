from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import torch

from drpe.data.simulator import SimConfig, recommend_top_k_from_matrices
from drpe.embeddings.io import load_embeddings
from drpe.evaluation.metrics import cohort_retention_means, engagement_depth_mean, retention_proxy_mean
from drpe.models.ranker import MultiObjectiveRanker, build_features
from drpe.rollout.guardrails import GuardrailConfig, GuardrailDecision, decide_rollout


@dataclass
class VariantStats:
    engagement_depth: float
    retention_proxy: float
    cohort_retention: Dict[str, float]


@dataclass
class RankerRolloutReport:
    baseline: VariantStats
    candidate: VariantStats
    decision: GuardrailDecision


@torch.no_grad()
def rerank_topk(
    *,
    ranker: MultiObjectiveRanker,
    cohort: str,
    topk: list[tuple[int, float, float]],
    items_quality: np.ndarray,
    items_popularity: np.ndarray,
    gamma_retention: float = 0.25,
) -> list[tuple[int, float, float]]:
    """Re-rank candidates using a learned ranker.

    Composite score:
      sigmoid(engagement_logit) + gamma * retention_pred

    Keep it simple, explainable, and production-adjacent.
    """
    X = []
    for rank, (item_id, score, affinity) in enumerate(topk, start=1):
        X.append(
            build_features(
                score=score,
                affinity=affinity,
                rank=rank,
                item_quality=float(items_quality[item_id]),
                item_popularity=float(items_popularity[item_id]),
                cohort=cohort,
            )
        )

    x = torch.tensor(np.stack(X), dtype=torch.float32)
    e_logit, r_pred = ranker(x)
    composite = torch.sigmoid(e_logit) + gamma_retention * r_pred

    order = torch.argsort(composite, descending=True).tolist()
    return [topk[i] for i in order]


def compare_rankers_for_rollout(
    *,
    embeddings_path: str,
    baseline_ranker: MultiObjectiveRanker,
    candidate_ranker: MultiObjectiveRanker,
    cfg: SimConfig,
    guardrails: GuardrailConfig = GuardrailConfig(),
    gamma_retention: float = 0.25,
    retention_fatigue_penalty: float = 0.0,
    candidate_retention_bias: float = 0.0,
) -> RankerRolloutReport:
    """Compare baseline vs candidate ranker on the same embeddings.

    This isolates ranking-layer changes while holding retrieval embeddings constant.

    retention_fatigue_penalty models a realistic failure mode: aggressive engagement
    optimization increases fatigue and reduces durability.

    candidate_retention_bias is a deterministic durability hit applied only to the
    candidate path to guarantee a guardrail-breaching example for demos.
    """

    users, items = load_embeddings(embeddings_path)

    cfg = SimConfig(
        **{
            **cfg.__dict__,
            "num_users": int(users.shape[0]),
            "num_items": int(items.shape[0]),
            "embedding_dim": int(users.shape[1]),
        }
    )

    rng = np.random.default_rng(cfg.seed)
    items_quality = rng.uniform(0.3, 1.0, cfg.num_items).astype(np.float32)
    items_pop = rng.beta(2, 8, cfg.num_items).astype(np.float32)

    from drpe.data.simulator import generate_world, UserState, _sigmoid
    from datetime import timedelta
    from drpe.data.schemas import SessionSummary

    rng2 = np.random.default_rng(cfg.seed)
    users_state, _ = generate_world(cfg)

    def run_with_ranker(ranker: MultiObjectiveRanker, model_version: str, *, apply_bias: bool) -> VariantStats:
        local_users = [UserState(**u.__dict__) for u in users_state]
        all_summaries: list[SessionSummary] = []

        for u in local_users:
            u.embed = users[u.user_id].astype(np.float32)
            for s in range(cfg.sessions_per_user):
                topk = recommend_top_k_from_matrices(
                    user_id=u.user_id,
                    true_pref=u.true_pref,
                    user_embed=u.embed,
                    items_vec=items,
                    items_quality=items_quality,
                    items_popularity=items_pop,
                    k=cfg.k,
                )
                topk = rerank_topk(
                    ranker=ranker,
                    cohort=u.cohort,
                    topk=topk,
                    items_quality=items_quality,
                    items_popularity=items_pop,
                    gamma_retention=gamma_retention,
                )

                now = u.last_active + timedelta(hours=int(rng2.integers(6, 72)))
                depth = 0.0

                for rank, (item_id, score, affinity) in enumerate(topk, start=1):
                    play_p = _sigmoid(2.0 * affinity - 0.8 * u.fatigue)
                    complete_p = _sigmoid(1.6 * affinity - 0.6 * u.fatigue)
                    played = rng2.random() < play_p
                    if played:
                        depth += 1.0
                        u.fatigue = float(min(1.0, u.fatigue + cfg.fatigue_gain_per_play))
                        if rng2.random() < complete_p:
                            depth += 0.5

                days_since = max(0.0, (now - u.last_active).total_seconds() / 86400.0)
                retention = (
                    u.base_return_prob
                    + cfg.retention_gain_per_depth * depth
                    - cfg.retention_decay_per_day * days_since
                    - retention_fatigue_penalty * u.fatigue
                    - (candidate_retention_bias if apply_bias else 0.0)
                )
                retention = float(np.clip(retention, 0.0, 1.0))

                all_summaries.append(
                    SessionSummary(
                        session_id=f"u{u.user_id}-s{s}-{int(now.timestamp())}",
                        user_id=u.user_id,
                        cohort=u.cohort,
                        started_at=now,
                        ended_at=now + timedelta(minutes=10),
                        k=cfg.k,
                        engagement_depth=float(depth),
                        retention_proxy=retention,
                        embedding_version="emb_v1",
                        model_version=model_version,
                    )
                )

                u.last_active = now
                u.fatigue = float(max(0.0, u.fatigue - cfg.fatigue_recovery_per_day * days_since))

        return VariantStats(
            engagement_depth=engagement_depth_mean(all_summaries),
            retention_proxy=retention_proxy_mean(all_summaries),
            cohort_retention=cohort_retention_means(all_summaries),
        )

    base = run_with_ranker(baseline_ranker, "ranker_v1", apply_bias=False)
    cand = run_with_ranker(candidate_ranker, "ranker_v2", apply_bias=True)

    decision = decide_rollout(
        baseline_retention=base.retention_proxy,
        candidate_retention=cand.retention_proxy,
        cohort_retention_baseline=base.cohort_retention,
        cohort_retention_candidate=cand.cohort_retention,
        cfg=guardrails,
    )

    return RankerRolloutReport(baseline=base, candidate=cand, decision=decision)
