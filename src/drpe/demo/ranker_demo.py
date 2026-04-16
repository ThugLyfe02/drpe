from __future__ import annotations

import argparse

from drpe.data.simulator import SimConfig
from drpe.reporting.model_card import ranker_rollout_card
from drpe.rollout.guardrails import GuardrailConfig
from drpe.rollout.ranker_rollout import compare_rankers_for_rollout
from drpe.models.ranker_io import load_ranker


def main() -> None:
    p = argparse.ArgumentParser(description="DRPE ranker rollout demo (safe vs risky candidate gated by durability)")
    p.add_argument("--emb", default="artifacts/embeddings_for_ranker.npz")
    p.add_argument("--ranker-v1", default="artifacts/ranker_v1.pt")
    p.add_argument("--ranker-v2", default="artifacts/ranker_v2.pt")

    p.add_argument("--users", type=int, default=200)
    p.add_argument("--items", type=int, default=800)
    p.add_argument("--sessions", type=int, default=2)
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--embed-dim", dest="embedding_dim", type=int, default=32)

    p.add_argument("--max-ret-drop", type=float, default=0.01)

    p.add_argument("--mode", choices=["safe", "risky"], default="safe")
    args = p.parse_args()

    cfg = SimConfig(
        seed=9,
        embedding_dim=args.embedding_dim,
        num_users=args.users,
        num_items=args.items,
        sessions_per_user=args.sessions,
        k=args.k,
    )

    base = load_ranker(args.ranker_v1)
    cand = load_ranker(args.ranker_v2)

    # Defaults are tuned for storytelling reliability:
    # - safe mode is typically allowed under a 1% retention drop guardrail
    # - risky mode is designed to trip the guardrail via fatigue-driven durability loss
    if args.mode == "safe":
        gamma = 0.25
        pen = 0.00
    else:
        gamma = 0.02
        pen = 0.65

    rep = compare_rankers_for_rollout(
        embeddings_path=args.emb,
        baseline_ranker=base,
        candidate_ranker=cand,
        cfg=cfg,
        guardrails=GuardrailConfig(max_retention_drop=args.max_ret_drop),
        gamma_retention=gamma,
        retention_fatigue_penalty=pen,
    )

    card = ranker_rollout_card(
        mode=args.mode,
        baseline_depth=rep.baseline.engagement_depth,
        baseline_ret=rep.baseline.retention_proxy,
        candidate_depth=rep.candidate.engagement_depth,
        candidate_ret=rep.candidate.retention_proxy,
        decision_allow=rep.decision.allow_rollout,
        decision_reason=rep.decision.reason,
        max_ret_drop=args.max_ret_drop,
        gamma_retention=gamma,
        retention_fatigue_penalty_candidate=pen,
    )

    print(card.render())


if __name__ == "__main__":
    main()
