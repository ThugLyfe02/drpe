from __future__ import annotations

import argparse

from drpe.data.simulator import SimConfig
from drpe.rollout.guardrails import GuardrailConfig
from drpe.rollout.ranker_rollout import compare_rankers_for_rollout
from drpe.models.ranker_io import load_ranker


def main() -> None:
    p = argparse.ArgumentParser(description="DRPE ranker rollout demo (v1 vs v2 ranker gated by durability)")
    p.add_argument("--emb", default="artifacts/embeddings_for_ranker.npz")
    p.add_argument("--ranker-v1", default="artifacts/ranker_v1.pt")
    p.add_argument("--ranker-v2", default="artifacts/ranker_v2.pt")

    p.add_argument("--users", type=int, default=200)
    p.add_argument("--items", type=int, default=800)
    p.add_argument("--sessions", type=int, default=2)
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--embed-dim", dest="embedding_dim", type=int, default=32)

    p.add_argument("--max-ret-drop", type=float, default=0.01)
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

    rep = compare_rankers_for_rollout(
        embeddings_path=args.emb,
        baseline_ranker=base,
        candidate_ranker=cand,
        cfg=cfg,
        guardrails=GuardrailConfig(max_retention_drop=args.max_ret_drop),
    )

    print(f"Baseline: depth={rep.baseline.engagement_depth:.3f} ret={rep.baseline.retention_proxy:.3f}")
    print(f"Candidate: depth={rep.candidate.engagement_depth:.3f} ret={rep.candidate.retention_proxy:.3f}")
    print(f"Decision: {rep.decision.allow_rollout} - {rep.decision.reason}")


if __name__ == "__main__":
    main()
