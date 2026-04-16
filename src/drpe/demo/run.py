from __future__ import annotations

import argparse

from drpe.data.simulator import SimConfig
from drpe.rollout.guardrails import GuardrailConfig
from drpe.rollout.rollout_from_artifacts import compare_embedding_artifacts
from drpe.training.train_two_tower import train, TrainConfig


def main() -> None:
    p = argparse.ArgumentParser(description="DRPE end-to-end demo: train v1/v2 embeddings, then run rollout gate")
    p.add_argument("--users", type=int, default=300)
    p.add_argument("--items", type=int, default=1500)
    p.add_argument("--sessions", type=int, default=2)
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--embed-dim", dest="embedding_dim", type=int, default=32)
    p.add_argument("--dim", type=int, default=64)
    p.add_argument("--epochs-v1", type=int, default=2)
    p.add_argument("--epochs-v2", type=int, default=1)
    p.add_argument("--lr-v1", type=float, default=1e-2)
    p.add_argument("--lr-v2", type=float, default=5e-3)
    p.add_argument("--seed-v1", type=int, default=7)
    p.add_argument("--seed-v2", type=int, default=8)
    p.add_argument("--max-ret-drop", type=float, default=0.01)
    p.add_argument("--max-emb-drift", type=float, default=0.12)
    args = p.parse_args()

    sim_v1 = SimConfig(
        seed=args.seed_v1,
        embedding_dim=args.embedding_dim,
        num_users=args.users,
        num_items=args.items,
        sessions_per_user=args.sessions,
        k=args.k,
    )

    sim_v2 = SimConfig(
        seed=args.seed_v2,
        embedding_dim=args.embedding_dim,
        num_users=args.users,
        num_items=args.items,
        sessions_per_user=args.sessions,
        k=args.k,
    )

    v1_path = "artifacts/emb_v1.npz"
    v2_path = "artifacts/emb_v2.npz"

    print("[1/3] train v1 embeddings...")
    train(
        TrainConfig(
            sim=sim_v1,
            dim=args.dim,
            epochs=args.epochs_v1,
            lr=args.lr_v1,
            seed=args.seed_v1,
            out_path=v1_path,
        )
    )

    print("[2/3] train v2 embeddings (warm-start from v1)...")
    train(
        TrainConfig(
            sim=sim_v2,
            dim=args.dim,
            epochs=args.epochs_v2,
            lr=args.lr_v2,
            seed=args.seed_v2,
            warm_start_path=v1_path,
            out_path=v2_path,
        )
    )

    print("[3/3] rollout gate (durability + drift)...")
    report = compare_embedding_artifacts(
        baseline_path=v1_path,
        candidate_path=v2_path,
        cfg=SimConfig(
            seed=9,
            embedding_dim=args.embedding_dim,
            num_users=args.users,
            num_items=args.items,
            sessions_per_user=args.sessions,
            k=args.k,
        ),
        guardrails=GuardrailConfig(
            max_retention_drop=args.max_ret_drop,
            max_embedding_mean_cosine_shift=args.max_emb_drift,
            max_embedding_mean_cosine_shift_per_cohort=args.max_emb_drift,
        ),
    )

    print(f"Baseline: depth={report.baseline.engagement_depth:.3f} ret={report.baseline.retention_proxy:.3f}")
    print(f"Candidate: depth={report.candidate.engagement_depth:.3f} ret={report.candidate.retention_proxy:.3f}")
    print(f"KL(depth,ret)= {report.depth_kl:.4f} {report.retention_kl:.4f}")
    print(f"EmbDrift mean users/items= {report.geom_users_mean:.4f} {report.geom_items_mean:.4f}")
    if report.cohort_user_mean_shift:
        print("Cohort user mean drift:")
        for c, v in sorted(report.cohort_user_mean_shift.items()):
            print(f"  {c}: {v:.4f}")
    print(f"Decision: {report.decision.allow_rollout} - {report.decision.reason}")


if __name__ == "__main__":
    main()
