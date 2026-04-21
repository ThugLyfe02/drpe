from __future__ import annotations

import argparse
import numpy as np

from drpe.data.simulator import SimConfig
from drpe.reporting.export import CardHeader, render_with_header, write_card
from drpe.reporting.model_card import embedding_rollout_card
from drpe.rollout.guardrails import GuardrailConfig
from drpe.rollout.rollout_from_artifacts import compare_embedding_artifacts
from drpe.training.train_two_tower import train, TrainConfig

# RecSysOps
from drpe.recsysops.cold_start import ColdStartSignals
from drpe.recsysops.integration import ExportPaths, build_blocked_incident, export_incident, export_ops_note
from drpe.recsysops.ramp_integration import assess
from drpe.recsysops.trace_integration import maybe_emit_trace
from drpe.recsysops.trace_sampler import TraceSampleConfig


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
    p.add_argument("--export", action="store_true", help="write model card to artifacts/model_cards")

    # Ops wiring
    p.add_argument("--emit-ops", action="store_true", help="emit ops artifacts (incidents/traces/ops notes)")

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
    guardrails = GuardrailConfig(
        max_retention_drop=args.max_ret_drop,
        max_embedding_mean_cosine_shift=args.max_emb_drift,
        max_embedding_mean_cosine_shift_per_cohort=args.max_emb_drift,
    )

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
        guardrails=guardrails,
    )

    card = embedding_rollout_card(
        baseline_depth=report.baseline.engagement_depth,
        baseline_ret=report.baseline.retention_proxy,
        candidate_depth=report.candidate.engagement_depth,
        candidate_ret=report.candidate.retention_proxy,
        depth_kl=report.depth_kl,
        ret_kl=report.retention_kl,
        emb_users_mean=report.geom_users_mean,
        emb_items_mean=report.geom_items_mean,
        cohort_user_drift=report.cohort_user_mean_shift,
        decision_allow=report.decision.allow_rollout,
        decision_reason=report.decision.reason,
        guardrail_max_ret_drop=guardrails.max_retention_drop,
        guardrail_max_emb_mean=guardrails.max_embedding_mean_cosine_shift,
    )

    print(card.render())

    if args.export:
        header = CardHeader(
            kind="embedding_rollout",
            version_left="emb_v1",
            version_right="emb_v2",
            thresholds={
                "max_retention_drop": f"{guardrails.max_retention_drop:.4f}",
                "max_embedding_mean_cosine_shift": (
                    f"{guardrails.max_embedding_mean_cosine_shift:.4f}"
                    if guardrails.max_embedding_mean_cosine_shift is not None
                    else "(none)"
                ),
                "max_embedding_mean_cosine_shift_per_cohort": (
                    f"{guardrails.max_embedding_mean_cosine_shift_per_cohort:.4f}"
                    if guardrails.max_embedding_mean_cosine_shift_per_cohort is not None
                    else "(none)"
                ),
            },
        )
        content = render_with_header(card, header)
        out = write_card("artifacts/model_cards/embedding_rollout.txt", content)
        print(f"\n[exported] {out}")

    # RecSysOps wiring (only when explicitly requested)
    if args.emit_ops:
        # Cold-start ramp note (demo signals)
        sig = ColdStartSignals(
            prior_quality=0.35,
            metadata_confidence=0.50,
            early_ctr=0.03,
            early_completion=0.20,
        )
        cs = assess(sig)
        if cs.risk >= 0.45:
            out = export_ops_note(
                "cold_start_ramp",
                {
                    "risk": cs.risk,
                    "rationale": cs.rationale,
                    "ramp": {
                        "stage": cs.stage,
                        "traffic_pct": cs.traffic_pct,
                        "stop_conditions": cs.stop_conditions,
                    },
                },
                paths=ExportPaths(),
            )
            print(f"[recsysops] exported cold-start ramp note: {out}")

        # If blocked, emit incident + a forced trace signature
        if not report.decision.allow_rollout:
            # Synthetic top-k score trace (privacy-safe)
            item_ids = np.arange(100, dtype=np.int64)
            scores = np.linspace(1.0, 0.0, 100, dtype=np.float32)

            trace_id = maybe_emit_trace(
                user_id=0,
                session_id="demo",
                model_version="emb_v2",
                cohort="core",
                feature_version="embedding_features_v1",
                item_ids=item_ids,
                scores=scores,
                cfg=TraceSampleConfig(sample_rate=1.0),
            )

            inc = build_blocked_incident(
                title="Embedding rollout blocked by durability/drift guardrails",
                summary=report.decision.reason,
                metric_name="retention_proxy",
                baseline=report.baseline.retention_proxy,
                current=report.candidate.retention_proxy,
                threshold=report.baseline.retention_proxy * (1.0 - guardrails.max_retention_drop),
                affected_surfaces=["home_feed"],
                affected_cohorts=["new", "core", "power"],
                trace_id=trace_id,
                suspected_causes=["embedding geometry drift", "distribution drift"],
            )
            out = export_incident(inc, paths=ExportPaths())
            print(f"[recsysops] exported incident: {out}")


if __name__ == "__main__":
    main()
