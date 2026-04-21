from __future__ import annotations

import argparse
import numpy as np

from drpe.data.simulator import SimConfig
from drpe.models.ranker_io import load_ranker
from drpe.reporting.export import CardHeader, render_with_header, write_card
from drpe.reporting.model_card import ranker_rollout_card
from drpe.rollout.guardrails import GuardrailConfig
from drpe.rollout.ranker_rollout import compare_rankers_for_rollout

# RecSysOps
from drpe.recsysops.cold_start import ColdStartSignals
from drpe.recsysops.integration import ExportPaths, build_blocked_incident, export_incident, export_ops_note
from drpe.recsysops.ramp_integration import assess
from drpe.recsysops.trace_integration import maybe_emit_trace


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
    p.add_argument("--export", action="store_true", help="write model card to artifacts/model_cards")

    # Ops knobs
    p.add_argument("--emit-ops", action="store_true", help="emit ops artifacts (incidents/traces/ops notes)")

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

    if args.mode == "safe":
        gamma = 0.25
        pen = 0.00
        bias = 0.0
    else:
        # deterministic “block” example
        gamma = 0.02
        pen = 0.65
        bias = 0.02

    rep = compare_rankers_for_rollout(
        embeddings_path=args.emb,
        baseline_ranker=base,
        candidate_ranker=cand,
        cfg=cfg,
        guardrails=GuardrailConfig(max_retention_drop=args.max_ret_drop),
        gamma_retention=gamma,
        retention_fatigue_penalty=pen,
        candidate_retention_bias=bias,
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

    # Export model card
    if args.export:
        header = CardHeader(
            kind="ranker_rollout",
            version_left="ranker_v1",
            version_right=f"ranker_v2({args.mode})",
            thresholds={"max_retention_drop": f"{args.max_ret_drop:.4f}"},
            notes=f"gamma_retention={gamma:.3f}, fatigue_penalty={pen:.3f}, candidate_bias={bias:.3f}",
        )
        content = render_with_header(card, header)
        out = write_card(f"artifacts/model_cards/ranker_{args.mode}.txt", content)
        print(f"\n[exported] {out}")

    # RecSysOps: cold-start ramp note (only when asked)
    if args.emit_ops:
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

        # RecSysOps: incident + trace when blocked
        if not rep.decision.allow_rollout:
            # minimal privacy-safe trace sample
            item_ids = np.arange(100, dtype=np.int64)
            scores = np.linspace(1.0, 0.0, 100, dtype=np.float32)

            trace_id = maybe_emit_trace(
                user_id=0,
                session_id="demo",
                model_version=f"ranker_v2({args.mode})",
                cohort="core",
                feature_version="ranker_features_v1",
                item_ids=item_ids,
                scores=scores,
            )

            inc = build_blocked_incident(
                title="Ranker rollout blocked by durability guardrail",
                summary=rep.decision.reason,
                baseline=rep.baseline.retention_proxy,
                current=rep.candidate.retention_proxy,
                threshold=rep.baseline.retention_proxy * (1.0 - args.max_ret_drop),
                affected_surfaces=["home_feed"],
                affected_cohorts=["new", "core", "power"],
                trace_id=trace_id,
                suspected_causes=[
                    "ranking tradeoff shifted toward engagement",
                    "fatigue proxy increased",
                    "candidate durability bias applied (demo)",
                ],
            )
            out = export_incident(inc, paths=ExportPaths())
            print(f"[recsysops] exported incident: {out}")


if __name__ == "__main__":
    main()
