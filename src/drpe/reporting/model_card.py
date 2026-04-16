from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional


def _pct(x: float) -> str:
    return f"{x*100:.2f}%"


def _delta(a: float, b: float) -> float:
    return b - a


def _reldelta(a: float, b: float) -> float:
    if a == 0:
        return 0.0
    return (b - a) / a


def _fmt_map(m: Dict[str, float], digits: int = 4) -> str:
    if not m:
        return "(none)"
    parts = [f"{k}={m[k]:.{digits}f}" for k in sorted(m.keys())]
    return ", ".join(parts)


@dataclass
class Card:
    title: str
    body: str

    def render(self) -> str:
        bar = "=" * 72
        return f"{bar}\n{self.title}\n{bar}\n{self.body}\n"


def embedding_rollout_card(
    *,
    baseline_depth: float,
    baseline_ret: float,
    candidate_depth: float,
    candidate_ret: float,
    depth_kl: float,
    ret_kl: float,
    emb_users_mean: float,
    emb_items_mean: float,
    cohort_user_drift: Dict[str, float],
    decision_allow: bool,
    decision_reason: str,
    guardrail_max_ret_drop: float,
    guardrail_max_emb_mean: Optional[float],
) -> Card:
    dd = _delta(baseline_depth, candidate_depth)
    dr = _delta(baseline_ret, candidate_ret)
    rr = _reldelta(baseline_ret, candidate_ret)

    body = (
        f"Baseline:   depth={baseline_depth:.3f}  retention={baseline_ret:.3f}\n"
        f"Candidate:  depth={candidate_depth:.3f}  retention={candidate_ret:.3f}\n"
        f"Lift:       depth={dd:+.3f}  retention={dr:+.3f} (rel {_pct(rr)})\n\n"
        f"Drift (distribution): KL(depth)={depth_kl:.4f}  KL(ret)={ret_kl:.4f}\n"
        f"Drift (geometry):     emb_mean(users)={emb_users_mean:.4f}  emb_mean(items)={emb_items_mean:.4f}\n"
        f"Cohort drift(users):  {_fmt_map(cohort_user_drift)}\n\n"
        f"Guardrails: max_ret_drop={_pct(guardrail_max_ret_drop)}"
        + (f"  max_emb_mean={guardrail_max_emb_mean:.4f}" if guardrail_max_emb_mean is not None else "")
        + "\n"
        f"Decision:   {str(decision_allow).upper()}  — {decision_reason}\n"
    )

    return Card(title="MODEL CARD — Embedding Rollout Gate", body=body)


def ranker_rollout_card(
    *,
    mode: str,
    baseline_depth: float,
    baseline_ret: float,
    candidate_depth: float,
    candidate_ret: float,
    decision_allow: bool,
    decision_reason: str,
    max_ret_drop: float,
    gamma_retention: float,
    retention_fatigue_penalty_candidate: float,
) -> Card:
    dd = _delta(baseline_depth, candidate_depth)
    dr = _delta(baseline_ret, candidate_ret)
    rr = _reldelta(baseline_ret, candidate_ret)

    body = (
        f"Mode:       {mode}\n"
        f"Baseline:   depth={baseline_depth:.3f}  retention={baseline_ret:.3f}\n"
        f"Candidate:  depth={candidate_depth:.3f}  retention={candidate_ret:.3f}\n"
        f"Lift:       depth={dd:+.3f}  retention={dr:+.3f} (rel {_pct(rr)})\n\n"
        f"Re-rank policy: gamma_retention={gamma_retention:.3f}  fatigue_penalty(candidate)={retention_fatigue_penalty_candidate:.3f}\n"
        f"Guardrail:   max_ret_drop={_pct(max_ret_drop)}\n"
        f"Decision:    {str(decision_allow).upper()}  — {decision_reason}\n"
    )

    return Card(title="MODEL CARD — Ranker Rollout Gate", body=body)
