from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Literal, Tuple


@dataclass
class ColdStartSignals:
    """Signals available *before* or *right at* launch.

    This is intentionally generic (Netflix/Spotify-style):
    - priors from similar titles/artists
    - early engagement quality from small cohorts
    - confidence of metadata
    """

    prior_quality: float  # 0..1
    metadata_confidence: float  # 0..1
    early_ctr: float  # 0..1
    early_completion: float  # 0..1


@dataclass
class ColdStartRisk:
    risk: float  # 0..1 (higher = riskier)
    rationale: Dict[str, float]


def risk_score(sig: ColdStartSignals) -> ColdStartRisk:
    """A simple risk score (not "good/bad" prediction).

    Higher risk means we should ramp slowly and watch guardrails.
    """
    # Risk increases when priors are weak, metadata is uncertain, and early quality is low.
    r_prior = 1.0 - sig.prior_quality
    r_meta = 1.0 - sig.metadata_confidence
    r_quality = 1.0 - (0.5 * sig.early_ctr + 0.5 * sig.early_completion)

    # weighted blend
    risk = 0.35 * r_prior + 0.25 * r_meta + 0.40 * r_quality
    risk = max(0.0, min(1.0, risk))

    return ColdStartRisk(
        risk=risk,
        rationale={
            "prior_gap": r_prior,
            "metadata_gap": r_meta,
            "early_quality_gap": r_quality,
        },
    )


RampStage = Literal["canary", "small", "medium", "full"]


@dataclass
class RampPlan:
    stage: RampStage
    traffic_pct: float
    stop_conditions: Dict[str, str]


def ramp_policy(risk: ColdStartRisk) -> RampPlan:
    """Convert risk into a safe ramp plan.

    This is the operational output: how to ship safely.
    """
    if risk.risk >= 0.75:
        return RampPlan(
            stage="canary",
            traffic_pct=1.0,
            stop_conditions={
                "retention_proxy_drop": "> 1%",
                "complaint_rate": "> baseline",
                "cohort_skew": "new users harmed",
            },
        )
    if risk.risk >= 0.45:
        return RampPlan(
            stage="small",
            traffic_pct=5.0,
            stop_conditions={
                "retention_proxy_drop": "> 1%",
                "repetition_rate": "> threshold",
            },
        )
    if risk.risk >= 0.20:
        return RampPlan(
            stage="medium",
            traffic_pct=20.0,
            stop_conditions={
                "retention_proxy_drop": "> 1%",
            },
        )

    return RampPlan(
        stage="full",
        traffic_pct=100.0,
        stop_conditions={
            "retention_proxy_drop": "> 1%",
        },
    )
