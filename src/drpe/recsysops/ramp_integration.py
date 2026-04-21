from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

from drpe.recsysops.cold_start import ColdStartSignals, risk_score, ramp_policy


@dataclass
class ColdStartAssessment:
    """Cold-start risk assessment + recommended safe ramp."""

    risk: float
    rationale: Dict[str, float]
    stage: str
    traffic_pct: float
    stop_conditions: Dict[str, str]


def assess(sig: ColdStartSignals) -> ColdStartAssessment:
    """Assess cold-start risk and produce a ramp plan."""
    r = risk_score(sig)
    rp = ramp_policy(r)
    return ColdStartAssessment(
        risk=r.risk,
        rationale=r.rationale,
        stage=rp.stage,
        traffic_pct=rp.traffic_pct,
        stop_conditions=rp.stop_conditions,
    )
