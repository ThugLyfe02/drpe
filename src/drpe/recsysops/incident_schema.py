from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


class Severity(str, Enum):
    sev0 = "sev0"  # user-impacting outage
    sev1 = "sev1"  # significant degradation
    sev2 = "sev2"  # partial degradation / cohort-specific
    sev3 = "sev3"  # warning / early signal


class IncidentStatus(str, Enum):
    open = "open"
    mitigated = "mitigated"
    resolved = "resolved"


class MetricBreach(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str  # e.g. retention_proxy, repetition_rate, p95_latency
    baseline: float
    current: float
    threshold: float
    direction: Literal["above", "below"]
    notes: Optional[str] = None


class ActionTaken(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action: str  # e.g. rollback_model, clamp_feature, reduce_traffic
    owner: Optional[str] = None
    ts: datetime = Field(default_factory=datetime.utcnow)
    details: Optional[Dict[str, str]] = None


class IncidentRecord(BaseModel):
    """A compact, shareable incident record schema.

    This is not a ticketing system. It's a *repeatable structure* that makes
    postmortems and automation easier.
    """

    model_config = ConfigDict(extra="forbid")

    incident_id: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    status: IncidentStatus = IncidentStatus.open
    severity: Severity

    title: str
    summary: str

    affected_surfaces: List[str] = Field(default_factory=list)  # e.g. home_feed, search, autoplay
    affected_cohorts: List[str] = Field(default_factory=list)  # e.g. new/core/power

    breached_metrics: List[MetricBreach] = Field(default_factory=list)

    suspected_causes: List[str] = Field(default_factory=list)
    trace_ids: List[str] = Field(default_factory=list)  # privacy-safe trace references

    mitigations: List[ActionTaken] = Field(default_factory=list)

    resolved_at: Optional[datetime] = None
    prevention_added: List[str] = Field(default_factory=list)  # e.g. "added guardrail: max_rep_rate"


def new_incident_id(prefix: str = "INC") -> str:
    """Generate a readable incident id without external dependencies."""
    ts = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    return f"{prefix}-{ts}"
