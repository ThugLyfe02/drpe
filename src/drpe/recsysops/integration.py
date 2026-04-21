from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from drpe.recsysops.incident_schema import (
    ActionTaken,
    IncidentRecord,
    IncidentStatus,
    MetricBreach,
    Severity,
    new_incident_id,
)


@dataclass
class ExportPaths:
    incidents_dir: str = "artifacts/incidents"
    traces_dir: str = "artifacts/traces"
    ops_dir: str = "artifacts/ops"


def _ensure_dir(p: str | Path) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


def export_json(path: str | Path, payload: Dict[str, Any]) -> Path:
    path = Path(path)
    _ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def build_blocked_incident(
    *,
    title: str,
    summary: str,
    severity: Severity = Severity.sev2,
    affected_surfaces: Optional[list[str]] = None,
    affected_cohorts: Optional[list[str]] = None,
    metric_name: str = "retention_proxy",
    baseline: float,
    current: float,
    threshold: float,
    direction: str = "below",
    trace_id: Optional[str] = None,
    suspected_causes: Optional[list[str]] = None,
) -> IncidentRecord:
    inc = IncidentRecord(
        incident_id=new_incident_id(),
        status=IncidentStatus.open,
        severity=severity,
        title=title,
        summary=summary,
        affected_surfaces=affected_surfaces or [],
        affected_cohorts=affected_cohorts or [],
        breached_metrics=[
            MetricBreach(
                name=metric_name,
                baseline=float(baseline),
                current=float(current),
                threshold=float(threshold),
                direction=direction,  # type: ignore[arg-type]
                notes="auto-emitted by DRPE demo",
            )
        ],
        suspected_causes=suspected_causes or [],
        trace_ids=[trace_id] if trace_id else [],
        mitigations=[
            ActionTaken(
                action="block_rollout",
                owner="drpe-demo",
                details={"reason": "guardrail breach"},
            )
        ],
        prevention_added=["add guardrail + alert", "add canary + rollback hook"],
    )
    return inc


def export_incident(incident: IncidentRecord, *, paths: ExportPaths = ExportPaths()) -> Path:
    d = _ensure_dir(paths.incidents_dir)
    out = d / f"{incident.incident_id}.json"
    return export_json(out, incident.model_dump())


def export_ops_note(name: str, payload: Dict[str, Any], *, paths: ExportPaths = ExportPaths()) -> Path:
    d = _ensure_dir(paths.ops_dir)
    ts = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    out = d / f"{name}-{ts}.json"
    return export_json(out, payload)
