from __future__ import annotations

from collections import defaultdict
from typing import Dict, List

from drpe.data.schemas import SessionSummary


def engagement_depth_mean(summaries: List[SessionSummary]) -> float:
    if not summaries:
        return 0.0
    return sum(s.engagement_depth for s in summaries) / len(summaries)


def retention_proxy_mean(summaries: List[SessionSummary]) -> float:
    if not summaries:
        return 0.0
    return sum(s.retention_proxy for s in summaries) / len(summaries)


def cohort_retention_means(summaries: List[SessionSummary]) -> Dict[str, float]:
    buckets = defaultdict(list)
    for s in summaries:
        buckets[s.cohort].append(s.retention_proxy)
    return {c: sum(v) / len(v) for c, v in buckets.items()}
