from __future__ import annotations

from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, Field

EventType = Literal["impression", "click", "play", "complete", "skip", "save"]


class Event(BaseModel):
    ts: datetime
    user_id: int
    item_id: int
    event_type: EventType
    rank: int = Field(ge=1)
    score: float  # model score used for ranking (baseline or learned)
    affinity: float  # simulator-side "true" affinity proxy
    session_id: str
    cohort: str  # e.g. new/core/power
    embedding_version: str
    model_version: str


class SessionSummary(BaseModel):
    session_id: str
    user_id: int
    cohort: str
    started_at: datetime
    ended_at: datetime
    k: int
    engagement_depth: float
    retention_proxy: float
    embedding_version: str
    model_version: str
