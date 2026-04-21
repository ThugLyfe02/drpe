from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import numpy as np


def _sha256(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


@dataclass
class TraceSampleConfig:
    sample_rate: float = 0.001  # 0.1%
    seed: int = 7
    redact_keys: tuple[str, ...] = ("raw_user_id", "email", "ip", "device_id")


def should_sample(trace_key: str, cfg: TraceSampleConfig) -> bool:
    """Deterministic sampling by hashing a stable key.

    This avoids per-request RNG (and makes sampling reproducible).
    """
    h = int(_sha256(f"{cfg.seed}:{trace_key}")[:8], 16)
    bucket = (h % 1_000_000) / 1_000_000.0
    return bucket < cfg.sample_rate


def redact(payload: Dict[str, Any], cfg: TraceSampleConfig) -> Dict[str, Any]:
    out = dict(payload)
    for k in cfg.redact_keys:
        if k in out:
            out[k] = "***REDACTED***"
    return out


def make_trace_id(*, user_id: int, session_id: str, model_version: str) -> str:
    """Privacy-safe trace id (no raw identifiers)."""
    return _sha256(f"u={user_id}|s={session_id}|m={model_version}")[:16]


def build_rank_trace(
    *,
    trace_id: str,
    feature_version: str,
    model_version: str,
    cohort: str,
    topk: List[Dict[str, Any]],
    notes: Optional[str] = None,
) -> Dict[str, Any]:
    """A privacy-safe rank trace payload.

    The goal is debugging without storing raw PII.

    topk should contain score summaries, not sensitive raw features.
    """
    return {
        "trace_id": trace_id,
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "feature_version": feature_version,
        "model_version": model_version,
        "cohort": cohort,
        "topk": topk,
        "notes": notes,
    }


def export_trace_jsonl(path: str, traces: List[Dict[str, Any]]) -> None:
    """Write traces to JSONL for inspection."""
    with open(path, "w", encoding="utf-8") as f:
        for t in traces:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")


def summarize_topk(item_ids: np.ndarray, scores: np.ndarray, k: int = 10) -> List[Dict[str, Any]]:
    """Return a minimal top-k summary suitable for traces."""
    idx = np.argsort(-scores)[:k]
    out: List[Dict[str, Any]] = []
    for rank, i in enumerate(idx, start=1):
        out.append({"rank": rank, "item_id": int(item_ids[i]), "score": float(scores[i])})
    return out
