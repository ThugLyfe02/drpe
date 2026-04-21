@"
from __future__ import annotations

from typing import Optional

import numpy as np

from drpe.recsysops.trace_sampler import (
    TraceSampleConfig,
    build_rank_trace,
    export_trace_jsonl,
    make_trace_id,
    should_sample,
    summarize_topk,
)

def maybe_emit_trace(
    *,
    user_id: int,
    session_id: str,
    model_version: str,
    cohort: str,
    feature_version: str,
    item_ids: np.ndarray,
    scores: np.ndarray,
    out_path: str = "artifacts/traces/rank_traces.jsonl",
    cfg: TraceSampleConfig = TraceSampleConfig(),
) -> Optional[str]:
    """Emit a privacy-safe sampled rank trace and return its trace_id."""
    trace_id = make_trace_id(user_id=user_id, session_id=session_id, model_version=model_version)

    if not should_sample(trace_id, cfg):
        return None

    topk = summarize_topk(item_ids=item_ids, scores=scores, k=10)
    trace = build_rank_trace(
        trace_id=trace_id,
        feature_version=feature_version,
        model_version=model_version,
        cohort=cohort,
        topk=topk,
        notes="auto-emitted by DRPE demo",
    )
    export_trace_jsonl(out_path, [trace])
    return trace_id
"@ | Out-File -Encoding utf8 .\src\drpe\recsysops\trace_integration.py