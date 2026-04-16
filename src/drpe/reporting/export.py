from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

from drpe.reporting.model_card import Card


@dataclass
class CardHeader:
    kind: str  # e.g. embedding_rollout | ranker_rollout
    version_left: str
    version_right: str
    thresholds: Dict[str, str]
    notes: Optional[str] = None


def render_with_header(card: Card, header: CardHeader) -> str:
    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    lines = [
        f"timestamp_utc: {ts}",
        f"kind: {header.kind}",
        f"compare: {header.version_left} -> {header.version_right}",
    ]
    if header.thresholds:
        lines.append("thresholds:")
        for k in sorted(header.thresholds.keys()):
            lines.append(f"  - {k}: {header.thresholds[k]}")
    if header.notes:
        lines.append(f"notes: {header.notes}")

    header_block = "\n".join(lines)
    sep = "-" * 72
    return f"{header_block}\n{sep}\n{card.render()}"


def write_card(path: str | Path, content: str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path
