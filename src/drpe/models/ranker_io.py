from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from drpe.models.ranker import MultiObjectiveRanker


@dataclass
class RankerArtifact:
    in_dim: int
    hidden: int
    dropout: float


def save_ranker(path: str | Path, model: MultiObjectiveRanker, meta: RankerArtifact) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "meta": {
            "in_dim": meta.in_dim,
            "hidden": meta.hidden,
            "dropout": meta.dropout,
        },
        "state_dict": model.state_dict(),
    }
    torch.save(payload, path)


def load_ranker(path: str | Path) -> MultiObjectiveRanker:
    payload = torch.load(path, map_location="cpu")
    meta = payload["meta"]
    model = MultiObjectiveRanker(
        in_dim=int(meta["in_dim"]),
        hidden=int(meta["hidden"]),
        dropout=float(meta["dropout"]),
    )
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model
