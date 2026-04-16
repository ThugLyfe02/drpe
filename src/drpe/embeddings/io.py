from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np


def save_embeddings(path: str | Path, users: np.ndarray, items: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, users=users, items=items)


def load_embeddings(path: str | Path) -> Tuple[np.ndarray, np.ndarray]:
    path = Path(path)
    data = np.load(path)
    return data["users"], data["items"]
