from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from drpe.data.schemas import Event


@dataclass
class ImplicitConfig:
    num_users: int
    num_items: int
    negatives_per_positive: int = 4
    seed: int = 7


class ImplicitFeedbackDataset(Dataset):
    """Builds (user_id, item_id, label) samples from simulator events.

    Positives: play/complete
    Negatives: sampled items not interacted with by that user (implicit negatives)

    This dataset is intentionally small & simple to keep DRPE focused on the
    end-to-end loop (train -> export -> rollout gate).
    """

    def __init__(self, events: List[Event], cfg: ImplicitConfig):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)

        # Collect user->positive items
        user_pos = {u: set() for u in range(cfg.num_users)}
        for e in events:
            if e.event_type in ("play", "complete"):
                user_pos[e.user_id].add(e.item_id)

        samples: List[Tuple[int, int, int]] = []

        for u, pos_items in user_pos.items():
            for it in pos_items:
                samples.append((u, it, 1))

                # sample implicit negatives
                for _ in range(cfg.negatives_per_positive):
                    neg = int(self.rng.integers(0, cfg.num_items))
                    tries = 0
                    while neg in pos_items and tries < 10:
                        neg = int(self.rng.integers(0, cfg.num_items))
                        tries += 1
                    samples.append((u, neg, 0))

        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        u, it, y = self.samples[idx]
        return (
            torch.tensor(u, dtype=torch.long),
            torch.tensor(it, dtype=torch.long),
            torch.tensor(y, dtype=torch.float32),
        )
