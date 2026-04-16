from __future__ import annotations

import numpy as np
import torch
from torch import nn


class TwoTower(nn.Module):
    """Minimal two-tower retrieval model.

    - user_id -> user embedding
    - item_id -> item embedding

    Score = dot(u, v)

    DRPE’s differentiation is lifecycle discipline (evaluation, drift, guardrails).
    The model stays intentionally simple and easy to reason about.
    """

    def __init__(self, num_users: int, num_items: int, dim: int = 64):
        super().__init__()
        self.user_emb = nn.Embedding(num_users, dim)
        self.item_emb = nn.Embedding(num_items, dim)
        nn.init.normal_(self.user_emb.weight, std=0.02)
        nn.init.normal_(self.item_emb.weight, std=0.02)

    def forward(self, user_ids: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
        u = self.user_emb(user_ids)
        v = self.item_emb(item_ids)
        return (u * v).sum(dim=-1)

    @torch.no_grad()
    def export_user_item_embeddings(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.user_emb.weight.detach().cpu(), self.item_emb.weight.detach().cpu()

    @torch.no_grad()
    def warm_start_from_np(self, users: np.ndarray, items: np.ndarray) -> None:
        """Warm-start model embeddings from numpy arrays.

        This is the production-relevant move: treat embedding updates like schema
        migrations. Warm-starting preserves geometry continuity and reduces drift.
        """
        if users.shape != tuple(self.user_emb.weight.shape):
            raise ValueError(f"users shape mismatch: {users.shape} vs {tuple(self.user_emb.weight.shape)}")
        if items.shape != tuple(self.item_emb.weight.shape):
            raise ValueError(f"items shape mismatch: {items.shape} vs {tuple(self.item_emb.weight.shape)}")

        self.user_emb.weight.copy_(torch.tensor(users, dtype=self.user_emb.weight.dtype))
        self.item_emb.weight.copy_(torch.tensor(items, dtype=self.item_emb.weight.dtype))
