from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import DataLoader

from drpe.data.simulator import SimConfig, run_simulation
from drpe.embeddings.io import save_embeddings
from drpe.models.two_tower import TwoTower
from drpe.training.dataset import ImplicitConfig, ImplicitFeedbackDataset


@dataclass
class TrainConfig:
    # simulator
    sim: SimConfig = SimConfig(num_users=300, num_items=1500, sessions_per_user=2, k=10)

    # model
    dim: int = 64

    # training
    lr: float = 1e-2
    batch_size: int = 2048
    epochs: int = 2
    negatives_per_positive: int = 4
    seed: int = 7

    # output
    out_path: str = "artifacts/embeddings_v1.npz"


def train(cfg: TrainConfig) -> str:
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    events, _ = run_simulation(cfg.sim, embedding_version="emb_train", model_version="rank_train")

    ds = ImplicitFeedbackDataset(
        events,
        ImplicitConfig(
            num_users=cfg.sim.num_users,
            num_items=cfg.sim.num_items,
            negatives_per_positive=cfg.negatives_per_positive,
            seed=cfg.seed,
        ),
    )
    dl = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True)

    model = TwoTower(cfg.sim.num_users, cfg.sim.num_items, dim=cfg.dim)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    model.train()
    for epoch in range(cfg.epochs):
        total = 0.0
        n = 0
        for u, it, y in dl:
            logits = model(u, it)
            loss = loss_fn(logits, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.item())
            n += 1
        # light logging
        print(f"epoch {epoch+1}/{cfg.epochs} loss={total/max(n,1):.4f} samples={len(ds)}")

    users_t, items_t = model.export_user_item_embeddings()
    users = users_t.numpy().astype(np.float32)
    items = items_t.numpy().astype(np.float32)
    # normalize rows for cosine stability
    users /= (np.linalg.norm(users, axis=1, keepdims=True) + 1e-8)
    items /= (np.linalg.norm(items, axis=1, keepdims=True) + 1e-8)

    save_embeddings(cfg.out_path, users, items)
    return cfg.out_path


if __name__ == "__main__":
    path = train(TrainConfig())
    print(f"saved embeddings: {path}")
