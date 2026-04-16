from __future__ import annotations

from dataclasses import dataclass, field

import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader

from drpe.data.simulator import SimConfig, run_simulation
from drpe.embeddings.io import load_embeddings, save_embeddings
from drpe.models.two_tower import TwoTower
from drpe.training.dataset import ImplicitConfig, ImplicitFeedbackDataset


@dataclass
class TrainConfig:
    # simulator (use default_factory to avoid mutable default)
    sim: SimConfig = field(
        default_factory=lambda: SimConfig(num_users=300, num_items=1500, sessions_per_user=2, k=10)
    )

    # model
    dim: int = 64

    # training
    lr: float = 1e-2
    batch_size: int = 2048
    epochs: int = 2
    negatives_per_positive: int = 4
    seed: int = 7

    # warm-start (optional)
    warm_start_path: str | None = None

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

    # Warm-start embeddings to preserve geometry continuity (production-relevant).
    if cfg.warm_start_path:
        users_np, items_np = load_embeddings(cfg.warm_start_path)
        model.warm_start_from_np(users_np, items_np)

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
        print(f"epoch {epoch+1}/{cfg.epochs} loss={total/max(n,1):.4f} samples={len(ds)}")

    users_t, items_t = model.export_user_item_embeddings()
    users = users_t.numpy().astype(np.float32)
    items = items_t.numpy().astype(np.float32)

    # normalize rows for cosine stability
    users /= (np.linalg.norm(users, axis=1, keepdims=True) + 1e-8)
    items /= (np.linalg.norm(items, axis=1, keepdims=True) + 1e-8)

    save_embeddings(cfg.out_path, users, items)
    return cfg.out_path


def _parse_args() -> TrainConfig:
    p = argparse.ArgumentParser(description="Train DRPE two-tower embeddings from simulator logs")

    # output + core knobs
    p.add_argument("--out", dest="out_path", default="artifacts/embeddings_v1.npz")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--dim", type=int, default=64)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--batch", dest="batch_size", type=int, default=2048)
    p.add_argument("--neg", dest="negatives_per_positive", type=int, default=4)

    # warm start
    p.add_argument("--warm-start", dest="warm_start_path", default=None)

    # simulator sizing (for quick local runs)
    p.add_argument("--users", type=int, default=300)
    p.add_argument("--items", type=int, default=1500)
    p.add_argument("--sessions", type=int, default=2)
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--embed-dim", dest="embedding_dim", type=int, default=32)

    a = p.parse_args()

    sim = SimConfig(
        seed=a.seed,
        embedding_dim=a.embedding_dim,
        num_users=a.users,
        num_items=a.items,
        k=a.k,
        sessions_per_user=a.sessions,
    )

    return TrainConfig(
        sim=sim,
        dim=a.dim,
        lr=a.lr,
        batch_size=a.batch_size,
        epochs=a.epochs,
        negatives_per_positive=a.negatives_per_positive,
        seed=a.seed,
        warm_start_path=a.warm_start_path,
        out_path=a.out_path,
    )


if __name__ == "__main__":
    cfg = _parse_args()
    path = train(cfg)
    print(f"saved embeddings: {path}")
