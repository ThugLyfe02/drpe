from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from drpe.data.simulator import SimConfig, run_simulation
from drpe.models.ranker import MultiObjectiveRanker, build_features


class RankerDataset(Dataset):
    """Supervised dataset from simulator events.

    We treat play/complete as engagement label.
    Retention label is session-level retention_proxy attached to each impression.
    """

    def __init__(self, cfg: SimConfig, seed: int = 7):
        self.cfg = cfg
        rng = np.random.default_rng(seed)

        events, summaries = run_simulation(cfg, embedding_version="emb_sim", model_version="rank_sim")

        # map session_id -> retention_proxy
        ret_by_session = {s.session_id: s.retention_proxy for s in summaries}

        # We need item_quality/popularity; simulator items are internal, so approximate
        # using stable RNG (consistent across runs for same seed).
        item_quality = rng.uniform(0.3, 1.0, cfg.num_items).astype(np.float32)
        item_pop = rng.beta(2, 8, cfg.num_items).astype(np.float32)

        # engagement label per (user,item,session,rank)
        engaged = set(
            (e.session_id, e.user_id, e.item_id)
            for e in events
            if e.event_type in ("play", "complete")
        )

        X = []
        y_eng = []
        y_ret = []

        for e in events:
            if e.event_type != "impression":
                continue
            lbl = 1.0 if (e.session_id, e.user_id, e.item_id) in engaged else 0.0
            ret = float(ret_by_session.get(e.session_id, 0.0))

            feats = build_features(
                score=e.score,
                affinity=e.affinity,
                rank=e.rank,
                item_quality=float(item_quality[e.item_id]),
                item_popularity=float(item_pop[e.item_id]),
                cohort=e.cohort,
            )
            X.append(feats)
            y_eng.append(lbl)
            y_ret.append(ret)

        self.X = torch.tensor(np.stack(X), dtype=torch.float32)
        self.y_eng = torch.tensor(np.array(y_eng), dtype=torch.float32)
        self.y_ret = torch.tensor(np.array(y_ret), dtype=torch.float32)

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int):
        return self.X[idx], self.y_eng[idx], self.y_ret[idx]


@dataclass
class RankerTrainCfg:
    sim: SimConfig
    epochs: int = 2
    lr: float = 1e-3
    batch: int = 2048
    hidden: int = 64
    dropout: float = 0.10

    # multi-objective weighting
    alpha_eng: float = 1.0
    beta_ret: float = 0.35


def train_ranker(cfg: RankerTrainCfg) -> MultiObjectiveRanker:
    ds = RankerDataset(cfg.sim, seed=cfg.sim.seed)
    dl = DataLoader(ds, batch_size=cfg.batch, shuffle=True)

    model = MultiObjectiveRanker(in_dim=ds.X.shape[1], hidden=cfg.hidden, dropout=cfg.dropout)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    loss_eng = torch.nn.BCEWithLogitsLoss()
    loss_ret = torch.nn.MSELoss()

    model.train()
    for ep in range(cfg.epochs):
        total = 0.0
        n = 0
        for x, y_e, y_r in dl:
            e_logit, r_pred = model(x)
            le = loss_eng(e_logit, y_e)
            lr_ = loss_ret(r_pred, y_r)
            loss = cfg.alpha_eng * le + cfg.beta_ret * lr_

            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.item())
            n += 1
        print(f"epoch {ep+1}/{cfg.epochs} loss={total/max(n,1):.4f} samples={len(ds)}")

    return model


def main() -> None:
    p = argparse.ArgumentParser(description="Train DRPE multi-objective ranker (engagement + retention proxy)")
    p.add_argument("--users", type=int, default=300)
    p.add_argument("--items", type=int, default=1500)
    p.add_argument("--sessions", type=int, default=2)
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--embed-dim", dest="embedding_dim", type=int, default=32)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--beta-ret", type=float, default=0.35)
    a = p.parse_args()

    sim = SimConfig(
        seed=7,
        embedding_dim=a.embedding_dim,
        num_users=a.users,
        num_items=a.items,
        sessions_per_user=a.sessions,
        k=a.k,
    )

    train_ranker(RankerTrainCfg(sim=sim, epochs=a.epochs, lr=a.lr, beta_ret=a.beta_ret))


if __name__ == "__main__":
    main()
