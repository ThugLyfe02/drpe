import numpy as np

from drpe.data.simulator import SimConfig, run_simulation_with_embeddings


def test_simulation_with_embeddings_runs():
    cfg = SimConfig(num_users=30, num_items=120, sessions_per_user=2, k=10, embedding_dim=16)

    rng = np.random.default_rng(0)
    users = rng.normal(0, 1, (cfg.num_users, cfg.embedding_dim)).astype(np.float32)
    items = rng.normal(0, 1, (cfg.num_items, cfg.embedding_dim)).astype(np.float32)
    users /= (np.linalg.norm(users, axis=1, keepdims=True) + 1e-8)
    items /= (np.linalg.norm(items, axis=1, keepdims=True) + 1e-8)

    items_quality = rng.uniform(0.3, 1.0, cfg.num_items).astype(np.float32)
    items_pop = rng.beta(2, 8, cfg.num_items).astype(np.float32)

    events, summaries = run_simulation_with_embeddings(
        cfg,
        users_embed=users,
        items_vec=items,
        items_quality=items_quality,
        items_popularity=items_pop,
        embedding_version="emb_test",
        model_version="rank_test",
    )

    assert len(events) > 0
    assert len(summaries) == cfg.num_users * cfg.sessions_per_user
