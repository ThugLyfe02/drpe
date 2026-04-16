import os

from drpe.data.simulator import SimConfig
from drpe.rollout.guardrails import GuardrailConfig
from drpe.rollout.rollout_from_artifacts import compare_embedding_artifacts
from drpe.training.train_two_tower import TrainConfig, train


def test_rollout_from_artifacts_smoke(tmp_path):
    # Train two small embedding sets with slightly different seeds
    base_path = tmp_path / "emb_v1.npz"
    cand_path = tmp_path / "emb_v2.npz"

    base_cfg = TrainConfig(
        sim=SimConfig(num_users=60, num_items=250, sessions_per_user=1, k=10, embedding_dim=16, seed=1),
        dim=16,
        epochs=1,
        batch_size=512,
        seed=1,
        out_path=str(base_path),
    )
    cand_cfg = TrainConfig(
        sim=SimConfig(num_users=60, num_items=250, sessions_per_user=1, k=10, embedding_dim=16, seed=2),
        dim=16,
        epochs=1,
        batch_size=512,
        seed=2,
        out_path=str(cand_path),
    )

    train(base_cfg)
    train(cand_cfg)

    assert os.path.exists(base_path)
    assert os.path.exists(cand_path)

    report = compare_embedding_artifacts(
        baseline_path=str(base_path),
        candidate_path=str(cand_path),
        cfg=SimConfig(num_users=60, num_items=250, sessions_per_user=1, k=10, embedding_dim=16, seed=3),
        guardrails=GuardrailConfig(max_retention_drop=0.50, max_embedding_mean_cosine_shift=1.0),
    )

    assert report.geom_users_mean >= 0.0
    assert report.geom_items_mean >= 0.0
