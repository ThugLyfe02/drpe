import os

from drpe.training.train_two_tower import TrainConfig, train


def test_two_tower_training_smoke(tmp_path):
    out = tmp_path / "embeddings_test.npz"

    cfg = TrainConfig(
        sim=TrainConfig().sim.__class__(num_users=80, num_items=300, sessions_per_user=1, k=10, embedding_dim=16),
        dim=16,
        epochs=1,
        batch_size=512,
        out_path=str(out),
    )

    path = train(cfg)
    assert os.path.exists(path)
