from drpe.data.simulator import SimConfig
from drpe.rollout.guardrails import GuardrailConfig
from drpe.rollout.rollout_from_artifacts import compare_embedding_artifacts
from drpe.training.train_two_tower import TrainConfig, train


def test_cohort_embedding_drift_guardrail_blocks(tmp_path):
    # Train v1
    base_path = tmp_path / "emb_v1.npz"
    train(
        TrainConfig(
            sim=SimConfig(num_users=60, num_items=250, sessions_per_user=1, k=10, embedding_dim=16, seed=1),
            dim=16,
            epochs=1,
            batch_size=512,
            seed=1,
            out_path=str(base_path),
        )
    )

    # Train v2 from scratch with different seed (higher drift)
    cand_path = tmp_path / "emb_v2.npz"
    train(
        TrainConfig(
            sim=SimConfig(num_users=60, num_items=250, sessions_per_user=1, k=10, embedding_dim=16, seed=2),
            dim=16,
            epochs=1,
            batch_size=512,
            seed=2,
            out_path=str(cand_path),
        )
    )

    report = compare_embedding_artifacts(
        baseline_path=str(base_path),
        candidate_path=str(cand_path),
        cfg=SimConfig(num_users=60, num_items=250, sessions_per_user=1, k=10, embedding_dim=16, seed=3),
        guardrails=GuardrailConfig(
            max_retention_drop=0.50,
            max_embedding_mean_cosine_shift=1.0,
            max_embedding_mean_cosine_shift_per_cohort=0.05,
        ),
    )

    assert report.decision.allow_rollout is False
