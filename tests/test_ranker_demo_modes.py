from drpe.data.simulator import SimConfig
from drpe.models.ranker import MultiObjectiveRanker
from drpe.rollout.guardrails import GuardrailConfig
from drpe.rollout.ranker_rollout import compare_rankers_for_rollout
from drpe.training.train_two_tower import TrainConfig, train


def test_risky_mode_can_trigger_block(tmp_path):
    emb_path = tmp_path / "emb.npz"
    train(
        TrainConfig(
            sim=SimConfig(num_users=60, num_items=250, sessions_per_user=1, k=10, embedding_dim=16, seed=7),
            dim=16,
            epochs=1,
            batch_size=512,
            seed=7,
            out_path=str(emb_path),
        )
    )

    base_ranker = MultiObjectiveRanker(in_dim=8)
    cand_ranker = MultiObjectiveRanker(in_dim=8)

    rep = compare_rankers_for_rollout(
        embeddings_path=str(emb_path),
        baseline_ranker=base_ranker,
        candidate_ranker=cand_ranker,
        cfg=SimConfig(num_users=60, num_items=250, sessions_per_user=1, k=10, embedding_dim=16, seed=9),
        guardrails=GuardrailConfig(max_retention_drop=0.0),
        gamma_retention=0.05,
        retention_fatigue_penalty=0.20,
    )

    # With max_retention_drop=0, any drop blocks. This ensures we exercise the path.
    assert rep.decision.reason
