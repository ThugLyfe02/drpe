from drpe.data.simulator import SimConfig
from drpe.training.train_ranker import RankerTrainCfg, train_ranker


def test_ranker_training_smoke():
    sim = SimConfig(num_users=60, num_items=250, sessions_per_user=1, k=10, embedding_dim=16, seed=7)
    model = train_ranker(RankerTrainCfg(sim=sim, epochs=1, lr=1e-3))
    assert model is not None
