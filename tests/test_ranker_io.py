from drpe.models.ranker import MultiObjectiveRanker
from drpe.models.ranker_io import RankerArtifact, load_ranker, save_ranker


def test_ranker_save_load_roundtrip(tmp_path):
    path = tmp_path / "ranker.pt"
    model = MultiObjectiveRanker(in_dim=8)
    save_ranker(path, model, RankerArtifact(in_dim=8, hidden=64, dropout=0.10))
    loaded = load_ranker(path)
    assert isinstance(loaded, MultiObjectiveRanker)
