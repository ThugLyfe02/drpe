import numpy as np

from drpe.drift.embedding_geometry import build_geometry_drift_report
from drpe.embeddings.versioning import generate_embedding_versions


def test_geometry_drift_report_shapes():
    emb = generate_embedding_versions(seed=1, embedding_dim=8, num_users=50, num_items=100, drift_strength=0.05)
    rep = build_geometry_drift_report(
        users_v1=emb.users_v1,
        users_v2=emb.users_v2,
        items_v1=emb.items_v1,
        items_v2=emb.items_v2,
        user_cohorts=emb.user_cohorts,
    )
    assert rep.users.n == 50
    assert rep.items.n == 100
    assert rep.users.mean_cosine_shift >= 0.0
    assert rep.items.mean_cosine_shift >= 0.0
    assert len(rep.cohort_user_mean_shift) > 0
