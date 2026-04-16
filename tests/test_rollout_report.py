from drpe.data.simulator import SimConfig
from drpe.rollout.guardrails import GuardrailConfig
from drpe.rollout.rollout_report import build_rollout_report
from drpe.rollout.rollout_simulation import default_candidate_cfg


def test_rollout_report_has_drift_and_decision():
    base = SimConfig(num_users=60, num_items=250, sessions_per_user=2, k=10)
    cand = default_candidate_cfg(base)
    report = build_rollout_report(
        baseline_cfg=base,
        candidate_cfg=cand,
        guardrails=GuardrailConfig(max_retention_drop=0.01, max_embedding_mean_cosine_shift=0.50),
    )

    assert report.drift.engagement_depth_kl >= 0.0
    assert report.drift.retention_proxy_kl >= 0.0
    assert report.drift.embedding_geometry.users.mean_cosine_shift >= 0.0
    assert report.drift.embedding_geometry.items.mean_cosine_shift >= 0.0
    assert report.decision.allow_rollout is False
