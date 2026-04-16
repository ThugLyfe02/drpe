from drpe.data.simulator import SimConfig
from drpe.rollout.guardrails import GuardrailConfig
from drpe.rollout.rollout_report import build_rollout_report
from drpe.rollout.rollout_simulation import default_candidate_cfg


def test_rollout_blocks_on_embedding_drift_when_enabled():
    base = SimConfig(num_users=60, num_items=250, sessions_per_user=2, k=10)
    cand = default_candidate_cfg(base)

    # Set an intentionally strict drift gate so it will block.
    report = build_rollout_report(
        baseline_cfg=base,
        candidate_cfg=cand,
        guardrails=GuardrailConfig(max_retention_drop=0.50, max_embedding_mean_cosine_shift=0.0001),
    )

    assert report.decision.allow_rollout is False
    assert "embedding drift" in report.decision.reason
