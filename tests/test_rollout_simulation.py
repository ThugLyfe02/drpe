from drpe.data.simulator import SimConfig
from drpe.rollout.guardrails import GuardrailConfig
from drpe.rollout.rollout_simulation import compare_for_rollout, default_candidate_cfg


def test_rollout_blocks_when_candidate_retention_worse():
    base = SimConfig(num_users=80, num_items=300, sessions_per_user=2, k=10)
    cand = default_candidate_cfg(base)
    decision_cfg = GuardrailConfig(max_retention_drop=0.01)

    comp = compare_for_rollout(baseline_cfg=base, candidate_cfg=cand, guardrails=decision_cfg)
    # Candidate is designed to have worse durability
    assert comp.decision.allow_rollout is False


def test_rollout_allows_when_guardrail_relaxed():
    base = SimConfig(num_users=80, num_items=300, sessions_per_user=2, k=10)
    cand = default_candidate_cfg(base)
    decision_cfg = GuardrailConfig(max_retention_drop=0.50)

    comp = compare_for_rollout(baseline_cfg=base, candidate_cfg=cand, guardrails=decision_cfg)
    assert comp.decision.allow_rollout is True
