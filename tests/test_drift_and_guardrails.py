import numpy as np

from drpe.drift.drift import embedding_drift, histogram_kl
from drpe.rollout.guardrails import GuardrailConfig, decide_rollout


def test_embedding_drift_basic():
    a = np.zeros((100, 8), dtype=np.float32)
    b = np.zeros((100, 8), dtype=np.float32)
    b[:, 0] = 1.0
    res = embedding_drift(a + 1e-6, b)  # avoid zero-norm
    assert res.n == 100
    assert res.mean_cosine_shift >= 0.0


def test_histogram_kl_zero_when_identical():
    x = np.random.default_rng(0).normal(0, 1, 1000)
    assert histogram_kl(x, x) == 0.0


def test_guardrail_blocks_on_retention_drop():
    cfg = GuardrailConfig(max_retention_drop=0.01)
    decision = decide_rollout(
        baseline_retention=0.50,
        candidate_retention=0.48,  # 4% drop
        cfg=cfg,
    )
    assert decision.allow_rollout is False


def test_guardrail_allows_when_retention_stable():
    cfg = GuardrailConfig(max_retention_drop=0.02)
    decision = decide_rollout(
        baseline_retention=0.50,
        candidate_retention=0.495,  # 1% drop
        cfg=cfg,
    )
    assert decision.allow_rollout is True
