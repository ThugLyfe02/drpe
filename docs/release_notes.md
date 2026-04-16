# DRPE Release Notes (v0)

## v0.1 — Systems-first rollout gating

**What changed**
- Added drift scoring utilities (KL drift + embedding cosine shift)
- Added rollout guardrails (retention proxy + optional embedding drift gates)
- Added rollout simulation + report (baseline vs candidate)
- Added PyTorch two-tower embedding training + export
- Added warm-start training to reduce embedding drift between versions

**Why it matters**
- Many recommenders optimize proxy metrics and silently degrade.
- DRPE treats embedding updates like schema migrations: measurable drift gates rollout.

**Demo outputs**
- Scratch retrain: drift ≈ 1.0 → rollout blocked
- Warm-start update: drift ≈ 0.008 → rollout allowed

**Next**
- Ranking layer (multi-objective) + online-style cohort evaluation
- Lightweight dashboard output
