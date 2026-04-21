# RecSysOps Runbook (DRPE)

This is a lightweight runbook for operating a recommendation system.

## 1) Detect
**Goal:** learn about issues before users complain.

Signals to watch:
- RetentionProxy guardrail (global + per cohort)
- Score distribution drift (KL)
- Embedding geometry drift (mean cosine shift)
- Repetition / diversity
- Latency tails + error rates

## 2) Predict
**Goal:** anticipate incidents using leading indicators.

Leading indicators:
- Drift velocity (rate of change)
- Cohort divergence
- Candidate churn rate (top-k instability)
- Latency tail growth

Output:
- risk-of-incident in next 6–12h
- recommended mitigation playbook

## 3) Diagnose
**Goal:** understand “why this happened,” quickly.

Tools:
- Privacy-safe rank traces (sampled)
- Feature attribution / counterfactual probes
- Cohort slicing (new/core/power)

## 4) Resolve
**Goal:** mitigate immediately and prevent recurrence.

Immediate mitigations:
- rollback to last good model
- clamp risky features (freshness, novelty)
- throttle rollout / revert canary

Prevention:
- add guardrail + alert
- add targeted test
- add a trace signature for earlier detection
