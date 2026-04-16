# DRPE Architecture (v0)

## Goal
Build a **drift‑resilient** personalization system that optimizes short‑term engagement while enforcing long‑term durability guardrails.

## Core metrics
- **Optimize:** EngagementDepth (depth > clicks)
- **Guardrail:** RetentionProxy stability (cohort‑aware)
- **Monitor:** embedding drift + feature drift + cohort variance

## Data model
The simulator generates:
- session events (impression/click/play/complete/skip)
- cohort labels (new/core/power)
- a retention proxy per session
- controlled embedding drift over time

## Pipeline (target)
User events → features → embeddings → retrieval (top‑K) → ranking (multi‑objective) → evaluation → drift monitor → rollout guardrails.

## Rollout control (new)
We treat candidate model updates as **staged rollouts**. A candidate is blocked if the retention proxy drops beyond the configured threshold (optionally cohort‑aware), even if engagement improves.

See `src/drpe/rollout/rollout_simulation.py` for a baseline vs candidate comparison that produces an explicit rollout decision.

## Why systems‑first
A recommendation model can show offline lift while silently degrading online due to drift, proxy misalignment, and cohort instability. DRPE is designed to detect and prevent those failure modes.
