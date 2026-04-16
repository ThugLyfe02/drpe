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

## Why systems‑first
A recommendation model can show offline lift while silently degrading online due to drift, proxy misalignment, and cohort instability. DRPE is designed to detect and prevent those failure modes.
