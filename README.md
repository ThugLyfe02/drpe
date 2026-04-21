# DRPE — Drift‑Resilient Personalization Engine

A systems‑first recommendation project focused on **durability**: embedding lifecycle, drift detection, guardrails, and staged rollouts.

Most portfolio recommenders stop at “train a model + report NDCG.” DRPE treats recommendation as a **production feedback system** that can silently degrade when proxies, drift, or cohort variance are ignored.

## What’s in this repo (v0)
- **Session‑based simulator** generating user events, cohorts, a retention proxy, and controlled drift
- **Metrics contract**: EngagementDepth + RetentionProxy + cohort breakdown
- **(Next)** drift scoring + rollout guardrails + retrieval/ranking models

## Quickstart (Windows PowerShell)
```powershell
cd $env:USERPROFILE\projects
# clone if you haven’t
# git clone https://github.com/ThugLyfe02/drpe.git
cd drpe

py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt

# install the package (src layout)
pip install -e .

pytest -q
```

## Design philosophy
- **Optimize**: EngagementDepth (depth > clicks)
- **Guardrail**: RetentionProxy stability (cohort‑aware)
- **Monitor**: embedding/feature drift + cohort variance
- **Rollout**: stage → observe → expand (rollback on guardrail breach)

See `docs/architecture.md` for the system spec.

## How to reproduce (5 minutes, Windows/PowerShell)

**Goal:** run end-to-end “RecSysOps” demos that *gate* model changes (embeddings + ranker) with durability guardrails, emit model cards, and (when blocked) automatically produce incident + trace artifacts.

### 1) Setup
```powershell
# from repo root
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .
pytest -q