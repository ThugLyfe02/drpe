# DRPE — Drift‑Resilient Personalization Engine

A systems‑first recommendation project focused on **durability**: embedding lifecycle, drift detection, guardrails, and staged rollouts.

DRPE is a durability‑first personalization sandbox: it simulates end‑to‑end recsys behavior, trains retrieval/ranking components, and gates every change with drift + cohort guardrails. The point isn’t a single offline metric spike — it’s proving you can ship safely when embeddings drift, cohorts diverge, or cold‑start launches go sideways. Each demo run outputs a model card, and blocked rollouts automatically emit an incident record plus a privacy‑safe trace signature.

Most portfolio recommenders stop at “train a model + report NDCG.” DRPE treats recommendation as a **production feedback system** that can silently degrade when proxies, drift, or cohort variance are ignored.

## What’s in this repo (v0)
- **Session‑based simulator** generating user events, cohorts, a retention proxy, and controlled drift
- **Metrics contract**: EngagementDepth + RetentionProxy + cohort breakdown
- **Embedding lifecycle**: warm‑start updates + geometry drift scoring
- **Ranker lifecycle**: multi‑objective re‑ranking gated by durability
- **RecSysOps**: incident schema + trace signatures + cold‑start risk → ramp plan

## Quickstart (Windows PowerShell)
```powershell
cd $env:USERPROFILE\projects
# clone if you haven’t
# git clone https://github.com/ThugLyfe02/drpe.git
cd drpe

py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip

# install the package (src layout)
pip install -e .

pytest -q
```

## Design philosophy
- **Optimize**: EngagementDepth (depth > clicks)
- **Guardrail**: RetentionProxy stability (cohort‑aware)
- **Monitor**: distribution drift + embedding geometry drift
- **Rollout**: stage → observe → expand (rollback on guardrail breach)

See `docs/architecture.md` for the system spec.

## How to reproduce (5 minutes, Windows/PowerShell)

**Goal:** run end‑to‑end “RecSysOps” demos that *gate* model changes (embeddings + ranker) with durability guardrails, emit model cards, and (when blocked) automatically produce incident + trace artifacts.

### 1) Setup
```powershell
# from repo root
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .
pytest -q
```

### 2) One‑command demo bundle (recommended)
Runs 4 scenarios and exports everything:
- **[A] Embedding rollout (normal)** → allowed
- **[B] Embedding rollout (forced block)** → incident + trace emitted
- **[C] Ranker rollout (safe)** → allowed
- **[D] Ranker rollout (risky)** → blocked + incident emitted

```powershell
python -m drpe.demo.all --export --emit-ops
```

### 3) Where outputs land (shareable evidence pack)
```powershell
dir artifacts\model_cards
dir artifacts\incidents
dir artifacts\traces
dir artifacts\ops
```

You should see:
- `artifacts/model_cards/*.txt` — model cards with lift + drift + cohort breakdown + decision
- `artifacts/incidents/INC-*.json` — structured IncidentRecord emitted when a rollout is blocked
- `artifacts/traces/rank_traces.jsonl` — privacy‑safe rank trace signatures
- `artifacts/ops/cold_start_ramp-*.json` — cold‑start risk assessment → safe ramp plan + stop conditions

### 4) Run demos individually (optional)
```powershell
# embeddings: normal run (usually allowed)
python -m drpe.demo.run --export --emit-ops

# embeddings: deterministic block to demonstrate incident + trace emission
python -m drpe.demo.run --export --emit-ops --force-block

# ranker: safe rollout (allowed)
python -m drpe.demo.ranker_demo --mode safe --export --emit-ops

# ranker: risky rollout (blocked by retention guardrail)
python -m drpe.demo.ranker_demo --mode risky --export --emit-ops
```

## Why this project is different
DRPE isn’t “just a model.” It’s a **release discipline**:
- recommendation changes are treated like **schema migrations**
- drift + cohort stability + retention proxy are **guardrails**, not afterthoughts
- blocked rollouts automatically generate a **structured incident record** + **debug trace signature**
- cold‑start launches are managed with **risk scoring → ramp policy**
