from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import List, Tuple

import numpy as np

from .schemas import Event, SessionSummary


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-x))


@dataclass
class SimConfig:
    seed: int = 7
    embedding_dim: int = 32
    num_users: int = 500
    num_items: int = 2000
    k: int = 20
    sessions_per_user: int = 8

    # Retention dynamics
    base_return_prob_min: float = 0.25
    base_return_prob_max: float = 0.70
    retention_gain_per_depth: float = 0.015
    retention_decay_per_day: float = 0.03

    # Behavior dynamics
    fatigue_gain_per_play: float = 0.03
    fatigue_recovery_per_day: float = 0.05

    # Drift
    drift_every_n_sessions: int = 6
    drift_strength: float = 0.03


@dataclass
class UserState:
    user_id: int
    cohort: str
    true_pref: np.ndarray
    embed: np.ndarray
    fatigue: float
    base_return_prob: float
    last_active: datetime


@dataclass
class ItemState:
    item_id: int
    vec: np.ndarray
    quality: float
    popularity: float


def _assign_cohort(rng: np.random.Generator) -> str:
    r = rng.random()
    if r < 0.25:
        return "new"
    if r < 0.75:
        return "core"
    return "power"


def generate_world(cfg: SimConfig) -> Tuple[List[UserState], List[ItemState]]:
    rng = np.random.default_rng(cfg.seed)
    now = datetime.now(timezone.utc)

    users: List[UserState] = []
    for uid in range(cfg.num_users):
        cohort = _assign_cohort(rng)
        true_pref = rng.normal(0, 1, cfg.embedding_dim).astype(np.float32)
        true_pref /= (np.linalg.norm(true_pref) + 1e-8)

        base_return = float(rng.uniform(cfg.base_return_prob_min, cfg.base_return_prob_max))
        users.append(
            UserState(
                user_id=uid,
                cohort=cohort,
                true_pref=true_pref,
                embed=true_pref.copy(),
                fatigue=0.0,
                base_return_prob=base_return,
                last_active=now - timedelta(days=int(rng.integers(0, 14))),
            )
        )

    items: List[ItemState] = []
    for iid in range(cfg.num_items):
        vec = rng.normal(0, 1, cfg.embedding_dim).astype(np.float32)
        vec /= (np.linalg.norm(vec) + 1e-8)
        quality = float(rng.uniform(0.3, 1.0))
        popularity = float(rng.beta(2, 8))
        items.append(ItemState(item_id=iid, vec=vec, quality=quality, popularity=popularity))

    return users, items


def recommend_top_k(user: UserState, items: List[ItemState], k: int) -> List[Tuple[int, float, float]]:
    """Return (item_id, score, affinity) for top-k items.

    score: dot(user.embed, item.vec) + small popularity bias (baseline retrieval)
    affinity: dot(user.true_pref, item.vec) * item.quality (simulated "true" match)
    """
    scores: List[Tuple[int, float, float]] = []
    for it in items:
        score = float(np.dot(user.embed, it.vec) + 0.05 * it.popularity)
        affinity = float(np.dot(user.true_pref, it.vec) * it.quality)
        scores.append((it.item_id, score, affinity))

    top = sorted(scores, key=lambda x: x[1], reverse=True)[:k]
    return top


def recommend_top_k_from_matrices(
    *,
    user_id: int,
    true_pref: np.ndarray,
    user_embed: np.ndarray,
    items_vec: np.ndarray,
    items_quality: np.ndarray,
    items_popularity: np.ndarray,
    k: int,
) -> List[Tuple[int, float, float]]:
    """Vectorized recommender using explicit embedding matrices.

    This is the key v1→v2 integration: rollout comparisons can now use different
    embedding versions for scoring, while keeping ground truth affinity separate.
    """
    # score uses current embedding geometry
    scores = items_vec @ user_embed + 0.05 * items_popularity
    # affinity uses true preference (ground truth proxy)
    affinity = (items_vec @ true_pref) * items_quality

    # top-k indices
    idx = np.argpartition(-scores, k - 1)[:k]
    # sort those indices by score
    idx = idx[np.argsort(-scores[idx])]

    out: List[Tuple[int, float, float]] = []
    for rank_i in idx:
        out.append((int(rank_i), float(scores[rank_i]), float(affinity[rank_i])))
    return out


def simulate_session(
    *,
    user: UserState,
    items: List[ItemState],
    cfg: SimConfig,
    rng: np.random.Generator,
    embedding_version: str,
    model_version: str,
    session_index: int,
) -> Tuple[List[Event], SessionSummary]:
    now = user.last_active + timedelta(hours=int(rng.integers(6, 72)))
    session_id = f"u{user.user_id}-s{session_index}-{int(now.timestamp())}"

    recs = recommend_top_k(user, items, cfg.k)

    events: List[Event] = []
    depth = 0.0

    for rank, (item_id, score, affinity) in enumerate(recs, start=1):
        events.append(
            Event(
                ts=now,
                user_id=user.user_id,
                item_id=item_id,
                event_type="impression",
                rank=rank,
                score=score,
                affinity=affinity,
                session_id=session_id,
                cohort=user.cohort,
                embedding_version=embedding_version,
                model_version=model_version,
            )
        )

        # Affinity helps; fatigue hurts.
        click_p = _sigmoid(2.2 * affinity - 1.0 * user.fatigue)
        play_p = _sigmoid(2.0 * affinity - 0.8 * user.fatigue)
        complete_p = _sigmoid(1.6 * affinity - 0.6 * user.fatigue)

        clicked = rng.random() < click_p
        if clicked:
            events.append(
                Event(
                    ts=now + timedelta(seconds=rank * 2),
                    user_id=user.user_id,
                    item_id=item_id,
                    event_type="click",
                    rank=rank,
                    score=score,
                    affinity=affinity,
                    session_id=session_id,
                    cohort=user.cohort,
                    embedding_version=embedding_version,
                    model_version=model_version,
                )
            )

        played = rng.random() < play_p
        if played:
            depth += 1.0
            user.fatigue = float(min(1.0, user.fatigue + cfg.fatigue_gain_per_play))
            events.append(
                Event(
                    ts=now + timedelta(seconds=rank * 3),
                    user_id=user.user_id,
                    item_id=item_id,
                    event_type="play",
                    rank=rank,
                    score=score,
                    affinity=affinity,
                    session_id=session_id,
                    cohort=user.cohort,
                    embedding_version=embedding_version,
                    model_version=model_version,
                )
            )

            completed = rng.random() < complete_p
            if completed:
                depth += 0.5
                events.append(
                    Event(
                        ts=now + timedelta(seconds=rank * 6),
                        user_id=user.user_id,
                        item_id=item_id,
                        event_type="complete",
                        rank=rank,
                        score=score,
                        affinity=affinity,
                        session_id=session_id,
                        cohort=user.cohort,
                        embedding_version=embedding_version,
                        model_version=model_version,
                    )
                )
        else:
            events.append(
                Event(
                    ts=now + timedelta(seconds=rank * 3),
                    user_id=user.user_id,
                    item_id=item_id,
                    event_type="skip",
                    rank=rank,
                    score=score,
                    affinity=affinity,
                    session_id=session_id,
                    cohort=user.cohort,
                    embedding_version=embedding_version,
                    model_version=model_version,
                )
            )

    days_since = max(0.0, (now - user.last_active).total_seconds() / 86400.0)
    retention = (
        user.base_return_prob
        + cfg.retention_gain_per_depth * depth
        - cfg.retention_decay_per_day * days_since
    )
    retention = float(np.clip(retention, 0.0, 1.0))

    summary = SessionSummary(
        session_id=session_id,
        user_id=user.user_id,
        cohort=user.cohort,
        started_at=now,
        ended_at=now + timedelta(minutes=10),
        k=cfg.k,
        engagement_depth=float(depth),
        retention_proxy=retention,
        embedding_version=embedding_version,
        model_version=model_version,
    )

    # Advance time + fatigue recovery
    user.last_active = now
    user.fatigue = float(max(0.0, user.fatigue - cfg.fatigue_recovery_per_day * days_since))

    return events, summary


def run_simulation(
    cfg: SimConfig,
    *,
    embedding_version: str = "emb_v1",
    model_version: str = "rank_v1",
) -> Tuple[List[Event], List[SessionSummary]]:
    rng = np.random.default_rng(cfg.seed)
    users, items = generate_world(cfg)

    all_events: List[Event] = []
    all_summaries: List[SessionSummary] = []

    session_counter = 0
    for u in users:
        for s in range(cfg.sessions_per_user):
            session_counter += 1

            # Controlled drift: perturb embedding vector occasionally.
            if session_counter % cfg.drift_every_n_sessions == 0:
                u.embed = (
                    u.embed
                    + rng.normal(0, cfg.drift_strength, cfg.embedding_dim).astype(np.float32)
                )
                u.embed /= (np.linalg.norm(u.embed) + 1e-8)

            ev, sm = simulate_session(
                user=u,
                items=items,
                cfg=cfg,
                rng=rng,
                embedding_version=embedding_version,
                model_version=model_version,
                session_index=s,
            )
            all_events.extend(ev)
            all_summaries.append(sm)

    return all_events, all_summaries


def run_simulation_with_embeddings(
    cfg: SimConfig,
    *,
    users_embed: np.ndarray,
    items_vec: np.ndarray,
    items_quality: np.ndarray,
    items_popularity: np.ndarray,
    embedding_version: str,
    model_version: str,
) -> Tuple[List[Event], List[SessionSummary]]:
    """Run simulator using explicit embedding matrices (v1/v2).

    This makes embedding geometry drift *real*: rollout comparisons can use different
    embedding versions for scoring without changing the rest of the simulator.
    """
    if users_embed.shape != (cfg.num_users, cfg.embedding_dim):
        raise ValueError(f"users_embed must be {(cfg.num_users, cfg.embedding_dim)}, got {users_embed.shape}")
    if items_vec.shape != (cfg.num_items, cfg.embedding_dim):
        raise ValueError(f"items_vec must be {(cfg.num_items, cfg.embedding_dim)}, got {items_vec.shape}")

    rng = np.random.default_rng(cfg.seed)

    # Create world, but overwrite embed/vec with provided matrices
    users, items = generate_world(cfg)

    for u in users:
        u.embed = users_embed[u.user_id].astype(np.float32)

    for it in items:
        it.vec = items_vec[it.item_id].astype(np.float32)
        it.quality = float(items_quality[it.item_id])
        it.popularity = float(items_popularity[it.item_id])

    all_events: List[Event] = []
    all_summaries: List[SessionSummary] = []

    for u in users:
        for s in range(cfg.sessions_per_user):
            now = u.last_active + timedelta(hours=int(rng.integers(6, 72)))
            session_id = f"u{u.user_id}-s{s}-{int(now.timestamp())}"

            # Vectorized recs from matrices
            recs = recommend_top_k_from_matrices(
                user_id=u.user_id,
                true_pref=u.true_pref,
                user_embed=u.embed,
                items_vec=items_vec,
                items_quality=items_quality,
                items_popularity=items_popularity,
                k=cfg.k,
            )

            events: List[Event] = []
            depth = 0.0

            for rank, (item_id, score, affinity) in enumerate(recs, start=1):
                events.append(
                    Event(
                        ts=now,
                        user_id=u.user_id,
                        item_id=item_id,
                        event_type="impression",
                        rank=rank,
                        score=score,
                        affinity=affinity,
                        session_id=session_id,
                        cohort=u.cohort,
                        embedding_version=embedding_version,
                        model_version=model_version,
                    )
                )

                click_p = _sigmoid(2.2 * affinity - 1.0 * u.fatigue)
                play_p = _sigmoid(2.0 * affinity - 0.8 * u.fatigue)
                complete_p = _sigmoid(1.6 * affinity - 0.6 * u.fatigue)

                if rng.random() < click_p:
                    events.append(
                        Event(
                            ts=now + timedelta(seconds=rank * 2),
                            user_id=u.user_id,
                            item_id=item_id,
                            event_type="click",
                            rank=rank,
                            score=score,
                            affinity=affinity,
                            session_id=session_id,
                            cohort=u.cohort,
                            embedding_version=embedding_version,
                            model_version=model_version,
                        )
                    )

                played = rng.random() < play_p
                if played:
                    depth += 1.0
                    u.fatigue = float(min(1.0, u.fatigue + cfg.fatigue_gain_per_play))
                    events.append(
                        Event(
                            ts=now + timedelta(seconds=rank * 3),
                            user_id=u.user_id,
                            item_id=item_id,
                            event_type="play",
                            rank=rank,
                            score=score,
                            affinity=affinity,
                            session_id=session_id,
                            cohort=u.cohort,
                            embedding_version=embedding_version,
                            model_version=model_version,
                        )
                    )

                    if rng.random() < complete_p:
                        depth += 0.5
                        events.append(
                            Event(
                                ts=now + timedelta(seconds=rank * 6),
                                user_id=u.user_id,
                                item_id=item_id,
                                event_type="complete",
                                rank=rank,
                                score=score,
                                affinity=affinity,
                                session_id=session_id,
                                cohort=u.cohort,
                                embedding_version=embedding_version,
                                model_version=model_version,
                            )
                        )
                else:
                    events.append(
                        Event(
                            ts=now + timedelta(seconds=rank * 3),
                            user_id=u.user_id,
                            item_id=item_id,
                            event_type="skip",
                            rank=rank,
                            score=score,
                            affinity=affinity,
                            session_id=session_id,
                            cohort=u.cohort,
                            embedding_version=embedding_version,
                            model_version=model_version,
                        )
                    )

            days_since = max(0.0, (now - u.last_active).total_seconds() / 86400.0)
            retention = (
                u.base_return_prob
                + cfg.retention_gain_per_depth * depth
                - cfg.retention_decay_per_day * days_since
            )
            retention = float(np.clip(retention, 0.0, 1.0))

            all_summaries.append(
                SessionSummary(
                    session_id=session_id,
                    user_id=u.user_id,
                    cohort=u.cohort,
                    started_at=now,
                    ended_at=now + timedelta(minutes=10),
                    k=cfg.k,
                    engagement_depth=float(depth),
                    retention_proxy=retention,
                    embedding_version=embedding_version,
                    model_version=model_version,
                )
            )

            all_events.extend(events)
            u.last_active = now
            u.fatigue = float(max(0.0, u.fatigue - cfg.fatigue_recovery_per_day * days_since))

    return all_events, all_summaries
