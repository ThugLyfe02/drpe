from drpe.reporting.model_card import embedding_rollout_card, ranker_rollout_card


def test_model_card_renders():
    c1 = embedding_rollout_card(
        baseline_depth=5.0,
        baseline_ret=0.50,
        candidate_depth=5.1,
        candidate_ret=0.49,
        depth_kl=0.01,
        ret_kl=0.02,
        emb_users_mean=0.01,
        emb_items_mean=0.02,
        cohort_user_drift={"core": 0.01},
        decision_allow=False,
        decision_reason="blocked",
        guardrail_max_ret_drop=0.01,
        guardrail_max_emb_mean=0.12,
    )
    assert "MODEL CARD" in c1.render()

    c2 = ranker_rollout_card(
        mode="safe",
        baseline_depth=6.0,
        baseline_ret=0.50,
        candidate_depth=6.1,
        candidate_ret=0.49,
        decision_allow=False,
        decision_reason="blocked",
        max_ret_drop=0.01,
        gamma_retention=0.25,
        retention_fatigue_penalty_candidate=0.0,
    )
    assert "Ranker Rollout Gate" in c2.render()
