from drpe.data.simulator import SimConfig, run_simulation


def test_simulation_runs():
    cfg = SimConfig(num_users=50, num_items=200, sessions_per_user=2, k=10)
    events, summaries = run_simulation(cfg)
    assert len(events) > 0
    assert len(summaries) == cfg.num_users * cfg.sessions_per_user
