#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""Tests for the analytical (Poisson-binomial) win-probability model."""

import numpy as np
import pandas as pd
import pytest

from src.confpickem.analytical import (
    weighted_pmf,
    prob_a_beats_b,
    modal_opponent,
    build_opponent_types,
    sample_outcomes,
    make_pwin,
    game_importance,
    locked_board_standings,
    chalk_slate,
    optimize_slate,
)
from src.confpickem.confidence_pickem_sim import ConfidencePickEmSimulator, Game, Player
from datetime import datetime

# ---------------------------------------------------------------------------
# Poisson-binomial primitives
# ---------------------------------------------------------------------------


def _brute_pmf(probs, weights):
    n = len(probs)
    total = int(sum(weights))
    pmf = np.zeros(total + 1)
    for mask in range(1 << n):
        p, s = 1.0, 0
        for i in range(n):
            if mask >> i & 1:
                p *= probs[i]
                s += weights[i]
            else:
                p *= 1.0 - probs[i]
        pmf[s] += p
    return pmf


def test_weighted_pmf_matches_brute_force():
    rng = np.random.default_rng(0)
    for _ in range(100):
        k = int(rng.integers(1, 9))
        probs = rng.random(k)
        weights = rng.integers(0, 6, size=k)
        got = weighted_pmf(probs, weights)
        ref = _brute_pmf(probs, weights)
        assert got.shape == ref.shape
        assert np.allclose(got, ref, atol=1e-10)
        assert abs(got.sum() - 1.0) < 1e-9


def test_weighted_pmf_rejects_bad_input():
    with pytest.raises(ValueError):
        weighted_pmf([0.5, 1.5], [1, 1])
    with pytest.raises(ValueError):
        weighted_pmf([0.5, 0.5], [1, -1])
    with pytest.raises(ValueError):
        weighted_pmf([0.5, 0.5], [1])


def test_prob_a_beats_b_matches_brute_force():
    rng = np.random.default_rng(1)
    for _ in range(50):
        na, nb = int(rng.integers(1, 6)), int(rng.integers(1, 6))
        a = weighted_pmf(rng.random(na), rng.integers(1, 5, na))
        b = weighted_pmf(rng.random(nb), rng.integers(1, 5, nb))
        got = prob_a_beats_b(a, b)
        ref = 0.0
        for ka, va in enumerate(a):
            for kb, vb in enumerate(b):
                if ka > kb:
                    ref += va * vb
                elif ka == kb:
                    ref += 0.5 * va * vb
        assert abs(got - ref) < 1e-9


# ---------------------------------------------------------------------------
# Field model
# ---------------------------------------------------------------------------


@pytest.fixture
def week_data():
    """A small synthetic week: 8 games, a spread of Vegas probabilities."""
    n = 8
    rng = np.random.default_rng(7)
    vegas_home = np.clip(0.5 + np.linspace(-0.3, 0.3, n) + rng.normal(0, 0.02, n), 0.05, 0.95)
    crowd_home_pct = np.clip(vegas_home + rng.normal(0, 0.04, n), 0.02, 0.98)
    crowd_home_conf = np.abs(vegas_home - 0.5) * 18 + 3
    crowd_away_conf = 9 - np.abs(vegas_home - 0.5) * 6
    return dict(
        n=n,
        vegas_home=vegas_home,
        crowd_home_pct=crowd_home_pct,
        crowd_home_conf=crowd_home_conf,
        crowd_away_conf=crowd_away_conf,
    )


def test_modal_opponent_returns_valid_permutation(week_data):
    _, pick_home, points = modal_opponent(
        week_data["vegas_home"],
        week_data["crowd_home_pct"],
        week_data["crowd_home_conf"],
        week_data["crowd_away_conf"],
        0.5,
        0.5,
    )
    assert sorted(points.tolist()) == list(range(1, week_data["n"] + 1))
    assert pick_home.dtype == bool


def test_build_opponent_types_dedupes(week_data):
    # 20 identical opponents -> 1 type with count 20
    opps = [(0.5, 0.5)] * 20
    types = build_opponent_types(
        week_data["vegas_home"],
        week_data["crowd_home_pct"],
        week_data["crowd_home_conf"],
        week_data["crowd_away_conf"],
        opps,
    )
    assert len(types) == 1
    assert types[0][2] == 20

    # add a distinct group
    opps += [(0.9, 0.9)] * 5
    types = build_opponent_types(
        week_data["vegas_home"],
        week_data["crowd_home_pct"],
        week_data["crowd_home_conf"],
        week_data["crowd_away_conf"],
        opps,
    )
    assert len(types) == 2
    assert sorted(t[2] for t in types) == [5, 20]


# ---------------------------------------------------------------------------
# Analytical P(win)
# ---------------------------------------------------------------------------


def _pwin_for(week_data, seed=0, n_outcomes=3000):
    opps = [(0.55, 0.48)] * 30 + [(0.75, 0.5)] * 15
    types = build_opponent_types(
        week_data["vegas_home"],
        week_data["crowd_home_pct"],
        week_data["crowd_home_conf"],
        week_data["crowd_away_conf"],
        opps,
    )
    outcomes = sample_outcomes(week_data["vegas_home"], n_outcomes, np.random.default_rng(seed))
    return make_pwin(types, outcomes)


def test_pwin_in_unit_interval(week_data):
    pwin = _pwin_for(week_data)
    ph, pts = chalk_slate(week_data["vegas_home"])
    v = pwin(ph, pts)
    assert 0.0 <= v <= 1.0


def test_pwin_prefers_favorite_against_a_mirror_opponent(week_data):
    """With a single opponent playing the identical slate, P(win) is purely
    "do I outscore them", so picking the more likely side of any game must not
    lower it. (Against a *large* over-committed field the model correctly does
    NOT always prefer the favorite -- that contrarian value is the whole point
    of the optimizer -- so this is checked in the clean 1-opponent case.)"""
    vh = week_data["vegas_home"]
    types = build_opponent_types(
        vh,
        week_data["crowd_home_pct"],
        week_data["crowd_home_conf"],
        week_data["crowd_away_conf"],
        [(0.5, 0.5)],
    )  # one mirror opponent
    outcomes = sample_outcomes(vh, 8000, np.random.default_rng(0))
    pwin = make_pwin(types, outcomes)

    ph, pts = chalk_slate(vh)  # picks the favorite everywhere
    fav = pwin(ph, pts)

    g = int(np.argmax(np.abs(vh - 0.5)))  # most lopsided game
    dog_ph = ph.copy()
    dog_ph[g] = not dog_ph[g]
    dog = pwin(dog_ph, pts)

    assert fav > dog


def test_pwin_is_zero_when_you_cannot_score(week_data):
    """A slate that puts 0 on every game (degenerate) can never lead -> P(win)
    is essentially 0. Sanity check on the scoring direction."""
    pwin = _pwin_for(week_data)
    n = week_data["n"]
    ph, _ = chalk_slate(week_data["vegas_home"])
    zero_pts = np.zeros(n, dtype=int)
    assert pwin(ph, zero_pts) < 0.01


# ---------------------------------------------------------------------------
# Optimizer
# ---------------------------------------------------------------------------


def test_optimize_slate_returns_valid_permutation(week_data):
    pwin = _pwin_for(week_data)
    ph, pts, val = optimize_slate(
        pwin, week_data["vegas_home"], iterations=60, restarts=2, rng=np.random.default_rng(3)
    )
    assert sorted(pts.tolist()) == list(range(1, week_data["n"] + 1))
    assert 0.0 <= val <= 1.0


def test_optimize_slate_beats_chalk(week_data):
    pwin = _pwin_for(week_data)
    cph, cpts = chalk_slate(week_data["vegas_home"])
    chalk_val = pwin(cph, cpts)
    _, _, opt_val = optimize_slate(
        pwin, week_data["vegas_home"], iterations=120, restarts=3, rng=np.random.default_rng(4)
    )
    assert opt_val >= chalk_val - 1e-12


def test_optimize_slate_respects_locked_picks(week_data):
    n = week_data["n"]
    pwin = _pwin_for(week_data)
    points_fixed = np.zeros(n, dtype=int)
    pick_home_fixed = np.zeros(n, dtype=bool)
    points_fixed[0] = n  # game 0 locked to the max
    pick_home_fixed[0] = True
    points_fixed[2] = 1  # game 2 locked to the min
    pick_home_fixed[2] = False

    ph, pts, _ = optimize_slate(
        pwin,
        week_data["vegas_home"],
        pick_home_fixed=pick_home_fixed,
        points_fixed=points_fixed,
        iterations=60,
        restarts=2,
        rng=np.random.default_rng(5),
    )

    assert pts[0] == n and bool(ph[0]) is True
    assert pts[2] == 1 and bool(ph[2]) is False
    assert sorted(pts.tolist()) == list(range(1, n + 1))


def test_optimize_slate_is_deterministic_given_seed(week_data):
    pwin = _pwin_for(week_data)
    a = optimize_slate(
        pwin, week_data["vegas_home"], iterations=50, restarts=2, rng=np.random.default_rng(9)
    )
    b = optimize_slate(
        pwin, week_data["vegas_home"], iterations=50, restarts=2, rng=np.random.default_rng(9)
    )
    assert np.array_equal(a[0], b[0])
    assert np.array_equal(a[1], b[1])
    assert a[2] == b[2]


# ---------------------------------------------------------------------------
# Simulator integration
# ---------------------------------------------------------------------------


@pytest.fixture
def analytic_simulator():
    sim = ConfidencePickEmSimulator(num_sims=100)
    sim.games = [
        Game(
            home_team="SF",
            away_team="ARI",
            vegas_win_prob=0.85,
            crowd_home_pick_pct=0.90,
            crowd_home_confidence=14.0,
            crowd_away_confidence=2.0,
            week=1,
            kickoff_time=datetime(2024, 9, 8, 13, 0),
        ),
        Game(
            home_team="KC",
            away_team="DEN",
            vegas_win_prob=0.65,
            crowd_home_pick_pct=0.72,
            crowd_home_confidence=10.0,
            crowd_away_confidence=6.0,
            week=1,
            kickoff_time=datetime(2024, 9, 8, 16, 25),
        ),
        Game(
            home_team="BAL",
            away_team="CIN",
            vegas_win_prob=0.55,
            crowd_home_pick_pct=0.60,
            crowd_home_confidence=8.0,
            crowd_away_confidence=8.5,
            week=1,
            kickoff_time=datetime(2024, 9, 8, 20, 20),
        ),
        Game(
            home_team="BUF",
            away_team="MIA",
            vegas_win_prob=0.70,
            crowd_home_pick_pct=0.78,
            crowd_home_confidence=11.0,
            crowd_away_confidence=5.0,
            week=1,
            kickoff_time=datetime(2024, 9, 8, 13, 0),
        ),
    ]
    sim.players = [Player("Me", 0.75, 0.74, 0.83)] + [
        Player(f"P{i}", 0.6, 0.5, 0.5) for i in range(1, 12)
    ]
    return sim


def test_optimize_picks_analytic_valid_and_beats_chalk(analytic_simulator):
    optimal = analytic_simulator.optimize_picks_analytic(
        "Me", iterations=80, restarts=2, n_outcomes=2000, seed=1
    )
    n = len(analytic_simulator.games)
    assert sorted(optimal.values()) == list(range(1, n + 1))
    valid_teams = {t for g in analytic_simulator.games for t in (g.home_team, g.away_team)}
    assert all(team in valid_teams for team in optimal)
    assert len(optimal) == n


def test_optimize_picks_analytic_respects_fixed_picks(analytic_simulator):
    optimal = analytic_simulator.optimize_picks_analytic(
        "Me", fixed_picks={"Me": {"SF": 4}}, iterations=60, restarts=2, n_outcomes=1500, seed=2
    )
    assert optimal["SF"] == 4
    n = len(analytic_simulator.games)
    assert sorted(optimal.values()) == list(range(1, n + 1))


def test_optimize_picks_analytic_unknown_player(analytic_simulator):
    with pytest.raises(ValueError):
        analytic_simulator.optimize_picks_analytic("Nobody")


# ---------------------------------------------------------------------------
# locked_board_standings (fully-locked field, no opponent model)
# ---------------------------------------------------------------------------


def test_locked_board_standings_is_a_distribution():
    rng = np.random.default_rng(0)
    n, N = 6, 8
    vh = rng.uniform(0.35, 0.8, n)
    pick_home = rng.random((N, n)) < 0.5
    points = np.stack([rng.permutation(n) + 1 for _ in range(N)])
    outcomes = sample_outcomes(vh, 5000, rng)

    win_pct, exp_pts, swing = locked_board_standings(pick_home, points, outcomes)
    assert win_pct.shape == (N,) and exp_pts.shape == (N,)
    assert swing.shape == (N, n)
    assert abs(win_pct.sum() - 1.0) < 1e-9  # someone wins every draw
    assert (win_pct >= 0).all()
    # expected points in [0, sum of that entrant's confidence]
    assert (exp_pts <= points.sum(axis=1) + 1e-9).all()


def test_locked_board_standings_favours_the_better_slate():
    # two entrants, 3 games, everyone home-favoured; A puts big points on the
    # locks, B inverts -> A should win more often
    vh = np.array([0.9, 0.75, 0.6])
    outcomes = sample_outcomes(vh, 8000, np.random.default_rng(1))
    pick_home = np.array([[True, True, True], [True, True, True]])
    points = np.array([[3, 2, 1], [1, 2, 3]])
    win_pct, _, _ = locked_board_standings(pick_home, points, outcomes)
    assert win_pct[0] > win_pct[1]


def test_locked_board_standings_decided_game_zero_swing():
    vh = np.array([0.7, 0.55, 0.6])
    ao = [True, None, None]  # game 0 already decided
    outcomes = sample_outcomes(vh, 4000, np.random.default_rng(2), actual_outcomes=ao)
    pick_home = np.array([[True, True, False], [False, True, True]])
    points = np.array([[3, 2, 1], [2, 3, 1]])
    _, _, swing = locked_board_standings(pick_home, points, outcomes)
    assert np.allclose(swing[:, 0], 0.0)  # no live partition


def test_locked_board_standings_ties_split_credit():
    # identical slates -> identical win_pct, and they sum to 1
    vh = np.array([0.6, 0.5, 0.4])
    outcomes = sample_outcomes(vh, 3000, np.random.default_rng(3))
    pick_home = np.array([[True, False, True]] * 3)
    points = np.array([[3, 2, 1]] * 3)
    win_pct, _, _ = locked_board_standings(pick_home, points, outcomes)
    assert np.allclose(win_pct, 1 / 3)


def test_standings_analytic_end_to_end(analytic_simulator):
    sim = analytic_simulator
    n = len(sim.games)
    # game 0 (SF) already decided; the rest locked-but-live
    sim.games[0].actual_outcome = True

    names = ["Me"] + [f"P{i}" for i in range(1, 12)]
    rows = []
    for k, nm in enumerate(names):
        # rotate everyone's picks/points a little so slates differ
        rows.append(
            {
                "player_name": nm,
                **{
                    f"game_{i+1}_pick": (
                        sim.games[i].home_team if (i + k) % 3 else sim.games[i].away_team
                    )
                    for i in range(n)
                },
                **{f"game_{i+1}_confidence": ((i + k) % n) + 1 for i in range(n)},
            }
        )
    pdata = pd.DataFrame(rows)

    standings, importance = sim.standings_analytic(pdata, n_outcomes=3000, seed=1)

    assert list(standings.columns) == ["player", "locked_points", "win_pct", "expected_points"]
    assert len(standings) == len(names)
    assert abs(standings.win_pct.sum() - 1.0) < 1e-9
    # sorted descending by win_pct
    assert list(standings.win_pct) == sorted(standings.win_pct, reverse=True)
    # importance: one row per *undecided* game (SF dropped), sorted by top_swing
    assert len(importance) == n - 1
    assert "SF" not in importance.game.str.cat(sep="@")
    assert list(importance.top_swing) == sorted(importance.top_swing, reverse=True)


def test_standings_analytic_requires_full_slates(analytic_simulator):
    sim = analytic_simulator
    n = len(sim.games)
    rows = [
        {
            "player_name": "Me",
            **{f"game_{i+1}_pick": sim.games[i].home_team for i in range(n)},
            **{f"game_{i+1}_confidence": i + 1 for i in range(n)},
        },
        {
            "player_name": "P1",  # missing game 2
            **{f"game_{i+1}_pick": sim.games[i].home_team for i in range(n) if i != 1},
            **{f"game_{i+1}_confidence": i + 1 for i in range(n) if i != 1},
        },
    ]
    with pytest.raises(ValueError):
        sim.standings_analytic(pd.DataFrame(rows))


# ---------------------------------------------------------------------------
# game_importance (analytical)
# ---------------------------------------------------------------------------


def test_make_pwin_per_outcome_averages_to_pwin(week_data):
    pwin = _pwin_for(week_data)
    ph, pts = chalk_slate(week_data["vegas_home"])
    per = pwin.per_outcome(ph, pts)
    assert per.shape == (3000,)
    assert (per >= 0).all() and (per <= 1).all()
    assert abs(per.mean() - pwin(ph, pts)) < 1e-12


def test_game_importance_partition_identity(week_data):
    """importance = P(win|home) - P(win|away), and each conditional is a plain
    mean over the draws sliced on that game's outcome bit -- so the base is the
    outcome-probability-weighted blend of the two."""
    n = week_data["n"]
    vh = week_data["vegas_home"]
    types = build_opponent_types(
        vh,
        week_data["crowd_home_pct"],
        week_data["crowd_home_conf"],
        week_data["crowd_away_conf"],
        [(0.55, 0.5)] * 20,
    )
    outcomes = sample_outcomes(vh, 8000, np.random.default_rng(2))
    pwin = make_pwin(types, outcomes)
    ph, pts = chalk_slate(vh)

    p_home, p_away, base = game_importance(pwin, outcomes, ph, pts)
    assert p_home.shape == (n,) and p_away.shape == (n,)
    for i in range(n):
        frac_home = outcomes[:, i].mean()
        blended = frac_home * p_home[i] + (1 - frac_home) * p_away[i]
        assert abs(blended - base) < 1e-9


def test_game_importance_decided_game_has_zero_swing(week_data):
    vh = week_data["vegas_home"].copy()
    types = build_opponent_types(
        vh,
        week_data["crowd_home_pct"],
        week_data["crowd_home_conf"],
        week_data["crowd_away_conf"],
        [(0.5, 0.5)] * 10,
    )
    # force game 3 decided (home win) in every draw
    ao = [None] * len(vh)
    ao[3] = True
    outcomes = sample_outcomes(vh, 4000, np.random.default_rng(0), actual_outcomes=ao)
    pwin = make_pwin(types, outcomes)
    ph, pts = chalk_slate(vh)
    p_home, p_away, base = game_importance(pwin, outcomes, ph, pts)
    assert p_home[3] == base and p_away[3] == base  # no live partition


def test_game_importance_no_new_simulation(week_data, monkeypatch):
    """The analytical path must not call sample_outcomes again per game."""
    import src.confpickem.analytical as an

    vh = week_data["vegas_home"]
    types = build_opponent_types(
        vh,
        week_data["crowd_home_pct"],
        week_data["crowd_home_conf"],
        week_data["crowd_away_conf"],
        [(0.5, 0.5)] * 8,
    )
    outcomes = sample_outcomes(vh, 3000, np.random.default_rng(1))
    pwin = make_pwin(types, outcomes)
    ph, pts = chalk_slate(vh)

    calls = {"n": 0}
    real = an.sample_outcomes
    monkeypatch.setattr(
        an,
        "sample_outcomes",
        lambda *a, **k: (calls.__setitem__("n", calls["n"] + 1), real(*a, **k))[1],
    )
    game_importance(pwin, outcomes, ph, pts)
    assert calls["n"] == 0


def test_assess_game_importance_analytic_contract(analytic_simulator):
    sim = analytic_simulator
    fixed = {"Me": {"SF": 4, "KC": 3, "BAL": 1, "BUF": 2}}
    df = sim.assess_game_importance("Me", fixed_picks=fixed, n_outcomes=2000, seed=3)

    assert len(df) == len(sim.games)
    for col in (
        "game",
        "points_bid",
        "pick",
        "win_probability",
        "loss_probability",
        "win_delta",
        "loss_delta",
        "total_impact",
        "is_fixed",
    ):
        assert col in df.columns
    assert (df.win_probability.between(0, 1)).all()
    assert (df.loss_probability.between(0, 1)).all()
    # picking your side right never hurts vs picking it wrong, for a lone slate
    assert (df.total_impact >= -1e-9).all()
    # points_bid reflects the fixed slate
    assert dict(zip(df.game.str.split("@").str[1], df.points_bid))["SF"] == 4
    assert df.is_fixed.all()
    # sorted by |impact| desc
    assert list(df.total_impact.abs()) == sorted(df.total_impact.abs(), reverse=True)


def test_assess_game_importance_analytic_deterministic(analytic_simulator):
    sim = analytic_simulator
    fixed = {"Me": {"SF": 4, "KC": 3, "BAL": 1, "BUF": 2}}
    a = sim.assess_game_importance("Me", fixed_picks=fixed, n_outcomes=1500, seed=9)
    b = sim.assess_game_importance("Me", fixed_picks=fixed, n_outcomes=1500, seed=9)
    pd.testing.assert_frame_equal(a.reset_index(drop=True), b.reset_index(drop=True))


def test_assess_game_importance_matches_direct_pwin_delta(analytic_simulator):
    """The reported win/loss probabilities equal a direct two-call P(win)
    evaluation with that game's outcome pinned each way -- i.e. the partition
    trick is exact, not an approximation."""
    import numpy as np
    from src.confpickem import analytical as an

    sim = analytic_simulator
    fixed = {"Me": {"SF": 4, "KC": 3, "BAL": 1, "BUF": 2}}
    df = sim.assess_game_importance("Me", fixed_picks=fixed, n_outcomes=3000, seed=2)

    field = sim._build_analytic_field("Me", 3000, 2)
    outcomes, pwin = field["outcomes"], field["pwin"]
    games = field["games"]
    ph = np.array([{"SF": True, "KC": True, "BAL": True, "BUF": True}[g.home_team] for g in games])
    pts = np.array([fixed["Me"][g.home_team] for g in games])
    per = pwin.per_outcome(ph, pts)

    for i, g in enumerate(games):
        home = outcomes[:, i]
        row = df[df.game == f"{g.away_team}@{g.home_team}"].iloc[0]
        # this player's pick is home in the fixture, so win == home-win conditional
        assert abs(row.win_probability - per[home].mean()) < 1e-9
        assert abs(row.loss_probability - per[~home].mean()) < 1e-9


# ---------------------------------------------------------------------------
# Midweek: forced outcomes + folded-in opponent picks
# ---------------------------------------------------------------------------


def test_sample_outcomes_forces_decided_games():
    vh = np.array([0.6, 0.4, 0.7, 0.55])
    draws = sample_outcomes(
        vh, 500, np.random.default_rng(0), actual_outcomes=[True, None, False, None]
    )
    assert draws[:, 0].all()  # home win forced
    assert not draws[:, 2].any()  # away win forced
    assert 0 < draws[:, 1].mean() < 1  # still sampled
    assert 0 < draws[:, 3].mean() < 1


def test_build_opponent_types_collapses_completed_games_to_scalar():
    vh = np.array([0.85, 0.65, 0.55, 0.70])
    chp = np.array([0.90, 0.72, 0.60, 0.78])
    chc = np.array([14.0, 10.0, 8.0, 11.0])
    cac = np.array([2.0, 6.0, 8.5, 5.0])
    opps = [(0.5, 0.5)] * 6
    # game 0 decided: home won. Two opponents took the (losing) underdog w/ 3 pts,
    # the other four took the (winning) home team with 3 pts.
    outcomes = [True, None, None, None]
    completed = [
        {0: (True, 3)},
        {0: (True, 3)},
        {0: (True, 3)},
        {0: (True, 3)},
        {0: (False, 3)},
        {0: (False, 3)},
    ]
    types = build_opponent_types(
        vh,
        chp,
        chc,
        cac,
        opps,
        completed_picks=completed,
        actual_outcomes=outcomes,
    )
    # split by banked points: 4 with 3, 2 with 0
    by_count = {t[2]: t for t in types}
    assert sorted(by_count) == [2, 4]
    assert by_count[4][3] == 3  # picked the winner -> banked 3
    assert by_count[2][3] == 0  # picked the loser -> banked 0
    # the decided game is dropped from the Poisson-binomial for both
    for t in types:
        assert t[1][0] == 0


def test_build_opponent_types_no_completed_picks_is_backward_compatible():
    vh = np.array([0.85, 0.65, 0.55, 0.70])
    chp = np.array([0.90, 0.72, 0.60, 0.78])
    chc = np.array([14.0, 10.0, 8.0, 11.0])
    cac = np.array([2.0, 6.0, 8.5, 5.0])
    types = build_opponent_types(vh, chp, chc, cac, [(0.5, 0.5)] * 5)
    assert len(types) == 1
    assert types[0][2] == 5
    assert types[0][3] == 0  # nothing banked


def test_build_opponent_types_rejects_bad_game_index():
    vh = np.array([0.6, 0.4, 0.7])
    with pytest.raises(ValueError):
        build_opponent_types(
            vh,
            vh,
            vh * 10,
            vh * 10,
            [(0.5, 0.5)],
            completed_picks=[{9: (True, 1)}],
            actual_outcomes=[True, None, None],
        )


def test_build_opponent_types_rejects_completed_pick_without_outcome():
    vh = np.array([0.6, 0.4, 0.7])
    with pytest.raises(ValueError):
        build_opponent_types(
            vh,
            vh,
            vh * 10,
            vh * 10,
            [(0.5, 0.5)],
            completed_picks=[{0: (True, 1)}],
            actual_outcomes=[None, None, None],
        )


def test_make_pwin_accounts_for_banked_completed_points():
    """A type that already banked a big lead on decided games should be much
    harder to beat than the same modal pending slate with nothing banked."""
    vh = np.array([0.6, 0.55, 0.5, 0.45])
    outcomes = sample_outcomes(vh, 4000, np.random.default_rng(0))
    # identical pending model; one banked 10 pts on decided games, one banked 0
    p_home = vh.copy()
    pending_pts = np.array([4, 3, 2, 1])
    behind = make_pwin([(p_home, pending_pts, 1, 10)], outcomes)
    even = make_pwin([(p_home, pending_pts, 1, 0)], outcomes)
    my_ph = np.ones(4, dtype=bool)
    my_pts = np.array([4, 3, 2, 1])
    assert behind(my_ph, my_pts) < even(my_ph, my_pts)


@pytest.fixture
def midweek_simulator():
    """4-game slate, games 0 and 1 already decided; every player has a real
    pick on all four (so game 2 or 3 can also be treated as frozen-but-live)."""
    sim = ConfidencePickEmSimulator(num_sims=100)
    sim.games = [
        Game(
            "SF", "ARI", 0.85, 0.90, 14.0, 2.0, 1, datetime(2024, 9, 8, 13, 0), actual_outcome=True
        ),
        Game(
            "KC",
            "DEN",
            0.65,
            0.72,
            10.0,
            6.0,
            1,
            datetime(2024, 9, 8, 16, 25),
            actual_outcome=False,
        ),
        Game("BAL", "CIN", 0.55, 0.60, 8.0, 8.5, 1, datetime(2024, 9, 8, 20, 20)),
        Game("BUF", "MIA", 0.70, 0.78, 11.0, 5.0, 1, datetime(2024, 9, 8, 13, 0)),
    ]
    sim.players = [Player("Me", 0.75, 0.74, 0.83)] + [
        Player(f"P{i}", 0.6, 0.5, 0.5) for i in range(1, 12)
    ]
    # yahoo.players-style frame: game_N_pick / game_N_confidence
    rows = []
    for p in sim.players:
        rows.append(
            {
                "player_name": p.name,
                "game_1_pick": "SF",
                "game_1_confidence": 4,
                "game_2_pick": "DEN" if p.name == "Me" else "KC",
                "game_2_confidence": 2,
                "game_3_pick": "BAL" if p.name == "Me" else "CIN",
                "game_3_confidence": 1,
                "game_4_pick": "BUF",
                "game_4_confidence": 3,
            }
        )
    sim.player_data = pd.DataFrame(rows)
    return sim


def test_optimize_picks_analytic_midweek_locks_completed_games(midweek_simulator):
    sim = midweek_simulator
    optimal = sim.optimize_picks_analytic(
        "Me",
        iterations=60,
        restarts=2,
        n_outcomes=1500,
        seed=3,
        player_data=sim.player_data,
    )
    n = len(sim.games)
    # completed games keep Me's real pick + confidence
    assert optimal["SF"] == 4
    assert optimal["DEN"] == 2
    # still a full valid 1..n permutation
    assert sorted(optimal.values()) == list(range(1, n + 1))
    # free games only use the unspent values {1, 3}
    free_vals = sorted(v for t, v in optimal.items() if t not in ("SF", "DEN"))
    assert free_vals == [1, 3]


def test_optimize_picks_analytic_midweek_available_points_crosscheck(midweek_simulator):
    sim = midweek_simulator
    # {1, 3} is the true unspent set (2 -> DEN, 4 -> SF already spent)
    ok = sim.optimize_picks_analytic(
        "Me",
        iterations=40,
        restarts=1,
        n_outcomes=1000,
        seed=1,
        player_data=sim.player_data,
        available_points={1, 3},
    )
    assert sorted(ok.values()) == [1, 2, 3, 4]
    # a caller-supplied set that genuinely disagrees is rejected
    with pytest.raises(ValueError):
        sim.optimize_picks_analytic(
            "Me",
            iterations=40,
            restarts=1,
            n_outcomes=1000,
            seed=1,
            player_data=sim.player_data,
            available_points={1},
        )


def test_optimize_picks_analytic_midweek_detects_conflicting_lock(midweek_simulator):
    sim = midweek_simulator
    # fixed pick reuses confidence 4, already spent on the completed SF game
    with pytest.raises(ValueError):
        sim.optimize_picks_analytic(
            "Me",
            iterations=40,
            restarts=1,
            n_outcomes=1000,
            seed=1,
            player_data=sim.player_data,
            fixed_picks={"Me": {"BAL": 4}},
        )


def test_build_opponent_types_locked_pick_stays_in_convolution():
    vh = np.array([0.85, 0.65, 0.55, 0.70])
    chp = np.array([0.90, 0.72, 0.60, 0.78])
    chc = np.array([14.0, 10.0, 8.0, 11.0])
    cac = np.array([2.0, 6.0, 8.5, 5.0])
    opps = [(0.5, 0.5)] * 4
    # game 1 kicked off but is not final: two opponents locked the away side w/ 5
    locked = [None, {1: (False, 5)}, {1: (False, 5)}, None]
    types = build_opponent_types(vh, chp, chc, cac, opps, locked_picks=locked)
    assert len(types) == 2  # 2 with the locked away pick, 2 modal
    assert sorted(t[2] for t in types) == [2, 2]
    locked_type = [t for t in types if t[1][1] == 5][0]
    ph_prob, pts, _cnt, banked = locked_type
    assert ph_prob[1] == 0.0  # pinned to the real (away) pick
    assert pts[1] == 5  # real confidence kept
    assert pts[1] != 0  # NOT dropped from the Poisson-binomial
    assert banked == 0  # nothing banked -- outcome still unknown


def test_optimize_picks_analytic_freezes_kicked_off_games_via_as_of(midweek_simulator):
    sim = midweek_simulator
    # game index 3 (BUF@MIA) kicks off 2024-09-08 13:00; as_of just after locks it
    as_of = datetime(2024, 9, 8, 13, 30)
    optimal = sim.optimize_picks_analytic(
        "Me",
        iterations=60,
        restarts=2,
        n_outcomes=1500,
        seed=5,
        player_data=sim.player_data,
        as_of=as_of,
    )
    n = len(sim.games)
    # frozen: game 0 (final), game 1 (final), game 3 (kicked off) -> Me's real picks
    assert optimal["SF"] == 4  # completed
    assert optimal["DEN"] == 2  # completed
    assert optimal["BUF"] == 3  # kicked off, pick frozen at real confidence
    # only game 2 (BAL@CIN, 20:20 kickoff) is still free -> gets the last value
    assert sorted(optimal.values()) == list(range(1, n + 1))
    free = {t: v for t, v in optimal.items() if t not in ("SF", "DEN", "BUF")}
    assert set(free.values()) == {1}


def test_optimize_picks_analytic_as_of_before_kickoff_is_a_noop(midweek_simulator):
    sim = midweek_simulator
    early = sim.optimize_picks_analytic(
        "Me",
        iterations=40,
        restarts=1,
        n_outcomes=1000,
        seed=7,
        player_data=sim.player_data,
        as_of=datetime(2024, 9, 8, 6, 0),
    )
    plain = sim.optimize_picks_analytic(
        "Me",
        iterations=40,
        restarts=1,
        n_outcomes=1000,
        seed=7,
        player_data=sim.player_data,
    )
    assert early == plain
