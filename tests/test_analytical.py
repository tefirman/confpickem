#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""Tests for the analytical (Poisson-binomial) win-probability model."""

import numpy as np
import pytest

from src.confpickem.analytical import (
    weighted_pmf,
    prob_a_beats_b,
    modal_opponent,
    build_opponent_types,
    sample_outcomes,
    make_pwin,
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
