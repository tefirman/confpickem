"""Offline tests for the season-long simulator (no Yahoo / no PickEmCache needed)."""

import numpy as np
import pandas as pd
import pytest

from confpickem import season_sim as ss
from confpickem.confidence_pickem_sim import ConfidencePickEmSimulator, Player


def _template(n=8, seed=0, source="t"):
    rng = np.random.default_rng(seed)
    p = np.clip(0.5 + rng.uniform(0.02, 0.4, n), 0, 1)
    return ss.WeekSpec(
        p_a=p,
        crowd_a=np.clip(p + 0.05, 0, 1),
        conf_a=np.linspace(n, 1, n),
        conf_b=np.linspace(1, n, n) / 2,
        source=source,
    )


def _templates(k=4, n=8):
    return [_template(n, seed=i) for i in range(k)]


def _skills(P, spread=0.0):
    base = np.tile([0.5, 0.7, 0.7], (P, 1))
    base[:, 0] += np.linspace(-spread, spread, P)
    return base


def _sim(P=6, banked=None, completed=2, total=6, **kw):
    names = [f"p{i}" for i in range(P)]
    points = np.zeros(P) if banked is None else np.asarray(banked, dtype=float)
    return ss.SeasonSimulator(
        names=names,
        points=points,
        skills=_skills(P),
        templates=_templates(),
        completed_weeks=completed,
        total_weeks=total,
        num_sims=800,
        **kw,
    )


# --- prizes -----------------------------------------------------------------


def test_season_payouts_pay_the_prize_pool_and_split_ties():
    totals = np.array(
        [[100.0, 90.0, 80.0, 70.0, 60.0, 50.0], [100.0, 100.0, 80.0, 70.0, 60.0, 50.0]]
    )
    dollars, place = ss.season_payouts(totals)
    assert dollars[0].tolist() == [2250, 1000, 500, 250, 0, 0]
    # tie for 1st/2nd splits $3250
    assert dollars[1, :2].tolist() == [1625, 1625]
    assert dollars.sum(axis=1).tolist() == [4000, 4000]
    assert place[1, 0, 0] == pytest.approx(0.5) and place[1, 0, 1] == pytest.approx(0.5)


def test_prize_for_one_matches_season_payouts():
    rng = np.random.default_rng(3)
    totals = rng.integers(50, 60, size=(200, 7)).astype(float)  # lots of ties
    dollars, _ = ss.season_payouts(totals)
    for me in (0, 3):
        others = np.delete(totals, me, axis=1)
        assert ss.prize_for_one(totals[:, me], others) == pytest.approx(dollars[:, me])


# --- simulator --------------------------------------------------------------


def test_run_probabilities_and_dollars_are_conserved():
    sim = _sim(banked=[50, 40, 30, 20, 10, 0])
    out = sim.run()
    assert out["p_place_1"].sum() == pytest.approx(1.0)
    assert out["p_top4"].sum() == pytest.approx(4.0)
    weeks_left = sim.total_weeks - sim.completed_weeks
    assert out["exp_weekly_wins"].sum() == pytest.approx(weeks_left)
    assert out["exp_season_dollars"].sum() == pytest.approx(sum(ss.SEASON_PRIZES))
    assert out["exp_dollars"].sum() == pytest.approx(
        sum(ss.SEASON_PRIZES) + weeks_left * ss.WEEKLY_PRIZE
    )


def test_big_lead_is_worth_more_than_a_big_deficit():
    out = _sim(banked=[400, 0, 0, 0, 0, 0], completed=4, total=6).run().set_index("player")
    assert out.loc["p0", "p_place_1"] > 0.9
    assert out.loc["p1", "p_place_1"] < 0.05


def test_run_is_deterministic_for_a_seed():
    a = _sim(banked=[5, 4, 3, 2, 1, 0]).run()
    b = _sim(banked=[5, 4, 3, 2, 1, 0]).run()
    pd.testing.assert_frame_equal(a, b)


def test_inactive_players_score_nothing_more():
    active = np.array([True, True, True, True, True, False])
    out = _sim(banked=[10, 10, 10, 10, 10, 10], active=active).run().set_index("player")
    assert out.loc["p5", "exp_final_points"] == pytest.approx(10.0)
    assert out.loc["p5", "exp_weekly_wins"] == 0


def test_skipping_weeks_lowers_expected_points():
    base = _sim(banked=[0] * 6).run().set_index("player")
    flaky = _sim(banked=[0] * 6, play_prob=np.array([0.5, 1, 1, 1, 1, 1])).run().set_index("player")
    assert flaky.loc["p0", "exp_final_points"] < 0.7 * base.loc["p0", "exp_final_points"]


def test_ability_edge_shifts_points_by_roughly_edge_times_weeks():
    P = 6
    mean = np.zeros(P)
    mean[0] = 10.0
    sim = _sim(banked=[0] * P, ability_mean=mean, ability_post_sd=np.zeros(P))
    flat = _sim(banked=[0] * P).run().set_index("player")
    out = sim.run().set_index("player")
    weeks_left = sim.total_weeks - sim.completed_weeks
    gain = out.loc["p0", "exp_final_points"] - flat.loc["p0", "exp_final_points"]
    assert gain == pytest.approx(10.0 * weeks_left, rel=0.2)


def test_requires_templates_and_weeks_to_simulate():
    with pytest.raises(ValueError):
        ss.SeasonSimulator(
            names=["a"], points=np.zeros(1), skills=_skills(1), templates=[], completed_weeks=1
        )
    weekly = np.ones((3, 18))
    with pytest.raises(ValueError):
        ss.SeasonSimulator.from_weekly(["a", "b", "c"], weekly, {}, _templates())


# --- standings helpers ------------------------------------------------------


def _standings(weekly, names=None):
    P, W = weekly.shape
    df = pd.DataFrame(
        {
            "rank": range(1, P + 1),
            "player_name": names or [f"p{i}" for i in range(P)],
            "total_points": weekly.sum(axis=1),
            "weeks_won": 0,
        }
    )
    for w in range(18):
        df[f"week_{w + 1}_points"] = weekly[:, w] if w < W else 0
    return df


def test_from_standings_banks_only_completed_weeks():
    weekly = np.full((5, 4), 80.0)
    weekly[:, 3] = 15.0  # the in-progress week: partial score
    standings = _standings(weekly)
    sim = ss.SeasonSimulator.from_standings(standings, {}, _templates(), completed_weeks=3)
    assert sim.points.tolist() == [240.0] * 5  # not total_points (255)
    assert sim.completed_weeks == 3


def test_infer_completed_weeks_ignores_partial_current_week():
    rng = np.random.default_rng(0)
    full = rng.integers(60, 100, size=(30, 5)).astype(float)
    partial = np.zeros((30, 1))
    partial[:4, 0] = 12  # four players have a Thursday score so far
    assert ss.infer_completed_weeks(np.hstack([full, partial, np.zeros((30, 12))])) == 5
    assert ss.infer_completed_weeks(np.hstack([full, np.zeros((30, 13))])) == 5
    # a league-wide dip (upset-heavy week) still counts as final
    dip = np.full((30, 1), 65.0)
    assert ss.infer_completed_weeks(np.hstack([full, dip, np.zeros((30, 12))])) == 6


def test_participation_rates_follow_recent_misses():
    weekly = np.full((5, 6), 80.0)
    weekly[1, 5] = 0  # just missed
    weekly[2, 2] = 0  # missed earlier, back since
    weekly[3, :2] = 0  # late joiner, not a miss
    weekly[4, 4:] = 0  # gone for two weeks
    p = ss.participation_rates(weekly)
    assert p[0] == ss.PLAY_RELIABLE and p[3] == ss.PLAY_RELIABLE
    assert p[1] == ss.PLAY_AFTER_MISS
    assert p[2] == ss.PLAY_AFTER_OLD_MISS
    assert not ss.infer_active(weekly)[4]


def test_estimate_ability_shrinks_toward_zero_and_ranks_players():
    weekly = np.full((4, 6), 80.0)
    weekly[0] += 20  # consistently 20 above... relative to the week's mean
    mean, sd = ss.estimate_ability(weekly)
    assert mean[0] > mean[1] > mean[3] - 1e-9 or mean[0] > 0
    assert mean[0] < 20 * 0.75  # shrunk
    assert (sd < ss.ABILITY_SD).all() and (sd > 0).all()
    none_mean, none_sd = ss.estimate_ability(np.zeros((3, 0)))
    assert none_mean.tolist() == [0, 0, 0]


def test_build_skill_array_matches_case_insensitively_and_fills_with_mean():
    known = {
        "Alice": dict(skill_level=0.2, crowd_following=0.4, confidence_following=0.6),
        "bob": dict(skill_level=0.4, crowd_following=0.6, confidence_following=0.8),
    }
    arr, matched = ss.build_skill_array(["ALICE", "Bob", "Zed"], known)
    assert matched == 2
    assert arr[0].tolist() == [0.2, 0.4, 0.6]
    assert arr[2] == pytest.approx([0.3, 0.5, 0.7])  # mean of the known


# --- weekly engine ----------------------------------------------------------


def test_simulate_week_scores_valid_range_and_frozen_picks():
    spec = _template(n=8)
    rng = np.random.default_rng(1)
    outcomes = ss.sample_outcomes(spec.p_a, 300, rng)
    sc = ss.simulate_week_scores(rng, spec, _skills(5), outcomes)
    assert sc.shape == (300, 5) and (sc >= 0).all() and (sc <= 36).all()

    # everyone's game 0 is frozen on side A with 8 points: a side-A win guarantees >= 8
    mask = np.zeros(8, dtype=bool)
    mask[0] = True
    pick = np.zeros((5, 8), dtype=bool)
    pick[:, 0] = True
    pts = np.zeros((5, 8), dtype=int)
    pts[:, 0] = 8
    frozen = ss.FrozenPicks(mask, pick, pts)
    sc2 = ss.simulate_week_scores(rng, spec, _skills(5), outcomes, frozen)
    assert (sc2[outcomes[:, 0]] >= 8).all()
    assert (sc2 <= 36).all()


def test_sample_outcomes_pins_decided_games():
    d = ss.sample_outcomes(
        np.full(3, 0.5), 50, np.random.default_rng(0), actual=[True, None, False]
    )
    assert d[:, 0].all() and not d[:, 2].any() and 0 < d[:, 1].mean() < 1


# --- compare_slates: expected dollars ---------------------------------------


@pytest.fixture
def league():
    names = [f"p{i}" for i in range(8)]
    sim = ConfidencePickEmSimulator(num_sims=100)
    sim.players = [Player(n, 0.5, 0.7, 0.7) for n in names]
    n = 6
    games = pd.DataFrame(
        {
            "home_team": [f"H{i}" for i in range(n)],
            "away_team": [f"A{i}" for i in range(n)],
            "vegas_win_prob": [0.75, 0.7, 0.65, 0.6, 0.58, 0.55],
            "crowd_home_pick_pct": [0.85, 0.8, 0.7, 0.6, 0.55, 0.5],
            "crowd_home_confidence": [10, 9, 8, 7, 6, 5.0],
            "crowd_away_confidence": [3, 3, 3, 3, 3, 3.0],
            "week": 3,
            "kickoff_time": pd.Timestamp("2026-09-20 13:00"),
            "game_index": range(1, n + 1),
        }
    )
    sim.add_games_from_dataframe(games)
    weekly = np.random.default_rng(5).integers(60, 100, size=(8, 2)).astype(float)
    standings = _standings(weekly, names)
    chalk = {f"H{i}": n - i for i in range(n)}
    contra = {f"A{i}": n - i for i in range(n)}
    return sim, standings, chalk, contra


def test_compare_slates_without_standings_is_unchanged(league):
    sim, _, chalk, contra = league
    out = sim.compare_slates("p0", {"chalk": chalk, "contra": contra}, n_outcomes=500)
    assert "expected_dollars" not in out.columns
    assert out.iloc[0]["label"] == "chalk"


def test_compare_slates_adds_dollars(league):
    sim, standings, chalk, contra = league
    out = sim.compare_slates(
        "p0",
        {"chalk": chalk, "contra": contra},
        n_outcomes=800,
        season_standings=standings,
        completed_weeks=2,
        week_templates=_templates(n=6),
    ).set_index("label")
    assert out.loc["chalk", "weekly_dollars"] == pytest.approx(
        500 * out.loc["chalk", "win_probability"]
    )
    for lab in out.index:
        assert out.loc[lab, "expected_dollars"] == pytest.approx(
            out.loc[lab, "weekly_dollars"] + out.loc[lab, "season_dollars"]
        )
        assert 0 <= out.loc[lab, "p_season_top4"] <= 1
    assert out.loc["chalk", "weekly_dollars"] > out.loc["contra", "weekly_dollars"]
    # the reference slate (best win %) has zero paired error against itself
    assert out.loc["chalk", "season_dollars_se"] == 0


def test_compare_slates_objective_expected_dollars_reorders(league):
    sim, standings, chalk, contra = league
    out = sim.compare_slates(
        "p0",
        {"chalk": chalk, "contra": contra},
        n_outcomes=500,
        season_standings=standings,
        completed_weeks=2,
        week_templates=_templates(n=6),
        objective="expected_dollars",
    )
    assert out["expected_dollars"].is_monotonic_decreasing
    assert out["rank"].tolist() == [1, 2]


def test_compare_slates_objective_validation(league):
    sim, standings, chalk, _ = league
    with pytest.raises(ValueError):
        sim.compare_slates("p0", {"chalk": chalk}, objective="expected_dollars")
    with pytest.raises(ValueError):
        sim.compare_slates("p0", {"chalk": chalk}, objective="bogus")
    with pytest.raises(ValueError):
        sim.compare_slates(
            "nobody-here",
            {"chalk": chalk},
            season_standings=standings,
            completed_weeks=2,
            week_templates=_templates(n=6),
        )
