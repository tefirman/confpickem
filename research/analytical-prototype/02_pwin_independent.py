#!/usr/bin/env python
"""Analytical P(win), independence approximation -- vs the old simulator.

Pieces:
  1. My score PMF: weighted_pmf(vegas_probs_for_my_picks, my_confidence).
  2. Each opponent's score PMF: from the validated field model
     (P(pick home) blend + confidence-blend-then-rank), we get their
     per-game (p_correct_i, points_i); PMF = weighted_pmf(p_correct, points).
  3. P(I finish 1st) ~= prod_j  prob_a_beats_b(my_pmf, opp_j_pmf).
     KNOWN WRONG: treats games as independent across entries; they share the
     same 16 outcomes. This is the baseline before the correlation fix.

Benchmark: run the simulator on the same week/field/my-slate and compare its
win_pct to this analytical number. If they're in the same ballpark (the
independence approx should *overstate* P(win) somewhat), the machinery is right
and correlation is the remaining gap.

Run:  python research/analytical-prototype/02_pwin_independent.py
"""
import sys, warnings, io, contextlib, json, time
sys.path.insert(0, "src")
sys.path.insert(0, "research/analytical-prototype")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from poisson_binomial import weighted_pmf, prob_a_beats_b

import confpickem.yahoo_pickem_scraper as _scraper
_orig = _scraper.PageCache.get_cached_content
_scraper.PageCache.get_cached_content = (
    lambda self, pt, wk, expiration=3600: _orig(self, pt, wk, expiration=10 ** 13))

from confpickem.yahoo_pickem_scraper import YahooPickEm
from confpickem.confidence_pickem_sim import ConfidencePickEmSimulator, Player
from confpickem.yahoo_pickem_integration import (
    convert_yahoo_to_simulator_format, convert_yahoo_picks_to_dataframe)

ME = "Firman's Educated Guesses"
LEAGUE_ID = 15435
COOKIES = "cookies.txt"
CACHE = {2024: "PickEmCache2024", 2025: "PickEmCache2025"}
CHECK_WEEKS = [(2024, 3), (2024, 12), (2025, 9), (2025, 12)]
SIM_N = 4000

_SKILLS = {}


def skills(year):
    if year not in _SKILLS:
        _SKILLS[year] = json.load(open("player_skills_{}.json".format(year)))["player_skills"]
    return _SKILLS[year]


def opponent_pmf(vegas_home, crowd_home_pct, chc, cac, cf, conf_foll, n):
    """(p_correct per game, integer points per game, score PMF) for one opponent."""
    p_home = np.clip(vegas_home * (1 - cf) + crowd_home_pct * cf, 0.0, 1.0)
    pick_home = p_home > 0.5                      # modal pick
    # P(their pick is correct) = P(pick home)*P(home wins) + P(pick away)*P(away wins),
    # but with a modal pick we use: p_correct = p_home if they pick home else 1 - p_home,
    # weighted toward vegas since that's what "correct" means.
    p_correct = np.where(pick_home, vegas_home, 1.0 - vegas_home)

    chosen = np.where(pick_home, chc, cac)
    opposing = np.where(pick_home, cac, chc)
    conf_diff = (chosen - opposing) / (chosen + opposing)
    vegas_conf = np.abs(vegas_home - 0.5) * 2 * n
    score = chosen * (1 + conf_diff) * conf_foll + vegas_conf * (1 - conf_foll)
    points = pd.Series(score).rank(method="first").to_numpy().astype(int)  # 1..n

    pmf = weighted_pmf(p_correct, points)
    return p_correct, points, pmf


def my_pmf_from_slate(slate, home, away, vegas_home):
    """slate: {team: points}. Returns (p_correct, points, score PMF)."""
    n = len(home)
    p_correct = np.zeros(n)
    points = np.zeros(n, dtype=int)
    for gi in range(n):
        if home[gi] in slate:
            points[gi] = slate[home[gi]]
            p_correct[gi] = vegas_home[gi]
        elif away[gi] in slate:
            points[gi] = slate[away[gi]]
            p_correct[gi] = 1.0 - vegas_home[gi]
        else:
            raise KeyError("slate missing game {}".format(gi))
    return p_correct, points, weighted_pmf(p_correct, points)


def analytical_pwin(my_pmf, opp_pmfs):
    p = 1.0
    for opp in opp_pmfs:
        p *= prob_a_beats_b(my_pmf, opp)
    return p


def sim_pwin(y, gdf, slate, year):
    """Simulator win_pct for ME playing `slate` against the modeled field."""
    names = y.players["player_name"].tolist()
    sk = skills(year)
    sim = ConfidencePickEmSimulator(num_sims=SIM_N)
    sim.add_games_from_dataframe(gdf.drop(columns=["actual_outcome"]))
    sim.players = [Player(nm, sk[nm]["skill_level"], sk[nm]["crowd_following"],
                          sk[nm]["confidence_following"]) if nm in sk
                   else Player(nm, 0.6, 0.5, 0.5) for nm in names]
    with contextlib.redirect_stdout(io.StringIO()):
        picks = sim.simulate_picks(fixed_picks={ME: slate})
        outcomes = sim.simulate_outcomes()
        stats = sim.analyze_results(picks, outcomes)
    return float(stats["win_pct"][ME])


def main():
    print("{:<12} {:>10} {:>12} {:>10}  {}".format(
        "week", "sim P(win)", "analytic", "ratio", "slate (top 5)"))
    for (year, week) in CHECK_WEEKS:
        y = YahooPickEm(week=week, league_id=LEAGUE_ID, cookies_file=COOKIES,
                        cache_dir=CACHE[year])
        gdf = convert_yahoo_to_simulator_format(y, ignore_results=False)
        n = len(gdf)
        home = gdf["home_team"].tolist()
        away = gdf["away_team"].tolist()
        vegas_home = gdf["vegas_win_prob"].to_numpy()
        crowd_home_pct = gdf["crowd_home_pick_pct"].to_numpy()
        chc = gdf["crowd_home_confidence"].to_numpy()
        cac = gdf["crowd_away_confidence"].to_numpy()
        sk = skills(year)

        # ME's slate = straight chalk (vegas favorite ranked by |p - .5|), a fixed reference
        order = np.argsort(-np.abs(vegas_home - 0.5))
        slate = {}
        for rank, gi in enumerate(order):
            team = home[gi] if vegas_home[gi] >= 0.5 else away[gi]
            slate[team] = n - rank

        _, _, my_pmf = my_pmf_from_slate(slate, home, away, vegas_home)

        opp_pmfs = []
        for _, prow in y.players.iterrows():
            nm = prow["player_name"]
            if nm == ME or nm not in sk:
                continue
            _, _, pmf = opponent_pmf(
                vegas_home, crowd_home_pct, chc, cac,
                sk[nm]["crowd_following"], sk[nm]["confidence_following"], n)
            opp_pmfs.append(pmf)

        a = analytical_pwin(my_pmf, opp_pmfs)
        s = sim_pwin(y, gdf, slate, year)
        top5 = sorted(slate.items(), key=lambda kv: -kv[1])[:5]
        print("{:<12} {:>10.4f} {:>12.4f} {:>10.2f}  {}".format(
            "{} wk{}".format(year, week), s, a, (a / s if s else float("nan")), top5))


if __name__ == "__main__":
    main()
