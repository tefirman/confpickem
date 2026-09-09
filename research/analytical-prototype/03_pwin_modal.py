#!/usr/bin/env python
"""Analytical P(win) with cross-entry correlation -- MODAL opponents, exact.

P(win) = sum over all 2^n game-outcome vectors o of
             P(o) * P(my score strictly highest | o)

P(o) = prod_i vegas_i^{o_i} (1 - vegas_i)^{1 - o_i}   (o_i = 1 iff home team wins)

Every opponent plays their single most-likely slate (modal pick + confidence
rank), so given o every score is deterministic and P(win|o) is an indicator
(ties split evenly). Fully vectorized over the 2^n outcomes.

Benchmark vs the simulator's win_pct on the 4 check weeks.

Run:  python research/analytical-prototype/03_pwin_modal.py
"""
import sys, warnings, io, contextlib, json, time
sys.path.insert(0, "src")
sys.path.insert(0, "research/analytical-prototype")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import confpickem.yahoo_pickem_scraper as _scraper
_orig = _scraper.PageCache.get_cached_content
_scraper.PageCache.get_cached_content = (
    lambda self, pt, wk, expiration=3600: _orig(self, pt, wk, expiration=10 ** 13))

from confpickem.yahoo_pickem_scraper import YahooPickEm
from confpickem.confidence_pickem_sim import ConfidencePickEmSimulator, Player
from confpickem.yahoo_pickem_integration import convert_yahoo_to_simulator_format

ME = "Firman's Educated Guesses"
LEAGUE_ID = 15435
COOKIES = "cookies.txt"
CACHE = {2024: "PickEmCache2024", 2025: "PickEmCache2025"}
CHECK_WEEKS = [(2024, 12), (2025, 9), (2025, 12), (2024, 3)]  # 13g first (fastest)
SIM_N = 6000

_SKILLS = {}


def skills(year):
    if year not in _SKILLS:
        _SKILLS[year] = json.load(open("player_skills_{}.json".format(year)))["player_skills"]
    return _SKILLS[year]


def modal_entry(vegas_home, crowd_home_pct, chc, cac, cf, conf_foll, n):
    p_home = np.clip(vegas_home * (1 - cf) + crowd_home_pct * cf, 0.0, 1.0)
    pick_home = p_home > 0.5
    chosen = np.where(pick_home, chc, cac)
    opposing = np.where(pick_home, cac, chc)
    conf_diff = (chosen - opposing) / (chosen + opposing)
    vegas_conf = np.abs(vegas_home - 0.5) * 2 * n
    score = chosen * (1 + conf_diff) * conf_foll + vegas_conf * (1 - conf_foll)
    points = pd.Series(score).rank(method="first").to_numpy().astype(int)
    return pick_home.astype(np.int8), points.astype(np.int32)


def pwin_modal(my_ph, my_pts, opp_ph, opp_pts, vegas_home):
    """my_ph/my_pts: [n]. opp_ph/opp_pts: [P, n]. Returns exact P(win)."""
    n = len(vegas_home)
    M = 1 << n
    idx = np.arange(M, dtype=np.int64)
    bits = ((idx[:, None] >> np.arange(n)[None, :]) & 1).astype(np.int8)  # [M, n]

    logp = np.where(bits == 1, np.log(vegas_home)[None, :],
                    np.log1p(-vegas_home)[None, :])
    outcome_p = np.exp(logp.sum(axis=1))                                 # [M]

    my_scores = (my_pts[None, :] * (my_ph[None, :] == bits)).sum(axis=1)  # [M]

    best = np.zeros(M, dtype=np.int64)
    ties = np.zeros(M, dtype=np.int64)
    for p in range(opp_ph.shape[0]):
        s = (opp_pts[p][None, :] * (opp_ph[p][None, :] == bits)).sum(axis=1)  # [M]
        newmax = s > best
        best = np.where(newmax, s, best)
        # track how many opps sit exactly at the running max (for tie credit)
        ties = np.where(newmax, 1, ties + (s == best).astype(np.int64))

    strict = my_scores > best
    tie = my_scores == best
    pwin = float(outcome_p[strict].sum())
    if tie.any():
        # tie credit: 1 / (1 + number of opponents tied at that score)
        pwin += float((outcome_p[tie] / (1 + ties[tie])).sum())
    return pwin


def sim_pwin(y, gdf, slate, year):
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
    print("{:<12} {:>4} {:>10} {:>12} {:>8}   {}".format(
        "week", "n", "sim", "modal", "ratio", "modal time"))
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

        order = np.argsort(-np.abs(vegas_home - 0.5))
        slate = {}
        my_ph = np.zeros(n, dtype=np.int8)
        my_pts = np.zeros(n, dtype=np.int32)
        for rank, gi in enumerate(order):
            pick_home = vegas_home[gi] >= 0.5
            team = home[gi] if pick_home else away[gi]
            slate[team] = n - rank
            my_ph[gi] = 1 if pick_home else 0
            my_pts[gi] = n - rank

        opp_ph, opp_pts = [], []
        for _, prow in y.players.iterrows():
            nm = prow["player_name"]
            if nm == ME or nm not in sk:
                continue
            ph, pts = modal_entry(vegas_home, crowd_home_pct, chc, cac,
                                  sk[nm]["crowd_following"],
                                  sk[nm]["confidence_following"], n)
            opp_ph.append(ph)
            opp_pts.append(pts)
        opp_ph = np.array(opp_ph)
        opp_pts = np.array(opp_pts)

        t = time.time()
        pm = pwin_modal(my_ph, my_pts, opp_ph, opp_pts, vegas_home)
        tm = time.time() - t
        ps = sim_pwin(y, gdf, slate, year)
        print("{:<12} {:>4} {:>10.4f} {:>12.4f} {:>8.2f}   {:.1f}s".format(
            "{} wk{}".format(year, week), n, ps, pm,
            pm / ps if ps else float("nan"), tm), flush=True)


if __name__ == "__main__":
    main()
