#!/usr/bin/env python
"""P(win) by importance-sampling the game-outcome vector; P(win | o) exact.

    P(win) = E_{o ~ Bernoulli(vegas)} [ P(my score strictly highest | o) ]

We draw K outcome vectors o (16-dim, cheap), and for each compute P(win | o)
*exactly*:
  - my score under o is deterministic (my slate is fixed).
  - opponent j's score under o is a weighted Bernoulli: each of their (modal)
    picks is "correct" w.p. p_home_j[i] if o_i = 1 else 1 - p_home_j[i],
    weighted by their confidence points. -> weighted_pmf -> P(my_score > opp_j | o).
  - P(win | o) ~= prod_j P(my_score > opp_j | o)   (independence ACROSS opponents
    given o -- their remaining randomness is individual pick noise, ~independent).

The only Monte-Carlo noise is in the 16-dim outcome draw, not the 55x16 pick
matrix the old simulator sampled -> far lower variance for the same K.

Benchmark vs the simulator's win_pct on the 4 check weeks.

Run:  python research/analytical-prototype/04_pwin_sampled.py
"""
import sys, warnings, io, contextlib, json, time
sys.path.insert(0, "src")
sys.path.insert(0, "research/analytical-prototype")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from poisson_binomial import weighted_pmf

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
CHECK_WEEKS = [(2024, 12), (2025, 9), (2025, 12), (2024, 3)]
SIM_N = 8000
K_OUTCOMES = 4000
SEED = 12345

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
    return p_home, pick_home, points


def dedupe_opponents(opp_list):
    """Collapse identical (pick_home, points) opponents. Returns list of
    (p_home[n], pick_home[n], points[n], count)."""
    buckets = {}
    for (p_home, pick_home, points) in opp_list:
        key = (tuple(pick_home.tolist()), tuple(points.tolist()))
        if key in buckets:
            buckets[key][3] += 1
        else:
            buckets[key] = [p_home, pick_home, points, 1]
    return list(buckets.values())


def pwin_sampled(my_ph, my_pts, opp_types, vegas_home, K=K_OUTCOMES, seed=SEED):
    """Importance-sampled P(win). opp_types: list of (p_home, pick_home, points, count)."""
    rng = np.random.default_rng(seed)
    n = len(vegas_home)
    O = rng.random((K, n)) < vegas_home[None, :]     # [K, n] bool, home-win draws

    my_scores = (my_pts[None, :] * (my_ph[None, :] == O)).sum(axis=1)  # [K]

    # accumulate log P(win | o_k)
    log_beat = np.zeros(K)
    for (p_home, pick_home, points, count) in opp_types:
        # For each drawn outcome, P(this opp's pick i is correct)
        # = p_home[i] where home won, else 1 - p_home[i]
        pc = np.where(O, p_home[None, :], 1.0 - p_home[None, :])   # [K, n]
        # P(my_score_k > opp score) for each k -- opp score PMF depends on o_k
        # weighted_pmf is per-row; loop rows but that's K, not K*P.
        for k in range(K):
            pmf = weighted_pmf(pc[k], points)
            s = int(my_scores[k])
            cdf = np.cumsum(pmf)
            p_le = cdf[s - 1] if s >= 1 and s - 1 < len(cdf) else (0.0 if s < 1 else cdf[-1])
            p_eq = pmf[s] if s < len(pmf) else 0.0
            log_beat[k] += count * np.log(max(p_le + 0.5 * p_eq, 1e-300))

    pwin_per_k = np.exp(log_beat)
    return float(pwin_per_k.mean()), float(pwin_per_k.std(ddof=1) / np.sqrt(K))


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
    print("{:<12} {:>4} {:>10} {:>12} {:>10} {:>8}  {}".format(
        "week", "n", "sim", "sampled", "stderr", "ratio", "time"))
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
        slate, my_ph, my_pts = {}, np.zeros(n, dtype=bool), np.zeros(n, dtype=int)
        for rank, gi in enumerate(order):
            pick_home = vegas_home[gi] >= 0.5
            team = home[gi] if pick_home else away[gi]
            slate[team] = n - rank
            my_ph[gi] = pick_home
            my_pts[gi] = n - rank

        opp_list = []
        for _, prow in y.players.iterrows():
            nm = prow["player_name"]
            if nm == ME or nm not in sk:
                continue
            opp_list.append(modal_entry(
                vegas_home, crowd_home_pct, chc, cac,
                sk[nm]["crowd_following"], sk[nm]["confidence_following"], n))
        opp_types = dedupe_opponents(opp_list)

        t = time.time()
        pw, se = pwin_sampled(my_ph, my_pts, opp_types, vegas_home)
        dt = time.time() - t
        ps = sim_pwin(y, gdf, slate, year)
        print("{:<12} {:>4} {:>10.4f} {:>12.4f} {:>10.4f} {:>8.2f}  {:.1f}s "
              "({} opp -> {} types)".format(
                  "{} wk{}".format(year, week), n, ps, pw, se,
                  pw / ps if ps else float("nan"), dt,
                  len(opp_list), len(opp_types)), flush=True)


if __name__ == "__main__":
    main()
