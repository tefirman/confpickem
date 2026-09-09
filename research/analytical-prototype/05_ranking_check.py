#!/usr/bin/env python
"""Does the sampled analytical P(win) RANK slates like the simulator does?

For an optimizer, absolute calibration matters less than monotonicity: if
analytical P(win) says slate A > slate B, does the simulator agree?

Test: generate a spread of candidate slates for a week -- chalk, several
random perturbations, a few deliberately-contrarian ones -- score each with
both the sampled analytical P(win) and a high-sim simulator run. Report
Spearman rank correlation and whether the analytical argmax is a simulator
top-few slate.

Run:  python research/analytical-prototype/05_ranking_check.py
"""
import sys, warnings, io, contextlib, json, time
sys.path.insert(0, "src")
sys.path.insert(0, "research/analytical-prototype")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

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
CHECK_WEEKS = [(2024, 3), (2025, 9), (2025, 12)]
N_SLATES = 24
SIM_N = 12000
K_OUTCOMES = 6000
SEED = 999

_SKILLS = {}


def skills(year):
    if year not in _SKILLS:
        _SKILLS[year] = json.load(open("player_skills_{}.json".format(year)))["player_skills"]
    return _SKILLS[year]


def modal_entry(vh, chp, chc, cac, cf, conf_foll, n):
    p_home = np.clip(vh * (1 - cf) + chp * cf, 0.0, 1.0)
    pick_home = p_home > 0.5
    chosen = np.where(pick_home, chc, cac)
    opposing = np.where(pick_home, cac, chc)
    conf_diff = (chosen - opposing) / (chosen + opposing)
    vegas_conf = np.abs(vh - 0.5) * 2 * n
    score = chosen * (1 + conf_diff) * conf_foll + vegas_conf * (1 - conf_foll)
    points = pd.Series(score).rank(method="first").to_numpy().astype(int)
    return p_home, pick_home, points


def dedupe(opp_list):
    b = {}
    for (ph_prob, ph, pts) in opp_list:
        key = (tuple(ph.tolist()), tuple(pts.tolist()))
        if key in b:
            b[key][3] += 1
        else:
            b[key] = [ph_prob, ph, pts, 1]
    return list(b.values())


def slate_arrays(slate, home, away, n):
    ph = np.zeros(n, dtype=bool)
    pts = np.zeros(n, dtype=int)
    for gi in range(n):
        if home[gi] in slate:
            ph[gi] = True
            pts[gi] = slate[home[gi]]
        else:
            ph[gi] = False
            pts[gi] = slate[away[gi]]
    return ph, pts


def pwin_sampled(my_ph, my_pts, opp_types, vh, O):
    K, n = O.shape
    my_scores = (my_pts[None, :] * (my_ph[None, :] == O)).sum(axis=1)
    log_beat = np.zeros(K)
    for (ph_prob, ph, pts, count) in opp_types:
        pc = np.where(O, ph_prob[None, :], 1.0 - ph_prob[None, :])
        for k in range(K):
            pmf = weighted_pmf(pc[k], pts)
            s = int(my_scores[k])
            cdf = np.cumsum(pmf)
            p_le = cdf[s - 1] if 1 <= s <= len(cdf) else (0.0 if s < 1 else cdf[-1])
            p_eq = pmf[s] if s < len(pmf) else 0.0
            log_beat[k] += count * np.log(max(p_le + 0.5 * p_eq, 1e-300))
    return float(np.exp(log_beat).mean())


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


def make_slates(home, away, vh, n, rng):
    """chalk + random perturbations + a few contrarian variants."""
    order = np.argsort(-np.abs(vh - 0.5))
    chalk = {}
    for rank, gi in enumerate(order):
        team = home[gi] if vh[gi] >= 0.5 else away[gi]
        chalk[team] = n - rank
    slates = [dict(chalk)]

    for _ in range(N_SLATES - 6):
        s = dict(chalk)
        # swap confidence between two random games
        for _ in range(rng.integers(1, 4)):
            teams = list(s.keys())
            a, b = rng.choice(len(teams), 2, replace=False)
            s[teams[a]], s[teams[b]] = s[teams[b]], s[teams[a]]
        slates.append(s)

    # contrarian: flip the pick on the k closest-to-tossup games, keep points
    for k in (1, 2, 3, 4, 5):
        s = dict(chalk)
        tossups = order[::-1][:k]  # least certain
        for gi in tossups:
            fav = home[gi] if vh[gi] >= 0.5 else away[gi]
            dog = away[gi] if vh[gi] >= 0.5 else home[gi]
            s[dog] = s.pop(fav)
        slates.append(s)
    return slates


def main():
    for (year, week) in CHECK_WEEKS:
        y = YahooPickEm(week=week, league_id=LEAGUE_ID, cookies_file=COOKIES,
                        cache_dir=CACHE[year])
        gdf = convert_yahoo_to_simulator_format(y, ignore_results=False)
        n = len(gdf)
        home = gdf["home_team"].tolist()
        away = gdf["away_team"].tolist()
        vh = gdf["vegas_win_prob"].to_numpy()
        chp = gdf["crowd_home_pick_pct"].to_numpy()
        chc = gdf["crowd_home_confidence"].to_numpy()
        cac = gdf["crowd_away_confidence"].to_numpy()
        sk = skills(year)

        opp_list = []
        for _, prow in y.players.iterrows():
            nm = prow["player_name"]
            if nm == ME or nm not in sk:
                continue
            opp_list.append(modal_entry(vh, chp, chc, cac,
                                        sk[nm]["crowd_following"],
                                        sk[nm]["confidence_following"], n))
        opp_types = dedupe(opp_list)

        rng = np.random.default_rng(SEED + week)
        slates = make_slates(home, away, vh, n, rng)
        O = rng.random((K_OUTCOMES, n)) < vh[None, :]

        ana, sim = [], []
        for s in slates:
            ph, pts = slate_arrays(s, home, away, n)
            ana.append(pwin_sampled(ph, pts, opp_types, vh, O))
            sim.append(sim_pwin(y, gdf, s, year))
        ana, sim = np.array(ana), np.array(sim)

        rho, _ = spearmanr(ana, sim)
        ana_best = int(np.argmax(ana))
        sim_rank_of_ana_best = int((sim > sim[ana_best]).sum()) + 1
        print("{} wk{}  ({} slates, {} opp->{} types)".format(
            year, week, len(slates), len(opp_list), len(opp_types)))
        print("  Spearman rho(analytical, sim) = {:.3f}".format(rho))
        print("  analytical argmax slate -> simulator rank {}/{}  "
              "(sim P(win) {:.4f} vs sim best {:.4f})".format(
                  sim_rank_of_ana_best, len(slates), sim[ana_best], sim.max()))
        # show the top-3 by each
        print("  top-3 by analytical:", np.argsort(-ana)[:3].tolist(),
              " top-3 by sim:", np.argsort(-sim)[:3].tolist())
        print(flush=True)


if __name__ == "__main__":
    main()
