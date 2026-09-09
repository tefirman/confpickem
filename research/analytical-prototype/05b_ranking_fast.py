#!/usr/bin/env python
"""Ranking check, instrumented: one week, per-slate timing, vectorized analytical.

Same question as 05: does sampled analytical P(win) rank slates like the sim?
This version:
  - vectorizes pwin_sampled's inner PMF over the K outcomes (no per-k Python loop)
  - one week, prints per slate so we see progress and per-call cost
"""
import sys, warnings, io, contextlib, json, time
sys.path.insert(0, "src")
sys.path.insert(0, "research/analytical-prototype")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

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
YEAR, WEEK = 2025, 9
N_SLATES = 30
SIM_N = 600
K_OUTCOMES = 5000
SEED = 999


def skills(year):
    return json.load(open("player_skills_{}.json".format(year)))["player_skills"]


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


def vectorized_weighted_pmf(pc, points):
    """PMF of sum_i points[i]*Bernoulli(pc[:,i]) for every row of pc at once.

    pc:     [K, n] success probs (already conditioned on the outcome draw)
    points: [n]    integer weights
    returns [K, T+1] where T = points.sum()
    """
    K, n = pc.shape
    T = int(points.sum())
    pmf = np.zeros((K, T + 1))
    pmf[:, 0] = 1.0
    size = 1
    for i in range(n):
        w = int(points[i])
        if w == 0:
            continue
        p = pc[:, i][:, None]                      # [K, 1]
        new = np.zeros((K, size + w))
        new[:, :size] += pmf[:, :size] * (1.0 - p)
        new[:, w:w + size] += pmf[:, :size] * p
        pmf[:, :size + w] = new
        size += w
    return pmf[:, :size]


def pwin_sampled(my_ph, my_pts, opp_types, vh, O):
    K, n = O.shape
    my_scores = (my_pts[None, :] * (my_ph[None, :] == O)).sum(axis=1)  # [K]
    log_beat = np.zeros(K)
    for (ph_prob, ph, pts, count) in opp_types:
        pc = np.where(O, ph_prob[None, :], 1.0 - ph_prob[None, :])     # [K, n]
        pmf = vectorized_weighted_pmf(pc, pts)                        # [K, T+1]
        cdf = np.cumsum(pmf, axis=1)
        idx = np.clip(my_scores - 1, 0, pmf.shape[1] - 1)
        p_le = np.where(my_scores >= 1, cdf[np.arange(K), idx], 0.0)
        eq_idx = np.clip(my_scores, 0, pmf.shape[1] - 1)
        p_eq = np.where(my_scores < pmf.shape[1], pmf[np.arange(K), eq_idx], 0.0)
        log_beat += count * np.log(np.maximum(p_le + 0.5 * p_eq, 1e-300))
    return float(np.exp(log_beat).mean())


def slate_arrays(slate, home, away, n):
    ph = np.zeros(n, dtype=bool)
    pts = np.zeros(n, dtype=int)
    for gi in range(n):
        if home[gi] in slate:
            ph[gi], pts[gi] = True, slate[home[gi]]
        else:
            ph[gi], pts[gi] = False, slate[away[gi]]
    return ph, pts


def sim_pwin(y, gdf, slate, sk):
    names = y.players["player_name"].tolist()
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
    order = np.argsort(-np.abs(vh - 0.5))
    chalk = {}
    for rank, gi in enumerate(order):
        chalk[home[gi] if vh[gi] >= 0.5 else away[gi]] = n - rank
    slates = [dict(chalk)]
    for _ in range(N_SLATES - 6):
        s = dict(chalk)
        for _ in range(rng.integers(1, 4)):
            t = list(s.keys())
            a, b = rng.choice(len(t), 2, replace=False)
            s[t[a]], s[t[b]] = s[t[b]], s[t[a]]
        slates.append(s)
    for k in (1, 2, 3, 4, 5):
        s = dict(chalk)
        for gi in order[::-1][:k]:
            fav = home[gi] if vh[gi] >= 0.5 else away[gi]
            dog = away[gi] if vh[gi] >= 0.5 else home[gi]
            s[dog] = s.pop(fav)
        slates.append(s)
    return slates


def main():
    y = YahooPickEm(week=WEEK, league_id=LEAGUE_ID, cookies_file=COOKIES,
                    cache_dir=CACHE[YEAR])
    gdf = convert_yahoo_to_simulator_format(y, ignore_results=False)
    n = len(gdf)
    home = gdf["home_team"].tolist()
    away = gdf["away_team"].tolist()
    vh = gdf["vegas_win_prob"].to_numpy()
    chp = gdf["crowd_home_pick_pct"].to_numpy()
    chc = gdf["crowd_home_confidence"].to_numpy()
    cac = gdf["crowd_away_confidence"].to_numpy()
    sk = skills(YEAR)

    opp_list = []
    for _, prow in y.players.iterrows():
        nm = prow["player_name"]
        if nm == ME or nm not in sk:
            continue
        opp_list.append(modal_entry(vh, chp, chc, cac,
                                    sk[nm]["crowd_following"],
                                    sk[nm]["confidence_following"], n))
    opp_types = dedupe(opp_list)
    print("{} wk{}: n={}, {} opp -> {} types".format(
        YEAR, WEEK, n, len(opp_list), len(opp_types)), flush=True)

    rng = np.random.default_rng(SEED + WEEK)
    slates = make_slates(home, away, vh, n, rng)
    O = rng.random((K_OUTCOMES, n)) < vh[None, :]

    ana, sim = [], []
    for i, s in enumerate(slates):
        ph, pts = slate_arrays(s, home, away, n)
        t0 = time.time()
        a = pwin_sampled(ph, pts, opp_types, vh, O)
        t1 = time.time()
        sp = sim_pwin(y, gdf, s, sk)
        t2 = time.time()
        ana.append(a)
        sim.append(sp)
        print("  slate {:>2}: analytic {:.4f} ({:.2f}s)   sim {:.4f} ({:.1f}s)".format(
            i, a, t1 - t0, sp, t2 - t1), flush=True)

    ana, sim = np.array(ana), np.array(sim)
    rho, _ = spearmanr(ana, sim)
    ab = int(np.argmax(ana))
    print("\nSpearman rho(analytic, sim) = {:.3f}".format(rho))
    print("analytic argmax -> sim rank {}/{}  (sim {:.4f} vs sim best {:.4f})".format(
        int((sim > sim[ab]).sum()) + 1, len(slates), sim[ab], sim.max()))
    print("top-3 analytic:", np.argsort(-ana)[:3].tolist(),
          " top-3 sim:", np.argsort(-sim)[:3].tolist())


if __name__ == "__main__":
    main()
