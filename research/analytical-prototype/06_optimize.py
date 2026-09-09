#!/usr/bin/env python
"""Slate optimizer on the sampled analytical P(win), backtested vs greedy/leverage/actual.

Objective: pwin_sampled(my slate, opponent types, outcome draws)  -- ~0.06s/eval,
no Monte-Carlo noise in the ranking (rho ~0.85 vs the simulator).

Optimizer: random-restart hill climb.
  - seed: chalk (favorite ranked by |vegas - .5|)
  - neighbor moves: (a) flip which team we pick in one game, keeping its points;
                    (b) swap the confidence points of two games.
  - accept strictly improving neighbors; restart from perturbed chalk.
Because an eval is 0.06s we can afford thousands of iterations.

For each backtest week we report, scored on the ACTUAL outcomes:
  rank, points-behind-1st, one-game-away (any single game flip -> we finish 1st),
  n live flips -- for greedy, leverage(lambda=1), and this analytical optimizer.

Run:  python research/analytical-prototype/06_optimize.py
Writes research/analytical-prototype/06_results.csv
"""
import sys, os, warnings, io, contextlib, json, time
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
from confpickem.yahoo_pickem_integration import (
    convert_yahoo_to_simulator_format, convert_yahoo_picks_to_dataframe)

ME = "Firman's Educated Guesses"
LEAGUE_ID = 15435
COOKIES = "cookies.txt"
CACHE = {2024: "PickEmCache2024", 2025: "PickEmCache2025"}
WEEKS = [(2024, 3), (2024, 6), (2024, 9), (2024, 12), (2024, 15),
         (2025, 2), (2025, 5), (2025, 9), (2025, 12), (2025, 15)]

K_OUTCOMES = 6000
HC_ITERS = 400
HC_RESTARTS = 4
SEED = 20240
CHECKPOINT = "research/analytical-prototype/06_results.csv"

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


def vectorized_weighted_pmf(pc, points):
    K, n = pc.shape
    T = int(points.sum())
    pmf = np.zeros((K, T + 1))
    pmf[:, 0] = 1.0
    size = 1
    for i in range(n):
        w = int(points[i])
        if w == 0:
            continue
        p = pc[:, i][:, None]
        new = np.zeros((K, size + w))
        new[:, :size] += pmf[:, :size] * (1.0 - p)
        new[:, w:w + size] += pmf[:, :size] * p
        pmf[:, :size + w] = new
        size += w
    return pmf[:, :size]


def make_pwin(opp_types, vh, O):
    """Closure: given (my_pick_home[n] bool, my_points[n] int) -> analytical P(win)."""
    K, n = O.shape
    # precompute per-type conditional correctness prob array [K, n]
    type_pc = []
    for (ph_prob, ph, pts, count) in opp_types:
        pc = np.where(O, ph_prob[None, :], 1.0 - ph_prob[None, :])
        type_pc.append((pc, pts, count))

    def pwin(my_ph, my_pts):
        my_scores = (my_pts[None, :] * (my_ph[None, :] == O)).sum(axis=1)
        log_beat = np.zeros(K)
        for (pc, pts, count) in type_pc:
            pmf = vectorized_weighted_pmf(pc, pts)
            cdf = np.cumsum(pmf, axis=1)
            idx = np.clip(my_scores - 1, 0, pmf.shape[1] - 1)
            p_le = np.where(my_scores >= 1, cdf[np.arange(K), idx], 0.0)
            eqi = np.clip(my_scores, 0, pmf.shape[1] - 1)
            p_eq = np.where(my_scores < pmf.shape[1], pmf[np.arange(K), eqi], 0.0)
            log_beat += count * np.log(np.maximum(p_le + 0.5 * p_eq, 1e-300))
        return float(np.exp(log_beat).mean())

    return pwin


def chalk_slate(home, away, vh, n):
    order = np.argsort(-np.abs(vh - 0.5))
    ph = np.zeros(n, dtype=bool)
    pts = np.zeros(n, dtype=int)
    for rank, gi in enumerate(order):
        ph[gi] = vh[gi] >= 0.5
        pts[gi] = n - rank
    return ph, pts


def _neighbor(ph, pts, n, rng):
    """Return a fresh (ph, pts) one move away: flip a pick or swap two confidences."""
    ph2, pts2 = ph.copy(), pts.copy()
    if rng.random() < 0.5:
        g = int(rng.integers(n))
        ph2[g] = not ph2[g]
    else:
        a, b = rng.choice(n, 2, replace=False)
        pts2[a], pts2[b] = pts2[b], pts2[a]
    return ph2, pts2


def hill_climb(pwin, home, away, vh, n, rng):
    best_ph, best_pts = chalk_slate(home, away, vh, n)
    best_val = pwin(best_ph, best_pts)

    for restart in range(HC_RESTARTS):
        if restart == 0:
            ph, pts = best_ph.copy(), best_pts.copy()
        else:
            ph, pts = chalk_slate(home, away, vh, n)
            for _ in range(int(rng.integers(2, 6))):
                a, b = rng.choice(n, 2, replace=False)
                pts[a], pts[b] = pts[b], pts[a]
            for _ in range(int(rng.integers(0, 3))):
                ph[int(rng.integers(n))] ^= True
        val = pwin(ph, pts)

        no_improve = 0
        for _ in range(HC_ITERS):
            cand_ph, cand_pts = _neighbor(ph, pts, n, rng)
            cand_val = pwin(cand_ph, cand_pts)
            if cand_val > val + 1e-9:
                ph, pts, val = cand_ph, cand_pts, cand_val
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= 250:
                    break

        if val > best_val:
            best_ph, best_pts, best_val = ph.copy(), pts.copy(), val

    return best_ph, best_pts, best_val


# ---- scoring on ACTUAL outcomes -------------------------------------------------

def field_points(picks_df, outcome_home_win, games_in_order):
    gi = {g: i for i, g in enumerate(games_in_order)}
    one = picks_df[picks_df["simulation"] == picks_df["simulation"].min()].copy()
    one["home_won"] = one["game"].map(gi).map(lambda i: bool(outcome_home_win[i]))
    one["correct"] = one["home_won"] == one["picked_home"].astype(bool)
    one["pts"] = one["correct"] * one["confidence"]
    return one.groupby("player")["pts"].sum()


def one_away(picks_df, actual, games_in_order):
    base = field_points(picks_df, actual, games_in_order)
    my, lead = base[ME], base.max()
    rank = int((base > my).sum() + 1)
    behind = int(lead - my)
    n_live = 0
    for gi in range(len(games_in_order)):
        flp = actual.copy()
        flp[gi] = not flp[gi]
        fp = field_points(picks_df, flp, games_in_order)
        if fp[ME] >= fp.max() and (fp[ME] > fp.drop(ME).max() or (fp == fp[ME]).sum() == 1):
            n_live += 1
    return rank, behind, n_live, int(n_live > 0)


def slate_dict(ph, pts, home, away, n):
    d = {}
    for gi in range(n):
        d[home[gi] if ph[gi] else away[gi]] = int(pts[gi])
    return d


def build_sim(gdf, y, year, num_sims):
    names = y.players["player_name"].tolist()
    sk = skills(year)
    sim = ConfidencePickEmSimulator(num_sims=num_sims)
    sim.add_games_from_dataframe(gdf.drop(columns=["actual_outcome"]))
    sim.players = [Player(nm, sk[nm]["skill_level"], sk[nm]["crowd_following"],
                          sk[nm]["confidence_following"]) if nm in sk
                   else Player(nm, 0.6, 0.5, 0.5) for nm in names]
    return sim


def greedy_and_leverage(gdf, y, year):
    sim = build_sim(gdf, y, year, 150)
    with contextlib.redirect_stdout(io.StringIO()):
        g = sim.optimize_picks(ME, confidence_range=3)
        lv = sim.optimize_picks_leverage(ME, lambda_lev=1.0, iterations=120,
                                         restarts=3, search_sims=60)
    return g, lv


def sim_win_pct(gdf, y, year, slate, num_sims=2000):
    """Simulator win_pct for ME playing `slate` -- sanity check on analytic P(win)."""
    sim = build_sim(gdf, y, year, num_sims)
    with contextlib.redirect_stdout(io.StringIO()):
        picks = sim.simulate_picks(fixed_picks={ME: slate})
        outcomes = sim.simulate_outcomes()
        stats = sim.analyze_results(picks, outcomes)
    return float(stats["win_pct"][ME])


def main():
    done = set()
    rows = []
    if os.path.exists(CHECKPOINT):
        prev = pd.read_csv(CHECKPOINT)
        done = set(zip(prev["year"], prev["week"]))
        rows = prev.to_dict("records")
        print("resuming; {} weeks done".format(len(done)), flush=True)

    for (year, week) in WEEKS:
        if (year, week) in done:
            print("{} wk{}: checkpointed".format(year, week), flush=True)
            continue
        t0 = time.time()
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
        gio = ["{}@{}".format(a, h) for a, h in zip(away, home)]
        actual = gdf["actual_outcome"].to_numpy(dtype=bool)
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

        rng = np.random.default_rng(SEED + week + year)
        O = rng.random((K_OUTCOMES, n)) < vh[None, :]
        pwin = make_pwin(opp_types, vh, O)

        t_opt = time.time()
        aph, apts, aval = hill_climb(pwin, home, away, vh, n, rng)
        opt_secs = time.time() - t_opt
        ana_slate = slate_dict(aph, apts, home, away, n)
        chalk_ph, chalk_pts = chalk_slate(home, away, vh, n)
        chalk_val = pwin(chalk_ph, chalk_pts)

        g_slate, lv_slate = greedy_and_leverage(gdf, y, year)

        def score(sl):
            pdf = convert_yahoo_picks_to_dataframe(y, num_sims=1, fixed_picks={ME: sl})
            return one_away(pdf, actual, gio)

        gr = score(g_slate)
        lr = score(lv_slate)
        ar = score(ana_slate)
        # actual entry
        adf = convert_yahoo_picks_to_dataframe(y, num_sims=1)
        act = one_away(adf, actual, gio)

        # simulator sanity check: does the analytic-optimized slate's P(win)
        # hold up under the (correlation-correct, if noisy) simulator, vs chalk?
        sim_chalk = sim_win_pct(gdf, y, year, slate_dict(chalk_ph, chalk_pts, home, away, n))
        sim_ana = sim_win_pct(gdf, y, year, ana_slate)

        row = dict(year=year, week=week, games=n, opp_types=len(opp_types),
                   chalk_pwin=chalk_val, ana_pwin=aval, opt_secs=opt_secs,
                   sim_chalk_winpct=sim_chalk, sim_ana_winpct=sim_ana,
                   act_rank=act[0], act_behind=act[1],
                   g_rank=gr[0], g_behind=gr[1], g_live=gr[2], g_oa=gr[3],
                   lv_rank=lr[0], lv_behind=lr[1], lv_live=lr[2], lv_oa=lr[3],
                   an_rank=ar[0], an_behind=ar[1], an_live=ar[2], an_oa=ar[3])
        rows.append(row)
        pd.DataFrame(rows).to_csv(CHECKPOINT, index=False)
        print("{} wk{:>2} ({:.0f}s): actual rk {:>2} | greedy rk {:>2} oa {} | "
              "lev rk {:>2} oa {} | ANALYTIC rk {:>2} oa {} live {}  "
              "| pwin {:.4f}->{:.4f}  sim_winpct {:.4f}->{:.4f}".format(
                  year, week, time.time() - t0, act[0],
                  gr[0], gr[3], lr[0], lr[3], ar[0], ar[3], ar[2],
                  chalk_val, aval, sim_chalk, sim_ana), flush=True)

    df = pd.DataFrame(rows)
    print("\n=== SUMMARY (n={}) ===".format(len(df)))
    print("{:<10} {:>5} {:>9} {:>10} {:>13} {:>10}".format(
        "variant", "wins", "one-away", "mean live", "median behind", "mean rank"))
    for tag, pfx in (("greedy", "g"), ("leverage", "lv"), ("analytic", "an")):
        print("{:<10} {:>5} {:>9} {:>10.1f} {:>13.0f} {:>10.1f}".format(
            tag,
            int((df[pfx + "_rank"] == 1).sum()),
            int(df[pfx + "_oa"].sum()),
            df[pfx + "_live"].mean(),
            df[pfx + "_behind"].median(),
            df[pfx + "_rank"].mean()))
    print("{:<10} {:>5} {:>9} {:>10} {:>13.0f} {:>10.1f}".format(
        "actual", int((df.act_rank == 1).sum()), "-", "-",
        df.act_behind.median(), df.act_rank.mean()))


if __name__ == "__main__":
    main()
