#!/usr/bin/env python
"""Analytical P(win) with cross-entry correlation via full outcome enumeration.

P(win) = sum over all 2^n game-outcome vectors o of
             P(o) * P(my score strictly highest | o)

where P(o) = prod_i  vegas_i^{o_i} (1 - vegas_i)^{1 - o_i}   (o_i = 1 iff home wins).

Two opponent models:
  MODAL  -- every opponent plays their single most-likely slate. Given o, every
            score is deterministic, so P(win|o) is an indicator. O(2^n * P) with
            a vectorized inner loop.
  UNCERT -- opponents keep pick uncertainty. Given o, opponent j's score is a
            weighted-Bernoulli over which of their (modal) picks came true; we
            take P(their pick i is "as modeled") = the modal pick prob, and
            P(pick i correct | o) follows. Convolve -> P(my_score > opp_j | o),
            then prod_j (independence *given o*, a mild assumption).

Benchmark both against the simulator's win_pct on the 4 check weeks.

Run:  python research/analytical-prototype/03_pwin_correlated.py
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
from confpickem.yahoo_pickem_integration import convert_yahoo_to_simulator_format

ME = "Firman's Educated Guesses"
LEAGUE_ID = 15435
COOKIES = "cookies.txt"
CACHE = {2024: "PickEmCache2024", 2025: "PickEmCache2025"}
CHECK_WEEKS = [(2024, 3), (2024, 12), (2025, 9), (2025, 12)]
SIM_N = 6000

_SKILLS = {}


def skills(year):
    if year not in _SKILLS:
        _SKILLS[year] = json.load(open("player_skills_{}.json".format(year)))["player_skills"]
    return _SKILLS[year]


def modal_entry(vegas_home, crowd_home_pct, chc, cac, cf, conf_foll, n):
    """(picked_home[n] bool, points[n] int) for one modeled entry."""
    p_home = np.clip(vegas_home * (1 - cf) + crowd_home_pct * cf, 0.0, 1.0)
    pick_home = p_home > 0.5
    chosen = np.where(pick_home, chc, cac)
    opposing = np.where(pick_home, cac, chc)
    conf_diff = (chosen - opposing) / (chosen + opposing)
    vegas_conf = np.abs(vegas_home - 0.5) * 2 * n
    score = chosen * (1 + conf_diff) * conf_foll + vegas_conf * (1 - conf_foll)
    points = pd.Series(score).rank(method="first").to_numpy().astype(int)
    return pick_home, points, p_home


def score_given_outcomes(pick_home, points, outcomes):
    """Total points for an entry under a boolean home-win outcome vector."""
    correct = (pick_home == outcomes)
    return int((points * correct).sum())


def all_outcome_probs(vegas_home):
    """Return (outcomes[2^n, n] bool, probs[2^n]) for every game-outcome vector."""
    n = len(vegas_home)
    idx = np.arange(1 << n)
    bits = ((idx[:, None] >> np.arange(n)[None, :]) & 1).astype(bool)  # bit i = game i home win
    # P(o) = prod_i p_i^{o_i} (1-p_i)^{1-o_i}
    logp = np.where(bits, np.log(vegas_home)[None, :], np.log1p(-vegas_home)[None, :])
    probs = np.exp(logp.sum(axis=1))
    return bits, probs


def pwin_modal(my_pick_home, my_points, opp_entries, vegas_home):
    """Exact P(win) with modal opponents, full 2^n enumeration (vectorized)."""
    n = len(vegas_home)
    bits, probs = all_outcome_probs(vegas_home)             # [M, n], [M]
    # my score under every outcome
    my_scores = (my_points[None, :] * (my_pick_home[None, :] == bits)).sum(axis=1)  # [M]
    # opponent scores: [P, M]
    opp_ph = np.array([e[0] for e in opp_entries])          # [P, n]
    opp_pts = np.array([e[1] for e in opp_entries])         # [P, n]
    # broadcast: for each opp p, each outcome m: sum_i pts[p,i] * (ph[p,i] == bits[m,i])
    # do it opp-by-opp to keep memory sane
    best_opp = np.zeros(len(probs))
    for p in range(len(opp_entries)):
        s = (opp_pts[p][None, :] * (opp_ph[p][None, :] == bits)).sum(axis=1)  # [M]
        best_opp = np.maximum(best_opp, s)
    win_mask = my_scores > best_opp
    tie_mask = my_scores == best_opp
    # count ties to split credit
    # (recompute how many opps share the max on tie outcomes)
    pwin = float(probs[win_mask].sum())
    if tie_mask.any():
        for m in np.where(tie_mask)[0]:
            k = 1
            for p in range(len(opp_entries)):
                s = int((opp_pts[p] * (opp_ph[p] == bits[m])).sum())
                if s == my_scores[m]:
                    k += 1
            pwin += probs[m] / k
    return pwin


def pwin_uncertain(my_pick_home, my_points, opp_entries, vegas_home):
    """P(win) keeping opponent pick uncertainty; independence across opps GIVEN o."""
    n = len(vegas_home)
    bits, probs = all_outcome_probs(vegas_home)
    my_scores = (my_points[None, :] * (my_pick_home[None, :] == bits)).sum(axis=1)

    # Pre-clip: only outcomes with non-trivial probability mass matter, but with
    # n<=16 just do them all. For each opp, precompute p_home.
    pwin = 0.0
    M = len(probs)
    # For speed, iterate outcomes; for each, build each opp's score PMF.
    for m in range(M):
        if probs[m] < 1e-12:
            continue
        o = bits[m]
        beat_prod = 1.0
        for (ph, pts, p_home) in opp_entries:
            # P(opp's pick i is correct | o): opp picks home w.p. p_home[i].
            # If home won (o[i]=1): correct w.p. p_home[i]; else w.p. 1-p_home[i].
            p_correct = np.where(o, p_home, 1.0 - p_home)
            opp_pmf = weighted_pmf(p_correct, pts)
            # P(my_score > opp score) with my score fixed at my_scores[m]
            k = my_scores[m]
            cdf = np.cumsum(opp_pmf)
            p_le = cdf[min(k - 1, len(cdf) - 1)] if k >= 1 else 0.0
            p_eq = opp_pmf[k] if k < len(opp_pmf) else 0.0
            beat_prod *= p_le + 0.5 * p_eq
        pwin += probs[m] * beat_prod
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
    print("{:<12} {:>10} {:>12} {:>12}   {:>7} {:>7}".format(
        "week", "sim", "modal", "uncertain", "M/sim", "U/sim"))
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

        # ME plays straight chalk (fixed reference slate)
        order = np.argsort(-np.abs(vegas_home - 0.5))
        slate = {}
        my_ph = np.zeros(n, dtype=bool)
        my_pts = np.zeros(n, dtype=int)
        for rank, gi in enumerate(order):
            pick_home = vegas_home[gi] >= 0.5
            team = home[gi] if pick_home else away[gi]
            slate[team] = n - rank
            my_ph[gi] = pick_home
            my_pts[gi] = n - rank

        opp_entries = []
        for _, prow in y.players.iterrows():
            nm = prow["player_name"]
            if nm == ME or nm not in sk:
                continue
            ph, pts, p_home = modal_entry(
                vegas_home, crowd_home_pct, chc, cac,
                sk[nm]["crowd_following"], sk[nm]["confidence_following"], n)
            opp_entries.append((ph, pts, p_home))

        t = time.time()
        pm = pwin_modal(my_ph, my_pts, opp_entries, vegas_home)
        tm = time.time() - t
        t = time.time()
        pu = pwin_uncertain(my_ph, my_pts, opp_entries, vegas_home)
        tu = time.time() - t
        ps = sim_pwin(y, gdf, slate, year)

        print("{:<12} {:>10.4f} {:>12.4f} {:>12.4f}   {:>7.2f} {:>7.2f}   "
              "(modal {:.1f}s, uncert {:.1f}s, n={})".format(
                  "{} wk{}".format(year, week), ps, pm, pu,
                  pm / ps if ps else float("nan"),
                  pu / ps if ps else float("nan"), tm, tu, n))


if __name__ == "__main__":
    main()
