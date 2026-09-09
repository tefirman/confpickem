"""'One game away' analysis for greedy vs leverage optimizer.

For each backtest week, rebuild both optimizers' slates, then ask: is there ANY
single game whose result, if flipped, would make our entry finish 1st? (Re-scoring
the whole field under each counterfactual -- a flip can cost the current leader
points too.) A week where some flip wins it for us was a "live shot" regardless of
final rank.

Run from repo root:  python scratchpad/one_game_away.py
Writes scratchpad/one_game_away_results.csv
"""
import sys, warnings, io, contextlib, json, time
sys.path.insert(0, "src")
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

NUM_SIMS = 150
CONF_RANGE = 3
LAMBDA_LEV = 1.0
LEV_ITERS, LEV_RESTARTS, LEV_SEARCH_SIMS = 120, 3, 60

_SKILLS = {}


def skills(year):
    if year not in _SKILLS:
        _SKILLS[year] = json.load(open("player_skills_{}.json".format(year)))["player_skills"]
    return _SKILLS[year]


def build_field(names, year):
    s = skills(year)
    return [Player(n, s[n]["skill_level"], s[n]["crowd_following"], s[n]["confidence_following"])
            if n in s else Player(n, 0.6, 0.5, 0.5) for n in names]


def field_points(picks_df, outcome_home_win, games_in_order):
    """DataFrame [player, points] for one outcome vector (first sim of picks_df)."""
    gi = {g: i for i, g in enumerate(games_in_order)}
    one = picks_df[picks_df["simulation"] == picks_df["simulation"].min()].copy()
    one["home_won"] = one["game"].map(gi).map(lambda i: bool(outcome_home_win[i]))
    one["correct"] = one["home_won"] == one["picked_home"].astype(bool)
    one["pts"] = one["correct"] * one["confidence"]
    return one.groupby("player")["pts"].sum()


def one_away(picks_df, actual_home_win, games_in_order):
    """Return dict: rank, pts_behind_first, n_live_flips, one_flip_to_win."""
    base = field_points(picks_df, actual_home_win, games_in_order)
    my = base[ME]
    lead = base.max()
    rank = int((base > my).sum() + 1)
    pts_behind = int(lead - my)

    n_live = 0
    for gi in range(len(games_in_order)):
        flipped = actual_home_win.copy()
        flipped[gi] = not flipped[gi]
        fp = field_points(picks_df, flipped, games_in_order)
        if fp[ME] >= fp.max() and (fp[ME] > fp.drop(ME).max() or (fp == fp[ME]).sum() == 1):
            n_live += 1
    return dict(rank=rank, pts_behind_first=pts_behind,
                n_live_flips=n_live, one_flip_to_win=int(n_live > 0))


def run():
    rows = []
    for (year, week) in WEEKS:
        t0 = time.time()
        y = YahooPickEm(week=week, league_id=LEAGUE_ID, cookies_file=COOKIES,
                        cache_dir=CACHE[year])
        gdf = convert_yahoo_to_simulator_format(y, ignore_results=False)
        gio = ["{}@{}".format(a, h) for a, h in zip(gdf["away_team"], gdf["home_team"])]
        actual = gdf["actual_outcome"].to_numpy(dtype=bool)

        sim = ConfidencePickEmSimulator(num_sims=NUM_SIMS)
        sim.add_games_from_dataframe(gdf.drop(columns=["actual_outcome"]))
        sim.players = build_field(y.players["player_name"].tolist(), year)

        with contextlib.redirect_stdout(io.StringIO()):
            greedy = sim.optimize_picks(ME, confidence_range=CONF_RANGE)
            lever = sim.optimize_picks_leverage(
                ME, lambda_lev=LAMBDA_LEV, iterations=LEV_ITERS,
                restarts=LEV_RESTARTS, search_sims=LEV_SEARCH_SIMS)

        gd = convert_yahoo_picks_to_dataframe(y, num_sims=1, fixed_picks={ME: greedy})
        ld = convert_yahoo_picks_to_dataframe(y, num_sims=1, fixed_picks={ME: lever})
        g = one_away(gd, actual, gio)
        l = one_away(ld, actual, gio)

        rows.append(dict(
            year=year, week=week, games=len(gio),
            g_rank=g["rank"], g_behind=g["pts_behind_first"],
            g_live_flips=g["n_live_flips"], g_one_away=g["one_flip_to_win"],
            l_rank=l["rank"], l_behind=l["pts_behind_first"],
            l_live_flips=l["n_live_flips"], l_one_away=l["one_flip_to_win"],
        ))
        print("{} wk{:>2}: greedy rank {:>2} behind {:>3} live {} oneaway {}  |  "
              "leverage rank {:>2} behind {:>3} live {} oneaway {}  ({:.0f}s)".format(
                  year, week, g["rank"], g["pts_behind_first"], g["n_live_flips"],
                  g["one_flip_to_win"], l["rank"], l["pts_behind_first"],
                  l["n_live_flips"], l["one_flip_to_win"], time.time() - t0))

    df = pd.DataFrame(rows)
    df.to_csv("scratchpad/one_game_away_results.csv", index=False)
    print()
    print("=== SUMMARY (n={}) ===".format(len(df)))
    print("                       greedy   leverage")
    print("weeks won              {:>6} {:>10}".format(
        int((df.g_rank == 1).sum()), int((df.l_rank == 1).sum())))
    print("weeks 'one game away'  {:>6} {:>10}".format(
        int(df.g_one_away.sum()), int(df.l_one_away.sum())))
    print("mean live-flip count   {:>6.1f} {:>10.1f}".format(
        df.g_live_flips.mean(), df.l_live_flips.mean()))
    print("mean pts behind 1st    {:>6.1f} {:>10.1f}".format(
        df.g_behind.mean(), df.l_behind.mean()))
    print("median pts behind 1st  {:>6.1f} {:>10.1f}".format(
        df.g_behind.median(), df.l_behind.median()))
    print("mean rank              {:>6.1f} {:>10.1f}".format(
        df.g_rank.mean(), df.l_rank.mean()))


if __name__ == "__main__":
    run()
