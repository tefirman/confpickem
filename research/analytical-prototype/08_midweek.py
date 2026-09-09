#!/usr/bin/env python
"""Midweek backtest: analytical vs greedy, PR A (issue #14).

For each (year, week) we simulate a *midweek* decision: pretend the first
``CUTOFF`` games of the slate are already played (their real outcomes known, and
every player's real pick/confidence on them locked), then optimize the remaining
games two ways and score both on the FULL real outcome vector:

  * greedy  -> ConfidencePickEmSimulator.optimize_picks(player_data=...)
  * analytic-> ConfidencePickEmSimulator.optimize_picks_analytic(player_data=...,
               available_points=<unspent>)

Reported per week (scored on actual results): final rank, points behind 1st, and
whether a single remaining-game flip would have won it ("one away").

Run:  python research/analytical-prototype/08_midweek.py
Writes research/analytical-prototype/08_midweek_results.csv
"""

import sys
import os
import io
import json
import time
import contextlib
import warnings

sys.path.insert(0, "src")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import confpickem.yahoo_pickem_scraper as _scraper

# never expire the cache -- backtest is fully offline
_orig = _scraper.PageCache.get_cached_content
_scraper.PageCache.get_cached_content = lambda self, pt, wk, expiration=3600: _orig(
    self, pt, wk, expiration=10**13
)

from confpickem.yahoo_pickem_scraper import YahooPickEm  # noqa: E402
from confpickem.confidence_pickem_sim import ConfidencePickEmSimulator, Player  # noqa: E402
from confpickem.yahoo_pickem_integration import (  # noqa: E402
    convert_yahoo_to_simulator_format,
    convert_yahoo_picks_to_dataframe,
)

ME = "Firman's Educated Guesses"
LEAGUE_ID = 15435
COOKIES = "cookies.txt"
CACHE = {2024: "PickEmCache2024", 2025: "PickEmCache2025"}
WEEKS = [(2025, 2), (2025, 3), (2025, 6), (2025, 9), (2025, 12)]
CUTOFF = 6  # games 1..CUTOFF treated as already played
K_OUTCOMES = 6000
AN_ITERS = 400
AN_RESTARTS = 4
GREEDY_SIMS = 150
SEED = 51
OUT = "research/analytical-prototype/08_midweek_results.csv"


def skills(year):
    path = "player_skills_{}.json".format(year)
    return json.load(open(path))["player_skills"]


def eligible_players(y, cutoff):
    """Player rows with a real pick + confidence on every game 1..cutoff.

    A midweek run needs everyone's spent confidence; players who skipped the
    week entirely (and the greedy engine's simulate_picks chokes on their NaN
    picks) are dropped so greedy vs analytic is scored against the same field.
    """
    keep = []
    for _, r in y.players.iterrows():
        ok = all(
            pd.notna(r.get("game_{}_pick".format(i + 1)))
            and pd.notna(r.get("game_{}_confidence".format(i + 1)))
            for i in range(cutoff)
        )
        if ok:
            keep.append(r["player_name"])
    return keep


def build_sim(gdf, y, year, num_sims, cutoff):
    """Sim with games 1..cutoff completed (actual_outcome set), rest pending."""
    names = eligible_players(y, cutoff)
    sk = skills(year)
    sim = ConfidencePickEmSimulator(num_sims=num_sims)
    full_outcomes = gdf["actual_outcome"].tolist()
    sim.add_games_from_dataframe(gdf.drop(columns=["actual_outcome"]))
    for i, g in enumerate(sim.games):
        g.actual_outcome = full_outcomes[i] if i < cutoff else None
    sim.players = [
        (
            Player(
                nm, sk[nm]["skill_level"], sk[nm]["crowd_following"], sk[nm]["confidence_following"]
            )
            if nm in sk
            else Player(nm, 0.6, 0.5, 0.5)
        )
        for nm in names
    ]
    return sim


def unspent_confidence(y, gdf, cutoff):
    """1..N minus the confidence values ME already spent on games 1..cutoff."""
    n = len(gdf)
    row = y.players[y.players["player_name"] == ME].iloc[0]
    spent = set()
    for i in range(cutoff):
        c = row.get("game_{}_confidence".format(i + 1))
        if pd.notna(c) and int(c) > 0:
            spent.add(int(c))
    return set(range(1, n + 1)) - spent


def score_slate(y, actual, gio, slate):
    """Rank / points-behind / one-away for ME playing `slate`, on real results."""
    pdf = convert_yahoo_picks_to_dataframe(y, num_sims=1, fixed_picks={ME: slate})
    return _one_away(pdf, actual, gio)


def _field_points(picks_df, outcome_home_win, games_in_order):
    gi = {g: i for i, g in enumerate(games_in_order)}
    one = picks_df[picks_df["simulation"] == picks_df["simulation"].min()].copy()
    one["home_won"] = one["game"].map(gi).map(lambda i: bool(outcome_home_win[i]))
    one["correct"] = one["home_won"] == one["picked_home"].astype(bool)
    one["pts"] = one["correct"] * one["confidence"]
    return one.groupby("player")["pts"].sum()


def _one_away(picks_df, actual, games_in_order):
    base = _field_points(picks_df, actual, games_in_order)
    my, lead = base[ME], base.max()
    rank = int((base > my).sum() + 1)
    behind = int(lead - my)
    n_live = 0
    for gi in range(len(games_in_order)):
        flp = actual.copy()
        flp[gi] = not flp[gi]
        fp = _field_points(picks_df, flp, games_in_order)
        if fp[ME] >= fp.max() and (fp[ME] > fp.drop(ME).max() or (fp == fp[ME]).sum() == 1):
            n_live += 1
    return rank, behind, n_live, int(n_live > 0)


def slate_from_picks(picks_dict):
    return {t: int(c) for t, c in picks_dict.items()}


def main():
    rows = []
    if os.path.exists(OUT):
        prev = pd.read_csv(OUT)
        rows = prev.to_dict("records")
        done = set(zip(prev.year, prev.week))
        print("resuming; {} done".format(len(done)), flush=True)
    else:
        done = set()

    for year, week in WEEKS:
        if (year, week) in done:
            continue
        t0 = time.time()
        y = YahooPickEm(week=week, league_id=LEAGUE_ID, cookies_file=COOKIES, cache_dir=CACHE[year])
        gdf = convert_yahoo_to_simulator_format(y, ignore_results=False)
        if "actual_outcome" not in gdf.columns or gdf["actual_outcome"].isna().any():
            print("{} wk{}: SKIP (incomplete results)".format(year, week), flush=True)
            continue
        n = len(gdf)
        away = gdf["away_team"].tolist()
        home = gdf["home_team"].tolist()
        gio = ["{}@{}".format(a, h) for a, h in zip(away, home)]
        actual = gdf["actual_outcome"].to_numpy(dtype=bool)
        avail = unspent_confidence(y, gdf, CUTOFF)
        keep = set(eligible_players(y, CUTOFF))
        pdata = y.players[y.players["player_name"].isin(keep)].reset_index(drop=True)
        print(
            "   {}/{} players eligible (rest skipped the week)".format(len(pdata), len(y.players)),
            flush=True,
        )

        # greedy midweek
        gsim = build_sim(gdf, y, year, GREEDY_SIMS, CUTOFF)
        with contextlib.redirect_stdout(io.StringIO()):
            g_pick = gsim.optimize_picks(
                ME, confidence_range=3, available_points=set(avail), player_data=pdata
            )
        g_slate = slate_from_picks(g_pick)

        # analytic midweek
        asim = build_sim(gdf, y, year, 100, CUTOFF)
        with contextlib.redirect_stdout(io.StringIO()):
            a_pick = asim.optimize_picks_analytic(
                ME,
                iterations=AN_ITERS,
                restarts=AN_RESTARTS,
                n_outcomes=K_OUTCOMES,
                seed=SEED,
                available_points=set(avail),
                player_data=pdata,
            )
        a_slate = slate_from_picks(a_pick)

        gr = score_slate(y, actual, gio, g_slate)
        ar = score_slate(y, actual, gio, a_slate)
        adf = convert_yahoo_picks_to_dataframe(y, num_sims=1)
        act = _one_away(adf, actual, gio)

        row = dict(
            year=year,
            week=week,
            games=n,
            cutoff=CUTOFF,
            unspent=len(avail),
            act_rank=act[0],
            act_behind=act[1],
            g_rank=gr[0],
            g_behind=gr[1],
            g_live=gr[2],
            g_oa=gr[3],
            an_rank=ar[0],
            an_behind=ar[1],
            an_live=ar[2],
            an_oa=ar[3],
        )
        rows.append(row)
        pd.DataFrame(rows).to_csv(OUT, index=False)
        print(
            "{} wk{:>2} ({:.0f}s)  actual rk {:>2} | greedy rk {:>2} "
            "behind {:>2} oa {} | ANALYTIC rk {:>2} behind {:>2} oa {}".format(
                year, week, time.time() - t0, act[0], gr[0], gr[1], gr[3], ar[0], ar[1], ar[3]
            ),
            flush=True,
        )

    df = pd.DataFrame(rows)
    print("\n=== MIDWEEK SUMMARY (n={}, cutoff={}) ===".format(len(df), CUTOFF))
    print(
        "{:<10} {:>5} {:>9} {:>13} {:>10}".format(
            "variant", "wins", "one-away", "median behind", "mean rank"
        )
    )
    for tag, pfx in (("greedy", "g"), ("analytic", "an")):
        print(
            "{:<10} {:>5} {:>9} {:>13.0f} {:>10.1f}".format(
                tag,
                int((df[pfx + "_rank"] == 1).sum()),
                int(df[pfx + "_oa"].sum()),
                df[pfx + "_behind"].median(),
                df[pfx + "_rank"].mean(),
            )
        )
    print(
        "{:<10} {:>5} {:>9} {:>13.0f} {:>10.1f}".format(
            "actual", int((df.act_rank == 1).sum()), "-", df.act_behind.median(), df.act_rank.mean()
        )
    )


if __name__ == "__main__":
    main()
