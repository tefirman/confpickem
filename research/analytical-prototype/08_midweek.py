#!/usr/bin/env python
"""Midweek backtest: analytical vs greedy, PR A (issue #14).

Runs against the code on the ``feat/analytical-midweek`` branch.

Two scenario families, both scored on the FULL real outcome vector:

  ("cutoff", k)
      First k games of the slate are already *finished* (outcomes known, every
      player's real pick/confidence on them locked). Optimize the rest.

  ("kickoff", f)
      Only f games are finished, but every game that has kicked off is *frozen*:
      picks locked, outcome still unknown. as_of is set one hour past the f-th
      game's kickoff, so the Thursday-nighter plus the whole Sunday-1pm wave are
      frozen-but-live -- the real mid-Sunday state, since the pool locks all
      entries once the first game starts. Greedy has no frozen-but-live concept,
      so it can only lock the f finished games and "re-optimizes" games it
      couldn't actually change -- that handicap is part of the finding.

Reported per (week, scenario): final rank, points behind 1st, one-away.

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
# scenario: ("cutoff", k) or ("kickoff", f)
SCENARIOS = [("cutoff", 6), ("cutoff", 2), ("kickoff", 2)]
K_OUTCOMES = 6000
AN_ITERS = 400
AN_RESTARTS = 4
GREEDY_SIMS = 120
SEED = 51
OUT = "research/analytical-prototype/08_midweek_results.csv"


def skills(year):
    path = "player_skills_{}.json".format(year)
    return json.load(open(path))["player_skills"]


def eligible_players(y, ngames):
    """Player rows with a real pick + confidence on *every* game.

    A midweek run needs everyone's spent confidence; players who skipped the
    week (and the greedy engine's simulate_picks chokes on their NaN picks) are
    dropped so greedy vs analytic is scored against the same field.
    """
    keep = []
    for _, r in y.players.iterrows():
        ok = all(
            pd.notna(r.get("game_{}_pick".format(i + 1)))
            and pd.notna(r.get("game_{}_confidence".format(i + 1)))
            for i in range(ngames)
        )
        if ok:
            keep.append(r["player_name"])
    return keep


def frozen_mask(gdf, scenario):
    """(final_idx set, frozen_idx set, as_of or None) for a scenario."""
    n = len(gdf)
    kinds, k = scenario
    kt = gdf["kickoff_time"]
    order = sorted(range(n), key=lambda i: kt.iloc[i])
    if kinds == "cutoff":
        final = set(order[:k])
        return final, set(final), None
    # "kickoff": k finished; as_of = 1h past the k-th game's kickoff, so every
    # game that has started (Thursday + the Sunday-1pm wave) is frozen-but-live.
    final = set(order[:k])
    as_of = kt.iloc[order[k - 1]] + pd.Timedelta(hours=1)
    frozen = {i for i in range(n) if kt.iloc[i] <= as_of} | final
    return final, frozen, as_of


def build_sim(gdf, y, year, num_sims, final_idx, names):
    sk = skills(year)
    sim = ConfidencePickEmSimulator(num_sims=num_sims)
    full_outcomes = gdf["actual_outcome"].tolist()
    sim.add_games_from_dataframe(gdf.drop(columns=["actual_outcome"]))
    for i, g in enumerate(sim.games):
        g.actual_outcome = full_outcomes[i] if i in final_idx else None
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


def my_spent(y, gdf, idxs):
    """Confidence ME spent on the given game indices."""
    row = y.players[y.players["player_name"] == ME].iloc[0]
    spent = set()
    for i in idxs:
        c = row.get("game_{}_confidence".format(i + 1))
        if pd.notna(c) and int(c) > 0:
            spent.add(int(c))
    return spent


def score_slate(y, actual, gio, slate):
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
        done = set(zip(prev.year, prev.week, prev.scenario))
        print("resuming; {} done".format(len(done)), flush=True)
    else:
        done = set()

    for year, week in WEEKS:
        y = YahooPickEm(week=week, league_id=LEAGUE_ID, cookies_file=COOKIES, cache_dir=CACHE[year])
        gdf = convert_yahoo_to_simulator_format(y, ignore_results=False)
        if "actual_outcome" not in gdf.columns or gdf["actual_outcome"].isna().any():
            print("{} wk{}: SKIP (incomplete results)".format(year, week), flush=True)
            continue
        n = len(gdf)
        gio = [
            "{}@{}".format(a, h) for a, h in zip(gdf["away_team"].tolist(), gdf["home_team"].tolist())
        ]
        actual = gdf["actual_outcome"].to_numpy(dtype=bool)
        names = eligible_players(y, n)
        pdata = y.players[y.players["player_name"].isin(set(names))].reset_index(drop=True)

        for scenario in SCENARIOS:
            tag = "{}{}".format(*scenario)
            if (year, week, tag) in done:
                continue
            t0 = time.time()
            final_idx, frozen_idx, as_of = frozen_mask(gdf, scenario)
            free_idx = [i for i in range(n) if i not in frozen_idx]
            avail = set(range(1, n + 1)) - my_spent(y, gdf, frozen_idx)
            print(
                "   {} wk{} [{}] {}/{} eligible | {} final, {} frozen, {} free".format(
                    year, week, tag, len(pdata), len(y.players),
                    len(final_idx), len(frozen_idx), len(free_idx),
                ),
                flush=True,
            )

            # greedy: can only lock the *finished* games
            gsim = build_sim(gdf, y, year, GREEDY_SIMS, final_idx, names)
            g_avail = set(range(1, n + 1)) - my_spent(y, gdf, final_idx)
            with contextlib.redirect_stdout(io.StringIO()):
                g_pick = gsim.optimize_picks(
                    ME, confidence_range=3, available_points=g_avail, player_data=pdata
                )
            g_slate = slate_from_picks(g_pick)

            # analytic: locks every frozen game via as_of
            asim = build_sim(gdf, y, year, 100, final_idx, names)
            with contextlib.redirect_stdout(io.StringIO()):
                a_pick = asim.optimize_picks_analytic(
                    ME,
                    iterations=AN_ITERS,
                    restarts=AN_RESTARTS,
                    n_outcomes=K_OUTCOMES,
                    seed=SEED,
                    player_data=pdata,
                    as_of=as_of,
                )
            a_slate = slate_from_picks(a_pick)

            gr = score_slate(y, actual, gio, g_slate)
            ar = score_slate(y, actual, gio, a_slate)
            act = _one_away(convert_yahoo_picks_to_dataframe(y, num_sims=1), actual, gio)

            rows.append(
                dict(
                    year=year, week=week, scenario=tag, games=n,
                    final=len(final_idx), frozen=len(frozen_idx), free=len(free_idx),
                    act_rank=act[0], act_behind=act[1],
                    g_rank=gr[0], g_behind=gr[1], g_live=gr[2], g_oa=gr[3],
                    an_rank=ar[0], an_behind=ar[1], an_live=ar[2], an_oa=ar[3],
                )
            )
            pd.DataFrame(rows).to_csv(OUT, index=False)
            print(
                "   -> ({:.0f}s) actual rk {:>2} | greedy rk {:>2} behind {:>2} oa {} "
                "| ANALYTIC rk {:>2} behind {:>2} oa {}".format(
                    time.time() - t0, act[0], gr[0], gr[1], gr[3], ar[0], ar[1], ar[3]
                ),
                flush=True,
            )

    df = pd.DataFrame(rows)
    for tag in df.scenario.unique():
        sub = df[df.scenario == tag]
        print("\n=== {} (n={}) ===".format(tag, len(sub)))
        print(
            "{:<10} {:>5} {:>9} {:>13} {:>10}".format(
                "variant", "wins", "one-away", "median behind", "mean rank"
            )
        )
        for name, pfx in (("greedy", "g"), ("analytic", "an")):
            print(
                "{:<10} {:>5} {:>9} {:>13.0f} {:>10.1f}".format(
                    name,
                    int((sub[pfx + "_rank"] == 1).sum()),
                    int(sub[pfx + "_oa"].sum()),
                    sub[pfx + "_behind"].median(),
                    sub[pfx + "_rank"].mean(),
                )
            )
        print(
            "{:<10} {:>5} {:>9} {:>13.0f} {:>10.1f}".format(
                "actual", int((sub.act_rank == 1).sum()), "-",
                sub.act_behind.median(), sub.act_rank.mean(),
            )
        )


if __name__ == "__main__":
    main()
