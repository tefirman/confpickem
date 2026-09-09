"""Lambda sweep for optimize_picks_leverage, with the 'one game away' metric.

For each backtest week: build greedy once, then optimize_picks_leverage at each
lambda in LAMBDAS. Score every slate against actual outcomes and compute the
one-game-away counterfactual (flip ANY single game, re-score the whole field,
did we finish 1st?).

Run from repo root:  python scratchpad/lambda_sweep.py
Writes scratchpad/lambda_sweep_results.csv and prints a summary per lambda.
Checkpoints per week so a laptop sleep / kill doesn't lose progress.
"""
import sys, os, warnings, io, contextlib, json, time
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
LAMBDAS = [0.0, 1.0, 2.0, 3.0, 5.0]   # 0.0 = pure win-prob hill climb
LEV_ITERS, LEV_RESTARTS, LEV_SEARCH_SIMS = 120, 3, 60

CHECKPOINT = "scratchpad/lambda_sweep_results.csv"
STATUS = "scratchpad/lambda_sweep_status.txt"
_SKILLS = {}
_T_START = time.time()
_TOTAL_STEPS = len(WEEKS) * (1 + len(LAMBDAS))   # optimizer runs across the whole sweep
_STEPS_DONE = 0


def _status(msg):
    """Overwrite a one-glance status file and echo a timestamped line to stdout."""
    global _STEPS_DONE
    elapsed = time.time() - _T_START
    frac = _STEPS_DONE / _TOTAL_STEPS if _TOTAL_STEPS else 0
    eta = (elapsed / frac - elapsed) if frac > 0 else 0
    line = "[{:5.1f}m elapsed | {}/{} runs ({:.0%}) | ETA ~{:.0f}m] {}".format(
        elapsed / 60, _STEPS_DONE, _TOTAL_STEPS, frac, eta / 60, msg)
    print(line, flush=True)
    try:
        with open(STATUS, "w") as f:
            f.write(line + "\n")
    except OSError:
        pass


def skills(year):
    if year not in _SKILLS:
        _SKILLS[year] = json.load(open("player_skills_{}.json".format(year)))["player_skills"]
    return _SKILLS[year]


def build_field(names, year):
    s = skills(year)
    return [Player(n, s[n]["skill_level"], s[n]["crowd_following"], s[n]["confidence_following"])
            if n in s else Player(n, 0.6, 0.5, 0.5) for n in names]


def field_points(picks_df, outcome_home_win, games_in_order):
    gi = {g: i for i, g in enumerate(games_in_order)}
    one = picks_df[picks_df["simulation"] == picks_df["simulation"].min()].copy()
    one["home_won"] = one["game"].map(gi).map(lambda i: bool(outcome_home_win[i]))
    one["correct"] = one["home_won"] == one["picked_home"].astype(bool)
    one["pts"] = one["correct"] * one["confidence"]
    return one.groupby("player")["pts"].sum()


def evaluate(picks, y, actual_home_win, games_in_order):
    pdf = convert_yahoo_picks_to_dataframe(y, num_sims=1, fixed_picks={ME: picks})
    base = field_points(pdf, actual_home_win, games_in_order)
    my, lead = base[ME], base.max()
    rank = int((base > my).sum() + 1)
    behind = int(lead - my)
    n_live = 0
    for gi in range(len(games_in_order)):
        flipped = actual_home_win.copy()
        flipped[gi] = not flipped[gi]
        fp = field_points(pdf, flipped, games_in_order)
        if fp[ME] >= fp.max() and (fp[ME] > fp.drop(ME).max() or (fp == fp[ME]).sum() == 1):
            n_live += 1
    return dict(rank=rank, behind=behind, live_flips=n_live, one_away=int(n_live > 0))


def run():
    done = set()
    if os.path.exists(CHECKPOINT):
        prev = pd.read_csv(CHECKPOINT)
        done = set(zip(prev["year"], prev["week"]))
        rows = prev.to_dict("records")
        print("resuming; {} weeks already done".format(len(done)))
    else:
        rows = []

    global _STEPS_DONE
    # count already-finished weeks toward the progress denominator
    _STEPS_DONE = len(done) * (1 + len(LAMBDAS))

    for (year, week) in WEEKS:
        if (year, week) in done:
            print("{} wk{:>2}: (checkpointed)".format(year, week))
            continue
        t0 = time.time()
        _status("{} wk{:>2}: loading + building field...".format(year, week))
        y = YahooPickEm(week=week, league_id=LEAGUE_ID, cookies_file=COOKIES,
                        cache_dir=CACHE[year])
        gdf = convert_yahoo_to_simulator_format(y, ignore_results=False)
        gio = ["{}@{}".format(a, h) for a, h in zip(gdf["away_team"], gdf["home_team"])]
        actual = gdf["actual_outcome"].to_numpy(dtype=bool)

        sim = ConfidencePickEmSimulator(num_sims=NUM_SIMS)
        sim.add_games_from_dataframe(gdf.drop(columns=["actual_outcome"]))
        sim.players = build_field(y.players["player_name"].tolist(), year)

        _status("{} wk{:>2}: greedy...".format(year, week))
        with contextlib.redirect_stdout(io.StringIO()):
            greedy = sim.optimize_picks(ME, confidence_range=CONF_RANGE)
        gm = evaluate(greedy, y, actual, gio)
        _STEPS_DONE += 1
        row = dict(year=year, week=week, games=len(gio),
                   g_rank=gm["rank"], g_behind=gm["behind"],
                   g_live=gm["live_flips"], g_oneaway=gm["one_away"])
        _status("{} wk{:>2}: greedy done -> rk {} live {} oa {}".format(
            year, week, gm["rank"], gm["live_flips"], gm["one_away"]))

        parts = ["greedy rk {:>2} live {} oneaway {}".format(
            gm["rank"], gm["live_flips"], gm["one_away"])]
        for lam in LAMBDAS:
            _status("{} wk{:>2}: leverage λ{:g}...".format(year, week, lam))
            with contextlib.redirect_stdout(io.StringIO()):
                lev = sim.optimize_picks_leverage(
                    ME, lambda_lev=lam, iterations=LEV_ITERS,
                    restarts=LEV_RESTARTS, search_sims=LEV_SEARCH_SIMS)
            lm = evaluate(lev, y, actual, gio)
            _STEPS_DONE += 1
            tag = "l{}".format(str(lam).replace(".", "_"))
            row[tag + "_rank"] = lm["rank"]
            row[tag + "_behind"] = lm["behind"]
            row[tag + "_live"] = lm["live_flips"]
            row[tag + "_oneaway"] = lm["one_away"]
            parts.append("λ{:g} rk {:>2} live {} oa {}".format(
                lam, lm["rank"], lm["live_flips"], lm["one_away"]))
            _status("{} wk{:>2}: λ{:g} done -> rk {} live {} oa {}".format(
                year, week, lam, lm["rank"], lm["live_flips"], lm["one_away"]))

        rows.append(row)
        pd.DataFrame(rows).to_csv(CHECKPOINT, index=False)
        _status("{} wk{:>2} COMPLETE ({:.0f}s): {}".format(
            year, week, time.time() - t0, "  |  ".join(parts)))

    _status("ALL WEEKS COMPLETE - writing summary")
    df = pd.DataFrame(rows)
    print("\n=== SUMMARY (n={}) ===".format(len(df)))
    hdr = "{:<10} {:>6} {:>10} {:>12} {:>12} {:>10}".format(
        "variant", "wins", "one-away", "mean live", "med behind", "mean rank")
    print(hdr)
    print("{:<10} {:>6} {:>10} {:>12.1f} {:>12.1f} {:>10.1f}".format(
        "greedy", int((df.g_rank == 1).sum()), int(df.g_oneaway.sum()),
        df.g_live.mean(), df.g_behind.median(), df.g_rank.mean()))
    for lam in LAMBDAS:
        tag = "l{}".format(str(lam).replace(".", "_"))
        print("{:<10} {:>6} {:>10} {:>12.1f} {:>12.1f} {:>10.1f}".format(
            "leverage λ{:g}".format(lam),
            int((df[tag + "_rank"] == 1).sum()), int(df[tag + "_oneaway"].sum()),
            df[tag + "_live"].mean(), df[tag + "_behind"].median(),
            df[tag + "_rank"].mean()))


if __name__ == "__main__":
    run()
