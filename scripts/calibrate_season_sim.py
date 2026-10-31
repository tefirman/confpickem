#!/usr/bin/env python
"""Backtest the season simulator against finished past seasons.

For each held-out season Y and cutoff week k, forecast the rest of the season
from the standings after week k -- using only information available then:

* player skills built from the *other* seasons' raw stats (leave-one-season-out);
  entrants with no history get the league-mean skills;
* future slates sampled from every cached week except Y's own weeks after k.

...and compare against what actually happened (final totals, weekly winners,
season prizes). Season = the weeks whose cached pages are final (some pages were
cached before their games were played, so a season may be shorter than 18).

Run from the repo root:  python scripts/calibrate_season_sim.py [--sims 5000]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from confpickem.analyze_player_skills import skills_from_raw_stats  # noqa: E402
from confpickem.apply_realistic_skills import combine_raw_stats  # noqa: E402
from confpickem.season_sim import (  # noqa: E402
    SEASON_PRIZES,
    WEEKLY_PRIZE,
    SeasonSimulator,
    available_cache_years,
    estimate_ability_sd,
    load_season_history,
    load_week_templates,
    season_payouts,
    _win_credit,
    spread_to_win_prob,
)


def skills_excluding(year, root="."):
    """Skills from every other season's raw stats."""
    raw = None
    for y in available_cache_years(root):
        path = Path(root) / f"player_skills_{y}.json"
        if y == year or not path.exists():
            continue
        stats = json.load(open(path)).get("raw_player_stats")
        if not stats:
            continue
        raw = stats if raw is None else combine_raw_stats(raw, stats)
    return skills_from_raw_stats(raw) if raw else {}


def actual_outcome(weekly, k, prizes):
    """Realised weekly wins after week k, and final season dollars (ties split)."""
    after = weekly[:, k:]
    wins = np.zeros(weekly.shape[0])
    for w in range(after.shape[1]):
        col = after[:, [w]].T
        wins += _win_credit(col)[0]
    dollars, _ = season_payouts(weekly.sum(axis=1)[None, :], prizes)
    return wins, dollars[0], weekly.sum(axis=1)


def run(sims, cutoffs, root=".", **sim_kwargs):
    rows = []
    summaries = []
    for year in (2024, 2025):
        hist = load_season_history(year, root)
        W = len(hist.weeks)
        skills_by_name = skills_excluding(year, root)
        for k in cutoffs:
            exclude = {(year, wk) for wk in hist.weeks[k:]}
            templates = load_week_templates(root, exclude=exclude)
            banked = hist.weekly[:, :k]
            pool = np.flatnonzero(banked.sum(axis=1) > 0)  # entrants seen by week k
            names = [hist.names[i] for i in pool]
            # ability SD from the *other* seasons only (no peeking at this one)
            others = [
                load_season_history(y, root).weekly
                for y in available_cache_years(root)
                if y != year
            ]
            tau, _ = estimate_ability_sd(others)
            sim = SeasonSimulator.from_weekly(
                names,
                banked[pool],
                skills_by_name,
                templates,
                total_weeks=W,
                ability_sd=tau,
                num_sims=sims,
                seed=51,
                **sim_kwargs,
            )
            matched = sim.skills_matched
            out = sim.run().set_index("player").loc[names]

            # --- reality ---
            wins, dollars, final = actual_outcome(hist.weekly, k, SEASON_PRIZES)
            real = pd.DataFrame(
                {"wins": wins[pool], "season_dollars": dollars[pool], "final": final[pool]},
                index=names,
            )
            newcomers = len(hist.names) - len(pool)

            df = out.join(real)
            df["year"], df["k"] = year, k
            df["missed"] = (hist.weekly[pool, k:] == 0).sum(axis=1)  # weeks not played after k
            # z of final points *relative to the field*: the field-wide level (how chalky
            # the remaining weeks turn out) is common to everyone and cancels in the ranking.
            live = df["active"].to_numpy()
            rel_real = df["final"] - df.loc[live, "final"].mean()
            rel_exp = df["exp_final_points"] - df.loc[live, "exp_final_points"].mean()
            df["z"] = ((rel_real - rel_exp) / df["sd_rel_points"]).where(live)
            df["real_dollars"] = df["season_dollars"] + df["wins"] * WEEKLY_PRIZE
            df["real_top4"] = df["season_dollars"] > 0
            df["real_first"] = df["final"] == df["final"].max()
            rows.append(df.reset_index())

            summaries.append(
                dict(
                    year=year,
                    k=k,
                    weeks_left=W - k,
                    players=len(pool),
                    newcomers=newcomers,
                    skills_matched=matched,
                    wk_pts_sim=float((out["exp_final_points"] - out["points"]).mean() / (W - k)),
                    wk_pts_real=float((hist.weekly[pool, k:].mean())),
                    tau=tau,
                    mean_z=float(df["z"].mean()),
                    sd_z=float(df["z"].std()),
                    frac_abs_z_gt2=float((df["z"].abs() > 2).mean()),
                    missed_pred=float(
                        ((1 - sim.play_prob) * sim.active * (W - k) + (~sim.active) * (W - k)).sum()
                    ),
                    missed_real=int((hist.weekly[pool, k:] == 0).sum()),
                )
            )
    return pd.concat(rows, ignore_index=True), pd.DataFrame(summaries)


def reliability(p, y, edges):
    cut = pd.cut(p, edges, include_lowest=True)
    g = pd.DataFrame({"p": p, "y": y, "bin": cut}).groupby("bin", observed=True)
    return g.agg(
        n=("y", "size"), predicted=("p", "mean"), actual=("y", "mean"), positives=("y", "sum")
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sims", type=int, default=5000)
    ap.add_argument("--cutoffs", default="4,8,12")
    ap.add_argument("--csv", help="write the per-player forecast/outcome rows here")
    args = ap.parse_args()
    cutoffs = [int(x) for x in args.cutoffs.split(",")]
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 30)

    df, summ = run(args.sims, cutoffs)
    if args.csv:
        df.to_csv(args.csv, index=False)

    print("\n== Per season / cutoff ==")
    print(summ.round(3).to_string(index=False))

    print("\n== Final points relative to the field, z-scored (should be mean 0, sd 1) ==")
    z = df[df["z"].notna()]
    for label, sub in (
        ("all active entrants", z),
        ("played every remaining week", z[z["missed"] == 0]),
        ("missed 1+ remaining weeks", z[z["missed"] > 0]),
    ):
        print(
            f"{label:<30} n={len(sub):<4} mean z={sub['z'].mean():+.2f}  "
            f"sd z={sub['z'].std():.2f}  "
            f"|z|>2: {(sub['z'].abs() > 2).mean():.1%} (nominal 4.6%)"
        )
    print(
        "(a missed week costs ~80 pts, so misses dominate the tails; players who played "
        "every week also read slightly high because the forecast prices in possible misses)"
    )

    print("\n== Weekly wins after the cutoff: predicted vs actual ==")
    df["wk_bucket"] = pd.qcut(df["exp_weekly_wins"], 4, duplicates="drop")
    print(
        df.groupby("wk_bucket", observed=True)
        .agg(n=("wins", "size"), predicted=("exp_weekly_wins", "sum"), actual=("wins", "sum"))
        .round(2)
        .to_string()
    )
    print(f"total predicted {df['exp_weekly_wins'].sum():.1f}  actual {df['wins'].sum():.1f}")

    print("\n== P(top 4) reliability ==")
    print(
        reliability(
            df["p_top4"], df["real_top4"].astype(float), [0, 0.02, 0.05, 0.10, 0.20, 0.35, 1]
        )
        .round(3)
        .to_string()
    )
    base = df.groupby(["year", "k"])["real_top4"].transform("mean")
    print(
        f"Brier model {((df['p_top4'] - df['real_top4']) ** 2).mean():.4f}  "
        f"vs flat-rate {((base - df['real_top4']) ** 2).mean():.4f}  "
        f"vs current-rank-only (top-4 now = 1) "
        f"{(((df['current_rank'] <= 4).astype(float) - df['real_top4']) ** 2).mean():.4f}"
    )

    print("\n== P(1st) reliability ==")
    print(
        reliability(df["p_place_1"], df["real_first"].astype(float), [0, 0.01, 0.03, 0.08, 0.2, 1])
        .round(3)
        .to_string()
    )

    print("\n== Expected dollars: predicted vs realised (by predicted quintile) ==")
    df["d_bucket"] = pd.qcut(df["exp_dollars"], 5, duplicates="drop")
    print(
        df.groupby("d_bucket", observed=True)
        .agg(
            n=("real_dollars", "size"),
            predicted=("exp_dollars", "mean"),
            realised=("real_dollars", "mean"),
        )
        .round(1)
        .to_string()
    )
    corr = df["exp_dollars"].corr(df["real_dollars"])
    print(
        f"corr(predicted, realised) = {corr:.3f}; "
        f"mean predicted ${df['exp_dollars'].mean():.0f} vs "
        f"realised ${df['real_dollars'].mean():.0f}"
    )

    print("\n== Favorite win rate vs the spread map (all cached games) ==")
    sp, fw = [], []
    for y in available_cache_years():
        try:
            h = load_season_history(y)
        except ValueError:
            continue
        sp.append(h.fav_spread)
        fw.append(h.fav_won)
    sp, fw = np.concatenate(sp), np.concatenate(fw)
    g = pd.DataFrame({"spread": sp, "won": fw})
    g["bucket"] = pd.cut(g["spread"], [-0.1, 2.5, 4.5, 7, 10, 30])
    g["model"] = spread_to_win_prob(g["spread"])
    print(
        g.groupby("bucket", observed=True)
        .agg(n=("won", "size"), model=("model", "mean"), actual=("won", "mean"))
        .round(3)
        .to_string()
    )
    print(f"all {len(g)} games: model {g['model'].mean():.3f}  actual {g['won'].mean():.3f}")


if __name__ == "__main__":
    main()
