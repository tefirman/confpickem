#!/usr/bin/env python
"""Prototype: value the opener (TNF / international / Thanksgiving) adaptively.

Instead of asking "which full slate is best if I commit to everything now?",
ask "which pick on the early-locking game is best given I'll re-optimize the
rest after seeing its result?" -- see
``ConfidencePickEmSimulator.optimize_opener_adaptive``.

Usage (repo root, needs cookies.txt + network):

    python scripts/adaptive_opener.py --week 12 [--player NAME] [--live-odds]
        [--game-idx N] [--restarts N] [--iterations N] [--top N]
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from confpickem.yahoo_pickem_scraper import YahooPickEm  # noqa: E402
from confpickem.yahoo_pickem_integration import convert_yahoo_to_simulator_format  # noqa: E402
from confpickem.confidence_pickem_sim import ConfidencePickEmSimulator, Player  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--week", type=int, required=True)
    ap.add_argument("--league-id", default="11465")
    ap.add_argument("--player", default="Firman's Educated Guesses")
    ap.add_argument("--live-odds", action="store_true")
    ap.add_argument("--odds-api-key", default=None)
    ap.add_argument(
        "--game-idx",
        type=int,
        default=None,
        help="game index to treat as the opener (default: earliest kickoff)",
    )
    ap.add_argument("--restarts", type=int, default=4)
    ap.add_argument("--iterations", type=int, default=400)
    ap.add_argument("--outcomes", type=int, default=6000)
    ap.add_argument(
        "--test-outcomes",
        type=int,
        default=8000,
        help="holdout draws used to score slates out-of-sample",
    )
    ap.add_argument("--seed", type=int, default=51)
    ap.add_argument("--top", type=int, default=8)
    args = ap.parse_args()

    yahoo = YahooPickEm(week=args.week, league_id=args.league_id, cookies_file="cookies.txt")
    if len(yahoo.games) == 0:
        print("No games found -- cookies expired or week not posted yet.")
        return 1
    if args.live_odds:
        from confpickem.live_odds_scraper import update_odds_with_live_data

        yahoo.games = update_odds_with_live_data(
            yahoo.games, week=args.week, odds_api_key=args.odds_api_key
        )

    games_df = convert_yahoo_to_simulator_format(yahoo, ignore_results=True)
    sim = ConfidencePickEmSimulator()
    sim.add_games_from_dataframe(games_df)
    try:
        skills = json.load(open("current_player_skills.json"))
    except FileNotFoundError:
        skills = {}
    sim.players = [
        Player(
            name=n,
            skill_level=skills.get(n, {}).get("skill_level", 0.6),
            crowd_following=skills.get(n, {}).get("crowd_following", 0.5),
            confidence_following=skills.get(n, {}).get("confidence_following", 0.5),
        )
        for n in yahoo.players["player_name"]
    ]

    t0 = time.time()
    res = sim.optimize_opener_adaptive(
        args.player,
        game_idx=args.game_idx,
        iterations=args.iterations,
        restarts=args.restarts,
        n_outcomes=args.outcomes,
        n_test=args.test_outcomes,
        seed=args.seed,
    )
    jt, jc, jv = res.attrs["joint"]
    print(
        f"\nOpener: {res.attrs['game']}  (home win prob {res.attrs['home_win_prob']:.1%})"
        f"   [{time.time() - t0:.0f}s]"
    )
    print(f"Joint-optimal slate's opener pick: {jt} {jc}  (P(win) {jv:.2%})\n")
    fmt = {
        c: "{:.2%}".format
        for c in ("commit", "adaptive", "gain", "gain_se", "p_if_win", "p_if_lose")
    }
    print("Top picks by adaptive value:")
    print(res.head(args.top).to_string(index=False, formatters=fmt))
    row = res[(res.team == jt) & (res.conf == jc)].iloc[0]
    best = res.iloc[0]
    print(
        f"\nJoint pick adaptive value {row.adaptive:.2%} vs best adaptive "
        f"({best.team} {int(best.conf)}) {best.adaptive:.2%} "
        f"-> {100 * (best.adaptive - row.adaptive):.2f} pts of P(win)"
    )
    print("\nBest confidence per team:")
    print(
        res.sort_values("adaptive", ascending=False)
        .groupby("team")
        .head(1)
        .to_string(index=False, formatters=fmt)
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
