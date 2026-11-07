#!/usr/bin/env python
"""
Live tracker: re-score a locked pool as the games play and log win probability over time.

Each tick re-scrapes Yahoo, re-pulls live odds, recomputes every entry's win %
and appends a snapshot to a JSONL history file (default NFL_Week<N>_live.jsonl).
Run it from the first Sunday kickoff; it stops by itself once every game is final.

Usage:
  live.py --week 10                         # refresh every 10 minutes until all games final
  live.py --week 10 --once                  # one snapshot, then exit
  live.py --week 10 --interval 5 --player "Educated Guesses"
"""

import sys
import time
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.confpickem.live_tracker import append_snapshot, take_snapshot


def format_snapshot(snapshot, player=None, top=8):
    """Console summary of one snapshot: game counts, stale warnings, top entries."""
    states = snapshot["games"]
    counts = {k: sum(s["status"] == k for s in states) for k in ("final", "live", "upcoming")}
    lines = [
        f"[{snapshot['timestamp']}] week {snapshot['week']}: {counts['final']} final, "
        f"{counts['live']} live, {counts['upcoming']} upcoming"
    ]
    if snapshot["stale_games"]:
        lines.append(
            "  ⚠️  no live line for in-progress game(s), using Yahoo's pregame odds: "
            + ", ".join(snapshot["stale_games"])
        )
    standings = snapshot["standings"]
    shown = standings[:top]
    if player:
        mine = [
            (i, e) for i, e in enumerate(standings) if player.casefold() in e["player"].casefold()
        ]
        if mine and mine[0][0] >= top:
            shown = shown + [mine[0][1]]
    for e in shown:
        rank = standings.index(e) + 1
        mark = "👉" if player and player.casefold() in e["player"].casefold() else "  "
        lines.append(
            f"  {mark}{rank:>3}. {e['player'][:28]:<28} {e['win_pct']:>6.1%}  "
            f"locked {e['locked_points']:>4.0f}  exp {e['expected_points']:>6.1f}"
        )
    return lines


def main():
    parser = argparse.ArgumentParser(
        description="Track entry win probabilities live as the games play",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--week", "-w", type=int, required=True, help="NFL week number")
    parser.add_argument("--league-id", "-l", type=int, default=11465)
    parser.add_argument("--cookies", default="cookies.txt")
    parser.add_argument("--odds-api-key", help="The Odds API key (or set ODDS_API_KEY)")
    parser.add_argument("--interval", type=float, default=10.0, help="Minutes between refreshes")
    parser.add_argument("--once", action="store_true", help="Take one snapshot and exit")
    parser.add_argument("--player", help="Highlight this entry in the console summary")
    parser.add_argument("--top", type=int, default=8, help="Entries to print per snapshot")
    parser.add_argument("--n-outcomes", type=int, default=6000, help="Outcome draws per snapshot")
    parser.add_argument("--history", help="JSONL history file (default NFL_Week<N>_live.jsonl)")
    args = parser.parse_args()

    if args.interval <= 0:
        print("❌ Error: --interval must be positive")
        return 1
    if not Path(args.cookies).exists():
        print(f"❌ Missing {args.cookies} -- export your Yahoo session cookies first")
        return 1

    history = args.history or f"NFL_Week{args.week}_live.jsonl"
    print(f"📈 Live tracker: week {args.week}, league {args.league_id} -> {history}")

    while True:
        try:
            snapshot = take_snapshot(
                week=args.week,
                league_id=args.league_id,
                cookies_file=args.cookies,
                odds_api_key=args.odds_api_key,
                n_outcomes=args.n_outcomes,
            )
        except Exception as e:
            # One bad tick (network blip, Yahoo hiccup) shouldn't kill a Sunday-long run.
            if args.once:
                print(f"❌ Snapshot failed: {e}")
                return 1
            print(f"⚠️  Snapshot failed ({e}); retrying in {args.interval:g} min")
        else:
            append_snapshot(history, snapshot)
            print("\n".join(format_snapshot(snapshot, args.player, args.top)))
            if snapshot["all_final"]:
                print("🏁 All games final.")
                return 0
        if args.once:
            return 0
        time.sleep(args.interval * 60)


if __name__ == "__main__":
    sys.exit(main())
