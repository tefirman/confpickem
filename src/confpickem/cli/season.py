#!/usr/bin/env python
"""
Season-long outlook: P(1st-4th) and expected dollars for every player.

Takes the current Yahoo standings, simulates every remaining week for every active
player (slates sampled from the weeks in PickEmCache<year>/, picks from each
player's skill model) and reports each player's chance at each season prize plus
expected dollars -- weekly prizes still to be won + season-prize equity.

Usage:
  season.py --week 8                      # weeks 1-7 final, simulate 8-18
  season.py --week 8 --player "Educated Guesses" --top 25
  season.py --week 8 --csv season_outlook.csv
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.confpickem.yahoo_pickem_scraper import YahooPickEm
from src.confpickem.season_sim import (
    SeasonSimulator,
    SEASON_PRIZES,
    WEEKLY_PRIZE,
    load_skills,
    load_week_templates,
)


def format_outlook(outlook, prizes, top=None, highlight=None):
    """Fixed-width table of a ``SeasonSimulator.run()`` result."""
    places = [f"P({_ordinal(j + 1)})" for j in range(len(prizes))]
    head = (
        f"{'#':>3} {'Player':<28} {'Pts':>5} {'Rk':>3} {'ExpPts':>7} "
        + " ".join(f"{p:>7}" for p in places)
        + f" {'Top-4':>7} {'Wkly$':>7} {'Szn$':>8} {'Total$':>8}"
    )
    lines = [head, "-" * len(head)]
    shown = outlook if top is None else outlook.head(top)
    for i, r in enumerate(shown.itertuples(index=False), 1):
        probs = " ".join(f"{getattr(r, f'p_place_{j + 1}'):>7.1%}" for j in range(len(prizes)))
        mark = "*" if highlight and r.player.casefold() == highlight.casefold() else " "
        flag = "" if r.active else "  (inactive)"
        lines.append(
            f"{i:>3}{mark}{r.player[:28]:<28} {r.points:>5.0f} {r.current_rank:>3} "
            f"{r.exp_final_points:>7.0f} {probs} {r.p_top4:>7.1%} "
            f"{r.exp_weekly_dollars:>7.0f} {r.exp_season_dollars:>8.0f} "
            f"{r.exp_dollars:>8.0f}{flag}"
        )
    return lines


def _ordinal(n):
    return f"{n}{'th' if 10 <= n % 100 <= 20 else {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')}"


def main():
    parser = argparse.ArgumentParser(
        description="Season-long P(1st-4th) and expected dollars from the current standings",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--week",
        "-w",
        type=int,
        required=True,
        help="Current NFL week: weeks before it are treated as final and "
        "it (and later weeks) are simulated in full",
    )
    parser.add_argument(
        "--completed-weeks",
        type=int,
        default=None,
        help="Override: number of final weeks (default: --week minus 1)",
    )
    parser.add_argument("--league-id", "-l", type=int, default=11465)
    parser.add_argument("--cookies", default="cookies.txt")
    parser.add_argument(
        "--skills",
        default="current_player_skills.json",
        help="Player skills JSON (default: current_player_skills.json)",
    )
    parser.add_argument(
        "--cache-root",
        default=".",
        help="Directory holding PickEmCache<year>/ folders to sample slates from",
    )
    parser.add_argument("--num-sims", "-n", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=51)
    parser.add_argument("--player", help="Highlight (and summarise) this entry")
    parser.add_argument("--top", type=int, default=None, help="Only print the top N by expected $")
    parser.add_argument("--csv", help="Also write the full table to this CSV")
    parser.add_argument("--weekly-prize", type=float, default=WEEKLY_PRIZE)
    parser.add_argument(
        "--season-prizes",
        default=",".join(str(int(p)) for p in SEASON_PRIZES),
        help="Comma-separated season prizes, 1st place first",
    )
    args = parser.parse_args()

    try:
        prizes = tuple(float(x) for x in args.season_prizes.split(","))
    except ValueError:
        print(f"❌ Error: could not parse --season-prizes {args.season_prizes!r}")
        return 1

    templates = load_week_templates(args.cache_root)
    if not templates:
        print(
            f"❌ Error: no PickEmCache<year>/ history under {args.cache_root!r} to sample "
            "future slates from"
        )
        return 1

    yahoo = YahooPickEm(week=args.week, league_id=args.league_id, cookies_file=args.cookies)
    standings = getattr(yahoo, "standings", None)
    if standings is None or len(standings) == 0:
        print("❌ Error: no standings scraped -- is cookies.txt current?")
        return 1

    skills = load_skills(args.skills)
    completed = args.completed_weeks if args.completed_weeks is not None else args.week - 1
    sim = SeasonSimulator.from_standings(
        standings,
        skills,
        templates,
        completed_weeks=completed,
        num_sims=args.num_sims,
        seed=args.seed,
        weekly_prize=args.weekly_prize,
        season_prizes=prizes,
    )

    print(
        f"📅 Weeks 1-{sim.completed_weeks} final; simulating weeks "
        f"{sim.completed_weeks + 1}-{sim.total_weeks} x {sim.num_sims:,} seasons"
    )
    print(
        f"👥 {len(sim.names)} entries ({int(sim.active.sum())} active), "
        f"{sim.skills_matched} with skill history; {len(templates)} historical slates"
    )
    if sim.skills_matched < len(sim.names):
        missing = len(sim.names) - sim.skills_matched
        print(f"   ({missing} without history get the league-mean skills)")

    outlook = sim.run()
    print()
    for line in format_outlook(outlook, prizes, args.top, args.player):
        print(line)
    print(
        f"\nExpected $ = weekly prizes still to win (${args.weekly_prize:.0f}/week) + season "
        f"prizes ({'/'.join(f'${p:,.0f}' for p in prizes)}). Ties split evenly."
    )

    if args.player:
        mine = outlook[outlook.player.str.casefold() == args.player.casefold()]
        if mine.empty:
            print(f"\n⚠️  {args.player!r} not found in standings")
        else:
            r = mine.iloc[0]
            print(
                f"\n⭐ {r.player}: rank {r.current_rank} on {r.points:.0f} pts -> "
                f"expected ${r.exp_dollars:,.0f} (weekly ${r.exp_weekly_dollars:,.0f} + "
                f"season ${r.exp_season_dollars:,.0f}); P(top 4) {r.p_top4:.1%}"
            )

    if args.csv:
        outlook.to_csv(args.csv, index=False)
        print(f"💾 Wrote {args.csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
