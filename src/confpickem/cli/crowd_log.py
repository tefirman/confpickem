#!/usr/bin/env python
"""
Log crowd pick % vs. line vs. result, and check whether the crowd carries any signal.

Snapshots the league-wide Yahoo pick distribution for all four pool types (no
spread/confidence, spread only, confidence only, spread + confidence), settles final
scores from ESPN, and reports whether crowd % predicts straight-up winners or covers.

Usage:
  crowd_log.py --snapshot --week 9            # snapshot all four pools for week 9
  crowd_log.py --snapshot --week 1 2 3 4 5    # several weeks (also backfills past ones)
  crowd_log.py --snapshot --week 9 --pools s sc
  crowd_log.py --settle                       # fetch ESPN finals for any unsettled games
  crowd_log.py --report                       # analyze cached history + the log
  crowd_log.py --report --no-history          # only the snapshot log

Run --snapshot shortly before kickoff to capture a true pre-game line and crowd; run
--settle after games finish. The log is JSONL ("snapshot" and "result" records).
Backfilling a past week works but its crowd % is post-game, and the report says so.
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

import pandas as pd

from src.confpickem.crowd_log import (
    POOLS,
    append_records,
    crowd_signal_report,
    current_season,
    history_from_cache,
    load_log,
    make_session,
    settle_log,
    snapshot_week,
)

DEFAULT_LOG = "crowd_lines.jsonl"


def main():
    parser = argparse.ArgumentParser(
        description="Log crowd pick % vs. lines/results and test for signal",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--snapshot", action="store_true", help="Scrape and append a snapshot")
    parser.add_argument("--settle", action="store_true", help="Add ESPN final scores to the log")
    parser.add_argument("--report", action="store_true", help="Print the signal report")
    parser.add_argument(
        "--week", "-w", type=int, nargs="+", help="NFL week(s) (required with --snapshot)"
    )
    parser.add_argument("--season", type=int, default=None, help="Season (default: current)")
    parser.add_argument(
        "--pools", nargs="+", choices=POOLS, default=list(POOLS), help="Pool types to snapshot"
    )
    parser.add_argument("--cookies", default="cookies.txt")
    parser.add_argument("--log", default=DEFAULT_LOG, help=f"JSONL log (default: {DEFAULT_LOG})")
    parser.add_argument(
        "--cache-root", default=".", help="Directory holding PickEmCache<year>/ folders"
    )
    parser.add_argument("--no-history", action="store_true", help="Report on the log only")
    args = parser.parse_args()

    if not (args.snapshot or args.settle or args.report):
        parser.error("nothing to do: pass --snapshot, --settle and/or --report")
    season = args.season or current_season()

    if args.snapshot:
        if not args.week:
            parser.error("--snapshot requires --week")
        session = make_session(args.cookies)
        for week in args.week:
            records = snapshot_week(session, week, season, pools=args.pools)
            n = append_records(args.log, records)
            print(f"📝 Logged {n} game-pool rows for {season} week {week} to {args.log}")

    if args.snapshot or args.settle:
        # --week limits settling to those weeks; a bare --settle covers every logged week
        n = settle_log(args.log, season, weeks=args.week)
        print(f"🏁 Settled {n} games from ESPN")

    if args.report:
        frames = []
        if not args.no_history:
            frames.append(history_from_cache(args.cache_root))
        logged = load_log(args.log)
        if not logged.empty:
            frames.append(logged.assign(_src=0))
        if not frames:
            print("❌ Error: no cached history or snapshot log to analyze")
            return 1
        df = pd.concat(frames, ignore_index=True)
        # a game both cached and logged: prefer the logged row (it has scores)
        df["_src"] = df["_src"].fillna(1) if "_src" in df else 1
        key = ["season", "week", "pool", "favorite", "underdog"]
        df = df.sort_values("_src").drop_duplicates(key, keep="first")
        print("\n".join(crowd_signal_report(df)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
