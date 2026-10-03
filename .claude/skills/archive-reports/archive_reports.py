"""Move finished weeks' NFL_Week* reports from the repo root into PreviousWeeks/.

Usage (from the repo root):
    python .claude/skills/archive-reports/archive_reports.py [--dry-run] [--keep-week N]

Keeps the most recent week's reports in the root (the one still being worked
on) and archives everything older. Weeks are ordered by (season, week), with
the season read from each report's _YYYYMMDD_HHMM timestamp -- so Week 18 of
last season is archived once Week 1 of the new season shows up. Never
overwrites a file already in PreviousWeeks/.
"""

import argparse
import re
import sys
from datetime import datetime, timedelta
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
ARCHIVE = REPO_ROOT / "PreviousWeeks"
REPORT = re.compile(r"NFL_Week(\d+)_.*_(\d{8})_\d{4}(?:_[^.]*)?\.(?:txt|html)$")


def season_week(path):
    m = REPORT.fullmatch(path.name)
    if not m:
        return None
    stamp = datetime.strptime(m.group(2), "%Y%m%d")
    # January games belong to the prior season (same rule as the scraper).
    return ((stamp - timedelta(days=90)).year, int(m.group(1)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--keep-week", type=int,
                        help="Week to keep in the root (default: the most recent one)")
    args = parser.parse_args()

    reports = {p: season_week(p) for p in REPO_ROOT.glob("NFL_Week*_*")}
    reports = {p: sw for p, sw in reports.items() if sw}
    if not reports:
        print("No reports in the repo root.")
        return 0

    latest = max(reports.values())
    keep = (latest[0], args.keep_week) if args.keep_week else latest
    to_move = sorted((p for p, sw in reports.items() if sw != keep), key=lambda p: p.name)

    moved, skipped = {}, []
    ARCHIVE.mkdir(exist_ok=True)
    for p in to_move:
        dest = ARCHIVE / p.name
        if dest.exists():
            skipped.append(p.name)
            continue
        if not args.dry_run:
            p.rename(dest)
        label = f"{reports[p][0]} Week {reports[p][1]}"
        moved[label] = moved.get(label, 0) + 1

    verb = "Would move" if args.dry_run else "Moved"
    kept = sum(1 for sw in reports.values() if sw == keep)
    print(f"Kept {kept} report(s) for {keep[0]} Week {keep[1]} in the repo root.")
    if moved:
        print(f"{verb} {sum(moved.values())} report(s) to PreviousWeeks/:")
        for label in sorted(moved, key=lambda s: tuple(int(x) for x in s.split(" Week "))):
            print(f"  {label}: {moved[label]}")
    else:
        print("Nothing to archive.")
    if skipped:
        print(f"Skipped {len(skipped)} already in PreviousWeeks/ (left in root): "
              + ", ".join(skipped))
    return 0


if __name__ == "__main__":
    sys.exit(main())
