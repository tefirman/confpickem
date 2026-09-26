"""Print the state of a pick'em week so the /picks skill can choose a mode.

Usage (from the repo root):
    python .claude/skills/picks/week_status.py [WEEK] [--league-id ID]

With no WEEK, uses the highest week among NFL_Week*_ reports in the repo root,
rolling forward one week once every game of that week has a final result.

Output is a small JSON object on stdout; diagnostics go to stderr.
"""

import argparse
import contextlib
import glob
import io
import json
import re
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))

from confpickem.yahoo_pickem_scraper import YahooPickEm  # noqa: E402

ET = "America/New_York"


def latest_report_week():
    weeks = [
        int(m.group(1))
        for f in glob.glob(str(REPO_ROOT / "NFL_Week*_*"))
        if (m := re.search(r"NFL_Week(\d+)_", Path(f).name))
    ]
    return max(weeks) if weeks else 1


def to_et(ts):
    # The scraper rewrites Yahoo's "EDT" to "EST" before parsing, so an aware
    # timestamp is an hour off during DST. Its wall-clock time is correct ET,
    # so drop whatever zone it carries and re-localize to America/New_York.
    ts = pd.Timestamp(ts)
    if ts.tzinfo is not None:
        ts = ts.tz_localize(None)
    return ts.tz_localize(ET)


def slate_lock(kickoffs):
    """When Yahoo locks every remaining entry: the first Sunday kickoff at or
    after 1:00 PM ET. Earlier Sunday games (international, ~9:30 AM ET) lock
    individually like Thursday's, so the slate stays open until the main
    window. Falls back to the first Sunday kickoff if there's no 1 PM game."""
    sunday = sorted(k for k in kickoffs if k.dayofweek == 6)
    main_window = [k for k in sunday if k.hour >= 13]
    if main_window:
        return main_window[0]
    return sunday[0] if sunday else None


def load(week, league_id):
    # The scraper prints progress chatter; keep stdout clean for the JSON.
    with contextlib.redirect_stdout(io.StringIO()):
        return YahooPickEm(week, league_id, str(REPO_ROOT / "cookies.txt"))


def status(week, league_id):
    yahoo = load(week, league_id)
    games = yahoo.games
    if games is None or len(games) == 0:
        return {"week": week, "error": "no games scraped -- cookies.txt is likely expired"}

    now = pd.Timestamp.now(tz=ET)
    rows = []
    for _, g in games.iterrows():
        kick = to_et(g["kickoff_time"])
        home_fav = bool(g["home_favorite"])
        home = g["favorite"] if home_fav else g["underdog"]
        away = g["underdog"] if home_fav else g["favorite"]
        rows.append({"away": away, "home": home, "kickoff": kick, "started": kick <= now})
    rows.sort(key=lambda r: r["kickoff"])

    lock = slate_lock([r["kickoff"] for r in rows])
    early = [r for r in rows if r["kickoff"].dayofweek == 6 and lock and r["kickoff"] < lock]
    started = sum(r["started"] for r in rows)
    player_count = 0 if yahoo.players is None else len(yahoo.players)
    finished = sum(1 for r in (yahoo.results or []) if r.get("winner"))

    if lock is not None and now >= lock:
        mode = "locked"
    elif started == 0:
        mode = "beginning"
    else:
        mode = "midweek"

    opener = rows[0]
    return {
        "week": week,
        "now_et": now.strftime("%a %b %d %I:%M %p"),
        "num_games": len(rows),
        "max_confidence": len(rows),
        "games_started": started,
        "games_finished": finished,
        "week_over": finished == len(rows),
        "players_scraped": player_count,
        "recommended_mode": mode,
        "opener": {
            "away": opener["away"],
            "home": opener["home"],
            "kickoff_et": opener["kickoff"].strftime("%a %b %d %I:%M %p"),
            "started": opener["started"],
        },
        "slate_lock_et": lock.strftime("%a %b %d %I:%M %p") if lock is not None else None,
        "early_sunday_games": [
            {
                "away": r["away"],
                "home": r["home"],
                "kickoff_et": r["kickoff"].strftime("%a %b %d %I:%M %p"),
                "started": r["started"],
            }
            for r in early
        ],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("week", type=int, nargs="?")
    parser.add_argument("--league-id", type=int, default=11465)
    args = parser.parse_args()

    if args.week is not None:
        result = status(args.week, args.league_id)
    else:
        week = latest_report_week()
        result = status(week, args.league_id)
        # Once every game has a result, the user is asking about next week.
        if result.get("week_over"):
            nxt = status(week + 1, args.league_id)
            if "error" not in nxt:
                result = nxt

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
