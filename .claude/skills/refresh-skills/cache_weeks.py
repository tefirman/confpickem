"""Save every finished week of a season into PickEmCache<season>/.

Usage (from the repo root):
    python .claude/skills/refresh-skills/cache_weeks.py [--season YEAR] [--force-week N]

Walks weeks 1, 2, ... and, for each week not already cached, fetches the
Yahoo pages straight into PickEmCache<season>/ -- the layout
analyze_player_skills.py reads. A week is kept only if every game has a
result; the first unfinished week is discarded and the walk stops there, so
the cache only ever holds final weeks. --force-week re-fetches one week (e.g.
if a stat correction changed a result).

Prints a JSON summary on stdout; scraper chatter is suppressed.
"""

import argparse
import contextlib
import io
import json
import sys
from datetime import date, timedelta
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))

from confpickem.yahoo_pickem_scraper import YahooPickEm  # noqa: E402

MAX_WEEKS = 18
# Pages analyze_player_skills.py reads. The scraper also caches the season-wide
# weekly_performance page, which isn't per-week data, so it's dropped.
KEEP = ("confidence_picks", "pick_distribution")
DROP = ("weekly_performance",)


def default_season():
    # Same rule as the scraper: January games belong to the prior season.
    return (date.today() - timedelta(days=90)).year


def week_files(cache_dir, week, page_types):
    for page in page_types:
        yield cache_dir / f"{page}_week{week}.html"
        yield cache_dir / f"{page}_week{week}_meta.json"


def is_cached(cache_dir, week):
    return all(p.exists() for p in week_files(cache_dir, week, KEEP))


def remove(cache_dir, week, page_types):
    for p in week_files(cache_dir, week, page_types):
        p.unlink(missing_ok=True)


def signed_out(cache_dir, week):
    """Yahoo serves its sign-in page (HTTP 200) for pages that need a login once
    cookies expire. The confidence-picks page is public, so it keeps working and
    hides the problem -- check the pick-distribution page, which isn't."""
    page = cache_dir / f"pick_distribution_week{week}.html"
    return page.exists() and "<title>Sign in | Yahoo</title>" in page.read_text(errors="ignore")


def fetch(cache_dir, week, league_id):
    """Fetch a week into cache_dir; return (num_games, num_final, num_players).

    Games are counted from the confidence-picks page's results: once a week is
    over, Yahoo's pick-distribution page no longer parses into games.
    """
    with contextlib.redirect_stdout(io.StringIO()):
        yahoo = YahooPickEm(week, league_id, str(REPO_ROOT / "cookies.txt"),
                            cache_dir=str(cache_dir))
    results = getattr(yahoo, "results", None) or []
    games = len(results)
    final = sum(1 for r in results if r.get("winner"))
    players = 0 if yahoo.players is None else len(yahoo.players)
    return games, final, players


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", type=int, default=default_season())
    parser.add_argument("--league-id", type=int, default=11465)
    parser.add_argument("--force-week", type=int, action="append", default=[])
    args = parser.parse_args()

    cache_dir = REPO_ROOT / f"PickEmCache{args.season}"
    cache_dir.mkdir(exist_ok=True)
    for week in args.force_week:
        remove(cache_dir, week, KEEP + DROP)

    added, already = [], []
    stopped = None
    for week in range(1, MAX_WEEKS + 1):
        if is_cached(cache_dir, week):
            already.append(week)
            continue
        # Clear any partial leftovers so the scraper fetches fresh pages.
        remove(cache_dir, week, KEEP + DROP)
        games, final, players = fetch(cache_dir, week, args.league_id)
        remove(cache_dir, week, DROP)
        if signed_out(cache_dir, week):
            remove(cache_dir, week, KEEP)
            print(json.dumps({"error": "Yahoo returned its sign-in page -- cookies.txt has "
                              "expired", "season": args.season, "week": week}))
            return 1
        if games == 0 or players == 0:
            remove(cache_dir, week, KEEP)
            if week == 1 or not (added or already):
                print(json.dumps({"error": "no games/players scraped -- cookies.txt is "
                                  "likely expired", "season": args.season, "week": week}))
                return 1
            stopped = {"week": week, "reason": "no data yet"}
            break
        if final < games:
            remove(cache_dir, week, KEEP)
            stopped = {"week": week, "reason": f"not final ({final}/{games} games decided)"}
            break
        added.append(week)

    cached = already + added
    print(json.dumps({
        "season": args.season,
        "cache_dir": cache_dir.name,
        "weeks_added": added,
        "weeks_already_cached": already,
        "latest_final_week": max(cached) if cached else None,
        "stopped_at": stopped,
    }, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
