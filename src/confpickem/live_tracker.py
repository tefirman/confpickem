"""Live win-probability tracker for a locked confidence pool.

Once Yahoo locks every entry (first Sunday kickoff) the only thing still moving
is the games. Each *snapshot* re-scrapes Yahoo for results, re-pulls live odds
(in-play lines for games underway), re-scores the board with
``ConfidencePickEmSimulator.standings_analytic`` and appends one JSON line to a
history file. The history file is the source of truth for the over-time chart,
so a restart (or a viewer opened later) never loses what already happened.
"""

import json
import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from .confidence_pickem_sim import ConfidencePickEmSimulator
from .live_odds_scraper import update_odds_with_live_data
from .yahoo_pickem_scraper import KICKOFF_TZ, YahooPickEm

logger = logging.getLogger(__name__)

YAHOO_FALLBACK = "Yahoo_Fallback"


def _same_game(result: Dict, home: str, away: str) -> bool:
    return {result["favorite"], result["underdog"]} == {home, away}


def build_games_frame(
    yahoo_games: pd.DataFrame,
    enhanced_games: pd.DataFrame,
    results: List[Dict],
    week: int,
) -> pd.DataFrame:
    """Simulator-format games frame from Yahoo's structure + live-odds probabilities.

    Home/away and the crowd columns come from Yahoo's *original* rows; only the
    win probability is taken from ``enhanced_games`` (matched by team, since an
    in-play line can flip which side is the betting favorite). Finished games
    are matched to ``results`` by team pair for the same reason.
    """
    rows = []
    for (_, orig), (_, live) in zip(yahoo_games.iterrows(), enhanced_games.iterrows()):
        if orig["home_favorite"]:
            home, away = orig["favorite"], orig["underdog"]
            crowd_home_pct = orig["favorite_pick_pct"] / 100.0
            home_conf, away_conf = orig["favorite_confidence"], orig["underdog_confidence"]
        else:
            home, away = orig["underdog"], orig["favorite"]
            crowd_home_pct = orig["underdog_pick_pct"] / 100.0
            home_conf, away_conf = orig["underdog_confidence"], orig["favorite_confidence"]

        home_prob = live["win_prob"] if live["favorite"] == home else 1.0 - live["win_prob"]

        actual = None
        for r in results:
            if r["winner"] and _same_game(r, home, away):
                actual = r["winner"] == home
                break

        rows.append(
            {
                "home_team": home,
                "away_team": away,
                "vegas_win_prob": float(home_prob),
                "crowd_home_pick_pct": crowd_home_pct,
                "crowd_home_confidence": home_conf,
                "crowd_away_confidence": away_conf,
                "week": week,
                "kickoff_time": orig["kickoff_time"],
                "actual_outcome": actual,
                "live_odds_source": live.get("live_odds_source", YAHOO_FALLBACK),
            }
        )
    return pd.DataFrame(rows)


def game_states(games: pd.DataFrame, now: Optional[pd.Timestamp] = None) -> List[Dict]:
    """Per-game status for the snapshot: final / live / upcoming, plus ``stale``
    for a game that is underway but only has Yahoo's pregame line (so its
    probability isn't tracking the game)."""
    now = now if now is not None else pd.Timestamp.now(tz=KICKOFF_TZ)
    states = []
    for g in games.itertuples(index=False):
        kickoff = pd.Timestamp(g.kickoff_time)
        if kickoff.tzinfo is None:
            kickoff = kickoff.tz_localize(KICKOFF_TZ)
        if g.actual_outcome is not None and not pd.isna(g.actual_outcome):
            status = "final"
        elif kickoff <= now:
            status = "live"
        else:
            status = "upcoming"
        states.append(
            {
                "home": g.home_team,
                "away": g.away_team,
                "home_win_prob": round(float(g.vegas_win_prob), 4),
                "status": status,
                "stale": status == "live" and g.live_odds_source == YAHOO_FALLBACK,
            }
        )
    return states


def take_snapshot(
    week: int,
    league_id: int,
    cookies_file: str = "cookies.txt",
    odds_api_key: Optional[str] = None,
    n_outcomes: int = 6000,
    cache_dir: str = ".cache",
    now: Optional[pd.Timestamp] = None,
) -> Dict:
    """Scrape, update odds, score the board; return one JSON-serializable snapshot.

    Intended for the locked state (first Sunday kickoff onward), like
    ``--mode locked``: entrants missing a pick score 0 on that game.
    """
    cache = Path(cache_dir)
    if cache.exists():
        shutil.rmtree(cache)

    yahoo = YahooPickEm(
        week=week, league_id=league_id, cookies_file=cookies_file, cache_dir=cache_dir
    )
    if len(yahoo.games) == 0:
        raise RuntimeError(
            "Yahoo returned no games -- cookies.txt has probably expired "
            "(or the league ID / week is wrong)"
        )

    # Only moneylines are needed for win probabilities; h2h alone is a third of the
    # Odds API credits of the default h2h,spreads,totals.
    enhanced = update_odds_with_live_data(
        yahoo.games, week=week, odds_api_key=odds_api_key, markets="h2h"
    )
    games = build_games_frame(yahoo.games, enhanced, yahoo.results, week)

    simulator = ConfidencePickEmSimulator()
    simulator.add_games_from_dataframe(games.drop(columns=["live_odds_source"]))
    standings, _ = simulator.standings_analytic(
        yahoo.players, n_outcomes=n_outcomes, fill_missing=True
    )

    states = game_states(games, now=now)
    return {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "week": week,
        "games": states,
        "all_final": all(s["status"] == "final" for s in states),
        "stale_games": [f"{s['away']} @ {s['home']}" for s in states if s["stale"]],
        "standings": [
            {
                "player": r.player,
                "win_pct": round(float(r.win_pct), 5),
                "expected_points": round(float(r.expected_points), 2),
                "locked_points": float(r.locked_points),
            }
            for r in standings.itertuples(index=False)
        ],
    }


def append_snapshot(path, snapshot: Dict) -> None:
    with open(path, "a") as f:
        f.write(json.dumps(snapshot) + "\n")


def read_snapshots(path, week: Optional[int] = None) -> List[Dict]:
    """All snapshots in ``path`` (optionally one week), oldest first. A missing
    file is an empty history; a torn final line (crash mid-write) is skipped. Two
    snapshots with the same timestamp (overlapping refreshes) collapse to the last."""
    p = Path(path)
    if not p.exists():
        return []
    by_timestamp: Dict[str, Dict] = {}
    for line in p.read_text().splitlines():
        try:
            snap = json.loads(line)
        except json.JSONDecodeError:
            continue
        if week is None or snap.get("week") == week:
            by_timestamp[snap["timestamp"]] = snap
    return list(by_timestamp.values())


def history_frame(snapshots: List[Dict]) -> pd.DataFrame:
    """Long-form ``timestamp, player, win_pct, expected_points, locked_points``
    -- one row per entrant per snapshot, ready to pivot and plot."""
    rows = [
        {"timestamp": pd.Timestamp(s["timestamp"]), **entry}
        for s in snapshots
        for entry in s["standings"]
    ]
    return pd.DataFrame(
        rows, columns=["timestamp", "player", "win_pct", "expected_points", "locked_points"]
    )


def win_pct_series(snapshots: List[Dict]) -> pd.DataFrame:
    """Wide win % history: one row per snapshot, one column per entrant."""
    df = history_frame(snapshots)
    if df.empty:
        return pd.DataFrame()
    # A repeated entrant name within one snapshot would make the pivot ambiguous.
    df = df.drop_duplicates(["timestamp", "player"], keep="last")
    return df.pivot(index="timestamp", columns="player", values="win_pct").sort_index()


def final_events(snapshots: List[Dict]) -> List[Dict]:
    """Games that went final between snapshots, as ``{timestamp, game}`` -- the
    first snapshot in which each game shows as final, for chart markers. Games
    already final in the first snapshot are skipped (nothing observed changing)."""
    events: List[Dict] = []
    seen_final = None
    for snap in snapshots:
        final = {f"{g['away']} @ {g['home']}" for g in snap["games"] if g["status"] == "final"}
        if seen_final is not None:
            events.extend(
                {"timestamp": pd.Timestamp(snap["timestamp"]), "game": g}
                for g in sorted(final - seen_final)
            )
        seen_final = final
    return events
