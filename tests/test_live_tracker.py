"""Tests for the live tracker (snapshot building, history file, CLI loop). Offline."""

import json
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd
import pytest

from src.confpickem import live_tracker
from src.confpickem.cli import live

PAST = pd.Timestamp("2025-11-09 13:00", tz="America/New_York")
NOW = pd.Timestamp("2025-11-09 16:00", tz="America/New_York")
LATER = pd.Timestamp("2025-11-09 20:20", tz="America/New_York")


def yahoo_games():
    return pd.DataFrame(
        {
            "favorite": ["KC", "Buf", "Dal"],
            "underdog": ["Den", "Mia", "Phi"],
            "win_prob": [0.7, 0.6, 0.4],
            "home_favorite": [True, False, True],
            "favorite_pick_pct": [80.0, 60.0, 50.0],
            "underdog_pick_pct": [20.0, 40.0, 50.0],
            "favorite_confidence": [10.0, 8.0, 5.0],
            "underdog_confidence": [3.0, 4.0, 5.0],
            "kickoff_time": [PAST, PAST, LATER],
        }
    )


def yahoo_results(first_winner="KC"):
    return [
        {"favorite": "KC", "underdog": "Den", "winner": first_winner},
        {"favorite": "Buf", "underdog": "Mia", "winner": None},
        {"favorite": "Dal", "underdog": "Phi", "winner": None},
    ]


def test_build_games_frame_uses_yahoo_structure_and_matches_results():
    orig = yahoo_games()
    enhanced = orig.copy()
    enhanced["live_odds_source"] = "OddsAPI"
    # In-play the underdog (Mia, Yahoo's home team) became the favorite.
    enhanced.loc[1, ["favorite", "underdog", "win_prob"]] = ["Mia", "Buf", 0.8]

    games = live_tracker.build_games_frame(orig, enhanced, yahoo_results(), week=10)

    assert list(games.home_team) == ["KC", "Mia", "Dal"]
    assert games.vegas_win_prob[1] == pytest.approx(0.8)  # Mia is home and favored
    assert games.vegas_win_prob[0] == pytest.approx(0.7)
    # Crowd stats come from Yahoo's original rows, untouched by the flip.
    assert games.crowd_home_pick_pct[1] == pytest.approx(0.4)
    assert games.actual_outcome[0] is True or games.actual_outcome[0] == True  # noqa: E712
    assert games.actual_outcome[1] is None


def test_build_games_frame_result_matched_even_when_favorite_flipped():
    orig = yahoo_games()
    enhanced = orig.copy()
    enhanced["live_odds_source"] = "OddsAPI"
    enhanced.loc[0, ["favorite", "underdog", "win_prob"]] = ["Den", "KC", 0.6]
    games = live_tracker.build_games_frame(orig, enhanced, yahoo_results("Den"), week=10)
    assert games.actual_outcome[0] == False  # noqa: E712  (Den is away, Den won)
    assert games.vegas_win_prob[0] == pytest.approx(0.4)  # KC home, Den favored at 0.6


def test_game_states_flag_stale_live_game():
    games = live_tracker.build_games_frame(
        yahoo_games(),
        yahoo_games().assign(live_odds_source=["OddsAPI", "Yahoo_Fallback", "Yahoo_Fallback"]),
        yahoo_results(),
        week=10,
    )
    states = live_tracker.game_states(games, now=NOW)
    assert [s["status"] for s in states] == ["final", "live", "upcoming"]
    # Only the underway game relying on Yahoo's pregame line is stale.
    assert [s["stale"] for s in states] == [False, True, False]


def test_history_roundtrip_skips_torn_line_and_filters_week(tmp_path):
    path = tmp_path / "h.jsonl"
    assert live_tracker.read_snapshots(path) == []
    snap = lambda ts, wk, p: {  # noqa: E731
        "timestamp": ts,
        "week": wk,
        "standings": [
            {"player": "A", "win_pct": p, "expected_points": 1.0, "locked_points": 2.0},
            {"player": "B", "win_pct": 1 - p, "expected_points": 1.0, "locked_points": 2.0},
        ],
    }
    live_tracker.append_snapshot(path, snap("2025-11-09T13:00:00", 10, 0.4))
    live_tracker.append_snapshot(path, snap("2025-11-09T13:10:00", 10, 0.5))
    live_tracker.append_snapshot(path, snap("2025-11-02T13:10:00", 9, 0.9))
    with open(path, "a") as f:
        f.write('{"timestamp": "2025-11-09T13:2')  # crash mid-write

    snaps = live_tracker.read_snapshots(path, week=10)
    assert len(snaps) == 2
    df = live_tracker.history_frame(snaps)
    assert list(df.columns) == [
        "timestamp",
        "player",
        "win_pct",
        "expected_points",
        "locked_points",
    ]
    assert len(df) == 4
    assert df[df.player == "A"].win_pct.tolist() == [0.4, 0.5]
    assert live_tracker.history_frame([]).empty


def fake_yahoo(cache_dir_unused=None, **kwargs):
    players = pd.DataFrame(
        {
            "player_name": ["Alice", "Bob"],
            "game_1_pick": ["KC", "Den"],
            "game_1_confidence": [3, 1],
            "game_2_pick": ["Buf", "Mia"],
            "game_2_confidence": [1, 3],
            "game_3_pick": ["Dal", "Phi"],
            "game_3_confidence": [2, 2],
        }
    )
    return SimpleNamespace(games=yahoo_games(), players=players, results=yahoo_results())


def test_take_snapshot_scores_board_with_mocked_network(tmp_path):
    requested = {}

    def fake_odds(games, week=None, odds_api_key=None, markets=None):
        requested["markets"] = markets
        return games.assign(live_odds_source="OddsAPI")

    with patch.object(live_tracker, "YahooPickEm", side_effect=fake_yahoo), patch.object(
        live_tracker, "update_odds_with_live_data", side_effect=fake_odds
    ):
        snap = live_tracker.take_snapshot(
            10, 1, "c.txt", n_outcomes=500, cache_dir=str(tmp_path / "cache"), now=NOW
        )

    assert requested["markets"] == "h2h"  # moneyline only: 1 credit/call
    assert snap["week"] == 10 and not snap["all_final"] and snap["stale_games"] == []
    by_player = {e["player"]: e for e in snap["standings"]}
    assert sum(e["win_pct"] for e in snap["standings"]) == pytest.approx(1.0, abs=1e-3)
    # Alice banked game 1 (KC won, 3 pts); Bob picked Den.
    assert by_player["Alice"]["locked_points"] == 3 and by_player["Bob"]["locked_points"] == 0
    assert by_player["Alice"]["win_pct"] > by_player["Bob"]["win_pct"]
    json.dumps(snap)  # must be serializable for the history file


def test_take_snapshot_errors_when_yahoo_returns_no_games(tmp_path):
    empty = SimpleNamespace(games=pd.DataFrame(), players=pd.DataFrame(), results=[])
    with patch.object(live_tracker, "YahooPickEm", return_value=empty):
        with pytest.raises(RuntimeError, match="cookies"):
            live_tracker.take_snapshot(10, 1, cache_dir=str(tmp_path / "cache"))


def _snapshot(all_final=False):
    return {
        "timestamp": "2025-11-09T16:00:00",
        "week": 10,
        "games": [{"status": "final"}, {"status": "live"}],
        "all_final": all_final,
        "stale_games": ["Mia @ Buf"],
        "standings": [
            {"player": "Alice", "win_pct": 0.6, "expected_points": 50.0, "locked_points": 3.0},
            {"player": "Bob", "win_pct": 0.4, "expected_points": 49.0, "locked_points": 0.0},
        ],
    }


def test_format_snapshot_warns_on_stale_and_marks_player():
    lines = live.format_snapshot(_snapshot(), player="bob")
    assert "1 final, 1 live, 0 upcoming" in lines[0]
    assert "Mia @ Buf" in lines[1]
    assert "👉" in lines[3] and "Bob" in lines[3]


def test_cli_once_writes_history(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "cookies.txt").write_text("")
    with patch.object(live, "take_snapshot", return_value=_snapshot()):
        with patch("sys.argv", ["live.py", "--week", "10", "--once"]):
            assert live.main() == 0
    lines = (tmp_path / "NFL_Week10_live.jsonl").read_text().splitlines()
    assert len(lines) == 1 and json.loads(lines[0])["week"] == 10


def test_cli_loop_retries_after_failure_and_stops_when_final(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "cookies.txt").write_text("")
    results = [RuntimeError("blip"), _snapshot(), _snapshot(all_final=True)]
    with patch.object(live, "take_snapshot", side_effect=results), patch.object(
        live.time, "sleep"
    ) as sleep:
        with patch("sys.argv", ["live.py", "--week", "10", "--interval", "5"]):
            assert live.main() == 0
    assert sleep.call_count == 2 and sleep.call_args[0][0] == 300
    assert len((tmp_path / "NFL_Week10_live.jsonl").read_text().splitlines()) == 2


def test_cli_once_failure_exits_nonzero(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "cookies.txt").write_text("")
    with patch.object(live, "take_snapshot", side_effect=RuntimeError("boom")):
        with patch("sys.argv", ["live.py", "--week", "10", "--once"]):
            assert live.main() == 1


def _snap(ts, finals, probs):
    games = [
        {"home": "KC", "away": "Den", "status": "final" if "KC" in finals else "live"},
        {"home": "Buf", "away": "Mia", "status": "final" if "Buf" in finals else "live"},
    ]
    return {
        "timestamp": ts,
        "week": 10,
        "games": games,
        "all_final": False,
        "stale_games": [],
        "standings": [
            {"player": p, "win_pct": v, "expected_points": 1.0, "locked_points": 0.0}
            for p, v in probs.items()
        ],
    }


def test_win_pct_series_and_final_events():
    snaps = [
        _snap("2025-11-09T13:00:00", {"KC"}, {"A": 0.4, "B": 0.6}),
        _snap("2025-11-09T13:15:00", {"KC"}, {"A": 0.5, "B": 0.5}),
        _snap("2025-11-09T13:30:00", {"KC", "Buf"}, {"A": 0.9, "B": 0.1}),
    ]
    wide = live_tracker.win_pct_series(snaps)
    assert list(wide.columns) == ["A", "B"] and wide.A.tolist() == [0.4, 0.5, 0.9]
    # KC was already final when tracking began, so only Buf is an observed event.
    events = live_tracker.final_events(snaps)
    assert [e["game"] for e in events] == ["Mia @ Buf"]
    assert events[0]["timestamp"] == pd.Timestamp("2025-11-09T13:30:00")
    assert live_tracker.win_pct_series([]).empty and live_tracker.final_events([]) == []


def _odds_api_game(markets):
    return {
        "home_team": "Buffalo Bills",
        "away_team": "Miami Dolphins",
        "commence_time": "2025-11-09T18:00:00Z",
        "bookmakers": [{"key": "fanduel", "markets": markets}],
    }


def test_parse_odds_game_h2h_only_uses_moneyline():
    from src.confpickem.live_odds_scraper import LiveOddsScraper

    h2h = {
        "key": "h2h",
        "outcomes": [
            {"name": "Buffalo Bills", "price": -200},
            {"name": "Miami Dolphins", "price": 170},
        ],
    }
    parsed = LiveOddsScraper("k", markets="h2h")._parse_odds_api_game(_odds_api_game([h2h]))
    assert parsed["home_win_prob"] > 0.6 and parsed["home_spread"] == 0.0


def test_parse_odds_game_without_moneyline_or_spread_is_dropped():
    from src.confpickem.live_odds_scraper import LiveOddsScraper

    totals = {"key": "totals", "outcomes": [{"name": "Over", "point": 44.5}]}
    assert LiveOddsScraper("k")._parse_odds_api_game(_odds_api_game([totals])) is None


def test_read_snapshots_collapses_duplicate_timestamps_keeping_last(tmp_path):
    path = tmp_path / "h.jsonl"
    live_tracker.append_snapshot(path, _snap("2025-11-09T13:00:00", set(), {"A": 0.4, "B": 0.6}))
    live_tracker.append_snapshot(path, _snap("2025-11-09T13:00:00", set(), {"A": 0.5, "B": 0.5}))
    live_tracker.append_snapshot(path, _snap("2025-11-09T13:10:00", set(), {"A": 0.7, "B": 0.3}))
    snaps = live_tracker.read_snapshots(path)
    assert len(snaps) == 2
    assert live_tracker.win_pct_series(snaps).A.tolist() == [0.5, 0.7]


def test_win_pct_series_tolerates_repeated_entrant_name():
    snap = _snap("2025-11-09T13:00:00", set(), {"A": 0.4, "B": 0.6})
    snap["standings"].append(dict(snap["standings"][0], win_pct=0.1))  # second "A"
    wide = live_tracker.win_pct_series([snap])
    assert list(wide.columns) == ["A", "B"] and wide.A.tolist() == [0.1]
