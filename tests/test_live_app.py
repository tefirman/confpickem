"""Smoke test for the Streamlit viewer using Streamlit's headless AppTest."""

import pytest

pytest.importorskip("streamlit")
pytest.importorskip("plotly")

from pathlib import Path  # noqa: E402

from streamlit.testing.v1 import AppTest  # noqa: E402

from src.confpickem import live_tracker  # noqa: E402

APP = str(Path(__file__).parent.parent / "src" / "confpickem" / "live_app.py")


def _snap(ts, a, b, all_final=False):
    return {
        "timestamp": ts,
        "week": 10,
        "games": [
            {"home": "KC", "away": "Den", "home_win_prob": 0.6, "status": "live", "stale": False}
        ],
        "all_final": all_final,
        "stale_games": ["Den @ KC"],
        "standings": [
            {
                "player": "Educated Guesses",
                "win_pct": a,
                "expected_points": 5.0,
                "locked_points": 1.0,
            },
            {"player": "Bob", "win_pct": b, "expected_points": 4.0, "locked_points": 0.0},
        ],
    }


def test_app_renders_chart_and_stale_warning(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for s in (
        _snap("2025-11-09T13:00:00", 0.4, 0.6),
        _snap("2025-11-09T13:15:00", 0.7, 0.3, all_final=True),
    ):
        live_tracker.append_snapshot("NFL_Week10_live.jsonl", s)
    at = AppTest.from_file(APP, default_timeout=30)
    at.run()
    assert not at.exception
    assert any("Den @ KC" in w.value for w in at.warning)
    assert any("2 snapshots" in c.value for c in at.caption)


def test_app_empty_history_prompts_refresh(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    at = AppTest.from_file(APP, default_timeout=30)
    at.run()
    at.sidebar.toggle[0].set_value(False).run()  # auto-refresh off: no network in tests
    assert not at.exception
    assert any("No snapshots yet" in i.value for i in at.info)


def test_app_renders_history_with_duplicate_timestamps(tmp_path, monkeypatch):
    """Overlapping refreshes once wrote several snapshots in the same second, which
    crashed the chart pivot ("Index contains duplicate entries")."""
    monkeypatch.chdir(tmp_path)
    for a in (0.4, 0.5, 0.6):
        live_tracker.append_snapshot(
            "NFL_Week10_live.jsonl", _snap("2025-11-09T13:00:00", a, 1 - a, all_final=True)
        )
    at = AppTest.from_file(APP, default_timeout=30)
    at.run()
    assert not at.exception
    assert any("1 snapshots" in c.value for c in at.caption)
