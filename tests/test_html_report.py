#!/usr/bin/env python
"""Tests for src.confpickem.html_report"""

import pandas as pd

from src.confpickem.html_report import generate_html_report, generate_locked_board_html_report


def _importance_df():
    return pd.DataFrame(
        [
            {
                "game": "KC@NO",
                "points_bid": 15,
                "pick": "KC",
                "win_probability": 0.68,
                "loss_probability": 0.12,
                "total_impact": 0.087,
            },
            {
                "game": "SF@ARI",
                "points_bid": 16,
                "pick": "SF",
                "win_probability": 0.71,
                "loss_probability": 0.15,
                "total_impact": -0.054,
            },
        ]
    ).sort_values("total_impact", ascending=False, key=abs)


def _all_win_probs(you_name="You"):
    return [
        {
            "player": "OneNDone",
            "win_pct": 0.41,
            "total_expected": 118.4,
            "current_pts": 68,
            "is_you": False,
        },
        {
            "player": you_name,
            "win_pct": 0.34,
            "total_expected": 112.6,
            "current_pts": 62,
            "is_you": True,
        },
    ]


def _base_kwargs(**overrides):
    kwargs = dict(
        week=4,
        league_id=11465,
        player_name="You",
        mode="beginning",
        algo_label="Analytical P(win)",
        sorted_picks=[("SF", 16), ("KC", 15)],
        remaining_games=[{"home": "SF", "away": "ARI"}, {"home": "NO", "away": "KC"}],
        opt_win=0.338,
        rand_win=0.181,
        importance_sorted=None,
        all_win_probs=_all_win_probs(),
        current_standings={},
        your_rank=None,
        your_points=None,
        num_remaining_games=2,
        total_games=2,
        summary_stats=None,
    )
    kwargs.update(overrides)
    return kwargs


def test_generate_html_report_returns_string():
    html = generate_html_report(**_base_kwargs())
    assert isinstance(html, str)
    assert "<title>Week 4 Optimization Report</title>" in html


def test_beginning_mode_omits_standings_columns():
    html = generate_html_report(**_base_kwargs())
    assert "HAS_STANDINGS = false" in html


def test_midweek_mode_includes_standings_columns():
    html = generate_html_report(
        **_base_kwargs(
            mode="midweek",
            current_standings={"You": 62, "OneNDone": 68},
            your_rank=2,
            your_points=62,
            importance_sorted=_importance_df(),
        )
    )
    assert "HAS_STANDINGS = true" in html
    assert "KC@NO".replace("@", " @ ") in html or "KC @ NO" in html


def test_picks_and_paste_format_present():
    html = generate_html_report(**_base_kwargs())
    assert "SF 16, KC 15" in html


def test_player_name_is_escaped():
    html = generate_html_report(
        **_base_kwargs(
            player_name="<script>alert(1)</script>",
            all_win_probs=_all_win_probs(you_name="<script>alert(1)</script>"),
        )
    )
    assert "<script>alert(1)</script>" not in html
    assert "&lt;script&gt;" in html


def test_importance_rows_include_win_and_loss_probability():
    html = generate_html_report(
        **_base_kwargs(
            mode="midweek",
            current_standings={"You": 62, "OneNDone": 68},
            importance_sorted=_importance_df(),
        )
    )
    # KC@NO row: win_probability 0.68, loss_probability 0.12
    assert '"winProb": 0.68' in html
    assert '"lossProb": 0.12' in html
    # SF@ARI row: win_probability 0.71, loss_probability 0.15
    assert '"winProb": 0.71' in html
    assert '"lossProb": 0.15' in html


def test_empty_importance_handled_gracefully():
    html = generate_html_report(**_base_kwargs(importance_sorted=None))
    assert "DATA.importance" in html
    assert '"importance": []' in html


def test_robustness_section_included_when_summary_stats_present():
    summary_stats = pd.DataFrame(
        [
            {
                "team": "SF",
                "frequency": 0.95,
                "appearances": 950,
                "avg_confidence": 15.2,
                "median_confidence": 16.0,
                "std_confidence": 1.1,
                "min_confidence": 10,
                "max_confidence": 16,
            },
        ]
    )
    html = generate_html_report(**_base_kwargs(summary_stats=summary_stats))
    assert "Pick Robustness" in html
    assert "robustness-table" in html


def test_robustness_section_omitted_when_no_summary_stats():
    html = generate_html_report(**_base_kwargs(summary_stats=None))
    assert "Pick Robustness" not in html


def test_rank_and_edge_values_rendered():
    html = generate_html_report(
        **_base_kwargs(
            mode="midweek",
            current_standings={"You": 62, "OneNDone": 68},
            your_rank=2,
            your_points=62,
        )
    )
    assert "#2" in html
    assert "33.8%" in html
    assert "18.1%" in html


# ---------------------------------------------------------------------------
# generate_locked_board_html_report
# ---------------------------------------------------------------------------


def _locked_standings():
    return pd.DataFrame(
        [
            {"player": "OneNDone", "locked_points": 42, "win_pct": 0.55, "expected_points": 99.1},
            {"player": "You", "locked_points": 38, "win_pct": 0.30, "expected_points": 90.4},
            {"player": "Jayparr", "locked_points": 10, "win_pct": 0.15, "expected_points": 70.2},
        ]
    )


def _locked_importance():
    return pd.DataFrame(
        [
            {"game": "NE@Sea", "vegas_home_win_pct": 0.62, "top_swing": 0.184},
            {"game": "KC@Den", "vegas_home_win_pct": 0.41, "top_swing": 0.052},
        ]
    )


def test_generate_locked_board_html_report_returns_string():
    html = generate_locked_board_html_report(
        week=5, league_id=11465, standings=_locked_standings(), importance=_locked_importance()
    )
    assert isinstance(html, str)
    assert "<title>Week 5 Live Standings</title>" in html


def test_locked_board_highlights_selected_player():
    html = generate_locked_board_html_report(
        week=5,
        league_id=11465,
        standings=_locked_standings(),
        importance=_locked_importance(),
        player_name="You",
    )
    assert '"you": true' in html
    assert "#2" in html  # You are rank 2 by win_pct order


def test_locked_board_omits_you_marker_when_no_player_given():
    html = generate_locked_board_html_report(
        week=5, league_id=11465, standings=_locked_standings(), importance=_locked_importance()
    )
    assert '"you": true' not in html


def test_locked_board_filled_players_note_rendered_and_escaped():
    html = generate_locked_board_html_report(
        week=5,
        league_id=11465,
        standings=_locked_standings(),
        importance=_locked_importance(),
        filled_players=["<script>alert(1)</script>"],
    )
    assert "auto-filled" in html
    assert "<script>alert(1)</script>" not in html
    assert "&lt;script&gt;" in html


def test_locked_board_no_filled_note_when_none_missing():
    html = generate_locked_board_html_report(
        week=5, league_id=11465, standings=_locked_standings(), importance=_locked_importance()
    )
    assert "auto-filled" not in html


def test_locked_board_handles_empty_importance():
    html = generate_locked_board_html_report(
        week=5,
        league_id=11465,
        standings=_locked_standings(),
        importance=pd.DataFrame(columns=["game", "vegas_home_win_pct", "top_swing"]),
    )
    assert '"importance": []' in html


def test_locked_board_game_names_present():
    html = generate_locked_board_html_report(
        week=5, league_id=11465, standings=_locked_standings(), importance=_locked_importance()
    )
    assert "NE @ Sea" in html
    assert "KC @ Den" in html


def _locked_importance_with_pick_columns():
    return pd.DataFrame(
        [
            {
                "game": "NE@Sea",
                "vegas_home_win_pct": 0.62,
                "top_swing": 0.184,
                "pick": "Sea",
                "points_bid": 12,
                "win_probability": 0.41,
                "loss_probability": 0.22,
                "win_delta": 0.11,
                "loss_delta": -0.08,
                "total_impact": 0.19,
            },
            {
                "game": "KC@Den",
                "vegas_home_win_pct": 0.41,
                "top_swing": 0.052,
                "pick": "KC",
                "points_bid": 5,
                "win_probability": 0.33,
                "loss_probability": 0.29,
                "win_delta": 0.03,
                "loss_delta": -0.01,
                "total_impact": 0.04,
            },
        ]
    )


def test_locked_board_with_player_name_shows_win_loss_probability():
    """When importance carries the per-player columns (standings_analytic was
    called with player_name), the locked-board report must render the same
    Correct/Wrong-style win/loss probability display as the optimizer report."""
    html = generate_locked_board_html_report(
        week=5,
        league_id=11465,
        standings=_locked_standings(),
        importance=_locked_importance_with_pick_columns(),
        player_name="You",
    )
    assert '"hasPickColumns": true' in html
    assert '"winProb": 0.41' in html
    assert '"lossProb": 0.22' in html
    assert '"pick": "Sea"' in html
    assert '"conf": 12' in html
    assert "impact on" in html.lower()


def test_locked_board_without_player_name_omits_win_loss_probability():
    html = generate_locked_board_html_report(
        week=5, league_id=11465, standings=_locked_standings(), importance=_locked_importance()
    )
    assert '"hasPickColumns": false' in html
    assert '"winProb"' not in html
    assert '"lossProb"' not in html
