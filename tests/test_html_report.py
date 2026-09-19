#!/usr/bin/env python
"""Tests for src.confpickem.html_report"""

import json

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


def test_picks_include_matchup_details_from_picked_teams_perspective():
    html = generate_html_report(
        **_base_kwargs(
            remaining_games=[
                {
                    "home": "SF",
                    "away": "ARI",
                    "spread": -6.5,
                    "favorite": "SF",
                    "home_win_prob": 0.78,
                    "home_pick_pct": 0.82,
                    "kickoff_time": pd.Timestamp("2026-09-20 13:00:00"),
                },
                {
                    "home": "NO",
                    "away": "KC",
                    "spread": -3.0,
                    "favorite": "KC",
                    "home_win_prob": 0.41,
                    "home_pick_pct": 0.35,
                    "kickoff_time": pd.Timestamp("2026-09-21 20:20:00"),
                },
            ]
        )
    )
    # SF is home and the favorite: spread/win/crowd stay as-is.
    assert '"spread": -6.5' in html
    assert '"winProb": 0.78' in html
    assert '"crowdPct": 0.82' in html
    assert '"kickoff": "Sun 1:00 PM"' in html
    # KC is away and the favorite: spread stays negative, win/crowd flip to KC's perspective.
    assert '"spread": -3.0' in html
    assert '"winProb": 0.59' in html
    assert '"crowdPct": 0.65' in html
    assert '"kickoff": "Mon 8:20 PM"' in html


def test_picks_matchup_details_missing_for_locked_games():
    html = generate_html_report(**_base_kwargs(remaining_games=[]))
    data_start = html.index("var DATA = ") + len("var DATA = ")
    data_json = html[data_start : html.index(";\n", data_start)]
    data = json.loads(data_json)
    for pick in data["picks"]:
        assert pick["locked"] is True
        assert pick["spread"] is None
        assert pick["winProb"] is None
        assert pick["crowdPct"] is None
        assert pick["kickoff"] is None


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


def _comparison_df():
    return pd.DataFrame(
        [
            {"label": "optimized", "win_probability": 0.42, "win_std": 0.31, "rank": 1},
            {"label": "manual", "win_probability": 0.38, "win_std": 0.22, "rank": 2},
        ]
    )


def test_comparison_section_included_when_comparison_df_present():
    html = generate_html_report(**_base_kwargs(comparison_df=_comparison_df()))
    assert "Slate Comparison" in html
    assert "comparison-table" in html
    assert '"label": "optimized"' in html
    assert '"win_std": 0.31' in html


def test_comparison_section_omitted_when_no_comparison_df():
    html = generate_html_report(**_base_kwargs(comparison_df=None))
    assert "Slate Comparison" not in html


def test_comparison_section_omitted_when_default_not_passed():
    html = generate_html_report(**_base_kwargs())
    assert "Slate Comparison" not in html


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


def _large_field_win_probs(your_index=48, size=50):
    return [
        {
            "player": f"Player{i+1}",
            "win_pct": 0.9 - i * 0.01,
            "total_expected": 150 - i,
            "current_pts": 60,
            "is_you": i == your_index,
        }
        for i in range(size)
    ]


def test_rank_denominator_uses_full_field_not_displayed_rows():
    """Current Rank's "/ N" must count every entrant, not just the displayed top 25."""
    html = generate_html_report(
        **_base_kwargs(
            mode="midweek",
            current_standings={f"Player{i+1}": 60 for i in range(50)},
            all_win_probs=_large_field_win_probs(),
            your_rank=49,
            your_points=62,
        )
    )
    assert "#49" in html
    idx = html.index("Current Rank")
    assert "/ 50" in html[idx : idx + 200]
    assert "/ 25" not in html[idx : idx + 200]


def test_stat_strip_subtitles_match_their_own_stat():
    """"of N tracked" belongs under Current Rank; the points/games note belongs
    under Games Remaining -- they were swapped in a previous version."""
    html = generate_html_report(
        **_base_kwargs(
            mode="midweek",
            current_standings={"You": 62, "OneNDone": 68},
            all_win_probs=_all_win_probs(),
            your_rank=2,
            your_points=62,
        )
    )
    rank_idx = html.index("Current Rank")
    rank_block = html[rank_idx : rank_idx + 300]
    games_idx = html.index("Games Remaining")
    games_block = html[games_idx : games_idx + 300]

    assert "tracked" in rank_block
    assert "tracked" not in games_block
    assert "completed games" in games_block
    assert "completed games" not in rank_block


def test_standings_table_includes_your_row_when_outside_top_25():
    html = generate_html_report(
        **_base_kwargs(
            mode="midweek",
            current_standings={f"Player{i+1}": 60 for i in range(50)},
            all_win_probs=_large_field_win_probs(),
            your_rank=49,
            your_points=62,
        )
    )
    data_start = html.index("var DATA = ") + len("var DATA = ")
    data_json = html[data_start : html.index(";\n", data_start)]
    data = json.loads(data_json)
    assert len(data["standings"]) == 26
    assert data["standings"][-1]["name"] == "Player49"
    assert data["standings"][-1]["rank"] == 49
    assert data["standings"][-1]["you"] is True


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
