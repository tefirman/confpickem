"""Tests for the season CLI helpers and the --compare-slate dollar columns."""

import pandas as pd

from src.confpickem.cli import optimize, season


def test_ordinal():
    assert [season._ordinal(i) for i in (1, 2, 3, 4, 11, 21)] == [
        "1st",
        "2nd",
        "3rd",
        "4th",
        "11th",
        "21st",
    ]


def test_format_outlook_marks_player_and_inactive():
    outlook = pd.DataFrame(
        {
            "player": ["Alice", "Bob"],
            "points": [100.0, 90.0],
            "current_rank": [1, 2],
            "active": [True, False],
            "exp_final_points": [900.0, 800.0],
            "p_place_1": [0.6, 0.1],
            "p_place_2": [0.3, 0.1],
            "p_place_3": [0.05, 0.1],
            "p_place_4": [0.03, 0.1],
            "p_top4": [0.98, 0.4],
            "exp_weekly_dollars": [120.0, 40.0],
            "exp_season_dollars": [1900.0, 300.0],
            "exp_dollars": [2020.0, 340.0],
        }
    )
    lines = season.format_outlook(outlook, (2250, 1000, 500, 250), highlight="bob")
    assert "P(1st)" in lines[0] and "P(4th)" in lines[0]
    assert lines[2].startswith("  1 Alice") and "60.0%" in lines[2]
    assert "*Bob" in lines[3] and "(inactive)" in lines[3]


def test_comparison_lines_without_dollars_keeps_original_columns():
    df = pd.DataFrame(
        {"rank": [1], "label": ["optimized"], "win_probability": [0.12], "win_std": [0.2]}
    )
    lines = optimize.comparison_lines(df, indent="   ")
    assert len(lines) == 3 and "Total $" not in lines[0]
    assert "12.0%" in lines[2]


def test_comparison_lines_with_dollars():
    df = pd.DataFrame(
        {
            "rank": [1, 2],
            "label": ["optimized", "alt"],
            "win_probability": [0.12, 0.08],
            "win_std": [0.2, 0.1],
            "weekly_dollars": [60.0, 40.0],
            "season_dollars": [100.0, 104.0],
            "expected_dollars": [160.0, 144.0],
            "p_season_top4": [0.1, 0.11],
            "season_dollars_se": [0.0, 3.2],
        }
    )
    lines = optimize.comparison_lines(df)
    assert "Total $" in lines[0] and "160.00" in lines[2] and "3.20" in lines[3]
    assert any("noise" in line for line in lines)
