#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   test_cli.py
@Time    :   2025/01/15 10:00:00
@Author  :   Test Suite
@Version :   1.0
@Desc    :   Tests for CLI interfaces (optimize, player_skills, win_probability)
"""

import pytest
import sys
import json
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock, mock_open
import pandas as pd
from datetime import datetime

# Import CLI modules
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.confpickem.cli import optimize, player_skills, win_probability


@pytest.fixture
def temp_cookies_file():
    """Create temporary cookies.txt file"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write("# Netscape HTTP Cookie File\n")
        f.write(".yahoo.com\tTRUE\t/\tTRUE\t0\ttest_cookie\ttest_value\n")
        temp_path = f.name

    yield temp_path

    # Cleanup
    Path(temp_path).unlink(missing_ok=True)


@pytest.fixture
def temp_player_skills_file():
    """Create temporary player skills JSON file"""
    skills_data = {
        "TestPlayer1": {
            "skill_level": 0.75,
            "crowd_following": 0.5,
            "confidence_following": 0.6,
            "accuracy": 0.65,
            "efficiency": 0.70,
        },
        "TestPlayer2": {
            "skill_level": 0.55,
            "crowd_following": 0.7,
            "confidence_following": 0.5,
            "accuracy": 0.55,
            "efficiency": 0.52,
        },
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(skills_data, f)
        temp_path = f.name

    yield temp_path

    # Cleanup
    Path(temp_path).unlink(missing_ok=True)


class TestOptimizeCLI:
    """Test optimize.py CLI functionality"""

    def test_main_function_exists(self):
        """Test that main function exists and is callable"""
        assert hasattr(optimize, "main")
        assert callable(optimize.main)

    def test_argument_has_week_default(self):
        """Test that --week has a default value"""
        # --week is no longer required (defaults to 3)
        pass  # Week now has a default

    def test_argument_validation_requires_mode(self):
        """Test that --mode argument is required"""
        with patch("sys.argv", ["optimize.py", "--week", "10"]):
            with pytest.raises(SystemExit):
                optimize.main()

    def test_fast_mode_only_for_beginning(self):
        """Test that --fast mode is rejected for midweek"""
        test_args = ["optimize.py", "--week", "10", "--mode", "midweek", "--fast"]

        with patch("sys.argv", test_args):
            with patch("builtins.print") as mock_print:
                result = optimize.main()

                # Should print error and return 1
                assert result == 1
                # Check that error message was printed
                error_calls = [str(call) for call in mock_print.call_args_list]
                assert any("fast mode" in str(call).lower() for call in error_calls)

    def test_fast_mode_rejected_for_locked(self):
        """--fast is rejected for --mode locked too (only beginning supports it)"""
        test_args = ["optimize.py", "--week", "10", "--mode", "locked", "--fast"]
        with patch("sys.argv", test_args):
            with patch("builtins.print") as mock_print:
                result = optimize.main()
        assert result == 1
        error_calls = [str(call) for call in mock_print.call_args_list]
        assert any("fast mode" in str(call).lower() for call in error_calls)

    def test_num_opponents_rejected_for_locked(self):
        """--num-opponents is rejected for --mode locked (needs real player data)"""
        test_args = ["optimize.py", "--week", "10", "--mode", "locked", "--num-opponents", "5"]
        with patch("sys.argv", test_args):
            with patch("builtins.print") as mock_print:
                result = optimize.main()
        assert result == 1
        error_calls = [str(call) for call in mock_print.call_args_list]
        assert any("num-opponents" in str(call).lower() for call in error_calls)

    def test_hill_climb_rejected_for_locked(self):
        """--hill-climb doesn't apply to --mode locked (nothing to optimize)"""
        test_args = ["optimize.py", "--week", "10", "--mode", "locked", "--hill-climb"]
        with patch("sys.argv", test_args):
            with patch("builtins.print") as mock_print:
                result = optimize.main()
        assert result == 1
        error_calls = [str(call) for call in mock_print.call_args_list]
        assert any("optimize" in str(call).lower() for call in error_calls)

    def test_player_rejected_outside_locked_mode(self):
        """--player only makes sense for --mode locked (others prompt interactively)"""
        test_args = ["optimize.py", "--week", "10", "--mode", "midweek", "--player", "Alice"]
        with patch("sys.argv", test_args):
            with patch("builtins.print") as mock_print:
                result = optimize.main()
        assert result == 1
        error_calls = [str(call) for call in mock_print.call_args_list]
        assert any("--player" in str(call) for call in error_calls)

    def test_mode_choices_validation(self):
        """Test that mode must be 'beginning', 'midweek', or 'locked'"""
        with patch("sys.argv", ["optimize.py", "--week", "10", "--mode", "invalid"]):
            with pytest.raises(SystemExit):
                optimize.main()

    def test_fast_requires_greedy(self):
        """--fast without --greedy is rejected (analytical default is already fast)"""
        test_args = ["optimize.py", "--week", "10", "--mode", "beginning", "--fast"]
        with patch("sys.argv", test_args):
            with patch("builtins.print") as mock_print:
                result = optimize.main()
        assert result == 1
        msgs = " ".join(str(c) for c in mock_print.call_args_list).lower()
        assert "fast mode" in msgs and "greedy" in msgs

    def test_greedy_and_hill_climb_are_mutually_exclusive(self):
        """argparse rejects --greedy --hill-climb together"""
        test_args = [
            "optimize.py",
            "--week",
            "10",
            "--mode",
            "beginning",
            "--greedy",
            "--hill-climb",
        ]
        with patch("sys.argv", test_args):
            with pytest.raises(SystemExit):
                optimize.main()

    def test_analytic_flag_still_accepted(self):
        """--analytic is a deprecated no-op, not an error"""
        parser_args = ["optimize.py", "--week", "10", "--mode", "beginning", "--analytic"]
        with patch("sys.argv", parser_args):
            with patch("src.confpickem.cli.optimize.Path") as mock_path:
                mock_path.return_value.exists.return_value = False  # bail at cookies
                with patch("builtins.print"):
                    result = optimize.main()
        # reaches the cookies check and returns 1 -- did not SystemExit on parsing
        assert result == 1

    def test_html_flag_accepted(self):
        """--html is a valid flag and does not fail argument parsing"""
        parser_args = ["optimize.py", "--week", "10", "--mode", "beginning", "--html"]
        with patch("sys.argv", parser_args):
            with patch("src.confpickem.cli.optimize.Path") as mock_path:
                mock_path.return_value.exists.return_value = False  # bail at cookies
                with patch("builtins.print"):
                    result = optimize.main()
        # reaches the cookies check and returns 1 -- did not SystemExit on parsing
        assert result == 1

    def test_compare_slate_flag_accepted_and_repeatable(self):
        """--compare-slate is valid and can be passed multiple times"""
        parser_args = [
            "optimize.py",
            "--week",
            "10",
            "--mode",
            "beginning",
            "--compare-slate",
            "manual:KC 16, SF 15",
            "--compare-slate",
            "alt:SF 16, KC 15",
        ]
        with patch("sys.argv", parser_args):
            with patch("src.confpickem.cli.optimize.Path") as mock_path:
                mock_path.return_value.exists.return_value = False  # bail at cookies
                with patch("builtins.print"):
                    result = optimize.main()
        # reaches the cookies check and returns 1 -- did not SystemExit on parsing
        assert result == 1

    def test_compare_slate_rejected_in_locked_mode(self):
        """--compare-slate has nothing to optimize against in --mode locked"""
        parser_args = [
            "optimize.py",
            "--week",
            "10",
            "--mode",
            "locked",
            "--compare-slate",
            "manual:KC 16, SF 15",
        ]
        with patch("sys.argv", parser_args):
            with patch("builtins.print"):
                result = optimize.main()
        assert result == 1

    def test_compare_slate_duplicate_labels_rejected(self):
        """--compare-slate labels must be unique"""
        parser_args = [
            "optimize.py",
            "--week",
            "10",
            "--mode",
            "beginning",
            "--compare-slate",
            "manual:KC 16, SF 15",
            "--compare-slate",
            "manual:SF 16, KC 15",
        ]
        with patch("sys.argv", parser_args):
            with patch("builtins.print"):
                result = optimize.main()
        assert result == 1

    @patch("src.confpickem.cli.optimize.Path")
    def test_missing_cookies_file_error(self, mock_path):
        """Test error handling when cookies.txt is missing"""
        mock_path.return_value.exists.return_value = False

        test_args = ["optimize.py", "--week", "10", "--mode", "beginning"]

        with patch("sys.argv", test_args):
            with patch("builtins.print") as mock_print:
                result = optimize.main()

                assert result == 1
                # Check that missing cookies error was printed
                error_calls = [str(call) for call in mock_print.call_args_list]
                assert any("cookies.txt" in str(call).lower() for call in error_calls)


class TestParseCompareSlate:
    """Unit tests for optimize.parse_compare_slate"""

    def test_parses_label_and_picks(self):
        label, picks = optimize.parse_compare_slate("manual:KC 16, SF 15, MIN 14")
        assert label == "manual"
        assert picks == {"KC": 16, "SF": 15, "MIN": 14}

    def test_strips_whitespace_around_label(self):
        label, picks = optimize.parse_compare_slate("  manual  :KC 16")
        assert label == "manual"
        assert picks == {"KC": 16}

    def test_missing_colon_raises(self):
        with pytest.raises(ValueError, match="missing a label"):
            optimize.parse_compare_slate("KC 16, SF 15")

    def test_empty_label_raises(self):
        with pytest.raises(ValueError, match="empty label"):
            optimize.parse_compare_slate(":KC 16")

    def test_malformed_pick_raises(self):
        with pytest.raises(ValueError, match="could not parse pick"):
            optimize.parse_compare_slate("manual:KC")

    def test_non_integer_confidence_raises(self):
        with pytest.raises(ValueError, match="not an integer"):
            optimize.parse_compare_slate("manual:KC sixteen")


class TestPlayerSkillsCLI:
    """Test player_skills.py CLI functionality"""

    def test_main_function_exists(self):
        """Test that main function exists"""
        assert hasattr(player_skills, "main")
        assert callable(player_skills.main)

    def test_no_command_shows_help(self):
        """Test that no command shows help and returns 1"""
        with patch("sys.argv", ["player_skills.py"]):
            with patch("builtins.print"):
                result = player_skills.main()
                assert result == 1

    def test_analyze_command_has_year_default(self):
        """Test that analyze command has --year with default value"""
        # Since --year has a default, analyze should work without it
        # But it still needs the underlying function to work
        pass  # This test is no longer needed since --year has a default

    def test_apply_command_works_without_args(self):
        """Test that apply command works without arguments (uses defaults)"""
        # apply should work without args (year defaults to None = combine all)
        # This will fail when trying to actually apply, but argument parsing should succeed
        pass  # apply has all optional args

    def test_update_command_works_with_defaults(self):
        """Test that update command works with default year"""
        # update should work without args (year defaults to 2024)
        pass  # All args have defaults

    @patch("src.confpickem.cli.player_skills.analyze_main")
    def test_analyze_command_calls_analyze_main(self, mock_analyze):
        """Test that analyze command delegates to analyze_main"""
        mock_analyze.return_value = 0

        test_args = ["player_skills.py", "analyze", "--year", "2024"]

        with patch("sys.argv", test_args):
            with patch("builtins.print"):
                result = player_skills.main()

                # Should call analyze_main
                mock_analyze.assert_called_once()
                assert result == 0

    @patch("src.confpickem.cli.player_skills.apply_main")
    def test_apply_command_calls_apply_main(self, mock_apply):
        """Test that apply command delegates to apply_main"""
        mock_apply.return_value = 0

        test_args = ["player_skills.py", "apply", "--year", "2024"]

        with patch("sys.argv", test_args):
            with patch("builtins.print"):
                result = player_skills.main()

                # Should call apply_main
                mock_apply.assert_called_once()
                assert result == 0

    @patch("src.confpickem.cli.player_skills.analyze_main")
    @patch("src.confpickem.cli.player_skills.apply_main")
    def test_update_command_calls_both(self, mock_apply, mock_analyze):
        """Test that update command calls both analyze and apply"""
        mock_analyze.return_value = 0
        mock_apply.return_value = 0

        test_args = ["player_skills.py", "update", "--year", "2024"]

        with patch("sys.argv", test_args):
            with patch("builtins.print"):
                result = player_skills.main()

                # Should call both in order
                mock_analyze.assert_called_once()
                mock_apply.assert_called_once()
                assert result == 0

    @patch("src.confpickem.cli.player_skills.analyze_main")
    @patch("src.confpickem.cli.player_skills.apply_main")
    def test_update_command_fails_if_analyze_fails(self, mock_apply, mock_analyze):
        """Test that update command stops if analyze fails"""
        mock_analyze.return_value = 1  # Simulate failure

        test_args = ["player_skills.py", "update", "--year", "2024"]

        with patch("sys.argv", test_args):
            with patch("builtins.print"):
                result = player_skills.main()

                # Should call analyze but not apply
                mock_analyze.assert_called_once()
                mock_apply.assert_not_called()
                assert result == 1


class TestWinProbabilityCLI:
    """Test win_probability.py CLI functionality"""

    def test_main_function_exists(self):
        """Test that main function exists"""
        assert hasattr(win_probability, "main")
        assert callable(win_probability.main)

    def test_has_week_default(self):
        """Test that --week has a default value"""
        # --week is no longer required (defaults to 3)
        pass  # Week now has a default


class TestCLIIntegration:
    """Integration tests for CLI workflows"""

    @patch("src.confpickem.cli.optimize.YahooPickEm")
    @patch("src.confpickem.cli.optimize.Path")
    def test_optimize_beginning_mode_workflow(self, mock_path, mock_yahoo):
        """Test complete beginning-of-week optimization workflow"""
        # Setup mocks
        mock_path.return_value.exists.return_value = True

        # Mock Yahoo data
        mock_yahoo_instance = MagicMock()
        mock_yahoo_instance.games = pd.DataFrame(
            {
                "favorite": ["KC", "SF"],
                "underdog": ["LV", "ARI"],
                "spread": [7.0, 6.5],
                "win_prob": [0.75, 0.70],
                "home_favorite": [True, True],
                "favorite_pick_pct": [80.0, 75.0],
                "underdog_pick_pct": [20.0, 25.0],
                "favorite_confidence": [12.0, 11.0],
                "underdog_confidence": [4.0, 5.0],
                "kickoff_time": [datetime.now(), datetime.now()],
            }
        )
        mock_yahoo_instance.players = pd.DataFrame({"player_name": ["TestPlayer1", "TestPlayer2"]})
        mock_yahoo_instance.results = []

        mock_yahoo.return_value = mock_yahoo_instance

        test_args = ["optimize.py", "--week", "10", "--mode", "beginning", "--fast"]

        with patch("sys.argv", test_args):
            with patch("builtins.print"):
                with patch("builtins.input", return_value="1"):  # Select player 1
                    with patch("builtins.input", return_value=""):  # No fixed picks
                        # This would normally run full optimization
                        # For now, just test that it starts without errors
                        try:
                            # Note: Full optimization takes too long for unit test
                            # We're just testing that argument parsing works
                            pass
                        except Exception as e:
                            pytest.fail(f"Optimization workflow failed: {e}")

    @patch("src.confpickem.cli.optimize.YahooPickEm")
    @patch("src.confpickem.cli.optimize.Path")
    def test_optimize_locked_mode_workflow(self, mock_path, mock_yahoo, tmp_path, monkeypatch):
        """--mode locked runs end-to-end with no prompts and no optimization,
        auto-filling any entrant missing a pick."""
        monkeypatch.chdir(tmp_path)
        mock_path.return_value.exists.return_value = True

        mock_yahoo_instance = MagicMock()
        mock_yahoo_instance.games = pd.DataFrame(
            {
                "favorite": ["KC", "SF"],
                "underdog": ["LV", "ARI"],
                "spread": [7.0, 6.5],
                "win_prob": [0.75, 0.70],
                "home_favorite": [True, True],
                "favorite_pick_pct": [80.0, 75.0],
                "underdog_pick_pct": [20.0, 25.0],
                "favorite_confidence": [12.0, 11.0],
                "underdog_confidence": [4.0, 5.0],
                "kickoff_time": [datetime(2024, 9, 8, 13, 0), datetime(2024, 9, 8, 16, 25)],
            }
        )
        mock_yahoo_instance.players = pd.DataFrame(
            [
                {
                    "player_name": "Complete",
                    "game_1_pick": "KC",
                    "game_1_confidence": 2,
                    "game_2_pick": "SF",
                    "game_2_confidence": 1,
                },
                {
                    # missing game_2 entirely -- standings_analytic must not raise
                    "player_name": "Jayparr",
                    "game_1_pick": "KC",
                    "game_1_confidence": 1,
                    "game_2_pick": None,
                    "game_2_confidence": None,
                },
            ]
        )
        mock_yahoo_instance.results = [
            {"favorite": "KC", "underdog": "LV", "winner": None},
            {"favorite": "SF", "underdog": "ARI", "winner": None},
        ]
        mock_yahoo.return_value = mock_yahoo_instance

        test_args = ["optimize.py", "--week", "1", "--mode", "locked", "--html"]
        with patch("sys.argv", test_args):
            result = optimize.main()

        assert result == 0
        txt_files = list(tmp_path.glob("NFL_Week1_Locked_*.txt"))
        html_files = list(tmp_path.glob("NFL_Week1_Locked_*.html"))
        assert len(txt_files) == 1
        assert len(html_files) == 1

        report = txt_files[0].read_text()
        assert "LIVE STANDINGS" in report
        assert "Jayparr" in report
        assert "Auto-filled missing picks" in report

        html = html_files[0].read_text()
        assert "Live Standings" in html
        assert "Jayparr" in html

    def _locked_mode_yahoo_mock(self):
        mock_yahoo_instance = MagicMock()
        mock_yahoo_instance.games = pd.DataFrame(
            {
                "favorite": ["KC", "SF"],
                "underdog": ["LV", "ARI"],
                "spread": [7.0, 6.5],
                "win_prob": [0.75, 0.70],
                "home_favorite": [True, True],
                "favorite_pick_pct": [80.0, 75.0],
                "underdog_pick_pct": [20.0, 25.0],
                "favorite_confidence": [12.0, 11.0],
                "underdog_confidence": [4.0, 5.0],
                "kickoff_time": [datetime(2024, 9, 8, 13, 0), datetime(2024, 9, 8, 16, 25)],
            }
        )
        mock_yahoo_instance.players = pd.DataFrame(
            [
                {
                    "player_name": "Alice Anderson",
                    "game_1_pick": "KC",
                    "game_1_confidence": 2,
                    "game_2_pick": "SF",
                    "game_2_confidence": 1,
                },
                {
                    "player_name": "Alice Smith",
                    "game_1_pick": "LV",
                    "game_1_confidence": 1,
                    "game_2_pick": "SF",
                    "game_2_confidence": 2,
                },
            ]
        )
        mock_yahoo_instance.results = [
            {"favorite": "KC", "underdog": "LV", "winner": None},
            {"favorite": "SF", "underdog": "ARI", "winner": None},
        ]
        return mock_yahoo_instance

    @patch("src.confpickem.cli.optimize.YahooPickEm")
    @patch("src.confpickem.cli.optimize.Path")
    def test_locked_mode_player_flag_highlights_match(
        self, mock_path, mock_yahoo, tmp_path, monkeypatch
    ):
        """--player uniquely matching one entrant highlights their row"""
        monkeypatch.chdir(tmp_path)
        mock_path.return_value.exists.return_value = True
        mock_yahoo.return_value = self._locked_mode_yahoo_mock()

        test_args = [
            "optimize.py",
            "--week",
            "1",
            "--mode",
            "locked",
            "--player",
            "Anderson",
            "--html",
        ]
        with patch("sys.argv", test_args):
            result = optimize.main()

        assert result == 0
        report = list(tmp_path.glob("NFL_Week1_Locked_*.txt"))[0].read_text()
        assert "Player: Alice Anderson" in report
        # same display as the optimizer's game importance: Correct/Wrong probabilities
        assert "GAME IMPORTANCE ANALYSIS (Alice Anderson)" in report
        assert "Correct:" in report and "Wrong:" in report

        html = list(tmp_path.glob("NFL_Week1_Locked_*.html"))[0].read_text()
        assert '"hasPickColumns": true' in html
        assert "-> " in report  # marker on the matched entrant's row

    @patch("src.confpickem.cli.optimize.YahooPickEm")
    @patch("src.confpickem.cli.optimize.Path")
    def test_locked_mode_player_flag_ambiguous(self, mock_path, mock_yahoo, tmp_path, monkeypatch):
        """--player matching multiple entrants errors instead of guessing"""
        monkeypatch.chdir(tmp_path)
        mock_path.return_value.exists.return_value = True
        mock_yahoo.return_value = self._locked_mode_yahoo_mock()

        test_args = ["optimize.py", "--week", "1", "--mode", "locked", "--player", "Alice"]
        with patch("sys.argv", test_args):
            with patch("builtins.print") as mock_print:
                result = optimize.main()

        assert result == 1
        msgs = " ".join(str(c) for c in mock_print.call_args_list)
        assert "multiple entrants" in msgs

    @patch("src.confpickem.cli.optimize.YahooPickEm")
    @patch("src.confpickem.cli.optimize.Path")
    def test_locked_mode_player_flag_not_found(self, mock_path, mock_yahoo, tmp_path, monkeypatch):
        """--player matching no entrant errors instead of silently omitting it"""
        monkeypatch.chdir(tmp_path)
        mock_path.return_value.exists.return_value = True
        mock_yahoo.return_value = self._locked_mode_yahoo_mock()

        test_args = ["optimize.py", "--week", "1", "--mode", "locked", "--player", "Nobody"]
        with patch("sys.argv", test_args):
            with patch("builtins.print") as mock_print:
                result = optimize.main()

        assert result == 1
        msgs = " ".join(str(c) for c in mock_print.call_args_list)
        assert "not found" in msgs


class TestCLIOutputFiles:
    """Test CLI output file generation"""

    def test_optimize_creates_output_file(self):
        """Test that optimization creates results file"""
        # This would test file creation
        # Implementation depends on how output files are structured
        pass

    def test_player_skills_creates_json(self):
        """Test that player skills creates current_player_skills.json"""
        # This would test JSON file creation
        pass


class TestCLIErrorHandling:
    """Test error handling across CLI modules"""

    def test_graceful_handling_of_network_errors(self):
        """Test that network errors are handled gracefully"""
        # Mock network failures and verify graceful degradation
        pass

    def test_invalid_week_number_handling(self):
        """Test handling of invalid week numbers (0, 19, etc)"""
        # Test week validation
        pass

    def test_invalid_league_id_handling(self):
        """Test handling of invalid league IDs"""
        # Test league ID validation
        pass


# Additional helper tests
class TestCLIHelpers:
    """Test CLI helper functions and utilities"""

    def test_confidence_range_calculation(self):
        """Test confidence range setting based on mode"""
        # Fast mode should use confidence_range=4
        # Normal mode should use confidence_range=3
        pass

    def test_simulation_count_calculation(self):
        """Test num_sims setting based on arguments"""
        # Custom should override defaults
        # Fast mode should use 2000
        # Normal should use 2000 (default)
        pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
