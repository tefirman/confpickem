import pytest
import numpy as np
import pandas as pd
from datetime import datetime
from confpickem.confidence_pickem_sim import ConfidencePickEmSimulator, Game, Player

@pytest.fixture
def sample_games():
    """Fixture providing a list of sample games for testing"""
    return [
        Game(
            home_team="SF",
            away_team="SEA",
            vegas_win_prob=0.7,
            crowd_home_pick_pct=0.8,
            crowd_home_confidence=13.5,
            crowd_away_confidence=3.5,
            week=1,
            kickoff_time=datetime(2024, 9, 8, 13, 0)
        ),
        Game(
            home_team="KC",
            away_team="DEN",
            vegas_win_prob=0.65,
            crowd_home_pick_pct=0.75,
            crowd_home_confidence=12.0,
            crowd_away_confidence=4.0,
            week=1,
            kickoff_time=datetime(2024, 9, 8, 16, 25)
        )
    ]

@pytest.fixture
def sample_players():
    """Fixture providing a list of sample players for testing"""
    return [
        Player(name="Player 1", skill_level=0.8, crowd_following=0.3, confidence_following=0.4),
        Player(name="Player 2", skill_level=0.6, crowd_following=0.7, confidence_following=0.6)
    ]

@pytest.fixture
def simulator(sample_games, sample_players):
    """Fixture providing a configured simulator instance"""
    sim = ConfidencePickEmSimulator(num_sims=1000)
    sim.games = sample_games
    sim.players = sample_players
    return sim

def test_simulator_initialization():
    """Test basic simulator initialization"""
    sim = ConfidencePickEmSimulator(num_sims=1000)
    assert sim.num_sims == 1000
    assert len(sim.games) == 0
    assert len(sim.players) == 0

def test_game_initialization():
    """Test game object initialization"""
    game = Game(
        home_team="SF",
        away_team="SEA",
        vegas_win_prob=0.7,
        crowd_home_pick_pct=0.8,
        crowd_home_confidence=13.5,
        crowd_away_confidence=3.5,
        week=1,
        kickoff_time=datetime(2024, 9, 8, 13, 0)
    )
    assert game.home_team == "SF"
    assert game.away_team == "SEA"
    assert game.vegas_win_prob == 0.7
    assert game.crowd_home_pick_pct == 0.8
    assert game.crowd_home_confidence == 13.5
    assert game.crowd_away_confidence == 3.5
    assert game.actual_outcome is None
    assert not game.picks_locked

def test_simulate_picks_shape(simulator):
    """Test the shape and structure of simulated picks"""
    picks_df = simulator.simulate_picks()
    
    # Check DataFrame structure
    expected_cols = ['simulation', 'player', 'week', 'game', 'picked_home', 'confidence']
    assert all(col in picks_df.columns for col in expected_cols)
    
    # Check dimensions
    expected_rows = simulator.num_sims * len(simulator.players) * len(simulator.games)
    assert len(picks_df) == expected_rows

def test_simulate_picks_values(simulator):
    """Test the values in simulated picks are within expected ranges"""
    picks_df = simulator.simulate_picks()
    
    # Check value ranges
    assert picks_df['simulation'].min() >= 0
    assert picks_df['simulation'].max() < simulator.num_sims
    assert picks_df['confidence'].min() >= 1
    assert picks_df['confidence'].max() <= len(simulator.games)
    assert picks_df['picked_home'].isin([True, False]).all()

def test_simulate_outcomes(simulator):
    """Test game outcome simulation"""
    outcomes = simulator.simulate_outcomes()
    
    # Check dimensions
    assert outcomes.shape == (simulator.num_sims, len(simulator.games))
    
    # Check values are binary
    assert np.array_equal(outcomes, outcomes.astype(bool))

def test_analyze_results(simulator):
    """Test results analysis"""
    picks_df = simulator.simulate_picks()
    outcomes = simulator.simulate_outcomes()
    stats = simulator.analyze_results(picks_df, outcomes)
    
    # Check stats contain expected keys
    expected_keys = ['expected_points', 'point_std', 'win_pct', 'value_at_risk']
    assert all(key in stats for key in expected_keys)
    
    # Check win percentages sum to approximately 1
    assert abs(stats['win_pct'].sum() - 1.0) < 0.01
    
    # Check all stats are non-negative
    assert (stats['expected_points'] >= 0).all()
    assert (stats['point_std'] >= 0).all()
    assert (stats['win_pct'] >= 0).all()
    assert (stats['value_at_risk'] >= 0).all()

def test_fixed_picks(simulator):
    """Test handling of fixed picks"""
    fixed_picks = {"SF": 2, "KC": 1}
    picks_df = simulator.simulate_picks(fixed_picks)
    
    # Filter for player with fixed picks (first player)
    player_picks = picks_df[picks_df['player'] == simulator.players[0].name]
    
    # Check SF pick
    sf_picks = player_picks[player_picks['game'].str.contains("SF")]
    assert sf_picks['confidence'].iloc[0] == 2
    assert sf_picks['picked_home'].all()  # SF is home team
    
    # Check KC pick
    kc_picks = player_picks[player_picks['game'].str.contains("KC")]
    assert kc_picks['confidence'].iloc[0] == 1
    assert kc_picks['picked_home'].all()  # KC is home team

def test_game_importance(simulator):
    """Test game importance calculation"""
    picks_df = simulator.simulate_picks()
    results = simulator.assess_game_importance(picks_df)
    
    # Check DataFrame structure
    expected_cols = ['game', 'points_bid', 'win_probability', 'loss_probability', 
                    'win_delta', 'loss_delta', 'total_impact']
    assert all(col in results.columns for col in expected_cols)
    
    # Check number of games analyzed
    assert len(results) == len(simulator.games)
    
    # Check probability ranges
    assert (results['win_probability'] >= 0).all() and (results['win_probability'] <= 1).all()
    assert (results['loss_probability'] >= 0).all() and (results['loss_probability'] <= 1).all()

def test_optimize_picks(simulator):
    """Test pick optimization"""
    fixed_picks = {"SF": 2}  # Fix one pick to test partial optimization
    optimal_picks = simulator.optimize_picks(fixed_picks)
    
    # Check that fixed pick is preserved
    assert optimal_picks.get("SF") == 2
    
    # Check that all confidence points are used exactly once
    points_used = list(optimal_picks.values())
    assert len(points_used) == len(simulator.games)
    assert len(set(points_used)) == len(points_used)
    assert min(points_used) >= 1
    assert max(points_used) <= len(simulator.games)

def test_error_handling():
    """Test error handling for invalid inputs"""
    sim = ConfidencePickEmSimulator(num_sims=1000)
    
    # Test invalid fixed picks
    with pytest.raises(ValueError):
        sim.simulate_picks(fixed_picks={"INVALID": 1})
    
    # Test invalid confidence points
    with pytest.raises(ValueError):
        sim.simulate_picks(fixed_picks={"SF": len(sim.games) + 1})
    
    # Test negative number of simulations
    with pytest.raises(ValueError):
        ConfidencePickEmSimulator(num_sims=-1)
