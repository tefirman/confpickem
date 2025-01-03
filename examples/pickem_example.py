#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   confidence_pickem_example.py
@Time    :   2024/12/09 14:30:00
@Author  :   Example Author
@Version :   1.0
@Desc    :   Example usage of ConfidencePickEmSimulator
'''

import sys
sys.path.append("../src/confpickem/")
from confidence_pickem_sim import ConfidencePickEmSimulator, Player, Game
import pandas as pd

def main():
    # Initialize simulator with 10,000 simulations
    simulator = ConfidencePickEmSimulator(num_sims=10000)

    # Create sample games data
    games_data = pd.DataFrame({
        'home_team': ['KC', 'DAL', 'SF', 'PHI', 'BUF'],
        'away_team': ['LV', 'NYG', 'SEA', 'MIN', 'NYJ'],
        'vegas_win_prob': [0.75, 0.68, 0.72, 0.65, 0.70],
        'crowd_home_pick_pct': [0.82, 0.75, 0.78, 0.70, 0.73],
        'crowd_home_confidence': [13.2, 11.5, 12.8, 10.5, 11.8],
        'crowd_away_confidence': [4.5, 5.2, 4.8, 6.2, 5.5],
        'week': [14, 14, 14, 14, 14],
        'actual_outcome': [None, None, None, None, None]  # None for games not yet played
    })

    # Load games into simulator
    simulator.add_games_from_dataframe(games_data)

    # Add players with different characteristics
    simulator.players = [
        Player("Expert Player", skill_level=0.9, crowd_following=0.2, confidence_following=0.3),
        Player("Crowd Follower", skill_level=0.5, crowd_following=0.9, confidence_following=0.8),
        Player("Average Joe", skill_level=0.5, crowd_following=0.5, confidence_following=0.5),
        Player("Contrarian", skill_level=0.7, crowd_following=0.1, confidence_following=0.4)
    ]

    # Run each of the individual pieces...
    picks_df = simulator.simulate_picks()
    outcomes = simulator.simulate_outcomes()
    stats = simulator.analyze_results(picks_df, outcomes)

    # ... or run them all in one function
    stats = simulator.simulate_all()

    # Print summary statistics
    print("\nExpected Points by Player:")
    print(stats['expected_points'])

    print("\nWin Percentages:")
    print(stats['win_pct'])

    print("\nPoint Standard Deviations:")
    print(stats['point_std'])

    print("\nValue at Risk (5th percentile):")
    print(stats['value_at_risk'])

    # Optimize picks for Expert Player
    print("\nOptimal Picks for Expert Player:")
    optimal_picks = simulator.optimize_picks()
    print(optimal_picks)

    # Example with some fixed picks
    fixed_picks = {'PHI': 5, 'BUF': 4}  # Force PHI with 5 points and BUF with 4 points
    print("\nOptimal Picks with Fixed Selections:")
    optimal_picks_fixed = simulator.optimize_picks(fixed_picks)
    print(optimal_picks_fixed)

if __name__ == "__main__":
    main()
