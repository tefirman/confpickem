#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   pickem_integration.py 
@Time    :   2024/12/12
@Desc    :   Integration script using actual Yahoo picks instead of simulating new ones
'''

from yahoo_pickem_scraper import YahooPickEm
from confidence_pickem_sim import ConfidencePickEmSimulator
import pandas as pd
import numpy as np

def convert_yahoo_to_simulator_format(yahoo_data: YahooPickEm, ignore_results: bool = False) -> pd.DataFrame:
    """
    Convert Yahoo Pick'em data into format needed for simulator
    
    Args:
        yahoo_data: YahooPickEm instance containing scraped data
        ignore_results: If True, will not include actual game outcomes even if known
        
    Returns:
        DataFrame containing game data in simulator format
    """
    # Vectorized team assignment
    home_teams = np.where(yahoo_data.games['home_favorite'], 
                         yahoo_data.games['favorite'], 
                         yahoo_data.games['underdog'])
    away_teams = np.where(yahoo_data.games['home_favorite'],
                         yahoo_data.games['underdog'],
                         yahoo_data.games['favorite'])
    
    # Vectorized probability calculations
    vegas_win_probs = yahoo_data.games['win_prob']
    crowd_home_pick_pcts = np.where(yahoo_data.games['home_favorite'],
                                   yahoo_data.games['favorite_pick_pct'],
                                   yahoo_data.games['underdog_pick_pct']) / 100
    
    crowd_home_confidences = np.where(yahoo_data.games['home_favorite'],
                                     yahoo_data.games['favorite_confidence'],
                                     yahoo_data.games['underdog_confidence'])
    crowd_away_confidences = np.where(yahoo_data.games['home_favorite'],
                                     yahoo_data.games['underdog_confidence'],
                                     yahoo_data.games['favorite_confidence'])
    
    # Only include actual outcomes if not ignoring results
    actual_outcomes = None
    if not ignore_results:
        # Create results lookup dictionary
        results_dict = {r['favorite']: r['winner'] for r in yahoo_data.results if r['winner']}
        actual_outcomes = [results_dict.get(game['favorite']) == home_team 
                          if game['favorite'] in results_dict else None
                          for home_team, game in zip(home_teams, yahoo_data.games.to_dict('records'))]
    
    # Get kickoff times
    kickoff_times = yahoo_data.games.kickoff_time.tolist()

    # Get game indices (1-based to match Yahoo's format)
    game_indices = range(1, len(home_teams) + 1)

    games_df = pd.DataFrame({
        'home_team': home_teams,
        'away_team': away_teams,
        'vegas_win_prob': vegas_win_probs,
        'crowd_home_pick_pct': crowd_home_pick_pcts,
        'crowd_home_confidence': crowd_home_confidences,
        'crowd_away_confidence': crowd_away_confidences,
        'week': yahoo_data.week,
        'kickoff_time': kickoff_times,
        'game_index': game_indices
    })
    
    # Only add actual outcomes if we're not ignoring results
    if actual_outcomes:
        games_df['actual_outcome'] = actual_outcomes
        
    return games_df

def convert_yahoo_picks_to_dataframe(yahoo_data: YahooPickEm, num_sims: int, alternative_picks: dict = None) -> pd.DataFrame:
    """
    Convert actual Yahoo picks into simulator DataFrame format, replicated for each simulation.
    Can override picks for a specific player with alternative picks.
    
    Args:
        yahoo_data: YahooPickEm instance containing scraped data
        num_sims: Number of simulations to run
        alternative_picks: Dictionary mapping team names to confidence points, e.g.:
            {'SF': 16, 'KC': 15, ...}
    """
    picks_list = []
    
    # Get mapping of teams to games
    games_df = convert_yahoo_to_simulator_format(yahoo_data)
    games = [f"{away}@{home}" for away, home in zip(games_df.away_team, games_df.home_team)]
    
    # Process each player's picks
    for _, player_row in yahoo_data.players.iterrows():
        player_name = player_row['player_name']
        
        # For your entry, convert alternative picks to game indices first
        game_to_pick = {}
        if player_name == "Firman's Educated Guesses" and alternative_picks:
            for game_idx, game in enumerate(games):
                home = game.split('@')[1]
                away = game.split('@')[0]
                if home in alternative_picks:
                    game_to_pick[game_idx + 1] = {'pick': home, 'confidence': alternative_picks[home]}
                elif away in alternative_picks:
                    game_to_pick[game_idx + 1] = {'pick': away, 'confidence': alternative_picks[away]}
        
        # Process each game
        for game_idx, game in enumerate(games):
            home_team = game.split('@')[1]
            away_team = game.split('@')[0]
            
            pick_col = f'game_{game_idx+1}_pick'
            conf_col = f'game_{game_idx+1}_confidence'
            
            # Use alternative pick if specified for this player and game
            if (player_name == "Firman's Educated Guesses" and 
                game_to_pick and 
                game_idx + 1 in game_to_pick):
                
                alt_pick = game_to_pick[game_idx + 1]['pick']
                confidence = game_to_pick[game_idx + 1]['confidence']
                picked_home = (alt_pick == home_team)
                
            elif pd.notna(player_row[pick_col]):
                picked_team = player_row[pick_col]
                confidence = player_row[conf_col]
                picked_home = (picked_team == home_team)
            else:
                continue
                
            # Replicate for each simulation
            for sim in range(num_sims):
                picks_list.append({
                    'simulation': sim,
                    'player': player_name,
                    'week': yahoo_data.week,
                    'game': game,
                    'picked_home': picked_home,
                    'confidence': confidence
                })
    
    return pd.DataFrame(picks_list)

def run_simulation(yahoo_data: YahooPickEm, num_sims: int = 10000, ignore_results: bool = False):
    """
    Run simulation using actual Yahoo picks. Can optionally ignore known game results.
    
    Args:
        yahoo_data: YahooPickEm instance containing scraped data
        num_sims: Number of simulations to run
        ignore_results: If True, will simulate all game outcomes regardless of known results
    """
    # Initialize simulator
    simulator = ConfidencePickEmSimulator(num_sims=num_sims)
    
    # Convert and load games data
    games_df = convert_yahoo_to_simulator_format(yahoo_data, ignore_results=ignore_results)
    simulator.add_games_from_dataframe(games_df)
    
    # Create picks DataFrame using actual picks
    picks_df = convert_yahoo_picks_to_dataframe(yahoo_data, num_sims)
    
    # Simulate outcomes
    outcomes = simulator.simulate_outcomes()
    
    # Analyze results using actual picks
    stats = simulator.analyze_results(picks_df, outcomes)
    
    return simulator, stats

# Example usage:
if __name__ == "__main__":
    # Initialize Yahoo scraper
    week = 1
    yahoo = YahooPickEm(week=week, league_id=6207, cookies_file='cookies.txt')
    
    # Run simulation with known results
    print("\nSimulation using known results:")
    simulator, stats = run_simulation(yahoo, ignore_results=False)
    print("\nExpected Points by Player:")
    print(stats['expected_points'])
    print("\nWin Percentages:")
    print(stats['win_pct'])
    print(f"Sum of win percentages: {stats['win_pct'].sum():.3f}")
    
    # Run simulation ignoring known results
    print("\nSimulation with all outcomes simulated:")
    simulator, stats = run_simulation(yahoo, ignore_results=True) 
    print("\nExpected Points by Player:")
    print(stats['expected_points'])
    print("\nWin Percentages:")
    print(stats['win_pct'].sort_values(ascending=False))
    print(f"Sum of win percentages: {stats['win_pct'].sum():.3f}")
