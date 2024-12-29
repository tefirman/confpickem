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
from datetime import datetime, timedelta
from pytz import timezone

def convert_yahoo_to_simulator_format(yahoo_data: YahooPickEm, ignore_results: bool = False) -> pd.DataFrame:
    """
    Convert Yahoo Pick'em data into format needed for simulator
    
    Args:
        yahoo_data: YahooPickEm instance containing scraped data
        ignore_results: If True, will not include actual game outcomes even if known
    """
    is_favorite_home = ~yahoo_data.games['favorite'].str.contains('@')
    
    # Vectorized team assignment
    home_teams = np.where(is_favorite_home, 
                         yahoo_data.games['favorite'], 
                         yahoo_data.games['underdog'])
    away_teams = np.where(is_favorite_home,
                         yahoo_data.games['underdog'],
                         yahoo_data.games['favorite'])
    
    # Vectorized probability calculations  
    vegas_win_probs = yahoo_data.games['favorite_pick_pct'] / 100
    crowd_home_pick_pcts = np.where(is_favorite_home,
                                   yahoo_data.games['favorite_pick_pct'],
                                   yahoo_data.games['underdog_pick_pct']) / 100
    
    crowd_home_confidences = np.where(is_favorite_home,
                                     yahoo_data.games['favorite_confidence'],
                                     yahoo_data.games['underdog_confidence'])
    crowd_away_confidences = np.where(is_favorite_home,
                                     yahoo_data.games['underdog_confidence'],
                                     yahoo_data.games['favorite_confidence'])
    
    # Create results lookup dictionary only if we're not ignoring results
    if not ignore_results:
        results_dict = {r['favorite']: r['winner'] for r in yahoo_data.results if r['winner']}
        actual_outcomes = [results_dict.get(game['favorite']) == home_team 
                          if game['favorite'] in results_dict else None
                          for home_team, game in zip(home_teams, yahoo_data.games.to_dict('records'))]
    else:
        actual_outcomes = [None] * len(home_teams)
    
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
        'actual_outcome': actual_outcomes,
        'kickoff_time': kickoff_times,
        'game_index': game_indices
    })
    
    return games_df

def convert_picks_to_simulator_format(game_picks: pd.DataFrame, game: pd.Series, num_sims: int) -> pd.DataFrame:
    """
    Convert Yahoo picks data for a single game into simulator format
    
    Args:
        game_picks: DataFrame containing player picks and confidence for this game
        game: Series containing game information (home_team, away_team, etc)
        num_sims: Number of simulations being run
    """
    # Create game identifier
    game_id = f"{game.away_team}@{game.home_team}"
    
    picks_list = []
    
    # Process each player's pick
    for _, row in game_picks.iterrows():
        player_name = row['player_name']
        pick_col = f'game_{game.game_index}_pick'
        conf_col = f'game_{game.game_index}_confidence'
        
        if pd.notna(row[pick_col]):
            picked_team = row[pick_col]
            confidence = row[conf_col]
            picked_home = (picked_team == game.home_team)
            
            # Replicate for each simulation
            for sim in range(num_sims):
                picks_list.append({
                    'simulation': sim,
                    'player': player_name,
                    'week': game.week,
                    'game': game_id,
                    'picked_home': picked_home,
                    'confidence': confidence
                })
    
    return pd.DataFrame(picks_list)

def convert_yahoo_picks_to_dataframe(yahoo_data: YahooPickEm, num_sims: int) -> pd.DataFrame:
    """
    Convert actual Yahoo picks into simulator DataFrame format, replicated for each simulation
    """
    picks_list = []
    
    # Get mapping of teams to games
    games_df = convert_yahoo_to_simulator_format(yahoo_data)
    games = [f"{away}@{home}" for away, home in zip(games_df.away_team, games_df.home_team)]
    
    # Process each player's picks
    for _, player_row in yahoo_data.players.iterrows():
        player_name = player_row['player_name']
        
        # Process each game
        for game_idx, game in enumerate(games):
            home_team = game.split('@')[1]
            
            pick_col = f'game_{game_idx+1}_pick'
            conf_col = f'game_{game_idx+1}_confidence'
            
            if pd.notna(player_row[pick_col]):
                picked_team = player_row[pick_col]
                confidence = player_row[conf_col]
                picked_home = (picked_team == home_team)
                
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

def run_partial_simulation(yahoo_data: YahooPickEm, current_time: datetime, 
                         num_sims: int = 10000):
    """
    Run simulation using actual Yahoo picks and results where known, simulating the rest.
    """
    # Initialize simulator
    simulator = ConfidencePickEmSimulator(num_sims=num_sims)
    
    # Convert and load games data
    games_df = convert_yahoo_to_simulator_format(yahoo_data)
    
    # Mark which games should use actual results vs simulation
    games_df['use_actual_result'] = games_df.apply(
        lambda x: x['kickoff_time'] < current_time and pd.notna(x['actual_outcome']),
        axis=1
    )
    
    # Add games to simulator
    simulator.add_games_from_dataframe(games_df)
    
    # Create picks DataFrame starting with known picks
    picks_df = pd.DataFrame()
    
    for _, game in games_df.iterrows():
        game_picks = yahoo_data.players[[
            'player_name', 
            f'game_{game.game_index}_pick',
            f'game_{game.game_index}_confidence'
        ]].copy()
        
        # Only use actual picks for games where picks are locked
        if game.kickoff_time <= current_time:
            # Use actual picks
            picks_df = pd.concat([picks_df, 
                convert_picks_to_simulator_format(game_picks, game, num_sims)
            ])
        else:
            # Simulate picks for future games
            simulated_picks = simulator.simulate_picks_for_game(
                game, 
                yahoo_data.players['player_name'].unique()
            )
            picks_df = pd.concat([picks_df, simulated_picks])
    
    # Simulate outcomes
    outcomes = simulator.simulate_outcomes()
    
    # Where we have actual results, override simulated ones
    for idx, game in games_df.iterrows():
        if game.use_actual_result:
            outcomes[:, idx] = game.actual_outcome
    
    # Analyze results
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

    # Run simulation at different points in time
    print("\nSimulation only Sunday outcomes simulated:")
    current_time = datetime(2024, 9, 1, 7, 0, tzinfo=timezone('EST')) + timedelta(days=7*week)  # Sunday at 7am ET
    simulator, stats = run_partial_simulation(yahoo, current_time)
    
    print("\nSimulation incorporating known results and picks:")
    print("\nExpected Points by Player:")
    print(stats['expected_points'])
    print("\nWin Percentages:")
    print(stats['win_pct'].sort_values(ascending=False))
    print(f"Sum of win percentages: {stats['win_pct'].sum():.3f}")