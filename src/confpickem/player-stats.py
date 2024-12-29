#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   player-stats.py
@Time    :   2024/12/29 00:38:50
@Author  :   Taylor Firman
@Version :   v0.1
@Contact :   tfirman@fredhutch.org
@Desc    :   Method for extracting average player statistics across the course of the season
'''

import pandas as pd
from yahoo_pickem_scraper import YahooPickEm

def calculate_player_stats(league_id: int, weeks: list, cookies_file: str):
    """
    Calculate player statistics based on historical performance.
    
    Args:
        league_id: Yahoo Pick'em league ID
        weeks: List of weeks to analyze
        cookies_file: Path to cookies.txt file
        
    Returns:
        DataFrame containing player statistics
    """
    all_picks = []
    all_games = []
    
    # Gather historical data
    for week in weeks:
        yahoo = YahooPickEm(week=week, league_id=league_id, cookies_file=cookies_file)
        
        # Add week number to dataframes
        yahoo.players['week'] = week
        yahoo.games['week'] = week
        
        all_picks.append(yahoo.players)
        all_games.append(yahoo.games)
    
    picks_df = pd.concat(all_picks, ignore_index=True)
    games_df = pd.concat(all_games, ignore_index=True)
    
    player_stats = []
    
    for player in picks_df['player_name'].unique():
        player_picks = picks_df[picks_df['player_name'] == player]
        
        # Calculate skill level based on correct pick percentage
        total_picks = 0
        correct_picks = 0
        
        # Calculate crowd following based on agreement with majority
        crowd_agreement = 0
        total_comparable = 0
        
        # Calculate confidence alignment
        confidence_alignment = 0
        total_confidence = 0
        
        for week in player_picks['week'].unique():
            week_picks = player_picks[player_picks['week'] == week]
            week_games = games_df[games_df['week'] == week]
            
            for game_num in range(len(week_games)):
                pick_col = f'game_{game_num+1}_pick'
                conf_col = f'game_{game_num+1}_confidence'
                correct_col = f'game_{game_num+1}_correct'
                
                if pd.notna(week_picks[pick_col].iloc[0]):
                    pick = week_picks[pick_col].iloc[0]
                    confidence = week_picks[conf_col].iloc[0]
                    
                    # Skill level calculation
                    total_picks += 1
                    if week_picks[correct_col].iloc[0]:
                        correct_picks += 1
                    
                    # Crowd following calculation
                    game = week_games.iloc[game_num]
                    picked_favorite = (pick == game['favorite'])
                    majority_picked_favorite = (game['favorite_pick_pct'] > 50)
                    
                    total_comparable += 1
                    if picked_favorite == majority_picked_favorite:
                        crowd_agreement += 1
                    
                    # Confidence alignment calculation
                    expected_conf = game['favorite_confidence'] if picked_favorite else game['underdog_confidence']
                    total_confidence += 1
                    confidence_alignment += 1 - abs(confidence - expected_conf) / max(confidence, expected_conf)
        
        skill_level = correct_picks / total_picks if total_picks > 0 else 0.5
        crowd_following = crowd_agreement / total_comparable if total_comparable > 0 else 0.5
        confidence_following = confidence_alignment / total_confidence if total_confidence > 0 else 0.5
        
        player_stats.append({
            'player': player,
            'skill_level': skill_level,
            'crowd_following': crowd_following,
            'confidence_following': confidence_following,
            'total_picks': total_picks
        })
    
    stats_df = pd.DataFrame(player_stats)
    
    # Normalize stats to 0-1 range
    for col in ['skill_level', 'crowd_following', 'confidence_following']:
        min_val = stats_df[col].min()
        max_val = stats_df[col].max()
        if max_val > min_val:
            stats_df[col] = (stats_df[col] - min_val) / (max_val - min_val)
        else:
            stats_df[col] = 0.5  # Default to middle value if no variation
    
    return stats_df

# Example usage:
if __name__ == "__main__":
    # Calculate stats for weeks 1-15
    stats = calculate_player_stats(
        league_id=6207,
        weeks=range(1, 16),
        cookies_file='cookies.txt'
    )
    
    print("\nPlayer Statistics:")
    print(stats.sort_values('skill_level', ascending=False).round(3))
    
    print("\nLeague Averages:")
    print(stats[['skill_level', 'crowd_following', 'confidence_following']].mean().round(3))
