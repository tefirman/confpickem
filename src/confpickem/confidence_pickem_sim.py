#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   confidence_pickem_sim.py
@Time    :   2024/12/07 13:38:47
@Author  :   Taylor Firman
@Version :   v0.1
@Contact :   tefirman@gmail.com
@Desc    :   Simulation script for NFL Confidence Pick 'Em groups
'''

from dataclasses import dataclass
from typing import List, Dict, Optional
import numpy as np
import pandas as pd
import scipy
from datetime import datetime

@dataclass
class Game:
    home_team: str
    away_team: str 
    vegas_win_prob: float
    crowd_home_pick_pct: float
    crowd_home_confidence: float
    crowd_away_confidence: float
    week: int
    kickoff_time: datetime  # Add kickoff time
    actual_outcome: Optional[bool] = None
    picks_locked: bool = False  # Whether picks have been revealed

@dataclass 
class Player:
    name: str
    skill_level: float  # 0-1 scale
    crowd_following: float  # 0-1 scale
    confidence_following: float  # 0-1 scale

class ConfidencePickEmSimulator:
    def __init__(self, num_sims: int = 10000):
        self.num_sims = num_sims
        self.games: List[Game] = []
        self.players: List[Player] = []
        
    def add_games_from_dataframe(self, games_df: pd.DataFrame):
        """Load games from a pandas DataFrame for efficient bulk loading"""
        for _, row in games_df.iterrows():
            self.games.append(Game(
                home_team=row['home_team'],
                away_team=row['away_team'],
                vegas_win_prob=row['vegas_win_prob'],
                crowd_home_pick_pct=row['crowd_home_pick_pct'],
                crowd_home_confidence=row['crowd_home_confidence'],
                crowd_away_confidence=row['crowd_away_confidence'],
                week=row['week'],
                kickoff_time=row['kickoff_time'],
                actual_outcome=row.get('actual_outcome', None)
            ))

    def simulate_picks(self, fixed_picks: Dict[str, int] = {}) -> pd.DataFrame:
        """Vectorized simulation of all picks and confidence points"""
        num_games = len(self.games)
        num_players = len(self.players)
        
        # Create matrices for efficient computation
        vegas_probs = np.array([g.vegas_win_prob for g in self.games])
        crowd_pcts = np.array([g.crowd_home_pick_pct for g in self.games])
        crowd_home_conf = np.array([g.crowd_home_confidence for g in self.games])
        crowd_away_conf = np.array([g.crowd_away_confidence for g in self.games])
        
        # Player characteristics matrices
        skill_levels = np.array([p.skill_level for p in self.players])
        crowd_following = np.array([p.crowd_following for p in self.players])
        conf_following = np.array([p.confidence_following for p in self.players])
        
        # Simulate picks for all players at once
        picks = np.zeros((self.num_sims, num_players, num_games))
        confidence = np.zeros((self.num_sims, num_players, num_games))
        
        for sim in range(self.num_sims):
            # Calculate pick probabilities
            base_probs = vegas_probs.reshape(1, -1) * (1 - crowd_following.reshape(-1, 1)) + \
                        crowd_pcts.reshape(1, -1) * crowd_following.reshape(-1, 1)
            
            # Add skill-based noise
            noise = np.random.normal(0, 0.1, (num_players, num_games)) * \
                   (1 - skill_levels.reshape(-1, 1))
            pick_probs = np.clip(base_probs + noise, 0.1, 0.9)
            
            # Generate picks
            picks[sim] = np.random.random((num_players, num_games)) < pick_probs
            
            # Calculate confidence points
            for p in range(num_players):
                player_picks = picks[sim, p]
                
                # Get relevant confidence values based on picks
                chosen_conf = np.where(player_picks, 
                                     crowd_home_conf,
                                     crowd_away_conf)
                opposing_conf = np.where(player_picks,
                                       crowd_away_conf, 
                                       crowd_home_conf)
                
                # Calculate confidence scores
                conf_diff = (chosen_conf - opposing_conf) / (chosen_conf + opposing_conf)
                vegas_conf = np.abs(vegas_probs - 0.5) * 2 * num_games
                
                # Blend signals
                blended_conf = chosen_conf * (1 + conf_diff) * conf_following[p] + \
                             vegas_conf * (1 - conf_following[p])
                
                # Add skill-based noise
                noise = np.random.normal(0, 2, num_games) * (1 - skill_levels[p])
                final_conf = np.clip(blended_conf + noise, 1, num_games)
                
                # Convert to ranks
                confidence[sim, p] = num_games + 1 - scipy.stats.rankdata(final_conf)
        
        # Convert to pandas dataframe
        results = []
        for sim in range(self.num_sims):
            for p_idx, player in enumerate(self.players):
                for g_idx, game in enumerate(self.games):
                    results.append({
                        'simulation': sim,
                        'player': player.name,
                        'week': game.week,
                        'game': f"{game.away_team}@{game.home_team}",
                        'picked_home': picks[sim, p_idx, g_idx],
                        'confidence': confidence[sim, p_idx, g_idx]
                    })
        picks_df = pd.DataFrame(results)

        # Update picks for the first player with provided fixed picks
        # Might further generalize this later, but not now...
        mask = (picks_df.player == self.players[0].name)
        for team, pts in fixed_picks.items():
            game_mask = picks_df[mask].game.str.contains(team)
            is_home = (picks_df.loc[mask & game_mask, 'game'].str.split('@').str[1] == team).astype(float)
            picks_df.loc[mask & game_mask, 'picked_home'] = is_home
            picks_df.loc[mask & game_mask, 'confidence'] = float(pts)
        
        return picks_df

    def simulate_outcomes(self) -> np.ndarray:
        """Simulate game outcomes efficiently"""
        vegas_probs = np.array([g.vegas_win_prob for g in self.games])
        actual_outcomes = np.array([g.actual_outcome for g in self.games])
        
        # Use actual outcomes where available
        outcomes = np.zeros((self.num_sims, len(self.games)))
        for i, outcome in enumerate(actual_outcomes):
            if outcome is not None:
                outcomes[:, i] = outcome
            else:
                outcomes[:, i] = np.random.random(self.num_sims) < vegas_probs[i]
                
        return outcomes

    def analyze_results(self, picks_df: pd.DataFrame, outcomes: np.ndarray) -> Dict:
        """Analyze simulation results and compute statistics with tiebreaker handling"""
        # Calculate points for each simulation
        picks_df['correct'] = picks_df.apply(
            lambda x: outcomes[x.simulation, 
                    [g.away_team + '@' + g.home_team for g in self.games].index(x.game)] == x.picked_home,
            axis=1
        )
        picks_df['points'] = picks_df.correct * picks_df.confidence
        
        # Aggregate results by simulation and player
        by_sim = picks_df.groupby(['simulation', 'player'])['points'].sum().reset_index()
        
        # Calculate win percentages with tiebreaker
        win_pcts = []
        for player in by_sim['player'].unique():
            wins = 0
            for sim in by_sim['simulation'].unique():
                sim_results = by_sim[by_sim['simulation'] == sim]
                max_points = sim_results['points'].max()
                
                # Find players tied for first
                tied_players = sim_results[sim_results['points'] == max_points]['player'].values
                
                if len(tied_players) == 1:
                    # Clear winner
                    wins += 1 if tied_players[0] == player else 0
                else:
                    # Tie - randomly select winner using consistent seed for reproducibility
                    np.random.seed(sim)  # Use simulation number as seed
                    winner = np.random.choice(tied_players)
                    wins += 1 if winner == player else 0
                    np.random.seed(None)  # Reset seed
            
            win_pct = wins / self.num_sims
            win_pcts.append({'player': player, 'win_pct': win_pct})
        
        win_pct_series = pd.Series(
            [x['win_pct'] for x in win_pcts],
            index=[x['player'] for x in win_pcts]
        )
        
        # Calculate other statistics
        stats = {
            'expected_points': by_sim.groupby('player')['points'].mean(),
            'point_std': by_sim.groupby('player')['points'].std(),
            'win_pct': win_pct_series,
            'value_at_risk': by_sim.groupby('player')['points'].quantile(0.05)
        }
        
        return stats

    def simulate_all(self, fixed_picks: Dict[str, int] = {}):
        # Run focused simulation
        picks_df = self.simulate_picks(fixed_picks)
        
        # Simulate outcomes and analyze results
        outcomes = self.simulate_outcomes()
        stats = self.analyze_results(picks_df, outcomes)
        return stats

    def optimize_picks(self, fixed_picks: Dict[str, int] = None) -> Dict[str, int]:
        """Optimize picks using simulation results"""
        # Initialize variables
        optimal = {}
        if fixed_picks is None:
            fixed_picks = {}
        
        # Track which points have been used
        used_points = set(fixed_picks.values())
        available_points = set(range(1, len(self.games) + 1)) - used_points
        
        # Start with fixed picks
        optimal.update(fixed_picks)
        
        # Sort games by confidence (most certain to least certain)
        remaining_games = [g for g in sorted(self.games, 
                                        key=lambda g: abs(g.vegas_win_prob - 0.5),
                                        reverse=True)
                        if g.home_team not in fixed_picks 
                        and g.away_team not in fixed_picks]
        
        # Assign picks for remaining games
        for game in remaining_games:
            if not available_points:  # Safety check
                break
            
            print(game)

            # Get highest available point value
            current_points = max(available_points)
            
            # Simulate both options
            home_picks = optimal.copy()
            home_picks[game.home_team] = current_points
            home_results = self.simulate_all(home_picks)
            away_picks = optimal.copy()
            away_picks[game.away_team] = current_points
            away_results = self.simulate_all(away_picks)
            
            print(home_results)
            print(away_results)

            # Choose better option
            if home_results['expected_points'][self.players[0].name] > away_results['expected_points'][self.players[0].name]:
                optimal[game.home_team] = current_points
            else:
                optimal[game.away_team] = current_points
            
            print(optimal)
            
            # Remove used points value
            available_points.remove(current_points)
        
        return optimal
