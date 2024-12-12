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
from typing import List, Dict, Optional, Tuple
import numpy as np
import pandas as pd
import scipy

# BRING THESE ALL TOGETHER!!!

@dataclass
class Game:
    home_team: str
    away_team: str 
    vegas_win_prob: float
    crowd_home_pick_pct: float
    crowd_home_confidence: float
    crowd_away_confidence: float
    week: int
    actual_outcome: Optional[bool] = None

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
                actual_outcome=row.get('actual_outcome', None)
            ))

    def simulate_picks(self) -> pd.DataFrame:
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
        
        return self._create_results_dataframe(picks, confidence)

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

    def _create_results_dataframe(self, picks: np.ndarray, confidence: np.ndarray) -> pd.DataFrame:
        """Convert simulation results to DataFrame for analysis"""
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
                    
        return pd.DataFrame(results)

    def analyze_results(self, picks_df: pd.DataFrame, outcomes: np.ndarray) -> Dict:
        """Analyze simulation results and compute statistics"""
        # Calculate points for each simulation
        picks_df['correct'] = picks_df.apply(
            lambda x: outcomes[x.simulation, 
                    [g.away_team + '@' + g.home_team for g in self.games].index(x.game)] == x.picked_home,
            axis=1
        )
        picks_df['points'] = picks_df.correct * picks_df.confidence
        
        # Aggregate results
        by_sim = picks_df.groupby(['simulation', 'player'])['points'].sum().reset_index()
        
        # Calculate win percentages more safely
        win_pcts = []
        for player in by_sim['player'].unique():
            player_points = by_sim[by_sim['player'] == player]['points'].values
            max_points_per_sim = by_sim.groupby('simulation')['points'].max().values
            win_pct = np.mean(player_points == max_points_per_sim)
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
                
            # Get highest available point value
            current_points = max(available_points)
            
            # Simulate both options
            home_results = self._simulate_with_pick(game.home_team, current_points)
            away_results = self._simulate_with_pick(game.away_team, current_points)
            
            # Choose better option
            if home_results['expected_value'] > away_results['expected_value']:
                optimal[game.home_team] = current_points
            else:
                optimal[game.away_team] = current_points
                
            # Remove used points value
            available_points.remove(current_points)
        
        return optimal

    def _simulate_with_pick(self, team: str, points: int) -> Dict:
        """Helper method to simulate outcomes with a specific pick"""
        # Run focused simulation
        picks_df = self.simulate_picks()
        outcomes = self.simulate_outcomes()
        
        # Find rows to update
        mask = (picks_df.player == self.players[0].name) & picks_df.game.str.contains(team)
        
        # Calculate if team is home team and convert to float
        is_home = picks_df.loc[mask, 'game'].apply(lambda x: team in x.split('@')[1])
        picks_df.loc[mask, 'picked_home'] = is_home.astype(float)
        picks_df.loc[mask, 'confidence'] = float(points)
        
        stats = self.analyze_results(picks_df, outcomes)
        
        return {
            'expected_value': stats['expected_points'][self.players[0].name],
            'win_pct': stats['win_pct'][self.players[0].name]
        }
