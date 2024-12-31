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

    def simulate_picks_for_game(self, game: pd.Series, player_names: list) -> pd.DataFrame:
        """
        Simulate picks for a single future game for specified players
        
        Args:
            game: Series containing game information (home_team, away_team, etc)
            player_names: List of player names to simulate picks for
            
        Returns:
            DataFrame containing simulated picks in standard format
        """
        # Create game identifier
        game_id = f"{game.away_team}@{game.home_team}"
        
        picks_list = []
        
        # Get player characteristics
        player_dict = {p.name: p for p in self.players}
        
        for player_name in player_names:
            player = player_dict.get(player_name)
            if not player:
                # If player not found in characteristics, use average values
                skill_level = 0.5
                crowd_following = 0.5
                conf_following = 0.5
            else:
                skill_level = player.skill_level
                crowd_following = player.crowd_following
                conf_following = player.confidence_following
                
            # Calculate pick probability
            base_prob = game.vegas_win_prob * (1 - crowd_following) + \
                    game.crowd_home_pick_pct * crowd_following
            
            # Add skill-based noise
            noise = np.random.normal(0, 0.1) * (1 - skill_level)
            pick_prob = np.clip(base_prob + noise, 0.1, 0.9)
            
            # Generate picks for each simulation
            for sim in range(self.num_sims):
                # Determine if home team picked
                picked_home = np.random.random() < pick_prob
                
                # Calculate confidence
                if picked_home:
                    chosen_conf = game.crowd_home_confidence
                    opposing_conf = game.crowd_away_confidence
                else:
                    chosen_conf = game.crowd_away_confidence
                    opposing_conf = game.crowd_home_confidence
                    
                # Calculate confidence score
                conf_diff = (chosen_conf - opposing_conf) / (chosen_conf + opposing_conf)
                vegas_conf = abs(game.vegas_win_prob - 0.5) * 2 * len(self.games)
                
                # Blend signals
                blended_conf = chosen_conf * (1 + conf_diff) * conf_following + \
                            vegas_conf * (1 - conf_following)
                
                # Add skill-based noise
                noise = np.random.normal(0, 2) * (1 - skill_level)
                final_conf = int(np.clip(blended_conf + noise, 1, len(self.games)))
                
                picks_list.append({
                    'simulation': sim,
                    'player': player_name,
                    'week': game.week,
                    'game': game_id,
                    'picked_home': picked_home,
                    'confidence': final_conf
                })
        
        return pd.DataFrame(picks_list)

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
            home_results = self._simulate_with_pick(game.home_team, current_points)
            away_results = self._simulate_with_pick(game.away_team, current_points)
            
            print(home_results)
            print(away_results)

            # Choose better option
            if home_results['expected_value'] > away_results['expected_value']:
                optimal[game.home_team] = current_points
            else:
                optimal[game.away_team] = current_points
            
            print(optimal)
            
            # Remove used points value
            available_points.remove(current_points)
        
        return optimal

    def optimize_picks_comprehensive(self, fixed_picks: Dict[str, int] = None, point_buckets: int = 4) -> Dict[str, int]:
        """
        Optimize picks by simulating possibilities with bucketed point values for efficiency.
        
        Args:
            fixed_picks: Dictionary mapping team names to point values for picks that should not be changed
            point_buckets: Number of buckets to divide point values into (excluding fixed picks)
            
        Returns:
            Dictionary mapping team names to optimal point values
        """
        # Initialize variables
        optimal = {}
        if fixed_picks is None:
            fixed_picks = {}
        
        # Start with fixed picks
        optimal.update(fixed_picks)
        
        # Track available points
        used_points = set(fixed_picks.values())
        available_points = set(range(1, len(self.games) + 1)) - used_points
        
        # Create point value buckets for remaining points
        sorted_points = sorted(list(available_points))
        points_per_bucket = max(1, len(sorted_points) // point_buckets)
        point_groups = [sorted_points[i:i + points_per_bucket] 
                    for i in range(0, len(sorted_points), points_per_bucket)]
        
        # Get remaining games that need picks
        remaining_games = [g for g in self.games 
                        if g.home_team not in fixed_picks 
                        and g.away_team not in fixed_picks]
        
        # For each remaining pick slot
        while remaining_games and available_points:
            best_ev = float('-inf')
            best_pick = None
            best_points = None
            
            # Try each remaining game
            for game in remaining_games:
                # Try representative points from each bucket
                for point_group in point_groups:
                    # Use median value from bucket as representative
                    points = point_group[len(point_group)//2]
                    if points not in available_points:
                        continue
                    
                    # Try home team
                    home_results = self._simulate_with_pick(game.home_team, points, optimal)
                    if home_results['win_pct'] > best_ev:
                        best_ev = home_results['win_pct']
                        best_pick = game.home_team
                        best_points = points
                    
                    # Try away team
                    away_results = self._simulate_with_pick(game.away_team, points, optimal)
                    if away_results['win_pct'] > best_ev:
                        best_ev = away_results['win_pct']
                        best_pick = game.away_team 
                        best_points = points
                    
                    print(f"Selected {best_pick} with {best_points} points (EV: {best_ev:.3f}), {datetime.now()}")
            
            # Fine-tune point value around best bucket
            if best_points is not None:
                current_bucket = next(group for group in point_groups if best_points in group)
                nearby_points = [p for p in available_points 
                            if abs(p - best_points) <= max(2, points_per_bucket//2)]
                
                for points in nearby_points:
                    results = self._simulate_with_pick(best_pick, points, optimal)
                    if results['win_pct'] > best_ev:
                        best_ev = results['win_pct']
                        best_points = points
                    
                    print(f"Selected {best_pick} with {best_points} points (EV: {best_ev:.3f}), {datetime.now()}")
            
            # Update optimal picks with best option found
            optimal[best_pick] = best_points
            available_points.remove(best_points)
            remaining_games = [g for g in remaining_games 
                            if g.home_team != best_pick
                            and g.away_team != best_pick]
            
            print(f"Finally selected {best_pick} with {best_points} points (EV: {best_ev:.3f}), {datetime.now()}")
            print(optimal)
        
        return optimal

    def _simulate_with_pick(self, team: str, points: int, current_picks: Dict[str, int]) -> Dict:
        """
        Helper method to simulate outcomes with a specific pick added to current picks
        
        Args:
            team: Team to simulate picking
            points: Point value to assign
            current_picks: Dictionary of picks already made
            
        Returns:
            Dictionary containing expected value and win percentage
        """
        # Create temporary picks dict with new pick added
        test_picks = current_picks.copy()
        test_picks[team] = points
        
        # Run focused simulation
        picks_df = self.simulate_picks()
        
        # Update picks for first player (the one we're optimizing for)
        mask = (picks_df.player == self.players[0].name)
        
        # Find the game containing our team and update the pick
        game_mask = picks_df[mask].game.str.contains(team)
        is_home = (picks_df.loc[mask & game_mask, 'game'].str.split('@').str[1] == team).astype(float)
        picks_df.loc[mask & game_mask, 'picked_home'] = is_home
        picks_df.loc[mask & game_mask, 'confidence'] = float(points)
        
        # Update other picks based on current_picks
        for team, pts in current_picks.items():
            game_mask = picks_df[mask].game.str.contains(team)
            is_home = (picks_df.loc[mask & game_mask, 'game'].str.split('@').str[1] == team).astype(float)
            picks_df.loc[mask & game_mask, 'picked_home'] = is_home
            picks_df.loc[mask & game_mask, 'confidence'] = float(pts)
        
        # Simulate outcomes and analyze results
        outcomes = self.simulate_outcomes()
        stats = self.analyze_results(picks_df, outcomes)
        
        return {
            'expected_value': stats['expected_points'][self.players[0].name],
            'win_pct': stats['win_pct'][self.players[0].name]
        }
