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
        if num_sims <= 0:
            raise ValueError("Number of simulations must be positive")
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

    def simulate_picks(self, fixed_picks: Dict[str, Dict[str, int]] = None, 
                      player_data: pd.DataFrame = None) -> pd.DataFrame:
        """Vectorized simulation of all picks and confidence points
        
        Args:
            fixed_picks: Dictionary mapping player names to their fixed picks.
                Structure: {
                    'Player Name': {
                        'SF': 16,  # Team abbreviation -> confidence points
                        'KC': 15,
                        # etc...
                    }
                }
                If player name is not in fixed_picks, all their picks will be simulated.
                For players in fixed_picks, any teams not specified will be simulated.
            player_data: DataFrame from yahoo.players with actual pick data for completed games.
                Used to calculate each player's already-used confidence levels.
        """
        num_games = len(self.games)
        if num_games == 0:
            raise ValueError("No games loaded for simulation")
        num_players = len(self.players)
        if num_players == 0:
            raise ValueError("No players loaded for simulation")
        
        # Calculate each player's used confidence levels from completed games
        player_used_confidence = {}
        if player_data is not None:
            for _, player_row in player_data.iterrows():
                player_name = player_row['player_name']
                used_confidence = set()
                
                # Check each game to see if it's completed and get confidence used
                for i, game in enumerate(self.games):
                    if game.actual_outcome is not None:  # Completed game
                        game_num = i + 1
                        pick = player_row.get(f'game_{game_num}_pick')
                        confidence = player_row.get(f'game_{game_num}_confidence')
                        
                        if pick and confidence:
                            used_confidence.add(int(confidence))
                
                player_used_confidence[player_name] = used_confidence
        fixed_picks = fixed_picks or {}
        
        # Validate fixed picks
        for player_name, picks in fixed_picks.items():
            # Validate player exists
            if player_name not in [p.name for p in self.players]:
                raise ValueError(f"Fixed picks specified for unknown player: {player_name}")
            
            # Validate teams exist
            game_teams = set()
            for game in self.games:
                game_teams.add(game.home_team)
                game_teams.add(game.away_team)
                
            for team in picks.keys():
                if team not in game_teams:
                    raise ValueError(f"Fixed pick specified for unknown team: {team}")
                    
            # Validate confidence points
            for points in picks.values():
                if not isinstance(points, int) or points < 1 or points > num_games:
                    raise ValueError(f"Invalid confidence points: {points}. Must be between 1 and {num_games}")

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
            # pick_probs = np.clip(base_probs + noise, 0.1, 0.9)
            pick_probs = np.clip(base_probs + noise, 0.0, 1.0)
            
            # Generate picks
            picks[sim] = np.random.random((num_players, num_games)) < pick_probs
            
            # Calculate confidence points
            for p, player in enumerate(self.players):
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
                
                # Get available confidence levels for this player (accounting for completed games)
                player_name = player.name
                used_conf = player_used_confidence.get(player_name, set())
                available_conf = set(range(1, num_games + 1)) - used_conf
                
                # Only rank games that haven't been completed (actual_outcome is None)
                incomplete_game_indices = [i for i, game in enumerate(self.games) if game.actual_outcome is None]
                
                if len(incomplete_game_indices) > 0 and len(available_conf) >= len(incomplete_game_indices):
                    # Get confidence scores only for incomplete games
                    incomplete_conf_scores = final_conf[incomplete_game_indices]
                    
                    # Create stable ranking using both confidence scores and game indices for tie-breaking
                    # This ensures deterministic behavior when confidence scores are tied
                    score_index_pairs = list(zip(-incomplete_conf_scores, incomplete_game_indices))  # Negative for descending
                    score_index_pairs.sort()  # Sort by confidence (desc), then by game index (asc) for ties
                    
                    available_conf_list = sorted(list(available_conf), reverse=True)  # Highest first
                    
                    # Initialize confidence array with zeros
                    game_confidences = np.zeros(num_games)
                    
                    # Assign confidence levels to incomplete games only
                    for rank, (_, game_idx) in enumerate(score_index_pairs):
                        if rank < len(available_conf_list):
                            game_confidences[game_idx] = available_conf_list[rank]
                    
                    confidence[sim, p] = game_confidences
                else:
                    # Fallback: use traditional ranking if no completed games or no available confidence
                    confidence[sim, p] = num_games + 1 - scipy.stats.rankdata(final_conf)
        
        # Convert to pandas dataframe
        results = []
        for sim in range(self.num_sims):
            for p_idx, player in enumerate(self.players):
                # Handle fixed picks for this player if they exist
                if player.name in fixed_picks:
                    player_fixed = fixed_picks[player.name]
                    
                    # Track available points to maintain valid confidence distribution
                    used_points = set()
                    
                    # Apply fixed picks
                    for g_idx, game in enumerate(self.games):
                        game_id = f"{game.away_team}@{game.home_team}"
                        
                        # Check if either team in this game has fixed picks
                        fixed_home = game.home_team in player_fixed
                        fixed_away = game.away_team in player_fixed
                        
                        if fixed_home or fixed_away:
                            if fixed_home:
                                team = game.home_team
                                picks[sim, p_idx, g_idx] = True
                            else:
                                team = game.away_team
                                picks[sim, p_idx, g_idx] = False
                                
                            pts = player_fixed[team]
                            confidence[sim, p_idx, g_idx] = pts
                            used_points.add(pts)
                    
                    # Adjust remaining confidence points to be valid
                    available_points = set(range(1, num_games + 1)) - used_points
                    remaining_indices = [i for i in range(num_games) 
                                    if confidence[sim, p_idx, i] not in used_points]
                    
                    if remaining_indices:
                        remaining_points = sorted(list(available_points), reverse=True)
                        remaining_confidence = confidence[sim, p_idx, remaining_indices]
                        rank_order = np.argsort(-remaining_confidence)
                        
                        for rank, idx in enumerate(rank_order):
                            confidence[sim, p_idx, remaining_indices[idx]] = remaining_points[rank]
                
                # Add results for this player
                for g_idx, game in enumerate(self.games):
                    game_id = f"{game.away_team}@{game.home_team}"
                    results.append({
                        'simulation': sim,
                        'player': player.name,
                        'week': game.week,
                        'game': game_id,
                        'picked_home': picks[sim, p_idx, g_idx],
                        'confidence': confidence[sim, p_idx, g_idx]
                    })
        
        return pd.DataFrame(results)

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

    def simulate_all(self, fixed_picks: Dict[str, int] = {}, player_data: pd.DataFrame = None):
        # Run focused simulation
        picks_df = self.simulate_picks(fixed_picks, player_data)
        
        # Simulate outcomes and analyze results
        outcomes = self.simulate_outcomes()
        stats = self.analyze_results(picks_df, outcomes)
        return stats

    def optimize_picks(self, player_name: str, fixed_picks: Dict[str, Dict[str, int]] = None, 
                    confidence_range: int = 3, available_points: set = None) -> Dict[str, int]:
        """Optimize picks using simulation results for a specific player.

        Args:
            player_name: Name of the player to optimize picks for
            fixed_picks: Dictionary mapping player names to their fixed picks
            confidence_range: Number of confidence values to explore for each game
            available_points: Set of confidence points available to use (if None, auto-calculate)

        Returns:
            Dict mapping team abbreviations to optimal confidence points
        """
        # Set consistent random seed for deterministic optimization
        np.random.seed(51)
        
        # Validate player exists
        if player_name not in [p.name for p in self.players]:
            raise ValueError(f"Unknown player: {player_name}")
            
        optimal = {}
        if fixed_picks is None:
            fixed_picks = {}
        
        # Get current player's fixed picks if they exist
        player_fixed = fixed_picks.get(player_name, {})
        optimal.update(player_fixed)

        # Track which points have been used
        used_points = list(player_fixed.values())
        
        # Validate no duplicate confidence points in fixed picks
        if len(used_points) != len(set(used_points)):
            raise ValueError("Fixed picks cannot have duplicate confidence points")
        
        used_points = set(used_points)
        
        # Use provided available_points or auto-calculate
        if available_points is None:
            # Use full confidence range (1 to total games) minus any used points
            # This is correct because confidence points 1-16 are available regardless of completed games
            available_points = set(range(1, len(self.games) + 1)) - used_points
        else:
            # Use the provided set, but remove any already used in fixed picks
            available_points = available_points - used_points

        # Sort games by certainty (most certain to least certain)
        # Process most certain games first so they get highest confidence
        # Skip games that are already completed (have actual_outcome) or have fixed picks
        remaining_games = [g for g in sorted(self.games,
                                        key=lambda g: abs(g.vegas_win_prob - 0.5), reverse=True)
                        if g.actual_outcome is None  # Skip completed games
                        and g.home_team not in player_fixed
                        and g.away_team not in player_fixed]

        # Assign picks for remaining games
        for game in remaining_games:
            if not available_points:  # Safety check
                break
            
            # Optional debug output (can be controlled via parameter)
            print(f"\nOptimizing: {game.away_team}@{game.home_team}")

            # Track best result for this game
            best_pick = None
            best_points = None
            best_win_prob = 0

            # Get range of points to try
            if confidence_range == 1:
                # Special case: only try the highest available point
                points_to_try = [max(available_points)]
            else:
                increment = max(1, (len(available_points) - 1) // (confidence_range - 1))
                points_to_try = sorted(available_points, reverse=True)[::increment]
            print(f"  Points to try: {points_to_try}")
            # points_to_try = sorted(available_points, reverse=True)[:confidence_range]
            
            # Try each team with different confidence points
            for current_points in points_to_try:
                # Try home team pick
                home_picks = fixed_picks.copy()
                if player_name not in home_picks:
                    home_picks[player_name] = {}
                home_picks[player_name] = optimal.copy()
                home_picks[player_name][game.home_team] = current_points
                
                # Simulate home team pick (with consistent seed)
                np.random.seed(51 + hash(f"{game.home_team}_{current_points}") % 10000)
                home_results = self.simulate_all(home_picks)
                home_prob = home_results['win_pct'][player_name]
                
                # Update best result if better (use >= with deterministic tie-breaking)
                if (home_prob > best_win_prob or 
                    (home_prob == best_win_prob and (best_pick is None or game.home_team < best_pick))):
                    best_win_prob = home_prob
                    best_pick = game.home_team
                    best_points = current_points

                # Try away team pick
                away_picks = fixed_picks.copy()
                if player_name not in away_picks:
                    away_picks[player_name] = {}
                away_picks[player_name] = optimal.copy()
                away_picks[player_name][game.away_team] = current_points
                
                # Simulate away team pick (with consistent seed)
                np.random.seed(51 + hash(f"{game.away_team}_{current_points}") % 10000)
                away_results = self.simulate_all(away_picks)
                away_prob = away_results['win_pct'][player_name]
                
                # Update best result if better (use >= with deterministic tie-breaking)
                if (away_prob > best_win_prob or 
                    (away_prob == best_win_prob and (best_pick is None or game.away_team < best_pick))):
                    best_win_prob = away_prob
                    best_pick = game.away_team
                    best_points = current_points

            # Add best pick/points combination to optimal picks
            optimal[best_pick] = best_points
            available_points.remove(best_points)

            print(f"  Chose {best_pick} with {best_points} points for win probability {best_win_prob:.4f}")
        
        return optimal
    
    def assess_game_importance(self, player_name: str, picks_df: pd.DataFrame = None, 
                            fixed_picks: Dict[str, Dict[str, int]] = None) -> pd.DataFrame:
        """
        Assess the relative importance of each game by calculating win probability
        changes between winning and losing each matchup.
        
        Args:
            player_name: Name of the player to analyze game importance for
            picks_df: DataFrame containing picks (optional - will simulate if not provided)
            fixed_picks: Dictionary mapping player names to their fixed picks.
                Structure: {
                    'Player Name': {
                        'SF': 16,  # Team abbreviation -> confidence points
                        'KC': 15,
                        # etc...
                    }
                }
                If player name is not in fixed_picks, all their picks will be simulated.
                For players in fixed_picks, any teams not specified will be simulated.
        
        Returns:
            DataFrame containing impact metrics for each game
        """
        # Generate picks if not provided
        if picks_df is None:
            picks_df = self.simulate_picks(fixed_picks if fixed_picks else {})
        
        # Get base simulation results
        outcomes = self.simulate_outcomes()
        base_stats = self.analyze_results(picks_df, outcomes)
        base_win_pct = base_stats['win_pct'][player_name]
        
        # Analyze each game
        game_impacts = []
        for game_idx, game in enumerate(self.games):
            game_id = f"{game.away_team}@{game.home_team}"
            
            # Get this player's pick for this game
            player_pick = picks_df[
                (picks_df.player == player_name) & 
                (picks_df.game == game_id)
            ].iloc[0]
            
            # Create forced outcome array
            forced_win = outcomes.copy()
            forced_win[:, game_idx] = player_pick.picked_home
            forced_loss = outcomes.copy()
            forced_loss[:, game_idx] = not player_pick.picked_home
            
            # Calculate win probabilities under each scenario
            win_stats = self.analyze_results(picks_df, forced_win)
            loss_stats = self.analyze_results(picks_df, forced_loss)
            
            win_prob = win_stats['win_pct'][player_name]
            loss_prob = loss_stats['win_pct'][player_name]

            # Check if this game has fixed picks
            is_fixed = False
            if fixed_picks and player_name in fixed_picks:
                player_fixed = fixed_picks[player_name]
                if game.home_team in player_fixed or game.away_team in player_fixed:
                    is_fixed = True
            
            game_impacts.append({
                'game': game_id,
                'points_bid': player_pick.confidence,
                'pick': game.home_team if player_pick.picked_home else game.away_team,
                'win_probability': win_prob,
                'loss_probability': loss_prob,
                'win_delta': win_prob - base_win_pct,
                'loss_delta': loss_prob - base_win_pct,
                'total_impact': win_prob - loss_prob,
                'is_fixed': is_fixed
            })
        
        results = pd.DataFrame(game_impacts)
        
        # Sort by absolute impact
        results = results.sort_values('total_impact', ascending=False, key=abs)
        
        return results
    
    def assess_remaining_game_importance(self, player_name: str, current_standings: dict, 
                                       player_picks: dict) -> pd.DataFrame:
        """
        Assess the importance of remaining games based on current standings and locked-in results.
        This is designed for mid-week/Sunday analysis when some games are completed.
        
        Args:
            player_name: Name of the player to analyze
            current_standings: Dict mapping player names to current points
            player_picks: Dict mapping players to their pick dictionaries
                Example: {'Player1': {'GB': 16, 'Cin': 15, ...}, 'Player2': {...}}
        
        Returns:
            DataFrame with remaining game importance analysis
        """
        if player_name not in current_standings:
            raise ValueError(f"Player {player_name} not found in current standings")
        
        # Get remaining games (those without actual outcomes)
        remaining_games = [g for g in self.games if g.actual_outcome is None]
        
        if not remaining_games:
            # No games remaining
            return pd.DataFrame()
        
        # Current player's position
        your_current_points = current_standings[player_name]
        sorted_standings = sorted(current_standings.items(), key=lambda x: x[1], reverse=True)
        your_current_rank = next(i for i, (name, _) in enumerate(sorted_standings, 1) if name == player_name)
        
        # Calculate maximum possible points for each player from remaining games
        max_remaining_points = {}
        for p_name, picks in player_picks.items():
            # Find confidence levels used in completed games
            used_conf = set()
            for game in self.games:
                if game.actual_outcome is not None:  # Completed game
                    for team in [game.home_team, game.away_team]:
                        if team in picks:
                            used_conf.add(picks[team])
                            break
            
            # Available confidence levels for remaining games
            all_conf = set(range(1, len(self.games) + 1))
            available_conf = all_conf - used_conf
            max_remaining = sum(sorted(list(available_conf), reverse=True)[:len(remaining_games)])
            max_remaining_points[p_name] = max_remaining
        
        # Analyze each remaining game
        game_impacts = []
        
        for game_idx, game in enumerate(remaining_games):
            game_id = f"{game.away_team}@{game.home_team}"
            
            # Get your pick and confidence for this game
            your_pick = None
            your_confidence = 0
            
            if player_name in player_picks:
                picks = player_picks[player_name]
                if game.home_team in picks:
                    your_pick = game.home_team
                    your_confidence = picks[game.home_team]
                elif game.away_team in picks:
                    your_pick = game.away_team  
                    your_confidence = picks[game.away_team]
            
            if your_pick is None:
                continue  # Skip games where we don't have picks
            
            # Simulate scenarios: you win this game vs you lose this game
            win_scenario_final = your_current_points + your_confidence
            loss_scenario_final = your_current_points
            
            # Count how many players you could beat/lose to based on this game
            players_you_could_pass = 0
            players_who_could_pass_you = 0
            
            for other_name, other_current_points in current_standings.items():
                if other_name == player_name:
                    continue
                    
                other_max_possible = other_current_points + max_remaining_points.get(other_name, 0)
                
                # If you win this game, could you pass them?
                if win_scenario_final > other_max_possible and your_current_points <= other_current_points:
                    players_you_could_pass += 1
                
                # If you lose this game, could they pass you?
                if other_max_possible > loss_scenario_final and other_current_points <= your_current_points:
                    players_who_could_pass_you += 1
            
            # Calculate importance based on Vegas probability and positional impact
            vegas_win_prob = game.vegas_win_prob if your_pick == game.home_team else (1.0 - game.vegas_win_prob)
            
            # Higher importance for:
            # - Close games (uncertainty)
            # - High confidence bids
            # - Games that affect many position changes
            # - Games late in remaining schedule (fewer chances left)
            
            uncertainty_factor = 1.0 - abs(vegas_win_prob - 0.5) * 2  # 0 to 1, higher for closer games
            confidence_factor = your_confidence / 16.0  # Normalize confidence
            position_factor = (players_you_could_pass + players_who_could_pass_you) / len(current_standings)
            scarcity_factor = (len(remaining_games) - game_idx) / len(remaining_games)  # Later games more important
            
            # Combined importance score (0 to 1)
            importance_score = (uncertainty_factor * 0.3 + 
                              confidence_factor * 0.3 + 
                              position_factor * 0.2 + 
                              scarcity_factor * 0.2)
            
            game_impacts.append({
                'game': game_id,
                'pick': your_pick,
                'points_bid': your_confidence,
                'vegas_win_prob': vegas_win_prob,
                'uncertainty_factor': uncertainty_factor,
                'confidence_factor': confidence_factor, 
                'position_factor': position_factor,
                'scarcity_factor': scarcity_factor,
                'importance_score': importance_score,
                'players_could_pass': players_you_could_pass,
                'players_could_pass_you': players_who_could_pass_you,
                'current_rank': your_current_rank
            })
        
        results = pd.DataFrame(game_impacts)
        
        if len(results) > 0:
            # Sort by importance score
            results = results.sort_values('importance_score', ascending=False)
        
        return results
