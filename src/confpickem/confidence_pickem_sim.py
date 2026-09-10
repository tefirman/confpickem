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
                # First, apply actual picks from completed games for all players
                if player_data is not None:
                    player_row = player_data[player_data['player_name'] == player.name]
                    if not player_row.empty:
                        player_row = player_row.iloc[0]
                        for g_idx, game in enumerate(self.games):
                            if game.actual_outcome is not None:  # Completed game
                                game_num = g_idx + 1
                                pick_col = f'game_{game_num}_pick'
                                conf_col = f'game_{game_num}_confidence'

                                if pick_col in player_data.columns and conf_col in player_data.columns:
                                    picked_team = player_row[pick_col]
                                    conf_value = player_row[conf_col]

                                    if pd.notna(picked_team) and pd.notna(conf_value) and conf_value > 0:
                                        # Set the pick
                                        picks[sim, p_idx, g_idx] = (picked_team == game.home_team)
                                        # Set the confidence
                                        confidence[sim, p_idx, g_idx] = int(conf_value)

                # Handle fixed picks for this player if they exist (for remaining games)
                if player.name in fixed_picks:
                    player_fixed = fixed_picks[player.name]

                    # Pinned game indices -> point value. Includes completed games
                    # applied above (their confidence is already spent) so the repack
                    # below never reuses those point values or reshuffles those slots.
                    pinned_points = {}
                    for g_idx, game in enumerate(self.games):
                        if game.actual_outcome is not None and confidence[sim, p_idx, g_idx] > 0:
                            pinned_points[g_idx] = int(confidence[sim, p_idx, g_idx])

                    # Apply this player's fixed picks
                    for g_idx, game in enumerate(self.games):
                        # Check if either team in this game has a fixed pick
                        fixed_home = game.home_team in player_fixed
                        fixed_away = game.away_team in player_fixed

                        if fixed_home or fixed_away:
                            if fixed_home:
                                picks[sim, p_idx, g_idx] = True
                                pts = player_fixed[game.home_team]
                            else:
                                picks[sim, p_idx, g_idx] = False
                                pts = player_fixed[game.away_team]

                            confidence[sim, p_idx, g_idx] = pts
                            pinned_points[g_idx] = pts

                    # Repack every non-pinned game with the leftover point values,
                    # ranked by the (noisy) confidence scores from the block above.
                    # Tracking pinned games *by index* (not by value) guarantees the
                    # result is a valid permutation of 1..num_games even when a fixed
                    # point value collides with one already sitting in a free slot.
                    free_indices = [i for i in range(num_games) if i not in pinned_points]
                    if free_indices:
                        leftover_points = sorted(
                            set(range(1, num_games + 1)) - set(pinned_points.values()),
                            reverse=True,
                        )
                        free_scores = confidence[sim, p_idx, free_indices]
                        rank_order = np.argsort(-free_scores)  # highest score first
                        for rank, idx in enumerate(rank_order):
                            confidence[sim, p_idx, free_indices[idx]] = leftover_points[rank]
                
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
                    confidence_range: int = 3, available_points: set = None,
                    player_data: pd.DataFrame = None) -> Dict[str, int]:
        """Optimize picks using simulation results for a specific player.

        Args:
            player_name: Name of the player to optimize picks for
            fixed_picks: Dictionary mapping player names to their fixed picks
            confidence_range: Number of confidence values to explore for each game
            available_points: Set of confidence points available to use (if None, auto-calculate)
            player_data: DataFrame containing actual player picks for completed games

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
                home_results = self.simulate_all(home_picks, player_data=player_data)
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
                away_results = self.simulate_all(away_picks, player_data=player_data)
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

    def _player_game_pick(self, player_row: pd.Series, game_idx: int,
                          game: Game) -> Optional[Tuple[bool, int]]:
        """``(pick_home, points)`` for one player on one game from ``yahoo.players``
        row data, or ``None`` if they have no recorded pick there."""
        pick = player_row.get(f'game_{game_idx + 1}_pick')
        conf = player_row.get(f'game_{game_idx + 1}_confidence')
        if pick is None or conf is None or pd.isna(pick) or pd.isna(conf):
            return None
        try:
            conf = int(conf)
        except (TypeError, ValueError):
            return None
        if conf <= 0:
            return None
        return (pick == game.home_team), conf

    def _build_analytic_field(self, player_name: str, n_outcomes: int, seed: int,
                              player_data: pd.DataFrame = None,
                              as_of: datetime = None,
                              max_opponent_types: int = 16):
        """Shared setup for the analytical ``P(win)`` methods.

        Builds the modeled-opponent field, the sampled outcome draws and the
        ``pwin`` closure, and reports which games are frozen (finished or
        kicked-off-but-live) plus this player's real pick/confidence on them.

        Returns a dict with keys: ``games``, ``n``, ``vegas_home``, ``rng``,
        ``outcomes``, ``pwin``, ``completed`` (bool list), ``frozen`` (bool
        list), ``my_frozen`` (``{game_idx: (pick_home, points)}``).
        """
        from . import analytical as _an

        if player_name not in [p.name for p in self.players]:
            raise ValueError(f"Unknown player: {player_name}")

        games = self.games
        n = len(games)
        if n == 0:
            raise ValueError("No games loaded")

        vegas_home = np.array([g.vegas_win_prob for g in games])
        crowd_home_pct = np.array([g.crowd_home_pick_pct for g in games])
        crowd_home_conf = np.array([g.crowd_home_confidence for g in games])
        crowd_away_conf = np.array([g.crowd_away_confidence for g in games])

        completed = [g.actual_outcome is not None for g in games]

        def _kicked_off(g):
            """Has ``g`` started as of ``as_of``? Robust to naive/aware and
            pandas.Timestamp kickoff times."""
            if as_of is None or g.kickoff_time is None:
                return False
            kt = g.kickoff_time
            try:
                return kt.timestamp() <= as_of.timestamp()
            except (TypeError, ValueError, OverflowError, OSError):
                kt_naive = kt.replace(tzinfo=None)
                as_naive = as_of.replace(tzinfo=None)
                return kt_naive <= as_naive

        locked_pending = [
            (not completed[i]) and (g.picks_locked or _kicked_off(g))
            for i, g in enumerate(games)]
        frozen = [completed[i] or locked_pending[i] for i in range(n)]

        def _row_for(name):
            if player_data is None:
                return None
            match = player_data[player_data['player_name'] == name]
            return match.iloc[0] if not match.empty else None

        opp_names = [p.name for p in self.players if p.name != player_name]
        opponents = [(p.crowd_following, p.confidence_following)
                     for p in self.players if p.name != player_name]
        opp_completed = opp_locked = None
        if any(frozen) and player_data is not None:
            opp_completed, opp_locked = [], []
            for name in opp_names:
                row = _row_for(name)
                done, live = {}, {}
                if row is not None:
                    for i, g in enumerate(games):
                        if not frozen[i]:
                            continue
                        got = self._player_game_pick(row, i, g)
                        if got is None:
                            continue
                        (done if completed[i] else live)[i] = got
                opp_completed.append(done or None)
                opp_locked.append(live or None)

        actual_outcomes = ([g.actual_outcome for g in games]
                           if any(completed) else None)
        opp_types = _an.build_opponent_types(
            vegas_home, crowd_home_pct, crowd_home_conf, crowd_away_conf,
            opponents, completed_picks=opp_completed,
            actual_outcomes=actual_outcomes, locked_picks=opp_locked,
            max_types=max_opponent_types if any(frozen) else None)
        rng = np.random.default_rng(seed)
        outcomes = _an.sample_outcomes(vegas_home, n_outcomes, rng,
                                       actual_outcomes=actual_outcomes)
        pwin = _an.make_pwin(opp_types, outcomes)

        my_frozen = {}
        my_row = _row_for(player_name)
        if my_row is not None:
            for i, g in enumerate(games):
                if frozen[i]:
                    got = self._player_game_pick(my_row, i, g)
                    if got is not None:
                        my_frozen[i] = got

        return dict(games=games, n=n, vegas_home=vegas_home, rng=rng,
                    outcomes=outcomes, pwin=pwin, completed=completed,
                    frozen=frozen, my_frozen=my_frozen)

    def optimize_picks_analytic(self, player_name: str,
                                fixed_picks: Dict[str, Dict[str, int]] = None,
                                iterations: int = 400, restarts: int = 4,
                                n_outcomes: int = 6000, seed: int = 51,
                                available_points: set = None,
                                player_data: pd.DataFrame = None,
                                as_of: datetime = None,
                                max_opponent_types: int = 16,
                                verbose: bool = False) -> Dict[str, int]:
        """Optimize picks against an exact analytical ``P(win)``.

        A weekly confidence score is a Poisson-binomial (weighted sum of
        Bernoullis), so ``P(you finish 1st)`` against a modeled field can be
        computed without Monte-Carlo noise -- see ``confpickem.analytical`` and
        ``docs/optimization-methodology.md``. This runs a random-restart hill
        climb on that objective, which in backtest beats the greedy
        :meth:`optimize_picks` on wins, "one game from winning" weeks, and mean
        finish.

        Midweek (some ``self.games`` are decided or already kicked off): pass
        ``player_data`` (``yahoo.players``) so that

        * every game whose picks are frozen -- finished, or kicked off but not
          yet final -- is locked to this player's real pick and confidence, and
          the free games are optimized over exactly the **unspent** confidence
          values;
        * the sampled outcome vectors are pinned to the real results on
          *finished* games (kicked-off-but-live games stay random);
        * each opponent's *actual* frozen picks are folded into the field model
          instead of their modal slate -- finished games as the points they
          banked, live games as a pinned pick still awaiting an outcome.

        A game counts as frozen if ``game.actual_outcome`` is set, if
        ``game.picks_locked`` is true, or if ``as_of`` is given and the game's
        ``kickoff_time`` is at or before it (the pool locks *all* picks once the
        first Sunday game starts, so mid-Sunday only a game or two is usually
        finished while the rest are frozen-but-live).

        Args:
            player_name: player to optimize for (must be in ``self.players``).
            fixed_picks: ``{player: {TEAM: confidence}}``; this player's entries
                are locked and everything else is optimized around them.
            iterations: hill-climb steps per restart.
            restarts: random restarts.
            n_outcomes: outcome-vector draws for the analytical P(win) estimate.
            seed: RNG seed (deterministic given identical inputs).
            available_points: optional set of confidence values the free games
                may use. If given it is validated against the values left
                unspent after completed games + fixed picks; normally you can
                leave it ``None`` and let ``player_data`` imply it.
            player_data: ``yahoo.players`` DataFrame -- required for midweek to
                know spent confidence and opponents' real picks.
            as_of: optional timestamp; any not-yet-final game that kicked off at
                or before it is treated as frozen (picks locked, outcome still
                live). Use ``datetime.now()`` for a live mid-Sunday run.
            max_opponent_types: midweek only -- cap on distinct modeled
                opponent types after folding in real frozen picks (each
                distinct history is otherwise its own type and ``P(win)`` slows
                by the same factor). The rarest histories past the cap are
                merged; 16 keeps evaluation fast with negligible effect on the
                chosen slate.
            verbose: print the chalk vs. optimized P(win).

        Returns:
            ``{TEAM: confidence}`` -- a full valid 1..N assignment.
        """
        from . import analytical as _an

        field = self._build_analytic_field(
            player_name, n_outcomes, seed, player_data=player_data,
            as_of=as_of, max_opponent_types=max_opponent_types)
        games, n = field['games'], field['n']
        vegas_home, rng = field['vegas_home'], field['rng']
        pwin, frozen, my_frozen = field['pwin'], field['frozen'], field['my_frozen']

        # --- this player's locked slots: fixed picks + every frozen game -----
        fixed_picks = fixed_picks or {}
        mine = fixed_picks.get(player_name, {})
        pick_home_fixed = np.zeros(n, dtype=bool)
        points_fixed = np.zeros(n, dtype=int)

        for i, (ph_i, pts_i) in my_frozen.items():
            pick_home_fixed[i], points_fixed[i] = ph_i, pts_i
        for i, g in enumerate(games):
            if g.home_team in mine:
                pick_home_fixed[i] = True
                points_fixed[i] = int(mine[g.home_team])
            elif g.away_team in mine:
                pick_home_fixed[i] = False
                points_fixed[i] = int(mine[g.away_team])

        locked_vals = points_fixed[points_fixed > 0].tolist()
        if len(locked_vals) != len(set(locked_vals)):
            raise ValueError("Locked picks (frozen games + fixed picks) reuse "
                             "a confidence value")
        unspent = set(range(1, n + 1)) - set(locked_vals)
        if available_points is not None:
            requested = set(available_points) - set(locked_vals)
            if requested != unspent:
                raise ValueError(
                    f"available_points {sorted(requested)} does not match the "
                    f"unspent confidence values {sorted(unspent)}")

        ph, pts, val = _an.optimize_slate(
            pwin, vegas_home,
            pick_home_fixed=pick_home_fixed if points_fixed.any() else None,
            points_fixed=points_fixed if points_fixed.any() else None,
            iterations=iterations, restarts=restarts, rng=rng)

        if verbose:
            cph, cpts = _an.chalk_slate(vegas_home)
            print(f"analytical P(win): chalk {pwin(cph, cpts):.4f} "
                  f"-> optimized {val:.4f}")

        return {(games[i].home_team if ph[i] else games[i].away_team): int(pts[i])
                for i in range(n)}

    def assess_game_importance(self, player_name: str, picks_df: pd.DataFrame = None,
                            fixed_picks: Dict[str, Dict[str, int]] = None,
                            player_data: pd.DataFrame = None,
                            n_outcomes: int = 6000, seed: int = 51,
                            as_of: datetime = None) -> pd.DataFrame:
        """
        Rank each game by how much its result swings your probability of
        finishing first, computed **analytically** -- no forced re-simulation.

        The player's slate (which team, how many points on each game) comes from
        ``picks_df`` if given, else from ``fixed_picks[player_name]`` plus any
        completed/locked games, else from one quick simulation. Holding that
        slate fixed, ``P(win | draw)`` is evaluated once over ``n_outcomes``
        importance-sampled game-outcome vectors (see ``confpickem.analytical``),
        and each game's importance is

            ``P(win | game i home win) - P(win | game i away win)``

        obtained by slicing those same draws on game ``i``'s outcome bit. A game
        already decided (or Vegas 0/1) has no live split -> importance 0.

        Args:
            player_name: player to analyze (must be in ``self.players``).
            picks_df: optional simulator picks DataFrame; its first simulation
                supplies this player's slate.
            fixed_picks: ``{player: {TEAM: confidence}}``; used for the slate
                when ``picks_df`` is not given, and to flag ``is_fixed``.
            player_data: ``yahoo.players`` -- lets completed/locked games be
                pinned and opponents' real picks fold into the field model.
            n_outcomes: outcome-vector draws for the analytical estimate.
            seed: RNG seed (deterministic given identical inputs).
            as_of: optional timestamp; not-yet-final games that kicked off by it
                count as locked (see :meth:`optimize_picks_analytic`).

        Returns:
            DataFrame with one row per game and columns ``game``, ``points_bid``,
            ``pick``, ``win_probability`` (P(win) if this game's pick hits),
            ``loss_probability`` (if it misses), ``win_delta`` / ``loss_delta``
            (vs. the unconditioned base), ``total_impact``
            (``win_probability - loss_probability``) and ``is_fixed``, sorted by
            ``|total_impact|`` descending.
        """
        from . import analytical as _an

        field = self._build_analytic_field(
            player_name, n_outcomes, seed, player_data=player_data, as_of=as_of)
        games, n = field['games'], field['n']
        outcomes, pwin, my_frozen = field['outcomes'], field['pwin'], field['my_frozen']

        # --- resolve this player's slate: pick_home + points per game --------
        pick_home = np.zeros(n, dtype=bool)
        points = np.zeros(n, dtype=int)
        got_all = False
        fp = (fixed_picks or {}).get(player_name, {})

        if picks_df is not None:
            first_sim = picks_df['simulation'].min() if 'simulation' in picks_df else None
            for i, game in enumerate(games):
                game_id = f"{game.away_team}@{game.home_team}"
                sel = picks_df[(picks_df.player == player_name)
                               & (picks_df.game == game_id)]
                if first_sim is not None:
                    sel = sel[sel.simulation == first_sim]
                if sel.empty:
                    break
                row = sel.iloc[0]
                pick_home[i] = bool(row.picked_home)
                points[i] = int(row.confidence)
            else:
                got_all = True

        if not got_all:
            for i, game in enumerate(games):
                if game.home_team in fp:
                    pick_home[i], points[i] = True, int(fp[game.home_team])
                elif game.away_team in fp:
                    pick_home[i], points[i] = False, int(fp[game.away_team])
                elif i in my_frozen:
                    pick_home[i], points[i] = my_frozen[i]
            if not points.all():  # slate still incomplete -> simulate one
                sim_df = self.simulate_picks(fixed_picks or {}, player_data)
                s0 = sim_df['simulation'].min()
                for i, game in enumerate(games):
                    if points[i]:
                        continue
                    game_id = f"{game.away_team}@{game.home_team}"
                    row = sim_df[(sim_df.player == player_name)
                                 & (sim_df.game == game_id)
                                 & (sim_df.simulation == s0)].iloc[0]
                    pick_home[i] = bool(row.picked_home)
                    points[i] = int(row.confidence)

        p_home_wins, p_away_wins, base = _an.game_importance(
            pwin, outcomes, pick_home, points)

        rows = []
        for i, game in enumerate(games):
            win_prob = p_home_wins[i] if pick_home[i] else p_away_wins[i]
            loss_prob = p_away_wins[i] if pick_home[i] else p_home_wins[i]
            is_fixed = bool(fp) and (game.home_team in fp or game.away_team in fp)
            rows.append({
                'game': f"{game.away_team}@{game.home_team}",
                'points_bid': int(points[i]),
                'pick': game.home_team if pick_home[i] else game.away_team,
                'win_probability': float(win_prob),
                'loss_probability': float(loss_prob),
                'win_delta': float(win_prob - base),
                'loss_delta': float(loss_prob - base),
                'total_impact': float(win_prob - loss_prob),
                'is_fixed': is_fixed,
            })

        results = pd.DataFrame(rows)
        return results.sort_values('total_impact', ascending=False, key=abs)

    def standings_analytic(self, player_data: pd.DataFrame,
                           n_outcomes: int = 6000, seed: int = 51):
        """Live league standings once every pick is locked -- no optimization.

        After the first Sunday kickoff a confidence pool locks *every* entry, so
        there is nothing left to choose: each entrant's slate is known and only
        the game outcomes are uncertain. This scores every entrant's real slate
        (from ``player_data`` == ``yahoo.players``) against ``n_outcomes``
        importance-sampled outcome vectors -- with the games already decided
        (``game.actual_outcome``) pinned to their real results -- and returns the
        current win probabilities plus how much each remaining game swings them.
        No opponent model, no Monte-Carlo pick sampling.

        Args:
            player_data: ``yahoo.players`` DataFrame -- every entrant must have a
                pick + confidence on every game (that is the point of the
                fully-locked state).
            n_outcomes: outcome-vector draws.
            seed: RNG seed (deterministic given identical inputs).

        Returns:
            ``(standings, importance)``:

            * ``standings`` -- one row per entrant: ``player``, ``locked_points``
              (already banked on decided games), ``win_pct``, ``expected_points``,
              sorted by ``win_pct`` descending.
            * ``importance`` -- one row per *undecided* game: ``game``,
              ``vegas_home_win_pct``, and ``top_swing`` = the largest
              ``|P(win | home) - P(win | away)|`` over all entrants (how much
              first place hinges on that game), sorted descending.
        """
        from . import analytical as _an

        games = self.games
        n = len(games)
        if n == 0:
            raise ValueError("No games loaded")

        names = player_data['player_name'].tolist()
        N = len(names)
        pick_home = np.zeros((N, n), dtype=bool)
        points = np.zeros((N, n), dtype=int)
        for e, name in enumerate(names):
            row = player_data.iloc[e]
            for i, g in enumerate(games):
                got = self._player_game_pick(row, i, g)
                if got is None:
                    raise ValueError(
                        f"{name!r} has no locked pick for "
                        f"{g.away_team}@{g.home_team}; standings_analytic needs "
                        f"every entrant's full slate")
                pick_home[e, i], points[e, i] = got

        vegas_home = np.array([g.vegas_win_prob for g in games])
        completed = [g.actual_outcome is not None for g in games]
        actual_outcomes = [g.actual_outcome for g in games] if any(completed) else None
        rng = np.random.default_rng(seed)
        outcomes = _an.sample_outcomes(vegas_home, n_outcomes, rng,
                                       actual_outcomes=actual_outcomes)

        win_pct, exp_pts, swing = _an.locked_board_standings(
            pick_home, points, outcomes)

        locked_points = np.array([
            sum(points[e, i] for i in range(n)
                if completed[i] and pick_home[e, i] == bool(games[i].actual_outcome))
            for e in range(N)])

        standings = pd.DataFrame({
            'player': names,
            'locked_points': locked_points,
            'win_pct': win_pct,
            'expected_points': exp_pts,
        }).sort_values('win_pct', ascending=False, ignore_index=True)

        imp_rows = []
        for i, g in enumerate(games):
            if completed[i]:
                continue
            imp_rows.append({
                'game': f"{g.away_team}@{g.home_team}",
                'vegas_home_win_pct': float(vegas_home[i]),
                'top_swing': float(np.abs(swing[:, i]).max()),
            })
        importance = pd.DataFrame(imp_rows).sort_values(
            'top_swing', ascending=False, ignore_index=True)

        return standings, importance

    def optimize_picks_hill_climb(self, player_name: str, fixed_picks: Dict[str, Dict[str, int]] = None,
                                   iterations: int = 1000, restarts: int = 10,
                                   available_points: set = None, player_data: pd.DataFrame = None,
                                   top_n: int = 1000) -> Tuple[Dict[str, int], pd.DataFrame]:
        """Optimize picks using hill climbing with random restarts.

        This is a local search optimization that explores the solution space more thoroughly
        than the greedy sequential approach. It can find better global optima by:
        1. Starting from random or greedy initial solutions
        2. Making small iterative improvements (swapping teams, swapping confidence)
        3. Restarting multiple times to avoid local minima

        Args:
            player_name: Name of the player to optimize picks for
            fixed_picks: Dictionary mapping player names to their fixed picks
            iterations: Number of hill climbing iterations per restart
            restarts: Number of random restarts
            available_points: Set of confidence points available to use (if None, auto-calculate)
            player_data: DataFrame with actual player picks for completed games
            top_n: Number of top combinations to analyze for summary statistics

        Returns:
            Tuple of (optimal_picks, summary_stats) where:
            - optimal_picks: Dict mapping team abbreviations to optimal confidence points
            - summary_stats: DataFrame with frequency and average points for each team in top N solutions
        """
        # Set consistent random seed for deterministic optimization
        np.random.seed(42)

        # Store player_data for use in evaluations
        self._optimization_player_data = player_data

        # Validate player exists
        if player_name not in [p.name for p in self.players]:
            raise ValueError(f"Unknown player: {player_name}")

        if fixed_picks is None:
            fixed_picks = {}

        # Get current player's fixed picks if they exist
        player_fixed = fixed_picks.get(player_name, {})

        # Track which points have been used in fixed picks
        used_points = set(player_fixed.values())

        # Validate no duplicate confidence points in fixed picks
        if len(player_fixed) != len(used_points):
            raise ValueError("Fixed picks cannot have duplicate confidence points")

        # Use provided available_points or auto-calculate
        if available_points is None:
            available_points = set(range(1, len(self.games) + 1)) - used_points
        else:
            available_points = available_points - used_points

        # Get games that need picks (not completed, not in fixed picks)
        games_to_pick = [g for g in self.games
                        if g.actual_outcome is None
                        and g.home_team not in player_fixed
                        and g.away_team not in player_fixed]

        if len(games_to_pick) == 0:
            print("No games to optimize (all fixed or completed)")
            return player_fixed.copy()

        if len(available_points) < len(games_to_pick):
            raise ValueError(f"Not enough confidence points ({len(available_points)}) for games ({len(games_to_pick)})")

        print(f"\n🔍 HILL CLIMBING OPTIMIZATION")
        print(f"   Games to optimize: {len(games_to_pick)}")
        print(f"   Iterations per restart: {iterations}")
        print(f"   Restarts: {restarts}")
        print(f"   Total evaluations: ~{iterations * restarts:,}")
        print(f"   Tracking top {top_n} combinations for summary statistics")

        best_overall_picks = None
        best_overall_prob = 0

        # Track all explored combinations: list of (picks_dict, win_probability)
        all_combinations = []

        for restart in range(restarts):
            print(f"\n🔄 Restart {restart + 1}/{restarts}")

            # Generate initial solution
            if restart == 0:
                # First restart: use greedy approach as starting point
                print("   Starting from greedy solution...")
                current_picks = self._generate_greedy_picks(
                    player_name, player_fixed, games_to_pick, available_points, fixed_picks
                )
            else:
                # Subsequent restarts: use random solutions
                print("   Starting from random solution...")
                current_picks = self._generate_random_picks(games_to_pick, available_points)

            # Combine with fixed picks
            current_picks.update(player_fixed)

            # Evaluate initial solution
            current_prob = self._evaluate_picks(player_name, current_picks, fixed_picks)
            print(f"   Initial win probability: {current_prob:.4f}")

            # Track this initial solution
            all_combinations.append((current_picks.copy(), current_prob))

            improvements = 0
            no_improvement_count = 0

            # Hill climbing iterations
            for i in range(iterations):
                # Generate neighbor solution
                neighbor_picks = self._get_neighbor_solution(
                    current_picks, games_to_pick, player_fixed
                )

                # Evaluate neighbor
                neighbor_prob = self._evaluate_picks(player_name, neighbor_picks, fixed_picks)

                # Track this neighbor solution
                all_combinations.append((neighbor_picks.copy(), neighbor_prob))

                # Accept if better
                if neighbor_prob > current_prob:
                    current_picks = neighbor_picks
                    current_prob = neighbor_prob
                    improvements += 1
                    no_improvement_count = 0
                else:
                    no_improvement_count += 1

                # Early stopping if no improvement for a while
                if no_improvement_count >= 100:
                    print(f"   Early stop at iteration {i+1} (no improvement for 100 iterations)")
                    break

                # Progress update every 100 iterations
                if (i + 1) % 100 == 0:
                    print(f"   Iteration {i+1}/{iterations}: {current_prob:.4f} ({improvements} improvements)")

            print(f"   Final win probability: {current_prob:.4f} ({improvements} total improvements)")

            # Update best overall solution
            if current_prob > best_overall_prob:
                best_overall_picks = current_picks.copy()
                best_overall_prob = current_prob
                print(f"   ⭐ New best solution!")

            # Print best solution so far after each restart (safety net for interruptions)
            print(f"\n   💾 Best solution so far (after {restart + 1}/{restarts} restarts):")
            print(f"      Win probability: {best_overall_prob:.4f}")
            if best_overall_picks:
                sorted_best = sorted(best_overall_picks.items(), key=lambda x: x[1], reverse=True)
                picks_summary = ", ".join(f"{team}({pts})" for team, pts in sorted_best)
                print(f"      Picks: {picks_summary}")

                # Save checkpoint to file
                checkpoint_file = "hill_climb_checkpoint.txt"
                with open(checkpoint_file, 'w') as f:
                    f.write(f"Hill Climbing Checkpoint - Restart {restart + 1}/{restarts}\n")
                    f.write(f"Win Probability: {best_overall_prob:.4f}\n")
                    f.write(f"\nBest Picks (sorted by confidence):\n")
                    for team, pts in sorted_best:
                        f.write(f"  {team}: {pts}\n")
                    f.write(f"\nCopy-paste format:\n")
                    paste_format = ", ".join(f"{team} {pts}" for team, pts in sorted_best)
                    f.write(f"{paste_format}\n")

        # Calculate summary statistics from top N combinations
        print(f"\n📊 CALCULATING SUMMARY STATISTICS...")
        print(f"   Total combinations explored: {len(all_combinations):,}")

        # Filter out combinations with zero win probability
        viable_combinations = [(picks, prob) for picks, prob in all_combinations if prob > 0]
        print(f"   Viable combinations (win prob > 0): {len(viable_combinations):,}")

        # Sort viable combinations by win probability (descending)
        viable_combinations.sort(key=lambda x: x[1], reverse=True)

        # Take top N from viable combinations
        top_combinations = viable_combinations[:top_n]
        actual_n = len(top_combinations)
        print(f"   Analyzing top {actual_n} combinations")

        # Build summary statistics for each team
        team_stats = {}

        for picks, win_prob in top_combinations:
            for team, points in picks.items():
                if team not in team_stats:
                    team_stats[team] = {
                        'appearances': 0,
                        'total_points': 0,
                        'point_values': []
                    }

                team_stats[team]['appearances'] += 1
                team_stats[team]['total_points'] += points
                team_stats[team]['point_values'].append(points)

        # Convert to DataFrame
        summary_rows = []
        for team, stats in team_stats.items():
            frequency = stats['appearances'] / actual_n
            avg_points = stats['total_points'] / stats['appearances']
            point_values = stats['point_values']
            median_points = float(np.median(point_values))
            std_points = float(np.std(point_values)) if len(point_values) > 1 else 0.0

            summary_rows.append({
                'team': team,
                'frequency': frequency,
                'appearances': stats['appearances'],
                'avg_confidence': avg_points,
                'median_confidence': median_points,
                'std_confidence': std_points,
                'min_confidence': min(point_values),
                'max_confidence': max(point_values)
            })

        summary_df = pd.DataFrame(summary_rows)
        summary_df = summary_df.sort_values('frequency', ascending=False)

        print(f"\n📈 SUMMARY STATISTICS (Top {actual_n} combinations):")
        print(f"   {'Team':<8} {'Frequency':<12} {'Avg Pts':<10} {'Range'}")
        print(f"   {'-'*8} {'-'*12} {'-'*10} {'-'*15}")

        for _, row in summary_df.head(15).iterrows():
            team = row['team']
            freq = row['frequency']
            avg_conf = row['avg_confidence']
            min_conf = row['min_confidence']
            max_conf = row['max_confidence']

            # Add visual indicator for very high frequency (>80%)
            indicator = "🔒" if freq > 0.8 else "  "

            print(f"   {team:<8} {freq:>6.1%} ({row['appearances']:>4})  {avg_conf:>5.1f}      {min_conf:.0f}-{max_conf:.0f} {indicator}")

        if len(summary_df) > 15:
            print(f"   ... and {len(summary_df) - 15} more teams")

        print(f"\n✅ Best win probability found: {best_overall_prob:.4f}")
        return best_overall_picks, summary_df

    def _generate_greedy_picks(self, player_name: str, player_fixed: Dict[str, int],
                               games_to_pick: List[Game], available_points: set,
                               fixed_picks: Dict[str, Dict[str, int]]) -> Dict[str, int]:
        """Generate initial picks using a simple greedy heuristic based on Vegas odds."""
        picks = {}
        remaining_points = sorted(list(available_points), reverse=True)

        # Sort games by certainty (most certain first)
        sorted_games = sorted(games_to_pick, key=lambda g: abs(g.vegas_win_prob - 0.5), reverse=True)

        for i, game in enumerate(sorted_games):
            if i >= len(remaining_points):
                break

            # Pick the team favored by Vegas
            if game.vegas_win_prob >= 0.5:
                picks[game.home_team] = remaining_points[i]
            else:
                picks[game.away_team] = remaining_points[i]

        return picks

    def _generate_random_picks(self, games_to_pick: List[Game],
                               available_points: set) -> Dict[str, int]:
        """Generate random picks for initial solution."""
        picks = {}
        points_list = list(available_points)
        np.random.shuffle(points_list)

        for i, game in enumerate(games_to_pick):
            if i >= len(points_list):
                break

            # Randomly pick home or away
            team = game.home_team if np.random.random() < 0.5 else game.away_team
            picks[team] = points_list[i]

        return picks

    def _evaluate_picks(self, player_name: str, picks: Dict[str, int],
                        all_fixed_picks: Dict[str, Dict[str, int]]) -> float:
        """Evaluate a set of picks and return win probability."""
        # Format picks for simulation
        formatted_picks = all_fixed_picks.copy() if all_fixed_picks else {}
        formatted_picks[player_name] = picks

        # Run simulation with player_data if available (for mid-week optimization)
        player_data = getattr(self, '_optimization_player_data', None)
        stats = self.simulate_all(formatted_picks, player_data=player_data)
        return stats['win_pct'][player_name]

    def _get_neighbor_solution(self, current_picks: Dict[str, int],
                               games_to_pick: List[Game],
                               player_fixed: Dict[str, int]) -> Dict[str, int]:
        """Generate a neighbor solution by making a small change.

        Possible changes:
        1. Swap which team we pick in a random game (50% probability)
        2. Swap confidence values between two random games (50% probability)
        """
        neighbor = current_picks.copy()

        # Get teams that are not fixed (can be modified)
        modifiable_teams = [team for team in neighbor.keys() if team not in player_fixed]

        if len(modifiable_teams) < 2:
            return neighbor  # Can't make meaningful changes

        if np.random.random() < 0.5:
            # Operation 1: Swap which team we pick in a game
            # Find a game where we picked one of the teams
            game_to_swap = None
            picked_team = None

            for game in games_to_pick:
                if game.home_team in modifiable_teams:
                    game_to_swap = game
                    picked_team = game.home_team
                    break
                elif game.away_team in modifiable_teams:
                    game_to_swap = game
                    picked_team = game.away_team
                    break

            if game_to_swap and picked_team:
                # Swap to the other team in this game
                other_team = game_to_swap.away_team if picked_team == game_to_swap.home_team else game_to_swap.home_team
                confidence = neighbor[picked_team]
                del neighbor[picked_team]
                neighbor[other_team] = confidence
        else:
            # Operation 2: Swap confidence values between two games
            if len(modifiable_teams) >= 2:
                team1, team2 = np.random.choice(modifiable_teams, size=2, replace=False)
                neighbor[team1], neighbor[team2] = neighbor[team2], neighbor[team1]

        return neighbor

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
