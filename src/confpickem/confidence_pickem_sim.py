from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import numpy as np
import pandas as pd

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
        
        # Calculate various statistics
        stats = {
            'expected_points': by_sim.groupby('player')['points'].mean(),
            'point_std': by_sim.groupby('player')['points'].std(),
            'win_pct': by_sim.groupby('player')['points'].apply(
                lambda x: (x == by_sim.groupby('simulation')['points'].max()).mean()
            ),
            'value_at_risk': by_sim.groupby('player')['points'].quantile(0.05)
        }
        
        return stats

    def optimize_picks(self, fixed_picks: Dict[str, int] = None) -> Dict[str, int]:
        """Optimize picks using simulation results"""
        # Run initial simulations
        picks_df = self.simulate_picks()
        outcomes = self.simulate_outcomes()
        stats = self.analyze_results(picks_df, outcomes)
        
        # Start with best expected value picks
        optimal = {}
        available_points = set(range(1, len(self.games) + 1))
        
        for game in sorted(self.games, 
                         key=lambda g: abs(g.vegas_win_prob - 0.5),
                         reverse=True):
            if fixed_picks and (game.home_team in fixed_picks or game.away_team in fixed_picks):
                continue
                
            # Simulate both options
            home_results = self._simulate_with_pick(game.home_team, next(iter(available_points)))
            away_results = self._simulate_with_pick(game.away_team, next(iter(available_points)))
            
            # Choose better option
            if home_results['expected_value'] > away_results['expected_value']:
                optimal[game.home_team] = max(available_points)
            else:
                optimal[game.away_team] = max(available_points)
            available_points.remove(max(available_points))
            
        # Add fixed picks
        if fixed_picks:
            optimal.update(fixed_picks)
            
        return optimal

    def _simulate_with_pick(self, team: str, points: int) -> Dict:
        """Helper method to simulate outcomes with a specific pick"""
        # Run focused simulation
        picks_df = self.simulate_picks()
        outcomes = self.simulate_outcomes()
        
        # Override pick for analysis
        picks_df.loc[
            (picks_df.player == self.players[0].name) & \
            picks_df.game.str.contains(team),
            ['picked_home', 'confidence']
        ] = [team in picks_df.game.str.split('@').str[1], points]
        
        stats = self.analyze_results(picks_df, outcomes)
        
        return {
            'expected_value': stats['expected_points'][self.players[0].name],
            'win_pct': stats['win_pct'][self.players[0].name]
        }
