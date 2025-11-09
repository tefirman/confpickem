#!/usr/bin/env python
"""Win probability calculator using available data"""

import sys
import argparse
from pathlib import Path
import pandas as pd
import numpy as np

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.confpickem.yahoo_pickem_scraper import YahooPickEm
from src.confpickem.confidence_pickem_sim import ConfidencePickEmSimulator, Player

def simulate_remaining_games(results, all_picks, num_sims=5000, simulate_from_game=None, yahoo_games=None):
    """Simulate remaining games using Vegas probabilities and calculate final standings
    
    Args:
        simulate_from_game: If specified, treat all games from this index onward as pending
                          (useful for "what if" scenarios like Sunday 10am kickoff)
    """
    
    if simulate_from_game is not None:
        # Simulate from a specific point in time
        completed_games = [(i, r) for i, r in enumerate(results) if i < simulate_from_game]
        pending_games = [(i, r) for i, r in enumerate(results) if i >= simulate_from_game]
    else:
        # Use actual current state
        pending_games = [(i, r) for i, r in enumerate(results) if not r['winner']]
        completed_games = [(i, r) for i, r in enumerate(results) if r['winner']]
    
    print(f"🎲 Simulating {len(pending_games)} remaining games with {num_sims:,} iterations...")
    print(f"📊 Using Vegas spreads to calculate realistic win probabilities:")
    
    # Show Vegas probabilities for pending games
    for game_idx, result in pending_games:
        spread = result.get('spread', 0.0)
        
        # Find matching game in yahoo.games to get actual Vegas probability
        vegas_prob = 0.5  # Default fallback
        if yahoo_games is not None:
            try:
                # Find matching game by teams
                result_teams = {result['favorite'], result['underdog']}
                for _, game_row in yahoo_games.iterrows():
                    game_teams = {game_row['favorite'], game_row['underdog']}
                    if result_teams == game_teams:
                        # Found matching game, get Vegas probability for favorite
                        if game_row['favorite'] == result['favorite']:
                            vegas_prob = game_row['win_prob']
                        else:
                            vegas_prob = 1.0 - game_row['win_prob']
                        break
            except:
                # Fallback to spread calculation if lookup fails
                if spread != 0:
                    vegas_prob = min(max(spread * 0.031 + 0.5, 0.0), 1.0)
        else:
            # Fallback to spread calculation if no yahoo_games provided
            if spread != 0:
                vegas_prob = min(max(spread * 0.031 + 0.5, 0.0), 1.0)
            
        print(f"   {result['favorite']} vs {result['underdog']} (spread: {spread}) -> {result['favorite']} {vegas_prob:.1%}")
    
    # Calculate current points for each player
    current_standings = {}
    for player_name, picks in all_picks.items():
        points = 0
        for game_idx, result in completed_games:
            game_num = game_idx + 1
            pick = picks.get(f'game_{game_num}_pick')
            confidence = picks.get(f'game_{game_num}_confidence', 0)
            
            if pick and pick == result['winner']:
                points += confidence
        
        current_standings[player_name] = points
    
    # Run simulations
    win_counts = {name: 0 for name in all_picks.keys()}
    
    for sim in range(num_sims):
        # Start with current points
        sim_standings = current_standings.copy()
        
        # Simulate each pending game using Vegas probabilities
        for game_idx, result in pending_games:
            # Get Vegas probability using same logic as display
            favorite_win_prob = 0.5  # Default fallback
            if yahoo_games is not None:
                try:
                    # Find matching game by teams
                    result_teams = {result['favorite'], result['underdog']}
                    for _, game_row in yahoo_games.iterrows():
                        game_teams = {game_row['favorite'], game_row['underdog']}
                        if result_teams == game_teams:
                            # Found matching game, get Vegas probability for favorite
                            if game_row['favorite'] == result['favorite']:
                                favorite_win_prob = game_row['win_prob']
                            else:
                                favorite_win_prob = 1.0 - game_row['win_prob']
                            break
                except:
                    # Fallback to spread calculation if lookup fails
                    spread = result.get('spread', 0.0)
                    if spread != 0:
                        favorite_win_prob = min(max(spread * 0.031 + 0.5, 0.0), 1.0)
            else:
                # Fallback to spread calculation if no yahoo_games provided
                spread = result.get('spread', 0.0)
                if spread != 0:
                    favorite_win_prob = min(max(spread * 0.031 + 0.5, 0.0), 1.0)
            
            # Simulate game outcome based on Vegas probability
            if np.random.random() < favorite_win_prob:
                winner = result['favorite']
            else:
                winner = result['underdog']
            
            # Add points for correct picks
            game_num = game_idx + 1
            for player_name, picks in all_picks.items():
                pick = picks.get(f'game_{game_num}_pick')
                confidence = picks.get(f'game_{game_num}_confidence', 0)
                
                if pick and pick == winner:
                    sim_standings[player_name] += confidence
        
        # Find winner of this simulation
        max_points = max(sim_standings.values())
        winners = [name for name, points in sim_standings.items() if points == max_points]
        
        # Handle ties randomly
        winner = np.random.choice(winners)
        win_counts[winner] += 1
    
    # Convert to probabilities
    win_probs = {name: count / num_sims for name, count in win_counts.items()}
    
    return win_probs, current_standings

def main():
    """Calculate win probabilities"""
    parser = argparse.ArgumentParser(description='Calculate NFL win probabilities')
    parser.add_argument('--week', '-w', type=int, default=3,
                       help='NFL week number (default: 3)')
    parser.add_argument('--league-id', '-l', type=int, default=15435,
                       help='Yahoo league ID (default: 15435)')
    args = parser.parse_args()

    week = args.week
    league_id = args.league_id

    print(f"🎯 WIN PROBABILITY CALCULATOR - WEEK {week}")
    print("🎲 Based on remaining game simulations")
    print("=" * 45)
    
    if not Path("cookies.txt").exists():
        print("❌ Missing cookies.txt file")
        return 1
    
    try:
        # Clear cache
        cache_dir = Path(".cache")
        if cache_dir.exists():
            import shutil
            shutil.rmtree(cache_dir)
        
        yahoo = YahooPickEm(week=week, league_id=league_id, cookies_file="cookies.txt")
        print(f"✅ Loaded {len(yahoo.players)} players, {len(yahoo.results)} results")
        
        # Extract all player picks
        print("🔍 Extracting everyone's picks...")
        all_picks = {}
        
        for _, player in yahoo.players.iterrows():
            player_name = player['player_name']
            picks = {}
            
            for i in range(1, len(yahoo.results) + 1):
                pick = player.get(f'game_{i}_pick')
                confidence = player.get(f'game_{i}_confidence')
                
                if pick and confidence:
                    picks[f'game_{i}_pick'] = pick
                    picks[f'game_{i}_confidence'] = confidence
            
            all_picks[player_name] = picks
        
        print(f"   📊 Extracted picks for {len(all_picks)} players")
        
        # Show game status
        completed = sum(1 for r in yahoo.results if r['winner'])
        pending = len(yahoo.results) - completed
        
        print(f"\n🎮 GAME STATUS:")
        print(f"   ✅ {completed} completed")
        print(f"   ⏳ {pending} pending")
        
        if pending == 0 and completed == len(yahoo.results):
            print("🏁 All games complete - final standings are set!")
            print("💡 But you can still simulate from earlier time points...")
        
        # Ask user what time point to simulate from
        print(f"\n⏰ SIMULATION TIME POINT:")
        print(f"   1. Current state ({completed} completed, {pending} pending)")
        print(f"   2. Sunday 10am kickoff (after Thu/Fri, before Sunday games)")
        print(f"   3. Custom point in time")
        
        try:
            choice = input("Choose simulation point (1-3): ").strip()
        except EOFError:
            choice = "1"  # Default to current state
        
        simulate_from_game = None
        
        if choice == "2":
            # Find first Sunday game (usually game 3+ after Thu/Fri games)
            # Look for games that aren't Thu/Fri
            first_sunday_game = 2  # Default assumption - first 2 are Thu/Fri
            simulate_from_game = first_sunday_game
            print(f"🕙 Simulating from Sunday 10am kickoff (games {first_sunday_game+1}+ pending)")
            
        elif choice == "3":
            try:
                game_input = input(f"Simulate from which game? (1-{len(yahoo.results)}): ").strip()
                game_num = int(game_input)
                if 1 <= game_num <= len(yahoo.results):
                    simulate_from_game = game_num - 1  # Convert to 0-based index
                    print(f"🕐 Simulating from after game {game_num}")
                else:
                    print("⚠️ Invalid game number, using current state")
            except (ValueError, EOFError):
                print("⚠️ Invalid input, using current state")
        else:
            print("📊 Using current actual state")
        
        # Run simulation
        win_probs, current_standings = simulate_remaining_games(
            yahoo.results, all_picks, num_sims=5000, simulate_from_game=simulate_from_game, yahoo_games=yahoo.games
        )
        
        # Show results
        print(f"\n🏆 WIN PROBABILITY RANKINGS:")
        print(f"{'Rank':<4} {'Player':<25} {'Win %':<8} {'Current'}")
        print("-" * 50)
        
        # Sort by win probability
        sorted_probs = sorted(win_probs.items(), key=lambda x: x[1], reverse=True)
        
        for i, (name, prob) in enumerate(sorted_probs[:20], 1):
            current_pts = current_standings.get(name, 0)
            print(f"{i:2d}.  {name:<25} {prob:6.1%}   {current_pts} pts")
        
        if len(sorted_probs) > 20:
            print(f"... and {len(sorted_probs) - 20} more players")
        
        # Look up specific player
        try:
            print(f"\n🔍 Look up specific player?")
            search = input("Player name (or Enter to skip): ").strip()
        except EOFError:
            search = ""
        
        if search:
            matches = [(name, prob) for name, prob in sorted_probs 
                      if search.lower() in name.lower()]
            
            if matches:
                for name, prob in matches:
                    rank = next(i for i, (n, _) in enumerate(sorted_probs, 1) if n == name)
                    current_pts = current_standings.get(name, 0)
                    
                    print(f"\n🎯 {name}:")
                    print(f"   Win Probability: {prob:.1%}")
                    print(f"   Current Rank: #{rank}")
                    print(f"   Current Points: {current_pts}")
                    
                    # Show what they need
                    leader_points = max(current_standings.values())
                    points_behind = leader_points - current_pts
                    
                    if points_behind > 0:
                        print(f"   Points Behind Leader: {points_behind}")
                    else:
                        print(f"   Currently Leading!")
                        
                    print(f"   Games Remaining: {pending}")
                    
                    # Add remaining game importance analysis for this player
                    if pending > 0:
                        try:
                            print(f"\n🔥 REMAINING GAME IMPORTANCE for {name}:")
                            print(f"   (Higher scores = more critical for your final position)")
                            
                            # Setup simulator for game importance analysis (reuse yahoo data structure)
                            importance_simulator = ConfidencePickEmSimulator(num_sims=100)  # Fast for this analysis
                            
                            # Convert games to simulator format using existing yahoo.games data
                            games_data = []
                            for _, game_row in yahoo.games.iterrows():
                                favorite, underdog = game_row['favorite'], game_row['underdog']
                                
                                if game_row['home_favorite']:
                                    home_team, away_team = favorite, underdog
                                    home_prob = game_row['win_prob']
                                    crowd_home_pct = game_row['favorite_pick_pct'] / 100.0
                                    home_conf, away_conf = game_row['favorite_confidence'], game_row['underdog_confidence']
                                else:
                                    home_team, away_team = underdog, favorite
                                    home_prob = 1.0 - game_row['win_prob']
                                    crowd_home_pct = game_row['underdog_pick_pct'] / 100.0
                                    home_conf, away_conf = game_row['underdog_confidence'], game_row['favorite_confidence']
                                
                                # Check if game has actual outcome from results
                                actual_outcome = None
                                for result in yahoo.results:
                                    if result.get('winner'):
                                        # Match by team names
                                        result_teams = {result.get('favorite', ''), result.get('underdog', '')}
                                        game_teams = {home_team, away_team}
                                        if result_teams == game_teams:
                                            actual_outcome = (result['winner'] == home_team)
                                            break
                                
                                games_data.append({
                                    'home_team': home_team,
                                    'away_team': away_team,
                                    'vegas_win_prob': home_prob,
                                    'crowd_home_pick_pct': crowd_home_pct,
                                    'crowd_home_confidence': home_conf,
                                    'crowd_away_confidence': away_conf,
                                    'week': week,
                                    'kickoff_time': game_row['kickoff_time'],
                                    'actual_outcome': actual_outcome
                                })
                            
                            importance_simulator.add_games_from_dataframe(pd.DataFrame(games_data))
                            
                            # Add players to the simulator
                            players = []
                            for _, player in yahoo.players.iterrows():
                                players.append(Player(
                                    name=player['player_name'],
                                    skill_level=0.6,
                                    crowd_following=0.5,
                                    confidence_following=0.5
                                ))
                            importance_simulator.players = players
                            
                            # Convert player picks to the format needed
                            player_picks = {}
                            for _, player_row in yahoo.players.iterrows():
                                player_name = player_row['player_name']
                                picks_dict = {}
                                
                                for i in range(1, len(yahoo.games) + 1):
                                    pick = player_row.get(f'game_{i}_pick')
                                    conf = player_row.get(f'game_{i}_confidence')
                                    if pick and conf:
                                        picks_dict[pick] = int(conf)
                                
                                player_picks[player_name] = picks_dict
                            
                            # Run remaining game importance analysis
                            # Create picks dataframe for the selected player
                            test_picks_for_importance = {name: player_picks[name]}
                            picks_df = importance_simulator.simulate_picks(test_picks_for_importance)
                            
                            # Create fixed_picks dict with completed games to exclude them from analysis
                            completed_picks = {}
                            for i, game in enumerate(importance_simulator.games):
                                if game.actual_outcome is not None:
                                    # This game is completed, mark player's pick as "fixed"
                                    player_pick = None
                                    for team_pick, conf in player_picks[name].items():
                                        if team_pick in [game.home_team, game.away_team]:
                                            player_pick = team_pick
                                            break
                                    if player_pick:
                                        completed_picks[player_pick] = player_picks[name][player_pick]
                            
                            fixed_picks_for_analysis = {name: completed_picks} if completed_picks else None
                            
                            importance_df = importance_simulator.assess_game_importance(
                                player_name=name,
                                picks_df=picks_df,
                                fixed_picks=fixed_picks_for_analysis
                            )
                            
                            if len(importance_df) > 0:
                                # Filter to only remaining games (non-fixed)
                                remaining_importance = importance_df[importance_df['is_fixed'] == False]
                                
                                if len(remaining_importance) > 0:
                                    # Show all remaining games by importance
                                    for i, (_, row) in enumerate(remaining_importance.iterrows()):
                                        game = row['game']
                                        pick = row['pick']
                                        conf = int(row['points_bid'])
                                        impact = row['total_impact']

                                        # Get actual win/loss probabilities from the importance analysis
                                        # These are the actual probabilities if you win vs lose this specific game
                                        correct_prob = row['win_probability']
                                        incorrect_prob = row['loss_probability']

                                        print(f"     {i+1}. {game:<15} → {pick:3} ({conf:2d} pts) {impact:+5.1%} (Correct: {correct_prob:4.1%}, Wrong: {incorrect_prob:4.1%})")

                                    print(f"   💡 Shows win probability change if you win vs lose each game")
                                else:
                                    print(f"     All remaining games analyzed")
                            else:
                                print(f"     No remaining games to analyze")
                                
                        except Exception as e:
                            print(f"     ⚠️ Could not analyze game importance: {e}")
            else:
                print(f"❌ No matches found for '{search}'")
        
        print(f"\n🎲 Simulation uses Vegas spreads for realistic win probabilities")
        print(f"💡 Based on 5000 Monte Carlo simulations with actual betting odds")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    main()