#!/usr/bin/env python
"""Win probability calculator using available data"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.confpickem.yahoo_pickem_scraper import YahooPickEm

def simulate_remaining_games(results, all_picks, num_sims=5000, simulate_from_game=None):
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
        favorite_win_prob = min(max(spread * 0.031 + 0.5, 0.0), 1.0)
        print(f"   {result['favorite']} vs {result['underdog']} (spread: {spread}) -> {result['favorite']} {favorite_win_prob:.1%}")
    
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
            # Convert spread to Vegas win probability (same formula as scraper)
            spread = result.get('spread', 0.0)
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
    print("🎯 WIN PROBABILITY CALCULATOR")
    print("🎲 Based on remaining game simulations")
    print("=" * 40)
    
    if not Path("cookies.txt").exists():
        print("❌ Missing cookies.txt file")
        return 1
    
    try:
        # Clear cache
        cache_dir = Path(".cache")
        if cache_dir.exists():
            import shutil
            shutil.rmtree(cache_dir)
        
        yahoo = YahooPickEm(week=1, league_id=15435, cookies_file="cookies.txt")
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
            yahoo.results, all_picks, num_sims=5000, simulate_from_game=simulate_from_game
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