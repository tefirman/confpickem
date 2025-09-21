#!/usr/bin/env python
"""Test and verify optimized picks accounting for completed mid-week games"""

import sys
import argparse
from pathlib import Path
import pandas as pd

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.confpickem.yahoo_pickem_scraper import YahooPickEm
from src.confpickem.confidence_pickem_sim import ConfidencePickEmSimulator, Player

def main():
    """Test your optimized picks"""
    parser = argparse.ArgumentParser(description='Test and verify mid-week optimized NFL picks')
    parser.add_argument('--week', '-w', type=int, default=3,
                       help='NFL week number (default: 3)')
    parser.add_argument('--league-id', '-l', type=int, default=15435,
                       help='Yahoo league ID (default: 15435)')
    args = parser.parse_args()

    week = args.week
    league_id = args.league_id

    print(f"🧪 MID-WEEK PICK VERIFICATION TOOL - WEEK {week}")
    print("🎯 Accounts for completed Thu/Fri games")
    print("=" * 45)
    
    if not Path("cookies.txt").exists():
        print("❌ Missing cookies.txt file")
        return 1
    
    print("📡 Loading league data...")
    try:
        # Clear cache for fresh data
        cache_dir = Path(".cache")
        if cache_dir.exists():
            import shutil
            shutil.rmtree(cache_dir)
        
        yahoo = YahooPickEm(week=week, league_id=league_id, cookies_file="cookies.txt")
        print(f"✅ Loaded {len(yahoo.games)} games, {len(yahoo.players)} players")
        
        # Setup simulator (use high accuracy for verification)
        simulator = ConfidencePickEmSimulator(num_sims=15000)
        
        # Convert games WITH actual outcomes for completed games
        games_data = []
        for _, game in yahoo.games.iterrows():
            favorite, underdog = game['favorite'], game['underdog']
            
            if game['home_favorite']:
                home_team, away_team = favorite, underdog
                home_prob = game['win_prob']
                crowd_home_pct = game['favorite_pick_pct'] / 100.0
                home_conf, away_conf = game['favorite_confidence'], game['underdog_confidence']
            else:
                home_team, away_team = underdog, favorite
                home_prob = 1.0 - game['win_prob']
                crowd_home_pct = game['underdog_pick_pct'] / 100.0
                home_conf, away_conf = game['underdog_confidence'], game['favorite_confidence']
            
            # Check if we have actual outcome for this game
            actual_outcome = None
            for completed in yahoo.results:
                if completed.get('winner'):
                    # Match by team names from completed game data
                    completed_teams = {completed.get('favorite', ''), completed.get('underdog', '')}
                    our_teams = {home_team, away_team}
                    if completed_teams == our_teams:  # Exact match of the two teams
                        # Determine if home team won
                        actual_outcome = (completed['winner'] == home_team)
                        print(f"   ✅ Found completed game: {completed['favorite']} vs {completed['underdog']}, winner: {completed['winner']}")
                        break
            
            games_data.append({
                'home_team': home_team,
                'away_team': away_team,
                'vegas_win_prob': home_prob,
                'crowd_home_pick_pct': crowd_home_pct,
                'crowd_home_confidence': home_conf,
                'crowd_away_confidence': away_conf,
                'week': week,
                'kickoff_time': game['kickoff_time'],
                'actual_outcome': actual_outcome  # Key difference from original test_picks.py!
            })
        
        simulator.add_games_from_dataframe(pd.DataFrame(games_data))
        
        # Show game status
        completed_count = sum(1 for game in simulator.games if game.actual_outcome is not None)
        remaining_count = len(simulator.games) - completed_count
        print(f"\n🎮 MID-WEEK SIMULATION SETUP:")
        print(f"   ✅ {completed_count} games with known outcomes")
        print(f"   🎲 {remaining_count} games to simulate")
        
        # Convert players
        players = []
        for _, player in yahoo.players.iterrows():
            players.append(Player(
                name=player['player_name'],
                skill_level=0.6,
                crowd_following=0.5,
                confidence_following=0.5
            ))
        
        simulator.players = players
        
        # Get player list
        player_names = [p.name for p in simulator.players]
        print(f"\n👥 Available players:")
        for i in range(0, min(10, len(player_names)), 2):
            row_players = player_names[i:i+2]
            for j, name in enumerate(row_players):
                print(f"   {i+j+1:2d}. {name}")
        if len(player_names) > 10:
            print(f"   ... and {len(player_names) - 10} more")
        
        # Get player name
        player_choice = input(f"\nEnter your player name or number: ").strip()
        
        selected_player = None
        try:
            idx = int(player_choice) - 1
            if 0 <= idx < len(player_names):
                selected_player = player_names[idx]
        except ValueError:
            matches = [p for p in player_names if player_choice.lower() in p.lower()]
            if len(matches) == 1:
                selected_player = matches[0]
            else:
                print(f"❌ Player not found: {player_choice}")
                return 1
        
        print(f"✅ Testing picks for: {selected_player}")
        
        # Get player's actual data from Yahoo
        your_player_data = None
        for _, player in yahoo.players.iterrows():
            if player['player_name'] == selected_player:
                your_player_data = player
                break
        
        if your_player_data is None:
            print(f"❌ Could not find player data for {selected_player}")
            return 1
        
        # Extract completed game picks automatically
        completed_picks = {}
        used_confidence = set()
        
        print(f"\n🎯 YOUR COMPLETED GAME PICKS (auto-detected):")
        for i, game in enumerate(simulator.games):
            if game.actual_outcome is not None:
                game_num = i + 1
                pick = your_player_data.get(f'game_{game_num}_pick')
                confidence = your_player_data.get(f'game_{game_num}_confidence')
                
                if pick and confidence:
                    completed_picks[pick] = int(confidence)  # Convert to int
                    used_confidence.add(int(confidence))     # Convert to int
                    was_correct = (pick == game.home_team and game.actual_outcome) or \
                                (pick == game.away_team and not game.actual_outcome)
                    status = "✅" if was_correct else "❌"
                    print(f"     {status} {pick} ({confidence} pts) - {'WIN' if was_correct else 'LOSS'}")
        
        # Get remaining teams only
        remaining_teams = set()
        for game in simulator.games:
            if game.actual_outcome is None:  # Only remaining games
                remaining_teams.add(game.home_team)
                remaining_teams.add(game.away_team)
        
        remaining_teams = sorted(list(remaining_teams))
        remaining_confidence = sorted(list(set(range(1, 17)) - used_confidence), reverse=True)
        
        print(f"\n📋 Available teams (remaining games only): {', '.join(remaining_teams)}")
        print(f"🎯 Available confidence points: {', '.join(map(str, remaining_confidence))}")
        
        # Get picks
        print(f"\n🎯 Enter your optimized picks:")
        print(f"   Format: 'Phi 16, KC 15, SF 14, TB 13, ...'")
        print(f"   (team confidence_points, separated by commas)")
        
        picks_input = input("Your picks: ").strip()
        
        if not picks_input:
            print("❌ No picks entered")
            return 1
        
        # Parse picks
        user_picks = {}
        try:
            for pick in picks_input.split(','):
                parts = pick.strip().split()
                if len(parts) >= 2:
                    team_input = parts[0].strip()
                    confidence = int(parts[1])
                    
                    # Find matching team (check remaining teams)
                    matched_team = None
                    for team in remaining_teams:
                        if team.lower() == team_input.lower():
                            matched_team = team
                            break
                    
                    if matched_team:
                        user_picks[matched_team] = confidence
                        print(f"✅ {team_input} -> {matched_team} ({confidence} pts)")
                    else:
                        print(f"❌ Team '{team_input}' not found")
                        
        except Exception as e:
            print(f"❌ Error parsing picks: {e}")
            return 1
        
        if not user_picks:
            print("❌ No valid picks found")
            return 1
        
        # Combine completed picks with user's remaining picks
        all_picks = {}
        all_picks.update(completed_picks)  # Add completed game picks
        all_picks.update(user_picks)       # Add remaining game picks
        
        remaining_games_count = len([g for g in simulator.games if g.actual_outcome is None])
        total_games = len(simulator.games)
        
        if len(user_picks) != remaining_games_count:
            print(f"⚠️  Warning: {len(user_picks)} remaining picks entered, expected {remaining_games_count}")
            missing = remaining_games_count - len(user_picks)
            if missing > 0:
                print(f"   Missing {missing} picks - they'll be assigned randomly")
        
        print(f"\n📊 Total picks: {len(all_picks)} ({len(completed_picks)} completed + {len(user_picks)} remaining)")
        
        if len(all_picks) != total_games:
            print(f"⚠️  Total picks ({len(all_picks)}) doesn't match total games ({total_games})")
            return 1
        
        print(f"\n🧪 RUNNING VERIFICATION SIMULATION...")
        print(f"   📊 Using 15,000 simulations for accuracy")
        print(f"   ⏱️  This will take 30-60 seconds...")
        
        # Test picks (combined completed + remaining)
        test_picks = {selected_player: all_picks}
        results = simulator.simulate_all(test_picks)
        
        # Also test random picks for comparison
        random_results = simulator.simulate_all({})
        
        # Get results
        your_win_prob = results['win_pct'][selected_player]
        random_win_prob = random_results['win_pct'][selected_player]
        baseline = 1.0 / len(simulator.players)
        
        print(f"\n🏆 VERIFICATION RESULTS:")
        print("=" * 40)
        print(f"📈 Your Optimized Picks:")
        print(f"   🎯 Win Probability: {your_win_prob:.1%}")
        print(f"   💪 vs Random: {your_win_prob/random_win_prob:.1f}x better")
        print(f"   🚀 vs Pure Luck: {your_win_prob/baseline:.1f}x better")
        print(f"   📊 Advantage: +{(your_win_prob - random_win_prob)*100:.1f} percentage points")
        
        print(f"\n📊 Comparison:")
        print(f"   🎯 Your picks: {your_win_prob:.1%}")
        print(f"   🎲 Random picks: {random_win_prob:.1%}")
        print(f"   🎪 Pure luck: {baseline:.1%}")
        
        # Show pick breakdown
        print(f"\n🎯 Your Pick Strategy:")
        sorted_picks = sorted(user_picks.items(), key=lambda x: x[1], reverse=True)
        
        high_conf = [p for p in sorted_picks if p[1] >= 13]
        med_conf = [p for p in sorted_picks if 7 <= p[1] < 13]
        low_conf = [p for p in sorted_picks if p[1] < 7]
        
        print(f"   🔥 High confidence (13-16): {len(high_conf)} games")
        for team, conf in high_conf:
            print(f"      {conf:2d}. {team}")
        
        print(f"   ⚖️  Medium confidence (7-12): {len(med_conf)} games") 
        print(f"   🤏 Low confidence (1-6): {len(low_conf)} games")
        
        # Add game importance analysis (win probability impact)
        print(f"\n🎯 GAME IMPORTANCE ANALYSIS:")
        print("   (Win probability change if you win vs lose each game)")
        try:
            # Create picks dataframe with all picks, then analyze importance
            test_picks_for_importance = {selected_player: all_picks}
            picks_df = simulator.simulate_picks(test_picks_for_importance)
            
            # Use the original assess_game_importance method
            importance_df = simulator.assess_game_importance(
                player_name=selected_player,
                picks_df=picks_df,
                fixed_picks={selected_player: completed_picks}  # Only completed games are truly "fixed"
            )
            
            if len(importance_df) > 0:
                # Filter to show only remaining games (non-fixed)
                remaining_importance = importance_df[importance_df['is_fixed'] == False]
                
                if len(remaining_importance) > 0:
                    # Sort by total impact and show ALL remaining games
                    remaining_sorted = remaining_importance.sort_values('total_impact', key=abs, ascending=False)
                    
                    print(f"   📊 All {len(remaining_sorted)} remaining games by win probability impact:")
                    for i, (_, row) in enumerate(remaining_sorted.iterrows()):
                        game = row['game']
                        pick = row['pick']
                        conf = int(row['points_bid'])
                        impact = row['total_impact']
                        
                        print(f"   {i+1:2d}. {game:<20} → {pick:3} ({conf:2d} pts) {impact:+5.1%}")
                        
                    print(f"   💡 Positive = helps if you win, Negative = hurts if you lose")
                else:
                    print(f"     All games are completed - no remaining games to analyze")
            else:
                print(f"     No games to analyze")
                
        except Exception as e:
            print(f"   ⚠️ Could not analyze game importance: {e}")
        
        print(f"\n✅ Verification complete!")
        
        if your_win_prob > 0.05:  # 5%+ in a 56 player league is very good
            print("🎉 Excellent picks! Well above average.")
        elif your_win_prob > 0.03:  # 3%+ is good
            print("👍 Good picks! Solid advantage.")
        else:
            print("🤔 Picks are okay, but might want to revisit strategy.")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())