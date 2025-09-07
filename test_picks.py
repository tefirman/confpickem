#!/usr/bin/env python
"""Test and verify optimized picks"""

import sys
from pathlib import Path
import pandas as pd

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.confpickem.yahoo_pickem_scraper import YahooPickEm
from src.confpickem.confidence_pickem_sim import ConfidencePickEmSimulator, Player

def main():
    """Test your optimized picks"""
    print("🧪 PICK VERIFICATION TOOL")
    print("=" * 30)
    
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
        
        yahoo = YahooPickEm(week=1, league_id=15435, cookies_file="cookies.txt")
        print(f"✅ Loaded {len(yahoo.games)} games, {len(yahoo.players)} players")
        
        # Setup simulator (use high accuracy for verification)
        simulator = ConfidencePickEmSimulator(num_sims=15000)
        
        # Convert games
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
            
            games_data.append({
                'home_team': home_team,
                'away_team': away_team,
                'vegas_win_prob': home_prob,
                'crowd_home_pick_pct': crowd_home_pct,
                'crowd_home_confidence': home_conf,
                'crowd_away_confidence': away_conf,
                'week': 1,
                'kickoff_time': game['kickoff_time'],
                'actual_outcome': None
            })
        
        simulator.add_games_from_dataframe(pd.DataFrame(games_data))
        
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
        
        # Get available teams
        available_teams = set()
        for _, game in yahoo.games.iterrows():
            available_teams.add(game['favorite'])
            available_teams.add(game['underdog'])
        available_teams = sorted(list(available_teams))
        
        print(f"\n📋 Available teams: {', '.join(available_teams)}")
        
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
                    
                    # Find matching team
                    matched_team = None
                    for team in available_teams:
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
        
        if len(user_picks) != 16:
            print(f"⚠️  Warning: {len(user_picks)} picks entered, expected 16")
            missing = 16 - len(user_picks)
            if missing > 0:
                print(f"   Missing {missing} picks - they'll be assigned randomly")
        
        print(f"\n🧪 RUNNING VERIFICATION SIMULATION...")
        print(f"   📊 Using 15,000 simulations for accuracy")
        print(f"   ⏱️  This will take 30-60 seconds...")
        
        # Test picks
        test_picks = {selected_player: user_picks}
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