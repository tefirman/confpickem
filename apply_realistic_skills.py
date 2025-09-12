#!/usr/bin/env python
"""Apply realistic 2024-derived skill levels to current players"""

import sys
from pathlib import Path
import json
import numpy as np

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.confpickem.yahoo_pickem_scraper import YahooPickEm
from src.confpickem.confidence_pickem_sim import ConfidencePickEmSimulator, Player

def load_skill_data():
    """Load the 2024 skill analysis"""
    try:
        with open('player_skills_2024.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print("❌ player_skills_2024.json not found. Run analyze_player_skills.py first.")
        return None

def match_players_to_skills(current_players, historical_skills):
    """Match current players to historical skill levels"""
    
    # Direct name matches first
    matched = {}
    unmatched = []
    
    historical_names = set(historical_skills.keys())
    
    for player in current_players:
        # Try exact match
        if player in historical_names:
            matched[player] = historical_skills[player]
        else:
            # Try fuzzy matching (remove spaces, case insensitive, etc)
            player_clean = player.lower().replace(' ', '').replace('_', '')
            
            found_match = False
            for hist_name, hist_data in historical_skills.items():
                hist_clean = hist_name.lower().replace(' ', '').replace('_', '')
                
                # Various matching strategies
                if (hist_clean == player_clean or 
                    player_clean in hist_clean or 
                    hist_clean in player_clean):
                    matched[player] = hist_data
                    found_match = True
                    break
            
            if not found_match:
                unmatched.append(player)
    
    return matched, unmatched

def assign_skills_to_unmatched(unmatched_players, skill_distribution):
    """Assign realistic skills to unmatched players using distribution"""
    
    # Get distribution parameters
    skill_mean = skill_distribution['skill_level']['mean']
    skill_std = skill_distribution['skill_level']['std']
    crowd_mean = skill_distribution['crowd_following']['mean']
    conf_mean = skill_distribution['confidence_following']['mean']
    conf_std = skill_distribution['confidence_following']['std']
    
    assignments = {}
    
    for player in unmatched_players:
        # Sample from realistic distributions
        skill = np.random.normal(skill_mean, skill_std)
        skill = max(0.3, min(0.9, skill))  # Clamp to valid range
        
        # Add some variation to crowd/confidence following
        crowd = np.random.normal(crowd_mean, 0.1)  # Small variation
        crowd = max(0.1, min(0.9, crowd))
        
        confidence = np.random.normal(conf_mean, conf_std)
        confidence = max(0.1, min(0.9, confidence))
        
        assignments[player] = {
            'skill_level': skill,
            'crowd_following': crowd,
            'confidence_following': confidence,
            'source': 'sampled'
        }
    
    return assignments

def create_realistic_simulator():
    """Create simulator with realistic 2024-based player skills"""
    
    print("🎯 CREATING REALISTIC PLAYER SIMULATOR")
    print("=" * 45)
    
    # Load skill data
    skill_data = load_skill_data()
    if not skill_data:
        return None
    
    historical_skills = skill_data['player_skills']
    distribution = skill_data['distribution_stats']
    
    print(f"📊 Loaded skills for {len(historical_skills)} historical players")
    
    # Get current players
    if not Path("cookies.txt").exists():
        print("❌ Missing cookies.txt file")
        return None
    
    # Clear cache for fresh data
    cache_dir = Path(".cache")
    if cache_dir.exists():
        import shutil
        shutil.rmtree(cache_dir)
    
    yahoo = YahooPickEm(week=2, league_id=15435, cookies_file="cookies.txt")
    current_player_names = [p['player_name'] for _, p in yahoo.players.iterrows()]
    
    print(f"👥 Current league has {len(current_player_names)} players")
    
    # Match players
    matched, unmatched = match_players_to_skills(current_player_names, historical_skills)
    
    print(f"\n🔍 PLAYER MATCHING:")
    print(f"   ✅ {len(matched)} players matched to 2024 data")
    print(f"   ❓ {len(unmatched)} players need estimated skills")
    
    if matched:
        print(f"\n✅ MATCHED PLAYERS:")
        for name, data in list(matched.items())[:10]:  # Show first 10
            skill = data['skill_level']
            acc = data.get('accuracy', 0)
            print(f"   {name:<25} skill={skill:.3f} (2024 accuracy: {acc:.1%})")
        if len(matched) > 10:
            print(f"   ... and {len(matched) - 10} more")
    
    if unmatched:
        print(f"\n❓ UNMATCHED PLAYERS (will use distribution sampling):")
        for name in unmatched[:10]:
            print(f"   {name}")
        if len(unmatched) > 10:
            print(f"   ... and {len(unmatched) - 10} more")
    
    # Assign skills to unmatched players
    unmatched_assignments = assign_skills_to_unmatched(unmatched, distribution)
    
    # Combine all assignments
    all_assignments = {**matched, **unmatched_assignments}
    
    print(f"\n🎮 CREATING SIMULATOR WITH REALISTIC SKILLS:")
    
    # Create players with realistic skills
    players = []
    skill_levels = []
    
    for _, player_row in yahoo.players.iterrows():
        name = player_row['player_name']
        
        if name in all_assignments:
            skill_data = all_assignments[name]
            skill = skill_data['skill_level']
            crowd = skill_data['crowd_following']
            confidence = skill_data['confidence_following']
            
            skill_levels.append(skill)
        else:
            # Fallback to distribution mean
            skill = distribution['skill_level']['mean']
            crowd = distribution['crowd_following']['mean']
            confidence = distribution['confidence_following']['mean']
            skill_levels.append(skill)
        
        players.append(Player(
            name=name,
            skill_level=skill,
            crowd_following=crowd,
            confidence_following=confidence
        ))
    
    # Show skill distribution
    print(f"   📈 Skill distribution: μ={np.mean(skill_levels):.3f}, σ={np.std(skill_levels):.3f}")
    print(f"   📊 Range: [{min(skill_levels):.3f}, {max(skill_levels):.3f}]")
    print(f"   🎯 vs 2024 data: μ={distribution['skill_level']['mean']:.3f}, σ={distribution['skill_level']['std']:.3f}")
    
    # Setup simulator (use basic game data since games scraper is broken)
    simulator = ConfidencePickEmSimulator(num_sims=5000)
    
    # Create basic games from results with all required fields
    games_data = []
    for _, result in enumerate(yahoo.results):
        spread = result.get('spread', 0.0)
        favorite_win_prob = min(max(spread * 0.031 + 0.5, 0.0), 1.0) if spread else 0.6
        
        games_data.append({
            'home_team': result['favorite'],  # Simplified
            'away_team': result['underdog'],
            'vegas_win_prob': favorite_win_prob,
            'crowd_home_pick_pct': 0.5,  # Default
            'crowd_home_confidence': 8,  # Default
            'crowd_away_confidence': 8,  # Default
            'week': 1,
            'kickoff_time': '1:00 PM ET',  # Default
            'actual_outcome': result['winner'] == result['favorite'] if result['winner'] else None
        })
    
    simulator.add_games_from_dataframe(pd.DataFrame(games_data))
    simulator.players = players
    
    return simulator, all_assignments

def main():
    """Create and test realistic simulator"""
    
    result = create_realistic_simulator()
    if not result:
        return 1
    
    simulator, assignments = result
    
    print(f"\n🧪 TESTING REALISTIC SIMULATOR:")
    
    # Test with a few players
    test_players = list(assignments.keys())[:3]
    
    for player_name in test_players:
        skill_data = assignments[player_name]
        print(f"\n🎯 Testing {player_name}:")
        print(f"   Skill: {skill_data['skill_level']:.3f}")
        print(f"   Crowd: {skill_data['crowd_following']:.3f}")  
        print(f"   Confidence: {skill_data['confidence_following']:.3f}")
        
        # Quick simulation test
        try:
            stats = simulator.simulate_all({})
            win_prob = stats['win_pct'][player_name]
            print(f"   Win probability: {win_prob:.1%}")
        except Exception as e:
            print(f"   ⚠️ Simulation test failed: {e}")
    
    # Save assignments for future use
    with open('current_player_skills.json', 'w') as f:
        json.dump(assignments, f, indent=2, default=str)
    
    print(f"\n💾 Player skill assignments saved to current_player_skills.json")
    print(f"💡 Use this realistic simulator for much better optimization!")
    
    return 0

if __name__ == "__main__":
    import pandas as pd  # Import here to avoid issues if not available
    main()