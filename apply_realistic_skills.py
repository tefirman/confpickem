#!/usr/bin/env python
"""Apply realistic skill levels to current players"""

import sys
from pathlib import Path
import json
import numpy as np
import argparse

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.confpickem.yahoo_pickem_scraper import YahooPickEm
from src.confpickem.confidence_pickem_sim import ConfidencePickEmSimulator, Player

def combine_raw_stats(stats1, stats2):
    """Combine raw player statistics from two different time periods"""
    combined = {}

    # Get all unique players
    all_players = set(stats1.keys()) | set(stats2.keys())

    for player in all_players:
        if player in stats1 and player in stats2:
            # Player exists in both - aggregate their stats
            s1 = stats1[player]
            s2 = stats2[player]

            combined[player] = {
                'total_picks': s1['total_picks'] + s2['total_picks'],
                'total_correct': s1['total_correct'] + s2['total_correct'],
                'total_points': s1['total_points'] + s2['total_points'],
                'total_possible_points': s1['total_possible_points'] + s2['total_possible_points'],
                'weeks_played': s1['weeks_played'] + s2['weeks_played'],
                'confidence_distribution': {
                    k: s1['confidence_distribution'].get(k, 0) + s2['confidence_distribution'].get(k, 0)
                    for k in set(s1['confidence_distribution'].keys()) | set(s2['confidence_distribution'].keys())
                },
                'pick_accuracy_by_confidence': {}
            }

            # Merge pick_accuracy_by_confidence
            all_confs = set(s1['pick_accuracy_by_confidence'].keys()) | set(s2['pick_accuracy_by_confidence'].keys())
            for conf in all_confs:
                c1 = s1['pick_accuracy_by_confidence'].get(conf, {'correct': 0, 'total': 0})
                c2 = s2['pick_accuracy_by_confidence'].get(conf, {'correct': 0, 'total': 0})
                combined[player]['pick_accuracy_by_confidence'][conf] = {
                    'correct': c1['correct'] + c2['correct'],
                    'total': c1['total'] + c2['total']
                }
        elif player in stats1:
            combined[player] = stats1[player]
        else:
            combined[player] = stats2[player]

    return combined

def calculate_skills_from_stats(raw_stats):
    """Calculate skill levels from raw statistics"""
    player_skills = {}

    for name, stats in raw_stats.items():
        if stats['total_picks'] < 20:  # Skip players with too little data
            continue

        accuracy = stats['total_correct'] / stats['total_picks']
        efficiency = stats['total_points'] / stats['total_possible_points'] if stats['total_possible_points'] > 0 else 0

        # Analyze confidence behavior
        total_conf_picks = sum(stats['confidence_distribution'].values())
        high_conf_usage = 0  # 13-16 point picks
        low_conf_usage = 0   # 1-4 point picks

        for conf, count in stats['confidence_distribution'].items():
            conf_int = int(conf) if isinstance(conf, str) else conf
            if conf_int >= 13:
                high_conf_usage += count
            elif conf_int <= 4:
                low_conf_usage += count

        high_conf_rate = high_conf_usage / total_conf_picks if total_conf_picks > 0 else 0
        low_conf_rate = low_conf_usage / total_conf_picks if total_conf_picks > 0 else 0

        # Estimate skill level (0.3 to 0.9 range)
        skill_level = max(0.3, min(0.9, 0.3 + accuracy * 0.6))

        # Estimate crowd following (high confidence on popular picks)
        crowd_following = 0.5  # Default for now

        # Estimate confidence following (how much they use extreme confidence levels)
        confidence_following = (high_conf_rate + low_conf_rate)  # 0-1 scale
        confidence_following = max(0.1, min(0.9, confidence_following))

        player_skills[name] = {
            'skill_level': skill_level,
            'crowd_following': crowd_following,
            'confidence_following': confidence_following,
            'accuracy': accuracy,
            'efficiency': efficiency,
            'weeks_played': stats['weeks_played'],
            'total_picks': stats['total_picks']
        }

    return player_skills

def load_skill_data(year=None):
    """Load skill analysis from specified year(s)

    If year is None, loads both 2024 and 2025, aggregates raw stats, and recalculates skills.
    If year is specified, only loads that year's data.
    """
    if year:
        # Load specific year
        filename = f'player_skills_{year}.json'
        try:
            with open(filename, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"❌ {filename} not found. Run analyze_player_skills.py --year {year} first.")
            return None
    else:
        # Load both years and combine raw stats with equal weight
        data_2024 = None
        data_2025 = None

        try:
            with open('player_skills_2024.json', 'r') as f:
                data_2024 = json.load(f)
                print("✅ Loaded 2024 player skills")
        except FileNotFoundError:
            print("⚠️ player_skills_2024.json not found")

        try:
            with open('player_skills_2025.json', 'r') as f:
                data_2025 = json.load(f)
                print("✅ Loaded 2025 player skills")
        except FileNotFoundError:
            print("⚠️ player_skills_2025.json not found")

        if not data_2024 and not data_2025:
            print("❌ No skill data found. Run analyze_player_skills.py first.")
            return None

        # Combine raw stats and recalculate skills
        if data_2024 and data_2025:
            # Check if raw stats are available
            if 'raw_player_stats' not in data_2024 or 'raw_player_stats' not in data_2025:
                print("⚠️ Raw stats not found in JSON files. Re-run analyze_player_skills.py to generate them.")
                print("   Falling back to using 2025 data only...")
                return data_2025

            print("📊 Combining raw statistics from both years with equal weight...")
            combined_raw = combine_raw_stats(data_2024['raw_player_stats'], data_2025['raw_player_stats'])

            print(f"   📈 Combined stats for {len(combined_raw)} players")

            # Recalculate skills from combined stats
            combined_skills = calculate_skills_from_stats(combined_raw)

            # Calculate new distribution stats
            skill_levels = [p['skill_level'] for p in combined_skills.values()]
            crowd_followings = [p['crowd_following'] for p in combined_skills.values()]
            confidence_followings = [p['confidence_following'] for p in combined_skills.values()]

            result = {
                'player_skills': combined_skills,
                'distribution_stats': {
                    'skill_level': {'mean': float(np.mean(skill_levels)), 'std': float(np.std(skill_levels))},
                    'crowd_following': {'mean': float(np.mean(crowd_followings)), 'std': float(np.std(crowd_followings))},
                    'confidence_following': {'mean': float(np.mean(confidence_followings)), 'std': float(np.std(confidence_followings))}
                }
            }

            print(f"   ✅ Recalculated skills based on combined data")
            return result
        else:
            # Return whichever one we have
            return data_2025 or data_2024

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

def create_realistic_simulator(year=None):
    """Create simulator with realistic player skills"""

    print("🎯 CREATING REALISTIC PLAYER SIMULATOR")
    print("=" * 45)

    # Load skill data
    skill_data = load_skill_data(year)
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
    print(f"   ✅ {len(matched)} players matched to historical data")
    print(f"   ❓ {len(unmatched)} players need estimated skills")
    
    if matched:
        print(f"\n✅ MATCHED PLAYERS:")
        for name, data in list(matched.items())[:10]:  # Show first 10
            skill = data['skill_level']
            acc = data.get('accuracy', 0)
            print(f"   {name:<25} skill={skill:.3f} (historical accuracy: {acc:.1%})")
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
    print(f"   🎯 vs historical data: μ={distribution['skill_level']['mean']:.3f}, σ={distribution['skill_level']['std']:.3f}")
    
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
    parser = argparse.ArgumentParser(description='Apply realistic skill levels to current players')
    parser.add_argument('--year', type=int, default=None,
                        help='Year to use (default: combine 2024 and 2025, with 2025 taking precedence)')
    args = parser.parse_args()

    result = create_realistic_simulator(args.year)
    if not result:
        return 1
    
    simulator, assignments = result
    
    # print(f"\n🧪 TESTING REALISTIC SIMULATOR:")
    
    # # Test with a few players
    # test_players = list(assignments.keys())[:3]
    
    # for player_name in test_players:
    #     skill_data = assignments[player_name]
    #     print(f"\n🎯 Testing {player_name}:")
    #     print(f"   Skill: {skill_data['skill_level']:.3f}")
    #     print(f"   Crowd: {skill_data['crowd_following']:.3f}")  
    #     print(f"   Confidence: {skill_data['confidence_following']:.3f}")
        
    #     # Quick simulation test
    #     try:
    #         stats = simulator.simulate_all({})
    #         win_prob = stats['win_pct'][player_name]
    #         print(f"   Win probability: {win_prob:.1%}")
    #     except Exception as e:
    #         print(f"   ⚠️ Simulation test failed: {e}")
    
    # Save assignments for future use
    with open('current_player_skills.json', 'w') as f:
        json.dump(assignments, f, indent=2, default=str)
    
    print(f"\n💾 Player skill assignments saved to current_player_skills.json")
    print(f"💡 Use this realistic simulator for much better optimization!")
    
    return 0

if __name__ == "__main__":
    import pandas as pd  # Import here to avoid issues if not available
    main()