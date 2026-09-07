#!/usr/bin/env python
"""Analyze player performance to derive realistic skill levels"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
from bs4.element import Comment
import json
from collections import defaultdict
import argparse

def parse_pick_distribution(dist_file):
    """Parse a pick_distribution HTML file for crowd pick % and crowd confidence.

    Returns a list of dicts keyed by favorite/underdog so it can be paired with
    the confidence_picks games (which use the same favorite/underdog labels and
    ordering). Missing/unparseable -> None.
    """
    try:
        with open(dist_file, 'r') as f:
            content = f.read()
    except OSError:
        return None

    soup = BeautifulSoup(content, 'html.parser')
    blocks = soup.find_all('div', class_='ysf-matchup-dist')
    if not blocks:
        return None

    crowd = []
    for block in blocks:
        try:
            teams = block.find_all('th')
            percentages = block.find_all('dd', class_='percent')
            if len(teams) < 2 or len(percentages) < 2:
                continue

            entry = {
                'favorite': teams[0].text.strip(),
                'underdog': teams[-1].text.strip(),
                'favorite_pick_pct': float(percentages[0].text.strip().replace('%', '')),
                'underdog_pick_pct': float(percentages[1].text.strip().replace('%', '')),
                'favorite_confidence': 8.0,
                'underdog_confidence': 8.0,
            }

            ft = block.find('div', class_='ft')
            if ft:
                conf_row = ft.find('tr', class_='odd first')
                if conf_row:
                    cells = conf_row.find_all('td')
                    if len(cells) >= 3:
                        try:
                            entry['favorite_confidence'] = float(cells[0].text.strip())
                            entry['underdog_confidence'] = float(cells[2].text.strip())
                        except ValueError:
                            pass

            crowd.append(entry)
        except Exception:
            continue

    return crowd or None


def parse_week_data(week_file):
    """Parse a single week's HTML file to extract player performance"""
    try:
        with open(week_file, 'r') as f:
            content = f.read()

        soup = BeautifulSoup(content, 'html.parser')
        
        # Find the main picks table
        table = soup.find('div', {'id': 'ysf-group-picks'})
        if not table:
            print(f"⚠️ No picks table found in {week_file}")
            return None, None
        
        # Parse game results from header rows
        all_rows = table.find_all('tr')
        header_rows = []
        
        # Find the actual header rows (skip empty rows)
        for row in all_rows:
            cols = row.find_all('td')
            if len(cols) > 5:  # Should have many columns for games
                header_rows.append(cols[1:-1])  # Skip first (label) and last (total) columns
            if len(header_rows) >= 3:  # We need favored, spread, underdog rows
                break
        
        if len(header_rows) < 3:
            print(f"⚠️ Could not find header rows in {week_file}")
            return None, None
            
        games_meta = header_rows
        games = []

        # Crowd pick %/confidence from the sibling pick_distribution file, if present.
        # Paired positionally (both pages list games in the same order).
        dist_file = str(week_file).replace('confidence_picks_week', 'pick_distribution_week')
        crowd_by_index = parse_pick_distribution(dist_file)

        # Parse each game from the header columns
        num_games = len(games_meta[0])
        for game in range(num_games):
            try:
                favorite = games_meta[0][game].text.strip()
                underdog = games_meta[2][game].text.strip()

                winner = None
                if "yspNflPickWin" in games_meta[0][game].get("class", []):
                    winner = favorite
                elif "yspNflPickWin" in games_meta[2][game].get("class", []):
                    winner = underdog

                entry = {
                    'favorite': favorite,
                    'underdog': underdog,
                    'winner': winner,
                }

                if crowd_by_index and game < len(crowd_by_index):
                    c = crowd_by_index[game]
                    # Only trust the crowd row if the teams line up
                    if c.get('favorite') == favorite and c.get('underdog') == underdog:
                        entry['favorite_pick_pct'] = c['favorite_pick_pct']
                        entry['favorite_confidence'] = c['favorite_confidence']
                        entry['underdog_confidence'] = c['underdog_confidence']

                games.append(entry)
            except Exception as e:
                continue
        
        # Parse each player's picks
        player_rows = []
        
        # Find rows with player data (skip headers and empty rows)
        for row in all_rows:
            cols = row.find_all('td')
            if len(cols) > 5:  # Should have many columns
                first_col = cols[0]
                # Check if this looks like a player row (has link or non-header text)
                if first_col.find('a') or (first_col.text.strip() not in ['Favored', 'Spread', 'Underdog', '']):
                    player_rows.append(row)
        
        players_data = []
        for row in player_rows:
            cols = row.find_all('td')
            if not cols or len(cols) <= 1:
                continue
            
            try:
                # Extract player name
                name_elem = cols[0]
                if name_elem.find('a'):
                    player_name = name_elem.find('a').text.strip()
                else:
                    player_name = name_elem.text.strip()
                
                # Skip if this is still a header row
                if player_name in ['Favored', 'Spread', 'Underdog', '']:
                    continue
                
                # Extract total points
                total_text = cols[-1].text.strip()
                try:
                    total_points = int(total_text) if total_text and total_text != '' else 0
                except ValueError:
                    total_points = 0
                
                player_data = {
                    'player_name': player_name,
                    'total_points': total_points,
                    'picks': []
                }
                
                # Parse each game pick
                game_cols = cols[1:-1]  # Skip name and total columns
                for i, col in enumerate(game_cols):
                    pick_text = col.text.strip()
                    
                    if pick_text and pick_text not in ["", "--"] and i < len(games):
                        try:
                            # Parse "Team(confidence)" format
                            if '(' in pick_text and ')' in pick_text:
                                team = pick_text.split('(')[0].strip()
                                confidence = int(pick_text.split('(')[1].replace(')', ''))
                                
                                # Determine if pick was correct
                                is_correct = None
                                if games[i]['winner']:
                                    is_correct = team.strip() == games[i]['winner'].strip()
                                
                                player_data['picks'].append({
                                    'team': team,
                                    'confidence': confidence,
                                    'correct': is_correct,
                                    'points': confidence if is_correct else 0,
                                    'game_index': i,
                                })
                        except Exception:
                            continue
                
                players_data.append(player_data)
                
            except Exception as e:
                continue
        
        return games, players_data
        
    except Exception as e:
        print(f"❌ Error parsing {week_file}: {e}")
        return None, None

def analyze_player_skills(year):
    """Analyze all weeks to derive player skill levels"""
    print(f"📊 ANALYZING {year} PLAYER SKILLS")
    print("=" * 40)

    cache_dir = Path(f"PickEmCache{year}")
    if not cache_dir.exists():
        print(f"❌ PickEmCache{year} directory not found")
        return None
    
    # Find all week files
    week_files = list(cache_dir.glob("confidence_picks_week*.html"))
    week_files = [f for f in week_files if not f.name.endswith("_test.html")]
    week_files.sort()
    
    print(f"🔍 Found {len(week_files)} weeks of data")
    
    # Collect all player data
    all_player_stats = defaultdict(lambda: {
        'total_picks': 0,
        'total_correct': 0,
        'total_points': 0,
        'total_possible_points': 0,
        'weeks_played': 0,
        'confidence_distribution': defaultdict(int),
        'pick_accuracy_by_confidence': defaultdict(lambda: {'correct': 0, 'total': 0}),
        # Crowd behavior (populated only for weeks where pick_distribution parsed)
        'crowd_agree': 0,        # picks matching the crowd majority side
        'crowd_comparable': 0,   # picks where a crowd majority was known
        'conf_align_sum': 0.0,   # sum of 1 - |conf - crowd_conf| / max(conf, crowd_conf)
        'conf_align_n': 0,       # picks where crowd confidence was known
    })
    
    total_games = 0
    processed_weeks = 0
    
    for week_file in week_files:
        print(f"📖 Processing {week_file.name}...")
        
        games, players_data = parse_week_data(week_file)
        if not games or not players_data:
            continue
        
        completed_games = [g for g in games if g['winner']]
        if len(completed_games) == 0:
            print(f"   ⚠️ No completed games, skipping...")
            continue
        
        processed_weeks += 1
        total_games += len(completed_games)
        
        print(f"   ✅ {len(completed_games)} games, {len(players_data)} players")
        
        for player in players_data:
            name = player['player_name']
            stats = all_player_stats[name]
            
            stats['weeks_played'] += 1
            
            # Analyze picks
            for pick_idx, pick in enumerate(player['picks']):
                if pick['correct'] is not None:  # Only count completed games
                    stats['total_picks'] += 1
                    stats['total_possible_points'] += pick['confidence']

                    if pick['correct']:
                        stats['total_correct'] += 1
                        stats['total_points'] += pick['confidence']

                    # Track confidence usage
                    conf = pick['confidence']
                    stats['confidence_distribution'][conf] += 1

                    # Track accuracy by confidence level
                    conf_stats = stats['pick_accuracy_by_confidence'][conf]
                    conf_stats['total'] += 1
                    if pick['correct']:
                        conf_stats['correct'] += 1

                    # Crowd behavior, when we have crowd data for this game
                    g_idx = pick.get('game_index', pick_idx)
                    g = games[g_idx] if g_idx < len(games) else {}
                    fav_pct = g.get('favorite_pick_pct')
                    if fav_pct is not None:
                        picked_favorite = (pick['team'] == g['favorite'])
                        majority_favorite = fav_pct > 50
                        stats['crowd_comparable'] += 1
                        if picked_favorite == majority_favorite:
                            stats['crowd_agree'] += 1

                    fav_conf = g.get('favorite_confidence')
                    und_conf = g.get('underdog_confidence')
                    if fav_conf is not None and und_conf is not None:
                        crowd_conf = fav_conf if pick['team'] == g['favorite'] else und_conf
                        pts = pick['confidence']
                        denom = max(pts, crowd_conf)
                        if denom > 0:
                            stats['conf_align_sum'] += 1 - abs(pts - crowd_conf) / denom
                            stats['conf_align_n'] += 1
    
    print(f"\n📈 ANALYSIS COMPLETE:")
    print(f"   📊 {processed_weeks} weeks processed")
    print(f"   🏈 {total_games} total games")
    print(f"   👥 {len(all_player_stats)} players analyzed")
    
    player_skills = skills_from_raw_stats(all_player_stats)
    return player_skills, all_player_stats


MIN_PICKS = 20  # players with fewer scored picks are excluded from the skill table


def skills_from_raw_stats(raw_stats):
    """Derive the three 0-1 behavioural knobs for every player with enough data.

    - skill_level: pick accuracy, spread across the league by z-score so the
      0.6-0.7 accuracy band maps to a usable 0-1 range (raw accuracy alone
      barely varies between players).
    - crowd_following: fraction of picks that sided with the crowd majority.
    - confidence_following: mean alignment between the player's assigned points
      and the crowd's average confidence for that side.

    crowd_following / confidence_following fall back to 0.5 for players with no
    crowd data (weeks where pick_distribution didn't parse).
    """
    eligible = {n: s for n, s in raw_stats.items() if s['total_picks'] >= MIN_PICKS}
    if not eligible:
        return {}

    accuracies = {n: s['total_correct'] / s['total_picks'] for n, s in eligible.items()}
    acc_vals = list(accuracies.values())
    acc_mean = float(np.mean(acc_vals))
    acc_std = float(np.std(acc_vals))
    # With a real league the accuracy spread is small (~0.03), so a z-score
    # spreads it into a usable range. With one or two players (tests, tiny
    # pools) there's no spread to work with - fall back to a direct monotonic
    # map so skill_level still tracks accuracy.
    use_zscore = len(eligible) >= 3 and acc_std > 1e-6

    player_skills = {}
    for name, stats in eligible.items():
        accuracy = accuracies[name]
        efficiency = (stats['total_points'] / stats['total_possible_points']
                      if stats['total_possible_points'] > 0 else 0)

        if use_zscore:
            z = (accuracy - acc_mean) / acc_std
            skill_level = float(np.clip(0.5 + z * 0.18, 0.15, 0.95))
        else:
            skill_level = float(np.clip(0.15 + accuracy * 0.8, 0.15, 0.95))

        if stats.get('crowd_comparable', 0) > 0:
            crowd_following = stats['crowd_agree'] / stats['crowd_comparable']
        else:
            crowd_following = 0.5

        if stats.get('conf_align_n', 0) > 0:
            confidence_following = stats['conf_align_sum'] / stats['conf_align_n']
        else:
            confidence_following = 0.5
        confidence_following = float(np.clip(confidence_following, 0.0, 1.0))

        player_skills[name] = {
            'skill_level': skill_level,
            'crowd_following': crowd_following,
            'confidence_following': confidence_following,
            'accuracy': accuracy,
            'efficiency': efficiency,
            'weeks_played': stats['weeks_played'],
            'total_picks': stats['total_picks'],
        }

    return player_skills

def main():
    """Analyze player skills and save results"""
    parser = argparse.ArgumentParser(description='Analyze player performance to derive realistic skill levels')
    parser.add_argument('--year', type=int, default=2024, help='Year to analyze (default: 2024)')
    args = parser.parse_args()

    result = analyze_player_skills(args.year)

    if not result:
        return 1

    player_skills, all_player_stats = result
    
    # Sort by number of picks (most data first)
    sorted_players = sorted(player_skills.items(), key=lambda x: x[1]['total_picks'], reverse=True)
    
    print(f"\n🎯 PLAYER SKILL ANALYSIS:")
    print(f"{'Player':<25} {'Skill':<6} {'Crowd':<6} {'Conf':<6} {'Acc%':<6} {'Eff%':<6} {'Picks'}")
    print("-" * 80)
    
    skill_levels = []
    crowd_followings = []
    confidence_followings = []
    
    for i, (name, data) in enumerate(sorted_players[:20], 1):  # Top 20
        skill = data['skill_level']
        crowd = data['crowd_following']
        conf = data['confidence_following']
        acc = data['accuracy']
        eff = data['efficiency']
        picks = data['total_picks']
        
        skill_levels.append(skill)
        crowd_followings.append(crowd)
        confidence_followings.append(conf)
        
        print(f"{name:<25} {skill:.3f}  {crowd:.3f}  {conf:.3f}  {acc:.1%}  {eff:.1%}  {picks}")
    
    # Show distribution statistics
    print(f"\n📊 SKILL DISTRIBUTION STATISTICS:")
    print(f"   Skill Level: μ={np.mean(skill_levels):.3f}, σ={np.std(skill_levels):.3f}, range=[{min(skill_levels):.3f}, {max(skill_levels):.3f}]")
    print(f"   Crowd Following: μ={np.mean(crowd_followings):.3f}, σ={np.std(crowd_followings):.3f}")
    print(f"   Confidence Following: μ={np.mean(confidence_followings):.3f}, σ={np.std(confidence_followings):.3f}")
    
    # Save results including raw stats for future aggregation
    results = {
        'player_skills': player_skills,
        'raw_player_stats': dict(all_player_stats),  # Save raw stats for combining years
        'distribution_stats': {
            'skill_level': {'mean': np.mean(skill_levels), 'std': np.std(skill_levels)},
            'crowd_following': {'mean': np.mean(crowd_followings), 'std': np.std(crowd_followings)},
            'confidence_following': {'mean': np.mean(confidence_followings), 'std': np.std(confidence_followings)}
        }
    }
    
    output_file = f'player_skills_{args.year}.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n💾 Results saved to {output_file}")
    print(f"💡 Use this data to create realistic player skill distributions!")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())