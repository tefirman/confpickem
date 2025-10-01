#!/usr/bin/env python
"""Unified CLI for confpickem - Confidence Pick'Em Pool Optimization"""

import sys
import argparse
from pathlib import Path
import pandas as pd
import json
from datetime import datetime
import numpy as np
from collections import defaultdict

from .yahoo_pickem_scraper import YahooPickEm
from .live_odds_scraper import update_odds_with_live_data
from .confidence_pickem_sim import ConfidencePickEmSimulator, Player


def cmd_optimize(args):
    """Optimize picks with optional live odds and mid-week support"""

    week = args.week
    league_id = args.league_id
    num_sims = args.num_sims
    mid_week = args.mid_week
    live_odds = args.live_odds
    odds_api_key = args.odds_api_key

    mode_str = "MID-WEEK " if mid_week else ""
    odds_str = "WITH LIVE ODDS " if live_odds else ""
    print(f"🎯 {mode_str}OPTIMIZATION {odds_str}- WEEK {week}")
    if mid_week and live_odds:
        print("🏈 Accounts for completed games + Live Vegas odds!")
    print("=" * 55)

    if not Path("cookies.txt").exists():
        print("❌ Missing cookies.txt file")
        return 1

    try:
        # Clear cache for fresh data if mid-week
        if mid_week:
            cache_dir = Path(".cache")
            if cache_dir.exists():
                import shutil
                shutil.rmtree(cache_dir)

        # Load Yahoo data
        print("📊 Loading Yahoo pick distribution and player data...")
        yahoo = YahooPickEm(week=week, league_id=league_id, cookies_file="cookies.txt")
        print(f"✅ Loaded {len(yahoo.players)} players, {len(yahoo.games)} games")

        if len(yahoo.games) == 0:
            print("❌ No games found! This usually means:")
            print("   🍪 Your cookies.txt file may be expired")
            print("   🔗 League ID might be incorrect")
            print("   📅 Week number might be wrong")
            return 1

        # Check for completed games (mid-week scenario)
        completed_games = []
        if mid_week:
            completed_games = [r for r in yahoo.results if r['winner']]
            print(f"🏆 {len(completed_games)} games completed, {len(yahoo.games) - len(completed_games)} remaining")

            if completed_games:
                print("   Completed games:")
                for game in completed_games:
                    print(f"     ✅ {game['winner']} beat opponent (spread: {game['spread']})")

        # Show original Yahoo odds
        print(f"\n📋 Original Yahoo odds:")
        for i, (_, game) in enumerate(yahoo.games.iterrows()):
            fav = game['favorite']
            dog = game['underdog']
            spread = game['spread']
            prob = game['win_prob']
            home_favorite = game.get('home_favorite', True)

            if home_favorite:
                home_team, away_team = fav, dog
            else:
                home_team, away_team = dog, fav

            print(f"   {i+1:2d}. {away_team:3s} @ {home_team:3s} | {fav} -{spread:4.1f} | Win prob: {prob:.1%}")

        # Update with live odds if requested
        enhanced_games = yahoo.games
        live_updates = 0
        if live_odds:
            print(f"\n🔄 Checking for live odds updates...")
            enhanced_games = update_odds_with_live_data(
                yahoo.games,
                week=week,
                odds_api_key=odds_api_key
            )

            # Show what changed
            print(f"\n📈 Updated odds:")
            for i, (_, game) in enumerate(enhanced_games.iterrows()):
                fav = game['favorite']
                dog = game['underdog']
                spread = game['spread']
                prob = game['win_prob']
                source = game.get('live_odds_source', 'Yahoo')
                home_favorite = game.get('home_favorite', True)

                if home_favorite:
                    home_team, away_team = fav, dog
                else:
                    home_team, away_team = dog, fav

                indicator = "🔴" if source != 'Yahoo_Fallback' else "  "
                print(f"   {i+1:2d}. {away_team:3s} @ {home_team:3s} | {fav} -{spread:4.1f} | Win prob: {prob:.1%} {indicator}")

                if source != 'Yahoo_Fallback':
                    live_updates += 1

            print(f"\n💡 Updated {live_updates}/{len(enhanced_games)} games with live odds")

        # Setup simulator
        print(f"\n🎲 Setting up optimization...")
        simulator = ConfidencePickEmSimulator(num_sims=num_sims)

        # Convert games to simulator format
        games_data = []
        for _, game_row in enhanced_games.iterrows():
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

            # Check for actual outcomes (mid-week)
            actual_outcome = None
            if mid_week:
                for completed in yahoo.results:
                    if completed['winner']:
                        game_teams = {completed['favorite'], completed['underdog']}
                        our_teams = {home_team, away_team}
                        if game_teams == our_teams:
                            actual_outcome = (completed['winner'] == home_team)
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

        simulator.add_games_from_dataframe(pd.DataFrame(games_data))

        # Load player skills if available
        try:
            with open('current_player_skills.json', 'r') as f:
                player_skills = json.load(f)
            print("✅ Using realistic 2024-derived player skills")
        except FileNotFoundError:
            print("⚠️ current_player_skills.json not found, using defaults")
            player_skills = {}

        # Add all league players
        players = []
        for _, player in yahoo.players.iterrows():
            name = player['player_name']
            if name in player_skills:
                skill_data = player_skills[name]
                skill = skill_data['skill_level']
                crowd = skill_data['crowd_following']
                confidence = skill_data['confidence_following']
            else:
                skill, crowd, confidence = 0.6, 0.5, 0.5

            players.append(Player(
                name=name,
                skill_level=skill,
                crowd_following=crowd,
                confidence_following=confidence
            ))

        simulator.players = players
        print(f"✅ Added {len(players)} league players")

        # Calculate current standings if mid-week
        completed_standings = {}
        if mid_week and completed_games:
            print(f"\n🎮 SIMULATION SETUP:")
            completed_count = sum(1 for game in simulator.games if game.actual_outcome is not None)
            remaining_count = len(simulator.games) - completed_count
            print(f"   ✅ {completed_count} games with known outcomes")
            print(f"   🎲 {remaining_count} games to simulate")
            print(f"   🔮 {num_sims:,} simulations per remaining game")

            # Calculate standings from completed games
            for _, player in yahoo.players.iterrows():
                player_name = player['player_name']
                points_earned = 0

                for i, game in enumerate(simulator.games):
                    if game.actual_outcome is not None:
                        game_num = i + 1
                        pick = player.get(f'game_{game_num}_pick')
                        confidence = player.get(f'game_{game_num}_confidence', 0)

                        if pick:
                            if (pick == game.home_team and game.actual_outcome) or \
                               (pick == game.away_team and not game.actual_outcome):
                                points_earned += confidence

                completed_standings[player_name] = points_earned

            # Show top performers
            sorted_standings = sorted(completed_standings.items(), key=lambda x: x[1], reverse=True)
            print(f"\n   Current leaders from completed games:")
            for i, (name, points) in enumerate(sorted_standings[:5], 1):
                print(f"     {i}. {name}: {points} points")

        # Player selection
        player_names = [p.name for p in simulator.players]
        print(f"\n👥 Select your player from {len(player_names)} total:")

        for i in range(0, min(15, len(player_names)), 3):
            row_players = player_names[i:i+3]
            for j, name in enumerate(row_players):
                current_points = completed_standings.get(name, 0) if mid_week and completed_games else 0
                points_str = f"({current_points} pts)" if mid_week and completed_games else ""
                print(f"   {i+j+1:2d}. {name:<25} {points_str}", end="")
            print()

        if len(player_names) > 15:
            print(f"   ... and {len(player_names) - 15} more")

        choice = input(f"\nPlayer (number or name): ").strip()

        # Find player
        selected = None
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(player_names):
                selected = player_names[idx]
        except ValueError:
            matches = [p for p in player_names if choice.lower() in p.lower()]
            if len(matches) == 1:
                selected = matches[0]
            else:
                print(f"❌ Player not found: {choice}")
                return 1

        print(f"✅ Selected: {selected}")

        # Calculate used/available confidence levels (mid-week)
        your_used_confidence = set()
        your_remaining_confidence = None
        if mid_week and completed_games:
            your_points = completed_standings.get(selected, 0)
            your_rank = sorted(completed_standings.values(), reverse=True).index(your_points) + 1
            print(f"📊 Your current position: #{your_rank} with {your_points} points from completed games")

            # Find actual confidence levels used
            your_player_data = None
            for _, player in yahoo.players.iterrows():
                if player['player_name'] == selected:
                    your_player_data = player
                    break

            if your_player_data is not None:
                print(f"\n🎯 YOUR COMPLETED GAME PICKS:")
                for i, game in enumerate(simulator.games):
                    if game.actual_outcome is not None:
                        game_num = i + 1
                        pick = your_player_data.get(f'game_{game_num}_pick')
                        confidence = your_player_data.get(f'game_{game_num}_confidence')

                        if pick and confidence:
                            your_used_confidence.add(confidence)
                            was_correct = (pick == game.home_team and game.actual_outcome) or \
                                        (pick == game.away_team and not game.actual_outcome)
                            status = "✅" if was_correct else "❌"
                            print(f"     {status} {pick} ({confidence} pts) - {'WIN' if was_correct else 'LOSS'}")

                # Calculate remaining confidence
                all_confidence = set(range(1, 17))
                your_remaining_confidence = all_confidence - your_used_confidence
                print(f"\n📊 CONFIDENCE LEVELS:")
                print(f"   Used in completed games: {sorted(your_used_confidence)}")
                print(f"   Available for remaining games: {sorted(your_remaining_confidence)}")

        # Get available teams for remaining games
        available_teams = set()
        remaining_games = []
        for i, game_sim in enumerate(simulator.games):
            if game_sim.actual_outcome is None:
                available_teams.add(game_sim.home_team)
                available_teams.add(game_sim.away_team)
                remaining_games.append({
                    'home': game_sim.home_team,
                    'away': game_sim.away_team
                })

        available_teams = sorted(list(available_teams))

        if mid_week and remaining_games:
            print(f"\n🏈 {len(remaining_games)} REMAINING GAMES TO OPTIMIZE:")
            for i, game in enumerate(remaining_games[:5], 1):
                print(f"   {i}. {game['away']} @ {game['home']}")
            if len(remaining_games) > 5:
                print(f"   ... and {len(remaining_games) - 5} more games")

            print(f"\n📋 Available teams for remaining games:")
            for i in range(0, len(available_teams), 8):
                row_teams = available_teams[i:i+8]
                print(f"   {' '.join(f'{team:<5}' for team in row_teams)}")

        # Fixed picks
        print(f"\n📌 Lock in any high-confidence picks?")
        print(f"   Examples: 'Phi 16, KC 15'")
        print(f"   Or press Enter to optimize all games")
        fixed_input = input("Fixed picks: ").strip()

        fixed_picks = None
        if fixed_input:
            fixed_picks = {}
            for pick in fixed_input.split(','):
                try:
                    parts = pick.strip().split()
                    if len(parts) >= 2:
                        team_input, conf = parts[0].strip(), int(parts[1])

                        # Find matching team
                        matched_team = None
                        teams_to_check = available_teams if mid_week else [t for g in yahoo.games.iterrows() for t in [g[1]['favorite'], g[1]['underdog']]]
                        for team in teams_to_check:
                            if team.lower() == team_input.lower():
                                matched_team = team
                                break

                        if matched_team:
                            fixed_picks[matched_team] = conf
                            print(f"✅ {team_input} -> {matched_team} ({conf} pts)")
                        else:
                            print(f"❌ Team '{team_input}' not available")
                except Exception as e:
                    print(f"⚠️ Could not parse '{pick}': {e}")

            if not fixed_picks:
                fixed_picks = None

        # Run optimization
        print(f"\n🚀 Starting optimization...")
        print(f"   🎲 {num_sims:,} simulations")
        if live_odds:
            print(f"   🔴 Enhanced with live Vegas odds ({live_updates} games)")
        if mid_week:
            print(f"   ✅ Using actual results for {len(completed_games)} completed games")

        try:
            fixed_formatted = {selected: fixed_picks} if fixed_picks else None
            remaining_confidence = your_remaining_confidence if mid_week and completed_games else None
            optimal_picks = simulator.optimize_picks(
                player_name=selected,
                fixed_picks=fixed_formatted,
                confidence_range=4,
                available_points=remaining_confidence
            )

            if optimal_picks:
                print("\n🏆 OPTIMIZATION RESULTS:")
                print("=" * 50)

                # Performance analysis
                optimal_fixed = {selected: optimal_picks}
                optimal_stats = simulator.simulate_all(optimal_fixed)
                random_stats = simulator.simulate_all({})

                opt_win = optimal_stats['win_pct'][selected]
                rand_win = random_stats['win_pct'][selected]

                print(f"📈 Win Probability:")
                print(f"   🎯 Optimized strategy: {opt_win:.1%}")
                print(f"   🎲 Random picks: {rand_win:.1%}")
                print(f"   💪 Advantage: +{(opt_win - rand_win)*100:.1f} percentage points")

                print(f"\n🎯 OPTIMIZED PICKS:")
                sorted_picks = sorted(optimal_picks.items(), key=lambda x: x[1], reverse=True)

                for team, conf in sorted_picks:
                    opponent = "Unknown"
                    for _, game in enhanced_games.iterrows():
                        if team in [game['favorite'], game['underdog']]:
                            opponent = game['underdog'] if team == game['favorite'] else game['favorite']
                            break

                    print(f"   {conf:2d}. {team} vs {opponent}")

                # Copy-paste format
                print(f"\n📋 COPY-PASTE FORMAT:")
                paste_format = ", ".join(f"{team} {conf}" for team, conf in sorted_picks)
                print(f"   {paste_format}")

                # Save results
                timestamp = datetime.now().strftime("%Y%m%d_%H%M")
                safe_name = selected.replace(' ', '_').replace('/', '_')
                mode_suffix = "MidWeek" if mid_week else ""
                odds_suffix = "LiveOdds" if live_odds else ""
                filename = f"NFL_Week{week}_{mode_suffix}{odds_suffix}_{safe_name}_{timestamp}.txt"

                with open(filename, 'w') as f:
                    f.write(f"NFL Week {week} Optimized Picks\n")
                    f.write(f"Player: {selected}\n")
                    f.write(f"Generated: {datetime.now()}\n")
                    f.write(f"Mode: {'Mid-week ' if mid_week else ''}{'Live odds ' if live_odds else ''}\n")
                    f.write(f"Win probability: {opt_win:.1%}\n\n")
                    f.write(f"OPTIMIZED PICKS:\n")
                    for team, conf in sorted_picks:
                        f.write(f"{conf:2d}. {team}\n")
                    f.write(f"\nCOPY-PASTE FORMAT:\n")
                    f.write(f"{paste_format}\n")

                print(f"\n💾 Results saved: {filename}")
                print("\n🎯 Optimization complete!")

            else:
                print("❌ Optimization failed")
                return 1

        except Exception as e:
            print(f"❌ Optimization error: {e}")
            import traceback
            traceback.print_exc()
            return 1

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


def cmd_winprob(args):
    """Calculate win probabilities for all players"""

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

        # Run simulation
        def simulate_remaining_games(results, all_picks, num_sims=5000):
            """Simulate remaining games using Vegas probabilities"""
            pending_games = [(i, r) for i, r in enumerate(results) if not r['winner']]
            completed_games = [(i, r) for i, r in enumerate(results) if r['winner']]

            print(f"🎲 Simulating {len(pending_games)} remaining games with {num_sims:,} iterations...")

            # Calculate current points
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
                sim_standings = current_standings.copy()

                # Simulate each pending game
                for game_idx, result in pending_games:
                    spread = result.get('spread', 0.0)
                    favorite_win_prob = min(max(spread * 0.031 + 0.5, 0.0), 1.0) if spread != 0 else 0.5

                    winner = result['favorite'] if np.random.random() < favorite_win_prob else result['underdog']

                    # Add points
                    game_num = game_idx + 1
                    for player_name, picks in all_picks.items():
                        pick = picks.get(f'game_{game_num}_pick')
                        confidence = picks.get(f'game_{game_num}_confidence', 0)

                        if pick and pick == winner:
                            sim_standings[player_name] += confidence

                # Find winner
                max_points = max(sim_standings.values())
                winners = [name for name, points in sim_standings.items() if points == max_points]
                winner = np.random.choice(winners)
                win_counts[winner] += 1

            win_probs = {name: count / num_sims for name, count in win_counts.items()}
            return win_probs, current_standings

        win_probs, current_standings = simulate_remaining_games(yahoo.results, all_picks)

        # Show results
        print(f"\n🏆 WIN PROBABILITY RANKINGS:")
        print(f"{'Rank':<4} {'Player':<25} {'Win %':<8} {'Current'}")
        print("-" * 50)

        sorted_probs = sorted(win_probs.items(), key=lambda x: x[1], reverse=True)

        for i, (name, prob) in enumerate(sorted_probs[:20], 1):
            current_pts = current_standings.get(name, 0)
            print(f"{i:2d}.  {name:<25} {prob:6.1%}   {current_pts} pts")

        if len(sorted_probs) > 20:
            print(f"... and {len(sorted_probs) - 20} more players")

        print(f"\n🎲 Simulation uses Vegas spreads for realistic win probabilities")
        print(f"💡 Based on 5000 Monte Carlo simulations")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


def cmd_test_picks(args):
    """Test and verify optimized picks"""

    week = args.week
    league_id = args.league_id

    print(f"🧪 PICK VERIFICATION TOOL - WEEK {week}")
    print("=" * 35)

    if not Path("cookies.txt").exists():
        print("❌ Missing cookies.txt file")
        return 1

    print("📡 Loading league data...")
    try:
        # Clear cache
        cache_dir = Path(".cache")
        if cache_dir.exists():
            import shutil
            shutil.rmtree(cache_dir)

        yahoo = YahooPickEm(week=week, league_id=league_id, cookies_file="cookies.txt")
        print(f"✅ Loaded {len(yahoo.games)} games, {len(yahoo.players)} players")

        # Setup simulator
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
                'week': week,
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

        # Get player
        player_names = [p.name for p in simulator.players]
        print(f"\n👥 Available players:")
        for i in range(0, min(10, len(player_names)), 2):
            row_players = player_names[i:i+2]
            for j, name in enumerate(row_players):
                print(f"   {i+j+1:2d}. {name}")
        if len(player_names) > 10:
            print(f"   ... and {len(player_names) - 10} more")

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

        picks_input = input("Your picks: ").strip()

        if not picks_input:
            print("❌ No picks entered")
            return 1

        # Parse picks
        user_picks = {}
        for pick in picks_input.split(','):
            parts = pick.strip().split()
            if len(parts) >= 2:
                team_input = parts[0].strip()
                confidence = int(parts[1])

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

        if not user_picks:
            print("❌ No valid picks found")
            return 1

        print(f"\n🧪 RUNNING VERIFICATION SIMULATION...")
        print(f"   📊 Using 15,000 simulations for accuracy")

        # Test picks
        test_picks = {selected_player: user_picks}
        results = simulator.simulate_all(test_picks)
        random_results = simulator.simulate_all({})

        your_win_prob = results['win_pct'][selected_player]
        random_win_prob = random_results['win_pct'][selected_player]
        baseline = 1.0 / len(simulator.players)

        print(f"\n🏆 VERIFICATION RESULTS:")
        print("=" * 40)
        print(f"📈 Your Optimized Picks:")
        print(f"   🎯 Win Probability: {your_win_prob:.1%}")
        print(f"   💪 vs Random: {your_win_prob/random_win_prob:.1f}x better")
        print(f"   🚀 vs Pure Luck: {your_win_prob/baseline:.1f}x better")

        print(f"\n✅ Verification complete!")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


def cmd_analyze_skills(args):
    """Analyze player skills from historical data"""

    from bs4 import BeautifulSoup

    print("📊 ANALYZING 2024 PLAYER SKILLS")
    print("=" * 40)

    cache_dir = Path("PickEmCache2024")
    if not cache_dir.exists():
        print("❌ PickEmCache2024 directory not found")
        return 1

    # Find all week files
    week_files = list(cache_dir.glob("confidence_picks_week*.html"))
    week_files = [f for f in week_files if not f.name.endswith("_test.html")]
    week_files.sort()

    print(f"🔍 Found {len(week_files)} weeks of data")

    # Parse and analyze (simplified version)
    print(f"⚠️  Full analysis requires parsing HTML files")
    print(f"💡 Run the original analyze_player_skills.py for complete analysis")

    return 0


def main():
    """Main entry point for CLI"""
    parser = argparse.ArgumentParser(
        description='confpickem - Confidence Pick\'Em Pool Optimization',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  confpickem optimize --week 3 --league-id 15435
  confpickem optimize --week 4 --mid-week --live-odds
  confpickem winprob --week 3
  confpickem test-picks --week 3
        '''
    )

    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Optimize command
    parser_optimize = subparsers.add_parser('optimize', help='Optimize picks')
    parser_optimize.add_argument('--week', '-w', type=int, default=3, help='NFL week number')
    parser_optimize.add_argument('--league-id', '-l', type=int, default=15435, help='Yahoo league ID')
    parser_optimize.add_argument('--num-sims', '-n', type=int, default=1000, help='Number of simulations')
    parser_optimize.add_argument('--mid-week', '-m', action='store_true', help='Mid-week mode (accounts for completed games)')
    parser_optimize.add_argument('--live-odds', action='store_true', help='Use live Vegas odds')
    parser_optimize.add_argument('--odds-api-key', '-k', type=str, help='The Odds API key')
    parser_optimize.set_defaults(func=cmd_optimize)

    # Winprob command
    parser_winprob = subparsers.add_parser('winprob', help='Calculate win probabilities')
    parser_winprob.add_argument('--week', '-w', type=int, default=3, help='NFL week number')
    parser_winprob.add_argument('--league-id', '-l', type=int, default=15435, help='Yahoo league ID')
    parser_winprob.set_defaults(func=cmd_winprob)

    # Test-picks command
    parser_test = subparsers.add_parser('test-picks', help='Test and verify picks')
    parser_test.add_argument('--week', '-w', type=int, default=3, help='NFL week number')
    parser_test.add_argument('--league-id', '-l', type=int, default=15435, help='Yahoo league ID')
    parser_test.set_defaults(func=cmd_test_picks)

    # Analyze-skills command
    parser_skills = subparsers.add_parser('analyze-skills', help='Analyze player skills from historical data')
    parser_skills.set_defaults(func=cmd_analyze_skills)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())