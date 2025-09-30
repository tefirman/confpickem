#!/usr/bin/env python
"""
Mid-week optimization with live odds integration
Accounts for completed games AND uses live Vegas odds when available
"""

import sys
import argparse
from pathlib import Path
import pandas as pd
import json
from datetime import datetime

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.confpickem.yahoo_pickem_scraper import YahooPickEm
from src.confpickem.live_odds_scraper import update_odds_with_live_data
from src.confpickem.confidence_pickem_sim import ConfidencePickEmSimulator, Player


def main():
    """Mid-week optimization with live odds integration"""
    parser = argparse.ArgumentParser(description='Mid-week NFL pick optimization with live Vegas odds')
    parser.add_argument('--week', '-w', type=int, default=4,
                       help='NFL week number (default: 4)')
    parser.add_argument('--league-id', '-l', type=int, default=15435,
                       help='Yahoo league ID (default: 15435)')
    parser.add_argument('--odds-api-key', '-k', type=str,
                       help='The Odds API key for live odds (optional)')
    parser.add_argument('--num-sims', '-n', type=int, default=2000,
                       help='Number of simulations (default: 2000)')
    args = parser.parse_args()

    print(f"🎯 MID-WEEK OPTIMIZATION WITH LIVE ODDS - WEEK {args.week}")
    print("🏈 Accounts for completed games + Live Vegas odds!")
    print("=" * 55)

    if not Path("cookies.txt").exists():
        print("❌ Missing cookies.txt file")
        return 1

    try:
        # Clear cache for fresh data
        cache_dir = Path(".cache")
        if cache_dir.exists():
            import shutil
            shutil.rmtree(cache_dir)

        # Load Yahoo data
        print("📊 Loading Yahoo pick distribution and player data...")
        yahoo = YahooPickEm(week=args.week, league_id=args.league_id, cookies_file="cookies.txt")
        print(f"✅ Loaded {len(yahoo.players)} players, {len(yahoo.games)} games")

        # Check for valid data extraction
        if len(yahoo.games) == 0:
            print("❌ No games found! This usually means:")
            print("   🍪 Your cookies.txt file may be expired")
            print("   🔗 League ID might be incorrect")
            print("   📅 Week number might be wrong")
            return 1

        # Check for completed games
        completed_games = [r for r in yahoo.results if r['winner']]
        print(f"🏆 {len(completed_games)} games completed, {len(yahoo.games) - len(completed_games)} remaining")

        if completed_games:
            print("   Completed games:")
            for game in completed_games:
                print(f"     ✅ {game['winner']} beat opponent (spread: {game['spread']})")

        # Show original Yahoo odds
        print(f"\n📋 Original Yahoo odds:")
        for i, (_, game) in enumerate(yahoo.games.iterrows()):
            fav = game['favorite']  # betting favorite
            dog = game['underdog']  # betting underdog
            spread = game['spread']
            prob = game['win_prob']
            home_favorite = game.get('home_favorite', True)

            # Determine actual home/away teams
            if home_favorite:
                home_team, away_team = fav, dog  # favorite is home
            else:
                home_team, away_team = dog, fav  # underdog is home

            print(f"   {i+1:2d}. {away_team:3s} @ {home_team:3s} | {fav} -{spread:4.1f} | Win prob: {prob:.1%}")

        # Update with live odds
        print(f"\n🔄 Checking for live odds updates...")
        enhanced_games = update_odds_with_live_data(
            yahoo.games,
            week=args.week,
            odds_api_key=args.odds_api_key
        )

        # Show what changed
        live_updates = 0
        print(f"\n📈 Updated odds:")
        for i, (_, game) in enumerate(enhanced_games.iterrows()):
            fav = game['favorite']  # betting favorite
            dog = game['underdog']  # betting underdog
            spread = game['spread']
            prob = game['win_prob']
            source = game.get('live_odds_source', 'Yahoo')
            home_favorite = game.get('home_favorite', True)

            # Determine actual home/away teams
            if home_favorite:
                home_team, away_team = fav, dog  # favorite is home
            else:
                home_team, away_team = dog, fav  # underdog is home

            indicator = "🔴" if source != 'Yahoo_Fallback' else "  "
            print(f"   {i+1:2d}. {away_team:3s} @ {home_team:3s} | {fav} -{spread:4.1f} | Win prob: {prob:.1%} {indicator}")

            if source != 'Yahoo_Fallback':
                live_updates += 1

        print(f"\n💡 Updated {live_updates}/{len(enhanced_games)} games with live odds")

        # Show detailed live odds for spot checking
        if live_updates > 0:
            print(f"\n🔍 LIVE VEGAS ODDS DETAILS (for spot checking):")
            for i, (_, game) in enumerate(enhanced_games.iterrows()):
                source = game.get('live_odds_source', 'Yahoo')
                if source != 'Yahoo_Fallback':
                    fav = game['favorite']  # betting favorite
                    dog = game['underdog']  # betting underdog
                    spread = game['spread']
                    home_favorite = game.get('home_favorite', True)

                    # Determine actual home/away teams
                    if home_favorite:
                        home_team, away_team = fav, dog  # favorite is home
                    else:
                        home_team, away_team = dog, fav  # underdog is home

                    # Get the raw live odds data if available
                    live_spread = game.get('live_spread', spread)
                    live_source_name = game.get('live_odds_source', 'Unknown')

                    print(f"   🔴 Game {i+1}: {away_team} @ {home_team}")
                    print(f"      Vegas Source: {live_source_name}")
                    print(f"      Live Spread: {fav} -{live_spread}")
                    print(f"      Converted Win Prob: {game['win_prob']:.1%}")

                    # Show original vs live if we have the data
                    original_spread = game.get('original_spread', 'N/A')
                    if original_spread != 'N/A' and abs(float(live_spread) - float(original_spread)) > 0.1:
                        print(f"      ⚠️  Significant change from Yahoo: {original_spread} → {live_spread}")
                    print()

        # Setup simulator with enhanced odds
        print(f"\n🎲 Setting up optimization with enhanced odds...")
        simulator = ConfidencePickEmSimulator(num_sims=args.num_sims)

        # Convert enhanced games to simulator format WITH actual outcomes
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

            # Check if we have actual outcome for this game
            actual_outcome = None
            for completed in yahoo.results:
                if completed['winner']:
                    # Match by team names regardless of home/away or favorite/underdog
                    game_teams = {completed['favorite'], completed['underdog']}
                    our_teams = {home_team, away_team}
                    if game_teams == our_teams:
                        # Determine if home team won
                        actual_outcome = (completed['winner'] == home_team)
                        break

            games_data.append({
                'home_team': home_team,
                'away_team': away_team,
                'vegas_win_prob': home_prob,
                'crowd_home_pick_pct': crowd_home_pct,
                'crowd_home_confidence': home_conf,
                'crowd_away_confidence': away_conf,
                'week': args.week,
                'kickoff_time': game_row['kickoff_time'],
                'actual_outcome': actual_outcome  # Key difference from beginning-of-week version!
            })

        simulator.add_games_from_dataframe(pd.DataFrame(games_data))

        # Load realistic player skills
        try:
            with open('current_player_skills.json', 'r') as f:
                player_skills = json.load(f)
            print("✅ Using realistic 2024-derived player skills")
        except FileNotFoundError:
            print("⚠️ current_player_skills.json not found, using defaults")
            player_skills = {}

        # Add all league players as opponents + the hypothetical player we're optimizing for
        players = []
        for _, player in yahoo.players.iterrows():
            name = player['player_name']
            if name in player_skills:
                skill_data = player_skills[name]
                skill = skill_data['skill_level']
                crowd = skill_data['crowd_following']
                confidence = skill_data['confidence_following']
            else:
                # Fallback to defaults
                skill, crowd, confidence = 0.6, 0.5, 0.5

            players.append(Player(
                name=name,
                skill_level=skill,
                crowd_following=crowd,
                confidence_following=confidence
            ))

        simulator.players = players
        print(f"✅ Added {len(players)} league players")

        # Show game status
        print(f"\n🎮 SIMULATION SETUP:")
        completed_count = sum(1 for game in simulator.games if game.actual_outcome is not None)
        remaining_count = len(simulator.games) - completed_count
        print(f"   ✅ {completed_count} games with known outcomes")
        print(f"   🎲 {remaining_count} games to simulate")
        print(f"   🔮 {args.num_sims:,} simulations per remaining game")

        # Show completed games impact
        if completed_count > 0:
            print(f"\n📊 COMPLETED GAMES IMPACT:")
            print("   (These results are locked in for everyone)")

            # Calculate current standings from completed games only
            completed_standings = {}
            for _, player in yahoo.players.iterrows():
                player_name = player['player_name']
                points_earned = 0

                for i, game in enumerate(simulator.games):
                    if game.actual_outcome is not None:
                        game_num = i + 1
                        pick = player.get(f'game_{game_num}_pick')
                        confidence = player.get(f'game_{game_num}_confidence', 0)

                        if pick:
                            # Check if pick was correct
                            if (pick == game.home_team and game.actual_outcome) or \
                               (pick == game.away_team and not game.actual_outcome):
                                points_earned += confidence

                completed_standings[player_name] = points_earned

            # Show top performers so far
            sorted_standings = sorted(completed_standings.items(), key=lambda x: x[1], reverse=True)
            print(f"   Current leaders from completed games:")
            for i, (name, points) in enumerate(sorted_standings[:5], 1):
                print(f"     {i}. {name}: {points} points")

        # Player selection
        player_names = [p.name for p in simulator.players]
        print(f"\n👥 Select your player from {len(player_names)} total:")

        for i in range(0, min(15, len(player_names)), 3):
            row_players = player_names[i:i+3]
            for j, name in enumerate(row_players):
                current_points = completed_standings.get(name, 0) if completed_count > 0 else 0
                print(f"   {i+j+1:2d}. {name:<25} ({current_points} pts so far)", end="")
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

        # Calculate YOUR actual used confidence levels from completed games
        your_used_confidence = set()
        your_remaining_confidence = None
        if completed_count > 0:
            your_points = completed_standings.get(selected, 0)
            your_rank = sorted(completed_standings.values(), reverse=True).index(your_points) + 1
            print(f"📊 Your current position: #{your_rank} with {your_points} points from completed games")

            # Find your actual confidence levels used in completed games
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

                # Calculate remaining confidence levels
                all_confidence = set(range(1, 17))  # All possible confidence levels
                your_remaining_confidence = all_confidence - your_used_confidence
                print(f"\n📊 CONFIDENCE LEVELS:")
                print(f"   Used in completed games: {sorted(your_used_confidence)}")
                print(f"   Available for remaining games: {sorted(your_remaining_confidence)}")
            else:
                print(f"⚠️ Could not find your player data")
                your_remaining_confidence = set(range(1, 15))  # Fallback

        # Get available teams for remaining games
        available_teams = set()
        remaining_games = []
        for i, game_sim in enumerate(simulator.games):
            if game_sim.actual_outcome is None:  # Game not completed
                available_teams.add(game_sim.home_team)
                available_teams.add(game_sim.away_team)
                remaining_games.append({
                    'home': game_sim.home_team,
                    'away': game_sim.away_team
                })

        available_teams = sorted(list(available_teams))

        print(f"\n🏈 {len(remaining_games)} REMAINING GAMES TO OPTIMIZE:")
        for i, game in enumerate(remaining_games[:5], 1):
            print(f"   {i}. {game['away']} @ {game['home']}")
        if len(remaining_games) > 5:
            print(f"   ... and {len(remaining_games) - 5} more games")

        print(f"\n📋 Available teams for remaining games:")
        for i in range(0, len(available_teams), 8):
            row_teams = available_teams[i:i+8]
            print(f"   {' '.join(f'{team:<5}' for team in row_teams)}")

        # Fixed picks for remaining games only
        print(f"\n📌 Lock in any high-confidence picks for REMAINING games?")
        print(f"   Examples: 'Phi 16, KC 15' (only for games not yet played)")
        print(f"   Or press Enter to optimize all remaining games")
        fixed_input = input("Fixed picks: ").strip()

        fixed_picks = None
        if fixed_input:
            fixed_picks = {}
            for pick in fixed_input.split(','):
                try:
                    parts = pick.strip().split()
                    if len(parts) >= 2:
                        team_input, conf = parts[0].strip(), int(parts[1])

                        # Find matching team (case insensitive)
                        matched_team = None
                        for team in available_teams:
                            if team.lower() == team_input.lower():
                                matched_team = team
                                break

                        if matched_team:
                            fixed_picks[matched_team] = conf
                            print(f"✅ {team_input} -> {matched_team} ({conf} pts)")
                        else:
                            print(f"❌ Team '{team_input}' not available in remaining games")
                except Exception as e:
                    print(f"⚠️ Could not parse '{pick}': {e}")

            if fixed_picks:
                print(f"\n📌 Fixed picks for remaining games: {fixed_picks}")
            else:
                print(f"\n⚠️ No valid fixed picks found")
                fixed_picks = None

        # Calculate time estimate
        games_to_optimize = len(remaining_games)
        if fixed_picks:
            games_to_optimize -= len(fixed_picks)

        estimated_minutes = games_to_optimize * 0.75  # Slightly slower due to actual results

        # Run mid-week optimization with live odds
        print(f"\n🎯 MID-WEEK OPTIMIZATION WITH LIVE ODDS:")
        print(f"   ✅ Using actual results for completed games")
        print(f"   🔴 Enhanced with live Vegas odds ({live_updates} games updated)")
        print(f"   🎲 {args.num_sims:,} simulations for remaining {games_to_optimize} games")
        print(f"   ⏱️  Estimated time: {estimated_minutes:.0f}-{estimated_minutes*1.5:.0f} minutes")
        print(f"\n🚀 Starting optimization...")

        try:
            fixed_formatted = {selected: fixed_picks} if fixed_picks else None
            # Use your actual remaining confidence levels (accounting for completed games)
            remaining_confidence = your_remaining_confidence if completed_count > 0 else None
            optimal_picks = simulator.optimize_picks(
                player_name=selected,
                fixed_picks=fixed_formatted,
                confidence_range=4,
                available_points=remaining_confidence
            )

            if optimal_picks:
                print("\n🏆 MID-WEEK LIVE ODDS OPTIMIZATION RESULTS:")
                print("=" * 50)

                # Performance analysis accounting for actual results
                optimal_fixed = {selected: optimal_picks}
                optimal_stats = simulator.simulate_all(optimal_fixed)
                random_stats = simulator.simulate_all({})

                opt_win = optimal_stats['win_pct'][selected]
                rand_win = random_stats['win_pct'][selected]

                print(f"📈 Win Probability (enhanced with live odds + actual results):")
                print(f"   🎯 Optimized strategy: {opt_win:.1%}")
                print(f"   🎲 Random remaining picks: {rand_win:.1%}")
                print(f"   💪 Live odds + mid-week advantage: +{(opt_win - rand_win)*100:.1f} percentage points")

                if completed_count > 0:
                    print(f"\n📊 Your position advantage:")
                    print(f"   🏃 Current rank: #{your_rank} ({your_points} pts from completed)")
                    print(f"   🎯 Optimized picks maximize remaining point potential")

                print(f"\n🎯 OPTIMIZED REMAINING PICKS:")
                sorted_picks = sorted(optimal_picks.items(), key=lambda x: x[1], reverse=True)

                for team, conf in sorted_picks:
                    # Find opponent and check if it's a remaining game
                    opponent = "Unknown"
                    is_remaining = False
                    for game in remaining_games:
                        if team in [game['home'], game['away']]:
                            opponent = game['away'] if team == game['home'] else game['home']
                            is_remaining = True
                            break

                    status = "📅" if is_remaining else "✅"
                    print(f"   {conf:2d}. {team} vs {opponent} {status}")

                # Game importance analysis
                print(f"\n🎯 GAME IMPORTANCE ANALYSIS:")
                print("   (How much each game affects your win probability)")
                importance_df = None
                try:
                    optimal_fixed_for_analysis = {selected: optimal_picks}
                    importance_df = simulator.assess_game_importance(
                        player_name=selected,
                        fixed_picks=optimal_fixed_for_analysis
                    )

                    # Sort by importance and show top games
                    importance_sorted = importance_df.sort_values('total_impact', ascending=False)

                    for i, (_, row) in enumerate(importance_sorted.head(8).iterrows()):
                        game_desc = row['game']  # Already formatted as "Away@Home"
                        pick = row['pick']
                        conf = int(row['points_bid'])  # Convert to int for formatting
                        importance = row['total_impact']

                        # Check if this is a remaining game by parsing the game string
                        away_team, home_team = game_desc.split('@')
                        is_remaining = any(
                            set([home_team, away_team]) == set([g['home'], g['away']])
                            for g in remaining_games
                        )
                        status = "📅" if is_remaining else "✅"

                        print(f"   {i+1:2d}. {game_desc:<20} → {pick:3} ({conf:2d} pts) +{importance*100:4.1f}% {status}")

                except Exception as e:
                    print(f"   ⚠️ Could not calculate game importance: {e}")

                # Show copy-paste friendly format for test_picks.py
                print(f"\n📋 COPY-PASTE FORMAT FOR test_picks.py:")
                sorted_picks = sorted(optimal_picks.items(), key=lambda x: x[1], reverse=True)
                paste_format = ", ".join(f"{team} {conf}" for team, conf in sorted_picks)
                print(f"   {paste_format}")

                # Projected standings based on win probabilities
                print(f"\n📊 PROJECTED FINAL STANDINGS:")
                print("   (Based on live odds + current win probabilities)")
                try:
                    # Get win probabilities for all players
                    win_probs = optimal_stats['win_pct']

                    # Sort players by win probability
                    sorted_standings = sorted(win_probs.items(), key=lambda x: x[1], reverse=True)

                    print("   Rank  Player                    Win%")
                    print("   " + "="*42)

                    for rank, (name, win_pct) in enumerate(sorted_standings[:15], 1):
                        # Highlight your position
                        marker = " 👤" if name == selected else ""
                        print(f"   {rank:2d}.   {name:<20} {win_pct:6.1%}{marker}")

                    if len(sorted_standings) > 15:
                        print(f"        ... and {len(sorted_standings) - 15} more players")

                    # Show your rank
                    your_rank = next(i for i, (name, _) in enumerate(sorted_standings, 1) if name == selected)
                    print(f"\n   🎯 Your projected rank: #{your_rank} out of {len(sorted_standings)}")

                except Exception as e:
                    print(f"   ⚠️ Could not calculate projected standings: {e}")

                # Save results
                timestamp = datetime.now().strftime("%Y%m%d_%H%M")
                safe_name = selected.replace(' ', '_').replace('/', '_')
                filename = f"NFL_Week{args.week}_MidWeekLiveOdds_{safe_name}_{timestamp}.txt"

                with open(filename, 'w') as f:
                    f.write(f"NFL Week {args.week} Mid-Week Optimized Picks with Live Odds\n")
                    f.write(f"Player: {selected}\n")
                    f.write(f"Generated: {datetime.now()}\n")
                    f.write(f"Method: Mid-week with live odds ({args.num_sims:,} sims, accounts for actual results)\n")
                    f.write(f"Live odds updates: {live_updates}/{len(enhanced_games)} games\n")
                    f.write(f"Projected win probability: {opt_win:.1%}\n")
                    f.write(f"Completed games: {completed_count}/{len(yahoo.games)}\n")
                    f.write(f"Current points: {your_points if completed_count > 0 else 'N/A'}\n")
                    f.write(f"Current rank: #{your_rank if completed_count > 0 else 'N/A'}\n")

                    # Add projected final rank if available
                    try:
                        win_probs = optimal_stats['win_pct']
                        sorted_standings = sorted(win_probs.items(), key=lambda x: x[1], reverse=True)
                        your_projected_rank = next(i for i, (name, _) in enumerate(sorted_standings, 1) if name == selected)
                        f.write(f"Projected final rank: #{your_projected_rank}/{len(sorted_standings)}\n")
                    except:
                        pass

                    f.write("\n")

                    # Add game importance analysis to file
                    f.write("OPTIMIZED PICKS (by strategic importance):\n")
                    if importance_df is not None:
                        try:
                            importance_sorted = importance_df.sort_values('total_impact', ascending=False)

                            for i, (_, row) in enumerate(importance_sorted.iterrows()):
                                game_desc = row['game']
                                pick = row['pick']
                                conf = int(row['points_bid'])
                                importance = row['total_impact']

                                # Check if this is a remaining game
                                away_team, home_team = game_desc.split('@')
                                is_remaining = any(
                                    set([home_team, away_team]) == set([g['home'], g['away']])
                                    for g in remaining_games
                                )
                                status = "📅" if is_remaining else "✅"

                                f.write(f"{i+1:2d}. {game_desc:<20} → {pick:3} ({conf:2d} pts) +{importance*100:4.1f}% {status}\n")

                        except Exception as e:
                            # Fallback to basic picks list if sorting fails
                            f.write("OPTIMIZED PICKS:\n")
                            for team, conf in sorted_picks:
                                f.write(f"{conf:2d}. {team}\n")
                    else:
                        # Fallback to basic picks list if game importance was not computed
                        f.write("OPTIMIZED PICKS:\n")
                        for team, conf in sorted_picks:
                            f.write(f"{conf:2d}. {team}\n")

                    # Add copy-paste format for easy testing
                    f.write(f"\nCOPY-PASTE FORMAT FOR test_picks.py:\n")
                    sorted_picks = sorted(optimal_picks.items(), key=lambda x: x[1], reverse=True)
                    paste_format = ", ".join(f"{team} {conf}" for team, conf in sorted_picks)
                    f.write(f"{paste_format}\n")

                print(f"\n💾 Results saved: {filename}")
                print("\n🎯 Mid-week live odds optimization complete!")
                print(f"💡 Strategy combines actual Thu/Fri results + live Vegas odds")
                print(f"   for maximum accuracy!")
            else:
                print("❌ Mid-week live odds optimization failed")

        except Exception as e:
            print(f"❌ Optimization error: {e}")
            return 1

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())