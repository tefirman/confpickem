#!/usr/bin/env python
"""
Unified NFL Confidence Pick'Em Optimization CLI

Consolidates all optimization functionality into a single interface:
- Beginning-of-week vs mid-week optimization
- Live Vegas odds integration
- Fast mode for quick decisions
- Player skill integration

Usage:
  # Beginning of week (all games pending)
  optimize.py --week 10 --mode beginning

  # Mid-week (some games completed)
  optimize.py --week 10 --mode midweek

  # With live Vegas odds
  optimize.py --week 10 --mode midweek --live-odds --odds-api-key YOUR_KEY

  # Fast mode (~85% accuracy, 10x speed)
  optimize.py --week 10 --mode beginning --fast
"""

import sys
import argparse
from pathlib import Path
import pandas as pd
import json
from datetime import datetime
import shutil

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.confpickem.yahoo_pickem_scraper import YahooPickEm
from src.confpickem.live_odds_scraper import update_odds_with_live_data
from src.confpickem.confidence_pickem_sim import ConfidencePickEmSimulator, Player


def main():
    """Unified optimization CLI"""
    parser = argparse.ArgumentParser(
        description='NFL Confidence Pick\'Em Optimization',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Beginning of week optimization
  %(prog)s --week 10 --mode beginning

  # Mid-week with completed games
  %(prog)s --week 10 --mode midweek

  # With live Vegas odds
  %(prog)s --week 10 --mode midweek --live-odds

  # Fast mode (10x faster, ~85%% accuracy)
  %(prog)s --week 10 --mode beginning --fast
        """
    )

    # Required arguments
    parser.add_argument('--week', '-w', type=int, default=3,
                       help='NFL week number (default: 3)')
    parser.add_argument('--league-id', '-l', type=int, default=15435,
                       help='Yahoo league ID (default: 15435)')

    # Mode selection
    parser.add_argument('--mode', '-m', choices=['beginning', 'midweek'], required=True,
                       help='Optimization mode: "beginning" (all games pending) or "midweek" (some games completed)')

    # Optional features
    parser.add_argument('--live-odds', action='store_true',
                       help='Use live Vegas odds (requires --odds-api-key or ODDS_API_KEY env var)')
    parser.add_argument('--odds-api-key', '-k', type=str,
                       help='The Odds API key for live odds')
    parser.add_argument('--fast', action='store_true',
                       help='Fast mode: ~85%% accuracy but 10x faster (only for beginning mode)')
    parser.add_argument('--num-sims', '-n', type=int,
                       help='Number of simulations (overrides defaults)')
    parser.add_argument('--no-cache', action='store_true',
                       help='Clear cache before loading data (forces fresh fetch)')

    args = parser.parse_args()

    # Validate arguments
    if args.fast and args.mode == 'midweek':
        print("❌ Error: --fast mode is only available for beginning-of-week optimization")
        return 1

    # Determine simulation count
    if args.num_sims:
        num_sims = args.num_sims
    elif args.fast:
        num_sims = 2000  # Fast mode uses limited confidence testing
    else:
        num_sims = 2000  # Default

    # Determine confidence range (for optimize_picks)
    confidence_range = 4  # Original scripts all used 4

    # Print banner
    mode_str = "MID-WEEK" if args.mode == 'midweek' else "BEGINNING-OF-WEEK"
    odds_str = " + LIVE ODDS" if args.live_odds else ""
    fast_str = " (FAST MODE)" if args.fast else ""

    print(f"🎯 NFL PICK OPTIMIZATION - {mode_str}{odds_str}{fast_str}")
    print(f"📅 Week {args.week} | League {args.league_id}")
    print("=" * 60)

    if not Path("cookies.txt").exists():
        print("❌ Missing cookies.txt file")
        print("   Please create cookies.txt with your Yahoo session cookies")
        return 1

    try:
        # Clear cache if requested or if using live odds
        if args.no_cache or args.live_odds:
            cache_dir = Path(".cache")
            if cache_dir.exists():
                shutil.rmtree(cache_dir)
                print("🗑️  Cache cleared for fresh data")

        # Load Yahoo data
        print(f"\n📊 Loading Yahoo data for week {args.week}...")
        yahoo = YahooPickEm(week=args.week, league_id=args.league_id, cookies_file="cookies.txt")
        print(f"✅ Loaded {len(yahoo.players)} players, {len(yahoo.games)} games")

        # Check for valid data
        if len(yahoo.games) == 0:
            print("❌ No games found! This usually means:")
            print("   🍪 Your cookies.txt file may be expired")
            print("   🔗 League ID might be incorrect")
            print("   📅 Week number might be wrong")
            return 1

        # Enhanced games with live odds if requested
        enhanced_games = yahoo.games
        live_updates = 0

        if args.live_odds:
            print(f"\n🔄 Fetching live Vegas odds...")
            enhanced_games = update_odds_with_live_data(
                yahoo.games,
                week=args.week,
                odds_api_key=args.odds_api_key
            )

            # Count updates
            live_updates = sum(1 for _, game in enhanced_games.iterrows()
                              if game.get('live_odds_source', 'Yahoo_Fallback') != 'Yahoo_Fallback')
            print(f"💡 Updated {live_updates}/{len(enhanced_games)} games with live Vegas odds")

        # Show game status
        completed_games = [r for r in yahoo.results if r['winner']]
        remaining_games_count = len(yahoo.games) - len(completed_games)

        print(f"\n🎮 GAME STATUS:")
        print(f"   ✅ {len(completed_games)} completed")
        print(f"   ⏳ {remaining_games_count} remaining")

        if args.mode == 'midweek' and len(completed_games) > 0:
            print(f"\n🏆 COMPLETED GAMES:")
            for game in completed_games[:5]:  # Show first 5
                winner = game.get('winner', 'Unknown')
                print(f"     ✅ {winner} won")
            if len(completed_games) > 5:
                print(f"     ... and {len(completed_games) - 5} more")

        # Setup simulator
        print(f"\n🎲 Setting up optimizer...")
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

            # Determine actual outcome if mid-week mode
            actual_outcome = None
            if args.mode == 'midweek':
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
                'week': args.week,
                'kickoff_time': game_row['kickoff_time'],
                'actual_outcome': actual_outcome
            })

        simulator.add_games_from_dataframe(pd.DataFrame(games_data))

        # Load player skills
        try:
            with open('current_player_skills.json', 'r') as f:
                player_skills = json.load(f)
            print("✅ Using realistic player skills from current_player_skills.json")
        except FileNotFoundError:
            print("⚠️  current_player_skills.json not found, using default skills")
            player_skills = {}

        # Add players
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
        current_standings = {}
        your_used_confidence = set()
        your_remaining_confidence = None

        if args.mode == 'midweek' and len(completed_games) > 0:
            print(f"\n📊 CURRENT STANDINGS (from completed games):")

            for _, player in yahoo.players.iterrows():
                player_name = player['player_name']
                points_earned = 0

                for i, game in enumerate(simulator.games):
                    if game.actual_outcome is not None:
                        game_num = i + 1
                        pick = player.get(f'game_{game_num}_pick')
                        conf = player.get(f'game_{game_num}_confidence', 0)

                        if pick:
                            if (pick == game.home_team and game.actual_outcome) or \
                               (pick == game.away_team and not game.actual_outcome):
                                points_earned += conf

                current_standings[player_name] = points_earned

            # Show top 5
            sorted_standings = sorted(current_standings.items(), key=lambda x: x[1], reverse=True)
            for i, (name, points) in enumerate(sorted_standings[:5], 1):
                print(f"   {i}. {name}: {points} points")

            if len(sorted_standings) > 5:
                print(f"   ... and {len(sorted_standings) - 5} more")

        # Player selection
        player_names = [p.name for p in simulator.players]
        print(f"\n👥 Select your player from {len(player_names)} total:")

        for i in range(0, min(15, len(player_names)), 3):
            row_players = player_names[i:i+3]
            for j, name in enumerate(row_players):
                current_points = current_standings.get(name, 0) if current_standings else 0
                print(f"   {i+j+1:2d}. {name:<25} ({current_points} pts)", end="")
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

        # Calculate available confidence levels for mid-week
        if args.mode == 'midweek' and len(completed_games) > 0:
            your_player_data = None
            for _, player in yahoo.players.iterrows():
                if player['player_name'] == selected:
                    your_player_data = player
                    break

            if your_player_data is not None:
                for i, game in enumerate(simulator.games):
                    if game.actual_outcome is not None:
                        game_num = i + 1
                        conf = your_player_data.get(f'game_{game_num}_confidence')
                        if conf:
                            your_used_confidence.add(conf)

                all_confidence = set(range(1, len(yahoo.games) + 1))
                your_remaining_confidence = all_confidence - your_used_confidence

                print(f"\n📊 YOUR CONFIDENCE LEVELS:")
                print(f"   Used: {sorted(your_used_confidence)}")
                print(f"   Available: {sorted(your_remaining_confidence)}")

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

        print(f"\n🏈 {len(remaining_games)} GAMES TO OPTIMIZE:")
        for i, game in enumerate(remaining_games[:5], 1):
            print(f"   {i}. {game['away']} @ {game['home']}")
        if len(remaining_games) > 5:
            print(f"   ... and {len(remaining_games) - 5} more")

        # Fixed picks
        print(f"\n📌 Lock in any high-confidence picks?")
        print(f"   Examples: 'SF 16, KC 15' or just press Enter to optimize all")
        fixed_input = input("Fixed picks: ").strip()

        fixed_picks = None
        if fixed_input:
            fixed_picks = {}
            for pick in fixed_input.split(','):
                try:
                    parts = pick.strip().split()
                    if len(parts) >= 2:
                        team_input, conf = parts[0].strip(), int(parts[1])

                        matched_team = None
                        for team in available_teams:
                            if team.lower() == team_input.lower():
                                matched_team = team
                                break

                        if matched_team:
                            fixed_picks[matched_team] = conf
                            print(f"✅ {matched_team} ({conf} pts)")
                        else:
                            print(f"❌ Team '{team_input}' not available")
                except Exception as e:
                    print(f"⚠️  Could not parse '{pick}': {e}")

            if not fixed_picks:
                fixed_picks = None

        # Calculate time estimate
        games_to_optimize = len(remaining_games)
        if fixed_picks:
            games_to_optimize -= len(fixed_picks)

        time_per_game = 0.03 if args.fast else 0.75
        estimated_minutes = games_to_optimize * time_per_game

        # Run optimization
        print(f"\n🚀 STARTING OPTIMIZATION:")
        print(f"   Mode: {mode_str}")
        if args.live_odds:
            print(f"   Live Odds: {live_updates} games updated")
        print(f"   Simulations: {num_sims:,} per game")
        print(f"   Games to optimize: {games_to_optimize}")
        if args.fast:
            print(f"   ⚡ Fast mode: ~85% accuracy, 10x speed")
        print(f"   ⏱️  Estimated time: {estimated_minutes:.0f}-{estimated_minutes*1.5:.0f} minutes")
        print()

        try:
            fixed_formatted = {selected: fixed_picks} if fixed_picks else None
            optimal_picks = simulator.optimize_picks(
                player_name=selected,
                fixed_picks=fixed_formatted,
                confidence_range=confidence_range,
                available_points=your_remaining_confidence if args.mode == 'midweek' else None
            )

            if optimal_picks:
                print("\n🏆 OPTIMIZATION RESULTS:")
                print("=" * 60)

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

                # Initialize midweek tracking variables
                your_rank = None
                your_points = None

                if args.mode == 'midweek' and current_standings:
                    your_points = current_standings.get(selected, 0)
                    sorted_standings = sorted(current_standings.items(), key=lambda x: x[1], reverse=True)
                    your_rank = next(i for i, (n, _) in enumerate(sorted_standings, 1) if n == selected)
                    print(f"\n📊 Current Position:")
                    print(f"   Rank: #{your_rank}")
                    print(f"   Points from completed: {your_points}")

                print(f"\n🎯 OPTIMIZED PICKS:")
                sorted_picks = sorted(optimal_picks.items(), key=lambda x: x[1], reverse=True)

                for team, conf in sorted_picks:
                    # Find opponent
                    opponent = "Unknown"
                    is_remaining = False
                    for game in remaining_games:
                        if team in [game['home'], game['away']]:
                            opponent = game['away'] if team == game['home'] else game['home']
                            is_remaining = True
                            break

                    status = "📅" if is_remaining else "✅"
                    print(f"   {conf:2d}. {team} vs {opponent} {status}")

                # Game importance
                print(f"\n🔥 GAME IMPORTANCE ANALYSIS:")
                print("   (Impact on your win probability)")

                try:
                    importance_df = simulator.assess_game_importance(
                        player_name=selected,
                        fixed_picks=optimal_fixed
                    )

                    importance_sorted = importance_df.sort_values('total_impact', ascending=False)

                    for i, (_, row) in enumerate(importance_sorted.head(8).iterrows()):
                        game_desc = row['game']
                        pick = row['pick']
                        conf = int(row['points_bid'])
                        importance = row['total_impact']
                        correct_prob = row['win_probability']
                        incorrect_prob = row['loss_probability']

                        # Check if remaining
                        away_team, home_team = game_desc.split('@')
                        is_remaining = any(
                            set([home_team, away_team]) == set([g['home'], g['away']])
                            for g in remaining_games
                        )
                        status = "📅" if is_remaining else "✅"

                        print(f"   {i+1:2d}. {game_desc:<20} → {pick:3} ({conf:2d} pts) {importance:+5.1%} (Correct: {correct_prob:4.1%}, Wrong: {incorrect_prob:4.1%}) {status}")

                except Exception as e:
                    print(f"   ⚠️  Could not calculate: {e}")

                # Copy-paste format
                print(f"\n📋 COPY-PASTE FORMAT:")
                paste_format = ", ".join(f"{team} {conf}" for team, conf in sorted_picks)
                print(f"   {paste_format}")

                # Save results
                mode_suffix = "MidWeek" if args.mode == 'midweek' else "BeginningWeek"
                odds_suffix = "_LiveOdds" if args.live_odds else ""
                fast_suffix = "_Fast" if args.fast else ""
                timestamp = datetime.now().strftime("%Y%m%d_%H%M")
                safe_name = selected.replace(' ', '_').replace('/', '_')
                filename = f"NFL_Week{args.week}_{mode_suffix}{odds_suffix}{fast_suffix}_{safe_name}_{timestamp}.txt"

                with open(filename, 'w') as f:
                    f.write(f"NFL Week {args.week} Optimized Picks\n")
                    f.write(f"Player: {selected}\n")
                    f.write(f"Generated: {datetime.now()}\n")
                    f.write(f"Mode: {mode_str}{odds_suffix}{fast_suffix}\n")
                    f.write(f"Simulations: {num_sims:,}\n")
                    if args.live_odds:
                        f.write(f"Live odds updates: {live_updates}/{len(enhanced_games)} games\n")
                    f.write(f"Win probability: {opt_win:.1%}\n")
                    f.write(f"Advantage: +{(opt_win - rand_win)*100:.1f} pp\n")
                    if args.mode == 'midweek' and your_rank is not None:
                        f.write(f"Current rank: #{your_rank}\n")
                        f.write(f"Current points: {your_points}\n")
                    f.write("\n")

                    f.write("OPTIMIZED PICKS:\n")
                    for team, conf in sorted_picks:
                        f.write(f"{conf:2d}. {team}\n")

                    # Write game importance analysis if available
                    try:
                        if 'importance_sorted' in locals():
                            f.write(f"\nGAME IMPORTANCE ANALYSIS:\n")
                            f.write("(Impact on your win probability)\n\n")
                            for i, (_, row) in enumerate(importance_sorted.head(8).iterrows()):
                                game_desc = row['game']
                                pick = row['pick']
                                conf = int(row['points_bid'])
                                importance = row['total_impact']
                                correct_prob = row['win_probability']
                                incorrect_prob = row['loss_probability']

                                # Check if remaining
                                away_team, home_team = game_desc.split('@')
                                is_remaining = any(
                                    set([home_team, away_team]) == set([g['home'], g['away']])
                                    for g in remaining_games
                                )
                                status = "[REMAINING]" if is_remaining else "[COMPLETE]"

                                f.write(f"{i+1:2d}. {game_desc:<20} -> {pick:3} ({conf:2d} pts) {importance:+5.1%} "
                                       f"(Correct: {correct_prob:4.1%}, Wrong: {incorrect_prob:4.1%}) {status}\n")
                    except Exception as e:
                        f.write(f"\nGame importance analysis: Could not calculate ({e})\n")

                    f.write(f"\nCOPY-PASTE FORMAT:\n")
                    f.write(f"{paste_format}\n")

                print(f"\n💾 Results saved: {filename}")
                print(f"\n✅ Optimization complete!")

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


if __name__ == "__main__":
    sys.exit(main())
