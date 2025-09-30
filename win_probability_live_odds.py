#!/usr/bin/env python
"""Win probability calculator using live Vegas odds for maximum accuracy"""

import sys
import argparse
from pathlib import Path
import pandas as pd
import numpy as np

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.confpickem.yahoo_pickem_scraper import YahooPickEm
from src.confpickem.live_odds_scraper import update_odds_with_live_data
from src.confpickem.confidence_pickem_sim import ConfidencePickEmSimulator, Player

def simulate_remaining_games_with_live_odds(results, all_picks, enhanced_games, num_sims=5000, simulate_from_game=None):
    """Simulate remaining games using live Vegas odds and calculate final standings

    Args:
        results: Yahoo results data
        all_picks: Player picks dictionary
        enhanced_games: Games data enhanced with live odds
        num_sims: Number of simulations to run
        simulate_from_game: If specified, treat all games from this index onward as pending
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
    print(f"📊 Using LIVE VEGAS ODDS for realistic win probabilities:")

    # Show Vegas probabilities for pending games
    for game_idx, result in pending_games:
        # Get enhanced game data with live odds
        enhanced_game = enhanced_games.iloc[game_idx]

        # Determine actual home/away teams
        fav = enhanced_game['favorite']  # betting favorite
        dog = enhanced_game['underdog']  # betting underdog
        home_favorite = enhanced_game.get('home_favorite', True)

        if home_favorite:
            home_team, away_team = fav, dog  # favorite is home
            favorite_win_prob = enhanced_game['win_prob']
        else:
            home_team, away_team = dog, fav  # underdog is home
            favorite_win_prob = enhanced_game['win_prob']

        spread = enhanced_game['spread']
        live_source = enhanced_game.get('live_odds_source', 'Yahoo_Fallback')
        source_indicator = "🔴" if live_source != 'Yahoo_Fallback' else "📊"

        print(f"   {source_indicator} {away_team} @ {home_team} (spread: {fav} -{spread}) → {fav} {favorite_win_prob:.1%}")

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

        # Simulate each pending game using live Vegas probabilities
        for game_idx, result in pending_games:
            # Get enhanced game data with live odds
            enhanced_game = enhanced_games.iloc[game_idx]

            # Determine actual home/away teams and probabilities
            fav = enhanced_game['favorite']  # betting favorite
            dog = enhanced_game['underdog']  # betting underdog
            home_favorite = enhanced_game.get('home_favorite', True)

            if home_favorite:
                home_team, away_team = fav, dog  # favorite is home
                home_win_prob = enhanced_game['win_prob']
            else:
                home_team, away_team = dog, fav  # underdog is home
                home_win_prob = 1.0 - enhanced_game['win_prob']  # flip probability for home team

            # Simulate game outcome based on live Vegas probability
            if np.random.random() < home_win_prob:
                winner = home_team
            else:
                winner = away_team

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
    """Calculate win probabilities using live Vegas odds"""
    parser = argparse.ArgumentParser(description='NFL win probabilities with live Vegas odds')
    parser.add_argument('--week', '-w', type=int, default=4,
                       help='NFL week number (default: 4)')
    parser.add_argument('--league-id', '-l', type=int, default=15435,
                       help='Yahoo league ID (default: 15435)')
    parser.add_argument('--odds-api-key', '-k', type=str,
                       help='The Odds API key for live odds (optional)')
    args = parser.parse_args()

    week = args.week
    league_id = args.league_id

    print(f"🎯 LIVE VEGAS ODDS WIN PROBABILITY CALCULATOR - WEEK {week}")
    print("🔴 Enhanced with real-time betting lines!")
    print("=" * 50)

    if not Path("cookies.txt").exists():
        print("❌ Missing cookies.txt file")
        return 1

    try:
        # Clear cache
        cache_dir = Path(".cache")
        if cache_dir.exists():
            import shutil
            shutil.rmtree(cache_dir)

        # Load Yahoo data
        print("📊 Loading Yahoo pick distribution and player data...")
        yahoo = YahooPickEm(week=week, league_id=league_id, cookies_file="cookies.txt")
        print(f"✅ Loaded {len(yahoo.players)} players, {len(yahoo.results)} results")

        # Update with live odds
        print(f"\n🔄 Enhancing with live Vegas odds...")
        enhanced_games = update_odds_with_live_data(
            yahoo.games,
            week=week,
            odds_api_key=args.odds_api_key
        )

        # Show enhancement results
        live_updates = sum(1 for _, game in enhanced_games.iterrows()
                          if game.get('live_odds_source', 'Yahoo_Fallback') != 'Yahoo_Fallback')
        print(f"💡 Enhanced {live_updates}/{len(enhanced_games)} games with live Vegas odds")

        # Extract all player picks
        print("\n🔍 Extracting everyone's picks...")
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

        # Run simulation with live odds
        win_probs, current_standings = simulate_remaining_games_with_live_odds(
            yahoo.results, all_picks, enhanced_games, num_sims=5000, simulate_from_game=simulate_from_game
        )

        # Show results
        print(f"\n🏆 LIVE VEGAS ODDS WIN PROBABILITY RANKINGS:")
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

                            # Setup simulator for game importance analysis
                            importance_simulator = ConfidencePickEmSimulator(num_sims=100)

                            # Convert enhanced games to simulator format
                            games_data = []
                            for _, game_row in enhanced_games.iterrows():
                                fav = game_row['favorite']  # betting favorite
                                dog = game_row['underdog']  # betting underdog
                                home_favorite = game_row.get('home_favorite', True)

                                if home_favorite:
                                    home_team, away_team = fav, dog  # favorite is home
                                    home_prob = game_row['win_prob']
                                    crowd_home_pct = game_row['favorite_pick_pct'] / 100.0
                                    home_conf, away_conf = game_row['favorite_confidence'], game_row['underdog_confidence']
                                else:
                                    home_team, away_team = dog, fav  # underdog is home
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

                                for i in range(1, len(enhanced_games) + 1):
                                    pick = player_row.get(f'game_{i}_pick')
                                    conf = player_row.get(f'game_{i}_confidence')
                                    if pick and conf:
                                        picks_dict[pick] = int(conf)

                                player_picks[player_name] = picks_dict

                            # Run remaining game importance analysis
                            test_picks_for_importance = {name: player_picks[name]}
                            picks_df = importance_simulator.simulate_picks(test_picks_for_importance)

                            # Create fixed_picks dict with completed games
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

                                        # Calculate absolute win probabilities for correct/incorrect scenarios
                                        baseline_prob = win_probs[name]
                                        correct_prob = baseline_prob + (impact / 2)
                                        incorrect_prob = baseline_prob - (impact / 2)

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

        print(f"\n🔴 Simulation uses LIVE VEGAS ODDS for maximum accuracy")
        print(f"💡 {live_updates}/{len(enhanced_games)} games enhanced with real betting lines")
        print(f"📊 Based on 5000 Monte Carlo simulations with actual Vegas probabilities")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0

if __name__ == "__main__":
    main()