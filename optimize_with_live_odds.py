#!/usr/bin/env python
"""
Enhanced optimization script with live odds integration
Uses live Vegas odds when available, falls back to Yahoo odds
"""

import sys
import argparse
from pathlib import Path
import pandas as pd

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.confpickem.yahoo_pickem_scraper import YahooPickEm
from src.confpickem.live_odds_scraper import update_odds_with_live_data
from src.confpickem.confidence_pickem_sim import ConfidencePickEmSimulator, Player


def main():
    """Enhanced optimization with live odds integration"""
    parser = argparse.ArgumentParser(description='Optimize picks with live Vegas odds')
    parser.add_argument('--week', '-w', type=int, default=3,
                       help='NFL week number (default: 3)')
    parser.add_argument('--league-id', '-l', type=int, default=15435,
                       help='Yahoo league ID (default: 15435)')
    parser.add_argument('--odds-api-key', '-k', type=str,
                       help='The Odds API key for live odds (optional)')
    parser.add_argument('--num-sims', '-n', type=int, default=1000,
                       help='Number of simulations (default: 1000)')
    args = parser.parse_args()

    print(f"🎯 ENHANCED OPTIMIZATION WITH LIVE ODDS - WEEK {args.week}")
    print("=" * 55)

    if not Path("cookies.txt").exists():
        print("❌ Missing cookies.txt file")
        return 1

    try:
        # Load Yahoo data
        print("📊 Loading Yahoo pick distribution and player data...")
        yahoo = YahooPickEm(week=args.week, league_id=args.league_id, cookies_file="cookies.txt")
        print(f"✅ Loaded {len(yahoo.players)} players, {len(yahoo.games)} games")

        # Show original Yahoo odds
        print(f"\n📋 Original Yahoo odds:")
        for i, (_, game) in enumerate(yahoo.games.iterrows()):
            fav = game['favorite']
            dog = game['underdog']
            spread = game['spread']
            prob = game['win_prob']
            print(f"   {i+1:2d}. {dog:3s} @ {fav:3s} | {fav} -{spread:4.1f} | Win prob: {prob:.1%}")

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
            fav = game['favorite']
            dog = game['underdog']
            spread = game['spread']
            prob = game['win_prob']
            source = game.get('live_odds_source', 'Yahoo')

            indicator = "🔴" if source != 'Yahoo_Fallback' else "  "
            print(f"   {i+1:2d}. {dog:3s} @ {fav:3s} | {fav} -{spread:4.1f} | Win prob: {prob:.1%} {indicator}")

            if source != 'Yahoo_Fallback':
                live_updates += 1

        print(f"\n💡 Updated {live_updates}/{len(enhanced_games)} games with live odds")

        # Setup simulator with enhanced odds
        print(f"\n🎲 Setting up optimization with enhanced odds...")
        simulator = ConfidencePickEmSimulator(num_sims=args.num_sims)

        # Convert enhanced games to simulator format
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

            games_data.append({
                'home_team': home_team,
                'away_team': away_team,
                'vegas_win_prob': home_prob,
                'crowd_home_pick_pct': crowd_home_pct,
                'crowd_home_confidence': home_conf,
                'crowd_away_confidence': away_conf,
                'week': args.week,
                'kickoff_time': game_row['kickoff_time']
            })

        simulator.add_games_from_dataframe(pd.DataFrame(games_data))

        # Add all league players as opponents + the hypothetical player we're optimizing for
        players = []
        for _, player in yahoo.players.iterrows():
            players.append(Player(
                name=player['player_name'],
                skill_level=0.6,  # Default skill level
                crowd_following=0.5,
                confidence_following=0.5
            ))

        # Add hypothetical player for optimization
        players.append(Player(
            name="LiveOdds_User",
            skill_level=0.65,  # Slightly above average
            crowd_following=0.3,  # Independent thinker
            confidence_following=0.4  # Moderate confidence following
        ))

        simulator.players = players
        print(f"✅ Added {len(players)-1} league players + 1 hypothetical player")

        # Run optimization
        print(f"\n🧠 Running pick optimization with {args.num_sims:,} simulations...")
        print(f"💡 Optimizing for a hypothetical player 'LiveOdds_User'")
        optimal_picks = simulator.optimize_picks(
            player_name="LiveOdds_User",
            fixed_picks=None,  # No fixed picks
            confidence_range=4,  # Default confidence range
            available_points=None  # Use all confidence levels 1-16
        )

        # Display results
        print(f"\n🏆 OPTIMIZED PICKS (Enhanced with Live Odds):")
        print(f"{'Game':<15} {'Pick':<4} {'Conf':<4} {'Reason'}")
        print("-" * 50)

        total_confidence = 0
        for pick, confidence in optimal_picks.items():
            total_confidence += confidence

            # Find the game for context
            game_info = "Unknown"
            for _, game in enhanced_games.iterrows():
                if pick in [game['favorite'], game['underdog']]:
                    opp = game['underdog'] if pick == game['favorite'] else game['favorite']
                    game_info = f"{opp} @ {pick}" if game['home_favorite'] else f"{pick} @ {opp}"
                    break

            # Simple reasoning
            reason = "Optimal EV"

            print(f"{game_info:<15} {pick:<4} {confidence:<4} {reason}")

        print(f"\nTotal confidence used: {total_confidence}/136")

        # Show comparison with crowd picks
        print(f"\n📊 CROWD vs OPTIMIZED COMPARISON:")
        print(f"{'Game':<15} {'Crowd Pick':<10} {'Your Pick':<10} {'Diff'}")
        print("-" * 55)

        for _, game in enhanced_games.iterrows():
            fav = game['favorite']
            dog = game['underdog']
            fav_pct = game['favorite_pick_pct']

            crowd_pick = fav if fav_pct > 50 else dog
            your_pick = None

            for pick in optimal_picks:
                if pick in [fav, dog]:
                    your_pick = pick
                    break

            if your_pick:
                game_name = f"{dog}@{fav}"
                agree = "✅" if crowd_pick == your_pick else "❌"
                print(f"{game_name:<15} {crowd_pick:<10} {your_pick:<10} {agree}")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    main()