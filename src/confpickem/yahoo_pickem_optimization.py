from confidence_pickem_sim import ConfidencePickEmSimulator, Player 
from yahoo_pickem_scraper import YahooPickEm
from yahoo_pickem_integration import convert_yahoo_to_simulator_format

def optimize_picks_for_week(week: int, league_id: int, cookies_file: str, 
                          fixed_picks: dict = None, num_sims: int = 1000,
                          ignore_results: bool = True):
    """
    Optimize picks for a given week, optionally ignoring known results
    
    Args:
        week: NFL week number
        league_id: Yahoo Pick'em league ID
        cookies_file: Path to cookies file with Yahoo authentication
        fixed_picks: Dictionary of team:points pairs to lock in
        num_sims: Number of simulations to run
        ignore_results: Whether to ignore known game results
    """
    # Initialize Yahoo scraper to get game data
    yahoo = YahooPickEm(week=week, league_id=league_id, cookies_file=cookies_file)

    # Create simulator instance
    simulator = ConfidencePickEmSimulator(num_sims=num_sims)

    # Add games from Yahoo data with ignore_results flag
    games_df = convert_yahoo_to_simulator_format(yahoo, ignore_results=ignore_results)
    simulator.add_games_from_dataframe(games_df)

    # Create players list with your player first
    players = [
        Player(
            name="Firman's Educated Guesses",
            skill_level=0.749,  # Can be updated with actual stats
            crowd_following=0.737,
            confidence_following=0.826
        )
    ] + [
        Player(
            name=f"Player {ind + 2}",
            skill_level=0.749,
            crowd_following=0.737,
            confidence_following=0.826
        ) for ind in range(7)
    ]

    simulator.players = players

    # If no fixed picks provided, initialize empty dict
    if fixed_picks is None:
        fixed_picks = {}

    # Optimize picks using comprehensive method
    optimal_picks = simulator.optimize_picks_comprehensive(fixed_picks=fixed_picks)
    
    # Print results in sorted order by confidence points
    print("\nOptimal Picks (sorted by confidence):")
    for team, points in sorted(optimal_picks.items(), key=lambda x: x[1], reverse=True):
        print(f"{team}: {points} points")
        
    return optimal_picks

# Example usage:
if __name__ == "__main__":
    # Example fixed picks - comment out to optimize all picks
    # fixed_picks = {
    #     'NE': 16, 
    #     'Buf': 15
    # }
    fixed_picks = {}
    # fixed_picks = {'Cin': 16, 'Buf': 15, 'Det': 11, 'NO': 8, 'Sea': 9, 'Mia': 7, 'NYJ': 3}
    
    optimal_picks = optimize_picks_for_week(
        week=1,
        league_id=6207,
        cookies_file='cookies.txt',
        fixed_picks=fixed_picks,
        num_sims=1000,
        ignore_results=True  # This ensures we don't use known outcomes
    )
