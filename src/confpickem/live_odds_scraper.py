#!/usr/bin/env python
"""
Live odds scraper for NFL games using multiple sources
Provides real-time Vegas odds and spreads
"""

import logging
import requests
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union
import json
import os

logger = logging.getLogger(__name__)


def moneyline_to_implied_prob(american_odds: float) -> float:
    """Convert American moneyline odds to (vig-included) implied win probability"""
    if american_odds > 0:
        return 100.0 / (american_odds + 100.0)
    return -american_odds / (-american_odds + 100.0)


def devig_moneyline_pair(home_odds: float, away_odds: float) -> float:
    """
    Convert a pair of American moneyline odds into a fair (no-vig) home win probability.

    Each side's raw implied probability includes the bookmaker's margin (the "vig"),
    so the two raw probabilities sum to slightly more than 1.0. Normalizing them to
    sum to 1.0 removes that margin and yields the market's true probability estimate.
    """
    home_implied = moneyline_to_implied_prob(home_odds)
    away_implied = moneyline_to_implied_prob(away_odds)
    return home_implied / (home_implied + away_implied)


class LiveOddsScraper:
    """Scrapes live NFL odds from multiple sources"""

    def __init__(self, odds_api_key: Optional[str] = None):
        # ESPN API (for game schedule/scores)
        self.espn_base_url = "https://site.api.espn.com/apis/site/v2/sports/football/nfl"

        # The Odds API (for betting odds)
        self.odds_api_key = odds_api_key or os.getenv('ODDS_API_KEY')
        self.odds_base_url = "https://api.the-odds-api.com/v4/sports/americanfootball_nfl/odds"

        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })

    def get_current_week(self) -> int:
        """Get current NFL week number"""
        try:
            url = f"{self.espn_base_url}/scoreboard"
            response = self.session.get(url)
            response.raise_for_status()
            data = response.json()

            # ESPN returns week info in the scoreboard
            week = data.get('week', {}).get('number', 1)
            return week
        except Exception as e:
            logger.warning("Could not get current week from ESPN: %s", e)
            # Fallback to date-based estimation
            now = datetime.now()
            if now.month < 3:  # Jan-Feb, probably still current season
                season_start = datetime(now.year - 1, 9, 1)
            else:  # March-Dec
                season_start = datetime(now.year, 9, 1) if now.month >= 9 else datetime(now.year - 1, 9, 1)

            weeks_since_start = (now - season_start).days // 7
            return min(max(1, weeks_since_start), 18)

    def _get_week_date_range(
        self, week: int, yahoo_games: Optional[pd.DataFrame] = None
    ) -> Tuple[pd.Timestamp, pd.Timestamp]:
        """
        Get date range for a given NFL week.

        Prefers deriving the range from yahoo_games['kickoff_time'] (the real
        schedule for this week) when available, padded by a day on each side to
        tolerate timezone edges and odds being posted before Yahoo's kickoff
        times settle. Falls back to a hardcoded per-season calendar only when
        Yahoo's kickoff times aren't available -- that calendar assumes a fixed
        Thursday-Sep-5 season start and drifts in years where the season starts
        on a different date.
        """
        if yahoo_games is not None and "kickoff_time" in yahoo_games.columns and len(yahoo_games):
            kickoff_times = pd.to_datetime(yahoo_games["kickoff_time"], utc=True)
            return (
                kickoff_times.min() - pd.Timedelta(days=1),
                kickoff_times.max() + pd.Timedelta(days=1),
            )

        # Determine current NFL season year based on current date
        now = datetime.now()
        if now.month >= 9:  # September or later = current year season
            season_year = now.year
        else:  # Before September = previous year season
            season_year = now.year - 1

        # NFL season week dates (Thursday to Wednesday pattern)
        week_dates = {
            1: (f'{season_year}-09-05', f'{season_year}-09-11'),   # Week 1: Thu Sep 5 - Wed Sep 11
            2: (f'{season_year}-09-12', f'{season_year}-09-18'),   # Week 2: Thu Sep 12 - Wed Sep 18
            3: (f'{season_year}-09-19', f'{season_year}-09-25'),   # Week 3: Thu Sep 19 - Wed Sep 25
            4: (f'{season_year}-09-26', f'{season_year}-10-02'),   # Week 4: Thu Sep 26 - Wed Oct 2
            5: (f'{season_year}-10-03', f'{season_year}-10-09'),   # Week 5: Thu Oct 3 - Wed Oct 9
            6: (f'{season_year}-10-10', f'{season_year}-10-16'),   # Week 6: Thu Oct 10 - Wed Oct 16
            7: (f'{season_year}-10-17', f'{season_year}-10-23'),   # Week 7: Thu Oct 17 - Wed Oct 23
            8: (f'{season_year}-10-24', f'{season_year}-10-30'),   # Week 8: Thu Oct 24 - Wed Oct 30
            9: (f'{season_year}-10-31', f'{season_year}-11-06'),   # Week 9: Thu Oct 31 - Wed Nov 6
            10: (f'{season_year}-11-07', f'{season_year}-11-13'),  # Week 10: Thu Nov 7 - Wed Nov 13
            11: (f'{season_year}-11-14', f'{season_year}-11-20'),  # Week 11: Thu Nov 14 - Wed Nov 20
            12: (f'{season_year}-11-21', f'{season_year}-11-27'),  # Week 12: Thu Nov 21 - Wed Nov 27 (Thanksgiving week)
            13: (f'{season_year}-11-28', f'{season_year}-12-04'),  # Week 13: Thu Nov 28 - Wed Dec 4
            14: (f'{season_year}-12-05', f'{season_year}-12-11'),  # Week 14: Thu Dec 5 - Wed Dec 11
            15: (f'{season_year}-12-12', f'{season_year}-12-18'),  # Week 15: Thu Dec 12 - Wed Dec 18
            16: (f'{season_year}-12-19', f'{season_year}-12-25'),  # Week 16: Thu Dec 19 - Wed Dec 25 (Christmas week)
            17: (f'{season_year}-12-26', f'{season_year + 1}-01-01'),  # Week 17: Thu Dec 26 - Wed Jan 1 (New Year week)
            18: (f'{season_year + 1}-01-02', f'{season_year + 1}-01-08'),  # Week 18: Thu Jan 2 - Wed Jan 8 (next year)
        }

        if week in week_dates:
            start_str, end_str = week_dates[week]
            week_start = pd.Timestamp(start_str, tz='UTC')
            week_end = pd.Timestamp(end_str + ' 23:59:59', tz='UTC')
        else:
            # Fallback calculation for weeks not explicitly defined
            week1_start = pd.Timestamp(f'{season_year}-09-05', tz='UTC')
            week_start = week1_start + pd.Timedelta(weeks=week-1)
            week_end = week_start + pd.Timedelta(days=6, hours=23, minutes=59)

        return week_start, week_end

    def get_live_odds(
        self, week: Optional[int] = None, yahoo_games: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Get live NFL odds for specified week from The Odds API.

        Args:
            week: NFL week number (if None, uses current week)
            yahoo_games: Yahoo's games DataFrame for this week, if available. Its
                kickoff_time column is used to pin down the real date range for the
                week instead of a hardcoded per-season calendar.

        Returns:
            DataFrame with columns: home_team, away_team, home_spread, total_points,
            home_win_prob. Empty if no API key is configured or no odds are available --
            callers should fall back to Yahoo's own odds in that case rather than
            treating an empty result as 50/50 games.
        """
        if week is None:
            week = self.get_current_week()

        if not self.odds_api_key:
            logger.warning(
                "No Odds API key provided - set ODDS_API_KEY environment variable "
                "or pass --odds-api-key. Falling back to Yahoo odds."
            )
            return pd.DataFrame()

        logger.debug("Using Odds API with key: %s...", self.odds_api_key[:8])
        odds_data = self._get_odds_from_api(week=week, yahoo_games=yahoo_games)
        if odds_data.empty:
            logger.warning("Odds API returned no games for week %s; falling back to Yahoo odds", week)
        else:
            logger.info("Retrieved live odds for %d games from Odds API", len(odds_data))
        return odds_data

    def _get_odds_from_api(
        self, week: int = 4, yahoo_games: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """Get odds from The Odds API and filter for specified NFL week"""
        try:
            params = {
                'apiKey': self.odds_api_key,
                'regions': 'us',
                'markets': 'h2h,spreads,totals',
                'bookmakers': 'draftkings,fanduel',
                'oddsFormat': 'american'
            }

            response = self.session.get(self.odds_base_url, params=params)
            response.raise_for_status()
            data = response.json()

            # Calculate date range for the specified NFL week
            week_start, week_end = self._get_week_date_range(week, yahoo_games=yahoo_games)
            logger.debug(
                "Odds API returned %d games; filtering for week %s (%s - %s)",
                len(data), week, week_start.strftime('%m/%d'), week_end.strftime('%m/%d'),
            )

            games_data = []
            for game in data:
                try:
                    # Parse game time and ensure it's timezone-aware
                    game_time = pd.to_datetime(game['commence_time'])
                    if game_time.tz is None:
                        game_time = game_time.tz_localize('UTC')
                    elif game_time.tz != week_start.tz:
                        game_time = game_time.tz_convert('UTC')

                    if not (week_start <= game_time <= week_end):
                        continue

                    game_data = self._parse_odds_api_game(game)
                    if game_data:
                        games_data.append(game_data)
                except Exception as e:
                    logger.warning("Failed to parse Odds API game: %s", e)
                    continue

            logger.debug("Filtered to %d games for week %s", len(games_data), week)
            return pd.DataFrame(games_data)

        except Exception as e:
            logger.warning("Odds API request failed: %s", e)
            return pd.DataFrame()

    def _parse_odds_api_game(self, game: Dict) -> Optional[Dict]:
        """Parse game from The Odds API response"""
        try:
            home_team = game['home_team']
            away_team = game['away_team']

            # Get the best available odds (prefer DraftKings, fallback to FanDuel)
            spread = 0.0
            total_points = 0.0
            home_moneyline: Optional[float] = None
            away_moneyline: Optional[float] = None

            bookmakers = game.get('bookmakers', [])
            for bookmaker in bookmakers:
                if bookmaker['key'] not in ('draftkings', 'fanduel'):
                    continue
                markets = bookmaker.get('markets', [])

                for market in markets:
                    if market['key'] == 'h2h':
                        for outcome in market['outcomes']:
                            if outcome['name'] == home_team:
                                home_moneyline = float(outcome['price'])
                            elif outcome['name'] == away_team:
                                away_moneyline = float(outcome['price'])

                    elif market['key'] == 'spreads':
                        for outcome in market['outcomes']:
                            if outcome['name'] == home_team:
                                spread = float(outcome['point'])
                                break

                    elif market['key'] == 'totals':
                        total_points = float(market['outcomes'][0]['point'])

                # Found a bookmaker with usable data; stop looking
                if home_moneyline is not None and away_moneyline is not None:
                    break

            # Prefer moneyline-derived probability: moneylines are the market's
            # direct probability quote, whereas spread-to-probability requires an
            # approximation curve that breaks down for large spreads (it either
            # saturates at 0/1 for blowout lines or is inaccurate near a pick'em).
            if home_moneyline is not None and away_moneyline is not None:
                home_win_prob = devig_moneyline_pair(home_moneyline, away_moneyline)
            else:
                # No moneyline available from either bookmaker; fall back to the
                # spread-based approximation so a game isn't dropped entirely.
                home_win_prob = min(max(-spread * 0.031 + 0.5, 0.01), 0.99)

            return {
                'home_team': home_team,
                'away_team': away_team,
                'home_spread': spread,
                'home_moneyline': home_moneyline,
                'away_moneyline': away_moneyline,
                'total_points': total_points,
                'home_win_prob': home_win_prob,
                'kickoff_time': pd.to_datetime(game['commence_time']),
                'game_completed': False,
                'winner': None,
                'source': 'OddsAPI'
            }

        except Exception as e:
            logger.warning("Error parsing Odds API game: %s", e)
            return None

    def update_yahoo_odds(self, yahoo_games: pd.DataFrame, live_odds: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Update Yahoo games DataFrame with live odds data

        Args:
            yahoo_games: DataFrame from YahooPickEm.games
            live_odds: DataFrame from get_live_odds() (if None, fetches fresh data)

        Returns:
            Updated DataFrame with live odds where available, Yahoo odds as fallback
        """
        if live_odds is None:
            live_odds = self.get_live_odds(yahoo_games=yahoo_games)

        if live_odds.empty:
            logger.info("No live odds available, keeping Yahoo odds")
            fallback = yahoo_games.copy()
            fallback['live_odds_source'] = 'Yahoo_Fallback'
            return fallback

        # Create team name mapping from Odds API (full names or abbreviations) to Yahoo (Title Case)
        team_name_mapping = {
            # Full team names from Odds API -> Yahoo Title Case
            'Arizona Cardinals': 'Ari', 'Atlanta Falcons': 'Atl', 'Baltimore Ravens': 'Bal', 'Buffalo Bills': 'Buf',
            'Carolina Panthers': 'Car', 'Chicago Bears': 'Chi', 'Cincinnati Bengals': 'Cin', 'Cleveland Browns': 'Cle',
            'Dallas Cowboys': 'Dal', 'Denver Broncos': 'Den', 'Detroit Lions': 'Det', 'Green Bay Packers': 'GB',
            'Houston Texans': 'Hou', 'Indianapolis Colts': 'Ind', 'Jacksonville Jaguars': 'Jax', 'Kansas City Chiefs': 'KC',
            'Los Angeles Chargers': 'LAC', 'Los Angeles Rams': 'LAR', 'Las Vegas Raiders': 'LV', 'Miami Dolphins': 'Mia',
            'Minnesota Vikings': 'Min', 'New England Patriots': 'NE', 'New Orleans Saints': 'NO', 'New York Giants': 'NYG',
            'New York Jets': 'NYJ', 'Philadelphia Eagles': 'Phi', 'Pittsburgh Steelers': 'Pit', 'Seattle Seahawks': 'Sea',
            'San Francisco 49ers': 'SF', 'Tampa Bay Buccaneers': 'TB', 'Tennessee Titans': 'Ten', 'Washington Commanders': 'Was',
            # Abbreviations from ESPN/other sources -> Yahoo Title Case
            'ARI': 'Ari', 'ATL': 'Atl', 'BAL': 'Bal', 'BUF': 'Buf',
            'CAR': 'Car', 'CHI': 'Chi', 'CIN': 'Cin', 'CLE': 'Cle',
            'DAL': 'Dal', 'DEN': 'Den', 'DET': 'Det', 'GB': 'GB',
            'HOU': 'Hou', 'IND': 'Ind', 'JAX': 'Jax', 'KC': 'KC',
            'LAC': 'LAC', 'LAR': 'LAR', 'LV': 'LV', 'MIA': 'Mia',
            'MIN': 'Min', 'NE': 'NE', 'NO': 'NO', 'NYG': 'NYG',
            'NYJ': 'NYJ', 'PHI': 'Phi', 'PIT': 'Pit', 'SEA': 'Sea',
            'SF': 'SF', 'TB': 'TB', 'TEN': 'Ten', 'WSH': 'Was',
            # Handle potential variations
            'WAS': 'Was', 'ARZ': 'Ari'
        }

        def normalize_team_name(team_name):
            """Normalize team name to Yahoo format"""
            return team_name_mapping.get(team_name, team_name)

        updated_games = yahoo_games.copy()
        matches_found = 0

        for idx, yahoo_game in updated_games.iterrows():
            # Get Yahoo teams (keep original format)
            yahoo_favorite = yahoo_game['favorite']
            yahoo_underdog = yahoo_game['underdog']
            yahoo_teams = {yahoo_favorite, yahoo_underdog}

            for _, live_game in live_odds.iterrows():
                # Normalize live odds team names to match Yahoo format
                live_home = normalize_team_name(live_game['home_team'])
                live_away = normalize_team_name(live_game['away_team'])
                live_teams = {live_home, live_away}

                if yahoo_teams == live_teams:
                    # Found matching game - preserve Yahoo's home/away structure but update odds

                    # Store original Yahoo data for comparison
                    original_spread = yahoo_game['spread']

                    # Yahoo structure: 'favorite' = betting favorite, 'underdog' = betting underdog
                    # 'home_favorite' = True if favorite is home, False if favorite is away
                    if yahoo_game.get('home_favorite', True):
                        # Favorite is home team
                        yahoo_home_team = yahoo_game['favorite']
                        yahoo_away_team = yahoo_game['underdog']
                    else:
                        # Favorite is away team (underdog is home)
                        yahoo_home_team = yahoo_game['underdog']
                        yahoo_away_team = yahoo_game['favorite']
                    yahoo_home_is_betting_favorite = yahoo_game.get('home_favorite', True)

                    # Determine which team is actually the home team in live data
                    if live_home == yahoo_home_team:
                        # Live data matches Yahoo structure
                        home_spread = live_game['home_spread']
                        home_win_prob = live_game['home_win_prob']
                    else:
                        # Live data has teams flipped - need to adjust
                        home_spread = -live_game['home_spread']  # Flip the spread
                        home_win_prob = 1.0 - live_game['home_win_prob']  # Flip the probability

                    # Determine who is the betting favorite and update accordingly
                    if home_spread < 0:
                        # Home team is betting favorite (negative spread means favored)
                        betting_favorite = yahoo_home_team
                        betting_underdog = yahoo_away_team
                        spread_magnitude = abs(home_spread)
                        favorite_win_prob = home_win_prob
                        home_is_betting_favorite = True
                    else:
                        # Away team is betting favorite (positive home spread means away is favored)
                        betting_favorite = yahoo_away_team
                        betting_underdog = yahoo_home_team
                        spread_magnitude = abs(home_spread)
                        favorite_win_prob = 1.0 - home_win_prob  # Away team's win prob
                        home_is_betting_favorite = False

                    # Update with live odds while preserving Yahoo's home/away structure
                    # Note: Yahoo's structure is confusing - it appears to use 'favorite'/'underdog'
                    # in the betting sense, not the home/away sense. But we're preserving home/away here.
                    updated_games.at[idx, 'favorite'] = betting_favorite  # Betting favorite
                    updated_games.at[idx, 'underdog'] = betting_underdog  # Betting underdog
                    updated_games.at[idx, 'spread'] = spread_magnitude
                    updated_games.at[idx, 'win_prob'] = favorite_win_prob  # Betting favorite's win probability
                    updated_games.at[idx, 'home_favorite'] = home_is_betting_favorite
                    updated_games.at[idx, 'kickoff_time'] = live_game['kickoff_time']

                    # Add live odds metadata
                    updated_games.at[idx, 'live_odds_source'] = live_game.get('source', 'LiveOdds')
                    updated_games.at[idx, 'live_spread'] = spread_magnitude
                    updated_games.at[idx, 'original_spread'] = original_spread
                    updated_games.at[idx, 'total_points'] = live_game.get('total_points', 0.0)
                    updated_games.at[idx, 'last_updated'] = datetime.now()

                    # live_game's home/away moneylines are in the *live data's* home-team
                    # frame. live_home == yahoo_home_team tells us that frame lines up with
                    # Yahoo's (rather than being flipped); home_is_betting_favorite tells us
                    # whether Yahoo's home team is the betting favorite. Together they say
                    # whether live_game's home_moneyline belongs to the favorite or underdog.
                    home_moneyline = live_game.get('home_moneyline')
                    away_moneyline = live_game.get('away_moneyline')
                    if pd.notna(home_moneyline) and pd.notna(away_moneyline):
                        live_home_is_favorite = (
                            home_is_betting_favorite if live_home == yahoo_home_team
                            else not home_is_betting_favorite
                        )
                        if live_home_is_favorite:
                            updated_games.at[idx, 'favorite_moneyline'] = home_moneyline
                            updated_games.at[idx, 'underdog_moneyline'] = away_moneyline
                        else:
                            updated_games.at[idx, 'favorite_moneyline'] = away_moneyline
                            updated_games.at[idx, 'underdog_moneyline'] = home_moneyline

                    matches_found += 1
                    logger.debug(
                        "Matched %s @ %s -> live spread %s", yahoo_away_team, yahoo_home_team, spread_magnitude
                    )
                    break

        logger.info("Updated %d/%d games with live odds", matches_found, len(yahoo_games))

        # Add metadata for games without live odds
        for idx, game in updated_games.iterrows():
            if 'live_odds_source' not in game or pd.isna(game['live_odds_source']):
                updated_games.at[idx, 'live_odds_source'] = 'Yahoo_Fallback'
                updated_games.at[idx, 'last_updated'] = datetime.now()

        return updated_games


def get_live_nfl_odds(week: Optional[int] = None, odds_api_key: Optional[str] = None) -> pd.DataFrame:
    """
    Convenience function to get live NFL odds

    Args:
        week: NFL week number (if None, uses current week)
        odds_api_key: The Odds API key (optional, can use environment variable)

    Returns:
        DataFrame with live odds data
    """
    scraper = LiveOddsScraper(odds_api_key=odds_api_key)
    return scraper.get_live_odds(week)


def update_odds_with_live_data(yahoo_games: pd.DataFrame, week: Optional[int] = None, odds_api_key: Optional[str] = None) -> pd.DataFrame:
    """
    Convenience function to update Yahoo odds with live data

    Args:
        yahoo_games: DataFrame from YahooPickEm.games
        week: NFL week number (if None, uses current week)
        odds_api_key: The Odds API key (optional, can use environment variable)

    Returns:
        Updated DataFrame with live odds
    """
    scraper = LiveOddsScraper(odds_api_key=odds_api_key)
    live_odds = scraper.get_live_odds(week, yahoo_games=yahoo_games)
    return scraper.update_yahoo_odds(yahoo_games, live_odds)
