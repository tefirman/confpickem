"""
Yahoo NFL Confidence Pick'em Package

This package provides tools for scraping and analyzing Yahoo NFL Confidence Pick'em pools,
including simulation capabilities for optimizing picks.

Main Components:
- YahooPickEm: Scraper for Yahoo Pick'em league data
- ConfidencePickEmSimulator: Simulator for analyzing and optimizing picks
- Integration utilities for connecting scraper data to simulator
"""

from .yahoo_pickem_scraper import (
    YahooPickEm,
    PageCache,
    calculate_player_stats
)

from .confidence_pickem_sim import (
    ConfidencePickEmSimulator,
    Game,
    Player
)

from .yahoo_pickem_integration import (
    convert_yahoo_to_simulator_format,
    convert_yahoo_picks_to_dataframe,
    run_simulation
)

__version__ = '0.1.0'

__all__ = [
    # Scraper classes and functions
    'YahooPickEm',
    'PageCache',
    'calculate_player_stats',
    
    # Simulator classes
    'ConfidencePickEmSimulator',
    'Game',
    'Player',
    
    # Integration functions
    'convert_yahoo_to_simulator_format',
    'convert_yahoo_picks_to_dataframe',
    'run_simulation',
]