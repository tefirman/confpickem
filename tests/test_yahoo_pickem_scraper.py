import pytest
from unittest.mock import patch, MagicMock, mock_open
import pandas as pd
from datetime import datetime
from pathlib import Path
import json
from bs4 import BeautifulSoup

from yahoo_pickem_scraper import YahooPickEm, PageCache

# Sample HTML content for testing
SAMPLE_PICK_DIST_HTML = """
<div class="ysf-matchup-dist yspmainmodule">
    <div class="hd">Thursday, Sep 7, 8:20 PM EDT</div>
    <table>
        <tr><th>KC</th></tr>
        <tr><td>-6.5</td></tr>
        <tr><th>DET</th></tr>
    </table>
    <dd class="team">KC</dd>
    <dd class="percent">75%</dd>
    <dd class="team">@ DET</dd>
    <dd class="percent">25%</dd>
    <div class="ft">
        <table>
            <tr class="odd first">
                <td>12.5</td>
                <td></td>
                <td>4.5</td>
            </tr>
            <tr class="odd">
                <td>-6.5 </td>
            </tr>
        </table>
    </div>
</div>
"""

SAMPLE_CONFIDENCE_PICKS_HTML = """
<div id="ysf-group-picks">
    <tr>
        <td>KC</td>
        <td>SF</td>
    </tr>
    <tr>
        <td>-6.5</td>
        <td>-7</td>
    </tr>
    <tr>
        <td>DET</td>
        <td>LAR</td>
    </tr>
    <tr>
        <td><a>Player 1</a></td>
        <td>KC (16)</td>
        <td>SF (15)</td>
        <td>31</td>
    </tr>
    <tr>
        <td><a>Player 2</a></td>
        <td>DET (14)</td>
        <td>SF (13)</td>
        <td>13</td>
    </tr>
</div>
"""

@pytest.fixture
def mock_session():
    """Create a mock requests session"""
    with patch('requests.Session') as mock:
        # Configure mock response for pick distribution page
        dist_response = MagicMock()
        dist_response.text = SAMPLE_PICK_DIST_HTML
        dist_response.raise_for_status.return_value = None

        # Configure mock response for confidence picks page
        picks_response = MagicMock()
        picks_response.text = SAMPLE_CONFIDENCE_PICKS_HTML
        picks_response.raise_for_status.return_value = None
        
        # Configure session to return appropriate response based on URL
        def get_response(url):
            if 'pickdistribution' in url:
                return dist_response
            elif 'grouppicks' in url:
                return picks_response
            return MagicMock()
            
        mock.return_value.get.side_effect = get_response
        yield mock.return_value

@pytest.fixture
def mock_cache():
    """Create a mock page cache"""
    with patch('yahoo_pickem_scraper.PageCache') as mock:
        instance = mock.return_value
        instance.get_cached_content.return_value = None
        yield instance

@pytest.fixture
def mock_cookiejar():
    """Create a mock cookie jar"""
    with patch('http.cookiejar.MozillaCookieJar') as mock:
        instance = mock.return_value
        instance.load.return_value = None
        yield instance

@pytest.fixture
def yahoo_pickem(mock_session, mock_cache, mock_cookiejar):
    """Create a YahooPickEm instance with mocked dependencies"""
    return YahooPickEm(
        week=1,
        league_id=12345,
        cookies_file='cookies.txt'
    )

def test_page_cache_initialization():
    """Test PageCache initialization"""
    cache_dir = ".test_cache"
    cache = PageCache(cache_dir)
    assert cache.cache_dir == Path(cache_dir)
    assert cache.cache_dir.exists()

def test_page_cache_paths():
    """Test PageCache path generation"""
    cache = PageCache(".test_cache")
    week = 1
    page_type = "pick_distribution"
    
    cache_path = cache.get_cache_path(page_type, week)
    meta_path = cache.get_metadata_path(page_type, week)
    
    assert str(cache_path).endswith(f"{page_type}_week{week}.html")
    assert str(meta_path).endswith(f"{page_type}_week{week}_meta.json")

@patch('builtins.open', new_callable=mock_open)
def test_page_cache_save(mock_file):
    """Test PageCache content saving"""
    cache = PageCache(".test_cache")
    content = "<html>Test content</html>"
    page_type = "test_page"
    week = 1
    
    cache.save_content(content, page_type, week)
    
    # Check that both content and metadata files were written
    assert mock_file.call_count == 2
    
def test_yahoo_pickem_initialization(yahoo_pickem):
    """Test YahooPickEm initialization"""
    assert yahoo_pickem.week == 1
    assert yahoo_pickem.league_id == 12345
    assert hasattr(yahoo_pickem, 'session')
    assert hasattr(yahoo_pickem, 'games')
    assert hasattr(yahoo_pickem, 'players')

def test_parse_pick_distribution(yahoo_pickem):
    """Test parsing of pick distribution page"""
    games_df = yahoo_pickem.games
    
    # Check DataFrame structure
    expected_cols = [
        'favorite', 'favorite_pick_pct', 'underdog', 'underdog_pick_pct',
        'home_favorite', 'favorite_confidence', 'underdog_confidence',
        'spread', 'win_prob', 'kickoff_time'
    ]
    assert all(col in games_df.columns for col in expected_cols)
    
    # Check sample data parsing
    assert games_df.iloc[0]['favorite'] == 'KC'
    assert games_df.iloc[0]['favorite_pick_pct'] == 75.0
    assert games_df.iloc[0]['underdog'] == 'DET'
    assert games_df.iloc[0]['underdog_pick_pct'] == 25.0
    assert games_df.iloc[0]['favorite_confidence'] == 12.5
    assert games_df.iloc[0]['underdog_confidence'] == 4.5
    assert games_df.iloc[0]['spread'] == -6.5

def test_parse_confidence_picks(yahoo_pickem):
    """Test parsing of confidence picks page"""
    players_df = yahoo_pickem.players
    
    # Check DataFrame structure
    assert 'player_name' in players_df.columns
    assert 'total_points' in players_df.columns
    
    # Check sample data parsing
    assert len(players_df) == 2
    assert players_df.iloc[0]['player_name'] == 'Player 1'
    assert players_df.iloc[0]['total_points'] == 31
    assert players_df.iloc[0]['game_1_pick'] == 'KC'
    assert players_df.iloc[0]['game_1_confidence'] == 16

def test_calculate_player_stats():
    """Test player statistics calculation"""
    with patch('yahoo_pickem_scraper.YahooPickEm') as mock_yahoo:
        # Configure mock YahooPickEm instances
        instance = MagicMock()
        instance.players = pd.DataFrame({
            'player_name': ['Player 1', 'Player 2'],
            'game_1_pick': ['KC', 'DET'],
            'game_1_confidence': [16, 14],
            'game_1_correct': [True, False],
            'week': [1, 1]
        })
        instance.games = pd.DataFrame({
            'favorite': ['KC'],
            'favorite_pick_pct': [75.0],
            'favorite_confidence': [12.5],
            'underdog_confidence': [4.5],
            'week': [1]
        })
        mock_yahoo.return_value = instance
        
        stats = calculate_player_stats(
            league_id=12345,
            weeks=[1],
            cookies_file='cookies.txt'
        )
        
        # Check DataFrame structure
        expected_cols = ['player', 'skill_level', 'crowd_following', 'confidence_following', 'total_picks']
        assert all(col in stats.columns for col in expected_cols)
        
        # Check stat ranges
        assert all(0 <= stats[col] <= 1 for col in ['skill_level', 'crowd_following', 'confidence_following'])

def test_error_handling(yahoo_pickem, mock_session):
    """Test error handling for failed requests"""
    mock_session.get.side_effect = Exception("Connection failed")
    
    with pytest.raises(Exception):
        yahoo_pickem.get_pick_distribution()

def test_cached_content(mock_cache):
    """Test cached content retrieval"""
    cached_html = "<html>Cached content</html>"
    mock_cache.get_cached_content.return_value = cached_html
    
    with patch('requests.Session'):
        yahoo = YahooPickEm(week=1, league_id=12345, cookies_file='cookies.txt')
        content = yahoo.get_page_content("http://test.com", "test_page")
        assert content == cached_html

def test_cookie_handling(mock_cookiejar):
    """Test cookie handling"""
    with patch('requests.Session'):
        YahooPickEm(week=1, league_id=12345, cookies_file='cookies.txt')
        mock_cookiejar.load.assert_called_once_with(ignore_discard=True, ignore_expires=True)

def test_session_headers(mock_session):
    """Test session header configuration"""
    with patch('http.cookiejar.MozillaCookieJar'):
        YahooPickEm(week=1, league_id=12345, cookies_file='cookies.txt')
        headers = mock_session.headers.update.call_args[0][0]
        assert 'User-Agent' in headers
        assert 'Accept' in headers
        assert 'Accept-Language' in headers
