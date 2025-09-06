import pytest
from unittest.mock import patch, MagicMock, mock_open
import pandas as pd
from pathlib import Path
import requests

from src.confpickem.yahoo_pickem_scraper import YahooPickEm, PageCache, calculate_player_stats

# Sample HTML content for testing
SAMPLE_PICK_DIST_HTML = """
    <div class="ysf-matchup-dist yspmainmodule">
        <div class="hd"><h4><span>Thursday, Sep 5, 8:20 pm EDT</span></h4></div>
        <div class="bd">
                        <dl class="favorite pick-preferred">
                <dt class="team">Favorite</dt>
                <dd class="team">@ <a href="https://sports.yahoo.com/nfl/teams/kansas-city/" target="sports">Kansas City</a> </dd>
                <dt class="percent"><span style="width:75%">Favorite Pick Percentage</span></dt>
                <dd class="percent">75%</dd>
            </dl>
            <dl class="underdog pick-loser">
                <dt class="team">Underdog</dt>
                <dd class="team"><a href="https://sports.yahoo.com/nfl/teams/baltimore/" target="sports">Baltimore</a> </dd>
                <dt class="percent"><span style="width:25%">Underdog Pick Percentage</span></dt>
                <dd class="percent">25%</dd>
            </dl>
        </div>
        <div class="ft">
        <table>
            <thead>
                <tr>
                    <th>KC</th>
                    <th></th>
                    <th>Bal</th>
                </tr>
            </thead>
            <tbody>
                <tr class="odd first">
                    <td>6.8</td>
                    <td class="label">Avg. Confidence</td>
                    <td>3.7</td>
                </tr>
                <tr class="even">
                    <td>15-1 (8-0)</td>
                    <td class="label">Record</td>
                    <td>11-5 (6-3)</td>
                </tr>
                <tr class="odd">
                    <td>3.0 pts</td>
                    <td class="label">Favorite</td>
                    <td>&nbsp;</td>
                </tr>
            </tbody>
        </table>
        </div>
    </div>    <div class="ysf-matchup-dist yspmainmodule">
        <div class="hd"><h4><span>Friday, Sep 6, 8:15 pm EDT</span></h4></div>
        <div class="bd">
                        <dl class="favorite pick-preferred">
                <dt class="team">Favorite</dt>
                <dd class="team">@ <a href="https://sports.yahoo.com/nfl/teams/philadelphia/" target="sports">Philadelphia</a> </dd>
                <dt class="percent"><span style="width:71%">Favorite Pick Percentage</span></dt>
                <dd class="percent">71%</dd>
            </dl>
            <dl class="underdog pick-loser">
                <dt class="team">Underdog</dt>
                <dd class="team"><a href="https://sports.yahoo.com/nfl/teams/green-bay/" target="sports">Green Bay</a> </dd>
                <dt class="percent"><span style="width:29%">Underdog Pick Percentage</span></dt>
                <dd class="percent">29%</dd>
            </dl>
        </div>
        <div class="ft">
        <table>
            <thead>
                <tr>
                    <th>Phi</th>
                    <th></th>
                    <th>GB</th>
                </tr>
            </thead>
            <tbody>
                <tr class="odd first">
                    <td>6.4</td>
                    <td class="label">Avg. Confidence</td>
                    <td>4.5</td>
                </tr>
                <tr class="even">
                    <td>13-3 (7-1)</td>
                    <td class="label">Record</td>
                    <td>11-5 (5-3)</td>
                </tr>
                <tr class="odd">
                    <td>2.5 pts</td>
                    <td class="label">Favorite</td>
                    <td>&nbsp;</td>
                </tr>
            </tbody>
        </table>
        </div>
    </div>    <div class="ysf-matchup-dist yspmainmodule">
        <div class="hd"><h4><span>Sunday, Sep 8, 1:00 pm EDT</span></h4></div>
        <div class="bd">
                        <dl class="favorite pick-preferred">
                <dt class="team">Favorite</dt>
                <dd class="team">@ <a href="https://sports.yahoo.com/nfl/teams/atlanta/" target="sports">Atlanta</a> </dd>
                <dt class="percent"><span style="width:77%">Favorite Pick Percentage</span></dt>
                <dd class="percent">77%</dd>
            </dl>
            <dl class="underdog pick-loser">
                <dt class="team">Underdog</dt>
                <dd class="team"><a href="https://sports.yahoo.com/nfl/teams/pittsburgh/" target="sports">Pittsburgh</a> </dd>
                <dt class="percent"><span style="width:23%">Underdog Pick Percentage</span></dt>
                <dd class="percent">23%</dd>
            </dl>
        </div>
        <div class="ft">
        <table>
            <thead>
                <tr>
                    <th>Atl</th>
                    <th></th>
                    <th>Pit</th>
                </tr>
            </thead>
            <tbody>
                <tr class="odd first">
                    <td>6.5</td>
                    <td class="label">Avg. Confidence</td>
                    <td>4.6</td>
                </tr>
                <tr class="even">
                    <td>8-8 (4-4)</td>
                    <td class="label">Record</td>
                    <td>10-6 (5-4)</td>
                </tr>
                <tr class="odd">
                    <td>3.0 pts</td>
                    <td class="label">Favorite</td>
                    <td>&nbsp;</td>
                </tr>
            </tbody>
        </table>
        </div>
    </div>    <div class="ysf-matchup-dist yspmainmodule">
        <div class="hd"><h4><span>Sunday, Sep 8, 1:00 pm EDT</span></h4></div>
        <div class="bd">
                        <dl class="favorite pick-preferred">
                <dt class="team">Favorite</dt>
                <dd class="team">@ <a href="https://sports.yahoo.com/nfl/teams/buffalo/" target="sports">Buffalo</a> </dd>
                <dt class="percent"><span style="width:96%">Favorite Pick Percentage</span></dt>
                <dd class="percent">96%</dd>
            </dl>
            <dl class="underdog pick-loser">
                <dt class="team">Underdog</dt>
                <dd class="team"><a href="https://sports.yahoo.com/nfl/teams/arizona/" target="sports">Arizona</a> </dd>
                <dt class="percent"><span style="width:4%">Underdog Pick Percentage</span></dt>
                <dd class="percent">4%</dd>
            </dl>
        </div>
        <div class="ft">
        <table>
            <thead>
                <tr>
                    <th>Buf</th>
                    <th></th>
                    <th>Ari</th>
                </tr>
            </thead>
            <tbody>
                <tr class="odd first">
                    <td>12.7</td>
                    <td class="label">Avg. Confidence</td>
                    <td>4.7</td>
                </tr>
                <tr class="even">
                    <td>13-3 (8-0)</td>
                    <td class="label">Record</td>
                    <td>7-9 (2-6)</td>
                </tr>
                <tr class="odd">
                    <td>6.5 pts</td>
                    <td class="label">Favorite</td>
                    <td>&nbsp;</td>
                </tr>
            </tbody>
        </table>
        </div>
    </div>

<!-- fantasy-sports-fe- -rhel7-production-bf1-89c65d566-gl6cn Thu Dec 25 07:34:25 UTC 2024 -->
"""

SAMPLE_CONFIDENCE_PICKS_HTML = """
<div id="ysf-group-picks" class="data-table">
    <table border="0" cellpadding="0" cellspacing="0" class="yspNflPickGroupPickTable yspNflPickGroupPickTablePadded">
        <tbody class="ysptblcontent1">
            <tr class="data-row even">
                <td class="l" scope="row">Favored</td>
                <td width="33" class="yspNflPickWin">KC</td>
                <td width="33" class="yspNflPickWin">Phi</td>
                <td width="33">Atl</td>
                <td width="33" class="yspNflPickWin">Buf</td>
                <td>&nbsp;</td>
            </tr>
            <tr class="data-row">
                <td class="l" scope="row">Spread</td>
                <td>3.0</td>
                <td>2.5</td>
                <td>3.0</td>
                <td>6.5</td>
                <td>&nbsp;</td>
            </tr>
            <tr class="data-row even">
                <td class="l" scope="row">Underdog</td>
                <td>Bal</td>
                <td>GB</td>
                <td class="yspNflPickWin">Pit</td>
                <td>Ari</td>
                <td>&nbsp;</td>
            </tr>


            <tr class="ysptblhead">
                <th colspan="17">Team Name</th>
                <th class="c" colspan="2" id="sum-header">Points</th>
            </tr>
            <tr class="data-row odd">
                <td scope="row" class="l"><a href="/pickem/12345/40">Player 1</a></td>
                <td class="ysf-pick-opponent incorrect">Bal<br>(2)</td>
                <td class="ysf-pick-opponent correct">Phi<br>(10)</td>
                <td class="correct">Pit<br>(3)</td>
                <td class="correct">Buf<br>(12)</td>
                <td class="sum"><strong>25</strong></td>
            </tr>
            <tr class="data-row even">
                <td scope="row" class="l"><a href="/pickem/12345/11">Player 2</a></td>
                <td class="correct">KC<br>(3)</td>
                <td class="ysf-pick-opponent correct">Phi<br>(4)</td>
                <td class="ysf-pick-opponent incorrect">Atl<br>(1)</td>
                <td class="correct">Buf<br>(16)</td>
                <td class="sum"><strong>23</strong></td>
            </tr>
        </tbody>
    </table>
</div>

<!-- fantasy-sports-fe- -rhel7-production-bf1-89c65d566-9jfkf Thu Jan  2 07:34:38 UTC 2025 -->
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
    with patch('src.confpickem.yahoo_pickem_scraper.PageCache') as mock:
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
    
    # Check sample data parsing from our realistic example
    assert games_df.iloc[0]['favorite'] == 'KC'
    assert games_df.iloc[0]['favorite_pick_pct'] == 75.0
    assert games_df.iloc[0]['underdog'] == 'Bal'
    assert games_df.iloc[0]['underdog_pick_pct'] == 25.0
    assert games_df.iloc[0]['favorite_confidence'] == 6.8
    assert games_df.iloc[0]['underdog_confidence'] == 3.7
    assert games_df.iloc[0]['spread'] == 3.0

def test_parse_confidence_picks(yahoo_pickem):
    """Test parsing of confidence picks page"""
    players_df = yahoo_pickem.players
    
    # Check DataFrame structure
    assert 'player_name' in players_df.columns
    assert 'total_points' in players_df.columns
    
    # Check sample data parsing from our realistic example
    assert len(players_df) == 2
    assert players_df.iloc[0]['player_name'] == 'Player 1'
    assert players_df.iloc[0]['total_points'] == 25

    # Check specific game picks
    assert players_df.iloc[0]['game_1_pick'] == 'Bal'
    assert players_df.iloc[0]['game_1_confidence'] == 2
    assert players_df.iloc[0]['game_1_correct'] == False
    
    assert players_df.iloc[1]['game_2_pick'] == 'Phi'
    assert players_df.iloc[1]['game_2_confidence'] == 4
    assert players_df.iloc[1]['game_2_correct'] == True

def test_calculate_player_stats():
    """Test player statistics calculation"""
    with patch('src.confpickem.yahoo_pickem_scraper.YahooPickEm') as mock_yahoo, \
         patch('http.cookiejar.MozillaCookieJar') as mock_cookiejar:
        
        # Configure mock cookie jar
        mock_jar = MagicMock()
        mock_jar.load = MagicMock()
        mock_cookiejar.return_value = mock_jar
        
        # Configure mock YahooPickEm instance
        instance = MagicMock()
        instance.players = pd.DataFrame({
            'player_name': ['Player 1', 'Player 2'],
            'game_1_pick': ['KC', 'KC'],
            'game_1_confidence': [5, 10],
            'game_1_correct': [True, True],
            'week': [1, 1]
        })
        instance.games = pd.DataFrame({
            'favorite': ['KC'],
            'favorite_pick_pct': [75.0],
            'favorite_confidence': [6.8],
            'underdog_confidence': [3.7],
            'week': [1]
        })
        mock_yahoo.return_value = instance
        
        stats = calculate_player_stats(
            league_id=12345,
            weeks=[1],
            cookies_file='mock_cookies.txt'
        )
        
        # Check DataFrame structure
        expected_cols = ['player', 'skill_level', 'crowd_following', 'confidence_following', 'total_picks']
        assert all(col in stats.columns for col in expected_cols)
        
        # Check stat ranges
        for col in ['skill_level', 'crowd_following', 'confidence_following']:
            assert (stats[col] >= 0).all()
            assert (stats[col] <= 1).all()

def test_error_handling(yahoo_pickem, mock_session):
    """Test error handling for failed requests"""
    # Mock at the lower get_page_content level directly
    with patch.object(yahoo_pickem, 'get_page_content') as mock_get_content:
        mock_get_content.side_effect = requests.exceptions.RequestException("Connection failed")
        
        with pytest.raises(requests.exceptions.RequestException):
            yahoo_pickem.get_pick_distribution()

def test_cached_content():
    """Test cached content retrieval"""
    cached_html = SAMPLE_PICK_DIST_HTML
    
    with patch('requests.Session') as mock_session, \
         patch('http.cookiejar.MozillaCookieJar') as mock_cookiejar, \
         patch('src.confpickem.yahoo_pickem_scraper.PageCache') as mock_cache_class:
        
        # Configure mock cookie jar
        mock_jar = MagicMock()
        mock_jar.load = MagicMock()
        mock_cookiejar.return_value = mock_jar
        
        # Configure mock cache instance
        mock_cache = MagicMock()
        mock_cache.get_cached_content.side_effect = [None, None, cached_html]  # First two for init, third for test
        mock_cache_class.return_value = mock_cache
        
        # Configure mock session
        mock_response1 = MagicMock()
        mock_response1.text = cached_html
        mock_response1.raise_for_status.return_value = None
        
        mock_response2 = MagicMock()
        mock_response2.text = SAMPLE_CONFIDENCE_PICKS_HTML
        mock_response2.raise_for_status.return_value = None
        
        mock_session.return_value.get.side_effect = [mock_response1, mock_response2]
        
        # Initialize YahooPickEm
        yahoo = YahooPickEm(week=1, league_id=12345, cookies_file='cookies.txt')
        
        # Test page content retrieval from cache
        test_content = yahoo.get_page_content("http://test.com", "test_page")
        assert test_content == cached_html
        
        # Verify cookie jar was properly initialized
        mock_cookiejar.assert_called_once_with('cookies.txt')
        mock_jar.load.assert_called_once_with(ignore_discard=True, ignore_expires=True)

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
