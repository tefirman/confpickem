#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   yahoo_pickem_scraper.py
@Time    :   2024/12/07 13:38:47
@Author  :   Taylor Firman
@Version :   v0.1
@Contact :   tefirman@gmail.com
@Desc    :   Scraper for Yahoo NFL Confidence Pick 'Em Groups
'''

import requests
import http.cookiejar
import pandas as pd
from bs4 import BeautifulSoup, Comment
import json
from datetime import datetime
from pathlib import Path
from datetime import timedelta

class PageCache:
    def __init__(self, cache_dir: str = ".cache"):
        """Initialize cache with specified directory"""
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
    
    def get_cache_path(self, page_type: str, week: int) -> Path:
        """Generate cache file path for given page type and week"""
        return self.cache_dir / f"{page_type}_week{week}.html"
    
    def get_metadata_path(self, page_type: str, week: int) -> Path:
        """Generate metadata file path for given page type and week"""
        return self.cache_dir / f"{page_type}_week{week}_meta.json"
    
    def get_cached_content(self, page_type: str, week: int, expiration: int = 86400) -> str:
        """Retrieve cached content if it exists and is not expired"""
        cache_path = self.get_cache_path(page_type, week)
        meta_path = self.get_metadata_path(page_type, week)
        
        if not cache_path.exists() or not meta_path.exists():
            return None
            
        with open(meta_path) as f:
            metadata = json.load(f)
            
        # Cache expires after 1 day
        cache_age = datetime.now() - datetime.fromisoformat(metadata['timestamp'])
        if cache_age.total_seconds() > expiration:
            return None
            
        with open(cache_path) as f:
            return f.read()
    
    def save_content(self, content: str, page_type: str, week: int):
        """Save content to cache with metadata"""
        cache_path = self.get_cache_path(page_type, week)
        meta_path = self.get_metadata_path(page_type, week)
        
        with open(cache_path, 'w') as f:
            f.write(content)
            
        metadata = {
            'timestamp': datetime.now().isoformat(),
            'page_type': page_type,
            'week': week
        }
        
        with open(meta_path, 'w') as f:
            json.dump(metadata, f)

class YahooPickEm:
    def __init__(self, week: int, league_id: int, cookies_file: str, cache_dir: str = ".cache"):
        """
        Initialize scraper using cookies from exported cookies.txt file
        
        Args:
            week (int): NFL week number
            league_id (int): Yahoo Pick'em league ID
            cookies_file (str): Path to exported cookies.txt file
            cache_dir (str): Directory to store cached pages
        """
        self.week = week
        self.league_id = league_id
        self.cache = PageCache(cache_dir)
        
        # Create session with cookies
        self.session = requests.Session()
        cookie_jar = http.cookiejar.MozillaCookieJar(cookies_file)
        cookie_jar.load(ignore_discard=True, ignore_expires=True)
        self.session.cookies = cookie_jar
        
        # Set common headers
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
        })
        
        # Load initial data
        self.get_pick_distribution()
        self.get_confidence_picks()

    def get_page_content(self, url: str, page_type: str) -> str:
        """
        Fetch page content using authenticated session or cache
        """
        cached_content = self.cache.get_cached_content(page_type, self.week)
        if cached_content:
            return cached_content
            
        response = self.session.get(url)
        response.raise_for_status()
        content = response.text
        
        self.cache.save_content(content, page_type, self.week)
        return content

    def get_pick_distribution(self):
        """
        Parse Yahoo Fantasy Pick Distribution page
        """
        url = f"https://football.fantasysports.yahoo.com/pickem/pickdistribution?gid=&week={self.week}&type=c"
        content = self.get_page_content(url, "pick_distribution")
        soup = BeautifulSoup(content, 'html.parser')
        
        # Pull season from commented timestamp
        comments = soup.find_all(string=lambda text: isinstance(text, Comment))
        last_comment = comments[-1]
        date_pulled = pd.to_datetime(" ".join(last_comment.strip().split()[-6:]),format="%a %b %d %H:%M:%S %Z %Y")
        season = (date_pulled - timedelta(days=90)).year # Accounting for week 18 being in January of the following year...

        # Find all game containers 
        games = soup.find_all('div', class_='ysf-matchup-dist')
        
        games_data = []
        for game in games:
            game_dict = {}
            
            # Get team names and pick percentages
            teams = game.find_all('th')
            teams_full = game.find_all('dd', class_='team')
            percentages = game.find_all('dd', class_='percent')
            
            game_dict['favorite'] = teams[0].text.strip()
            game_dict['favorite_pick_pct'] = float(percentages[0].text.strip().replace('%', ''))
            game_dict['underdog'] = teams[-1].text.strip()
            game_dict['underdog_pick_pct'] = float(percentages[1].text.strip().replace('%', ''))
            game_dict['home_favorite'] = teams_full[-1].text.strip().startswith("@ ")
            
            # Get confidence values
            ft = game.find('div', class_='ft')
            confidence_row = ft.find('tr', class_="odd first").find_all('td')
            game_dict['favorite_confidence'] = float(confidence_row[0].text.strip())
            game_dict['underdog_confidence'] = float(confidence_row[2].text.strip())

            # Get confidence values
            ft = game.find('div', class_='ft')
            spread_row = ft.find('tr', class_="odd").find_all('td')
            game_dict['spread'] = float(spread_row[0].text.strip().split()[0])
            game_dict['win_prob'] = min(max(game_dict['spread'] * 0.031 + 0.5,0.0),1.0)

            # Parse kickoff time
            time_element = game.find('div', class_='hd')
            if time_element:
                kickoff_time = pd.to_datetime(time_element.text.strip().replace(" EDT"," EST"), format="%A, %b %d, %I:%M %p %Z")
                kickoff_time = kickoff_time.replace(year=season)
                game_dict['kickoff_time'] = kickoff_time

            games_data.append(game_dict)
            
        self.games = pd.DataFrame(games_data)

    def get_confidence_picks(self):
        """
        Parse group picks page for confidence pool
        """
        url = f"https://football.fantasysports.yahoo.com/pickem/{self.league_id}/grouppicks?week={self.week}"
        content = self.get_page_content(url, "confidence_picks")
        soup = BeautifulSoup(content, 'html.parser')
        
        # Find the main picks table
        table = soup.find('div', {'id': 'ysf-group-picks'})

        # Parse game results from header
        games_meta = [row.find_all('td')[1:-1] for row in table.find_all('tr')[:3]]
        games = []
        for game in range(len(games_meta[0])):
            favorite = games_meta[0][game].text.strip()
            underdog = games_meta[2][game].text.strip()
            spread = games_meta[1][game].text.strip()
            if "Off" in spread: # Need to figure out what to do here...
                spread = 0.0
            else:
                spread = float(spread)
            
            winner = None
            if "yspNflPickWin" in games_meta[0][game].get("class",[]):
                winner = favorite
            elif "yspNflPickWin" in games_meta[2][game].get("class",[]):
                winner = underdog

            games.append({
                'favorite': favorite,
                'underdog': underdog,
                'spread': spread,
                'winner': winner
            })
        
        # Parse each player's picks
        players = []
        for row in table.find_all('tr')[3:]:  # Skip header rows
            cols = row.find_all('td')
            if not cols or len(cols) <= 1:
                continue
                
            player_data = {
                'player_name': cols[0].find('a').text.strip() if cols[0].find('a') else cols[0].text.strip(),
                'total_points': int(cols[-1].text.strip()) if cols[-1].text.strip() != '' else 0
            }
            
            # Parse each pick
            for i, col in enumerate(cols[1:-1]):
                if col.text.strip() not in ["", "--"]:
                    pick_text = col.text.strip()
                    team = pick_text.split('(')[0].strip()
                    confidence = int(pick_text.split('(')[1].replace(')', ''))
                    is_correct = team.strip() == games[i]['winner'].strip() if games[i]['winner'] else None
                    
                    player_data[f'game_{i+1}_pick'] = team
                    player_data[f'game_{i+1}_confidence'] = confidence
                    player_data[f'game_{i+1}_correct'] = is_correct
                    player_data[f'game_{i+1}_points'] = confidence if is_correct else 0
                else:
                    player_data[f'game_{i+1}_pick'] = None
                    player_data[f'game_{i+1}_confidence'] = None
                    player_data[f'game_{i+1}_correct'] = None
                    player_data[f'game_{i+1}_points'] = 0
                    
            players.append(player_data)
        
        self.players = pd.DataFrame(players)
        self.results = games

def calculate_player_stats(league_id: int, weeks: list, cookies_file: str):
    """
    Calculate player statistics based on historical performance.
    
    Args:
        league_id: Yahoo Pick'em league ID
        weeks: List of weeks to analyze
        cookies_file: Path to cookies.txt file
        
    Returns:
        DataFrame containing player statistics
    """
    all_picks = []
    all_games = []
    
    # Gather historical data
    for week in weeks:
        yahoo = YahooPickEm(week=week, league_id=league_id, cookies_file=cookies_file)
        
        # Add week number to dataframes
        yahoo.players['week'] = week
        yahoo.games['week'] = week
        
        all_picks.append(yahoo.players)
        all_games.append(yahoo.games)
    
    picks_df = pd.concat(all_picks, ignore_index=True)
    games_df = pd.concat(all_games, ignore_index=True)
    
    player_stats = []
    
    for player in picks_df['player_name'].unique():
        player_picks = picks_df[picks_df['player_name'] == player]
        
        # Calculate skill level based on correct pick percentage
        total_picks = 0
        correct_picks = 0
        
        # Calculate crowd following based on agreement with majority
        crowd_agreement = 0
        total_comparable = 0
        
        # Calculate confidence alignment
        confidence_alignment = 0
        total_confidence = 0
        
        for week in player_picks['week'].unique():
            week_picks = player_picks[player_picks['week'] == week]
            week_games = games_df[games_df['week'] == week]
            
            for game_num in range(len(week_games)):
                pick_col = f'game_{game_num+1}_pick'
                conf_col = f'game_{game_num+1}_confidence'
                correct_col = f'game_{game_num+1}_correct'
                
                if pd.notna(week_picks[pick_col].iloc[0]):
                    pick = week_picks[pick_col].iloc[0]
                    confidence = week_picks[conf_col].iloc[0]
                    
                    # Skill level calculation
                    total_picks += 1
                    if week_picks[correct_col].iloc[0]:
                        correct_picks += 1
                    
                    # Crowd following calculation
                    game = week_games.iloc[game_num]
                    picked_favorite = (pick == game['favorite'])
                    majority_picked_favorite = (game['favorite_pick_pct'] > 50)
                    
                    total_comparable += 1
                    if picked_favorite == majority_picked_favorite:
                        crowd_agreement += 1
                    
                    # Confidence alignment calculation
                    expected_conf = game['favorite_confidence'] if picked_favorite else game['underdog_confidence']
                    total_confidence += 1
                    confidence_alignment += 1 - abs(confidence - expected_conf) / max(confidence, expected_conf)
        
        skill_level = correct_picks / total_picks if total_picks > 0 else 0.5
        crowd_following = crowd_agreement / total_comparable if total_comparable > 0 else 0.5
        confidence_following = confidence_alignment / total_confidence if total_confidence > 0 else 0.5
        
        player_stats.append({
            'player': player,
            'skill_level': skill_level,
            'crowd_following': crowd_following,
            'confidence_following': confidence_following,
            'total_picks': total_picks
        })
    
    stats_df = pd.DataFrame(player_stats)
    
    # Normalize stats to 0-1 range
    for col in ['skill_level', 'crowd_following', 'confidence_following']:
        min_val = stats_df[col].min()
        max_val = stats_df[col].max()
        if max_val > min_val:
            stats_df[col] = (stats_df[col] - min_val) / (max_val - min_val)
        else:
            stats_df[col] = 0.5  # Default to middle value if no variation
    
    return stats_df
