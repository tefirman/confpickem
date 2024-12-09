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
from bs4 import BeautifulSoup

class YahooPickEm:
    def __init__(self, week: int, league_id: int, cookies_file: str):
        """
        Initialize scraper using cookies from exported cookies.txt file
        
        Args:
            week (int): NFL week number
            league_id (int): Yahoo Pick'em league ID
            cookies_file (str): Path to exported cookies.txt file
        """
        self.week = week
        self.league_id = league_id
        
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

    def get_page_content(self, url: str) -> str:
        """
        Fetch page content using authenticated session
        """
        response = self.session.get(url)
        response.raise_for_status()
        return response.text

    def get_pick_distribution(self):
        """
        Parse Yahoo Fantasy Pick Distribution page
        """
        url = f"https://football.fantasysports.yahoo.com/pickem/pickdistribution?gid=&week={self.week}&type=c"
        content = self.get_page_content(url)
        soup = BeautifulSoup(content, 'html.parser')
        
        # Find all game containers 
        games = soup.find_all('div', class_='ysf-matchup-dist')
        
        games_data = []
        for game in games:
            game_dict = {}
            
            # Get team names and pick percentages
            teams = game.find_all('dd', class_='team')
            percentages = game.find_all('dd', class_='percent')
            
            game_dict['favorite'] = teams[0].text.strip().replace('@ ', '')
            game_dict['favorite_pick_pct'] = float(percentages[0].text.strip().replace('%', ''))
            game_dict['underdog'] = teams[1].text.strip().replace('@ ', '') 
            game_dict['underdog_pick_pct'] = float(percentages[1].text.strip().replace('%', ''))
            
            # Get confidence values
            ft = game.find('div', class_='ft')
            confidence_row = ft.find('tr', class_="odd first").find_all('td')
            game_dict['favorite_confidence'] = float(confidence_row[0].text.strip())
            game_dict['underdog_confidence'] = float(confidence_row[2].text.strip())
            
            games_data.append(game_dict)
            
        self.games = pd.DataFrame(games_data)

    def get_confidence_picks(self):
        """
        Parse group picks page for confidence pool
        """
        url = f"https://football.fantasysports.yahoo.com/pickem/{self.league_id}/grouppicks?week={self.week}"
        content = self.get_page_content(url)
        soup = BeautifulSoup(content, 'html.parser')
        
        # Find the main picks table
        table = soup.find('div', {'id': 'ysf-group-picks'})

        # Parse game results from header
        games_meta = [row.find_all('td')[1:-1] for row in table.find_all('tr')[:3]]
        games = []
        for game in range(len(games_meta[0])):
            favorite = games_meta[0][game].text.strip()
            underdog = games_meta[2][game].text.strip()
            spread = float(games_meta[1][game].text.strip())
            
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
                if col.text.strip():
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

# Example usage:
# league = YahooPickEm(week=13, league_id=6207, cookies_file='cookies.txt')
# print(league.games)  # Shows pick distribution
# print(league.players)  # Shows everyone's picks
# print(league.results)  # Shows game results
