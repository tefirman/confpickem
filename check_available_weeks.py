#!/usr/bin/env python
"""Check what weeks and data are available in Yahoo league"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent / "src"))

from src.confpickem.yahoo_pickem_scraper import YahooPickEm

def check_weeks():
    """Check multiple weeks to find current NFL season data"""
    league_id = 15435
    
    # Check current date to determine season
    from datetime import datetime
    current_date = datetime.now()
    print(f"Current date: {current_date.strftime('%Y-%m-%d')}")
    
    # Try different weeks to find active NFL season
    for week in [1, 18, 19, 20, 21]:  # Current season might be in playoffs
        print(f"\\n🔍 Checking Week {week}...")
        try:
            yahoo = YahooPickEm(week=week, league_id=league_id, cookies_file="cookies.txt")
            
            print(f"✅ Week {week} data found:")
            print(f"   📊 Games: {len(yahoo.games)}")
            print(f"   👥 Players: {len(yahoo.players)}")
            
            if len(yahoo.games) > 0:
                print("   🏈 Sample games:")
                for idx in range(min(3, len(yahoo.games))):
                    game = yahoo.games.iloc[idx]
                    print(f"      {game['underdog']} @ {game['favorite']}")
            
            if len(yahoo.players) > 0:
                print("   👥 Sample players:")
                player_names = yahoo.players['player_name'].head(5).tolist()
                print(f"      {', '.join(player_names)}...")
                
            # If we found substantial data, this might be the active week
            if len(yahoo.games) >= 10 and len(yahoo.players) >= 5:
                print(f"\\n🎯 Week {week} looks like active NFL season data!")
                return week
                
        except Exception as e:
            print(f"❌ Week {week} failed: {str(e)}")
    
    print("\\n🤔 No active NFL season found. League might be inactive or between seasons.")
    return None

if __name__ == "__main__":
    active_week = check_weeks()
    if active_week:
        print(f"\\n✅ Use Week {active_week} for optimization!")
    else:
        print("\\n❌ No suitable week found for optimization.")