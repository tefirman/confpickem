# NFL Week 1 Pick Optimization Guide

## Quick Start

1. **Get your cookies file** (one-time setup):
   - Install browser extension like "cookies.txt" for Chrome/Firefox
   - Login to Yahoo Fantasy Sports
   - Export cookies to `cookies.txt` file
   - Place the file in the project root directory

2. **Run the optimization**:
   ```bash
   python optimize_week1_picks.py
   ```

3. **Follow the prompts**:
   - Select your player name
   - Optionally specify any fixed picks
   - Review the optimized recommendations

## What the Script Does

### Data Collection
- Fetches live Week 1 game data from Yahoo Pick'em
- Pulls current player standings and pick distributions
- Converts data to simulation format

### Optimization Process
- Runs 5,000+ Monte Carlo simulations
- Tests different confidence allocations for each game
- Finds the combination that maximizes your win probability
- Accounts for game certainty, crowd picks, and opponent behavior

### Results Analysis
- **Win Probability**: Shows your expected win rate vs random picks
- **Pick Rankings**: Displays recommended confidence allocation (1-16)
- **Game Impact**: Identifies which games matter most for your chances
- **Improvement**: Quantifies expected performance boost

## Example Output

```
🏆 Recommended Picks (sorted by confidence):
16. KC vs DEN (75% Vegas)
15. SF vs ARI (70% Vegas) 
14. BUF vs NYJ (68% Vegas)
...
 2. BAL vs CIN (52% Vegas)
 1. LAR vs SEA (51% Vegas)

📊 Optimization Analysis:
  📈 Optimized picks: 34.2%
  🎲 Random picks: 18.7%
  📊 Baseline (1/16): 6.2%
  ⬆️  Improvement: +15.5 percentage points
```

## Advanced Usage

### Fixed Picks
If you're confident about certain games, specify them:
```
Fixed picks: SF 16, KC 15, BUF 14
```

### Custom Parameters
Edit the script to adjust:
- `num_sims`: Number of simulations (more = more accurate, slower)
- `confidence_range`: How many confidence levels to test per game

## Troubleshooting

**"Cookies file not found"**
- Make sure `cookies.txt` is in the project root
- Ensure you're logged into Yahoo Fantasy Sports
- Re-export cookies if they're old

**"Failed to fetch Yahoo data"**
- Check your internet connection
- Verify the league ID (15435) is correct
- Make sure it's actually Week 1 and games haven't started yet

**Optimization taking too long**
- Reduce `num_sims` from 5000 to 1000-2000
- Reduce `confidence_range` from 3 to 2

## Tips for Better Results

1. **Run early in the week**: Before injury news affects lines
2. **Update cookies regularly**: They expire after a few days
3. **Consider fixed picks**: Lock in your strongest convictions
4. **Review game impact**: Focus on high-impact games for manual adjustments
5. **Monitor line movements**: Re-run if significant odds changes occur

Good luck! 🏈