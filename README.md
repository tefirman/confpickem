<table>
<tr>
  <td><img src="assets/ConfPickEmLogo.png" width="400" alt="confpickem logo"></td>
  <td>
    <h1>confpickem - NFL Confidence Pick'em Analyzer</h1>
    A Python package for analyzing and optimizing picks for NFL Confidence Pick'em pools. This package provides tools for:
    - Scraping Yahoo Pick'em league data
    - Analyzing pick distributions and trends
    - Simulating outcomes and optimizing picks
    - Evaluating different picking strategies
  </td>
</tr>
</table>

## Installation

You can install the package using pip:

```bash
pip install confpickem
```

Or install from source:

```bash
git clone https://github.com/tefirman/confpickem.git
cd confpickem
pip install -e .
```

## Quick Start - Command Line Interface

The easiest way to use confpickem is through the unified CLI:

```bash
# Optimize picks for a specific week
confpickem optimize --week 3 --league-id 15435

# Mid-week optimization with live Vegas odds
confpickem optimize --week 4 --mid-week --live-odds --odds-api-key YOUR_API_KEY

# Calculate win probabilities for all players
confpickem winprob --week 3 --league-id 15435

# Test and verify your optimized picks
confpickem test-picks --week 3 --league-id 15435

# Analyze player skills from historical data
confpickem analyze-skills
```

### Available Commands

- **`optimize`** - Optimize picks with optional live odds and mid-week support
  - `--week, -w` - NFL week number (default: 3)
  - `--league-id, -l` - Yahoo league ID (default: 15435)
  - `--num-sims, -n` - Number of simulations (default: 1000)
  - `--mid-week, -m` - Mid-week mode (accounts for completed games)
  - `--live-odds` - Use live Vegas odds
  - `--odds-api-key, -k` - The Odds API key for live odds

- **`winprob`** - Calculate win probabilities for all players
  - `--week, -w` - NFL week number
  - `--league-id, -l` - Yahoo league ID

- **`test-picks`** - Test and verify optimized picks
  - `--week, -w` - NFL week number
  - `--league-id, -l` - Yahoo league ID

- **`analyze-skills`** - Analyze player skills from historical data

## Python API

You can also use confpickem as a Python library:

```python
from confpickem import YahooPickEm, ConfidencePickEmSimulator

# Initialize scraper with your league info
yahoo = YahooPickEm(
    week=1,
    league_id=YOUR_LEAGUE_ID,
    cookies_file='cookies.txt'
)

# Setup simulator
simulator = ConfidencePickEmSimulator(num_sims=1000)
simulator.add_games_from_yahoo(yahoo)

# Optimize picks
optimal_picks = simulator.optimize_picks(
    player_name="YourName",
    confidence_range=4
)

print(f"Optimized picks: {optimal_picks}")
```

## Features

### Yahoo Data Scraping
- Scrape pick distributions and crowd confidence levels
- Track actual picks and results from your league
- Cache responses to avoid excessive requests

### Simulation and Analysis
- Monte Carlo simulation of game outcomes
- Player skill modeling and analysis
- Pick optimization algorithms
- Risk/reward analysis

### Strategy Optimization
- Evaluate different picking strategies
- Optimize confidence point assignments
- Analyze pick correlations and game importance

## Dependencies

- Python ≥ 3.8
- requests
- pandas 
- numpy
- beautifulsoup4
- scipy

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.