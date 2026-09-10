<table>
<tr>
  <td width="225"><img src="https://raw.githubusercontent.com/tefirman/confpickem/refs/heads/main/assets/ConfPickEmLogo.png" width="100%" alt="confpickem logo"></td>
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

## Quick Start

### Command Line Interface (Recommended)

The easiest way to use confpickem is through the command-line tools:

```bash
# Optimize your picks (analytical optimizer, runs in seconds)
confpickem --week 10 --mode beginning

# Re-optimize once games start -- locks what's finished or kicked off
confpickem --week 10 --mode midweek

# Check win probabilities for all players
confpickem-win-probability --week 10 --live-odds

# Update player skills from historical data
confpickem-player-skills update --years 2024,2025
```

New here? **[docs/USAGE.md](docs/USAGE.md)** walks through a full week end to end
(getting `cookies.txt`, the weekly run, reading the output).

**Installation:** Install the package to get these commands:
```bash
pip install -e .   # From project root
```

See the [CLI Documentation](CLI_README.md) for full details.

### Python API

```python
import json
from confpickem import YahooPickEm, ConfidencePickEmSimulator, Player
from confpickem.yahoo_pickem_integration import convert_yahoo_to_simulator_format

yahoo = YahooPickEm(week=10, league_id=YOUR_LEAGUE_ID, cookies_file='cookies.txt')

sim = ConfidencePickEmSimulator(num_sims=2000)
sim.add_games_from_dataframe(convert_yahoo_to_simulator_format(yahoo, ignore_results=True))

skills = json.load(open('current_player_skills.json'))   # or {} for average skills
sim.players = [
    Player(nm,
           skills.get(nm, {}).get('skill_level', 0.6),
           skills.get(nm, {}).get('crowd_following', 0.5),
           skills.get(nm, {}).get('confidence_following', 0.5))
    for nm in yahoo.players['player_name']
]

picks = sim.optimize_picks_analytic("Your Yahoo Name", verbose=True)
print(picks)   # {TEAM: confidence}, a full 1..N assignment
```

See **[examples/yahoo_pickem_demo.ipynb](examples/yahoo_pickem_demo.ipynb)** for
the full flow (sanity checks, game importance, mid-week re-optimization).

## Features

### 🎯 Unified CLI Tools
- **optimize.py** - Comprehensive pick optimization with live odds support
- **win_probability.py** - Monte Carlo win probability calculator
- **player_skills.py** - Historical performance analysis and skill modeling

### 📊 Yahoo Data Scraping
- Scrape pick distributions and crowd confidence levels
- Track actual picks and results from your league
- Cache responses to avoid excessive requests

### 🎲 Simulation and Analysis
- Monte Carlo simulation of game outcomes
- Player skill modeling and analysis
- Pick optimization algorithms
- Risk/reward and game importance analysis

### 🔴 Live Vegas Odds Integration
- Real-time betting line integration via The Odds API
- More accurate win probabilities than Yahoo spreads
- Automatic fallback to Yahoo data when API unavailable

### 🧠 Strategy Optimization
- Analytical Poisson-binomial `P(win)` optimizer (default; noise-free, runs in seconds)
- Optimize confidence point assignments
- Mid-week / mid-Sunday re-optimization — locks games already finished or kicked off
- `--greedy` / `--hill-climb` for the older simulation-based optimizers

## Dependencies

- Python ≥ 3.8
- requests
- pandas 
- numpy
- beautifulsoup4
- scipy

## Methodology

The pick optimizer computes your probability of winning the week **analytically**
(the weekly score is a Poisson-binomial distribution) rather than by Monte Carlo
simulation. This matters: a naive hill climb on *simulated* win probability
optimizes noise, not strategy, and never beats "pick the favorites." The
analytical objective is noise-free, ~250× faster to evaluate, and in a 29-week
backtest produced 2 outright weekly wins vs. the greedy optimizer's 1, at a mean
finish of 34th vs. 43rd.

This is the **default** optimizer (`confpickem --week N --mode beginning`). It
handles mid-week runs too — games already finished or kicked off are locked to
your submitted picks and the rest optimized around them. Pass `--greedy` for the
old sequential optimizer or `--hill-climb` for the slower simulation search.

See **[docs/optimization-methodology.md](docs/optimization-methodology.md)** for
the full write-up.

## Documentation

- **[Usage Walkthrough](docs/USAGE.md)** - A full week end to end, from the command line
- **[Python API Demo](examples/yahoo_pickem_demo.ipynb)** - The same flow from Python
- **[Optimization Methodology](docs/optimization-methodology.md)** - How the optimizer works and why
- **[CLI Tools Guide](CLI_README.md)** - Full flag reference for the command-line tools
- **[CLI Tools (in package)](src/confpickem/cli/README.md)** - Detailed CLI documentation
- **[GitHub Repository](https://github.com/tefirman/confpickem)** - Source code and issues

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.