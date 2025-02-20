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

## Quick Start

```python
from confpickem import YahooPickEm, ConfidencePickEmSimulator, run_simulation

# Initialize scraper with your league info
yahoo = YahooPickEm(
    week=1,
    league_id=YOUR_LEAGUE_ID,
    cookies_file='cookies.txt'
)

# Run simulation with actual picks
simulator, stats = run_simulation(yahoo)

# Print expected points and win percentages
print("\nExpected Points by Player:")
print(stats['expected_points'])
print("\nWin Percentages:")
print(stats['win_pct'])
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