# ConfPickEm CLI Tools

Command-line interface for NFL Confidence Pick'Em optimization and analysis.

## Installation

First, install the package to get the CLI commands:

```bash
cd /path/to/confpickem
pip install -e .
```

This creates convenient commands: `confpickem`, `confpickem-win-probability`, and `confpickem-player-skills`.

## Quick Start

```bash
# Optimize picks for mid-week with live odds
confpickem --week 10 --mode midweek --live-odds

# Check win probabilities
confpickem-win-probability --week 10 --live-odds

# Update player skills
confpickem-player-skills update --weeks 3,4,5,6,7,8,9 --week 10
```

## Available Commands

### 1. `confpickem` - Pick Optimization

Optimizes your confidence pick assignments using Monte Carlo simulation.

**Usage:**
```bash
confpickem --week WEEK --mode MODE [OPTIONS]
```

**Modes:**
- `beginning` - All games are pending (start of week)
- `midweek` - Some games completed (mid-week optimization)

**Options:**
```
--week, -w         NFL week number (required)
--league-id, -l    Yahoo league ID (default: 15435)
--mode, -m         'beginning' or 'midweek' (required)
--live-odds        Use live Vegas odds
--odds-api-key, -k The Odds API key
--fast             Fast mode (~85% accuracy, 10x speed)
--num-sims, -n     Number of simulations
--no-cache         Clear cache before loading
--hill-climb       Use hill climbing instead of greedy (better results, slower)
--hc-iterations    Iterations per restart (default: 1000)
--hc-restarts      Random restarts (default: 10)
--hc-top-n         Top combinations for robustness analysis (default: 1000)
```

**Examples:**
```bash
# Mid-week with live odds (most accurate)
confpickem --week 10 --mode midweek --live-odds

# Beginning of week, fast mode
confpickem --week 10 --mode beginning --fast

# Custom simulations
confpickem --week 10 --mode beginning --num-sims 10000

# Hill climbing with recommended starting parameters
confpickem --week 18 --mode midweek --live-odds --hill-climb \
  --hc-iterations 100 --hc-restarts 5 --num-sims 500 --hc-top-n 250
```

**Hill Climbing vs Greedy:**
- **Greedy (default):** Fast, picks the best option at each step. Good for quick decisions.
- **Hill Climbing:** Explores more of the solution space via random restarts. Better results but slower. The robustness analysis shows how often each team appears across top solutions, helping identify "lock" picks vs. volatile ones.

**Interactive Features:**

The optimizer will prompt you to:
1. **Select your player** - Choose which player to optimize for
2. **Enter fixed picks** (optional) - Lock in specific picks you want to keep
   - Format: `PHI:16,KC:15,SF:14` (TEAM:CONFIDENCE pairs, comma-separated)
   - Useful for constraining optimization or testing specific pick combinations
   - Leave blank to optimize all games freely

---

### 2. `confpickem-win-probability` - Win Probability Calculator

Calculates win probabilities for all players using Monte Carlo simulation.

**Usage:**
```bash
confpickem-win-probability --week WEEK [OPTIONS]
```

**Options:**
```
--week, -w         NFL week number (required)
--league-id, -l    Yahoo league ID (default: 15435)
--live-odds        Use live Vegas odds
--odds-api-key, -k The Odds API key
--num-sims, -n     Number of simulations (default: 5000)
```

**Examples:**
```bash
# Basic win probabilities
confpickem-win-probability --week 10

# With live Vegas odds
confpickem-win-probability --week 10 --live-odds

# More simulations for accuracy
confpickem-win-probability --week 10 --num-sims 10000
```

---

### 3. `confpickem-player-skills` - Player Skills Management

Analyzes historical player performance and applies realistic skill levels to the simulator.

**Usage:**
```bash
confpickem-player-skills COMMAND [OPTIONS]
```

**Commands:**
- `analyze` - Analyze historical performance
- `apply` - Apply saved skills to simulator
- `update` - Do both: analyze then apply

**Options:**
```
analyze:
  --year, -y       Year to analyze (default: 2024)
  --league-id, -l  Yahoo league ID (default: 15435)

apply:
  --year, -y       Year to use (default: combine all available years)
  --league-id, -l  Yahoo league ID (default: 15435)

update:
  --years, -y      Years to analyze, comma-separated (e.g., 2024,2025)
                   If not specified, skips analysis and applies all available
  --league-id, -l  Yahoo league ID (default: 15435)
```

**Examples:**
```bash
# Analyze a single season
confpickem-player-skills analyze --year 2024

# Apply skills from all available years (combines 2024 + 2025 if both exist)
confpickem-player-skills apply

# Apply skills from a specific year only
confpickem-player-skills apply --year 2025

# Analyze multiple years and apply combined skills
confpickem-player-skills update --years 2024,2025

# Just apply all available years (no analysis)
confpickem-player-skills update
```

---

## Prerequisites

### Required Files

1. **cookies.txt** - Yahoo session cookies in project root
2. **current_player_skills.json** (optional) - Player skill levels (generated by player_skills.py)

### Environment Variables

```bash
# Optionally set your Odds API key
export ODDS_API_KEY=your_key_here

# Then you can omit --odds-api-key flag
python src/confpickem/cli/optimize.py --week 10 --mode midweek --live-odds
```

---

## Typical Weekly Workflow

**Beginning of Week:**
```bash
# 1. Update player skills (optional, do once per season or periodically)
confpickem-player-skills update --weeks 3,4,5,6,7,8,9 --week 10

# 2. Get your optimal picks
confpickem --week 10 --mode beginning --live-odds
```

**Mid-Week (After Thursday/Friday Games):**
```bash
# Re-optimize with completed game results and live odds
confpickem --week 10 --mode midweek --live-odds
```

**Check Your Position:**
```bash
# See everyone's win probabilities
confpickem-win-probability --week 10 --live-odds
```

---

## Tips & Best Practices

### 1. Use Live Odds
Live Vegas odds are more accurate than Yahoo spreads:
```bash
--live-odds --odds-api-key YOUR_KEY
```

### 2. Mid-Week Re-Optimization
After Thursday/Friday games, re-optimize with `--mode midweek` to account for completed results.

### 3. Fast Mode for Quick Decisions
Use `--fast` when you need results quickly (~85% accuracy):
```bash
--mode beginning --fast
```

### 4. Use Hill Climbing for Better Results
When accuracy matters more than speed, use hill climbing:
```bash
confpickem --week 10 --mode midweek --live-odds --hill-climb \
  --hc-iterations 100 --hc-restarts 5 --num-sims 500 --hc-top-n 250
```
The robustness analysis helps identify which picks are "locks" (appear in 90%+ of top solutions) vs. uncertain.

### 5. Update Player Skills Periodically
Refresh player skills when new season data is available:
```bash
# Analyze both seasons and apply combined skills
confpickem-player-skills update --years 2024,2025
```

### 6. Set Environment Variables
Set `ODDS_API_KEY` once instead of passing it every time.

---

## Getting Help

Each command has comprehensive help:
```bash
confpickem --help
confpickem-win-probability --help
confpickem-player-skills --help
```

For subcommands:
```bash
confpickem-player-skills analyze --help
```

---

## Troubleshooting

### Command Not Found
Make sure you've installed the package:
```bash
cd /path/to/confpickem
pip install -e .
```

### Missing cookies.txt
Ensure `cookies.txt` exists in project root with valid Yahoo session cookies.

### Odds API Errors
Check your API key and rate limits at [The Odds API](https://the-odds-api.com/).

### Slow Performance
- Use `--fast` mode for optimization
- Reduce `--num-sims` (trade accuracy for speed)
- Use cached data (don't use `--no-cache`)
