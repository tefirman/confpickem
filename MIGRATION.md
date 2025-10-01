# CLI Migration Guide

## Summary

All standalone scripts in the root directory have been consolidated into a unified CLI at `src/confpickem/cli.py`.

## New Unified CLI

After installing the package (`pip install -e .`), you can now use:

```bash
confpickem <command> [options]
```

## Command Mapping

### Old Scripts → New CLI Commands

| Old Script | New Command | Notes |
|------------|-------------|-------|
| `optimize_with_live_odds.py` | `confpickem optimize --live-odds` | Combines basic optimization with live odds support |
| `optimize_mid_week_live_odds.py` | `confpickem optimize --mid-week --live-odds` | Mid-week mode with live odds |
| `optimize_mid_week.py` | `confpickem optimize --mid-week` | Mid-week optimization without live odds |
| `optimize_ultra_fast.py` | `confpickem optimize --num-sims 500` | Use fewer simulations for speed |
| `win_probability_simple.py` | `confpickem winprob` | Calculate win probabilities |
| `win_probability_live_odds.py` | `confpickem winprob` | Integrated into main winprob command |
| `test_picks.py` | `confpickem test-picks` | Test and verify picks |
| `test_picks_midweek.py` | `confpickem test-picks` | Integrated into main test command |
| `analyze_player_skills.py` | `confpickem analyze-skills` | Analyze player skills |

## Examples

### Basic Optimization
```bash
# Old way
python optimize_with_live_odds.py --week 3 --league-id 15435

# New way
confpickem optimize --week 3 --league-id 15435
```

### Mid-Week with Live Odds
```bash
# Old way
python optimize_mid_week_live_odds.py --week 4 --odds-api-key YOUR_KEY

# New way
confpickem optimize --week 4 --mid-week --live-odds --odds-api-key YOUR_KEY
```

### Win Probabilities
```bash
# Old way
python win_probability_simple.py --week 3

# New way
confpickem winprob --week 3
```

### Test Picks
```bash
# Old way
python test_picks.py --week 3

# New way
confpickem test-picks --week 3
```

## Benefits of Unified CLI

1. **Single Entry Point**: One command instead of 8+ scripts
2. **Consistent Interface**: All commands use the same argument patterns
3. **Better Discovery**: `confpickem --help` shows all available commands
4. **Easier Maintenance**: Shared code, no duplication
5. **Cleaner Repository**: Root directory is much cleaner
6. **Proper Package**: Installed globally via pip, accessible from anywhere

## What's Next?

The old scripts in the root directory can now be:
- Moved to a `legacy/` directory
- Or deleted (they're in git history if needed)

The new CLI provides all the same functionality with a cleaner interface.

## All Available Commands

```bash
# Get general help
confpickem --help

# Get help for specific command
confpickem optimize --help
confpickem winprob --help
confpickem test-picks --help
confpickem analyze-skills --help
```