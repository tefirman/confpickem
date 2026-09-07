# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
pip install -e ".[dev]"        # Install package + dev tools (pytest, black, flake8, mypy)

pytest                         # Run full suite (auto-runs with --cov per pyproject.toml addopts)
pytest tests/test_confidence_pickem_sim.py            # Single test file
pytest tests/test_cli.py::test_optimize_beginning_mode -x   # Single test, stop on first failure
pytest --no-cov                # Skip coverage when iterating quickly

black .                        # Format (line-length 100)
flake8 .                       # Lint (max-line-length 100, ignores E203/W503)
mypy src/confpickem            # Type check
```

CI (`.github/workflows/test.yml`) runs `pytest` across Python 3.8–3.13 on every push/PR to `main`. `conftest.py` prepends `src/` to `sys.path`, so tests import `confpickem` without an editable install.

The three console entry points (defined in `[project.scripts]`) all resolve to `main()` in `src/confpickem/cli/`:
- `confpickem` / `confpickem-optimize` → `cli/optimize.py`
- `confpickem-win-probability` → `cli/win_probability.py`
- `confpickem-player-skills` → `cli/player_skills.py`

See `CLI_README.md` for full flag documentation. (`README_OPTIMIZATION.md` is stale — it references a deleted `optimize_week1_picks.py`.)

## Architecture

Pipeline: **scrape Yahoo → convert to simulator format → Monte Carlo simulate → optimize confidence assignments**.

### Core modules (`src/confpickem/`)

- **`yahoo_pickem_scraper.py`** — `YahooPickEm(week, league_id, cookies_file)` scrapes three Yahoo pages and exposes them as attributes: `.games` (DataFrame: favorite/underdog, spread, `win_prob`, crowd `*_pick_pct`, crowd `*_confidence`, `home_favorite`, `kickoff_time`), `.players` (DataFrame: per-player `game_N_pick` / `game_N_confidence` columns), `.results` (list of dicts with `winner`). `PageCache` writes HTML + `_meta.json` into `.cache/` with a 1-day expiration.

- **`yahoo_pickem_integration.py`** — bridges scraper → simulator. `convert_yahoo_to_simulator_format(yahoo_data, ignore_results=False)` normalizes everything to **home-team perspective** (Yahoo data is favorite/underdog-relative) and produces the games DataFrame the simulator expects. `run_simulation(yahoo)` is the one-call path used by the Python API examples.

- **`confidence_pickem_sim.py`** — the engine. `Game` and `Player` are dataclasses; `Player` carries three 0–1 behavioral knobs: `skill_level`, `crowd_following`, `confidence_following`. `ConfidencePickEmSimulator`:
  - `add_games_from_dataframe()`, then `simulate_picks(fixed_picks, player_data)` — fully vectorized (NumPy matrices of shape `(num_sims, num_players, num_games)`). `fixed_picks` is `{player_name: {TEAM: confidence_int}}`; `player_data` (from `yahoo.players`) lets midweek runs exclude confidence values already spent on completed games.
  - `simulate_outcomes()` → boolean win matrix; `analyze_results(picks_df, outcomes)` → expected points, win %, game importance.
  - `optimize_picks(player_name, ...)` — **greedy**: assign highest confidence to the pick that most raises win probability, descending.
  - `optimize_picks_hill_climb(player_name, hc_iterations, hc_restarts, hc_top_n, ...)` — random-restart hill climbing; explores more of the space and reports per-team robustness across the top-N solutions. Writes progress to `hill_climb_checkpoint.txt` between restarts.

- **`live_odds_scraper.py`** — `LiveOddsScraper(odds_api_key)`. Pulls schedule/scores from ESPN's public API and betting lines from **The Odds API** (`ODDS_API_KEY` env var or `--odds-api-key`). `update_odds_with_live_data()` overwrites Yahoo's implied `win_prob` with live-derived probabilities; **falls back to Yahoo data per-game** when the API is unavailable (look for `live_odds_source == 'Yahoo_Fallback'`).

- **`analyze_player_skills.py`** — parses cached HTML in `PickEmCache<year>/` (per season) into raw per-player hit/miss stats → `player_skills_<year>.json`.
- **`apply_realistic_skills.py`** — combines one or more `player_skills_<year>.json`, converts raw stats to the three 0–1 knobs, fuzzy-matches historical names to the current roster, assigns distribution-sampled skills to unmatched players → `current_player_skills.json`, which `optimize.py` loads automatically if present (else default skills).

### Modes

- `beginning` — all games pending; may synthesize opponents (`--num-opponents`) and supports `--fast` (2000 sims, reduced confidence search).
- `midweek` — some games complete; requires real `yahoo.players` data to know spent confidence. `--fast` and `--num-opponents` are rejected here.

## Repo-specific conventions

- **Runtime data lives in the repo root and is gitignored**: `cookies.txt` (Mozilla cookie-jar format, Yahoo session, expires in days), `current_player_skills*.json`, `player_skills_*.json`, `PickEmCache*/`, `PreviousWeeks/`, `.cache/`, `hill_climb_checkpoint.txt`, and generated `NFL_Week*_*.txt` reports. Default league ID is `15435`.
- Packaging uses **hatchling** (not setuptools, despite a leftover `[tool.setuptools]` block). Bump `version` in `pyproject.toml` and `__version__` in `src/confpickem/__init__.py` together. A GitHub **release** triggers `publish.yml` → PyPI.
- `--live-odds` and `--no-cache` both wipe `.cache/` before loading so odds aren't served stale.
- Tests run offline — network-touching code (Yahoo, ESPN, Odds API) is mocked; keep it that way.
