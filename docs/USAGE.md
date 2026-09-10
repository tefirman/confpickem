# Using confpickem

Start-to-finish for a real week, from the command line. For the Python API see
[`examples/yahoo_pickem_demo.ipynb`](../examples/yahoo_pickem_demo.ipynb); for
every flag see [`CLI_README.md`](../CLI_README.md); for how the optimizer works
see [`optimization-methodology.md`](optimization-methodology.md).

---

## One-time setup

```bash
pip install -e ".[dev]"     # from a clone, or: pip install confpickem
```

Two files go in the directory you run the command from (normally the repo root):

### 1. `cookies.txt` — your Yahoo session

The scraper reads Yahoo as *you*, so it needs your logged-in session cookies in
Mozilla / Netscape cookie-jar format. Easiest path:

1. Log in to `football.fantasysports.yahoo.com` in your browser.
2. Install a "cookies.txt" export extension (e.g. *Get cookies.txt LOCALLY* for
   Chrome/Firefox), open it on a Yahoo Fantasy page, and **Export** →
   `cookies.txt`.
3. Move that file to the repo root.

It expires after a few days — when scraping starts failing with parse errors or
"No games found", re-export it.

### 2. `current_player_skills.json` — how each opponent picks (optional but wanted)

This models each league member's tendencies (skill, how much they follow the
crowd, how they spread confidence). It's derived from cached copies of past Yahoo
weeks under `PickEmCache<year>/` — every `confpickem` run for a week leaves those
behind, so once you've run a season's worth (or have the directories from a
prior season):

```bash
confpickem-player-skills update --years 2024,2025
```

That analyzes each `PickEmCache<year>/` into `player_skills_<year>.json`, then
combines them into `current_player_skills.json`. Refresh it every few weeks as
more of the season is cached. Without the file the optimizer falls back to
average skills for everyone — still usable, just a rougher field model.

Default league is `15435`; pass `--league-id` for your own.

---

## Each week

### Start of the week — all games pending

```bash
confpickem --week 10 --mode beginning
```

You'll be prompted for:

- **your player name** — type enough of your Yahoo display name to match one row;
- **picks to lock** — `SF 16, KC 15` to force those, or just Enter to optimize
  everything.

Add `--live-odds` (with `--odds-api-key` or the `ODDS_API_KEY` env var) to use
live betting lines instead of Yahoo's implied ones.

### After games start — mid-week / mid-Sunday

```bash
confpickem --week 10 --mode midweek
```

The moment the first Sunday game kicks off, Yahoo locks **every** entry, so this
mode:

- locks each game that's **finished or already kicked off** to the pick and
  confidence you submitted, and
- optimizes only the games still open, over the confidence values you haven't
  spent.

It figures out which games are locked from kickoff times vs. now, so just run it
whenever — Sunday morning, at halftime, Monday afternoon.

---

## Reading the output

```
🎯 NFL PICK OPTIMIZATION - BEGINNING-OF-WEEK | ANALYTIC
...
analytical P(win): chalk 0.0054 -> optimized 0.0408
...
🏆 OPTIMIZATION RESULTS
📈 Win Probability:
   🎯 Optimized strategy: 5.1%
   🎲 Random picks: 1.6%
   💪 Advantage: +3.5 percentage points

OPTIMIZED PICKS:
16. Chi
15. Buf
...

📋 COPY-PASTE FORMAT:
   Chi 16, Buf 15, Jax 14, ...

GAME IMPORTANCE ANALYSIS
 1. NYJ@TB   -> NYJ (13 pts) +14.2%  [REMAINING]
 2. Dal@Chi  -> Chi (16 pts)  +7.6%
 ...
```

- **`analytical P(win): chalk X -> optimized Y`** — your modeled win probability
  playing every favorite by Vegas certainty (`chalk`) vs. the optimized slate.
  In a ~50-person pool a single-digit-percent Y that's several times `chalk` is
  the expected shape; the edge is controlled differentiation, not a high number.
- **Win Probability block** — the same idea cross-checked on the (noisier)
  Monte-Carlo simulator: optimized slate vs. a random entrant.
- **OPTIMIZED PICKS / COPY-PASTE FORMAT** — the slate. Paste the copy-paste line
  straight into Yahoo's pick grid.
- **GAME IMPORTANCE ANALYSIS** — how much each game's result swings *your* win
  probability. It's a win-probability swing, not a points swing: a favorite the
  whole crowd is on can top the list even at low confidence, because an upset
  there re-orders the entire field. `[REMAINING]` marks games not yet decided.

A `NFL_Week10_BeginningWeek_Analytic_<you>_<timestamp>.txt` report with all of
this is written to the current directory.

---

## Switching optimizers

The analytical optimizer is the default and runs in a few seconds. The old ones
are still there:

| flag | what it does |
|---|---|
| *(none)* | analytical Poisson-binomial `P(win)` search — **default** |
| `--greedy` | old sequential optimizer; `--greedy --fast` for a quick rough pass (beginning mode only) |
| `--hill-climb` | simulation hill-climb; its robustness report shows which picks are "locks" vs. volatile across the top solutions |

`--an-outcomes` / `--an-iterations` tune the analytical run if you ever need it
faster or more thorough; the defaults are fine for a normal week.

---

## When it breaks

| symptom | fix |
|---|---|
| `No games found` / HTML parse errors | `cookies.txt` is stale — re-export it |
| `current_player_skills.json not found` | optional; run `confpickem-player-skills update --years ...` (needs `PickEmCache<year>/` dirs) or proceed with average skills |
| Wrong league | pass `--league-id <yours>` |
| Odds API errors | check the key and your rate limit at the-odds-api.com, or drop `--live-odds` |
| Slow | you're probably on `--greedy` / `--hill-climb`; the default analytical run is the fast one |
