---
name: picks
description: Run this week's confidence pick'em optimizer scenarios for "Firman's Educated Guesses" -- detects the week and mode (beginning / midweek / locked), runs a neutral optimization plus all-in scenarios on each side of the week's opening game (usually Thursday Night Football), scores them head-to-head with --compare-slate, and summarizes which slate to submit. Use when the user asks to run/refresh their picks, the weekly scenarios, or /picks.
argument-hint: "[week] [extra scenario, e.g. \"SEA 16\" or \"KC 16, BUF 15\"]..."
---

# Weekly pick scenarios

Run from the repo root. Use `.venv/bin/confpickem` if it exists, else `confpickem`.

## 1. Figure out the week

```bash
.venv/bin/python .claude/skills/picks/week_status.py [WEEK]
```

Pass WEEK if the user gave one; otherwise the script infers it from the latest
`NFL_Week*` report and rolls forward once that week has a result for every game.
It prints JSON with `week`, `max_confidence` (= number of games), `players_scraped`,
`recommended_mode`, and the `opener` game (earliest kickoff, with `started`).

Stop and tell the user to re-export `cookies.txt` (Mozilla cookie-jar format,
repo root) if the output has an `error` or `players_scraped` is 0 -- the Yahoo
session has expired. Don't try to work around it.

## 2. Pick the scenarios

| `recommended_mode` | Scenarios |
|---|---|
| `beginning` (opener not started) | `<AWAY>AllIn`, `<HOME>AllIn`, then `Neutral` |
| `midweek` | `Neutral` only |
| `locked` | one locked-mode run, no scenarios (see step 4) |

- **All-in** = that opener team pinned at `max_confidence` (e.g. `Atl 16`), everything
  else optimized. The opener is the one pick that locks before any new information
  arrives, so it's the decision worth stress-testing; later games get re-optimized
  in midweek runs anyway.
- **Extra scenarios**: any fixed-pick strings the user passed as arguments (e.g.
  `SEA 16`, `KC 16, BUF 15`, or a bare `DAL` to pin a team but let the optimizer
  choose its confidence) become additional scenarios in any non-locked mode. Label
  each one from its picks, CamelCase with no spaces or colons (`Sea16`, `Kc16Buf15`).
- Use Yahoo's team abbreviations exactly as `week_status.py` prints them.
- Run `Neutral` **last** (step 3 feeds it the others' slates).

## 3. Run each scenario

Live odds: add `--live-odds` if `ODDS_API_KEY` is set in the environment
(`[ -n "$ODDS_API_KEY" ]`). If it isn't, run without it and say so in the
summary -- never ask the user to paste the key into chat.

The optimizer prompts on stdin for the player, then fixed picks, so pipe both:

```bash
printf '%s\n%s\n' "Educated Guesses" "<FIXED PICKS or empty>" | \
  .venv/bin/confpickem --week <W> --mode <MODE> --html [--live-odds] [--compare-slate ...]
```

After **each** run:
1. Grab the report path from the `💾 Results saved: <file>.txt` line.
2. Rename both `<file>.txt` and `<file>.html` to `<file>_<Label>.txt/.html`.
   Do this before the next run -- runs in the same minute share a timestamp and
   would overwrite each other.
3. Record the `COPY-PASTE FORMAT` slate and `Win probability` from the report.

For the final `Neutral` run, pass one `--compare-slate "<Label>:<copy-paste slate>"`
per earlier scenario. The optimizer scores its own picks (labelled `optimized`)
and every supplied slate on the same seeded outcome draws, so the resulting
`SLATE COMPARISON` table (win % and win std) is apples-to-apples -- unlike the
per-run win probabilities, which come from separately optimized runs.

If a run errors, show the error output and stop rather than continuing with a
partial set.

## 4. Locked mode

Everything's locked -- no scenarios, no prompts:

```bash
.venv/bin/confpickem --week <W> --mode locked --html --player "Educated Guesses"
```

Summarize: current rank and points, win probability, who's ahead, and the top
remaining games by swing on first place.

## 5. Summarize

Keep it short:
- Week, mode, whether live odds were used, and the opener matchup.
- The `SLATE COMPARISON` table (`optimized` = Neutral).
- Recommendation: the highest win % slate. If the top two are within ~1 pp, say
  it's effectively a tie, and point to win std as the tiebreaker (lower = steadier,
  higher = more boom-or-bust -- which is better depends on whether the user
  is protecting a lead or chasing).
- The recommended slate in copy-paste format, ready to enter on Yahoo.
- Paths to the renamed `.html` reports.
