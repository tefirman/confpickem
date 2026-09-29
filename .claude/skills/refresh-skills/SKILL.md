---
name: refresh-skills
description: Refresh the league's player skill models after a week is final -- caches every finished week of the season into PickEmCache<season>/, re-analyzes the season, recombines it with past seasons into current_player_skills.json (which /picks and confpickem load automatically), snapshots the result, and reports which players' knobs moved most. Use when the user asks to update/refresh player skills, or after a week wraps up.
argument-hint: "[--season YEAR] [--force-week N]"
---

# Refresh player skills

Run from the repo root. Use `.venv/bin/` executables if `.venv` exists. Pass any
arguments the user gave (`--season`, `--force-week N`) through to step 2.

## 1. Save the current skills for comparison

```bash
cp current_player_skills.json "$SCRATCH/skills_before.json"
```

(`$SCRATCH` = your scratchpad directory.) If this is the first refresh of the
season -- no `current_player_skills_<season>_*.json` exists yet -- also keep a
permanent copy as `current_player_skills_<season>_preseason.json`.

## 2. Cache the finished weeks

```bash
.venv/bin/python .claude/skills/refresh-skills/cache_weeks.py [--season YEAR] [--force-week N]
```

It fetches each week not already in `PickEmCache<season>/`, keeps it only if every
game has a result, and stops at the first unfinished week. JSON output:
`weeks_added`, `weeks_already_cached`, `latest_final_week`, `stopped_at`.

- `error` → stop. Almost always expired Yahoo cookies: tell the user to re-export
  `cookies.txt` (Mozilla cookie-jar format, repo root). The confidence-picks page
  is public and keeps working after cookies expire, so don't be fooled by
  partial success elsewhere.
- `weeks_added` empty → skills are already current through `latest_final_week`.
  Say so and stop, unless the user asked to re-run anyway.

## 3. Re-analyze and recombine

```bash
.venv/bin/confpickem-player-skills update --years <season>
```

This rewrites `player_skills_<season>.json` from the whole season's cache, then
combines every `player_skills_<year>.json` in the repo root (backups like
`_orig` are ignored) into `current_player_skills.json`, fuzzy-matching
historical names to the current roster. It needs valid cookies (it fetches the
current roster) and wipes `.cache/`, which is harmless.

## 4. Snapshot

```bash
cp current_player_skills.json current_player_skills_<season>_week<latest_final_week>.json
```

## 5. Report

```bash
.venv/bin/python .claude/skills/refresh-skills/diff_skills.py "$SCRATCH/skills_before.json" current_player_skills.json
```

Summarize briefly:
- Weeks added and the season's cached range (e.g. "2026 weeks 1–3").
- Which seasons were combined (from the `update` output).
- The biggest movers, and **Firman's Educated Guesses** specifically if it moved.
- Players added/dropped, and any unmatched roster names the `update` output
  flagged (those get distribution-sampled skills rather than their own history).
- The snapshot filename.
