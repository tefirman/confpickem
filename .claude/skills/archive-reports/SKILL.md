---
name: archive-reports
description: Tidy the repo root by moving finished weeks' NFL_Week* .txt/.html reports into PreviousWeeks/, keeping only the week currently being worked on. Use when the user asks to clean up, archive, or tidy old reports.
argument-hint: "[--keep-week N]"
---

# Archive old reports

Run from the repo root. Everything involved is gitignored local data, so
there's nothing to commit.

1. Preview:
   ```bash
   .venv/bin/python .claude/skills/archive-reports/archive_reports.py --dry-run [--keep-week N]
   ```
   By default it keeps the most recent (season, week) in the root, reading the
   season from each report's timestamp. Pass `--keep-week N` if the user names
   a different week to keep.

2. If the preview looks right -- it only moves weeks older than the one being
   kept -- run it again without `--dry-run`. Moves are plain renames into
   `PreviousWeeks/` and never overwrite an existing file there.

3. Report which weeks were archived and how many files, plus any it skipped
   because a same-named file was already in `PreviousWeeks/`. Don't
   delete or rename skipped files; mention them so the user can decide.
