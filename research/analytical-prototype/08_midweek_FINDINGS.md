# PR A — analytical midweek path (issue #14)

> Runs against the code on the `feat/analytical-midweek` branch (PR A). On this
> research branch alone the imports resolve to the pre-PR-A engine and the
> script will not reproduce these numbers — check out PR A's branch to re-run.

`optimize_picks_analytic` now takes `player_data` and (for a live run) `as_of`,
and models three game states instead of two:

1. **Free** — pick open, outcome unknown. Optimized.
2. **Finished** (`Game.actual_outcome` set) — your real pick + confidence are
   locked; the outcome bit is forced in every draw; each opponent's real pick
   collapses to a scalar `completed_points` (points banked) and drops out of the
   Poisson-binomial.
3. **Frozen but live** (`Game.picks_locked`, or `as_of` past `kickoff_time`) —
   picks are locked pool-wide but the game hasn't ended. Your pick + confidence
   are locked; opponents' real picks pin their `p_home` to 1/0 but the game
   **stays in the convolution** because its outcome is still sampled.

The free games are optimized over exactly the confidence you haven't spent on
*any* frozen game. `build_opponent_types(max_types=…)` merges the rarest real
histories so `make_pwin` stays fast; `make_pwin` also precomputes each
opponent type's score PMF/CDF once (they never depend on the candidate slate),
which took a midweek call from ~240 s to ~5 s.

## Backtest (`08_midweek.py`)

5 real 2025 weeks × 3 scenarios, replaying each week's real field, scoring on the
full real outcome vector. Players who skipped a week entirely are dropped from
the field for both optimizers. `cutoff{k}` = first k games finished, rest free.
`kickoff2` = 2 finished but every game that has kicked off (Thursday + the
Sunday-1pm wave) is frozen-but-live; greedy has no such concept, so it can only
lock the 2 finished games — that handicap is part of the comparison.

| scenario | | wins | one-away | median behind | mean rank |
|---|---|---|---|---|---|
| **cutoff6** (10 free) | analytic | **1** | **2** | 35 | **35.2** |
| | greedy | 0 | 1 | 38 | 47.0 |
| | actual | 0 | — | 29 | 40.4 |
| **cutoff2** (14 free) | analytic | **1** | **2** | 28 | **40.2** |
| | greedy | 0 | 1 | 54 | 52.4 |
| | actual | 0 | — | 29 | 40.4 |
| **kickoff2** (~8 frozen-live, ~6 free) | analytic | **1** | 1 | 30 | **40.8** |
| | greedy | 0 | 1 | 54 | 52.4 |
| | actual | 0 | — | 29 | 40.4 |

Per week (analytic rank / greedy rank):
- wk2  56/56, 56/56, 56/56 — wipeout week for everyone (real player 47th)
- wk3  56/56, **37**/55, 41/55
- wk6  **1**/54, **1**/55, **1**/55
- wk9  12/17, 54/42, 54/42
- wk12 51/52, 53/54, 52/54

Same directional result as the #13 beginning-of-week backtest, and it **holds
with only 2 games decided** — the mid-Sunday case the user asked about. Fewer
completed games means a less-fragmented field (`max_types` rarely binds) and
more room to differentiate, so if anything the 2-game case is easier than the
6-game one; the mean-rank gap over greedy is actually larger there (40 vs 52 on
`cutoff2`, 35 vs 47 on `cutoff6`). The `kickoff2` frozen-but-live handling tracks
`cutoff2` closely — it doesn't degrade anything and is strictly more correct than
greedy's "re-optimize games I can't change."

Still higher-variance than greedy (weeks 2, 3 near last), but greedy had its own
blow-ups (five finishes at 52nd or worse across the matrix). 5 weeks is a
go/no-go signal for PR C, not a calibrated estimate.

Raw numbers: `08_midweek_results.csv`.
