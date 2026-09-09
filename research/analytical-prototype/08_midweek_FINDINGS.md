# PR A — analytical midweek path (issue #14)

> Runs against the code on the `feat/analytical-midweek` branch (PR A). On this
> research branch alone the imports resolve to the pre-PR-A engine and the
> script will not reproduce these numbers — check out PR A's branch to re-run.

`optimize_picks_analytic` now takes `player_data` / `available_points` and, when
some `self.games` carry an `actual_outcome`, runs a real midweek optimization:

1. **Locked scored games.** Each completed game is pinned to *this* player's real
   pick + confidence (read from `yahoo.players`), on top of any `fixed_picks`.
   The free games are then optimized over exactly the **unspent** confidence
   values (`1..N` minus what completed games + fixed picks already used). A
   caller-supplied `available_points` is cross-checked against that set and a
   collision between a fixed pick and an already-spent value raises.
2. **Forced outcome draws.** `sample_outcomes(actual_outcomes=...)` pins the
   decided games' columns to their real results in every importance-sampling
   draw, so my banked points are deterministic and the pending-game correlation
   structure is preserved.
3. **Opponents' real completed picks in the field model.** For each modeled
   opponent, their actual pick/confidence on decided games is collapsed to a
   scalar `completed_points` (points already banked) and dropped from the
   Poisson-binomial; only the still-pending modal slate feeds the convolution.
   Without this the field would fan out to one type per distinct completed-game
   history; `build_opponent_types(max_types=…)` additionally merges the rarest
   histories so `make_pwin` stays fast (default cap 16 via
   `optimize_picks_analytic(max_opponent_types=16)`).

`make_pwin` was also refactored to precompute each opponent type's score
PMF/CDF once (they never depend on my slate) — a midweek `optimize_picks_analytic`
call dropped from ~240 s to ~5 s, and the beginning-of-week path got faster too.

## Backtest (`08_midweek.py`)

5 real 2025 weeks, treating games 1–6 as already played (their outcomes known,
every eligible player's picks on them locked), optimizing the remaining 8–10
games two ways and scoring on the full real outcome vector. Players who skipped
a week entirely are dropped from the field for both methods (the greedy engine's
`simulate_picks` can't handle their NaN picks).

| variant | wins | one-away | median pts behind 1st | mean rank |
|---|---|---|---|---|
| **analytic midweek** | **1** | **2** | 35 | **35.2** |
| greedy midweek | 0 | 0 | 36 | 51.6 |
| actually submitted | 0 | — | 29 | 40.4 |

Per week (rank / behind): wk2 an 56/54 vs gr 54/36; wk3 an 56/71 vs gr 56/82;
wk6 an **1/0** vs gr 51/29; wk9 an 12/6 vs gr 46/18; wk12 an 51/35 vs gr 51/36.

Same directional result as the beginning-of-week backtest in #13: the analytic
optimizer wins more, gets "one game away" more, and has a much better mean
finish, at the cost of higher variance (two last-place weeks here — but greedy
had four finishes at 51st or worse). 5 weeks is thin; treat as a go/no-go signal
for PR C, not a calibrated estimate.

Raw numbers: `08_midweek_results.csv`.
