# Optimizer investigation — why `optimize_picks` finishes mid-pack, and why the fixes didn't help

**Status: negative result. This branch is not merged.**

Late-2025 season it became clear that `optimize_picks` (and `optimize_picks_hill_climb`)
produce entries that land near the median of a ~55-person Yahoo confidence pool and
never sustain a top finish. This is the record of four attempts to fix that, the
backtests that evaluated them, and the conclusion about why none worked.

The two prior PRs it depends on **were** merged and are keepers:

- **#10 — invalid confidence-vector bug fix.** `simulate_picks` re-packed the non-fixed
  games after applying `fixed_picks` by matching on *confidence value*, so a non-fixed
  game whose pre-filled confidence collided with a fixed point value was dropped from
  the re-pack — leaving a duplicated value and a gap. An entry with one fixed pick got a
  **silently inflated `win_pct`** (illegal scorecards double-count). Fixed to track pinned
  games by index. Necessary — `win_pct` was not a real number before this.
- **#11 — player-skill derivation rework.** The old skill knobs barely varied
  (`skill_level` std ~0.02, `crowd_following` hardcoded to 0.5). Reworked to derive all
  three from the cached results + pick-distribution crowd data: `skill_level` std → 0.15,
  `crowd_following` std → 0.10, actually computed. Necessary for any realistic field model.

## The four levers

| # | Lever | What it changed | Backtest result |
|---|---|---|---|
| 1 | Bug fix (#10) | honest `win_pct` | mean optimizer rank 28.2 → **36.5** (slightly worse) |
| 2 | Real skills (#11) | varied field instead of 50 clones | mean rank → **41.7** (worse again) |
| 3 | Leverage term | `win_pct + λ·leverage`, rewards confidence where the sharp line beats the crowd's pick % | wash vs. greedy; see below |
| 4 | Hill-climb search | random-restart local search instead of greedy-by-certainty (this is `λ=0` in the sweep) | wash vs. greedy |

Backtest = 10 historical weeks (2024 wk 3/6/9/12/15, 2025 wk 2/5/9/12/15) replayed offline
from `PickEmCache2024/2025`, optimizing the `Firman's Educated Guesses` entry against the
real field's actual picks, scored on the **real outcomes**.

## The λ sweep (`lambda_sweep.py` → `lambda_sweep_results.csv`)

10 weeks × {greedy, leverage λ ∈ 0, 1, 2, 3, 5}. λ=0 is a pure win-probability hill climb
(no leverage term) — the control that isolates "search method" from "leverage".

Per-week rank:

| Week | greedy | λ0 | λ1 | λ2 | λ3 | λ5 |
|------|-------:|---:|---:|---:|---:|---:|
| 2024 wk3  | 45 | **4**  | 20 | **1**  | 42 | 48 |
| 2024 wk6  | 53 | 53 | 49 | 53 | 53 | 53 |
| 2024 wk9  | 53 | 53 | 53 | 53 | 53 | 53 |
| 2024 wk12 | **5**  | 50 | 21 | 51 | 49 | 52 |
| 2024 wk15 | 49 | 51 | 50 | 51 | 51 | 52 |
| 2025 wk2  | 56 | 56 | 56 | 56 | 56 | 56 |
| 2025 wk5  | 2  | 16 | 3  | **55** | **1**  | 9  |
| 2025 wk9  | 13 | 17 | 55 | 55 | 50 | **7**  |
| 2025 wk12 | 54 | **2**  | **2**  | 52 | 50 | 51 |
| 2025 wk15 | 19 | 7  | 18 | **2**  | 49 | 54 |

Aggregate (n=10):

| variant | wins | one-away | mean live-flips | median pts behind 1st | mean rank |
|---------|-----:|---------:|----------------:|----------------------:|---------:|
| greedy      | 0 | 3 | 0.6 | 32.5 | 34.9 |
| leverage λ0 | 0 | 3 | 0.6 | 27.0 | 30.9 |
| leverage λ1 | 0 | 4 | 0.6 | 23.5 | 32.7 |
| leverage λ2 | 1 | 2 | 0.7 | 42.0 | 42.9 |
| leverage λ3 | 1 | 1 | 0.6 | 37.0 | 45.4 |
| leverage λ5 | 0 | 2 | 0.6 | 50.0 | 43.5 |

## Conclusion

**The objective is noise-dominated, so the search optimizes noise, not signal.**

- **6 of 10 weeks were competitive; each had a different "winning" configuration**
  (λ2, greedy, greedy, λ3, λ5, λ0). No method is coherently better.
- **λ ordering is chaotic.** 2025 wk5: rank `2 / 16 / 3 / 55 / 1 / 9` across λ = `0/1/2/3/5`
  plus greedy. Same week, same field, same objective — λ2 finishes last, λ3 finishes first.
  That is the hill climb landing in different local optima of a flat, noisy surface, made
  worse by `optimize_picks`' per-candidate `np.random.seed(51 + hash(f"{team}_{pts}") % 10000)`
  which is process-dependent (PYTHONHASHSEED) and non-reproducible.
- **`win_pct` for a mid-pack entry is ~1–2%** and at `num_sims=150` its standard error
  swamps the differences between candidate slates.
- **3–4 of the 10 weeks are unwinnable post-hoc** — the pool winner had a near-perfect
  slate (wk6 winner went 105/105) that no optimizer could match after the fact.

The `one_game_away.py` metric (flip any single game, re-score the whole field, did we
finish 1st?) was built to see past all-or-nothing rank. On one informal run it favoured
leverage (4 vs 3 "one game away" weeks, more live-flips); the controlled λ sweep did not
reproduce that separation. See `one_game_away_results.csv` (informal) vs. the λ sweep.

## Why this is a framing problem, not a dead end

Everything above optimizes **P(finish 1st) against a simulated field, in a single realized
world, this week, in isolation.** Better-posed alternatives:

1. **Theoretical objective.** Score *policies* (not slates) by P(win) over many resampled
   **seasons**, resampling both game outcomes and the field's picks, with enough sims that
   the estimate is stable. A good slate wins across possible worlds, not the one that
   happened.
2. **Analytical instead of Monte Carlo.** Your score is `Σ conf_i · 1[pick_i correct]` — a
   weighted sum of Bernoullis, i.e. a Poisson-binomial with a closed-form distribution.
   Each modelled opponent's total is also Poisson-binomial. `P(beat the field)` is a
   convolution with a shared-outcome correlation correction. This is **differentiable and
   noise-free** — "move 3 points from game A to B, ΔP(win) = x" with no simulation.
3. **Portfolio / correlation framing.** The crowd is already near-optimal on chalk; the
   edge is *differentiation* — a slate that wins when it wins by being contrarian in the
   right high-confidence spots, accepting a 40th-place median. (2) makes this measurable.

Recommended: build (2), model the field from the #11 skill data, and use it to answer
"given a realistic field, what degree and placement of contrarianism maximises P(win)?"

## Files

- `lambda_sweep.py` / `lambda_sweep_results.csv` — the controlled sweep above.
- `one_game_away.py` / `one_game_away_results.csv` — the "one game away" counterfactual metric
  (informal run; greedy vs. a single λ=1 leverage config).
- `backtest_leverage_results.csv` — greedy vs. leverage λ=1, 10 weeks, from the notebook run.
- `backtest_skills_results.csv` — greedy only, uniform vs. season-matched skill field.

## Code on this branch (not merged)

`src/confpickem/confidence_pickem_sim.py` adds `optimize_picks_leverage()` and
`_leverage_score()`. Left intact in case the objective is ever reframed — the
leverage-weighted hill climb itself is fine, only its results are negative.

Known issue if revisited: `_get_neighbor_solution` (used by both hill-climb methods)
always mutates the *first* modifiable game rather than a random one — limits exploration.
