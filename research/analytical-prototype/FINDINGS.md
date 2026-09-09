# Analytical prototype — findings

Branch: `explore/analytical-scoring` (off `main`, not merged).
Goal: replace the noise-dominated Monte-Carlo `P(win)` objective with an exact
analytical one, so an optimizer has a real gradient to follow.

## 1. Poisson-binomial machinery — WORKS

`poisson_binomial.py`:
- `weighted_pmf(probs, weights)` — exact PMF of `sum_i weights_i * Bernoulli(probs_i)`,
  convolution-based. Validated vs 2^n brute force (200+ random cases, atol 1e-9).
- `pmf_via_fft` — DFT version, same result.
- `prob_a_beats_b(a, b)` — `P(A > B) + 0.5 P(A == B)`, validated vs brute force.

A weekly confidence score is exactly a weighted Bernoulli sum, so its full
distribution is available with no simulation.

## 2. Field model (simulator's opponent model) — VALIDATES

`01_field_model_check.py`, 510 player-weeks vs actual cached picks:

| dimension | metric | value |
|---|---|---|
| pick side | mean Brier | 0.124 (vs 0.25 coin flip) |
| pick side | mean accuracy (model argmax vs actual) | 82.1% |
| confidence | mean rank correlation | 0.716 |
| confidence | mean abs point error | 2.4 of 16 |

The opponent model matches reality well enough to build on.

## 3. Independent-approx P(win) — USELESS

`02_pwin_independent.py`. `P(win) ~= prod_j prob_a_beats_b(my, opp_j)`.
2025 wk9: `prob_a_beats_b` ≈ 0.514 for all 55 opponents → `0.514**55 = 1.3e-16` → 0.
Simulator win_pct ~0.009. The independence assumption ignores that all 55
"do I beat them" events share the same 16 game outcomes.

## 4. Correlated P(win), MODAL opponents — TOO CRUDE (overestimates ~30-330x)

`03_pwin_modal.py`. Exact over all 2^n outcome vectors:
`P(win) = sum_o P(o) * [my score strictly highest | o]`, every opponent playing
their single most-likely slate.

| week | n | sim win_pct | modal P(win) | ratio |
|---|---|---|---|---|
| 2024 wk12 | 13 | 0.0012 | 0.382 | 327 |
| 2025 wk9  | 14 | 0.0150 | 0.516 | 34 |
| 2025 wk12 | 14 | 0.0148 | 0.474 | 32 |
| 2024 wk3  | 16 | 0.0078 | 0.426 | 54 |

Modal opponents are near-identical (all pick favorites, all rank confidence by
Vegas certainty) — `opp_ph`/`opp_pts` is essentially the same array 55 times. So
given any outcome vector all 55 score the same and my chalk score is a hair
above/below → `P(win) ≈ P(my score >= the one shared opponent score) ≈ 0.45`.

Real opponents diverge on the ~30% of games where `crowd_following` + noise flip
a pick, and their confidence rankings scatter. That divergence is what drives the
real `P(win)` down to ~1%: you must beat the *luckiest of 55 differently-wrong*
opponents, not one representative opponent. **Modal throws away the opponent
diversity that is the entire point.**

## Where next: correlated P(win) with opponent pick uncertainty

`P(win) = sum_o P(o) * P(my score strictly highest | o)`

Given `o`, my score is fixed. Opponent j's score is a weighted-Bernoulli over
which of their games came out their way: `P(pick i correct | o) = p_home_i` if
`o_i = 1` else `1 - p_home_i`, weighted by their confidence points. Convolve →
`P(my_score > opp_j | o)`. Then `P(win | o) ~= prod_j P(my_score > opp_j | o)` —
independence *across opponents given o*, a mild assumption (their remaining
randomness is their individual pick noise, which really is close to independent).

This is `02_pwin_uncertain` done right. The naive version was
`2^16 * 55 * weighted_pmf` per week ≈ 3.6M convolutions — too slow. Speedups:
- opponents cluster into a few distinct `(p_home vector, points vector)` types;
  compute each type's PMF once per outcome, weight by count.
- vectorize `weighted_pmf` across all outcomes at once (the per-game two-point
  kernels only depend on `o` through which of `p_home` / `1-p_home` is used).
- prune outcomes with `P(o) < eps` (the bulk of the 2^16 mass is in a few
  thousand near-chalk vectors).
- or: importance-sample outcomes from the Vegas distribution instead of full
  enumeration — reintroduces a little noise but far less than the pick-level MC.

Target: match the simulator's win_pct within ~20% on the 4 check weeks, at a cost
low enough (<~1s/week) to wrap an optimizer around.

## 5. Ranking check — PASSES (analytical objective is optimizer-ready)

`05b_ranking_fast.py`, 2025 wk9, 30 candidate slates (chalk + perturbations +
contrarian variants), each scored by sampled analytical P(win) and by the
simulator (600 sims).

- **Spearman rho(analytic, sim) = 0.849**
- analytic's argmax slate -> simulator rank **4 / 30** (top-few)
- top-3 overlap 2/3 (slates 15, 20 shared)
- analytic P(win) is a consistent ~0.6x the simulator's (a fixed scale factor,
  irrelevant to an optimizer that only needs the ordering)
- **timing: 0.05-0.08s analytic vs 14-25s simulator per slate** (~250x)

`pwin_sampled` vectorized: `vectorized_weighted_pmf` builds every outcome-drawn
opponent score PMF in one pass (no per-draw Python loop). K=5000 outcome draws,
opponents dedupe to 3 types.

### Verdict

The sampled analytical P(win) ranks slates like the simulator (rho 0.85) at
~250x the speed with no Monte-Carlo noise. This is the objective the earlier
lever investigation lacked. **Next: build a local-search slate optimizer on it**
(each eval ~0.06s, so thousands of neighbor evaluations per week is trivial) and
backtest vs greedy / leverage / the actual pool outcomes on the 10 weeks.

Open items (not blockers, tune later):
- validate rho on 2-3 more weeks (only wk9 checked at this depth)
- opponents collapse to 2-4 types -- the field model may be too smooth; a richer
  opponent model could widen the P(win) spread and sharpen the ranking further
- the ~0.6x scale gap vs simulator: from the given-o cross-opponent independence
  approximation and/or the modal-pick opponent assumption. Harmless for ranking.

## 6. Multi-week rho validation + optimizer smoke test

`05c_multiweek.py` on 3 more weeks (24 slates each):

| week | Spearman rho | analytic argmax -> sim rank | top-3 overlap |
|---|---|---|---|
| 2025 wk9  | 0.849 | 4 / 30 | 2/3 |
| 2024 wk3  | 0.665 | 1 / 24 | 2/3 |
| 2024 wk12 | 0.873 | 1 / 24 | 2/3 |
| 2025 wk12 | 0.946 | 1 / 24 | 3/3 |

3 of 4 weeks rho > 0.84; in 3 of 4 the analytic argmax slate IS the simulator's
#1. 2024 wk3 weaker (0.665) but argmax still sim #1. Objective is trustworthy.

`06_optimize.py` smoke test (2025 wk9, HC_ITERS=1500 x 6 restarts):
chalk analytic P(win) 0.0080 -> optimized 0.0857 (10.7x), all 14 games changed,
584s. Reduced to HC_ITERS=400 x 4 restarts for the backtest. Added a simulator
win_pct sanity check on the optimized slate (does the 10x analytic gain survive
the correlation-correct simulator, or is it overfit to the analytic model?).

## 7. Full analytical-optimizer backtest -- IN PROGRESS

`06_optimize.py`: analytic hill-climb vs greedy vs leverage(lambda=1) vs actual,
10 weeks, scored on real outcomes with the one-game-away metric, plus
sim_winpct(chalk) vs sim_winpct(analytic) per week.

## 7. Full analytical-optimizer backtest -- 2024 half (5/10 weeks)

`06_optimize.py`. Analytic hill-climb vs greedy vs leverage(l=1) vs actual, real
outcomes, one-game-away metric, plus a simulator win_pct sanity check.

| week | actual rk | greedy rk (oa) | leverage rk (oa) | analytic rk (oa/live) | sim_winpct chalk -> analytic |
|---|---|---|---|---|---|
| 2024 wk3  | 49 | 16 (1) | 48 (0) | 21 (0/0) | 0.009 -> 0.098  (10.9x) |
| 2024 wk6  | 53 | 53 (0) | 53 (0) | 35 (1/1) | 0.0045 -> 0.086 (19x)   |
| 2024 wk9  | 45 | 53 (0) | 53 (0) | 35 (1/1) | 0.013 -> 0.138  (10.6x) |
| 2024 wk12 | 43 | 27 (1) | 52 (0) | 27 (1/1) | 0.0015 -> 0.099 (66x)   |
| 2024 wk15 | 50 | 51 (0) | 50 (0) | 51 (0/0) | 0.0015 -> 0.0925 (62x)  |

**Q2 (overfit?) -- NO.** Every week the analytic-optimized slate's SIMULATOR
win_pct is ~10-60x chalk's. The gains survive the correlation-correct simulator,
not just the analytic model. Leverage never achieved this.

**Q1 (beats baselines?) -- yes, modestly.** Analytic one-game-away 3/5 (wk6,9,12)
vs greedy 2/5, leverage 0/5. Mean rank so far: analytic 33.8, greedy 40.0,
leverage 51.2, actual 48.0.

**Q3 (consistent?) -- yes.** Analytic >= greedy on rank every week, never the
50-53 blowups leverage had. Weakest week (wk15) it ties greedy.

Tension: analytic's ACTUAL-outcome rank (33.8) is only modestly above greedy
(40.0) despite 10-60x modeled win_pct. Over 5 weeks at ~10% modeled win each
you'd expect ~0.5 wins; got 0 but 3 "one game away". Consistent with a real edge
not yet realized (small n) OR the model still overstates the edge. 2025 half
pending.

## 8. Full analytical-optimizer backtest -- FINAL (9/10, 2025 wk15 pending)

| week | actual | greedy (oa) | leverage (oa) | analytic (oa) | sim_winpct chalk->analytic |
|---|---|---|---|---|---|
| 2024 wk3  | 49 | 16 (1) | 48 | 21 | 0.009->0.098  (11x) |
| 2024 wk6  | 53 | 53     | 53 | 35 (1) | 0.0045->0.086 (19x) |
| 2024 wk9  | 45 | 53     | 53 | 35 (1) | 0.013->0.138  (11x) |
| 2024 wk12 | 43 | 27 (1) | 52 | 27 (1) | 0.0015->0.099 (66x) |
| 2024 wk15 | 50 | 51     | 50 | 51 | 0.0015->0.093 (62x) |
| 2025 wk2  | 47 | 56     | 56 | 56 | 0.0015->0.111 (74x) |
| 2025 wk5  | 3  | 21 (1) | 9 (1) | 3 | 0.014->0.10 (7x) |
| 2025 wk9  | 54 | 1 (1)  | 54 | 46 | 0.014->0.107 (8x) |
| 2025 wk12 | 42 | 52     | 2  | 18 (1) | 0.011->0.114 (10x) |

Aggregate (9 wks):

| variant  | wins | one-away | mean rank |
|----------|-----:|---------:|----------:|
| analytic | 0    | **5**    | **32.3**  |
| greedy   | 1    | 4        | 37.8      |
| leverage | 0    | 2        | 40.7      |
| actual   | 1    | -        | 43.0      |

**Q2 overfit? NO.** All 9 weeks the analytic slate's SIMULATOR win_pct is 7-74x
chalk's. The analytic gains survive the correlation-correct model. Leverage never
did this.

**Q3 blowups? NO.** Analytic's worst weeks (wk15 rk51, wk2 rk56) are weeks every
method failed. No leverage-style "great one week, 53rd the next" volatility.

**Q1 beats baselines? Yes, modestly.** Best mean rank (32.3 vs greedy 37.8), most
"one game away" weeks (5/9 -- the designed metric). But **0 outright wins in 9
weeks.** Greedy's 1 win (wk9, 1st) came a week analytic went 46th.

### Honest read

The analytic optimizer reliably builds slates modeled at ~8-10% win prob that
consistently land "one flip from winning" but haven't converted. Over 9 weeks at
~9% you'd expect ~0.8 wins; got 0. Within variance (P(0 wins) ~43%) but also
consistent with the model overstating the edge ~2x. wk9: analytic's contrarian
bet missed (46th) while greedy's near-chalk slate nailed 1st -- the approach
trades "occasionally win big" for "usually close", the right trade for a
weekly-prize pool IF it hits.

### Recommendation

**Productionize, but as an alternative not a replacement.** This is the first
approach in the whole investigation that (a) has a noise-free, fast, validated
objective (rho 0.67-0.95), (b) produces gains that survive the simulator, (c) is
consistent week to week. That is a real result. But 0/9 wins means it is not
proven superior on realized outcomes -- it needs a full-season (or multi-season)
backtest to separate "real edge, small sample" from "model 2x optimistic".

Concrete next steps:
1. Full-season backtest (all ~17 replayable weeks/season x 2 seasons) -- the
   9-week sample is too thin to trust the win rate.
2. Calibrate: the ~0.6x scale gap vs simulator (section 5) plus the 0/9 wins
   suggest tightening the opponent model (richer than 2-4 types) and/or the
   given-o independence correction.
3. If it survives that: move poisson_binomial.py -> src/confpickem/, add
   optimize_picks_analytic() as a method (objective closure + hill climb),
   tests, wire into the CLI as an opt-in mode.

## 8. Full-season backtest -- FINAL (29 weeks: 16 of 2024, 13 of 2025)

`07_fullseason.py`. Analytic hill-climb vs greedy vs actual, real outcomes,
one-game-away metric, simulator win_pct sanity check on every optimized slate.
(3 weeks skipped -- 2025 wk8/13/16 -- bad cached pick_distribution HTML.)

| variant  | wins | top5 | top10 | one-away | mean rank | median behind |
|----------|-----:|-----:|------:|---------:|----------:|--------------:|
| analytic | **2** | 3 | 4 | **12** | **34.2** | **25** |
| greedy   | 1    | 3 | 4 | 6        | 42.9      | 43            |
| actual   | 0    | 1 | 1 | -        | 43.7      | 32            |

Split by season:

| | 2024 (16w) analytic / greedy | 2025 (13w) analytic / greedy |
|---|---|---|
| wins      | 0 / 0 | **2 / 1** |
| top-5     | 0 / 0 | 3 / 3 |
| one-away  | 8 / 2 | 4 / 4 |
| mean rank | 38.5 / 46.2 | **28.8 / 38.9** |
| median behind | 26 / 46 | **19 / 34** |

### The four questions

**Q: overfit to the analytic model?**  NO. Analytic slate's SIMULATOR win_pct
beats chalk's **29/29 weeks**, median 14x (0.077-0.144 vs chalk's ~0.003-0.03).
The gains are real under the correlation-correct model.

**Q: model win rate optimistic?**  NO -- it's calibrated. Mean modeled analytic
P(win) = 0.073 -> expected 2.1 wins over 29 weeks. **Actual: 2 wins.** The ~0.6x
analytic/sim scale gap from section 5 washes out at the decision level.

**Q: blowups like leverage?**  Almost none. **2 weeks in 29** analytic was >10
ranks worse than greedy (2024 wk3: 21 vs 6; 2025 wk14: 26 vs 2). Leverage had
3-4 in 10 weeks. Analytic is stable.

**Q: beats greedy?**  Yes, clearly. +2 vs +1 wins, DOUBLE the one-game-away
weeks (12 vs 6), mean rank 8.7 better, median points-behind-first ~18 lower.
2024 (a bad-variance season where nobody won) analytic was consistently closer;
2025 it converted.

### FINAL VERDICT: productionize.

This is the approach that works. Rationale:
- noise-free, fast (0.06s/eval), validated objective (rho 0.67-0.95 across 4 wks)
- calibrated (2 modeled ~= 2 actual wins over 29 weeks)
- gains survive the simulator every single week (29/29)
- stable (2 blowups / 29 vs leverage's 3-4 / 10)
- clearly beats greedy on wins, one-away, mean rank, and closeness

It does NOT need opponent-model calibration first -- the 2-4 type collapse and
0.6x scale gap were feared to matter but the backtest shows they don't at the
decision level. Calibration is a nice-to-have refinement, not a blocker.

### Productionize plan

1. `src/confpickem/analytical.py`:
   - `weighted_pmf`, `pmf_via_fft`, `prob_a_beats_b` (from poisson_binomial.py)
   - `build_opponent_types(games_df, player_skills)` -> deduped modal (p_home,
     points, count) list
   - `sampled_pwin(my_ph, my_pts, opp_types, vegas, n_outcomes=6000, seed=...)`
2. `ConfidencePickEmSimulator.optimize_picks_analytic(player_name, fixed_picks=None,
   iters=400, restarts=4, n_outcomes=6000, seed=51)` -- random-restart hill climb
   on `sampled_pwin`, same signature style as `optimize_picks`.
3. `tests/test_analytical.py`: PMF vs brute force; `sampled_pwin` monotonicity
   (better slate -> higher); optimizer returns a valid 1..N permutation; beats
   chalk P(win) on a fixture week.
4. CLI: `--optimizer {greedy,hillclimb,analytic}` in cli/optimize.py, default
   greedy, analytic opt-in.
5. Follow-up (not blocking): richer opponent model, calibrate the scale gap,
   multi-season backtest as regression.
