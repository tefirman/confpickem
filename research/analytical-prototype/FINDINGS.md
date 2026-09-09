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
