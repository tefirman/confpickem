# How the pick optimizer works — and why the obvious approach doesn't

This is the story of building a confidence-pool optimizer that beats "pick the
favorites." The short version: the natural approach (search for the slate that
maximizes your simulated win probability) fails, and it fails for a subtle reason
worth understanding. The fix is to compute win probability *analytically* instead
of by simulation.

## The problem

An NFL Confidence Pick'em pool: each week you pick a winner for all ~16 games and
assign each pick a distinct confidence value 1..N. You score the confidence value
for every correct pick. In a typical Yahoo pool ~50 people play, and each week's
prize goes to the single highest score.

The naive strategy — pick every favorite, rank confidence by Vegas certainty —
lands you near the **median** of the pool essentially every week. It never wins,
because ~50 people are all playing near-chalk and the winner each week is whoever
high-confidenced two or three upsets that happened to hit. The crowd is already
close to optimal on the favorites. **The edge, if there is one, is
differentiation**: a slate that wins *when it wins* by diverging from the field in
the right high-confidence spots, accepting a 40th-place finish the rest of the
time.

So the optimization target is `P(you finish 1st)`, and the question is how to
compute it well enough to optimize against.

## What doesn't work: hill-climbing on simulated win probability

The package already had a Monte Carlo simulator: model each opponent's picks from
their historical skill, simulate thousands of seasons, count how often you come
first. The obvious move is to wrap a local search around it — start from chalk,
try swapping picks and confidence values, keep the changes that raise your
simulated `win_pct`.

We tried several versions of this: a plain greedy assignment, a random-restart
hill climb, and a "leverage"-weighted objective that explicitly rewarded
contrarian high-confidence picks. **None of them beat chalk on a backtest**, and
the leverage sweep produced this:

| 2025 Week 5, same field, same objective | | | | | |
|---|---|---|---|---|---|
| leverage weight λ | 0 | 1 | 2 | 3 | 5 |
| resulting finish (of 56) | 16th | 3rd | **55th** | **1st** | 9th |

λ=2 finishes dead last; λ=3 finishes first. That is not a strategy parameter
doing anything — it's **the search landing in a different random spot each run**,
because the objective it's climbing is noise.

### Why the objective is noise

`P(you finish 1st)` for a mid-pack entry is about **1–2%**. Estimated from 150
simulated seasons, its standard error is larger than the real differences between
candidate slates. The optimizer isn't finding the best slate; it's finding
whichever slate the simulation noise happened to favor on that run. More
simulations would help, but the cost is already the bottleneck, and you'd need
*far* more to resolve 0.1% differences in a 1% quantity.

This is the whole lesson: **a search is only as good as its objective, and a
noisy objective produces a search that optimizes noise.**

## What works: computing P(win) analytically

Your weekly score is

```
S = Σ_i  c_i · 1[pick i is correct]
```

a sum of independent-ish Bernoulli terms weighted by your confidence values
`c_i`. With integer weights, `S` has a **Poisson-binomial distribution** — its
full probability mass function is a short convolution, computed exactly with no
simulation. The same is true for each modeled opponent's score.

The catch is that `P(you beat opponent A)` and `P(you beat opponent B)` are *not*
independent — every entry is scored against the **same 16 game outcomes**. If the
favorites all hit, you beat nearly everyone at once; if there are upsets, you
lose to nearly everyone at once. Treating them as independent gives
`P(win) ≈ Π_j P(beat opponent j) ≈ 0.51^55 ≈ 10⁻¹⁶` — nonsense.

The fix is to condition on the outcome vector:

```
P(win) = Σ over game-outcome vectors o:  P(o) · P(your score is strictly highest | o)
```

We importance-sample `o` from the Vegas probabilities (a 16-dimensional draw —
cheap and low-variance, unlike sampling the full 55×16 pick matrix), and for each
`o` compute `P(win | o)` exactly from the opponents' conditional score
distributions. Modeled opponents collapse to a handful of distinct types, so this
runs in **~0.06 seconds per evaluation** with no Monte Carlo noise in the
ranking.

### Validation

Across four test weeks, the analytical `P(win)` ranks candidate slates the same
way the (slow, noisy) simulator does:

| week | Spearman ρ (analytical vs simulator) | analytical's top slate → simulator's rank |
|---|---|---|
| 2025 wk9 | 0.85 | 4 / 30 |
| 2024 wk3 | 0.67 | 1 / 24 |
| 2024 wk12 | 0.87 | 1 / 24 |
| 2025 wk12 | 0.95 | 1 / 24 |

The analytical number runs a consistent ~0.6× of the simulator's, but an
optimizer only needs the *ordering*, and ρ ≈ 0.85 says the ordering is right — at
250× the speed.

## Results

We ran a local search (random-restart hill climb) on the analytical `P(win)`
across 29 backtestable weeks (2024 weeks 2–17, 2025 weeks 1–17, minus a few with
unparseable cached data), replaying each week's real field and scoring on the
real outcomes. Baselines: the existing greedy optimizer, and the actual entry
submitted that week.

| variant | wins | top-5 | one game from winning* | mean finish | median points behind 1st |
|---|---|---|---|---|---|
| **analytical optimizer** | **2** | 3 | **12** | **34.2** | **25** |
| greedy optimizer | 1 | 3 | 6 | 42.9 | 43 |
| actually submitted | 0 | 1 | — | 43.7 | 32 |

\* weeks where flipping any single game's result would have moved you to 1st —
re-scoring the whole field under that counterfactual.

Split by season:

| | 2024 (16 wk) analytical / greedy | 2025 (13 wk) analytical / greedy |
|---|---|---|
| wins | 0 / 0 | **2 / 1** |
| mean finish | 38.5 / 46.2 | **28.8 / 38.9** |
| median points behind | 26 / 46 | **19 / 34** |

### Why we trust it

- **Not overfit to its own model.** The analytical-optimized slate's *simulator*
  win probability beats chalk's in **29 of 29 weeks** (median ~14×). The gains
  survive the independent, correlation-correct model.
- **Calibrated.** Mean modeled `P(win)` over the 29 weeks was 0.073, predicting
  ~2.1 wins. It won 2.
- **Stable.** Only 2 weeks in 29 did it finish more than 10 places worse than
  greedy. The leverage optimizer had 3–4 such blow-ups in 10 weeks.
- **2024 was a no-winner season** — every method and every real player stayed
  mid-pack all year. The analytical optimizer was still consistently *closer* to
  first. In 2025 it converted.

## Midweek

Once any game is locked, `optimize_picks_analytic(player_data=…, as_of=…)`
re-optimizes only what's still open. A pool locks *every* entry the moment the
first Sunday game kicks off, so a live mid-Sunday run typically has just a game
or two **finished** and the rest **frozen but not yet decided**. Three states:

- **Free** — pick open, outcome unknown. Optimized.
- **Finished** — your pick and confidence are locked to what you submitted; the
  outcome-vector draws are pinned to the real result; each opponent's real pick
  collapses to a scalar (the points they banked) and drops out of the
  Poisson-binomial, since it carries no variance.
- **Frozen but live** (kicked off per `as_of`, or `Game.picks_locked`) — your
  pick and confidence are locked, and opponents' real picks pin their side of
  that game, but it **stays in the convolution** because the outcome is still
  sampled.

The open games are optimized over the confidence you haven't spent on *any*
frozen game. Knowing opponents' real picks breaks the modal collapse, so the
rarest distinct histories past a cap (`max_opponent_types`, default 16) are
merged to keep evaluation fast.

A 5-week × 3-scenario 2025 spot check (finished / frozen-live / free split
varied per scenario) tracked the beginning-of-week result in every case —
analytical vs greedy mean finish: 35th vs 47th with 6 games decided, 40th vs
52nd with only 2 decided, 41st vs 52nd in the realistic mid-Sunday "2 finished,
rest frozen-live" case. Analytical won a week in all three and kept the same
higher-variance profile. Fewer decided games, if anything, helps — the field is
less fragmented and there's more room to differentiate.

## Game importance

`assess_game_importance` ranks each game by how much its result swings your
`P(win)`, and it reuses the same machinery. Hold your slate fixed, evaluate
`P(win | draw)` once over the sampled outcome vectors, then for each game
partition those draws by that game's outcome bit:

```
importance(i) = P(win | game i home win) − P(win | game i away win)
```

Both conditionals are a plain mean over a slice of the *same* draws — no forced
re-simulation, no Monte-Carlo noise, one pass for the whole slate (~0.07 s for
16 games). A game already decided, or a Vegas 0/1, has no live partition, so its
importance is 0. Note the metric is a **win-probability** swing, not a points
swing: a heavy favorite the whole crowd is on can carry large importance even at
low confidence, because an upset there re-orders the entire field at once.

## Fully-locked board

Once the first Sunday game kicks off the pool locks **every** entry, so nothing
is left to optimize — you just want the live standings. `standings_analytic`
skips the opponent model entirely: it scores each entrant's *real* slate (from
`yahoo.players`) against `n_outcomes` outcome draws — decided games pinned to
their results — and counts firsts. Because the field is the actual field, not a
model, the win probabilities are a true distribution (they sum to 1). It also
returns, per still-live game, the largest `|P(win | home) − P(win | away)|`
across all entrants — how much first place hinges on that game. Milliseconds for
a full league.

## Takeaways

1. In a large weekly pool, "pick better" is nearly a dead end — the crowd is
   already sharp on the favorites. The lever is controlled differentiation.
2. Optimizing `P(win)` by simulation fails not because the search is bad but
   because a ~1% probability estimated from a few thousand samples is too noisy
   to hill-climb.
3. A weekly confidence score is a weighted sum of Bernoullis, so its distribution
   — and `P(win)` against a modeled field — is available in closed form once you
   account for the outcome-vector correlation across entries. That objective is
   noise-free, ~250× faster to evaluate, and produces slates that win.

---

The chronological research log — the prototype scripts and every approach that
didn't pan out (a bug fix, a player-skill rework, a "leverage" objective, plain
hill climbing) — lives on the
[`explore/analytical-scoring`](https://github.com/tefirman/confpickem/tree/explore/analytical-scoring/research/analytical-prototype)
branch under `research/analytical-prototype/`.
