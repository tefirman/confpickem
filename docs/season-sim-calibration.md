# Season simulator: calibration against past seasons

`scripts/calibrate_season_sim.py` backtests `SeasonSimulator` on the two finished
seasons in `PickEmCache<year>/`. For each season and each cutoff week k in
{4, 8, 12} it forecasts the rest of the season from the standings after week k,
using only what was knowable then:

- skills from the **other** seasons' raw stats (leave-one-season-out); newcomers get
  the league-mean skills;
- the ability-spread parameter estimated from the other seasons only;
- future slates sampled from every cached week except that season's own weeks after k.

It then compares against what actually happened. Usable data is **2024 weeks 2–17**
and **2025 weeks 1–17** — some cached pages were saved before their games were played
(2024 wk1/wk18, 2025 wk18), so each "season" is 16–17 weeks. The 2026 season is
unfinished and only feeds the slate pool and the ability estimate.

## What the first version got wrong

The first version reused the weekly simulator's pick model unchanged. Backtesting it
showed three problems, each visible in the real data:

| Problem | Evidence | Fix |
|---|---|---|
| Week-to-week noise too small | SD of scores across players within a week: real 10.4–11.3, simulated 7.8–8.1 | `SEASON_NOISE_SCALE = 2.8` (widens pick and confidence noise; matches 10.1 before the ability term) |
| No persistent skill differences | Real split-half reliability of relative score 0.14–0.40, true player SD ≈ 2.5–4 pts/week; the skill knobs produced SD ≈ 0.4 | Per-player ability edge, estimated from the season so far and shrunk (`estimate_ability`, `ABILITY_SD = 3.0`) |
| Missed weeks treated as rare and random | Players with no misses skip 1.7% of later weeks; after missing last week, 48%; after missing two in a row, 82% | State-based `participation_rates` + `infer_active` (rates fit on ~1,500 player-cutoffs for the reliable state but only ~20 each for the miss states — rough) |

The weekly optimizer's opponent model is untouched; these changes live only in
`season_sim.py`.

## Results (6,000 sims per forecast; 327 active entrants pooled over 6 forecasts)

**Final points relative to the field, z-scored** (should be mean 0, SD 1):

| Group | n | mean z | SD z | \|z\|>2 (nominal 4.6%) |
|---|---|---|---|---|
| All active entrants | 327 | +0.00 | 1.81 | 7.3% |
| Played every remaining week | 300 | +0.33 | 0.99 | 4.0% |
| Missed 1+ remaining weeks | 27 | −3.62 | 3.89 | 44% |

For players who kept playing, the spread of outcomes is right. The tails come from
missed weeks (~80 points each) that nobody can foresee individually; the model
prices them in only through the participation probabilities. Predicted vs actual
missed player-weeks over the six forecasts: 89 vs 88 in total, but individual
forecasts range from 12 vs 20 to 23 vs 14.

**Weekly wins after the cutoff:** 51.0 predicted vs 50.0 actual overall. By predicted
quartile: 4.9 / 9.5, 9.5 / 8.5, 14.6 / 12.9, 22.0 / 19.1 (predicted / actual) — the
bottom quartile wins about twice as often as forecast (9.5 events vs 4.9 expected;
borderline, Poisson p ≈ 0.04), a hint that low-ranked players' weekly scores have
fatter tails than modeled.

**P(top 4 in the season)** — reliability:

| Predicted bin | n | predicted | actual |
|---|---|---|---|
| ≤ 2% | 184 | 0.5% | 1.1% |
| 2–5% | 38 | 3.3% | 0.0% |
| 5–10% | 38 | 7.4% | 10.5% |
| 10–20% | 34 | 14.9% | 11.8% |
| 20–35% | 25 | 26.8% | 28.0% |
| > 35% | 13 | 56.2% | 76.9% |

Brier score 0.0495, against 0.0747 for a flat rate and 0.0723 for "whoever is in the
top 4 now". The top bin looks under-confident (10 of 13 vs 7.3 expected), and P(1st)
shows the same lean (the top bin: 6 players predicted 24.5%, actual 33%), but both
are 2–3 hits from nothing.

**Expected dollars** (weekly + season), by predicted quintile (predicted / realised):
$27 / $56, $56 / $70, $92 / $75, $161 / $122, $408 / $412. Correlation 0.49; overall
mean $149 predicted vs $148 realised.

**Spread → win-probability map** (all 601 cached games with a result): the
favorite wins 67.9% vs 65.1% implied. By spread bucket the gaps run both ways
(+5 pts at 2.5–4.5, +7 at 4.5–7, −6 at 7–10), none beyond ~2 standard errors.

## How much to trust it

- **It is two seasons.** Entrants within a season share the same game results, so the
  327 forecasts are far fewer than 327 independent observations; the top-bin and
  quartile discrepancies above are not statistically distinguishable from luck.
  What the backtest supports is: scoring dispersion is calibrated, the totals add up
  (weekly wins, dollars), and the probabilities beat naive baselines. It does not
  support quoting a P(1st) to better than a few points.
- **Season-level shocks are unforecastable.** 2024's remaining weeks averaged ~10
  points per player-week above the forecast (a chalky year); this moves everyone
  together and washes out of rankings but not out of absolute points.
- **Constants are fit on the same seasons they are tested on, in part.** The noise
  scale (2.8) and participation rates were chosen using all three seasons. Only the
  ability SD and the skills are leave-one-season-out. Expect a slightly rosier
  backtest than a true out-of-sample one.
- **Not modeled:** ties in games, roster changes (new entrants after the cutoff were
  ignored), the in-progress week's partial results (the current week is simulated in
  full), and any strategy adaptation by players late in the season.

## Reproducing

```bash
python scripts/calibrate_season_sim.py --sims 6000 [--csv forecasts.csv]
```
