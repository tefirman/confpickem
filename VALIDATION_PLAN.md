# Validation & Backtesting Framework

## Overview

This document outlines the systematic validation approach for the confpickem optimizer. The goal is to build confidence in the tool through rigorous testing against historical data.

## Problem Statement

After two seasons of using the optimizer, key questions remain:
- Are the optimizer's recommendations actually optimal?
- Which optimization algorithm (greedy vs. hill climbing) performs better?
- Are win probability estimates accurate and well-calibrated?
- Is the underlying simulation model sound?

Without objective proof, it's difficult to trust the optimizer's recommendations, especially during critical mid-week re-optimization after Thursday games.

## Available Historical Data

- **2+ seasons** of weekly pick data (50+ players per week)
- Yahoo HTML files containing:
  - All player picks for each week
  - Game results
  - Pick distributions
- Vegas spreads for each week

## Validation Phases

### Phase 1: Validate Input Data (Yahoo Spreads)

**Goal:** Verify that Yahoo's implied win probabilities are well-calibrated against actual NFL outcomes.

**Method:**
1. Extract Yahoo spreads from historical data
2. Convert spreads to implied win probabilities
3. Load actual NFL game results from Yahoo HTML data
4. Calculate calibration: When Yahoo said a team was 60% favorites, did they win ~60% of the time?

**Success Criteria:**
- Calibration error < 5% across probability buckets
- No systematic bias toward favorites/underdogs

**Output:**
- Terminal report showing calibration by probability bucket
- CSV file: `yahoo_spread_calibration.csv`
- Visualization: Calibration curve (predicted vs. actual)

**Why This Matters:**
If Yahoo spreads are poorly calibrated, all downstream analysis is suspect. This must pass before continuing.

---

### Phase 2: Validate Win Probability Calculation (Real Picks vs Real Results)

**Goal:** Verify that our win probability calculations are accurate for actual player picks.

**Method:**
1. For each historical week, load all 50 players' actual picks
2. For each player, calculate their predicted win probability using our simulation model
3. Compare predictions against actual outcomes (who won that week)
4. Bucket by predicted probability (0-10%, 10-20%, etc.) and check calibration

**Success Criteria:**
- Players with 15% predicted win probability should win ~15% of the time
- Calibration error < 5% across buckets
- Consistent across both seasons

**Output:**
- Terminal report: Calibration by probability bucket
- CSV file: `win_probability_calibration.csv` (one row per player per week)
- Visualization: Actual win rate vs. predicted probability

**Why This Matters:**
This validates that our core win probability math is correct when applied to real picks.

---

### Phase 3: Validate Simulation Engine (Simulated Picks vs Real Results)

**Goal:** Verify that our Monte Carlo simulation produces well-calibrated win probabilities.

**Method:**
1. For each historical week, generate thousands of random pick sets (similar to how we handle unknown opponent picks)
2. For each random pick set, calculate predicted win probability using simulation
3. Evaluate those pick sets against actual NFL results
4. Check calibration: Do pick sets with 15% predicted win probability actually win ~15% when tested against real outcomes?

**Success Criteria:**
- Simulated pick sets show same calibration as real picks (Phase 2)
- No systematic over/under-confidence in probability estimates
- Consistent across different random seeds

**Output:**
- Terminal report: Calibration of simulated pick sets
- CSV file: `simulation_calibration.csv`
- Comparison chart: Real picks vs. simulated picks calibration

**Why This Matters:**
This validates the Monte Carlo simulation engine itself, independent of the optimizer. If this fails, the optimizer can't be trusted even if its logic is perfect.

**Note:** Only proceed with this phase if Phase 1 shows Yahoo spreads are well-calibrated.

---

### Phase 4: Validate Optimization Algorithms

**Goal:** Determine which optimization algorithm performs best historically.

**Method:**
1. For each historical week (Friday evening state after Thursday game):
   - Load Thursday game result + opponent Thursday picks
   - Run both greedy and hill climbing optimizers
   - Generate recommended picks for Sunday+ games
2. Evaluate each algorithm's picks against simulated outcomes (thousands of simulations)
3. Compare performance metrics across weeks and seasons

**Comparison Metrics:**
- **Expected points:** Average points scored across simulations
- **Win probability:** Percentage of simulations where picks win the week
- **Robustness:** Consistency of recommendations across different random seeds
- **Crowd differentiation:** How often does algorithm pick against the crowd?

**Algorithms to Test:**
- Greedy (baseline)
- Hill Climbing (various parameter combinations)
  - Conservative: 100 iterations, 5 restarts, 500 sims
  - Moderate: 1000 iterations, 10 restarts, 500 sims (current recommended)
  - Aggressive: 2000 iterations, 20 restarts, 1000 sims

**Success Criteria:**
- Identify algorithm with highest average win probability
- Find parameter sweet spot (diminishing returns analysis)
- Hill climbing should show more crowd differentiation than greedy

**Output:**
- Terminal report: Algorithm comparison summary
- CSV file: `algorithm_comparison.csv` (one row per week per algorithm)
- Visualization: Win probability distributions by algorithm
- Recommendation: Best algorithm + parameters for production use

**Why This Matters:**
This tells us which algorithm to trust going forward and validates that optimization actually works better than random/naive strategies.

---

### Phase 5: Solution Space Visualization (PCA/t-SNE)

**Goal:** Build intuition about the pick strategy landscape and understand tradeoffs.

**Method:**
1. For a representative week, generate thousands of pick sets using hill climbing
2. For each pick set, calculate key features:
   - Expected points
   - Win probability
   - Crowd differentiation score
   - Risk/variance
   - Points assigned to each game
3. Apply dimensionality reduction (PCA and t-SNE) to visualize in 2D
4. Color points by win probability, size by expected points

**Insights to Extract:**
- Are there distinct clusters of valid strategies?
- What tradeoffs exist between different pick approaches?
- Can we identify "safe" vs. "risky" regions?
- Does the optimizer consistently find the high-probability regions?

**Output:**
- Interactive visualization (plotly/matplotlib)
- Cluster analysis report
- PNG files: `solution_space_pca.png`, `solution_space_tsne.png`

**Why This Matters:**
This builds trust by making the abstract "optimal picks" concept tangible and visual. It helps users understand why certain picks are recommended and what alternatives exist.

---

## Implementation Plan

### Notebook Structure

All validation phases will be implemented as Jupyter notebooks for interactive exploration and visual feedback.

**Main Notebook:**
```
notebooks/
├── validation_framework.ipynb  # Master notebook with all phases
└── figures/                     # Generated visualizations
    ├── phase1_calibration_curve.png
    ├── phase2_win_probability.png
    ├── phase3_simulation_check.png
    ├── phase4_algorithm_comparison.png
    └── phase5_solution_space.png
```

**Notebook Sections:**
1. Setup & Data Loading
2. Phase 1: Yahoo Spread Calibration
3. Phase 2: Win Probability Validation
4. Phase 3: Simulation Engine Check
5. Phase 4: Algorithm Comparison
6. Phase 5: Solution Space Visualization
7. Summary & Recommendations

Each phase can be run independently or all together sequentially.

### Default Parameters for Backtesting

To balance computation time vs. thoroughness:

**Greedy:**
- `num_sims`: 5000 (middle ground)

**Hill Climbing:**
- `iterations`: 1000
- `restarts`: 10
- `num_sims`: 500
- `top_n`: 250

These can be overridden via CLI flags for experimentation.

### Output Structure

**Notebook outputs** (inline visualizations and tables):
- Calibration curves and statistics
- Win probability distributions
- Algorithm comparison charts
- Solution space visualizations

**Exported files** (for reference and further analysis):
```
validation_results/
├── phase1_yahoo_calibration.csv
├── phase2_win_probability_by_week.csv
├── phase3_simulation_calibration.csv
├── phase4_algorithm_comparison.csv
└── figures/
    ├── phase1_calibration_curve.png
    ├── phase2_win_probability.png
    ├── phase3_comparison.png
    ├── phase4_algorithm_distributions.png
    └── phase5_solution_space_*.png
```

The notebook itself serves as the primary report, with CSV exports available for deeper analysis if needed.

## Success Definition

The validation framework succeeds if:

1. ✅ Yahoo spreads are well-calibrated (Phase 1)
2. ✅ Win probability calculations are accurate (Phase 2)
3. ✅ Simulation engine is sound (Phase 3)
4. ✅ We identify the best optimization algorithm (Phase 4)
5. ✅ Solution space visualization builds intuitive understanding (Phase 5)

If any phase fails, we identify the problem and fix it before proceeding.

## Future Experiments

Once validation is complete, we can explore:
- **Parameter sensitivity:** How much do num_sims, iterations, restarts matter?
- **Live odds vs. Yahoo spreads:** Does live odds API provide meaningful edge?
- **Player skill modeling:** Does accounting for player skill improve optimization?
- **Crowd prediction:** Can we predict crowd behavior to optimize earlier in the week?
- **Mid-week re-optimization value:** Is it worth re-running after Thursday/Saturday games?

## Notes & Caveats

- **Simulated vs. real results:** Most validation uses simulations (testing the model), but Phase 1-2 use real results (testing calibration)
- **Sample size:** Two seasons of data is ~34 weeks total. Some statistical tests may have limited power.
- **Non-independence:** Players in the same league may influence each other's picks
- **Hindsight bias:** We can't test "Wednesday blind optimization" meaningfully without predicting the crowd

## Related Issues

Once this validation framework is implemented, we can file issues for:
- Simplification: Default all player skills to average
- Simplification: Make Yahoo spreads default, live odds optional
- Feature: Implement backtesting CLI tool
- Analysis: Run full validation suite and document results
- Decision: Choose default optimization algorithm based on Phase 4 results
