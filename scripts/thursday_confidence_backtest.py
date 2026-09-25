#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""Synthetic backtest of the "max confidence on Thursday night" strategy.

Some players lock their highest confidence value (16 in a 16-game week) onto
the Thursday night game before the rest of the week's picks are even due,
independent of what the optimizer would choose. This script asks, using the
same analytical P(finish 1st) model ``optimize_picks_analytic`` searches:

1. Left alone, how often does the optimizer *itself* put max confidence on
   the early game anyway?
2. When it doesn't, what does forcing max confidence onto that game cost in
   P(finish 1st) -- isolating the "which slot gets your 16" decision from
   "which team you picked" by pinning the optimizer's own preferred side?
3. How much worse is it if, like the real players described, you also guess
   the side yourself (approximated here as leaning to the Vegas favorite)
   rather than the optimizer's independently preferred side?

No real Yahoo/PickEmCache data is used or required -- this environment has
none cached (it's gitignored and requires a live ``cookies.txt``). Each
"week" is synthetically generated: Vegas win probabilities, crowd pick
percentages and crowd confidence stats drawn from distributions that mirror
what ``tests/test_analytical.py`` and real Yahoo weeks look like, and a field
of opponents with varied crowd-following / confidence-following behavior
(skill_level doesn't factor into the analytical opponent-type model, so it's
irrelevant here).

Run: python scripts/thursday_confidence_backtest.py [--weeks N] [--games N]
"""

import argparse
import sys
from datetime import datetime, timedelta

import numpy as np

sys.path.insert(0, "src")

from confpickem.confidence_pickem_sim import ConfidencePickEmSimulator, Game, Player
from confpickem import analytical as _an


def make_week_games(rng: np.random.Generator, n_games: int) -> list:
    """A synthetic week's worth of games, one flagged as the Thursday opener."""
    vegas_home = np.clip(0.5 + rng.normal(0, 0.16, n_games), 0.03, 0.97)
    crowd_home_pct = np.clip(vegas_home + rng.normal(0, 0.05, n_games), 0.02, 0.98)
    lean = np.abs(vegas_home - 0.5)
    crowd_home_conf = lean * 18 + 3 + rng.normal(0, 0.5, n_games)
    crowd_away_conf = 9 - lean * 6 + rng.normal(0, 0.5, n_games)

    thursday_kickoff = datetime(2025, 11, 6, 20, 15)
    games = []
    for i in range(n_games):
        kickoff = thursday_kickoff if i == 0 else thursday_kickoff + timedelta(
            days=3, hours=int(rng.integers(0, 8)))
        games.append(Game(
            home_team=f"HOME{i}", away_team=f"AWAY{i}",
            vegas_win_prob=float(vegas_home[i]),
            crowd_home_pick_pct=float(crowd_home_pct[i]),
            crowd_home_confidence=float(crowd_home_conf[i]),
            crowd_away_confidence=float(crowd_away_conf[i]),
            week=1, kickoff_time=kickoff,
        ))
    return games


def make_field(rng: np.random.Generator, n_opponents: int) -> list:
    """A pool of opponents with varied crowd/confidence-following behavior."""
    players = []
    for i in range(n_opponents):
        skill = float(np.clip(rng.normal(0.5, 0.18), 0.05, 0.95))
        crowd_following = float(np.clip(rng.normal(0.5, 0.2), 0.05, 0.95))
        confidence_following = float(np.clip(rng.normal(0.5, 0.2), 0.05, 0.95))
        players.append(Player(f"Opp{i}", skill, crowd_following, confidence_following))
    return players


def run_week(seed: int, n_games: int, n_opponents: int,
             n_outcomes: int, iterations: int, restarts: int) -> dict:
    rng = np.random.default_rng(seed)
    games = make_week_games(rng, n_games)

    sim = ConfidencePickEmSimulator(num_sims=10)
    sim.games = games
    me = Player("Me", 0.7, 0.3, 0.3)
    sim.players = [me] + make_field(rng, n_opponents)

    field = sim._build_analytic_field("Me", n_outcomes, seed, player_data=None, as_of=None)
    vegas_home, pwin, n = field["vegas_home"], field["pwin"], field["n"]

    ph_free, pts_free, val_free = _an.optimize_slate(
        pwin, vegas_home, iterations=iterations, restarts=restarts,
        rng=np.random.default_rng(seed * 7 + 1))

    thursday_points_free = int(pts_free[0])
    optimizer_maxed_thursday = thursday_points_free == n

    # Condition 2: force max confidence onto Thursday, optimizer's own side.
    points_fixed = np.zeros(n, dtype=int)
    pick_home_fixed = np.zeros(n, dtype=bool)
    points_fixed[0] = n
    pick_home_fixed[0] = ph_free[0]
    _, _, val_forced_optimal_side = _an.optimize_slate(
        pwin, vegas_home, pick_home_fixed=pick_home_fixed, points_fixed=points_fixed,
        iterations=iterations, restarts=restarts, rng=np.random.default_rng(seed * 7 + 2))

    # Condition 3: force max confidence onto Thursday, guessing the Vegas favorite
    # (approximating a real player's own read rather than the optimizer's pick).
    points_fixed_chalk = np.zeros(n, dtype=int)
    pick_home_fixed_chalk = np.zeros(n, dtype=bool)
    points_fixed_chalk[0] = n
    pick_home_fixed_chalk[0] = vegas_home[0] >= 0.5
    _, _, val_forced_chalk = _an.optimize_slate(
        pwin, vegas_home, pick_home_fixed=pick_home_fixed_chalk, points_fixed=points_fixed_chalk,
        iterations=iterations, restarts=restarts, rng=np.random.default_rng(seed * 7 + 3))

    return dict(
        seed=seed,
        n_games=n,
        thursday_vegas_prob=float(vegas_home[0]),
        optimizer_maxed_thursday=optimizer_maxed_thursday,
        thursday_points_free=thursday_points_free,
        val_free=val_free,
        cost_optimal_side=val_free - val_forced_optimal_side,
        cost_chalk_side=val_free - val_forced_chalk,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weeks", type=int, default=150)
    parser.add_argument("--games", type=int, default=14)
    parser.add_argument("--opponents", type=int, default=60)
    parser.add_argument("--outcomes", type=int, default=4000)
    parser.add_argument("--iterations", type=int, default=250)
    parser.add_argument("--restarts", type=int, default=6)
    args = parser.parse_args()

    results = [
        run_week(seed, args.games, args.opponents, args.outcomes,
                  args.iterations, args.restarts)
        for seed in range(args.weeks)
    ]

    maxed = [r for r in results if r["optimizer_maxed_thursday"]]
    not_maxed = [r for r in results if not r["optimizer_maxed_thursday"]]
    costs_optimal = np.array([r["cost_optimal_side"] for r in not_maxed])
    costs_chalk = np.array([r["cost_chalk_side"] for r in results])

    print(f"Weeks simulated: {len(results)} ({args.games} games each, "
          f"{args.opponents} modeled opponents)\n")

    print(f"Optimizer independently maxed Thursday's confidence: "
          f"{len(maxed)}/{len(results)} ({100 * len(maxed) / len(results):.1f}%)")

    if len(not_maxed):
        print(f"\nOn the {len(not_maxed)} weeks it didn't, forcing max confidence onto "
              f"Thursday (optimizer's own side) cost, in P(finish 1st):")
        print(f"  mean  {costs_optimal.mean():+.4f}")
        print(f"  median{np.median(costs_optimal):+.4f}")
        print(f"  worst {costs_optimal.min():+.4f}")
        print(f"  best  {costs_optimal.max():+.4f}")

    print(f"\nForcing max confidence onto Thursday's Vegas favorite (a real player's "
          f"gut read, not the optimizer's pick), across all {len(results)} weeks:")
    print(f"  mean cost  {costs_chalk.mean():+.4f}")
    print(f"  median cost{np.median(costs_chalk):+.4f}")
    print(f"  worst cost {costs_chalk.min():+.4f}")
    print(f"  best cost  {costs_chalk.max():+.4f}")
    print(f"  weeks it helped (negative cost): "
          f"{(costs_chalk < -1e-9).sum()}/{len(results)}")
    print(f"  weeks it hurt (positive cost):   "
          f"{(costs_chalk > 1e-9).sum()}/{len(results)}")


if __name__ == "__main__":
    main()
