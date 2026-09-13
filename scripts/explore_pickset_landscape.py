#!/usr/bin/env python
"""Exploratory script for issue #8: visualize the landscape of pick sets.

Samples the pick-set combinations explored by ``optimize_picks_hill_climb``,
encodes each as a signed-confidence vector (one dimension per game: +confidence
if the home team was picked, -confidence if the away team was picked), and
projects them to 2D with PCA and t-SNE to see whether "good" pick sets cluster
into a few distinct strategies (chalk vs. contrarian vs. balanced) or are
scattered/noisy -- i.e. whether the optimizer is finding a real local
neighborhood of good solutions or just getting lucky.

This is exploratory (see notes/issue-populate-game-picks-locked.md-style
one-offs and notes/HILL_CLIMB_SUMMARY_STATS.md, its direct precursor) and not
wired into the CLI. Requires scikit-learn and matplotlib, which are NOT
project dependencies -- install with:

    pip install scikit-learn matplotlib

Usage:

    python scripts/explore_pickset_landscape.py [--games N] [--iterations N]
        [--restarts N] [--out FILE]
"""
import argparse
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from confpickem.confidence_pickem_sim import ConfidencePickEmSimulator, Game, Player

TEAM_POOL = [
    "KC", "BUF", "PHI", "SF", "DAL", "BAL", "DET", "MIA",
    "CIN", "HOU", "GB", "LAR", "NYJ", "MIN", "SEA", "JAX",
    "LV", "NYG", "CAR", "ATL", "CHI", "NE", "TEN", "IND",
    "CLE", "PIT", "DEN", "LAC", "ARI", "WAS", "TB", "NO",
]


def build_synthetic_week(num_games: int, seed: int = 7) -> ConfidencePickEmSimulator:
    """Build a simulator with a plausible slate of games and an opponent field.

    No live scrape / cookies needed -- spreads are sampled to look like a
    realistic NFL week (a handful of near-toss-ups, a handful of heavy
    favorites, most in between).
    """
    rng = np.random.default_rng(seed)
    teams = rng.choice(TEAM_POOL, size=2 * num_games, replace=False)
    kickoff = datetime.now() + timedelta(days=3)

    games_data = pd.DataFrame({
        "home_team": teams[:num_games],
        "away_team": teams[num_games:],
        "vegas_win_prob": np.clip(rng.normal(0.6, 0.15, num_games), 0.51, 0.95),
        "crowd_home_pick_pct": np.clip(rng.normal(0.6, 0.15, num_games), 0.05, 0.95),
        "crowd_home_confidence": rng.uniform(4, 13, num_games),
        "crowd_away_confidence": rng.uniform(4, 13, num_games),
        "week": 1,
        "kickoff_time": kickoff,
        "actual_outcome": None,
    })

    # Hill climbing re-evaluates a full Monte Carlo run per candidate (~1-5s at
    # 500-2000 sims), so keep this modest -- accuracy of any one win-prob
    # estimate matters far less here than being able to explore many candidates.
    simulator = ConfidencePickEmSimulator(num_sims=500)
    simulator.add_games_from_dataframe(games_data)
    simulator.players = [
        Player("Me", skill_level=0.7, crowd_following=0.3, confidence_following=0.4),
        Player("Expert", skill_level=0.9, crowd_following=0.2, confidence_following=0.3),
        Player("Crowd Follower", skill_level=0.5, crowd_following=0.9, confidence_following=0.8),
        Player("Average Joe", skill_level=0.5, crowd_following=0.5, confidence_following=0.5),
        Player("Contrarian", skill_level=0.6, crowd_following=0.1, confidence_following=0.4),
        Player("Chalk Eater", skill_level=0.55, crowd_following=0.7, confidence_following=0.6),
    ]
    return simulator


def vectorize(games: list, picks: dict) -> np.ndarray:
    """Signed-confidence encoding: +conf if home picked, -conf if away picked.

    Keeps confidence-adjacent picks (e.g. swapping 15<->16) close in space,
    unlike a raw one-hot/permutation encoding -- see issue #8's distance-metric
    concern.
    """
    vec = np.zeros(len(games))
    for i, g in enumerate(games):
        if g.home_team in picks:
            vec[i] = picks[g.home_team]
        elif g.away_team in picks:
            vec[i] = -picks[g.away_team]
    return vec


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=int, default=16, help="number of games in the synthetic week")
    parser.add_argument("--iterations", type=int, default=300, help="hill-climb iterations per restart")
    parser.add_argument("--restarts", type=int, default=15, help="hill-climb random restarts")
    parser.add_argument("--seed", type=int, default=7, help="synthetic-week RNG seed")
    parser.add_argument("--out", default="pickset_landscape.png", help="output image path")
    args = parser.parse_args()

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.decomposition import PCA
        from sklearn.manifold import TSNE
    except ImportError as exc:
        sys.exit(f"Missing dependency ({exc}). Install with: pip install scikit-learn matplotlib")

    print(f"Building a synthetic {args.games}-game week...")
    simulator = build_synthetic_week(args.games, seed=args.seed)

    print(f"Running hill-climb optimization ({args.restarts} restarts x {args.iterations} iterations)...")
    _, _, all_combinations = simulator.optimize_picks_hill_climb(
        "Me", iterations=args.iterations, restarts=args.restarts,
        top_n=len(TEAM_POOL) * 100, return_all_combinations=True,
    )
    print(f"Explored {len(all_combinations):,} combinations.")

    viable = [(picks, prob, restart) for picks, prob, restart in all_combinations if prob > 0]
    viable.sort(key=lambda x: x[1], reverse=True)
    top = viable[:1000]
    print(f"Clustering the top {len(top)} viable combinations (win prob > 0).")

    vectors = np.array([vectorize(simulator.games, picks) for picks, _, _ in top])
    win_probs = np.array([prob for _, prob, _ in top])
    restarts = np.array([restart for _, _, restart in top])

    pca_coords = PCA(n_components=2, random_state=0).fit_transform(vectors)
    perplexity = min(30, max(5, len(top) // 10))
    tsne_coords = TSNE(n_components=2, random_state=0, perplexity=perplexity,
                        init="pca").fit_transform(vectors)

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    for ax, coords, title in [(axes[0], pca_coords, "PCA"), (axes[1], tsne_coords, "t-SNE")]:
        scatter = ax.scatter(coords[:, 0], coords[:, 1], c=win_probs, cmap="viridis", s=18, alpha=0.8)
        ax.set_title(f"{title} of top {len(top)} pick sets")
        ax.set_xlabel("component 1")
        ax.set_ylabel("component 2")
    fig.colorbar(scatter, ax=axes[1], label="win probability", shrink=0.8)

    restart_scatter = axes[2].scatter(tsne_coords[:, 0], tsne_coords[:, 1], c=restarts,
                                       cmap="tab20", s=18, alpha=0.8)
    axes[2].set_title("t-SNE colored by restart (restart 0 = greedy start)")
    axes[2].set_xlabel("component 1")
    axes[2].set_ylabel("component 2")
    fig.colorbar(restart_scatter, ax=axes[2], label="restart index", shrink=0.8)

    fig.suptitle("Pick-set landscape (signed-confidence encoding)")
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"Saved plot to {args.out}")


if __name__ == "__main__":
    main()
