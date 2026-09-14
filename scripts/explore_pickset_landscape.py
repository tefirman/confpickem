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
        [--restarts N] [--temperature T] [--perturb-fraction F] [--out FILE]

    # Real Yahoo week instead of a synthetic one (needs cookies.txt + network):
    python scripts/explore_pickset_landscape.py --yahoo --week 2 [--league-id N]
        [--player NAME] [--restarts N] [--iterations N] [--out FILE]

Findings so far (synthetic 16-game weeks, small 6-player field): clusters map
almost 1:1 to hill-climb restarts -- each restart converges into its own
tight, largely isolated neighborhood rather than the search discovering a
shared "good strategy" region from multiple starting points. The team-set
overlap between restarts correlates with win probability (more overlap with
the best restart's picks = better score), which argues the good neighborhood
is reachable, not an isolated fluke -- but restarts starting from a fully
random slate rarely climb there within a normal iteration budget.
--temperature (simulated annealing) did NOT close this gap in testing (two
temperatures tried, both came back flat vs. plain hill climbing).
--perturb-fraction (seed some restarts as a few random moves away from the
greedy solution, instead of fully random) DID close it -- perturbed-greedy
restarts landed within ~1-2% of the greedy restart's win probability, vs.
random restarts trailing by ~15-25 points at the same iteration budget. That
result is specific to this exploration script's opt-in
perturbed_restart_fraction parameter; it has not been applied to
optimize_picks_hill_climb's actual defaults or the CLI.

Findings on real data (--yahoo, a real week with 58 real opponents): the
per-candidate Monte Carlo evaluation optimize_picks_hill_climb uses is both
too slow (~15s/eval at a field this size, vs ~1s for the 6-player synthetic
field) and too noisy at a sim count that would make it fast, so --yahoo uses
analytic_hill_climb_with_history instead -- the same exact, closed-form
Poisson-binomial P(win) engine as optimize_picks_analytic (~1ms/eval, no
Monte Carlo noise). The resulting win-probability landscape is much flatter
than the synthetic run's: 17/20 restarts land within 10% of the best solution
found (vs. one dominant restart out of 10 in the synthetic case). At first
glance the PCA/t-SNE plot still shows many separated clusters, which reads
like "many different competitive strategies" -- but a direct picks diff
across the most win-probability-competitive restarts shows they agree on
14-16 of 16 games; the "most different" competitive pair disagreed on only 2
games (both close to toss-ups). So the apparent visual spread is mostly the
signed-confidence encoding amplifying a couple of genuinely-close-call-game
swaps, not evidence of many fundamentally different good strategies -- the
real finding is that the optimizer converges to essentially one dominant
strategy plus a small set of interchangeable picks on the toss-up games,
which is a more useful trust-building signal than "lots of diverse options"
would have been.
"""
import argparse
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

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


def build_yahoo_week(week: int, league_id: int, cookies_file: str = "cookies.txt",
                      num_sims: int = 500):
    """Build a simulator from a live Yahoo Pick'em week -- real games, real
    opponent field. Requires a valid cookies_file (see CLAUDE.md) and network
    access. Mirrors the "beginning of week" real-opponent path in
    cli/optimize.py: opponent skills come from current_player_skills.json if
    present, else defaults.

    Returns (simulator, player_name) -- player_name is whichever yahoo.players
    entry is passed as ``your_name``, or the first player if not given.
    """
    from confpickem.yahoo_pickem_scraper import YahooPickEm
    from confpickem.yahoo_pickem_integration import convert_yahoo_to_simulator_format

    yahoo = YahooPickEm(week=week, league_id=league_id, cookies_file=cookies_file)

    games_df = convert_yahoo_to_simulator_format(yahoo, ignore_results=True)
    simulator = ConfidencePickEmSimulator(num_sims=num_sims)
    simulator.add_games_from_dataframe(games_df)

    skills_path = REPO_ROOT / "current_player_skills.json"
    player_skills = {}
    if skills_path.exists():
        with open(skills_path) as f:
            player_skills = json.load(f)

    players = []
    for _, row in yahoo.players.iterrows():
        name = row["player_name"]
        skill_data = player_skills.get(name, {})
        players.append(Player(
            name=name,
            skill_level=skill_data.get("skill_level", 0.6),
            crowd_following=skill_data.get("crowd_following", 0.5),
            confidence_following=skill_data.get("confidence_following", 0.5),
        ))
    simulator.players = players
    return simulator, yahoo.players


def analytic_hill_climb_with_history(simulator, player_name: str,
                                      iterations: int = 300, restarts: int = 15,
                                      n_outcomes: int = 6000, seed: int = 51):
    """Random-restart hill climb on the exact analytical P(win) (see
    confpickem.analytical), tracking every candidate explored -- the
    candidate-tracking equivalent of optimize_picks_hill_climb, but using the
    same no-Monte-Carlo-noise engine as optimize_picks_analytic.

    Needed for real Yahoo data: with a real ~50+ player field,
    optimize_picks_hill_climb's per-candidate Monte Carlo evaluation
    (simulate_all over the whole field) costs seconds each, and gets noisy at
    the sim counts that make it fast. pwin from the analytical field model is
    a closed-form Poisson-binomial calculation -- both fast and exact.

    Returns (best_picks_dict, all_combinations) where all_combinations is a
    list of (picks_dict, win_probability, restart_index) triples, matching
    optimize_picks_hill_climb(..., return_all_combinations=True)'s format.
    """
    from confpickem import analytical as _an

    field = simulator._build_analytic_field(player_name, n_outcomes, seed)
    games, n, pwin = field['games'], field['n'], field['pwin']
    vegas_home = field['vegas_home']
    rng = np.random.default_rng(seed)

    def to_dict(pick_home, points):
        return {(games[i].home_team if pick_home[i] else games[i].away_team): int(points[i])
                for i in range(n)}

    all_combinations = []
    best_picks, best_val = None, -1.0
    free = np.arange(n)

    for restart in range(restarts):
        if restart == 0:
            ph, pts = _an.chalk_slate(vegas_home)
        else:
            ph, pts = _an.chalk_slate(vegas_home)
            for _ in range(int(rng.integers(2, 6))):
                a, b = rng.choice(free, size=2, replace=False)
                pts[a], pts[b] = pts[b], pts[a]
            for _ in range(int(rng.integers(0, 3))):
                ph[int(rng.choice(free))] ^= True

        val = pwin(ph, pts)
        all_combinations.append((to_dict(ph, pts), float(val), restart))

        no_improve = 0
        for _ in range(iterations):
            cand_ph, cand_pts = _an._neighbor(ph, pts, free, rng)
            cand_val = pwin(cand_ph, cand_pts)
            all_combinations.append((to_dict(cand_ph, cand_pts), float(cand_val), restart))
            if cand_val > val + 1e-9:
                ph, pts, val = cand_ph, cand_pts, cand_val
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= 250:
                    break

        if val > best_val:
            best_picks, best_val = to_dict(ph, pts), val
        print(f"  restart {restart + 1}/{restarts}: {val:.4f} (best so far: {best_val:.4f})")

    return best_picks, all_combinations


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
    parser.add_argument("--temperature", type=float, default=0.0,
                         help="simulated-annealing initial temperature (0 = plain hill climb)")
    parser.add_argument("--perturb-fraction", type=float, default=0.0,
                         help="fraction of non-greedy restarts seeded near the greedy "
                              "solution instead of fully random (0 = all random)")
    parser.add_argument("--yahoo", action="store_true",
                         help="use a live Yahoo Pick'em week instead of a synthetic one "
                              "(requires cookies.txt and network access)")
    parser.add_argument("--week", type=int, default=2, help="Yahoo week number (--yahoo only)")
    parser.add_argument("--league-id", type=int, default=11465, help="Yahoo league ID (--yahoo only)")
    parser.add_argument("--player", default="Firman's Educated Guesses",
                         help="which yahoo.players entry to optimize for (--yahoo only)")
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

    if args.yahoo:
        print(f"Fetching live Yahoo week {args.week} (league {args.league_id})...")
        simulator, yahoo_players = build_yahoo_week(args.week, args.league_id)
        if args.player not in yahoo_players["player_name"].values:
            sys.exit(f"Player {args.player!r} not found in league. "
                     f"Available: {sorted(yahoo_players['player_name'].tolist())}")
        player_name = args.player
        print(f"Loaded {len(simulator.games)} games, {len(simulator.players)} players.")
    else:
        print(f"Building a synthetic {args.games}-game week...")
        simulator = build_synthetic_week(args.games, seed=args.seed)
        player_name = "Me"

    if args.yahoo:
        # Real fields are ~50+ players -- optimize_picks_hill_climb's per-candidate
        # Monte Carlo eval (simulate_all over the whole field) is too slow/noisy at
        # that size, so use the exact analytical P(win) engine instead. This path
        # doesn't (yet) support --temperature/--perturb-fraction.
        print(f"Running analytical hill-climb optimization "
              f"({args.restarts} restarts x {args.iterations} iterations)...")
        _, all_combinations = analytic_hill_climb_with_history(
            simulator, player_name, iterations=args.iterations, restarts=args.restarts,
        )
    else:
        print(f"Running hill-climb optimization ({args.restarts} restarts x {args.iterations} iterations, "
              f"temperature={args.temperature}, perturb_fraction={args.perturb_fraction})...")
        _, _, all_combinations = simulator.optimize_picks_hill_climb(
            player_name, iterations=args.iterations, restarts=args.restarts,
            top_n=len(TEAM_POOL) * 100, return_all_combinations=True,
            initial_temperature=args.temperature,
            perturbed_restart_fraction=args.perturb_fraction,
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
