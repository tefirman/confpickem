#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   analytical.py
@Author  :   Taylor Firman
@Contact :   tefirman@gmail.com
@Desc    :   Analytical (Poisson-binomial) win-probability model for confidence
             pools, and a local-search optimizer built on it.

Why this exists
---------------
A weekly confidence score is a weighted sum of independent Bernoulli terms
(``S = sum_i c_i * 1[pick i correct]``), so its full distribution is a
Poisson-binomial -- available in closed form, no simulation. Computing
``P(you finish 1st)`` this way is noise-free and ~250x faster to evaluate than a
Monte-Carlo estimate, which lets a local search actually optimize it. See
``docs/optimization-methodology.md`` for the full write-up and backtest.

The one subtlety: whether you beat opponent A and whether you beat opponent B are
correlated -- every entry is scored against the same game outcomes. We handle
that by importance-sampling the game-outcome vector and computing ``P(win | o)``
exactly for each draw.
'''

from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Poisson-binomial primitives
# ---------------------------------------------------------------------------

def weighted_pmf(probs: Sequence[float], weights: Sequence[int]) -> np.ndarray:
    """PMF of ``S = sum_i weights[i] * Bernoulli(probs[i])``.

    Returns a 1-D array ``pmf`` of length ``sum(weights) + 1`` with
    ``pmf[k] == P(S == k)``. Exact, O(n * sum(weights)).
    """
    probs = np.asarray(probs, dtype=float)
    weights = np.asarray(weights, dtype=int)
    if probs.shape != weights.shape:
        raise ValueError("probs and weights must have the same length")
    if np.any((probs < 0) | (probs > 1)):
        raise ValueError("probs must be in [0, 1]")
    if np.any(weights < 0):
        raise ValueError("weights must be non-negative integers")

    total = int(weights.sum())
    pmf = np.zeros(total + 1)
    pmf[0] = 1.0
    size = 1
    for p, w in zip(probs, weights):
        if w == 0:
            continue
        new = np.zeros(size + w)
        new[:size] += pmf[:size] * (1.0 - p)
        new[w:w + size] += pmf[:size] * p
        pmf[:size + w] = new
        size += w
    return pmf[:size]


def _batched_weighted_pmf(pc: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """``weighted_pmf`` for every row of ``pc`` at once.

    ``pc`` is ``[K, n]`` success probabilities (already conditioned on an
    outcome draw); ``weights`` is ``[n]``. Returns ``[K, T + 1]`` where
    ``T = weights.sum()``.
    """
    K, n = pc.shape
    total = int(weights.sum())
    pmf = np.zeros((K, total + 1))
    pmf[:, 0] = 1.0
    size = 1
    for i in range(n):
        w = int(weights[i])
        if w == 0:
            continue
        p = pc[:, i][:, None]
        new = np.zeros((K, size + w))
        new[:, :size] += pmf[:, :size] * (1.0 - p)
        new[:, w:w + size] += pmf[:, :size] * p
        pmf[:, :size + w] = new
        size += w
    return pmf[:, :size]


def prob_a_beats_b(pmf_a: np.ndarray, pmf_b: np.ndarray,
                   tie_credit: float = 0.5) -> float:
    """``P(A > B) + tie_credit * P(A == B)`` for independent scores A, B."""
    n = max(len(pmf_a), len(pmf_b))
    a = np.zeros(n)
    a[:len(pmf_a)] = pmf_a
    b = np.zeros(n)
    b[:len(pmf_b)] = pmf_b
    cdf_b = np.cumsum(b)
    cdf_b_lt = np.concatenate(([0.0], cdf_b[:-1]))  # P(B <= k-1)
    return float(np.dot(a, cdf_b_lt) + tie_credit * np.dot(a, b))


# ---------------------------------------------------------------------------
# Field model
# ---------------------------------------------------------------------------

# ``(p_home[n], points[n], count)`` or, midweek, ``(..., completed_points)``.
# ``completed_points`` is the fixed score this type already banked on decided
# games; 0 / omitted for a beginning-of-week field.
OpponentType = Tuple[np.ndarray, np.ndarray, int]


def modal_opponent(vegas_home: np.ndarray, crowd_home_pct: np.ndarray,
                   crowd_home_conf: np.ndarray, crowd_away_conf: np.ndarray,
                   crowd_following: float, confidence_following: float,
                   ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One modeled opponent's ``(p_home, pick_home, points)``.

    Mirrors ``ConfidencePickEmSimulator.simulate_picks``' opponent model (the one
    validated against real picks): pick probability is a Vegas/crowd blend
    weighted by ``crowd_following``; confidence is a blended score ranked into
    ``1..n``. ``pick_home`` / ``points`` are the modal (noise-free) choice.
    """
    n = len(vegas_home)
    p_home = np.clip(
        vegas_home * (1.0 - crowd_following) + crowd_home_pct * crowd_following,
        0.0, 1.0)
    pick_home = p_home > 0.5
    chosen = np.where(pick_home, crowd_home_conf, crowd_away_conf)
    opposing = np.where(pick_home, crowd_away_conf, crowd_home_conf)
    conf_diff = (chosen - opposing) / (chosen + opposing)
    vegas_conf = np.abs(vegas_home - 0.5) * 2 * n
    score = (chosen * (1.0 + conf_diff) * confidence_following
             + vegas_conf * (1.0 - confidence_following))
    # highest score -> n points, lowest -> 1
    order = np.argsort(np.argsort(score))  # ranks 0..n-1
    points = (order + 1).astype(int)
    return p_home, pick_home, points


# game_idx -> (pick_home, points)
OpponentCompletedPicks = Dict[int, Tuple[bool, int]]
OpponentLockedPicks = Dict[int, Tuple[bool, int]]


def build_opponent_types(vegas_home: np.ndarray, crowd_home_pct: np.ndarray,
                         crowd_home_conf: np.ndarray, crowd_away_conf: np.ndarray,
                         opponents: Sequence[Tuple[float, float]],
                         completed_picks: Optional[
                             Sequence[Optional[OpponentCompletedPicks]]] = None,
                         actual_outcomes: Optional[
                             Sequence[Optional[bool]]] = None,
                         locked_picks: Optional[
                             Sequence[Optional[OpponentLockedPicks]]] = None,
                         max_types: Optional[int] = None,
                         ) -> List[OpponentType]:
    """Deduplicate modeled opponents into ``(p_home, points, count[, completed])``
    types.

    ``opponents`` is a sequence of ``(crowd_following, confidence_following)``
    pairs -- one per real opponent. Opponents with identical modal
    ``(pick_home, points)`` collapse to a single weighted type, which is what
    makes the analytical P(win) fast.

    Two optional sequences, each parallel to ``opponents`` and each entry
    ``None`` or ``{game_idx: (pick_home, points)}`` with that opponent's real
    submitted pick + confidence, handle games the pool has already locked:

    * ``completed_picks`` -- games that have *finished*. Paired with
      ``actual_outcomes`` (length ``n`` home-win booleans), each is collapsed to
      a **scalar** ``completed_points`` (the points that opponent banked) and
      dropped from the Poisson-binomial -- the outcome is known, so it carries
      no variance. The 4th tuple element is that scalar; omitted / 0 otherwise.
    * ``locked_picks`` -- games that have *kicked off but not finished* (picks
      frozen, result still unknown). The opponent's ``p_home`` there is pinned
      to their real pick (1.0 / 0.0) and their real points kept, but the game
      **stays in the convolution** because its outcome is still sampled.

    Once real picks are known the modal collapse weakens (each distinct history
    is its own type), so ``max_types`` optionally caps the result: the
    ``max_types`` most common types are kept and every rarer one merged into the
    single most-common of them (its ``count`` absorbs them). Keeps ``make_pwin``
    fast at a small cost in tail fidelity.
    """
    n = len(vegas_home)
    if completed_picks is None:
        completed_picks = [None] * len(opponents)
    if locked_picks is None:
        locked_picks = [None] * len(opponents)
    if actual_outcomes is None:
        actual_outcomes = [None] * n

    buckets: Dict[Tuple, list] = {}
    for (cf, conf_foll), done, locked in zip(
            opponents, completed_picks, locked_picks):
        p_home, pick_home, points = modal_opponent(
            vegas_home, crowd_home_pct, crowd_home_conf, crowd_away_conf,
            cf, conf_foll)
        p_home = p_home.copy()
        pick_home = pick_home.copy()
        points = points.copy()
        completed_points = 0
        if locked:
            for gi, (ph_i, pts_i) in locked.items():
                if not 0 <= gi < n:
                    raise ValueError(
                        f"locked pick game index {gi} out of range")
                pick_home[gi] = bool(ph_i)
                points[gi] = int(pts_i)
                p_home[gi] = 1.0 if ph_i else 0.0  # real pick, outcome still live
        if done:
            for gi, (ph_i, pts_i) in done.items():
                if not 0 <= gi < n:
                    raise ValueError(
                        f"completed pick game index {gi} out of range")
                if actual_outcomes[gi] is None:
                    raise ValueError(
                        f"completed pick for game {gi} but no actual outcome")
                if bool(ph_i) == bool(actual_outcomes[gi]):
                    completed_points += int(pts_i)
                pick_home[gi] = bool(ph_i)
                points[gi] = 0  # decided -> out of the Poisson-binomial
        key = (tuple(pick_home.tolist()), tuple(points.tolist()),
               completed_points)
        if key in buckets:
            buckets[key][2] += 1
        else:
            buckets[key] = [p_home, points, 1, completed_points]

    types = [(pc, pts, cnt, comp) for pc, pts, cnt, comp in buckets.values()]
    if max_types is not None and len(types) > max_types:
        types.sort(key=lambda t: t[2], reverse=True)
        keep, tail = types[:max_types], types[max_types:]
        merged_count = keep[0][2] + sum(t[2] for t in tail)
        keep[0] = (keep[0][0], keep[0][1], merged_count, keep[0][3])
        types = keep
    return types


# ---------------------------------------------------------------------------
# Analytical P(win)
# ---------------------------------------------------------------------------

def sample_outcomes(vegas_home: np.ndarray, n_outcomes: int,
                    rng: np.random.Generator,
                    actual_outcomes: Optional[Sequence[Optional[bool]]] = None,
                    ) -> np.ndarray:
    """``[n_outcomes, n]`` boolean draws: did the home team win each game.

    ``actual_outcomes`` (midweek): an optional length-``n`` sequence; where an
    entry is not ``None`` that game is already decided, so its column is forced
    to the real result in every draw instead of being sampled from
    ``vegas_home``.
    """
    draws = rng.random((n_outcomes, len(vegas_home))) < vegas_home[None, :]
    if actual_outcomes is not None:
        for gi, res in enumerate(actual_outcomes):
            if res is not None:
                draws[:, gi] = bool(res)
    return draws


def make_pwin(opponent_types: Sequence[OpponentType], outcomes: np.ndarray,
              ) -> Callable[[np.ndarray, np.ndarray], float]:
    """Build ``pwin(my_pick_home, my_points) -> P(you finish 1st)``.

    ``outcomes`` is the ``[K, n]`` boolean array from :func:`sample_outcomes`.
    Each opponent type's score PMF/CDF depends only on ``outcomes`` and that
    type's modal slate -- never on the candidate ``my_pick_home`` /
    ``my_points`` -- so the returned closure precomputes them once. Every
    ``pwin`` call is then just an index into those tables plus a reduction,
    which is what makes a hill climb over thousands of candidates affordable.

    P(win) = mean over outcome draws o of  prod over opponent types t of
             P(my score > that type's score | o) ** count_t
    -- i.e. correlation across entries is captured by conditioning on o, and
    residual per-opponent independence *given o* is assumed.
    """
    K, n = outcomes.shape
    rows = np.arange(K)
    # Each opponent type's score PMF/CDF depends only on the (fixed) outcome
    # draws and that type's modal slate -- never on my picks -- so build them
    # once here. Each pwin() call then just indexes the precomputed tables.
    prepared = []
    for t in opponent_types:
        p_home, points, count = t[0], np.asarray(t[1], dtype=int), t[2]
        completed_points = int(t[3]) if len(t) > 3 else 0
        pc = np.where(outcomes, p_home[None, :], 1.0 - p_home[None, :])  # [K, n]
        pmf = _batched_weighted_pmf(pc, points)                # [K, T+1]
        cdf = np.cumsum(pmf, axis=1)
        prepared.append((pmf, cdf, pmf.shape[1], count, completed_points))

    def pwin(my_pick_home: np.ndarray, my_points: np.ndarray) -> float:
        my_scores = (my_points[None, :]
                     * (my_pick_home[None, :] == outcomes)).sum(axis=1)  # [K]
        log_beat = np.zeros(K)
        for pmf, cdf, width, count, completed_points in prepared:
            # this type's pending score must clear my lead over what it already
            # banked on decided games
            target = my_scores - completed_points
            le_idx = np.clip(target - 1, 0, width - 1)
            p_le = np.where(target >= 1, cdf[rows, le_idx], 0.0)
            eq_idx = np.clip(target, 0, width - 1)
            p_eq = np.where((target >= 0) & (target < width),
                            pmf[rows, eq_idx], 0.0)
            log_beat += count * np.log(np.maximum(p_le + 0.5 * p_eq, 1e-300))
        return float(np.exp(log_beat).mean())

    return pwin


# ---------------------------------------------------------------------------
# Slate helpers + optimizer
# ---------------------------------------------------------------------------

def chalk_slate(vegas_home: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Favorite in every game, confidence ranked by ``|vegas - 0.5|``.

    Returns ``(pick_home[n] bool, points[n] int)``.
    """
    n = len(vegas_home)
    order = np.argsort(-np.abs(vegas_home - 0.5))
    pick_home = vegas_home >= 0.5
    points = np.zeros(n, dtype=int)
    for rank, gi in enumerate(order):
        points[gi] = n - rank
    return pick_home, points


def _neighbor(pick_home: np.ndarray, points: np.ndarray,
              free: np.ndarray, rng: np.random.Generator,
              ) -> Tuple[np.ndarray, np.ndarray]:
    """One move away, touching only ``free`` games: flip a pick, or swap two
    confidence values (both between free games, so the permutation stays valid)."""
    ph, pts = pick_home.copy(), points.copy()
    if rng.random() < 0.5 or len(free) < 2:
        ph[int(rng.choice(free))] ^= True
    else:
        a, b = rng.choice(free, size=2, replace=False)
        pts[a], pts[b] = pts[b], pts[a]
    return ph, pts


def _seed_slate(vegas_home: np.ndarray, locked: np.ndarray,
                pick_home_fixed: Optional[np.ndarray],
                points_fixed: Optional[np.ndarray],
                ) -> Tuple[np.ndarray, np.ndarray]:
    """Chalk on the free games using the leftover confidence values; fixed games
    kept exactly. Always a valid 1..n permutation."""
    n = len(vegas_home)
    ph, pts = chalk_slate(vegas_home)
    if not locked.any():
        return ph, pts
    ph = ph.copy()
    pts = pts.copy()
    ph[locked] = pick_home_fixed[locked]
    pts[locked] = points_fixed[locked]
    free = np.where(~locked)[0]
    leftover = sorted(set(range(1, n + 1)) - set(points_fixed[locked].tolist()),
                      reverse=True)
    # most-certain free game gets the largest leftover value
    order = free[np.argsort(-np.abs(vegas_home[free] - 0.5))]
    for gi, val in zip(order, leftover):
        pts[gi] = val
    return ph, pts


def optimize_slate(pwin: Callable[[np.ndarray, np.ndarray], float],
                   vegas_home: np.ndarray,
                   pick_home_fixed: Optional[np.ndarray] = None,
                   points_fixed: Optional[np.ndarray] = None,
                   iterations: int = 400, restarts: int = 4,
                   rng: Optional[np.random.Generator] = None,
                   ) -> Tuple[np.ndarray, np.ndarray, float]:
    """Random-restart hill climb on ``pwin``, seeded from chalk.

    ``pick_home_fixed`` / ``points_fixed``: optional ``[n]`` arrays; where
    ``points_fixed[i] > 0`` that game's pick and confidence are held constant
    (used to respect already-locked midweek picks). The free games always carry
    a valid permutation of the leftover confidence values.

    Returns ``(pick_home, points, pwin_value)`` -- ``points`` is always a
    permutation of ``1..n``.
    """
    rng = rng or np.random.default_rng(51)
    n = len(vegas_home)
    locked = (points_fixed > 0) if points_fixed is not None else np.zeros(n, bool)
    free = np.where(~locked)[0]

    base_ph, base_pts = _seed_slate(vegas_home, locked, pick_home_fixed, points_fixed)
    best_ph, best_pts = base_ph.copy(), base_pts.copy()
    best_val = pwin(best_ph, best_pts)

    for restart in range(restarts):
        if restart == 0:
            ph, pts = base_ph.copy(), base_pts.copy()
        else:
            ph, pts = base_ph.copy(), base_pts.copy()
            if len(free) >= 2:
                for _ in range(int(rng.integers(2, 6))):
                    a, b = rng.choice(free, size=2, replace=False)
                    pts[a], pts[b] = pts[b], pts[a]
                for _ in range(int(rng.integers(0, 3))):
                    ph[int(rng.choice(free))] ^= True
        val = pwin(ph, pts)

        no_improve = 0
        for _ in range(iterations):
            cand_ph, cand_pts = _neighbor(ph, pts, free, rng)
            cand_val = pwin(cand_ph, cand_pts)
            if cand_val > val + 1e-9:
                ph, pts, val = cand_ph, cand_pts, cand_val
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= 250:
                    break

        if val > best_val:
            best_ph, best_pts, best_val = ph.copy(), pts.copy(), val

    return best_ph, best_pts, best_val
