#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   season_sim.py
@Author  :   Taylor Firman
@Contact :   tefirman@gmail.com
@Desc    :   Season-long Monte Carlo for the confidence pick'em league: weekly
             prize + season-total prizes, from the current standings.

What it models
--------------
The league pays ``WEEKLY_PRIZE`` to each week's top scorer and ``SEASON_PRIZES``
to the top finishers by total points. Given the standings so far, every
remaining week is simulated for every active player:

* The week's slate (win probabilities, crowd pick share, crowd confidence) is
  not known in advance, so it is *sampled* from the weeks already on disk in
  ``PickEmCache<year>/`` -- a whole historical slate at a time, which keeps the
  game count, spread mix and crowd behaviour internally consistent.
* Game outcomes are drawn from those win probabilities once per simulated week
  and shared by every player (so the field's scores are correlated, as they are
  in real life).
* Each player's picks and confidence points follow the same behavioural model as
  ``ConfidencePickEmSimulator.simulate_picks`` (skill / crowd-following /
  confidence-following knobs), vectorised across sims x players x games.

Ties split a prize evenly among the tied players.
"""

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .analytical import _win_credit

WEEKLY_PRIZE = 500.0
SEASON_PRIZES = (2250.0, 1000.0, 500.0, 250.0)
SEASON_WEEKS = 18
SPREAD_SLOPE = 0.031  # win-prob per point of spread; matches yahoo_pickem_scraper
# Calibration (scripts/calibrate_season_sim.py, docs/season-sim-calibration.md):
# the weekly simulator's pick/confidence noise gives a within-week spread of ~8 pts
# between players; real weeks show ~10.4. Widening the noise 2.8x matches it.
SEASON_NOISE_SCALE = 2.8
# Real players also differ persistently (SD of true weekly ability, pts/week,
# estimated from the 2024-2026 seasons; the skill knobs alone give ~0.4).
ABILITY_SD = 3.0
WEEKLY_SD = 10.0  # a single player's week-to-week SD, used to shrink season-to-date results
DEFAULT_SKILLS = (0.6, 0.5, 0.5)  # skill_level, crowd_following, confidence_following


# ---------------------------------------------------------------------------
# Week slates
# ---------------------------------------------------------------------------


@dataclass
class WeekSpec:
    """One week's slate, all arrays length ``n_games`` and from "side A"'s view.

    For a live week side A is the home team; for sampled historical templates it
    is the favorite (the simulation is symmetric, so which side is "A" is only a
    labelling choice).
    """

    p_a: np.ndarray  # P(side A wins), 0-1
    crowd_a: np.ndarray  # share of the crowd picking side A, 0-1
    conf_a: np.ndarray  # crowd's mean confidence when picking side A
    conf_b: np.ndarray  # ... and when picking side B
    source: str = ""  # e.g. "2025 wk3"

    @property
    def n(self) -> int:
        return len(self.p_a)


def spread_to_win_prob(spread) -> np.ndarray:
    """Favorite's win probability from its point spread (Yahoo's linear map)."""
    return np.clip(np.asarray(spread, dtype=float) * SPREAD_SLOPE + 0.5, 0.0, 1.0)


def week_spec_from_games(games) -> WeekSpec:
    """WeekSpec (side A = home) from ``Game`` objects."""
    return WeekSpec(
        p_a=np.array([g.vegas_win_prob for g in games], dtype=float),
        crowd_a=np.array([g.crowd_home_pick_pct for g in games], dtype=float),
        conf_a=np.array([g.crowd_home_confidence for g in games], dtype=float),
        conf_b=np.array([g.crowd_away_confidence for g in games], dtype=float),
        source="live",
    )


def available_cache_years(root: str = ".") -> List[int]:
    """Seasons with a ``PickEmCache<year>/`` directory under ``root``."""
    years = []
    for path in Path(root).glob("PickEmCache*"):
        m = re.fullmatch(r"PickEmCache(\d{4})", path.name)
        if m and path.is_dir():
            years.append(int(m.group(1)))
    return sorted(years)


def _week_files(cache_dir: Path) -> List[Tuple[int, Path]]:
    out = []
    for path in cache_dir.glob("confidence_picks_week*.html"):
        m = re.fullmatch(r"confidence_picks_week(\d+)\.html", path.name)
        if m:
            out.append((int(m.group(1)), path))
    return sorted(out)


def load_week_templates(
    root: str = ".",
    years: Optional[Iterable[int]] = None,
    exclude: Optional[Iterable[Tuple[int, int]]] = None,
) -> List[WeekSpec]:
    """Historical slates to sample future weeks from.

    Reads every cached ``PickEmCache<year>/confidence_picks_week<N>.html`` (and
    its sibling ``pick_distribution`` page for crowd data). A week is kept only
    if every game has a spread, crowd data and a result, so templates are complete
    and pages cached before their games were played are skipped.

    Args:
        root: directory holding the ``PickEmCache<year>`` folders.
        years: restrict to these seasons (default: all found).
        exclude: ``(year, week)`` pairs to leave out -- backtests use this so a
            forecast never samples from weeks that hadn't happened yet.
    """
    from .analyze_player_skills import parse_week_data

    excluded = set(exclude or ())
    wanted = list(years) if years is not None else available_cache_years(root)
    templates: List[WeekSpec] = []
    for year in wanted:
        cache_dir = Path(root) / f"PickEmCache{year}"
        for week, path in _week_files(cache_dir):
            if (year, week) in excluded:
                continue
            games, _ = _quiet_parse(parse_week_data, path)
            spec = _spec_from_parsed_games(games, f"{year} wk{week}")
            if spec is not None:
                templates.append(spec)
    return templates


MAX_UNRESOLVED = 1  # a tied game has no winner; tolerate one per slate


def _unresolved(games) -> int:
    return sum(1 for g in games if not g.get("winner"))


def _quiet_parse(parse_week_data, path):
    import contextlib
    import io

    with contextlib.redirect_stdout(io.StringIO()):
        return parse_week_data(path)


def _spec_from_parsed_games(games, source: str) -> Optional[WeekSpec]:
    if not games:
        return None
    needed = ("spread", "favorite_pick_pct", "favorite_confidence", "underdog_confidence")
    if any(any(g.get(k) is None for k in needed) for g in games):
        return None
    if _unresolved(games) > MAX_UNRESOLVED:
        return None  # slate not played yet (or page cached before results): stale
    return WeekSpec(
        p_a=spread_to_win_prob([g["spread"] for g in games]),
        crowd_a=np.array([g["favorite_pick_pct"] for g in games], dtype=float) / 100.0,
        conf_a=np.array([g["favorite_confidence"] for g in games], dtype=float),
        conf_b=np.array([g["underdog_confidence"] for g in games], dtype=float),
        source=source,
    )


# ---------------------------------------------------------------------------
# Player skills
# ---------------------------------------------------------------------------


def _casefold_lookup(mapping: Mapping[str, Any]) -> Dict[str, Any]:
    return {str(k).casefold(): v for k, v in mapping.items()}


def build_skill_array(
    names: Sequence[str],
    skills_by_name: Optional[Dict[str, Dict]],
) -> Tuple[np.ndarray, int]:
    """``[P, 3]`` array of (skill_level, crowd_following, confidence_following).

    Names match exactly, then case-insensitively. Players with no history get the
    mean of the known players (not the CLI's flat 0.6/0.5/0.5 default, which sits
    far from this league's actual mean and would make newcomers look odd).
    Returns ``(array, number_matched)``.
    """
    skills_by_name = skills_by_name or {}
    folded = _casefold_lookup(skills_by_name)
    keys = ("skill_level", "crowd_following", "confidence_following")
    rows: List[Optional[List[float]]] = []
    for name in names:
        entry = skills_by_name.get(name) or folded.get(str(name).casefold())
        rows.append([float(entry[k]) for k in keys] if entry else None)
    known = [r for r in rows if r is not None]
    default = list(np.mean(known, axis=0)) if known else list(DEFAULT_SKILLS)
    arr = np.array([r if r is not None else default for r in rows], dtype=float)
    return arr, len(known)


def load_skills(path: str = "current_player_skills.json") -> Dict[str, Dict]:
    import json

    try:
        with open(path) as f:
            return json.load(f)
    except FileNotFoundError:
        return {}


# ---------------------------------------------------------------------------
# One week for the whole field
# ---------------------------------------------------------------------------


@dataclass
class FrozenPicks:
    """Real picks for games already locked, for midweek simulation.

    ``mask[n]`` marks the locked games; ``pick_a[P, n]`` / ``points[P, n]`` are
    each player's real pick (True = side A) and confidence there (0 = no pick,
    scores nothing). Free games are re-simulated over each player's *unspent*
    confidence values.
    """

    mask: np.ndarray
    pick_a: np.ndarray
    points: np.ndarray


def sample_outcomes(
    p_a: np.ndarray,
    num_sims: int,
    rng: np.random.Generator,
    actual: Optional[Sequence[Optional[bool]]] = None,
) -> np.ndarray:
    """``[S, n]`` bool: did side A win. ``actual`` pins decided games."""
    draws = rng.random((num_sims, len(p_a))) < p_a[None, :]
    if actual is not None:
        for i, res in enumerate(actual):
            if res is not None:
                draws[:, i] = bool(res)
    return draws


def simulate_week_scores(
    rng: np.random.Generator,
    spec: WeekSpec,
    skills: np.ndarray,
    outcomes: np.ndarray,
    frozen: Optional[FrozenPicks] = None,
    chunk: int = 1000,
    noise_scale: float = 1.0,
) -> np.ndarray:
    """Weekly scores ``[S, P]`` for every player on shared ``outcomes`` ``[S, n]``.

    Same behavioural model as ``ConfidencePickEmSimulator.simulate_picks``: pick
    probability is a Vegas/crowd blend (weight ``crowd_following``) plus noise
    that shrinks with ``skill_level``; confidence points rank a blend of the
    crowd's confidence and the Vegas margin (weight ``confidence_following``)
    plus skill-shrunk noise. ``noise_scale`` multiplies both noise terms (1.0 is the
    weekly simulator's calibration; the season simulator widens it, see
    :data:`SEASON_NOISE_SCALE`).
    """
    S, n = outcomes.shape
    P = len(skills)
    skill, cf, cff = skills[:, 0], skills[:, 1], skills[:, 2]
    base = spec.p_a[None, :] * (1 - cf[:, None]) + spec.crowd_a[None, :] * cf[:, None]
    vegas_conf = np.abs(spec.p_a - 0.5) * 2 * n
    sloppy = (1 - skill)[None, :, None]

    free_idx = np.arange(n)
    leftover = None
    if frozen is not None and frozen.mask.any():
        free_idx = np.flatnonzero(~frozen.mask)
        nf = len(free_idx)
        leftover = np.zeros((P, nf), dtype=int)
        for p in range(P):
            spent = set(int(v) for v in frozen.points[p][frozen.mask] if v > 0)
            avail = sorted(set(range(1, n + 1)) - spent)
            leftover[p] = avail[-nf:] if nf else []

    scores = np.zeros((S, P))
    for lo in range(0, S, chunk):
        hi = min(S, lo + chunk)
        c = hi - lo
        noise = rng.normal(0, 0.1 * noise_scale, (c, P, n)) * sloppy
        p_pick = np.clip(base[None] + noise, 0.0, 1.0)
        pick_a = rng.random((c, P, n)) < p_pick

        chosen = np.where(pick_a, spec.conf_a, spec.conf_b)
        opposing = np.where(pick_a, spec.conf_b, spec.conf_a)
        denom = np.where(chosen + opposing == 0, 1.0, chosen + opposing)
        conf_diff = (chosen - opposing) / denom
        blended = (
            chosen * (1 + conf_diff) * cff[None, :, None]
            + vegas_conf[None, None, :] * (1 - cff)[None, :, None]
        )
        final = np.clip(blended + rng.normal(0, 2 * noise_scale, (c, P, n)) * sloppy, 1, n)
        final = final + rng.random((c, P, n)) * 1e-6  # break ties at the clip edges

        if leftover is None or frozen is None:
            ranks = np.argsort(np.argsort(final, axis=2), axis=2)
            points = ranks + 1
        else:
            points = np.zeros((c, P, n), dtype=int)
            sub = final[:, :, free_idx]
            ranks = np.argsort(np.argsort(sub, axis=2), axis=2)
            points[:, :, free_idx] = leftover[np.arange(P)[None, :, None], ranks]
            m = frozen.mask
            points[:, :, m] = frozen.points[None, :, m]
            pick_a[:, :, m] = frozen.pick_a[None, :, m]

        correct = pick_a == outcomes[lo:hi, None, :]
        scores[lo:hi] = (points * correct).sum(axis=2)
    return scores


# ---------------------------------------------------------------------------
# Prizes
# ---------------------------------------------------------------------------


def _prize_cumsum(prizes: Sequence[float], width: int) -> np.ndarray:
    arr = np.zeros(width)
    k = min(width, len(prizes))
    arr[:k] = prizes[:k]
    return np.concatenate(([0.0], np.cumsum(arr)))


def season_payouts(
    totals: np.ndarray, prizes: Sequence[float] = SEASON_PRIZES, chunk: int = 2000
) -> Tuple[np.ndarray, np.ndarray]:
    """Season prize money and place probabilities from final totals.

    Args:
        totals: ``[S, P]`` final season points.
    Returns:
        ``(dollars[S, P], place[S, P, len(prizes)])`` -- ``place[s, p, j]`` is the
        share of finishing in position ``j`` (a tie over positions ``lo..hi-1``
        splits evenly, so shares are fractional).
    """
    S, P = totals.shape
    cum = _prize_cumsum(prizes, P)
    dollars = np.zeros((S, P))
    place = np.zeros((S, P, len(prizes)))
    for lo in range(0, S, chunk):
        t = totals[lo : lo + chunk]
        greater = (t[:, None, :] > t[:, :, None]).sum(axis=2)  # strictly ahead
        at_least = (t[:, None, :] >= t[:, :, None]).sum(axis=2)  # incl. self
        width = at_least - greater
        dollars[lo : lo + chunk] = (cum[at_least] - cum[greater]) / width
        for j in range(len(prizes)):
            place[lo : lo + chunk, :, j] = ((greater <= j) & (j < at_least)) / width
    return dollars, place


def prize_for_one(
    me: np.ndarray, others: np.ndarray, prizes: Sequence[float] = SEASON_PRIZES
) -> np.ndarray:
    """Season prize for a single player: ``me`` ``[S]`` against ``others`` ``[S, Q]``."""
    cum = _prize_cumsum(prizes, others.shape[1] + 1)
    greater = (others > me[:, None]).sum(axis=1)
    at_least = (others >= me[:, None]).sum(axis=1) + 1
    return (cum[at_least] - cum[greater]) / (at_least - greater)


# ---------------------------------------------------------------------------
# Season simulator
# ---------------------------------------------------------------------------


@dataclass
class SeasonSimulator:
    """Simulate the rest of the season from the standings so far.

    Args:
        names: player names, one per row of ``points``.
        points: ``[P]`` season points banked through ``completed_weeks``.
        skills: ``[P, 3]`` behavioural knobs (see :func:`build_skill_array`).
        templates: historical slates to sample the remaining weeks from.
        completed_weeks: weeks already final; ``completed_weeks + 1 .. total_weeks``
            are simulated.
        active: ``[P]`` bool; inactive players keep their total and score nothing
            more (see :func:`infer_active`).
        ability_mean / ability_post_sd: ``[P]`` posterior mean / SD (points per
            week) of each player's persistent edge over the field, from their
            season so far (see :func:`estimate_ability`). Drawn once per simulated
            season and added to every week they play. ``None`` = no edge.
        noise_scale: widens the pick noise (see :data:`SEASON_NOISE_SCALE`).
        play_prob: ``[P]`` chance an active player submits picks in a given week;
            a skipped week scores 0 (see :func:`participation_rates`).
        banked_weekly_wins: ``[P]`` weekly prizes already won (reporting only).
    """

    names: List[str]
    points: np.ndarray
    skills: np.ndarray
    templates: List[WeekSpec]
    completed_weeks: int
    total_weeks: int = SEASON_WEEKS
    active: np.ndarray = None  # type: ignore[assignment]  # filled in __post_init__
    play_prob: np.ndarray = None  # type: ignore[assignment]  # filled in __post_init__
    ability_mean: np.ndarray = None  # type: ignore[assignment]  # filled in __post_init__
    ability_post_sd: np.ndarray = None  # type: ignore[assignment]  # filled in __post_init__
    noise_scale: float = SEASON_NOISE_SCALE
    banked_weekly_wins: np.ndarray = None  # type: ignore[assignment]  # filled in __post_init__
    num_sims: int = 10000
    seed: int = 51
    weekly_prize: float = WEEKLY_PRIZE
    season_prizes: Sequence[float] = SEASON_PRIZES
    skills_matched: int = field(default=0)

    def __post_init__(self):
        if not self.templates:
            raise ValueError(
                "No historical week templates to sample from -- need "
                "PickEmCache<year>/ data on disk"
            )
        self.points = np.asarray(self.points, dtype=float)
        if self.active is None:
            self.active = np.ones(len(self.names), dtype=bool)
        if self.play_prob is None:
            self.play_prob = np.ones(len(self.names))
        if self.ability_mean is None:
            self.ability_mean = np.zeros(len(self.names))
        if self.ability_post_sd is None:
            self.ability_post_sd = np.zeros(len(self.names))
        if self.banked_weekly_wins is None:
            self.banked_weekly_wins = np.zeros(len(self.names))

    @classmethod
    def from_standings(
        cls,
        standings: pd.DataFrame,
        skills_by_name: Optional[Dict],
        templates: List[WeekSpec],
        completed_weeks: Optional[int] = None,
        inactive_after: int = 2,
        **kwargs,
    ) -> "SeasonSimulator":
        """Build from ``YahooPickEm.standings`` (``get_standings()``).

        Banked points are the sum of the first ``completed_weeks`` weekly columns --
        *not* ``total_points``, which also holds any partial score from the week in
        progress (that week is re-simulated in full). ``completed_weeks=None``
        infers it with :func:`infer_completed_weeks`.
        """
        if standings is None or len(standings) == 0:
            raise ValueError("Empty standings")
        weekly = weekly_points_matrix(standings)
        if completed_weeks is None:
            completed_weeks = infer_completed_weeks(weekly)
        won = standings["weeks_won"].to_numpy(dtype=float) if "weeks_won" in standings else None
        return cls.from_weekly(
            standings["player_name"].tolist(),
            weekly[:, : int(completed_weeks)],
            skills_by_name,
            templates,
            inactive_after=inactive_after,
            banked_weekly_wins=won,
            **kwargs,
        )

    @classmethod
    def from_weekly(
        cls,
        names: List[str],
        weekly_done: np.ndarray,
        skills_by_name: Optional[Dict],
        templates: List[WeekSpec],
        inactive_after: int = 2,
        ability_sd: float = ABILITY_SD,
        **kwargs,
    ) -> "SeasonSimulator":
        """Build from a ``[P, completed_weeks]`` matrix of weekly points."""
        completed_weeks = weekly_done.shape[1]
        if not 0 <= completed_weeks < kwargs.get("total_weeks", SEASON_WEEKS):
            raise ValueError(f"{completed_weeks} completed weeks leaves nothing to simulate")
        skills, matched = build_skill_array(names, skills_by_name)
        a_mean, a_sd = estimate_ability(weekly_done, ability_sd)
        return cls(
            names=list(names),
            points=weekly_done.sum(axis=1),
            skills=skills,
            templates=templates,
            completed_weeks=completed_weeks,
            active=infer_active(weekly_done, inactive_after),
            play_prob=participation_rates(weekly_done),
            ability_mean=a_mean,
            ability_post_sd=a_sd,
            skills_matched=matched,
            **kwargs,
        )

    def draw_ability(self, rng: np.random.Generator, num_sims: int) -> np.ndarray:
        """``[S, P]`` persistent per-season ability draws (points per week)."""
        return self.ability_mean[None, :] + self.ability_post_sd[None, :] * rng.standard_normal(
            (num_sims, len(self.names))
        )

    # -- core -----------------------------------------------------------------

    def simulate_weeks(
        self,
        first_week: int,
        rng: np.random.Generator,
        num_sims: Optional[int] = None,
        ability: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Simulate weeks ``first_week .. total_weeks``.

        ``ability`` is an ``[S, P]`` draw from :meth:`draw_ability`; pass the same
        draw to every call that belongs to one simulated season (default: draw one).

        Returns ``(added_points[S, P], weekly_wins[S, P])`` -- points scored over
        those weeks and (tie-split) weekly prizes won.
        """
        S = num_sims or self.num_sims
        P = len(self.names)
        added = np.zeros((S, P))
        wins = np.zeros((S, P))
        act = np.flatnonzero(self.active)
        if len(act) == 0:
            return added, wins
        skills = self.skills[act]
        play_prob = self.play_prob[act]
        if ability is None:
            ability = self.draw_ability(rng, S)
        ability = ability[:, act]
        for _ in range(first_week, self.total_weeks + 1):
            which = rng.integers(len(self.templates), size=S)
            for t in np.unique(which):
                rows = np.flatnonzero(which == t)
                spec = self.templates[t]
                outcomes = sample_outcomes(spec.p_a, len(rows), rng)
                sc = simulate_week_scores(rng, spec, skills, outcomes, noise_scale=self.noise_scale)
                sc = np.maximum(sc + ability[rows], 0.0)
                sc = sc * (rng.random(sc.shape) < play_prob[None, :])  # skipped week = 0
                added[np.ix_(rows, act)] += sc
                wins[np.ix_(rows, act)] += _win_credit(sc)
        return added, wins

    def run(self) -> pd.DataFrame:
        """P(1st..4th), expected dollars and expected final points per player."""
        rng = np.random.default_rng(self.seed)
        added, wins = self.simulate_weeks(self.completed_weeks + 1, rng)
        totals = self.points[None, :] + added
        live = self.active
        rel = totals - totals[:, live].mean(axis=1, keepdims=True)  # vs. the active field
        dollars, place = season_payouts(totals, self.season_prizes)
        weekly_dollars = wins.mean(axis=0) * self.weekly_prize
        season_dollars = dollars.mean(axis=0)
        out = pd.DataFrame(
            {
                "player": self.names,
                "points": self.points,
                "active": self.active,
                "exp_final_points": totals.mean(axis=0),
                "sd_final_points": totals.std(axis=0),
                "sd_rel_points": rel.std(axis=0),
            }
        )
        for j in range(len(self.season_prizes)):
            out[f"p_place_{j + 1}"] = place[:, :, j].mean(axis=0)
        out["p_top4"] = place.sum(axis=2).mean(axis=0)
        out["exp_weekly_wins"] = wins.mean(axis=0)
        out["exp_weekly_dollars"] = weekly_dollars
        out["exp_season_dollars"] = season_dollars
        out["exp_dollars"] = weekly_dollars + season_dollars
        out["banked_weekly_dollars"] = self.banked_weekly_wins * self.weekly_prize
        out["current_rank"] = (-out["points"]).rank(method="min").astype(int)
        return out.sort_values("exp_dollars", ascending=False, ignore_index=True)


# ---------------------------------------------------------------------------
# Standings helpers
# ---------------------------------------------------------------------------


def weekly_points_matrix(standings: pd.DataFrame) -> np.ndarray:
    """``[P, W]`` per-week points from the ``week_<N>_points`` columns."""
    cols = sorted(
        (c for c in standings.columns if re.fullmatch(r"week_\d+_points", c)),
        key=lambda c: int(c.split("_")[1]),
    )
    if not cols:
        raise ValueError("standings has no week_<N>_points columns")
    return standings[cols].to_numpy(dtype=float)


def infer_completed_weeks(weekly: np.ndarray, min_frac: float = 0.8) -> int:
    """Count leading weeks that look final.

    Yahoo's weekly-performance table lists every week through the current one, and
    the current week holds zeros / partial scores until it ends. League totals
    swing a lot week to week (chalky weeks average ~95 points, upset-heavy ones
    ~60), so totals can't tell a short week from an unfinished one. Instead a
    week counts as final when nearly as many players have scored in it as in the
    typical earlier week *and* the average scorer is no lower than ``min_frac``
    of the lowest earlier average. Monday night in progress is the ambiguous
    case -- pass ``completed_weeks`` explicitly when it matters.
    """
    done = 0
    counts: List[int] = []
    means: List[float] = []
    for w in range(weekly.shape[1]):
        col = weekly[:, w]
        scored = col[col > 0]
        n_scored = len(scored)
        mean = float(scored.mean()) if n_scored else 0.0
        if n_scored == 0:
            break
        if counts and (n_scored < min_frac * np.median(counts) or mean < min_frac * min(means)):
            break
        if not counts and mean < 30:  # first week can't be judged against history
            break
        counts.append(n_scored)
        means.append(mean)
        done = w + 1
    return done


# Chance a player submits picks in a given future week, by what they've done so far.
# Fit to the 2024-2026 seasons (scripts/calibrate_season_sim.py): the "never missed"
# rate rests on ~1,500 player-cutoffs; the miss states on only ~20 each, so treat
# those as rough.
PLAY_RELIABLE = 0.983  # no missed week since first appearing
PLAY_AFTER_MISS = 0.5  # missed the most recent week (but not the one before)
PLAY_AFTER_OLD_MISS = 0.9  # missed earlier, but played the most recent week


def participation_rates(weekly_done: np.ndarray) -> np.ndarray:
    """Per-player chance of submitting picks in a given future week.

    Missing a week is far from random: players who have never missed skip ~1.7%
    of later weeks, but one who just missed a week skips about half and one who
    missed two in a row has usually quit (:func:`infer_active` marks those
    inactive). A zero before a player's first scored week is a late joiner, not a
    miss.
    """
    P, W = weekly_done.shape
    p = np.full(P, PLAY_RELIABLE)
    if W == 0:
        return p
    played = weekly_done > 0
    for i in range(P):
        if not played[i].any():
            continue
        row = played[i, int(np.argmax(played[i])) :]  # from first appearance
        if row.all():
            continue
        p[i] = PLAY_AFTER_OLD_MISS if row[-1] else PLAY_AFTER_MISS
    return p


def estimate_ability(
    weekly_done: np.ndarray, ability_sd: float = ABILITY_SD, weekly_sd: float = WEEKLY_SD
) -> Tuple[np.ndarray, np.ndarray]:
    """Posterior (mean, SD) of each player's persistent points-per-week edge.

    A player's average margin over the week's mean scorer, across the weeks they
    played, is a noisy read on a true edge ``~ N(0, ability_sd^2)``: it carries
    ``weekly_sd^2 / n`` of noise, so it is shrunk toward 0 by
    ``B = tau^2 / (tau^2 + sigma^2 / n)``.
    """
    P, W = weekly_done.shape
    if W == 0 or ability_sd <= 0:
        return np.zeros(P), np.zeros(P)
    played = weekly_done > 0
    week_mean = np.array(
        [weekly_done[played[:, j], j].mean() if played[:, j].any() else 0.0 for j in range(W)]
    )
    rel = np.where(played, weekly_done - week_mean[None, :], 0.0)
    n = played.sum(axis=1)
    seen = n > 0
    avg = np.zeros(P)
    avg[seen] = rel[seen].sum(axis=1) / n[seen]
    shrink = ability_sd**2 / (ability_sd**2 + weekly_sd**2 / np.maximum(n, 1))
    shrink = np.where(seen, shrink, 0.0)
    return shrink * avg, ability_sd * np.sqrt(1.0 - shrink)


def estimate_ability_sd(
    weekly_matrices: Sequence[np.ndarray],
    min_played: float = 0.7,
) -> Tuple[float, float]:
    """Method-of-moments ``(ability_sd, weekly_sd)`` from finished seasons.

    ``weekly_sd`` is the typical within-player week-to-week SD of the margin over
    the week's mean scorer; ``ability_sd`` is what is left of the spread of
    player averages once that sampling noise (``weekly_sd^2 / n``) is removed.
    Players who played under ``min_played`` of the weeks are skipped.
    """
    taus, sigmas = [], []
    for weekly in weekly_matrices:
        P, W = weekly.shape
        played = weekly > 0
        week_mean = np.array([weekly[played[:, j], j].mean() for j in range(W)])
        rel = np.where(played, weekly - week_mean[None, :], np.nan)
        ok = played.mean(axis=1) >= min_played
        n = played.sum(axis=1)
        sigma2 = float(np.nanmean(np.nanvar(rel[ok], axis=1)))
        pm = np.nanmean(rel[ok], axis=1)
        tau2 = max(0.0, float(np.var(pm) - np.mean(sigma2 / n[ok])))
        taus.append(tau2)
        sigmas.append(sigma2)
    return float(np.sqrt(np.mean(taus))), float(np.sqrt(np.mean(sigmas)))


def infer_active(weekly_done: np.ndarray, inactive_after: int = 2) -> np.ndarray:
    """False for players who scored zero in each of the last ``inactive_after``
    completed weeks (they've stopped submitting picks)."""
    P, W = weekly_done.shape
    if W < inactive_after or inactive_after <= 0:
        return np.ones(P, dtype=bool)
    return weekly_done[:, W - inactive_after :].sum(axis=1) > 0


# ---------------------------------------------------------------------------
# Expected dollars for a candidate slate this week
# ---------------------------------------------------------------------------


def slate_season_equity(
    season: SeasonSimulator,
    me: int,
    spec: WeekSpec,
    outcomes: np.ndarray,
    my_scores: Dict[str, np.ndarray],
    frozen: Optional[FrozenPicks] = None,
    ref: Optional[str] = None,
) -> Dict[str, Tuple[float, float, float]]:
    """Season-prize equity of each candidate slate for the week in progress.

    Everything except ``me``'s own slate is held common across candidates:

    * the week's ``outcomes`` ``[K, n]`` are the very draws that produced
      ``my_scores``, and every rival's score this week is simulated on them -- so
      the correlation between your week and theirs (a chalk slate that wins big
      also lifts every chalk rival) is preserved;
    * the weeks after this one are simulated once (``K`` sims) and reused;
    * ``frozen`` supplies rivals' real picks on locked games (midweek).

    Args:
        season: the :class:`SeasonSimulator`; ``season.completed_weeks`` is the
            last week *before* this one and ``season.points`` the matching bank.
        me: index of the player in ``season.names``.
        spec: this week's slate (side A = home).
        outcomes: ``[K, n]`` bool draws of side-A wins.
        my_scores: ``{label: [K] week score under that slate}``.
        frozen: real locked picks for every *other* player, rows in
            ``season.names`` order with ``me`` removed.

        ref: label the standard errors are measured against (default: first).

    Returns:
        ``{label: (season_dollars, p_top4, diff_se)}`` -- expected season-prize
        dollars and probability of a paying finish, each averaged over the ``K``
        draws, and the standard error of ``season_dollars`` minus ``ref``'s. The
        draws are shared, so that paired error is far tighter than the error on
        either level -- it is the one to judge a between-slate gap by.
    """
    K = outcomes.shape[0]
    P = len(season.names)
    rng = np.random.default_rng(season.seed + 1)
    this_week = season.completed_weeks + 1
    ability = season.draw_ability(rng, K)  # one persistent edge per simulated season

    if this_week < season.total_weeks:
        future, _ = season.simulate_weeks(this_week + 1, rng, num_sims=K, ability=ability)
    else:
        future = np.zeros((K, P))

    others = np.array([i for i in range(P) if i != me])
    act = np.array([i for i in others if season.active[i]], dtype=int)
    week_pts = np.zeros((K, P))
    if len(act):
        f = None
        if frozen is not None:
            keep = np.array([list(others).index(i) for i in act])
            f = FrozenPicks(frozen.mask, frozen.pick_a[keep], frozen.points[keep])
        # Rivals' edge applies only to the still-open share of the slate; your own
        # score this week comes from the actual slate, so it gets none.
        open_share = 1.0 if f is None else float((~f.mask).mean())
        sc = np.maximum(
            simulate_week_scores(
                rng, spec, season.skills[act], outcomes, f, noise_scale=season.noise_scale
            )
            + open_share * ability[:, act],
            0.0,
        )
        plays = rng.random(sc.shape) < season.play_prob[act][None, :]
        if f is not None:
            plays |= (f.points[:, f.mask] > 0).any(axis=1)[None, :]
        week_pts[:, act] = sc * plays

    rivals = season.points[others][None, :] + week_pts[:, others] + future[:, others]
    me_base = season.points[me] + future[:, me]
    ref = ref if ref in my_scores else next(iter(my_scores))
    per_sim = {
        label: prize_for_one(me_base + mine, rivals, season.season_prizes)
        for label, mine in my_scores.items()
    }
    out = {}
    for label, mine in my_scores.items():
        top4 = prize_for_one(me_base + mine, rivals, (1.0,) * len(season.season_prizes)).mean()
        diff = per_sim[label] - per_sim[ref]
        out[label] = (
            float(per_sim[label].mean()),
            float(top4),
            float(diff.std(ddof=1) / np.sqrt(K)) if K > 1 else 0.0,
        )
    return out


# ---------------------------------------------------------------------------
# Past seasons (for backtesting)
# ---------------------------------------------------------------------------


@dataclass
class SeasonHistory:
    """A finished (or partly finished) season rebuilt from ``PickEmCache<year>/``.

    ``weekly[P, W]`` is each player's points per week (0 = no picks submitted)
    over ``weeks`` -- only weeks whose pages were cached *after* the games were
    played (some cached pages are stale, so a season may have gaps at its ends);
    ``fav_spread`` / ``fav_won`` list every decided game's spread and whether the
    favorite won, for checking the spread -> win-probability map.
    """

    year: int
    weeks: List[int]
    names: List[str]
    weekly: np.ndarray
    fav_spread: np.ndarray
    fav_won: np.ndarray


def load_season_history(year: int, root: str = ".") -> SeasonHistory:
    from .analyze_player_skills import parse_week_data

    cache_dir = Path(root) / f"PickEmCache{year}"
    per_week: Dict[int, Dict[str, int]] = {}
    spreads: List[float] = []
    fav_won: List[bool] = []
    for week, path in _week_files(cache_dir):
        games, players = _quiet_parse(parse_week_data, path)
        if not games or not players or _unresolved(games) > MAX_UNRESOLVED:
            continue  # unparseable, or cached before the games were played
        if sum(p["total_points"] > 0 for p in players) < 10:
            continue
        per_week[week] = {p["player_name"]: p["total_points"] for p in players}
        for g in games:
            if g.get("winner"):
                spreads.append(g.get("spread", 0.0))
                fav_won.append(g["winner"] == g["favorite"])
    if not per_week:
        raise ValueError(f"No parsed weeks in {cache_dir}")
    weeks = sorted(per_week)
    names = sorted({n for w in per_week.values() for n in w})
    weekly = np.zeros((len(names), len(weeks)))
    index = {n: i for i, n in enumerate(names)}
    for j, week in enumerate(weeks):
        for n, v in per_week[week].items():
            weekly[index[n], j] = v
    return SeasonHistory(year, weeks, names, weekly, np.array(spreads), np.array(fav_won))
