"""
Crowd pick % vs. line vs. result: a log and a first-pass signal check.

Hypothesis (issue #50): Yahoo's crowd pick percentages carry some information about
game outcomes beyond what the betting line already prices in. Most likely they do
not -- this module exists to collect the data and test it, not to assert it.

Yahoo publishes the league-wide pick distribution for four kinds of pool, and the
crowds differ a lot (a 10-point favorite is picked by ~97% straight up but only
~65% against the spread):

    ``n``  no spread, no confidence      ``s``   spread only
    ``c``  confidence only               ``sc``  spread and confidence

The forward log (``snapshot_week`` / ``append_records`` / ``settle_log``) records all
four per game, then settles final scores from ESPN so spread pools can be tested
against *covering*. ``history_from_cache`` adds the ``PickEmCache<year>/`` weeks --
confidence pool only, straight-up winners only, and the line/crowd are whatever Yahoo
showed when the page was saved, not necessarily pre-kickoff.

Everything reduces to one tidy per-game-per-pool table (``load_log`` /
``history_from_cache``) that ``crowd_signal_report`` consumes.
"""

import contextlib
import http.cookiejar
import io
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup

from .yahoo_pickem_scraper import parse_kickoff

POOLS = ("n", "s", "c", "sc")
POOL_LABELS = {
    "n": "no spread/no confidence",
    "s": "spread only",
    "c": "confidence only",
    "sc": "spread + confidence",
}
SPREAD_POOLS = ("s", "sc")

SPREAD_SLOPE = 0.031  # same spread -> win prob conversion the scraper uses
BUCKET_EDGES = (0.0, 0.6, 0.75, 0.85, 0.92, 1.0)
ATS_BUCKET_EDGES = (0.5, 0.6, 0.7, 1.0)  # crowd strength: share on its preferred side
BREAKEVEN_ATS = 0.524  # win rate needed to beat -110 vig

DISTRIBUTION_URL = "https://football.fantasysports.yahoo.com/pickem/pickdistribution"
ESPN_SCOREBOARD = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard"
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
)

GAME_KEY = ("season", "week", "favorite", "underdog")
POOL_GAME_KEY = ("season", "week", "pool", "favorite", "underdog")

# ESPN abbreviations that differ from Yahoo's (compared case-insensitively)
TEAM_ALIASES = {"wsh": "was", "arz": "ari", "jac": "jax", "la": "lar"}


def current_season(now: Optional[datetime] = None) -> int:
    """NFL season a date falls in (January-March belong to the previous season)."""
    now = now or datetime.now()
    return now.year - 1 if now.month <= 3 else now.year


# ---------------------------------------------------------------------------
# Yahoo pick distribution (all four pool types)
# ---------------------------------------------------------------------------


def make_session(cookies_file: str) -> requests.Session:
    """A requests session carrying the exported Yahoo cookies."""
    jar = http.cookiejar.MozillaCookieJar(cookies_file)
    jar.load(ignore_discard=True, ignore_expires=True)
    session = requests.Session()
    session.cookies = jar  # type: ignore[assignment]
    session.headers.update({"User-Agent": USER_AGENT})
    return session


def fetch_distribution(session: requests.Session, week: int, pool: str) -> str:
    """Raw HTML of the league-wide pick distribution for one pool type."""
    if pool not in POOLS:
        raise ValueError(f"unknown pool {pool!r}; expected one of {POOLS}")
    params = {"gid": "", "week": week, "type": "" if pool == "n" else pool}
    resp = session.get(DISTRIBUTION_URL, params=params, timeout=30)
    resp.raise_for_status()
    return resp.text


def parse_distribution(html: str, season: int) -> List[Dict[str, Any]]:
    """One dict per game: teams, spread, crowd %s and kickoff.

    Unlike ``YahooPickEm.get_pick_distribution`` this tolerates the non-confidence
    pools, whose "Avg. Confidence" cells read ``N/A``. A game with no posted line
    gets ``spread=None``.
    """
    soup = BeautifulSoup(html, "html.parser")
    games = []
    for g in soup.find_all("div", class_="ysf-matchup-dist"):
        abbr = [th.text.strip() for th in g.find_all("th")]
        pct = g.find_all("dd", class_="percent")
        if len(abbr) < 2 or len(pct) < 2:
            continue
        try:
            fav_pct = float(pct[0].text.strip().rstrip("% "))
            und_pct = float(pct[1].text.strip().rstrip("% "))
        except ValueError:
            continue

        spread = None
        for td in g.select("div.ft td"):
            m = re.fullmatch(r"([\d.]+) pts", td.text.strip())
            if m:
                spread = float(m.group(1))
                break

        kickoff = None
        hd = g.find("div", class_="hd")
        if hd:
            try:
                kickoff = parse_kickoff(hd.text, season).isoformat()
            except ValueError:
                pass

        teams = g.find_all("dd", class_="team")
        games.append(
            {
                "favorite": abbr[0],
                "underdog": abbr[-1],
                "spread": spread,
                "favorite_pick_pct": fav_pct,
                "underdog_pick_pct": und_pct,
                "kickoff_time": kickoff,
                # an "@" marks the home team
                "favorite_home": teams[0].text.strip().startswith("@") if teams else None,
            }
        )
    return games


def snapshot_week(
    session: requests.Session,
    week: int,
    season: int,
    pools: Sequence[str] = POOLS,
    snapshot_time: Optional[datetime] = None,
    pause: float = 1.0,
) -> List[Dict[str, Any]]:
    """Snapshot records (one per game per pool) for a week."""
    import time

    when = snapshot_time or datetime.now().astimezone()  # tz-aware: compared to kickoff later
    stamp = when.isoformat(timespec="seconds")
    records: List[Dict[str, Any]] = []
    for i, pool in enumerate(pools):
        if i and pause:
            time.sleep(pause)
        for g in parse_distribution(fetch_distribution(session, week, pool), season):
            head = {"kind": "snapshot", "snapshot_time": stamp, "season": season, "week": week}
            records.append({**head, "pool": pool, **g})
    return records


# ---------------------------------------------------------------------------
# Results (ESPN final scores)
# ---------------------------------------------------------------------------


def norm_team(abbr: str) -> str:
    key = abbr.strip().casefold()
    return TEAM_ALIASES.get(key, key)


def parse_espn_week(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Home/away scores and completion for each game in an ESPN scoreboard payload."""
    out = []
    for ev in payload.get("events", []):
        comp = ev["competitions"][0]
        sides = {c["homeAway"]: c for c in comp["competitors"]}
        if "home" not in sides or "away" not in sides:
            continue
        out.append(
            {
                "home": norm_team(sides["home"]["team"]["abbreviation"]),
                "away": norm_team(sides["away"]["team"]["abbreviation"]),
                "home_score": int(sides["home"].get("score") or 0),
                "away_score": int(sides["away"].get("score") or 0),
                "completed": bool(ev["status"]["type"]["completed"]),
            }
        )
    return out


def fetch_espn_week(season: int, week: int) -> List[Dict[str, Any]]:
    resp = requests.get(
        ESPN_SCOREBOARD,
        params={"dates": season, "seasontype": 2, "week": week},
        timeout=30,
    )
    resp.raise_for_status()
    return parse_espn_week(resp.json())


def result_records(
    games: Iterable[Tuple[str, str]],
    espn_games: Iterable[Dict[str, Any]],
    season: int,
    week: int,
    settled_time: Optional[datetime] = None,
) -> List[Dict[str, Any]]:
    """Result records for the ``(favorite, underdog)`` games ESPN reports as final."""
    final = {frozenset((e["home"], e["away"])): e for e in espn_games if e["completed"]}
    stamp = (settled_time or datetime.now().astimezone()).isoformat(timespec="seconds")
    records = []
    for fav, und in games:
        e = final.get(frozenset((norm_team(fav), norm_team(und))))
        if e is None:
            continue
        fav_home = norm_team(fav) == e["home"]
        records.append(
            {
                "kind": "result",
                "settled_time": stamp,
                "season": season,
                "week": week,
                "favorite": fav,
                "underdog": und,
                "favorite_score": e["home_score"] if fav_home else e["away_score"],
                "underdog_score": e["away_score"] if fav_home else e["home_score"],
            }
        )
    return records


# ---------------------------------------------------------------------------
# The log
# ---------------------------------------------------------------------------


def append_records(path: str, records: Iterable[Dict[str, Any]]) -> int:
    """Append records to a JSONL log, one line each. Returns how many were written."""
    n = 0
    with open(path, "a") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
            n += 1
    return n


def read_log(path: str) -> List[Dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        return []
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]


def settle_log(
    path: str,
    season: int,
    weeks: Optional[Iterable[int]] = None,
    fetch: Callable[[int, int], List[Dict[str, Any]]] = fetch_espn_week,
) -> int:
    """Append result records for logged games that ESPN now reports as final.

    Only games without a result yet are looked up, and only weeks that still have
    some. Returns the number of results written.
    """
    rows = read_log(path)
    wanted = set(weeks) if weeks is not None else None
    settled = {
        (r["season"], r["week"], r["favorite"], r["underdog"]) for r in rows if _is_result(r)
    }
    pending: Dict[int, set] = {}
    for r in rows:
        if _is_result(r) or r["season"] != season:
            continue
        if wanted is not None and r["week"] not in wanted:
            continue
        key = (r["season"], r["week"], r["favorite"], r["underdog"])
        if key not in settled:
            pending.setdefault(r["week"], set()).add((r["favorite"], r["underdog"]))
    written = 0
    for week in sorted(pending):
        written += append_records(
            path, result_records(sorted(pending[week]), fetch(season, week), season, week)
        )
    return written


def _is_result(rec: Dict[str, Any]) -> bool:
    return rec.get("kind") == "result"


def load_log(path: str) -> pd.DataFrame:
    """One row per game per pool from a log, with final scores where settled.

    Takes the last snapshot at or before kickoff for the line/crowd (``pregame`` says
    whether one existed; otherwise the last snapshot is used and the crowd may have
    seen the result).
    """
    rows = read_log(path)
    snaps = pd.DataFrame([r for r in rows if not _is_result(r)])
    if snaps.empty:
        return pd.DataFrame()
    snaps = snaps.sort_values("snapshot_time")
    out = []
    for _, grp in snaps.groupby(list(POOL_GAME_KEY), sort=False):
        kick = grp["kickoff_time"]
        pregame = False
        row = grp.iloc[-1]
        if kick.notna().all():
            snap_t = pd.to_datetime(grp["snapshot_time"], utc=True)
            kick_t = pd.to_datetime(kick, utc=True)
            early = grp[snap_t <= kick_t]
            if len(early):
                pregame, row = True, early.iloc[-1]
        rec = row.to_dict()
        rec["pregame"] = pregame
        out.append(rec)
    df = pd.DataFrame(out).drop(columns=["kind"], errors="ignore")

    results = pd.DataFrame([r for r in rows if _is_result(r)])
    if results.empty:
        df["favorite_score"] = np.nan
        df["underdog_score"] = np.nan
        return df
    results = results.sort_values("settled_time").drop_duplicates(list(GAME_KEY), keep="last")
    keep = list(GAME_KEY) + ["favorite_score", "underdog_score"]
    return df.merge(results[keep], on=list(GAME_KEY), how="left")


def _week_files(cache_dir: Path) -> List[Tuple[int, Path]]:
    found = []
    for p in cache_dir.glob("confidence_picks_week*.html"):
        m = re.search(r"week(\d+)", p.name)
        if m:
            found.append((int(m.group(1)), p))
    return sorted(found)


def history_from_cache(root: str = ".") -> pd.DataFrame:
    """Every cached ``PickEmCache<year>/`` game as a confidence-pool row.

    Outcomes are straight-up winners only (no scores in the cached pages).
    """
    from .analyze_player_skills import parse_week_data

    rows = []
    for cache_dir in sorted(Path(root).glob("PickEmCache*")):
        m = re.fullmatch(r"PickEmCache(\d{4})", cache_dir.name)
        if not m:
            continue
        for week, path in _week_files(cache_dir):
            with contextlib.redirect_stdout(io.StringIO()):
                games, _ = parse_week_data(path)
            for g in games or []:
                rows.append(
                    {
                        "season": int(m.group(1)),
                        "week": week,
                        "pool": "c",
                        "favorite": g.get("favorite"),
                        "underdog": g.get("underdog"),
                        "spread": g.get("spread"),
                        "favorite_pick_pct": g.get("favorite_pick_pct"),
                        "winner": g.get("winner"),
                    }
                )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Keep games with a real line and crowd %, add outcome and analysis columns.

    ``fav_won`` (straight up) and ``fav_covered`` (against the spread, pushes are NaN)
    are NaN where the outcome isn't known; ``su_games`` / ``ats_games`` filter on them.
    """
    if df.empty:
        return df
    d = df.dropna(subset=["spread", "favorite_pick_pct"]).copy()
    d = d[d["spread"] > 0]  # drop pick'ems / "Off" lines: no favorite to speak of
    if "pool" not in d:
        d["pool"] = "c"

    won = pd.Series(np.nan, index=d.index)
    covered = pd.Series(np.nan, index=d.index)
    if "winner" in d:
        known = d["winner"].notna()
        won[known] = (d["winner"] == d["favorite"])[known].astype(float)
    if "favorite_score" in d:
        margin = d["favorite_score"] - d["underdog_score"]
        ok = margin.notna()
        won[ok] = np.where(margin[ok] > 0, 1.0, np.where(margin[ok] < 0, 0.0, np.nan))
        diff = margin - d["spread"]
        covered[ok] = np.where(diff[ok] > 0, 1.0, np.where(diff[ok] < 0, 0.0, np.nan))
        d["fav_margin"] = margin
    d["fav_won"] = won
    d["fav_covered"] = covered
    d["implied"] = (0.5 + SPREAD_SLOPE * d["spread"]).clip(0.01, 0.99)
    d["crowd"] = d["favorite_pick_pct"] / 100.0
    return d


def su_games(d: pd.DataFrame) -> pd.DataFrame:
    return d[d["fav_won"].notna()] if not d.empty else d


def ats_games(d: pd.DataFrame) -> pd.DataFrame:
    return d[d["fav_covered"].notna()] if not d.empty else d


def bucket_table(
    d: pd.DataFrame,
    outcome: str = "fav_won",
    edges: Iterable[float] = BUCKET_EDGES,
    baseline: Optional[str] = "implied",
) -> pd.DataFrame:
    """Favorite ``outcome`` rate (and edge over ``baseline``), bucketed by crowd %."""
    b = pd.cut(d["crowd"], list(edges), include_lowest=True)
    spec = dict(n=(outcome, "size"), crowd=("crowd", "mean"), won=(outcome, "mean"))
    if baseline:
        spec["implied"] = (baseline, "mean")
    g = d.groupby(b, observed=True).agg(**spec)
    if baseline:
        g["edge"] = g["won"] - g["implied"]
    g["se"] = np.sqrt(g["won"] * (1 - g["won"]) / g["n"])
    return g


def _logit(p: np.ndarray) -> np.ndarray:
    return np.log(p / (1 - p))


def _logit_fit(
    X: np.ndarray, y: np.ndarray, names: Sequence[str]
) -> Dict[str, Tuple[float, float]]:
    from scipy.optimize import minimize

    def nll(b):
        z = X @ b
        return -(y * z - np.logaddexp(0, z)).sum()

    fit = minimize(nll, np.zeros(X.shape[1]), method="BFGS")
    p = 1 / (1 + np.exp(-(X @ fit.x)))
    cov = np.linalg.inv((X * (p * (1 - p))[:, None]).T @ X)
    se = np.sqrt(np.diag(cov))
    return {n: (float(c), float(s)) for n, c, s in zip(names, fit.x, se)}


def logit_fit(d: pd.DataFrame) -> Dict[str, Tuple[float, float]]:
    """``fav_won ~ logit(implied) + logit(crowd)`` as ``{term: (coef, se)}``.

    If crowd % adds information beyond the line, its coefficient is clearly > 0
    (crowd right where the line is wrong) or < 0 (fade the crowd). Near zero within
    its standard error means no detectable signal.
    """
    X = np.c_[
        np.ones(len(d)), _logit(d["implied"].values), _logit(d["crowd"].clip(0.02, 0.98).values)
    ]
    return _logit_fit(X, d["fav_won"].values.astype(float), ("const", "line", "crowd"))


def ats_logit_fit(d: pd.DataFrame) -> Dict[str, Tuple[float, float]]:
    """``fav_covered ~ logit(crowd)``: > 0 means the spread crowd is right on covers."""
    X = np.c_[np.ones(len(d)), _logit(d["crowd"].clip(0.02, 0.98).values)]
    return _logit_fit(X, d["fav_covered"].values.astype(float), ("const", "crowd"))


def min_detectable_edge(n: int, p: float = 0.7, z: float = 1.96 + 0.84) -> float:
    """Rough smallest win-rate edge detectable at 80% power with ``n`` games."""
    return z * float(np.sqrt(p * (1 - p) / n)) if n else float("nan")


def pool_comparison(d: pd.DataFrame) -> List[str]:
    """Average favorite % per pool on the games every pool has, and its tie to the line."""
    if d.empty or d["pool"].nunique() < 2:
        return []
    wide = d.pivot_table(
        index=list(GAME_KEY) + ["spread"], columns="pool", values="favorite_pick_pct"
    ).dropna()
    pools = [p for p in POOLS if p in wide.columns]
    if len(pools) < 2 or wide.empty:
        return []
    spread = wide.index.get_level_values("spread").to_numpy(dtype=float)
    lines = [f"Pool comparison ({len(wide)} games in every pool):"]
    for p in pools:
        corr = float(np.corrcoef(spread, wide[p].to_numpy())[0, 1])
        lines.append(
            f"  {p:<3} {POOL_LABELS[p]:<24} avg {wide[p].mean():>5.1f}% on favorite, "
            f"corr with spread {corr:+.2f}"
        )
    return lines


def _su_section(d: pd.DataFrame) -> List[str]:
    su = su_games(d[d["pool"] == "c"])
    if su.empty:
        return ["No settled confidence-pool games with a line and crowd % to analyze."]
    lines = [
        "Straight up, confidence pool crowd vs. spread-implied win probability:",
        f"  {len(su)} settled games ({su['season'].nunique()} season(s)); favorite won "
        f"{su['fav_won'].mean():.1%} vs {su['implied'].mean():.1%} implied by the line, "
        f"crowd backed the favorite {su['crowd'].mean():.1%} of the time",
        "",
        f"  {'Crowd on fav':<14} {'n':>5} {'crowd':>7} {'implied':>8} "
        f"{'won':>7} {'edge':>7} {'se':>6}",
    ]
    for rng, r in bucket_table(su).iterrows():
        lines.append(
            f"  {str(rng):<14} {int(r['n']):>5} {r['crowd']:>7.1%} {r['implied']:>8.1%} "
            f"{r['won']:>7.1%} {r['edge']:>+7.1%} {r['se']:>6.1%}"
        )
    fit = logit_fit(su)
    c, s = fit["crowd"]
    lines += [
        "",
        f"  Logit: crowd coef {c:+.2f} (se {s:.2f}), line coef {fit['line'][0]:+.2f} "
        f"(se {fit['line'][1]:.2f}) -> crowd adds "
        + ("a detectable signal" if abs(c) > 2 * s else "no detectable signal beyond the line"),
        f"  Power: with {len(su)} games only edges over ~{min_detectable_edge(len(su)):.0%} "
        "are detectable; real betting edges are usually a few points.",
        "  (The favorite beating its implied rate in most buckets mostly reflects the crude "
        "linear spread->probability conversion, not crowd signal.)",
    ]
    return lines


def _ats_section(d: pd.DataFrame) -> List[str]:
    lines: List[str] = []
    for pool in SPREAD_POOLS:
        a = ats_games(d[d["pool"] == pool])
        if a.empty:
            continue
        n = len(a)
        crowd_side_fav = a["crowd"] > 0.5
        decisive = a[a["crowd"] != 0.5]
        right = (decisive["fav_covered"] == (decisive["crowd"] > 0.5).astype(float)).mean()
        se = float(np.sqrt(0.25 / len(decisive))) if len(decisive) else float("nan")
        lines += [
            "",
            f"Against the spread, {pool} pool ({POOL_LABELS[pool]}): {n} games with a result "
            f"({int(crowd_side_fav.sum())} crowd-on-favorite), favorite covered "
            f"{a['fav_covered'].mean():.1%}",
            f"  The crowd's side covered {right:.1%} (se {se:.1%} around 50%; "
            f"{BREAKEVEN_ATS:.1%} breaks even at -110; fading it covers {1 - right:.1%})",
        ]
        a = a.assign(strength=np.maximum(a["crowd"], 1 - a["crowd"]))
        a = a.assign(crowd_right=np.where(a["crowd"] > 0.5, a["fav_covered"], 1 - a["fav_covered"]))
        a = a[a["crowd"] != 0.5]
        b = pd.cut(a["strength"], list(ATS_BUCKET_EDGES), include_lowest=True)
        g = a.groupby(b, observed=True)["crowd_right"].agg(["size", "mean"])
        for rng, r in g.iterrows():
            lines.append(
                f"    crowd {str(rng):<12} n={int(r['size']):>4}  side covered {r['mean']:.1%}"
            )
        if len(a) > 10:
            c, s = ats_logit_fit(a)["crowd"]
            lines.append(
                f"  Logit: covers ~ crowd coef {c:+.2f} (se {s:.2f}) -> "
                + ("a detectable signal" if abs(c) > 2 * s else "no detectable signal")
            )
        lines.append(
            f"  Power: with {n} games only edges over ~{min_detectable_edge(n, 0.5):.0%} "
            "are detectable."
        )
    if not lines:
        return ["", "No spread-pool results yet (needs --snapshot of the s/sc pools and --settle)."]
    return lines


def crowd_signal_report(df: pd.DataFrame) -> List[str]:
    """Text report: pool comparison, straight-up check, and against-the-spread check."""
    d = prepare(df)
    if d.empty:
        return ["No settled games with a line and crowd % to analyze."]
    lines = pool_comparison(d)
    if lines:
        lines.append("")
    lines += _su_section(d)
    lines += _ats_section(d)
    if "pregame" in d:
        logged = d["pregame"].dropna().astype(bool)  # cached-history rows carry no flag
        n_post = int((~logged).sum())
        if n_post:
            lines += [
                "",
                f"Note: {n_post} logged game-pool rows have no pre-kickoff snapshot (backfilled "
                "after the fact); their crowd % may reflect the result.",
            ]
    return lines
