"""Tests for the crowd % / line / result log and signal report (all offline)."""

import json
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from src.confpickem import crowd_log

EST = timezone(timedelta(hours=-5))


def _at(*args):
    return datetime(*args, tzinfo=EST)


def _game_html(fav, und, fav_pct, spread, when="Sunday, Nov 8, 1:00 pm EST", und_home=True):
    at_fav, at_und = ("", "@ ") if und_home else ("@ ", "")
    return f"""
    <div class="ysf-matchup-dist yspmainmodule">
      <div class="hd"><h4><span>{when}</span></h4></div>
      <div class="bd">
        <dl class="favorite"><dd class="team">{at_fav}<a>{fav} Team</a> (-{spread} pts)</dd>
          <dd class="percent">{fav_pct}%</dd></dl>
        <dl class="underdog"><dd class="team">{at_und}<a>{und} Team</a> ({spread} pts)</dd>
          <dd class="percent">{100 - fav_pct}%</dd></dl>
      </div>
      <div class="ft"><table>
        <thead><tr><th>{fav}</th><th></th><th>{und}</th></tr></thead>
        <tbody>
          <tr class="odd first"><td>N/A</td><td class="label">Avg. Confidence</td><td>N/A</td></tr>
          <tr class="even"><td>6-2</td><td class="label">Record</td><td>4-5</td></tr>
          <tr class="odd"><td>{spread} pts</td><td class="label">Favorite</td><td> </td></tr>
        </tbody></table></div>
    </div>"""


def test_parse_distribution_tolerates_na_confidence():
    html = _game_html("Jax", "Bal", 69, 10.0) + _game_html(
        "KC", "Den", 55, 3.5, when="Sunday, Nov 8, 4:25 pm EST", und_home=False
    )
    games = crowd_log.parse_distribution(html, 2026)
    assert [g["favorite"] for g in games] == ["Jax", "KC"]
    assert games[0]["spread"] == 10.0 and games[0]["favorite_pick_pct"] == 69.0
    assert games[0]["underdog_pick_pct"] == 31.0
    assert games[0]["favorite_home"] is False and games[1]["favorite_home"] is True
    assert pd.Timestamp(games[0]["kickoff_time"]).hour == 13


def test_parse_distribution_missing_line_is_none():
    html = _game_html("Jax", "Bal", 69, 10.0).replace("10.0 pts</td><td", "Off</td><td")
    html = html.replace("(-10.0 pts)", "").replace("(10.0 pts)", "")
    assert crowd_log.parse_distribution(html, 2026)[0]["spread"] is None


def test_fetch_distribution_rejects_unknown_pool():
    with pytest.raises(ValueError):
        crowd_log.fetch_distribution(None, 1, "x")


def test_snapshot_week_logs_every_pool(monkeypatch):
    seen = []

    def fake_fetch(session, week, pool):
        seen.append(pool)
        return _game_html("Jax", "Bal", {"n": 96, "s": 69}.get(pool, 80), 10.0)

    monkeypatch.setattr(crowd_log, "fetch_distribution", fake_fetch)
    recs = crowd_log.snapshot_week(None, 9, 2026, pools=("n", "s"), pause=0)
    assert seen == ["n", "s"]
    assert [(r["pool"], r["favorite_pick_pct"]) for r in recs] == [("n", 96.0), ("s", 69.0)]
    assert all(r["kind"] == "snapshot" and r["season"] == 2026 and r["week"] == 9 for r in recs)


def _espn(home, away, hs, as_, completed=True):
    return {
        "competitions": [
            {
                "competitors": [
                    {"homeAway": "home", "team": {"abbreviation": home}, "score": str(hs)},
                    {"homeAway": "away", "team": {"abbreviation": away}, "score": str(as_)},
                ]
            }
        ],
        "status": {"type": {"completed": completed}},
    }


def test_parse_espn_week_and_result_matching_aliases_and_sides():
    payload = {
        "events": [
            _espn("GB", "CAR", 27, 19),
            _espn("WSH", "DAL", 10, 17),
            _espn("KC", "DEN", 0, 0, completed=False),
        ]
    }
    espn = crowd_log.parse_espn_week(payload)
    assert espn[1]["home"] == "was"
    games = [("Car", "GB"), ("Dal", "Was"), ("KC", "Den")]  # Yahoo names; first is favorite
    recs = crowd_log.result_records(games, espn, 2026, 8)
    assert [(r["favorite"], r["favorite_score"], r["underdog_score"]) for r in recs] == [
        ("Car", 19, 27),  # favorite was the away team
        ("Dal", 17, 10),  # Yahoo "Was" matches ESPN "WSH"
    ]  # the unfinished game is skipped


def test_settle_log_only_looks_up_unsettled_games(tmp_path):
    log = str(tmp_path / "log.jsonl")
    snap = {"kind": "snapshot", "season": 2026, "pool": "c", "week": 8, "spread": 3.0}
    crowd_log.append_records(
        log,
        [
            {**snap, "favorite": "Car", "underdog": "GB"},
            {**snap, "favorite": "Car", "underdog": "GB", "pool": "s"},  # same game, other pool
            {**snap, "week": 9, "favorite": "KC", "underdog": "Den"},
        ],
    )
    calls = []

    def fetch(season, week):
        calls.append(week)
        return crowd_log.parse_espn_week({"events": [_espn("GB", "CAR", 27, 19)]})

    assert crowd_log.settle_log(log, 2026, fetch=fetch) == 1  # week 9 game isn't final
    assert sorted(calls) == [8, 9]
    calls.clear()
    assert crowd_log.settle_log(log, 2026, weeks=[8], fetch=fetch) == 0
    assert calls == []  # week 8 is already settled: no lookup
    assert crowd_log.settle_log(log, 2026, weeks=[9], fetch=fetch) == 0
    assert calls == [9]


def test_load_log_takes_last_pregame_snapshot_and_merges_scores(tmp_path):
    log = str(tmp_path / "log.jsonl")
    kick = _at(2026, 11, 8, 13, 0).isoformat()
    base = {
        "kind": "snapshot",
        "season": 2026,
        "week": 9,
        "pool": "s",
        "favorite": "Bal",
        "underdog": "Cin",
        "kickoff_time": kick,
        "favorite_pick_pct": 60.0,
    }
    crowd_log.append_records(
        log,
        [
            {**base, "snapshot_time": _at(2026, 11, 8, 11, 0).isoformat(), "spread": 6.5},
            {**base, "snapshot_time": _at(2026, 11, 8, 12, 30).isoformat(), "spread": 7.5},
            {**base, "snapshot_time": _at(2026, 11, 8, 18, 0).isoformat(), "spread": 9.0},
            {  # a game only ever seen after kickoff
                **base,
                "favorite": "KC",
                "underdog": "Den",
                "snapshot_time": _at(2026, 11, 8, 18, 0).isoformat(),
                "spread": 3.0,
            },
            {
                "kind": "result",
                "settled_time": _at(2026, 11, 8, 19, 0).isoformat(),
                "season": 2026,
                "week": 9,
                "favorite": "Bal",
                "underdog": "Cin",
                "favorite_score": 24,
                "underdog_score": 20,
            },
        ],
    )
    df = crowd_log.load_log(log).set_index("favorite")
    assert df.loc["Bal", "spread"] == 7.5 and bool(df.loc["Bal", "pregame"])
    assert df.loc["Bal", "favorite_score"] == 24
    assert df.loc["KC", "spread"] == 3.0 and not bool(df.loc["KC", "pregame"])
    assert pd.isna(df.loc["KC", "favorite_score"])


def test_load_log_missing_file(tmp_path):
    assert crowd_log.load_log(str(tmp_path / "nope.jsonl")).empty
    assert crowd_log.read_log(str(tmp_path / "nope.jsonl")) == []


def test_prepare_su_and_ats_outcomes():
    df = pd.DataFrame(
        {
            "season": 2026,
            "week": 1,
            "pool": "s",
            "favorite": list("ABCDEFG"),
            "underdog": list("uvwxyzt"),
            "spread": [3.0, 3.0, 3.0, 3.0, 0.0, 3.0, 3.0],
            "favorite_pick_pct": [80.0, 80.0, 80.0, 80.0, 50.0, None, 80.0],
            "favorite_score": [24, 24, 24, 20, 21, 30, 23],
            "underdog_score": [20, 20, 23, 20, 20, 10, 20],
        }
    )
    d = crowd_log.prepare(df).set_index("favorite")
    assert list(d.index) == list("ABCDG")  # pick'em and missing-crowd rows dropped
    assert list(d["fav_covered"].iloc[:2]) == [1.0, 1.0]  # +4 covers -3
    assert d.loc["C", "fav_covered"] == 0.0 and d.loc["C", "fav_won"] == 1.0  # won, no cover
    assert pd.isna(d.loc["D", "fav_won"])  # a tie isn't a win either way
    assert d.loc["D", "fav_covered"] == 0.0  # tied game, but a loss laying 3
    assert pd.isna(d.loc["G", "fav_covered"]) and d.loc["G", "fav_won"] == 1.0  # push
    assert len(crowd_log.ats_games(d)) == 4
    assert abs(d.loc["A", "implied"] - (0.5 + 0.031 * 3.0)) < 1e-9


def test_prepare_cache_rows_use_winner_for_su_only():
    df = pd.DataFrame(
        {
            "season": [2025, 2025],
            "week": [1, 1],
            "favorite": ["A", "B"],
            "underdog": ["u", "v"],
            "spread": [3.0, 3.0],
            "favorite_pick_pct": [80.0, 80.0],
            "winner": ["A", None],
        }
    )
    d = crowd_log.prepare(df)
    assert (d["pool"] == "c").all()
    assert len(crowd_log.su_games(d)) == 1 and len(crowd_log.ats_games(d)) == 0


def _synthetic(n, crowd_matters, seed=0, ats=False):
    rng = np.random.default_rng(seed)
    spread = rng.uniform(1, 14, n)
    crowd = rng.uniform(0.3, 0.98, n)
    p = np.full(n, 0.5) if ats else 0.5 + 0.031 * spread
    if crowd_matters:
        p = p + 0.4 * (crowd - 0.65)
    p = np.clip(p, 0.02, 0.98)
    won = rng.random(n) < p
    df = pd.DataFrame(
        {
            "season": 2026,
            "week": 1,
            "pool": "s" if ats else "c",
            "favorite": [f"F{i}" for i in range(n)],
            "underdog": [f"U{i}" for i in range(n)],
            "spread": spread,
            "favorite_pick_pct": crowd * 100,
        }
    )
    if ats:  # margin lands just either side of the spread
        margin = spread + np.where(won, 0.5, -0.5)
        df["favorite_score"] = margin + 20
        df["underdog_score"] = 20.0
    else:
        df["winner"] = [f"F{i}" if w else f"U{i}" for i, w in enumerate(won)]
    return df


def test_logit_fit_detects_planted_crowd_signal_and_not_noise():
    signal = crowd_log.logit_fit(crowd_log.su_games(crowd_log.prepare(_synthetic(6000, True))))
    noise = crowd_log.logit_fit(crowd_log.su_games(crowd_log.prepare(_synthetic(6000, False))))
    c, s = signal["crowd"]
    assert c > 2 * s
    c0, s0 = noise["crowd"]
    assert abs(c0) < 3 * s0


def test_ats_logit_fit_detects_planted_signal():
    d = crowd_log.ats_games(crowd_log.prepare(_synthetic(6000, True, ats=True)))
    c, s = crowd_log.ats_logit_fit(d)["crowd"]
    assert c > 2 * s
    quiet = crowd_log.ats_games(crowd_log.prepare(_synthetic(6000, False, seed=1, ats=True)))
    c0, s0 = crowd_log.ats_logit_fit(quiet)["crowd"]
    assert abs(c0) < 3 * s0


def test_bucket_table_counts_every_game():
    d = crowd_log.su_games(crowd_log.prepare(_synthetic(500, False)))
    assert crowd_log.bucket_table(d)["n"].sum() == len(d)


def test_pool_comparison_needs_overlapping_pools():
    base = _synthetic(50, False)
    spread_pool = base.assign(pool="s", favorite_pick_pct=base["favorite_pick_pct"] * 0.5)
    lines = crowd_log.pool_comparison(crowd_log.prepare(pd.concat([base, spread_pool])))
    assert "50 games in every pool" in lines[0] and len(lines) == 3
    assert crowd_log.pool_comparison(crowd_log.prepare(base)) == []


def test_report_handles_empty_su_only_and_ats():
    assert "No settled games" in crowd_log.crowd_signal_report(pd.DataFrame())[0]
    su = "\n".join(crowd_log.crowd_signal_report(_synthetic(400, False)))
    assert "Logit:" in su and "Power:" in su and "No spread-pool results yet" in su
    both = pd.concat([_synthetic(400, False), _synthetic(300, False, seed=2, ats=True)])
    text = "\n".join(crowd_log.crowd_signal_report(both))
    assert "Against the spread, s pool" in text and "crowd's side covered" in text


def test_min_detectable_edge_shrinks_with_n():
    assert crowd_log.min_detectable_edge(4000) < crowd_log.min_detectable_edge(400)


def test_log_records_are_json_lines(tmp_path):
    log = tmp_path / "log.jsonl"
    crowd_log.append_records(str(log), [{"a": 1}, {"a": 2}])
    assert [json.loads(line) for line in log.read_text().splitlines()] == [{"a": 1}, {"a": 2}]
