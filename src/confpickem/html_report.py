"""
HTML report generation for confpickem optimization results.

Renders the same picks / win-probability / game-importance / standings data
that the CLI already prints and writes to a .txt file into a single
self-contained, styled HTML file for easier sharing and scanning.
"""

import html
import json
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd


def _esc(value) -> str:
    return html.escape(str(value), quote=True)


def _build_picks_rows(
    sorted_picks: List[tuple],
    remaining_games: List[Dict[str, str]],
) -> List[Dict]:
    """Map (team, confidence) pairs to the same fields the CLI prints per pick."""
    rows = []
    for team, conf in sorted_picks:
        opponent = "Unknown"
        is_remaining = False
        for game in remaining_games:
            if team in (game["home"], game["away"]):
                opponent = game["away"] if team == game["home"] else game["home"]
                is_remaining = True
                break
        rows.append(
            {
                "conf": int(conf),
                "team": team,
                "opp": opponent,
                "locked": not is_remaining,
            }
        )
    return rows


def _build_importance_rows(
    importance_sorted: pd.DataFrame, remaining_games: List[Dict[str, str]], limit: int = 8
) -> List[Dict]:
    rows = []
    for _, row in importance_sorted.head(limit).iterrows():
        game_desc = row["game"]
        away_team, home_team = game_desc.split("@")
        is_remaining = any(
            {home_team, away_team} == {g["home"], g["away"]} for g in remaining_games
        )
        rows.append(
            {
                "game": game_desc.replace("@", " @ "),
                "pick": row["pick"],
                "conf": int(row["points_bid"]),
                "impact": float(row["total_impact"]),
                "winProb": float(row["win_probability"]),
                "lossProb": float(row["loss_probability"]),
                "locked": not is_remaining,
            }
        )
    return rows


def _build_standings_rows(
    all_win_probs: List[Dict],
    has_standings: bool,
    limit: int = 25,
) -> List[Dict]:
    rows = []
    for i, p in enumerate(all_win_probs[:limit], 1):
        total_exp = p["total_expected"]
        current_pts = p["current_pts"]
        rows.append(
            {
                "rank": i,
                "name": p["player"],
                "win_pct": float(p["win_pct"]),
                "total": float(total_exp),
                "current": float(current_pts) if has_standings else None,
                "remaining": float(total_exp - current_pts) if has_standings else None,
                "you": bool(p["is_you"]),
            }
        )
    return rows


def _build_robustness_rows(
    summary_stats: Optional[pd.DataFrame], optimal_picks: Dict[str, int], limit: int = 20
) -> List[Dict]:
    if summary_stats is None or len(summary_stats) == 0:
        return []
    rows = []
    for _, row in summary_stats.head(limit).iterrows():
        rows.append(
            {
                "team": row["team"],
                "frequency": float(row["frequency"]),
                "appearances": int(row["appearances"]),
                "avg_confidence": float(row["avg_confidence"]),
                "median_confidence": float(row["median_confidence"]),
                "std_confidence": float(row["std_confidence"]),
                "min_confidence": float(row["min_confidence"]),
                "max_confidence": float(row["max_confidence"]),
                "in_optimal": row["team"] in optimal_picks,
            }
        )
    return rows


def _build_comparison_rows(slate_comparison: Optional[pd.DataFrame]) -> List[Dict]:
    if slate_comparison is None or len(slate_comparison) == 0:
        return []
    rows = []
    for _, row in slate_comparison.iterrows():
        rows.append(
            {
                "label": row["label"],
                "rank": int(row["rank"]),
                "winProb": float(row["win_probability"]),
                "winStd": float(row["win_std"]),
                "downside": float(row["downside_win_probability"]),
                "isOptimizer": row["label"] == "optimizer",
            }
        )
    return rows


def generate_html_report(
    *,
    week: int,
    league_id: int,
    player_name: str,
    mode: str,
    algo_label: str,
    sorted_picks: List[tuple],
    remaining_games: List[Dict[str, str]],
    opt_win: float,
    rand_win: float,
    importance_sorted: Optional[pd.DataFrame],
    all_win_probs: List[Dict],
    current_standings: Dict[str, float],
    your_rank: Optional[int],
    your_points: Optional[float],
    num_remaining_games: int,
    total_games: int,
    summary_stats: Optional[pd.DataFrame] = None,
    slate_comparison: Optional[pd.DataFrame] = None,
    generated_at: Optional[datetime] = None,
) -> str:
    """Build a single self-contained HTML report string.

    All arguments are values `cli/optimize.py` already computes during a run --
    this function does no simulation of its own, it only renders.
    """
    generated_at = generated_at or datetime.now()
    has_standings = mode == "midweek" and bool(current_standings)

    picks_rows = _build_picks_rows(sorted_picks, remaining_games)
    importance_rows = (
        _build_importance_rows(importance_sorted, remaining_games)
        if importance_sorted is not None and len(importance_sorted) > 0
        else []
    )
    standings_rows = _build_standings_rows(all_win_probs, has_standings)
    robustness_rows = _build_robustness_rows(summary_stats, dict(sorted_picks))
    comparison_rows = _build_comparison_rows(slate_comparison)

    paste_format = ", ".join(f"{team} {conf}" for team, conf in sorted_picks)

    max_impact = max((abs(r["impact"]) for r in importance_rows), default=0.0) or 1.0
    max_win_pct = max((r["win_pct"] for r in standings_rows), default=0.0) or 1.0
    max_comparison_win_pct = max((r["winProb"] for r in comparison_rows), default=0.0) or 1.0

    mode_label = "Mid-Week" if mode == "midweek" else "Beginning-of-Week"
    edge_pp = (opt_win - rand_win) * 100

    data = {
        "picks": picks_rows,
        "importance": importance_rows,
        "standings": standings_rows,
        "robustness": robustness_rows,
        "comparison": comparison_rows,
        "maxImpact": max_impact,
        "maxWinPct": max_win_pct,
        "maxComparisonWinPct": max_comparison_win_pct,
        "pasteFormat": paste_format,
    }
    data_json = json.dumps(data).replace("</", "<\\/")

    summary_note = (
        f"{your_points:g} pts from {total_games - num_remaining_games} completed games"
        if has_standings and your_points is not None
        else f"confidence 1&ndash;{num_remaining_games} still in play"
    )
    rank_value = f"#{your_rank}" if your_rank is not None else "&mdash;"
    rank_sub = f"of {len(standings_rows)} tracked" if standings_rows else "no league standings"

    robustness_section = ""
    if robustness_rows:
        robustness_section = f"""
  <section class="panel">
    <div class="panel-head">
      <span class="panel-title">Pick Robustness</span>
      <span class="panel-note">frequency across top hill-climb solutions</span>
    </div>
    <div class="table-scroll">
      <table class="standings" id="robustness-table">
        <thead>
          <tr>
            <th>Team</th>
            <th class="num">Frequency</th>
            <th class="num">Avg</th>
            <th class="num">Median</th>
            <th class="num">Std</th>
            <th class="num">Range</th>
            <th>Signal</th>
          </tr>
        </thead>
        <tbody><!-- rows injected by script --></tbody>
      </table>
    </div>
  </section>
"""

    comparison_section = ""
    if comparison_rows:
        comparison_section = """
  <section class="panel">
    <div class="panel-head">
      <span class="panel-title">Slate Comparison</span>
      <span class="panel-note">win probability vs. risk, same field for every slate</span>
    </div>
    <div class="table-scroll">
      <table class="standings" id="comparison-table">
        <thead>
          <tr>
            <th>Rank</th>
            <th>Slate</th>
            <th class="num">Win %</th>
            <th>&nbsp;</th>
            <th class="num">Std Dev</th>
            <th class="num">Worst 10% Wks</th>
          </tr>
        </thead>
        <tbody><!-- rows injected by script --></tbody>
      </table>
    </div>
    <div class="comparison-legend">
      <span><b>Std Dev</b> -- how much this slate's win probability swings between
      simulated weeks; higher means more boom/bust risk from a few high-impact picks.</span>
      <span><b>Worst 10% Wks</b> -- average win probability in the worst 10% of
      simulated weeks; your floor when things go wrong.</span>
    </div>
  </section>
"""

    return f"""<title>Week {week} Optimization Report</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,600;9..144,700&family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600;700&display=swap">
<style>
  :root{{
    --bg:#F7F5F1;
    --surface:#FFFFFF;
    --surface-2:#EFEBE2;
    --ink:#1D1B16;
    --ink-soft:#5B5749;
    --ink-faint:#8B8776;
    --line:#DDD7C8;
    --line-strong:#C7BFA9;
    --accent:#2F6B4F;
    --accent-ink:#FFFFFF;
    --accent-soft:#E3EDE6;
    --gold:#9C7A3C;
    --gold-soft:#F1E7D2;
    --good:#3A8A5C;
    --good-soft:#E4F0E7;
    --warn:#B8860B;
    --warn-soft:#F6EDDA;
    --bad:#B84A3E;
    --bad-soft:#F7E8E5;
    --shadow: 0 1px 2px rgba(29,27,22,0.06), 0 6px 20px -8px rgba(29,27,22,0.12);
    --focus: #1D6FB8;
  }}
  @media (prefers-color-scheme: dark){{
    :root:not([data-theme="light"]){{
      --bg:#121815;
      --surface:#182019;
      --surface-2:#1E2721;
      --ink:#EDEAE1;
      --ink-soft:#B4AF9F;
      --ink-faint:#7C7869;
      --line:#2B342C;
      --line-strong:#3B463C;
      --accent:#4CAF7A;
      --accent-ink:#0D140F;
      --accent-soft:#1D3327;
      --gold:#D9A441;
      --gold-soft:#2E2718;
      --good:#45A06E;
      --good-soft:#1C3226;
      --warn:#B8860B;
      --warn-soft:#332A14;
      --bad:#D14F5A;
      --bad-soft:#3A2220;
      --shadow: 0 1px 2px rgba(0,0,0,0.4), 0 8px 24px -10px rgba(0,0,0,0.6);
      --focus: #6FB2E8;
    }}
  }}
  :root[data-theme="dark"]{{
    --bg:#121815;
    --surface:#182019;
    --surface-2:#1E2721;
    --ink:#EDEAE1;
    --ink-soft:#B4AF9F;
    --ink-faint:#7C7869;
    --line:#2B342C;
    --line-strong:#3B463C;
    --accent:#4CAF7A;
    --accent-ink:#0D140F;
    --accent-soft:#1D3327;
    --gold:#D9A441;
    --gold-soft:#2E2718;
    --good:#45A06E;
    --good-soft:#1C3226;
    --warn:#B8860B;
    --warn-soft:#332A14;
    --bad:#D14F5A;
    --bad-soft:#3A2220;
    --shadow: 0 1px 2px rgba(0,0,0,0.4), 0 8px 24px -10px rgba(0,0,0,0.6);
    --focus: #6FB2E8;
  }}

  *{{box-sizing:border-box;}}
  body{{
    margin:0;
    background:var(--bg);
    color:var(--ink);
    font-family:"IBM Plex Sans", system-ui, sans-serif;
    font-size:14px;
    line-height:1.5;
  }}
  ::selection{{ background: var(--accent-soft); }}
  a{{ color: var(--accent); }}
  :focus-visible{{ outline: 2px solid var(--focus); outline-offset: 2px; }}

  .wrap{{
    max-width:1180px;
    margin:0 auto;
    padding:28px 24px 64px;
    display:flex;
    flex-direction:column;
    gap:22px;
  }}

  .masthead{{
    display:flex;
    align-items:flex-end;
    justify-content:space-between;
    gap:24px;
    flex-wrap:wrap;
    border-bottom: 2px solid var(--ink);
    padding-bottom:16px;
  }}
  .masthead-left{{ display:flex; flex-direction:column; gap:6px; }}
  .eyebrow{{
    font-family:"IBM Plex Mono", monospace;
    font-size:11.5px;
    letter-spacing:0.12em;
    text-transform:uppercase;
    color:var(--ink-faint);
  }}
  h1{{
    font-family:"Fraunces", Georgia, serif;
    font-weight:600;
    font-size:clamp(28px, 4vw, 40px);
    margin:0;
    text-wrap:balance;
    letter-spacing:-0.01em;
  }}
  .meta-line{{
    font-size:13px;
    color:var(--ink-soft);
    display:flex;
    gap:14px;
    flex-wrap:wrap;
    font-family:"IBM Plex Mono", monospace;
  }}
  .meta-line b{{ color:var(--ink); font-weight:600; }}
  .algo-badge{{
    font-family:"IBM Plex Mono", monospace;
    font-size:12px;
    font-weight:600;
    letter-spacing:0.02em;
    color:var(--accent-ink);
    background:var(--accent);
    border-radius:5px;
    padding:6px 12px;
    white-space:nowrap;
  }}

  .summary-strip{{
    display:grid;
    grid-template-columns:repeat(4,1fr);
    gap:1px;
    background:var(--line);
    border:1px solid var(--line);
    border-radius:10px;
    overflow:hidden;
    box-shadow:var(--shadow);
  }}
  @media (max-width: 720px){{ .summary-strip{{ grid-template-columns:repeat(2,1fr); }} }}
  .stat{{
    background:var(--surface);
    padding:16px 18px;
    display:flex;
    flex-direction:column;
    gap:4px;
    min-width:0;
  }}
  .stat-label{{
    font-size:11px;
    text-transform:uppercase;
    letter-spacing:0.08em;
    color:var(--ink-faint);
    font-weight:600;
  }}
  .stat-value{{
    font-family:"IBM Plex Mono", monospace;
    font-variant-numeric:tabular-nums;
    font-size:26px;
    font-weight:600;
    letter-spacing:-0.01em;
  }}
  .stat-value.accent{{ color:var(--accent); }}
  .stat-sub{{
    font-size:12.5px;
    color:var(--ink-soft);
  }}
  .stat-sub .good{{ color:var(--good); font-weight:600; }}

  .board{{
    display:grid;
    grid-template-columns: 1.15fr 1fr;
    gap:20px;
    align-items:start;
  }}
  @media (max-width: 880px){{ .board{{ grid-template-columns:1fr; }} }}

  .panel{{
    background:var(--surface);
    border:1px solid var(--line);
    border-radius:10px;
    box-shadow:var(--shadow);
    overflow:hidden;
  }}
  .panel-head{{
    display:flex;
    align-items:baseline;
    justify-content:space-between;
    gap:12px;
    padding:14px 18px;
    border-bottom:1px solid var(--line);
    background:var(--surface-2);
  }}
  .panel-title{{
    font-family:"Fraunces", Georgia, serif;
    font-size:17px;
    font-weight:600;
  }}
  .panel-note{{
    font-size:11.5px;
    color:var(--ink-faint);
    font-family:"IBM Plex Mono", monospace;
  }}
  .panel-body{{ padding:6px 0 4px; }}

  .ticket{{ list-style:none; margin:0; padding:0; }}
  .ticket-row{{
    display:grid;
    grid-template-columns:44px 1fr auto auto;
    align-items:center;
    gap:12px;
    padding:9px 18px;
    border-bottom:1px solid var(--line);
  }}
  .ticket-row:last-child{{ border-bottom:none; }}
  .ticket-row:hover{{ background:var(--surface-2); }}
  .conf-chip{{
    font-family:"IBM Plex Mono", monospace;
    font-variant-numeric:tabular-nums;
    font-weight:700;
    font-size:14px;
    width:32px;
    height:32px;
    border-radius:7px;
    display:flex;
    align-items:center;
    justify-content:center;
    background:var(--gold-soft);
    color:var(--gold);
    border:1px solid var(--line-strong);
  }}
  .conf-chip.locked{{
    background:var(--surface-2);
    color:var(--ink-faint);
    border-style:dashed;
  }}
  .matchup{{ display:flex; flex-direction:column; gap:1px; min-width:0; }}
  .pick-team{{ font-weight:600; font-size:14.5px; }}
  .vs-opp{{ font-size:12px; color:var(--ink-faint); }}
  .status-pill{{
    font-family:"IBM Plex Mono", monospace;
    font-size:10.5px;
    font-weight:600;
    letter-spacing:0.04em;
    text-transform:uppercase;
    padding:3px 8px;
    border-radius:99px;
    white-space:nowrap;
  }}
  .status-pill.upcoming{{ background:var(--accent-soft); color:var(--accent); }}
  .status-pill.locked{{ background:var(--surface-2); color:var(--ink-faint); }}

  .copy-row{{
    display:flex;
    align-items:center;
    gap:10px;
    padding:12px 18px;
    background:var(--surface-2);
    border-top:1px solid var(--line);
  }}
  .copy-text{{
    font-family:"IBM Plex Mono", monospace;
    font-size:12px;
    color:var(--ink-soft);
    overflow-x:auto;
    white-space:nowrap;
    flex:1;
    padding:6px 0;
  }}
  .copy-btn{{
    font-family:"IBM Plex Sans", sans-serif;
    font-size:12.5px;
    font-weight:600;
    color:var(--accent-ink);
    background:var(--accent);
    border:none;
    border-radius:6px;
    padding:7px 14px;
    cursor:pointer;
    white-space:nowrap;
  }}
  .copy-btn:hover{{ filter:brightness(1.08); }}
  .copy-btn:active{{ transform: translateY(1px); }}

  .importance-list{{ padding:14px 18px 16px; display:flex; flex-direction:column; gap:11px; }}
  .imp-row{{ display:flex; flex-direction:column; gap:4px; }}
  .imp-top{{
    display:flex;
    justify-content:space-between;
    align-items:baseline;
    gap:8px;
    font-size:13px;
  }}
  .imp-game{{ font-weight:600; }}
  .imp-pick{{ color:var(--ink-faint); font-size:12px; }}
  .imp-val{{
    font-family:"IBM Plex Mono", monospace;
    font-variant-numeric:tabular-nums;
    font-weight:600;
    font-size:13px;
  }}
  .imp-track{{
    position:relative;
    height:8px;
    background:var(--surface-2);
    border-radius:5px;
    overflow:hidden;
  }}
  .imp-fill{{
    position:absolute;
    top:0; bottom:0;
    border-radius:5px;
  }}
  .imp-mid{{
    position:absolute;
    top:-2px; bottom:-2px; left:50%;
    width:1px;
    background:var(--line-strong);
  }}
  .imp-outcomes{{
    display:flex;
    justify-content:space-between;
    font-family:"IBM Plex Mono", monospace;
    font-variant-numeric:tabular-nums;
    font-size:11.5px;
  }}
  .imp-outcomes .if-wrong{{ color:var(--bad); }}
  .imp-outcomes .if-right{{ color:var(--good); }}
  .imp-outcomes .outcome-label{{ color:var(--ink-faint); font-weight:400; }}
  .empty-note{{
    padding:20px 18px;
    color:var(--ink-faint);
    font-size:13px;
  }}

  .table-scroll{{ overflow-x:auto; }}
  table.standings{{
    width:100%;
    border-collapse:collapse;
    font-size:13.5px;
    min-width:560px;
  }}
  table.standings th{{
    text-align:left;
    font-size:11px;
    text-transform:uppercase;
    letter-spacing:0.06em;
    color:var(--ink-faint);
    font-weight:600;
    padding:10px 18px;
    border-bottom:1px solid var(--line);
    white-space:nowrap;
  }}
  table.standings td{{
    padding:9px 18px;
    border-bottom:1px solid var(--line);
    white-space:nowrap;
  }}
  table.standings td.num, table.standings th.num{{
    font-family:"IBM Plex Mono", monospace;
    font-variant-numeric:tabular-nums;
    text-align:right;
  }}
  table.standings tbody tr:hover{{ background:var(--surface-2); }}
  table.standings tbody tr.you{{ background:var(--accent-soft); }}
  table.standings tbody tr.you td{{ font-weight:600; }}
  .rank-badge{{
    display:inline-flex;
    align-items:center;
    justify-content:center;
    width:22px; height:22px;
    border-radius:5px;
    font-family:"IBM Plex Mono", monospace;
    font-size:12px;
    font-weight:700;
    color:var(--ink-soft);
  }}
  .rank-badge.gold{{ background:var(--gold-soft); color:var(--gold); }}
  .you-tag{{
    font-family:"IBM Plex Mono", monospace;
    font-size:10px;
    font-weight:700;
    letter-spacing:0.04em;
    background:var(--accent);
    color:var(--accent-ink);
    padding:2px 6px;
    border-radius:4px;
    margin-left:8px;
  }}
  .bar-cell{{ display:flex; align-items:center; gap:8px; }}
  .bar-track{{
    width:80px;
    height:6px;
    background:var(--surface-2);
    border-radius:4px;
    overflow:hidden;
    flex-shrink:0;
  }}
  .bar-fill{{ height:100%; background:var(--accent); border-radius:4px; }}

  .signal-lock{{ color:var(--good); }}
  .signal-confident{{ color:var(--accent); }}
  .signal-moderate{{ color:var(--warn); }}
  .signal-uncertain{{ color:var(--ink-faint); }}

  .comparison-legend{{
    display:flex;
    flex-direction:column;
    gap:4px;
    padding:12px 18px 16px;
    border-top:1px solid var(--line);
    background:var(--surface-2);
    font-size:11.5px;
    color:var(--ink-soft);
  }}
  .comparison-legend b{{ color:var(--ink); }}
  .comparison-row.is-optimizer{{ background:var(--accent-soft); }}
  .comparison-row.is-optimizer td{{ font-weight:600; }}

  footer{{
    display:flex;
    justify-content:space-between;
    gap:12px;
    flex-wrap:wrap;
    padding-top:6px;
    font-size:11.5px;
    color:var(--ink-faint);
    font-family:"IBM Plex Mono", monospace;
  }}
</style>

<div class="wrap">

  <header class="masthead">
    <div class="masthead-left">
      <span class="eyebrow">Confidence Pick&rsquo;Em &middot; League {_esc(league_id)}</span>
      <h1>Week {_esc(week)} Optimization Report</h1>
      <div class="meta-line">
        <span>Player: <b>{_esc(player_name)}</b></span>
        <span>&middot;</span>
        <span>Mode: <b>{_esc(mode_label)}</b></span>
        <span>&middot;</span>
        <span>Generated {_esc(generated_at.strftime('%Y-%m-%d %H:%M'))}</span>
      </div>
    </div>
    <span class="algo-badge">{_esc(algo_label)}</span>
  </header>

  <section class="summary-strip" aria-label="Headline results">
    <div class="stat">
      <span class="stat-label">Win Probability</span>
      <span class="stat-value accent">{opt_win*100:.1f}%</span>
      <span class="stat-sub">vs. <span class="good">{rand_win*100:.1f}%</span> on random picks</span>
    </div>
    <div class="stat">
      <span class="stat-label">Optimizer Edge</span>
      <span class="stat-value">{edge_pp:+.1f}<span style="font-size:15px;color:var(--ink-faint);">pp</span></span>
      <span class="stat-sub">over an unoptimized slate</span>
    </div>
    <div class="stat">
      <span class="stat-label">Current Rank</span>
      <span class="stat-value">{rank_value}<span style="font-size:15px;color:var(--ink-faint);"> / {len(standings_rows)}</span></span>
      <span class="stat-sub">{summary_note}</span>
    </div>
    <div class="stat">
      <span class="stat-label">Games Remaining</span>
      <span class="stat-value">{num_remaining_games}<span style="font-size:15px;color:var(--ink-faint);"> / {total_games}</span></span>
      <span class="stat-sub">{rank_sub}</span>
    </div>
  </section>

  <section class="board">

    <div class="panel">
      <div class="panel-head">
        <span class="panel-title">Optimized Picks</span>
        <span class="panel-note">sorted by confidence</span>
      </div>
      <div class="panel-body">
        <ul class="ticket" id="ticket-list"></ul>
      </div>
      <div class="copy-row">
        <code class="copy-text" id="copy-text">{_esc(paste_format)}</code>
        <button class="copy-btn" id="copy-btn" type="button">Copy</button>
      </div>
    </div>

    <div class="panel">
      <div class="panel-head">
        <span class="panel-title">Game Importance</span>
        <span class="panel-note">swing in win&nbsp;%</span>
      </div>
      <div class="importance-list" id="importance-list"></div>
    </div>

  </section>

  <section class="panel">
    <div class="panel-head">
      <span class="panel-title">Simulated Final Standings</span>
      <span class="panel-note">{"win % / expected points" if has_standings else "beginning-of-week projection"}</span>
    </div>
    <div class="table-scroll">
      <table class="standings" id="standings-table">
        <thead>
          <tr id="standings-head"></tr>
        </thead>
        <tbody></tbody>
      </table>
    </div>
  </section>
{comparison_section}
{robustness_section}
  <footer>
    <span>confpickem &middot; {_esc(algo_label)}</span>
    <span>Generated by confpickem-optimize</span>
  </footer>

</div>

<script>
(function(){{
  var DATA = {data_json};
  var HAS_STANDINGS = {str(has_standings).lower()};

  var ticket = document.getElementById('ticket-list');
  if(DATA.picks.length === 0){{
    ticket.innerHTML = '<li class="empty-note">No picks to display.</li>';
  }}
  DATA.picks.forEach(function(p){{
    var li = document.createElement('li');
    li.className = 'ticket-row';

    var chip = document.createElement('span');
    chip.className = 'conf-chip' + (p.locked ? ' locked' : '');
    chip.textContent = p.conf;

    var matchup = document.createElement('span');
    matchup.className = 'matchup';
    var team = document.createElement('span');
    team.className = 'pick-team';
    team.textContent = p.team;
    var vs = document.createElement('span');
    vs.className = 'vs-opp';
    vs.textContent = 'vs ' + p.opp;
    matchup.appendChild(team);
    matchup.appendChild(vs);

    var pill = document.createElement('span');
    pill.className = 'status-pill ' + (p.locked ? 'locked' : 'upcoming');
    pill.textContent = p.locked ? 'Locked' : 'Upcoming';

    li.appendChild(chip);
    li.appendChild(matchup);
    li.appendChild(document.createElement('span'));
    li.appendChild(pill);
    ticket.appendChild(li);
  }});

  var impList = document.getElementById('importance-list');
  if(DATA.importance.length === 0){{
    impList.innerHTML = '<div class="empty-note">Game importance unavailable for this run.</div>';
  }}
  DATA.importance.forEach(function(d){{
    var row = document.createElement('div');
    row.className = 'imp-row';

    var top = document.createElement('div');
    top.className = 'imp-top';
    var left = document.createElement('span');
    var gameEl = document.createElement('span');
    gameEl.className = 'imp-game';
    gameEl.textContent = d.game;
    var pickEl = document.createElement('span');
    pickEl.className = 'imp-pick';
    pickEl.textContent = ' \\u2192 ' + d.pick + ' (' + d.conf + ' pts)';
    left.appendChild(gameEl);
    left.appendChild(pickEl);

    var val = document.createElement('span');
    val.className = 'imp-val';
    val.style.color = d.impact < 0 ? 'var(--bad)' : 'var(--accent)';
    val.textContent = (d.impact >= 0 ? '+' : '') + (Math.round(d.impact*1000)/10) + 'pp';
    top.appendChild(left);
    top.appendChild(val);

    var track = document.createElement('div');
    track.className = 'imp-track';
    var mid = document.createElement('div');
    mid.className = 'imp-mid';
    var fill = document.createElement('div');
    fill.className = 'imp-fill';
    var pct = Math.abs(d.impact) / DATA.maxImpact * 48;
    if(d.impact >= 0){{
      fill.style.left = '50%';
      fill.style.width = pct + '%';
      fill.style.background = 'var(--accent)';
    }} else {{
      fill.style.left = (50 - pct) + '%';
      fill.style.width = pct + '%';
      fill.style.background = 'var(--bad)';
    }}
    track.appendChild(fill);
    track.appendChild(mid);

    var outcomes = document.createElement('div');
    outcomes.className = 'imp-outcomes';
    var wrongEl = document.createElement('span');
    wrongEl.className = 'if-wrong';
    wrongEl.innerHTML = '<span class="outcome-label">If wrong:</span> ' + (Math.round(d.lossProb*1000)/10) + '%';
    var rightEl = document.createElement('span');
    rightEl.className = 'if-right';
    rightEl.innerHTML = '<span class="outcome-label">If right:</span> ' + (Math.round(d.winProb*1000)/10) + '%';
    outcomes.appendChild(wrongEl);
    outcomes.appendChild(rightEl);

    row.appendChild(top);
    row.appendChild(track);
    row.appendChild(outcomes);
    impList.appendChild(row);
  }});

  var head = document.getElementById('standings-head');
  var headCells = ['Rank', 'Player', 'Win %', ''];
  if(HAS_STANDINGS){{
    headCells = headCells.concat(['Total Exp.', 'Current', 'Remaining']);
  }} else {{
    headCells = headCells.concat(['Exp. Points']);
  }}
  headCells.forEach(function(text){{
    var th = document.createElement('th');
    if(['Win %','Total Exp.','Current','Remaining','Exp. Points'].indexOf(text) !== -1){{
      th.className = 'num';
    }}
    th.textContent = text;
    head.appendChild(th);
  }});

  var tbody = document.querySelector('#standings-table tbody');
  if(DATA.standings.length === 0){{
    var tr0 = document.createElement('tr');
    var td0 = document.createElement('td');
    td0.className = 'empty-note';
    td0.colSpan = headCells.length;
    td0.textContent = 'No standings data available.';
    tr0.appendChild(td0);
    tbody.appendChild(tr0);
  }}
  DATA.standings.forEach(function(s){{
    var tr = document.createElement('tr');
    if(s.you) tr.className = 'you';

    var tdRank = document.createElement('td');
    var badge = document.createElement('span');
    badge.className = 'rank-badge' + (s.rank === 1 ? ' gold' : '');
    badge.textContent = s.rank;
    tdRank.appendChild(badge);

    var tdName = document.createElement('td');
    tdName.textContent = s.name;
    if(s.you){{
      var tag = document.createElement('span');
      tag.className = 'you-tag';
      tag.textContent = 'YOU';
      tdName.appendChild(tag);
    }}

    var tdWin = document.createElement('td');
    tdWin.className = 'num';
    tdWin.textContent = (Math.round(s.win_pct*1000)/10) + '%';

    var tdBar = document.createElement('td');
    var barWrap = document.createElement('div');
    barWrap.className = 'bar-cell';
    var track = document.createElement('div');
    track.className = 'bar-track';
    var fill = document.createElement('div');
    fill.className = 'bar-fill';
    fill.style.width = (DATA.maxWinPct > 0 ? (s.win_pct / DATA.maxWinPct * 100) : 0) + '%';
    track.appendChild(fill);
    barWrap.appendChild(track);
    tdBar.appendChild(barWrap);

    var tdTotal = document.createElement('td');
    tdTotal.className = 'num';
    tdTotal.textContent = s.total.toFixed(1);

    tr.appendChild(tdRank);
    tr.appendChild(tdName);
    tr.appendChild(tdWin);
    tr.appendChild(tdBar);
    tr.appendChild(tdTotal);

    if(HAS_STANDINGS){{
      var tdCurrent = document.createElement('td');
      tdCurrent.className = 'num';
      tdCurrent.textContent = s.current !== null ? s.current : '\\u2014';
      var tdRemaining = document.createElement('td');
      tdRemaining.className = 'num';
      tdRemaining.textContent = s.remaining !== null ? ('+' + s.remaining.toFixed(1)) : '\\u2014';
      tr.appendChild(tdCurrent);
      tr.appendChild(tdRemaining);
    }}

    tbody.appendChild(tr);
  }});

  var robustBody = document.querySelector('#robustness-table tbody');
  if(robustBody){{
    DATA.robustness.forEach(function(r){{
      var tr = document.createElement('tr');

      var signalClass = 'signal-uncertain';
      var signalText = 'Uncertain';
      if(r.frequency > 0.9){{ signalClass = 'signal-lock'; signalText = 'Lock it in'; }}
      else if(r.frequency > 0.7){{ signalClass = 'signal-confident'; signalText = 'Very confident'; }}
      else if(r.frequency > 0.5){{ signalClass = 'signal-confident'; signalText = 'Confident'; }}
      else if(r.frequency > 0.3){{ signalClass = 'signal-moderate'; signalText = 'Moderate'; }}

      var tdTeam = document.createElement('td');
      tdTeam.textContent = r.team + (r.in_optimal ? ' \\u2192' : '');

      var tdFreq = document.createElement('td');
      tdFreq.className = 'num';
      tdFreq.textContent = (Math.round(r.frequency*1000)/10) + '% (' + r.appearances + ')';

      var tdAvg = document.createElement('td');
      tdAvg.className = 'num';
      tdAvg.textContent = r.avg_confidence.toFixed(1);

      var tdMed = document.createElement('td');
      tdMed.className = 'num';
      tdMed.textContent = r.median_confidence.toFixed(1);

      var tdStd = document.createElement('td');
      tdStd.className = 'num';
      tdStd.textContent = r.std_confidence.toFixed(2);

      var tdRange = document.createElement('td');
      tdRange.className = 'num';
      tdRange.textContent = r.min_confidence.toFixed(0) + '\\u2013' + r.max_confidence.toFixed(0);

      var tdSignal = document.createElement('td');
      tdSignal.className = signalClass;
      tdSignal.textContent = signalText;

      tr.appendChild(tdTeam);
      tr.appendChild(tdFreq);
      tr.appendChild(tdAvg);
      tr.appendChild(tdMed);
      tr.appendChild(tdStd);
      tr.appendChild(tdRange);
      tr.appendChild(tdSignal);
      robustBody.appendChild(tr);
    }});
  }}

  var comparisonBody = document.querySelector('#comparison-table tbody');
  if(comparisonBody){{
    DATA.comparison.forEach(function(c){{
      var tr = document.createElement('tr');
      tr.className = 'comparison-row' + (c.isOptimizer ? ' is-optimizer' : '');

      var tdRank = document.createElement('td');
      var badge = document.createElement('span');
      badge.className = 'rank-badge' + (c.rank === 1 ? ' gold' : '');
      badge.textContent = c.rank;
      tdRank.appendChild(badge);

      var tdLabel = document.createElement('td');
      tdLabel.textContent = c.label;
      if(c.isOptimizer){{
        var tag = document.createElement('span');
        tag.className = 'you-tag';
        tag.textContent = 'OPTIMIZER';
        tdLabel.appendChild(tag);
      }}

      var tdWin = document.createElement('td');
      tdWin.className = 'num';
      tdWin.textContent = (Math.round(c.winProb*1000)/10) + '%';

      var tdBar = document.createElement('td');
      var barWrap = document.createElement('div');
      barWrap.className = 'bar-cell';
      var track = document.createElement('div');
      track.className = 'bar-track';
      var fill = document.createElement('div');
      fill.className = 'bar-fill';
      fill.style.width = (DATA.maxComparisonWinPct > 0 ? (c.winProb / DATA.maxComparisonWinPct * 100) : 0) + '%';
      track.appendChild(fill);
      barWrap.appendChild(track);
      tdBar.appendChild(barWrap);

      var tdStd = document.createElement('td');
      tdStd.className = 'num';
      tdStd.textContent = c.winStd.toFixed(4);

      var tdDownside = document.createElement('td');
      tdDownside.className = 'num';
      tdDownside.textContent = (Math.round(c.downside*1000)/10) + '%';

      tr.appendChild(tdRank);
      tr.appendChild(tdLabel);
      tr.appendChild(tdWin);
      tr.appendChild(tdBar);
      tr.appendChild(tdStd);
      tr.appendChild(tdDownside);
      comparisonBody.appendChild(tr);
    }});
  }}

  document.getElementById('copy-btn').addEventListener('click', function(){{
    var text = document.getElementById('copy-text').textContent;
    var btn = this;
    function done(ok){{
      btn.textContent = ok ? 'Copied!' : 'Copy failed';
      setTimeout(function(){{ btn.textContent = 'Copy'; }}, 1500);
    }}
    if(navigator.clipboard && navigator.clipboard.writeText){{
      navigator.clipboard.writeText(text).then(function(){{ done(true); }}, function(){{ done(false); }});
    }} else {{
      done(false);
    }}
  }});
}})();
</script>
"""
