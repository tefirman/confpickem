"""Streamlit viewer for the live tracker (launch with ``confpickem-app``).

Shows every entry's win probability over time from the JSONL history written by
``take_snapshot``. "Refresh now" takes a snapshot immediately; with auto-refresh
on, the page takes one whenever the newest snapshot is older than the interval.
Run it alongside (or instead of) ``confpickem-live`` -- both append to the same file.
"""

import os
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

try:
    from confpickem import live_tracker as lt
except ImportError:  # running from a source checkout without an install
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from src.confpickem import live_tracker as lt

# Categorical slots from the validated reference palette, fixed order.
SERIES_LIGHT = [
    "#2a78d6",
    "#eb6834",
    "#1baf7a",
    "#eda100",
    "#e87ba4",
    "#008300",
    "#6250d6",
    "#e34948",
]
SERIES_DARK = [
    "#3987e5",
    "#d95926",
    "#199e70",
    "#c98500",
    "#d55181",
    "#008300",
    "#9085e9",
    "#e66767",
]
MUTED = {"light": "#c9c8c3", "dark": "#52514e"}


def is_dark() -> bool:
    try:
        return st.context.theme.type == "dark"
    except Exception:  # older Streamlit: no theme info
        return False


def pick_entrants(wide: pd.DataFrame, player: str, top_n: int) -> list:
    """Default lines to emphasize: the highlighted entry first, then the current
    top ``top_n`` by win %."""
    latest = wide.iloc[-1].sort_values(ascending=False)
    chosen = []
    if player:
        chosen += [p for p in wide.columns if player.casefold() in p.casefold()][:1]
    chosen += [p for p in latest.index if p not in chosen][: max(top_n - len(chosen), 0)]
    return chosen


def build_chart(wide: pd.DataFrame, events: list, chosen: list, player: str) -> go.Figure:
    dark = is_dark()
    palette = SERIES_DARK if dark else SERIES_LIGHT
    fig = go.Figure()
    for name in wide.columns:
        if name not in chosen:
            fig.add_trace(
                go.Scatter(
                    x=wide.index,
                    y=wide[name],
                    mode="lines",
                    name=name,
                    showlegend=False,
                    line=dict(color=MUTED["dark" if dark else "light"], width=1),
                    hovertemplate=f"{name}: %{{y:.1%}}<extra></extra>",
                )
            )
    for i, name in enumerate(chosen):
        mine = bool(player) and player.casefold() in name.casefold()
        fig.add_trace(
            go.Scatter(
                x=wide.index,
                y=wide[name],
                mode="lines+markers",
                name=name,
                line=dict(color=palette[i % len(palette)], width=3.5 if mine else 2),
                marker=dict(size=6),
                hovertemplate=f"{name}: %{{y:.1%}}<extra></extra>",
            )
        )
    if events:
        fig.add_trace(
            go.Scatter(
                x=[e["timestamp"] for e in events],
                y=[0] * len(events),
                mode="markers",
                name="Game final",
                marker=dict(symbol="diamond", size=9, color="#888888"),
                text=[e["game"] for e in events],
                hovertemplate="Final: %{text}<extra></extra>",
            )
        )
        for e in events:
            fig.add_vline(
                x=e["timestamp"], line=dict(color="#888888", width=1, dash="dot"), opacity=0.35
            )
    fig.update_layout(
        template="plotly_dark" if dark else "plotly_white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=520,
        margin=dict(l=10, r=10, t=10, b=10),
        hovermode="closest",
        yaxis=dict(title="Win probability", tickformat=".0%", rangemode="tozero"),
        xaxis=dict(title=None),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    return fig


def main():
    st.set_page_config(page_title="Live Pick'em Tracker", page_icon="🏈", layout="wide")
    st.title("Live Pick'em Tracker")

    with st.sidebar:
        week = st.number_input("Week", 1, 22, value=int(os.getenv("CONFPICKEM_WEEK", 10)))
        league_id = st.number_input("League ID", value=11465, step=1)
        cookies = st.text_input("Cookies file", "cookies.txt")
        history = st.text_input("History file", f"NFL_Week{week}_live.jsonl")
        player = st.text_input("Your entry (substring)", "Educated Guesses")
        interval = st.number_input("Auto-refresh interval (min)", 1.0, 120.0, 15.0, step=1.0)
        auto = st.toggle("Auto-refresh", value=True)
        top_n = st.slider("Entries highlighted", 2, 8, 6)
        api_key = st.text_input("Odds API key", os.getenv("ODDS_API_KEY", ""), type="password")

    def refresh():
        try:
            snap = lt.take_snapshot(
                week=int(week),
                league_id=int(league_id),
                cookies_file=cookies,
                odds_api_key=api_key or None,
            )
            lt.append_snapshot(history, snap)
            st.session_state.pop("error", None)
        except Exception as e:  # keep the page up; show what went wrong
            st.session_state["error"] = f"{type(e).__name__}: {e}"

    if st.button("Refresh now", type="primary"):
        with st.spinner("Scraping Yahoo, pulling odds, re-scoring..."):
            refresh()

    @st.fragment(run_every=60)
    def board():
        snaps = lt.read_snapshots(history, week=int(week))
        due = (
            not snaps
            or (datetime.now() - datetime.fromisoformat(snaps[-1]["timestamp"])).total_seconds()
            >= interval * 60
        )
        if auto and due and not (snaps and snaps[-1]["all_final"]):
            with st.spinner("Auto-refreshing..."):
                refresh()
            snaps = lt.read_snapshots(history, week=int(week))

        if st.session_state.get("error"):
            st.error(f"Last refresh failed: {st.session_state['error']}")
        if not snaps:
            st.info("No snapshots yet -- click Refresh now (needs every entry locked).")
            return

        last = snaps[-1]
        counts = {
            k: sum(g["status"] == k for g in last["games"]) for k in ("final", "live", "upcoming")
        }
        st.caption(
            f"Updated {last['timestamp']} · {counts['final']} final · {counts['live']} live · "
            f"{counts['upcoming']} upcoming · {len(snaps)} snapshots"
            + (" · all games final" if last["all_final"] else "")
        )
        if last["stale_games"]:
            st.warning(
                "No live line for in-progress game(s), using pregame odds: "
                + ", ".join(last["stale_games"])
            )

        wide = lt.win_pct_series(snaps)
        chosen = st.multiselect(
            "Entries to color",
            list(wide.columns),
            default=pick_entrants(wide, player, top_n),
            max_selections=8,
        )
        st.plotly_chart(
            build_chart(wide, lt.final_events(snaps), chosen, player), use_container_width=True
        )

        with st.expander("Table view"):
            table = pd.DataFrame(last["standings"]).rename(
                columns={
                    "player": "Entry",
                    "win_pct": "Win %",
                    "expected_points": "Exp pts",
                    "locked_points": "Locked pts",
                }
            )
            st.dataframe(
                table,
                use_container_width=True,
                hide_index=True,
                column_config={"Win %": st.column_config.NumberColumn(format="percent")},
            )
            st.dataframe(pd.DataFrame(last["games"]), use_container_width=True, hide_index=True)

    board()


main()
