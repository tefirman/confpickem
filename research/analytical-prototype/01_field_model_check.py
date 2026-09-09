#!/usr/bin/env python
"""Does the simulator's opponent model match real picks in the cache?

Before any analytical P(win), validate the field model. For each backtest week
and each real player we have BOTH:
  - the model's predicted P(pick home) and E[confidence] per game
    (from that player's skill knobs + the game's vegas/crowd numbers)
  - what they actually picked / how much confidence they actually assigned

Compare. If the model is badly off, the analytical P(win) built on it is sand.

Run from repo root:  python research/analytical-prototype/01_field_model_check.py
"""
import sys, warnings, json
sys.path.insert(0, "src")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import confpickem.yahoo_pickem_scraper as _scraper
_orig = _scraper.PageCache.get_cached_content
_scraper.PageCache.get_cached_content = (
    lambda self, pt, wk, expiration=3600: _orig(self, pt, wk, expiration=10 ** 13))

from confpickem.yahoo_pickem_scraper import YahooPickEm
from confpickem.yahoo_pickem_integration import convert_yahoo_to_simulator_format

LEAGUE_ID = 15435
COOKIES = "cookies.txt"
CACHE = {2024: "PickEmCache2024", 2025: "PickEmCache2025"}
WEEKS = [(2024, 3), (2024, 6), (2024, 9), (2024, 12), (2024, 15),
         (2025, 2), (2025, 5), (2025, 9), (2025, 12), (2025, 15)]

_SKILLS = {}


def skills(year):
    if year not in _SKILLS:
        _SKILLS[year] = json.load(open("player_skills_{}.json".format(year)))["player_skills"]
    return _SKILLS[year]


def model_pick_home_prob(vegas_home, crowd_home_pct, crowd_following):
    """Simulator's mean P(pick home) -- base blend, noise integrates out to ~0 mean."""
    return np.clip(vegas_home * (1 - crowd_following) + crowd_home_pct * crowd_following,
                   0.0, 1.0)


def model_confidence_rank(vegas_home, crowd_home_conf, crowd_away_conf,
                          picked_home, conf_following, n_games):
    """Simulator's pre-noise 'confidence score' for a game (higher -> more points).

    Mirrors the blended_conf formula in simulate_picks. Returned as a raw score;
    the sim then ranks these into 1..n. We compare the *ranking*, not the raw value.
    """
    chosen = np.where(picked_home, crowd_home_conf, crowd_away_conf)
    opposing = np.where(picked_home, crowd_away_conf, crowd_home_conf)
    conf_diff = (chosen - opposing) / (chosen + opposing)
    vegas_conf = np.abs(vegas_home - 0.5) * 2 * n_games
    return chosen * (1 + conf_diff) * conf_following + vegas_conf * (1 - conf_following)


def main():
    rows = []
    for (year, week) in WEEKS:
        y = YahooPickEm(week=week, league_id=LEAGUE_ID, cookies_file=COOKIES,
                        cache_dir=CACHE[year])
        gdf = convert_yahoo_to_simulator_format(y, ignore_results=False)
        n = len(gdf)
        sk = skills(year)

        home = gdf["home_team"].tolist()
        away = gdf["away_team"].tolist()
        vegas_home = gdf["vegas_win_prob"].to_numpy()
        crowd_home_pct = gdf["crowd_home_pick_pct"].to_numpy()
        chc = gdf["crowd_home_confidence"].to_numpy()
        cac = gdf["crowd_away_confidence"].to_numpy()

        for _, prow in y.players.iterrows():
            name = prow["player_name"]
            s = sk.get(name)
            if s is None:
                continue
            cf = s["crowd_following"]
            conf_foll = s["confidence_following"]

            picked_home = np.zeros(n, dtype=bool)
            actual_conf = np.zeros(n, dtype=float)
            have = np.zeros(n, dtype=bool)
            for gi in range(n):
                pick = prow.get("game_{}_pick".format(gi + 1))
                conf = prow.get("game_{}_confidence".format(gi + 1))
                if pd.isna(pick) or pd.isna(conf) or conf == 0:
                    continue
                have[gi] = True
                picked_home[gi] = (pick == home[gi])
                actual_conf[gi] = float(conf)

            if have.sum() < n:      # skip players who sat out games this week
                continue

            # --- pick-side check ---
            p_home = model_pick_home_prob(vegas_home, crowd_home_pct, cf)
            # Brier + accuracy of "model says pick home if p_home > .5"
            brier = np.mean((p_home - picked_home.astype(float)) ** 2)
            pred_home = p_home > 0.5
            pick_acc = np.mean(pred_home == picked_home)

            # --- confidence-rank check ---
            model_score = model_confidence_rank(
                vegas_home, chc, cac, picked_home, conf_foll, n)
            # highest score -> n points, lowest -> 1 point
            model_pts = pd.Series(model_score).rank(method="first").to_numpy()
            conf_rank_corr = np.corrcoef(model_pts, actual_conf)[0, 1]
            conf_mae = np.mean(np.abs(model_pts - actual_conf))

            rows.append(dict(
                year=year, week=week, player=name,
                brier=brier, pick_acc=pick_acc,
                conf_rank_corr=conf_rank_corr, conf_mae=conf_mae))

    df = pd.DataFrame(rows)
    df.to_csv("research/analytical-prototype/01_field_model_check.csv", index=False)

    print("=== FIELD MODEL vs REAL PICKS  (n = {} player-weeks) ===\n".format(len(df)))
    print("PICK SIDE  (model: P(pick home) from vegas/crowd blend + player's crowd_following)")
    print("  mean Brier score      {:.3f}   (0 = perfect, 0.25 = coin flip)".format(df.brier.mean()))
    print("  mean pick accuracy    {:.1%}   (model's argmax vs actual pick)".format(df.pick_acc.mean()))
    print("  worst-decile accuracy {:.1%}".format(df.pick_acc.quantile(0.1)))
    print()
    print("CONFIDENCE  (model ranks a blended score into 1..n; compare to actual points)")
    print("  mean rank correlation {:.3f}".format(df.conf_rank_corr.mean()))
    print("  mean abs point error  {:.1f}  points".format(df.conf_mae.mean()))
    print("  median abs point error {:.1f} points".format(df.conf_mae.median()))
    print()
    # baseline: how well does "everyone picks the vegas favorite at crowd-implied confidence" do?
    print("For reference, a naive baseline check is in the CSV; inspect per-week if these look off.")


if __name__ == "__main__":
    main()
