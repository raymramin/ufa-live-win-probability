#!/usr/bin/env python
"""Offline demo: synthetic close game → train → smooth live chart."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd

from live_win_prob.features import build_feature_frame
from live_win_prob.model import train_smooth_win_model
from live_win_prob.plot import write_win_prob_svg
from live_win_prob.smooth import score_game_live


def _synthetic_history(n_games: int = 40, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for g in range(n_games):
        gid = f"SYN-{g:03d}"
        home_won = int(rng.random() > 0.45)
        score_h = score_a = 0
        elapsed = 0.0
        poss_home = 1
        # ~35 possessions
        for p in range(35):
            yards = float(rng.uniform(20, 90))
            side = float(rng.uniform(0, 1))
            skill = float(rng.normal(0, 1))
            # possession length 4–10 throws
            for t in range(int(rng.integers(4, 10))):
                elapsed += float(rng.uniform(4, 12))
                yards = max(5.0, yards - float(rng.uniform(2, 12)))
                side = float(np.clip(side + rng.normal(0, 0.08), 0, 1))
                goal = yards < 12 and rng.random() < 0.35
                turn = (not goal) and rng.random() < 0.12
                q = min(4, 1 + int(elapsed // 720))
                time_left = max(0.0, 720 - (elapsed % 720))
                rows.append(
                    {
                        "GameID": gid,
                        "game_quarter": q,
                        "time_left": time_left,
                        "quarter_point": p,
                        "possession_num": p,
                        "possession_throw": t,
                        "home_team_score": score_h,
                        "away_team_score": score_a,
                        "is_home_team": poss_home,
                        "ThrowerX": (side * 27.5) * (1 if rng.random() > 0.5 else -1),
                        "ThrowerY": 100 - yards,
                        "ReceiverX": 0.0,
                        "ReceiverY": 100 - max(0, yards - 8),
                        "turnover": 1 if turn else 0,
                        "DroppedThrow": 0,
                        "stall": 0,
                        "Thrower": "Ace Handler" if skill > 0 else "New Player",
                        "home_won_true": home_won,
                    }
                )
                if goal:
                    if poss_home:
                        score_h += 1
                    else:
                        score_a += 1
                    rows[-1]["ReceiverY"] = 105
                    rows[-1]["home_team_score"] = score_h
                    rows[-1]["away_team_score"] = score_a
                    poss_home = 1 - poss_home
                    yards = 80
                    break
                if turn:
                    poss_home = 1 - poss_home
                    yards = 100 - yards
                    break
        # force final score consistency with label
        if home_won and score_h <= score_a:
            score_h = score_a + 1
        if (not home_won) and score_a <= score_h:
            score_a = score_h + 1
        for r in rows:
            if r["GameID"] == gid:
                r["final_home"] = score_h
                r["final_away"] = score_a
    return pd.DataFrame(rows)


def main() -> None:
    raw = _synthetic_history()
    games = (
        raw.groupby("GameID", as_index=False)
        .agg(HomeScore=("final_home", "max"), AwayScore=("final_away", "max"))
    )
    priors = pd.DataFrame(
        {
            "full_name": ["Ace Handler", "New Player"],
            "skill_z": [1.2, -1.1],
            "completion_percentage": [0.94, 0.82],
            "OE": [0.55, 0.35],
        }
    )
    feats = build_feature_frame(raw, player_priors=priors, games=games)
    # overwrite label from synthetic truth
    truth = raw.groupby("GameID")["home_won_true"].max()
    feats["home_won"] = feats["GameID"].map(truth)

    model_path = ROOT / "models" / "smooth_win_model.joblib"
    model = train_smooth_win_model(feats, output_path=model_path, gam_lam=6.0)
    print(f"Saved {model_path}")

    demo_id = feats["GameID"].iloc[0]
    game = feats[feats["GameID"] == demo_id].copy()
    scored = score_game_live(game, model, team="home")
    fig = write_win_prob_svg(
        scored,
        ROOT / "figures" / "demo_synthetic_home.svg",
        title=f"Synthetic demo — {demo_id}",
        team_label="Home",
    )
    print(f"Wrote {fig}")
    print(
        "volatility mean|step|=",
        round(float(scored["win_prob"].diff().abs().mean()), 4),
        "final=",
        round(float(scored["win_prob"].iloc[-1]), 3),
    )


if __name__ == "__main__":
    main()
