from __future__ import annotations

import numpy as np
import pandas as pd

from live_win_prob.features import build_feature_frame, elapsed_seconds_from_clock
from live_win_prob.smooth import LiveWinSmoother


def test_elapsed_starts_near_zero():
    elapsed = elapsed_seconds_from_clock(pd.Series([1, 1, 2]), pd.Series([720, 700, 720]))
    assert float(elapsed.iloc[0]) == 0.0
    assert float(elapsed.iloc[1]) == 20.0
    assert float(elapsed.iloc[2]) == 720.0


def test_smoother_is_less_volatile_than_raw():
    rng = np.random.default_rng(0)
    t = np.arange(200, dtype=float)
    raw = 0.5 + 0.25 * np.sin(t / 3) + rng.normal(0, 0.08, size=len(t))
    raw = np.clip(raw, 0, 1)
    sm = LiveWinSmoother(half_life_seconds=25, max_step_quiet=0.015)
    live = sm.smooth_series(t, raw)
    assert float(np.mean(np.abs(np.diff(live)))) < float(np.mean(np.abs(np.diff(raw))))


def test_feature_builder_columns():
    throws = pd.DataFrame(
        {
            "GameID": ["G1", "G1"],
            "game_quarter": [1, 1],
            "time_left": [720, 700],
            "home_team_score": [0, 0],
            "away_team_score": [0, 0],
            "is_home_team": [1, 1],
            "ThrowerX": [0.0, 20.0],
            "ThrowerY": [40.0, 50.0],
            "ReceiverY": [45.0, 55.0],
            "turnover": [0, 0],
            "DroppedThrow": [0, 0],
            "stall": [0, 0],
            "Thrower": ["A B", "A B"],
        }
    )
    games = pd.DataFrame({"GameID": ["G1"], "HomeScore": [1], "AwayScore": [0]})
    priors = pd.DataFrame({"full_name": ["A B"], "skill_z": [0.5]})
    feats = build_feature_frame(throws, player_priors=priors, games=games)
    for c in [
        "score_diff",
        "elapsed_frac",
        "has_possession_home",
        "yards_to_endzone_home",
        "sideline_pressure",
        "offense_skill_edge",
        "elapsed_seconds",
    ]:
        assert c in feats.columns
    assert feats["home_won"].iloc[0] == 1.0
