"""Production-style win features (FV → field_position) — ported for local training/compare."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

REGULATION_SECONDS = 4 * 720


def resolve_shownspace_models_dir() -> Path:
    explicit = (os.getenv("SHOWNSPACE_MODELS_DIR") or "").strip()
    if explicit:
        return Path(explicit).expanduser().resolve()
    candidates = [
        Path.home() / "Desktop" / "shownspace_backend" / "models",
        Path(__file__).resolve().parents[3] / "shownspace_backend" / "models",
        Path(__file__).resolve().parents[2] / "models" / "deployed_source",
    ]
    for path in candidates:
        if (path / "win_model.joblib").is_file():
            return path.resolve()
    raise FileNotFoundError(
        "Could not find ShownSpace production models. Set SHOWNSPACE_MODELS_DIR "
        "to a folder containing win_model.joblib and fv_model.joblib."
    )


def _num(frame: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if col not in frame.columns:
        return pd.Series(default, index=frame.index, dtype=float)
    return pd.to_numeric(frame[col], errors="coerce").fillna(default)


def add_game_time_left(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    quarter = _num(out, "game_quarter", 1)
    time_left = _num(out, "time_left", 720)
    out["game_time_left"] = time_left + (4 - quarter) * 720
    out.loc[quarter == 5, "game_time_left"] = time_left[quarter == 5] - 300
    out.loc[quarter >= 6, "game_time_left"] = 100
    return out


def build_production_win_features(
    throws: pd.DataFrame,
    *,
    fv_model=None,
) -> pd.DataFrame:
    """
    Build features the deployed CombinedWinModel expects.

    If ``fv_model`` is provided, field_position uses FV predictions.
    Otherwise field_position falls back to a distance-based proxy so training
    still works without the production FV artifact.
    """
    frame = throws.copy()
    sort_cols = [
        c
        for c in [
            "GameID",
            "game_quarter",
            "quarter_point",
            "possession_num",
            "possession_throw",
            "event_id",
            "throw_id",
        ]
        if c in frame.columns
    ]
    if sort_cols:
        frame = frame.sort_values(sort_cols).reset_index(drop=True)

    home = _num(frame, "home_team_score")
    away = _num(frame, "away_team_score")
    frame["score_diff"] = home - away

    # Pre-goal score adjustment (matches production advanced_scoring)
    recv_y = _num(frame, "ReceiverY", 50)
    turnover = _num(frame, "turnover")
    is_home = _num(frame, "is_home_team")
    is_goal = (recv_y >= 100) & (turnover == 0)
    frame.loc[is_goal & (is_home == 1), "score_diff"] -= 1
    frame.loc[is_goal & (is_home == 0), "score_diff"] += 1

    frame = add_game_time_left(frame)
    quarter = _num(frame, "game_quarter", 1)
    frame["time_remaining_in_game"] = _num(frame, "game_time_left").where(
        quarter <= 4, _num(frame, "time_left", 720)
    )

    # Distance / possession (production enrichment)
    turnover_dist = 100 - (120 - recv_y).clip(lower=20, upper=100)
    frame["distance_to_endzone"] = (100 - recv_y).where(~(turnover == 1), turnover_dist)
    frame.loc[is_goal, "distance_to_endzone"] = 75

    change_possession = is_goal | (turnover == 1)
    frame["has_possession"] = is_home.astype(float)
    flipped = (1.0 - is_home.astype(float)).to_numpy()
    frame.loc[change_possession, "has_possession"] = flipped[change_possession.to_numpy()]

    if fv_model is not None:
        thrower = frame.copy()
        thrower["thrower_distance_to_endzone"] = 100 - _num(thrower, "ThrowerY", 50)
        receiver = frame.copy()
        receiver["ThrowerX"] = _num(receiver, "ReceiverX")
        receiver["ThrowerY"] = _num(receiver, "ReceiverY")
        receiver["thrower_distance_to_endzone"] = 100 - _num(receiver, "ThrowerY")
        opponent = receiver.copy()
        opponent["ThrowerX"] = -_num(opponent, "ThrowerX")
        opponent["ThrowerY"] = (120 - recv_y).clip(lower=20, upper=100)
        opponent["thrower_distance_to_endzone"] = 100 - _num(opponent, "ThrowerY")

        def _fv(df: pd.DataFrame) -> np.ndarray:
            x = np.column_stack(
                [
                    _num(df, "ThrowerX").to_numpy(),
                    _num(df, "thrower_distance_to_endzone").to_numpy(),
                    _num(df, "time_left", 720).to_numpy(),
                ]
            )
            proba = fv_model.predict_proba(np.nan_to_num(x, nan=0.0))
            proba = np.asarray(proba)
            return proba[:, 1] if proba.ndim > 1 else proba

        fv_receiver = _fv(receiver)
        fv_opponent = _fv(opponent)
        frame["fv_receiver"] = fv_receiver
        frame["fv_opponent"] = fv_opponent

        field = np.where(is_home.to_numpy() >= 0.5, fv_receiver, -fv_receiver)
        # turnovers: flip to opponent FV with home sign convention
        to_mask = (turnover == 1).to_numpy()
        field = np.where(
            to_mask,
            np.where(is_home.to_numpy() >= 0.5, -fv_opponent, fv_opponent),
            field,
        )
        # goals: fixed constants
        goal_mask = is_goal.to_numpy()
        field = np.where(
            goal_mask,
            np.where(is_home.to_numpy() >= 0.5, -0.592, 0.592),
            field,
        )
        frame["field_position"] = field
    else:
        # Proxy: home-centric field advantage from yards + possession sign
        yards = (100 - recv_y).clip(0, 100) / 100.0
        signed = np.where(is_home >= 0.5, 1.0 - yards, -(1.0 - yards))
        frame["field_position"] = signed

    if "GameID" in frame.columns:
        max_scores = (
            frame.groupby("GameID")[["home_team_score", "away_team_score"]]
            .max()
            .rename(columns={"home_team_score": "max_home", "away_team_score": "max_away"})
            .reset_index()
        )
        frame = frame.merge(max_scores, on="GameID", how="left")
        frame["win"] = (frame["max_home"] > frame["max_away"]).astype(int)
        frame = frame.drop(columns=["max_home", "max_away"], errors="ignore")

    # Elapsed axis for charts (0 → end)
    q = _num(frame, "game_quarter", 1)
    tl = _num(frame, "time_left", 720)
    reg = (q.clip(upper=4) - 1).clip(lower=0) * 720 + (720 - tl.clip(0, 720))
    ot = REGULATION_SECONDS + (300 - tl).clip(lower=0)
    frame["elapsed_seconds"] = reg.where(q <= 4, ot).clip(lower=0)
    return frame


def load_deployed_artifacts(model_dir: Path | None = None) -> dict:
    """Load production win (+ optional FV) joblibs with CombinedWinModel import path."""
    model_dir = model_dir or resolve_shownspace_models_dir()
    training_dir = Path(__file__).resolve().parents[1]
    # joblib may pickle as in_game_win_model.CombinedWinModel
    sys.path.insert(0, str(training_dir / "deployed"))
    sys.path.insert(0, str(training_dir))

    # Alias module name expected by historical pickles
    import combined_win_model as _cwm

    sys.modules.setdefault("in_game_win_model", _cwm)
    sys.modules.setdefault("model_training.in_game_win_model", _cwm)

    win = joblib.load(model_dir / "win_model.joblib")
    fv = None
    fv_path = model_dir / "fv_model.joblib"
    if fv_path.is_file():
        fv = joblib.load(fv_path)
    return {"win_model": win, "fv_model": fv, "model_dir": model_dir}
