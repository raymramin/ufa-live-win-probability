"""Smooth LogisticGAM win model."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from pygam import LogisticGAM, s, te

from live_win_prob.features import FEATURE_COLS


class SmoothWinModel:
    """High-regularization GAM for live-friendly win probability."""

    def __init__(self, gam: LogisticGAM, feature_cols: list[str] | None = None):
        self.gam = gam
        self.feature_cols = list(feature_cols or FEATURE_COLS)

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        x = frame[self.feature_cols].fillna(0.0).to_numpy(dtype=float)
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        proba = self.gam.predict_proba(x)
        proba = np.asarray(proba, dtype=float)
        if proba.ndim > 1:
            proba = proba[:, 1]
        return np.clip(proba, 0.0, 1.0)

    def predict_team(self, frame: pd.DataFrame, team: str = "home") -> np.ndarray:
        p_home = self.predict_proba(frame)
        if team == "away":
            return 1.0 - p_home
        return p_home

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @staticmethod
    def load(path: str | Path) -> "SmoothWinModel":
        return joblib.load(path)


def train_smooth_win_model(
    feature_frame: pd.DataFrame,
    *,
    output_path: str | Path | None = None,
    gam_lam: float = 8.0,
) -> SmoothWinModel:
    """
    Train a heavily smoothed LogisticGAM.

    ``gam_lam`` default is intentionally high so field/skill effects stay gentle.
    """
    frame = feature_frame.dropna(subset=FEATURE_COLS + ["home_won"]).copy()
    if frame.empty:
        raise ValueError("No training rows with features + home_won label")
    if frame["home_won"].nunique() < 2:
        raise ValueError("Training labels must include both wins and losses")

    x = frame[FEATURE_COLS].fillna(0.0).to_numpy(dtype=float)
    y = frame["home_won"].astype(int).to_numpy()

    # te(score_edge, time) is the Bayesian backbone: a one-goal lead late
    # is stronger evidence than the same lead early. Field/skill terms stay weak.
    gam = LogisticGAM(
        te(0, 1, n_splines=[8, 10], lam=gam_lam * 0.7)
        + s(2, n_splines=4, lam=gam_lam * 1.8)  # possession
        + s(3, n_splines=5, lam=gam_lam * 2.5)  # yards
        + s(4, n_splines=4, lam=gam_lam * 3.0)  # sideline
        + s(5, n_splines=4, lam=gam_lam * 3.0)  # skill
    )
    gam.fit(x, y)
    model = SmoothWinModel(gam)
    if output_path is not None:
        model.save(output_path)
    return model
