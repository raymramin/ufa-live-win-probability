"""Deployed (Game Center) CombinedWinModel — ported from shownspace_backend."""

from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
from pygam import LogisticGAM, s
from xgboost import XGBClassifier


class CombinedWinModel:
    """Production win model: GAM when clock > 30s, else shallow XGB."""

    def __init__(self, locfit_model: LogisticGAM, xgb_model: XGBClassifier, tree_cutoff: int = 30):
        self.locfit_model = locfit_model
        self.xgb_model = xgb_model
        self.tree_cutoff = tree_cutoff

    def predict(self, new_data: pd.DataFrame) -> np.ndarray:
        frame = new_data.copy()
        frame["row_id"] = np.arange(len(frame))
        time_remaining = pd.to_numeric(frame["time_remaining_in_game"], errors="coerce")

        locfit_input = frame[time_remaining > self.tree_cutoff].copy()
        tree_input = frame[time_remaining <= self.tree_cutoff].copy()
        preds: list[pd.DataFrame] = []

        if not locfit_input.empty:
            x = locfit_input[["score_diff", "time_remaining_in_game", "field_position"]].fillna(0).values
            proba = self.locfit_model.predict_proba(x)
            locfit_input["pred"] = proba[:, 1] if getattr(proba, "ndim", 1) > 1 else proba
            preds.append(locfit_input)

        if not tree_input.empty:
            x = tree_input[
                ["score_diff", "time_remaining_in_game", "distance_to_endzone", "has_possession"]
            ].fillna(0).values
            tree_input["pred"] = self.xgb_model.predict_proba(x)[:, 1]
            preds.append(tree_input)

        if not preds:
            return np.array([])
        combined = pd.concat(preds, ignore_index=True).sort_values("row_id")
        return combined["pred"].to_numpy(dtype=float)


def train_combined_win_model(
    modelling_data: pd.DataFrame,
    output_path: str = "models/deployed_win_model.joblib",
    random_state: int = 42,
) -> CombinedWinModel:
    """Retrain a production-style CombinedWinModel (for local experiments only)."""
    tree_cutoff = 30
    locfit_data = modelling_data[modelling_data["time_remaining_in_game"] > tree_cutoff].copy()
    tree_data = modelling_data[modelling_data["time_remaining_in_game"] <= tree_cutoff].copy()

    locfit_model = LogisticGAM(
        s(0, n_splines=5, lam=0.1) + s(1, n_splines=6, lam=0.1) + s(2, n_splines=4, lam=0.1)
    )
    locfit_model.fit(
        locfit_data[["score_diff", "time_remaining_in_game", "field_position"]].fillna(0).values,
        locfit_data["win"].astype(int).values,
    )

    xgb_model = XGBClassifier(
        objective="binary:logistic",
        n_estimators=75,
        max_depth=2,
        learning_rate=0.1,
        random_state=int(random_state),
        n_jobs=1,
        subsample=1.0,
        colsample_bytree=1.0,
    )
    xgb_model.fit(
        tree_data[
            ["score_diff", "time_remaining_in_game", "distance_to_endzone", "has_possession"]
        ].fillna(0).values,
        tree_data["win"].astype(int).values,
    )

    model = CombinedWinModel(locfit_model, xgb_model, tree_cutoff)
    joblib.dump(model, output_path)
    return model
