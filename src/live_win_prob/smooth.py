"""Event-gated EMA smoother for live charts."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class LiveWinSmoother:
    """
    Convert raw model probabilities into a stream suitable for live display.

    - Ordinary throws: tiny allowed step + EMA
    - Goals / turnovers: larger allowed step
    - Final row: optional snap to 0 or 1
    """

    half_life_seconds: float = 25.0
    max_step_quiet: float = 0.015
    max_step_event: float = 0.10
    initial: float = 0.50

    def smooth_series(
        self,
        elapsed_seconds: np.ndarray | pd.Series,
        p_raw: np.ndarray | pd.Series,
        *,
        is_goal: np.ndarray | pd.Series | None = None,
        is_turnover: np.ndarray | pd.Series | None = None,
        final_home_won: float | None = None,
        team: str = "home",
    ) -> np.ndarray:
        t = np.asarray(pd.to_numeric(elapsed_seconds, errors="coerce"), dtype=float)
        p = np.asarray(pd.to_numeric(p_raw, errors="coerce"), dtype=float)
        n = len(p)
        goals = np.zeros(n, dtype=bool) if is_goal is None else np.asarray(is_goal, dtype=bool)
        turns = np.zeros(n, dtype=bool) if is_turnover is None else np.asarray(is_turnover, dtype=bool)

        out = np.empty(n, dtype=float)
        live = float(self.initial)
        prev_t = float(t[0]) if n else 0.0

        for i in range(n):
            if not np.isfinite(p[i]):
                out[i] = live
                continue
            dt = max(0.0, float(t[i]) - prev_t) if np.isfinite(t[i]) else 0.0
            prev_t = float(t[i]) if np.isfinite(t[i]) else prev_t

            event = bool(goals[i] or turns[i])
            max_step = self.max_step_event if event else self.max_step_quiet
            capped = float(np.clip(p[i], live - max_step, live + max_step))

            if self.half_life_seconds <= 0:
                alpha = 1.0
            else:
                alpha = 1.0 - (0.5 ** (dt / self.half_life_seconds))
                alpha = float(np.clip(alpha, 0.05, 1.0))

            live = (1.0 - alpha) * live + alpha * capped
            live = float(np.clip(live, 0.0, 1.0))
            out[i] = live

        if final_home_won is not None and n:
            end = float(final_home_won) if team == "home" else 1.0 - float(final_home_won)
            out[-1] = float(np.clip(end, 0.0, 1.0))
        return out


def score_game_live(
    feature_frame: pd.DataFrame,
    model,
    *,
    team: str = "home",
    smoother: LiveWinSmoother | None = None,
) -> pd.DataFrame:
    """Attach raw + live win probability columns for one or more games."""
    smoother = smoother or LiveWinSmoother()
    out = feature_frame.copy()
    out["win_prob_raw"] = model.predict_team(out, team=team)

    pieces = []
    group_key = "GameID" if "GameID" in out.columns else None
    groups = out.groupby(group_key, sort=False) if group_key else [(None, out)]

    for _, g in groups:
        g = g.sort_values("elapsed_seconds").copy()
        final = None
        if "home_won" in g.columns and g["home_won"].notna().any():
            final = float(g["home_won"].dropna().iloc[-1])
        g["win_prob"] = smoother.smooth_series(
            g["elapsed_seconds"].to_numpy(),
            g["win_prob_raw"].to_numpy(),
            is_goal=g.get("is_goal_event"),
            is_turnover=g.get("is_turnover_event"),
            final_home_won=final,
            team=team,
        )
        pieces.append(g)

    return pd.concat(pieces, ignore_index=True) if pieces else out
