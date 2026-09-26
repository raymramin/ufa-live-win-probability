"""Event-gated, context-aware EMA smoother for live charts."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class LiveWinSmoother:
    """
    Convert raw model probabilities into a stream suitable for live display.

    Bayesian / context rules baked into step sizes:
    - Ordinary throws: tiny allowed step + EMA (weak new evidence)
    - Goals: larger steps that grow later in the game
    - Turnovers / negative throws: moderate steps, damped early and when leading
    - Final row: optional snap to 0 or 1
    """

    half_life_seconds: float = 30.0
    max_step_quiet: float = 0.012
    max_step_goal: float = 0.12
    max_step_turnover: float = 0.045
    initial: float = 0.50

    def _max_step(
        self,
        *,
        is_goal: bool,
        is_turnover: bool,
        score_diff_raw: float,
        elapsed_frac: float,
    ) -> float:
        el = float(np.clip(elapsed_frac, 0.0, 1.5))
        # How "safe" the scoreboard already feels (raw goals, capped).
        lead_cushion = float(min(1.0, abs(score_diff_raw) / 3.0))

        if is_goal:
            # Same goal means more late than early — match score × time prior.
            return float(self.max_step_goal * (0.55 + 0.45 * min(el, 1.0)))

        if is_turnover:
            # Negative throws should not crater WP: more recovery time early,
            # and a lead late already carries most of the belief.
            early_dampen = 1.0 - 0.35 * (1.0 - min(el, 1.0))
            lead_dampen = 1.0 - 0.45 * lead_cushion * min(el, 1.0)
            return float(self.max_step_turnover * early_dampen * lead_dampen)

        return float(self.max_step_quiet)

    def smooth_series(
        self,
        elapsed_seconds: np.ndarray | pd.Series,
        p_raw: np.ndarray | pd.Series,
        *,
        is_goal: np.ndarray | pd.Series | None = None,
        is_turnover: np.ndarray | pd.Series | None = None,
        score_diff_raw: np.ndarray | pd.Series | None = None,
        elapsed_frac: np.ndarray | pd.Series | None = None,
        final_home_won: float | None = None,
        team: str = "home",
    ) -> np.ndarray:
        t = np.asarray(pd.to_numeric(elapsed_seconds, errors="coerce"), dtype=float)
        p = np.asarray(pd.to_numeric(p_raw, errors="coerce"), dtype=float)
        n = len(p)
        goals = np.zeros(n, dtype=bool) if is_goal is None else np.asarray(is_goal, dtype=bool)
        turns = np.zeros(n, dtype=bool) if is_turnover is None else np.asarray(is_turnover, dtype=bool)
        if score_diff_raw is None:
            sd = np.zeros(n, dtype=float)
        else:
            sd = np.asarray(pd.to_numeric(score_diff_raw, errors="coerce"), dtype=float)
            sd = np.nan_to_num(sd, nan=0.0)
        if elapsed_frac is None:
            # Fallback from elapsed seconds vs regulation if frac not provided.
            ef = np.clip(t / 2880.0, 0.0, 1.5)
            ef = np.nan_to_num(ef, nan=0.5)
        else:
            ef = np.asarray(pd.to_numeric(elapsed_frac, errors="coerce"), dtype=float)
            ef = np.nan_to_num(ef, nan=0.5)

        out = np.empty(n, dtype=float)
        live = float(self.initial)
        prev_t = float(t[0]) if n else 0.0

        for i in range(n):
            if not np.isfinite(p[i]):
                out[i] = live
                continue
            dt = max(0.0, float(t[i]) - prev_t) if np.isfinite(t[i]) else 0.0
            prev_t = float(t[i]) if np.isfinite(t[i]) else prev_t

            max_step = self._max_step(
                is_goal=bool(goals[i]),
                is_turnover=bool(turns[i]),
                score_diff_raw=float(sd[i]),
                elapsed_frac=float(ef[i]),
            )
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
        score_ctx = g["score_diff_raw"] if "score_diff_raw" in g.columns else g.get("score_diff")
        g["win_prob"] = smoother.smooth_series(
            g["elapsed_seconds"].to_numpy(),
            g["win_prob_raw"].to_numpy(),
            is_goal=g.get("is_goal_event"),
            is_turnover=g.get("is_turnover_event"),
            score_diff_raw=score_ctx,
            elapsed_frac=g.get("elapsed_frac"),
            final_home_won=final,
            team=team,
        )
        pieces.append(g)

    return pd.concat(pieces, ignore_index=True) if pieces else out
