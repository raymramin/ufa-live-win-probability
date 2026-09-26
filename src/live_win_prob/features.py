"""Feature engineering for smooth live win probability."""

from __future__ import annotations

import numpy as np
import pandas as pd

FEATURE_COLS = [
    "score_diff",
    "elapsed_frac",
    "has_possession_home",
    "yards_to_endzone_home",
    "sideline_pressure",
    "offense_skill_edge",
]

REGULATION_SECONDS = 4 * 720  # 2880


def elapsed_seconds_from_clock(
    game_quarter: pd.Series | np.ndarray | float,
    time_left: pd.Series | np.ndarray | float,
) -> pd.Series:
    """Elapsed game seconds from quarter clock (starts at 0)."""
    q = pd.to_numeric(game_quarter, errors="coerce")
    tl = pd.to_numeric(time_left, errors="coerce")
    # Quarters 1-4: each 720s. OT (q>=5): continue past regulation using time_left as remaining in OT period.
    reg = (q.clip(upper=4) - 1).clip(lower=0) * 720 + (720 - tl.clip(lower=0, upper=720))
    ot = REGULATION_SECONDS + (300 - tl).clip(lower=0)  # soft OT handling
    elapsed = reg.where(q <= 4, ot)
    return elapsed.fillna(0.0).clip(lower=0.0)


def _num(frame: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if col not in frame.columns:
        return pd.Series(default, index=frame.index, dtype=float)
    return pd.to_numeric(frame[col], errors="coerce").fillna(default)


def build_feature_frame(
    throws: pd.DataFrame,
    *,
    player_priors: pd.DataFrame | None = None,
    games: pd.DataFrame | None = None,
    team: str = "home",
) -> pd.DataFrame:
    """
    Build model features + bookkeeping columns from a throws-like frame.

    Parameters
    ----------
    throws:
        Must include score, clock, coordinates, turnover flags.
    player_priors:
        Optional columns: full_name (or PlayerID), skill_z
    games:
        Optional GameID, HomeScore, AwayScore for labels / end snap.
    team:
        ``home`` or ``away`` — output ``win_prob`` is always for this team after scoring.
    """
    if throws.empty:
        return pd.DataFrame()

    frame = throws.copy()
    sort_cols = [c for c in ["GameID", "game_quarter", "quarter_point", "possession_num", "possession_throw", "event_id", "throw_id"] if c in frame.columns]
    if sort_cols:
        frame = frame.sort_values(sort_cols).reset_index(drop=True)

    home = _num(frame, "home_team_score")
    away = _num(frame, "away_team_score")
    # Raw goal difference for labels / live smoother context.
    frame["score_diff_raw"] = home - away
    # Soft score edge for the model: ±1 goal stays meaningful, blowouts saturate.
    # Bayesian idea — the scoreboard is evidence, not a linear dial.
    frame["score_diff"] = np.tanh(frame["score_diff_raw"] / 2.0)

    q = _num(frame, "game_quarter", 1)
    tl = _num(frame, "time_left", 720)
    frame["elapsed_seconds"] = elapsed_seconds_from_clock(q, tl)
    # Per-game normalize elapsed into [0, 1+] using observed max (live: use regulation as denom).
    if "GameID" in frame.columns:
        max_elapsed = frame.groupby("GameID")["elapsed_seconds"].transform("max").clip(lower=REGULATION_SECONDS)
    else:
        max_elapsed = max(float(frame["elapsed_seconds"].max()), float(REGULATION_SECONDS))
    frame["elapsed_frac"] = (frame["elapsed_seconds"] / max_elapsed).clip(0.0, 1.5)

    is_home = _num(frame, "is_home_team")
    turnover = _num(frame, "turnover")
    dropped = _num(frame, "DroppedThrow")
    stall = _num(frame, "stall")
    recv_y = _num(frame, "ReceiverY", 50)
    throw_x = _num(frame, "ThrowerX")

    is_goal = (recv_y >= 100) & (turnover == 0)
    change_poss = is_goal | (turnover == 1) | (stall == 1)
    # After goal/turnover possession flips away from the throwing team.
    has_poss_throwing = is_home.copy()
    has_poss_throwing = has_poss_throwing.where(~change_poss, 1.0 - is_home)
    # For ongoing throws, throwing team has possession.
    frame["has_possession_home"] = is_home.where(~change_poss, has_poss_throwing)
    frame["is_goal_event"] = is_goal.astype(int)
    frame["is_turnover_event"] = ((turnover == 1) | (stall == 1) | (dropped == 1)).astype(int)

    # Home-centric yards to endzone: when home has disc, use 100-ReceiverY; when away has disc, mirror.
    yards_for_thrower = (100 - recv_y).clip(0, 100)
    yards_home = yards_for_thrower.where(frame["has_possession_home"] >= 0.5, (100 - yards_for_thrower))
    frame["yards_to_endzone_home"] = yards_home
    frame["sideline_pressure"] = (throw_x.abs() / 27.5).clip(0.0, 1.0)

    # Skill prior: map thrower name → skill_z
    skill = pd.Series(0.0, index=frame.index, dtype=float)
    if player_priors is not None and not player_priors.empty and "Thrower" in frame.columns:
        priors = player_priors.copy()
        if "skill_z" not in priors.columns:
            priors["skill_z"] = 0.0
        key = "full_name" if "full_name" in priors.columns else None
        if key:
            lookup = priors.drop_duplicates(key).set_index(key)["skill_z"]
            names = frame["Thrower"].astype(str).str.strip()
            skill = names.map(lookup).fillna(0.0).astype(float)
    # Positive skill helps the team that currently possesses.
    sign = np.where(frame["has_possession_home"] >= 0.5, 1.0, -1.0)
    frame["offense_skill_edge"] = skill.to_numpy() * sign

    # Labels
    frame["home_won"] = np.nan
    if games is not None and not games.empty and "GameID" in frame.columns:
        g = games.copy()
        g["home_won"] = (
            pd.to_numeric(g["HomeScore"], errors="coerce") > pd.to_numeric(g["AwayScore"], errors="coerce")
        ).astype(float)
        frame = frame.merge(g[["GameID", "home_won"]], on="GameID", how="left", suffixes=("", "_g"))
        if "home_won_g" in frame.columns:
            frame["home_won"] = frame["home_won"].fillna(frame["home_won_g"])
            frame = frame.drop(columns=["home_won_g"])

    frame["team"] = team
    return frame


def build_player_skill_priors(player_stats: pd.DataFrame) -> pd.DataFrame:
    """Collapse player_stats into a name→skill_z table."""
    if player_stats is None or player_stats.empty:
        return pd.DataFrame(columns=["full_name", "skill_z"])

    frame = player_stats.copy()
    if "full_name" not in frame.columns:
        if {"FirstName", "LastName"}.issubset(frame.columns):
            frame["full_name"] = (
                frame["FirstName"].astype(str).str.strip() + " " + frame["LastName"].astype(str).str.strip()
            )
        else:
            return pd.DataFrame(columns=["full_name", "skill_z"])

    # Prefer most recent year per player name
    if "Year" in frame.columns:
        frame = frame.sort_values("Year").groupby("full_name", as_index=False).tail(1)

    comp = pd.to_numeric(frame.get("completion_percentage"), errors="coerce")
    oe = pd.to_numeric(frame.get("OE"), errors="coerce")
    # z-ish scale without leaking: center/scale within table
    def _z(s: pd.Series) -> pd.Series:
        s = s.astype(float)
        mu, sd = s.mean(skipna=True), s.std(skipna=True)
        if sd is None or sd == 0 or np.isnan(sd):
            return s.fillna(0.0) * 0.0
        return ((s - mu) / sd).fillna(0.0).clip(-2.5, 2.5)

    skill = 0.6 * _z(comp) + 0.4 * _z(oe)
    out = pd.DataFrame({"full_name": frame["full_name"].astype(str).str.strip(), "skill_z": skill.to_numpy()})
    return out.drop_duplicates("full_name")
