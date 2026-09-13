#!/usr/bin/env python
"""Retrain a production-style CombinedWinModel into models/ (gitignored)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "training" / "deployed"))

from live_win_prob.db import read_table
from production_features import build_production_win_features
from combined_win_model import train_combined_win_model


def main() -> None:
    throws = read_table("throws")
    games = read_table("games")
    # Use a manageable recent subset
    g = games.dropna(subset=["GameID", "HomeScore", "AwayScore"]).copy()
    g = g[g["HomeScore"] != g["AwayScore"]]
    if "Year" in g.columns:
        g = g.sort_values("Year")
    ids = g["GameID"].astype(str).tail(300).tolist()
    sub = throws[throws["GameID"].astype(str).isin(ids)].copy()
    feats = build_production_win_features(sub, fv_model=None)
    feats = feats.merge(g[["GameID"]], on="GameID", how="inner")
    out = ROOT / "models" / "deployed_win_model.joblib"
    out.parent.mkdir(parents=True, exist_ok=True)
    train_combined_win_model(feats.dropna(subset=["win"]), output_path=str(out))
    print(f"Wrote {out} rows={len(feats)}")


if __name__ == "__main__":
    main()
