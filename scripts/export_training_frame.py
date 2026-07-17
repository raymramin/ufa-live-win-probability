#!/usr/bin/env python
"""Export a training frame from ShownSpace tables into data/training_frame.parquet."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from live_win_prob.db import read_table
from live_win_prob.features import build_feature_frame, build_player_skill_priors


def main() -> None:
    print("Loading throws / games / player_stats ...")
    throws = read_table("throws")
    games = read_table("games")
    player_stats = read_table("player_stats")
    priors = build_player_skill_priors(player_stats)
    feats = build_feature_frame(throws, player_priors=priors, games=games)
    out = ROOT / "data" / "training_frame.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    feats.to_parquet(out, index=False)
    print(f"Wrote {out} rows={len(feats)} games={feats['GameID'].nunique() if 'GameID' in feats else '?'}")
    print(f"Labeled rows={feats['home_won'].notna().sum()}")


if __name__ == "__main__":
    main()
