#!/usr/bin/env python
"""Score one game and write elapsed-time win% SVG (0 → end)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd

from live_win_prob.db import read_table
from live_win_prob.features import build_feature_frame, build_player_skill_priors
from live_win_prob.model import SmoothWinModel
from live_win_prob.plot import write_win_prob_svg
from live_win_prob.smooth import LiveWinSmoother, score_game_live


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--game-id", required=True)
    parser.add_argument("--team", choices=["home", "away"], default="home")
    parser.add_argument("--model", type=Path, default=ROOT / "models" / "smooth_win_model.joblib")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--half-life", type=float, default=25.0)
    args = parser.parse_args()

    if not args.model.is_file():
        raise SystemExit(f"Missing model {args.model}. Run demo_synthetic.py or train_model.py first.")

    throws = read_table("throws", where=f"\"GameID\" = '{args.game_id}'")
    if throws.empty:
        raise SystemExit(f"No throws for {args.game_id}")
    games = read_table("games", where=f"\"GameID\" = '{args.game_id}'")
    priors = build_player_skill_priors(read_table("player_stats"))
    feats = build_feature_frame(throws, player_priors=priors, games=games, team=args.team)
    model = SmoothWinModel.load(args.model)
    scored = score_game_live(
        feats,
        model,
        team=args.team,
        smoother=LiveWinSmoother(half_life_seconds=args.half_life),
    )
    out = args.out or ROOT / "figures" / f"{args.game_id.replace('-', '_')}_{args.team}.svg"
    write_win_prob_svg(
        scored,
        out,
        title=f"{args.game_id} — {args.team} win %",
        team_label=args.team.title(),
    )
    vol = float(scored["win_prob"].diff().abs().mean())
    print(f"Wrote {out}")
    print(f"points={len(scored)} final={scored['win_prob'].iloc[-1]:.3f} mean|Δ|={vol:.4f}")


if __name__ == "__main__":
    main()
