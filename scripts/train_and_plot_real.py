#!/usr/bin/env python
"""Train on DB + plot a real game (default SLC-OAK) without writing full parquet."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd

from live_win_prob.db import read_table
from live_win_prob.features import build_feature_frame, build_player_skill_priors
from live_win_prob.model import train_smooth_win_model
from live_win_prob.plot import write_win_prob_html, write_win_prob_svg
from live_win_prob.smooth import LiveWinSmoother, score_game_live


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--game-id", default="2023-05-13-SLC-OAK")
    parser.add_argument("--team", choices=["home", "away"], default="home")
    parser.add_argument("--max-train-games", type=int, default=400)
    parser.add_argument("--lam", type=float, default=8.0)
    args = parser.parse_args()

    print("Loading tables...")
    throws = read_table("throws")
    games = read_table("games")
    priors = build_player_skill_priors(read_table("player_stats"))

    # Keep games with decisive scores
    g = games.dropna(subset=["GameID", "HomeScore", "AwayScore"]).copy()
    g = g[g["HomeScore"] != g["AwayScore"]]
    # Prefer years overlapping throws
    if "Year" in g.columns:
        g = g.sort_values("Year")
    train_ids = g["GameID"].astype(str).tail(args.max_train_games).tolist()
    if args.game_id not in train_ids:
        train_ids.append(args.game_id)

    sub = throws[throws["GameID"].astype(str).isin(train_ids)].copy()
    print(f"Training throws={len(sub)} games={sub['GameID'].nunique()}")
    feats = build_feature_frame(sub, player_priors=priors, games=g)
    labeled = feats.dropna(subset=["home_won"])
    print(f"Labeled rows={len(labeled)} win-rate={labeled['home_won'].mean():.3f}")

    model_path = ROOT / "models" / "smooth_win_model.joblib"
    model = train_smooth_win_model(labeled, output_path=model_path, gam_lam=args.lam)
    print(f"Saved {model_path}")

    game = feats[feats["GameID"].astype(str) == args.game_id].copy()
    if game.empty:
        raise SystemExit(f"No features for {args.game_id}")
    scored = score_game_live(
        game,
        model,
        team=args.team,
        smoother=LiveWinSmoother(half_life_seconds=25.0),
    )
    out = ROOT / "figures" / f"{args.game_id.replace('-', '_')}_{args.team}_live.svg"
    html = ROOT / "figures" / f"{args.game_id.replace('-', '_')}_{args.team}_live.html"
    write_win_prob_svg(
        scored,
        out,
        title=f"{args.game_id} — live smooth {args.team} win %",
        team_label=args.team.title(),
        team=args.team,
    )
    write_win_prob_html(
        scored,
        html,
        title=f"{args.game_id} — smooth live win probability",
        team=args.team,
    )
    raw_vol = float(scored["win_prob_raw"].diff().abs().mean())
    live_vol = float(scored["win_prob"].diff().abs().mean())
    print(f"Wrote {out}")
    print(f"Wrote {html}")
    print(f"raw mean|step|={raw_vol:.4f}  live mean|step|={live_vol:.4f}  final={scored['win_prob'].iloc[-1]:.3f}")
    print(f"elapsed 0 -> {scored['elapsed_seconds'].max():.0f}s")


if __name__ == "__main__":
    main()
