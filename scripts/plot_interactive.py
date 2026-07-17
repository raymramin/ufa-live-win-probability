#!/usr/bin/env python
"""Score one game and write SVG + interactive HTML (reuse model if present)."""

from __future__ import annotations

import argparse
import sys
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from live_win_prob.db import read_table
from live_win_prob.features import build_feature_frame, build_player_skill_priors
from live_win_prob.model import SmoothWinModel, train_smooth_win_model
from live_win_prob.plot import attach_player_display_names, write_win_prob_html, write_win_prob_svg
from live_win_prob.smooth import LiveWinSmoother, score_game_live


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--game-id", default="2023-05-13-SLC-OAK")
    parser.add_argument("--team", choices=["home", "away"], default="home")
    parser.add_argument("--model", type=Path, default=ROOT / "models" / "smooth_win_model.joblib")
    parser.add_argument("--open", action="store_true", help="Open interactive HTML in browser")
    parser.add_argument("--retrain", action="store_true")
    parser.add_argument("--max-train-games", type=int, default=400)
    parser.add_argument("--lam", type=float, default=8.0)
    args = parser.parse_args()

    print("Loading throws / games / player_stats / players...")
    throws = read_table("throws")
    games = read_table("games")
    players = read_table("players")
    priors = build_player_skill_priors(read_table("player_stats"))
    g = games.dropna(subset=["GameID", "HomeScore", "AwayScore"]).copy()
    g = g[g["HomeScore"] != g["AwayScore"]]

    model_path = args.model
    if args.retrain or not model_path.is_file():
        if "Year" in g.columns:
            g = g.sort_values("Year")
        train_ids = g["GameID"].astype(str).tail(args.max_train_games).tolist()
        if args.game_id not in train_ids:
            train_ids.append(args.game_id)
        sub = throws[throws["GameID"].astype(str).isin(train_ids)].copy()
        feats_all = build_feature_frame(sub, player_priors=priors, games=g)
        labeled = feats_all.dropna(subset=["home_won"])
        train_smooth_win_model(labeled, output_path=model_path, gam_lam=args.lam)
        print(f"Trained {model_path}")

    model = SmoothWinModel.load(model_path)
    game_throws = throws[throws["GameID"].astype(str) == args.game_id].copy()
    if game_throws.empty:
        raise SystemExit(f"No throws for {args.game_id}")
    feats = build_feature_frame(game_throws, player_priors=priors, games=g, team=args.team)
    scored = score_game_live(
        feats,
        model,
        team=args.team,
        smoother=LiveWinSmoother(half_life_seconds=25.0),
    )
    scored = attach_player_display_names(scored, players)

    safe = args.game_id.replace("-", "_")
    svg = ROOT / "figures" / f"{safe}_{args.team}_live.svg"
    html = ROOT / "figures" / f"{safe}_{args.team}_live.html"
    write_win_prob_svg(
        scored,
        svg,
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
    print(f"Wrote {svg}")
    print(f"Wrote {html}")
    print(f"points={len(scored)} final={scored['win_prob'].iloc[-1]:.3f}")
    if args.open:
        webbrowser.open(html.resolve().as_uri())


if __name__ == "__main__":
    main()
