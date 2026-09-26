#!/usr/bin/env python
"""Build side-by-side deployed vs proposal comparison site (multi-game, Next/Prev)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "training" / "deployed"))

import numpy as np
import pandas as pd

from live_win_prob.db import read_table
from live_win_prob.features import build_feature_frame, build_player_skill_priors, elapsed_seconds_from_clock
from live_win_prob.model import SmoothWinModel, train_smooth_win_model
from live_win_prob.plot import _format_clock, _play_description, _period_boundaries, attach_player_display_names
from live_win_prob.smooth import LiveWinSmoother, score_game_live
from production_features import build_production_win_features, load_deployed_artifacts


def _pick_games(throws: pd.DataFrame, games: pd.DataFrame, n: int, prefer: list[str]) -> list[str]:
    counts = throws["GameID"].astype(str).value_counts()
    eligible = counts[counts >= 250].index.tolist()
    g = games.dropna(subset=["GameID", "HomeScore", "AwayScore"]).copy()
    g["GameID"] = g["GameID"].astype(str)
    labeled = set(g.loc[g["HomeScore"] != g["AwayScore"], "GameID"])
    eligible = [gid for gid in eligible if gid in labeled]
    out: list[str] = []
    for gid in prefer:
        if gid in eligible and gid not in out:
            out.append(gid)
    for gid in eligible:
        if gid not in out:
            out.append(gid)
        if len(out) >= n:
            break
    return out[:n]


def _clip_y(val) -> float | None:
    y = pd.to_numeric(val, errors="coerce")
    if pd.isna(y):
        return None
    return round(float(np.clip(float(y), 0.0, 120.0)), 1)


def _series_points(frame: pd.DataFrame, wp_col: str) -> list[dict]:
    rows = []
    for _, r in frame.iterrows():
        start_y = _clip_y(r.get("ThrowerY"))
        end_y = _clip_y(r.get("ReceiverY"))
        if end_y is None:
            end_y = start_y
        dist = pd.to_numeric(r.get("throw_distance"), errors="coerce")
        if pd.isna(dist) and start_y is not None and end_y is not None:
            dist = abs(end_y - start_y)
        dist = round(float(dist), 1) if pd.notna(dist) else None
        rows.append(
            {
                "t": round(float(r["elapsed_seconds"]), 2),
                "wp": round(float(r[wp_col]), 4),
                "clock": _format_clock(r.get("game_quarter"), r.get("time_left")),
                "play": _play_description(r),
                "home_score": int(pd.to_numeric(r.get("home_team_score"), errors="coerce") or 0),
                "away_score": int(pd.to_numeric(r.get("away_team_score"), errors="coerce") or 0),
                "start_y": start_y,
                "end_y": end_y,
                "throw_dist": dist,
            }
        )
    return rows


def _vol(wp: pd.Series) -> dict:
    s = pd.to_numeric(wp, errors="coerce").dropna()
    if len(s) < 2:
        return {"mean_abs_step": None, "jumps_gt_5pp": 0}
    steps = s.diff().abs().dropna()
    return {
        "mean_abs_step": round(float(steps.mean()), 4),
        "jumps_gt_5pp": int((steps > 0.05).sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-games", type=int, default=8)
    parser.add_argument("--team", choices=["home", "away"], default="home")
    parser.add_argument("--prefer", action="append", default=["2023-05-13-SLC-OAK"])
    parser.add_argument("--skip-train", action="store_true", help="Reuse models/smooth_win_model.joblib if present")
    args = parser.parse_args()

    print("Loading DB tables...")
    throws = read_table("throws")
    games = read_table("games")
    players = read_table("players")
    priors = build_player_skill_priors(read_table("player_stats"))

    game_ids = _pick_games(throws, games, args.n_games, args.prefer or [])
    print(f"Games: {game_ids}")

    # Train / load proposal model on a broad subset
    g = games.dropna(subset=["GameID", "HomeScore", "AwayScore"]).copy()
    g = g[g["HomeScore"] != g["AwayScore"]]
    if "Year" in g.columns:
        g = g.sort_values("Year")
    train_ids = g["GameID"].astype(str).tail(400).tolist()
    for gid in game_ids:
        if gid not in train_ids:
            train_ids.append(gid)
    model_path = ROOT / "models" / "smooth_win_model.joblib"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    if args.skip_train and model_path.is_file():
        proposal = SmoothWinModel.load(model_path)
        print(f"Loaded proposal model from {model_path}")
    else:
        train_throws = throws[throws["GameID"].astype(str).isin(train_ids)].copy()
        feats_all = build_feature_frame(train_throws, player_priors=priors, games=g)
        proposal = train_smooth_win_model(
            feats_all.dropna(subset=["home_won"]),
            output_path=model_path,
            gam_lam=8.0,
        )
        print(f"Proposal model -> {model_path}")

    # Deployed artifacts
    try:
        deployed = load_deployed_artifacts()
        print(f"Deployed models from {deployed['model_dir']}")
    except Exception as exc:
        print(f"WARNING: could not load deployed joblibs ({exc}); using raw proposal as fallback")
        deployed = None

    site_data = {"team": args.team, "games": []}
    for gid in game_ids:
        gt = throws[throws["GameID"].astype(str) == gid].copy()
        if gt.empty:
            continue
        away, home = gid.split("-")[-2], gid.split("-")[-1]
        gg = g[g["GameID"].astype(str) == gid]

        # Proposal path
        feats = build_feature_frame(gt, player_priors=priors, games=g, team=args.team)
        scored_new = score_game_live(
            feats, proposal, team=args.team, smoother=LiveWinSmoother(half_life_seconds=30.0)
        )
        scored_new = attach_player_display_names(scored_new, players)

        # Deployed path
        if deployed is not None:
            prod_feats = build_production_win_features(gt, fv_model=deployed.get("fv_model"))
            keep = prod_feats.dropna(
                subset=["time_remaining_in_game", "score_diff", "has_possession", "field_position"]
            ).copy()
            keep["win_prob_home"] = np.clip(
                np.asarray(deployed["win_model"].predict(keep), dtype=float), 0.0, 1.0
            )
            if args.team == "away":
                keep["win_prob"] = 1.0 - keep["win_prob_home"]
            else:
                keep["win_prob"] = keep["win_prob_home"]
            keep["elapsed_seconds"] = elapsed_seconds_from_clock(
                keep["game_quarter"], keep["time_left"]
            )
            keep = attach_player_display_names(keep, players)
            old_frame = keep.sort_values("elapsed_seconds")
        else:
            old_frame = scored_new.copy()
            old_frame["win_prob"] = scored_new.get("win_prob_raw", scored_new["win_prob"])

        t_max = float(
            max(
                scored_new["elapsed_seconds"].max(),
                old_frame["elapsed_seconds"].max(),
                1.0,
            )
        )
        final_home_won = None
        if not gg.empty:
            final_home_won = int(gg.iloc[0]["HomeScore"] > gg.iloc[0]["AwayScore"])

        entry = {
            "gameId": gid,
            "away": away,
            "home": home,
            "focusRole": "Home" if args.team == "home" else "Away",
            "focusTeam": home if args.team == "home" else away,
            "finalHomeScore": int(gg.iloc[0]["HomeScore"]) if not gg.empty else None,
            "finalAwayScore": int(gg.iloc[0]["AwayScore"]) if not gg.empty else None,
            "homeWon": final_home_won,
            "tMax": round(t_max, 1),
            "periods": [{"t": t, "label": lab} for t, lab in _period_boundaries(t_max)],
            "deployed": {
                "label": "Deployed (Game Center)",
                "points": _series_points(old_frame, "win_prob"),
                "volatility": _vol(old_frame["win_prob"]),
            },
            "proposal": {
                "label": "Proposal (smooth live)",
                "points": _series_points(scored_new, "win_prob"),
                "volatility": _vol(scored_new["win_prob"]),
            },
        }
        site_data["games"].append(entry)
        print(
            f"  {gid}: deployed vol={entry['deployed']['volatility']} "
            f"proposal vol={entry['proposal']['volatility']}"
        )

    out_dir = ROOT / "site" / "data"
    out_dir.mkdir(parents=True, exist_ok=True)
    data_path = out_dir / "games.json"
    data_path.write_text(json.dumps(site_data), encoding="utf-8")
    print(f"Wrote {data_path}")

    compare_path = ROOT / "site" / "compare.html"
    compare_path.write_text(_compare_html(), encoding="utf-8")
    print(f"Wrote {compare_path}")

    # Mirror into figures for Pages convenience
    (ROOT / "figures" / "compare.html").write_text(_compare_html(data_url="../site/data/games.json"), encoding="utf-8")


def _compare_html(data_url: str = "data/games.json") -> str:
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>Deployed vs Proposal — UFA Win Probability</title>
  <style>
    :root {{ --ink:#142033; --muted:#5b6b7c; --line:#d7dee7; --accent:#1f4b99;
             --old:#c45c26; --new:#1f4b99; --bg:#f7f9fc; --card:#fff; --disc:#3d4f63; }}
    * {{ box-sizing: border-box; }}
    body {{ margin:0; font-family:"Segoe UI",system-ui,sans-serif; background:var(--bg); color:var(--ink); }}
    .wrap {{ max-width: 1180px; margin: 20px auto; padding: 0 16px 40px; }}
    h1 {{ font-family: Georgia, serif; font-size: 1.5rem; margin: 0 0 8px; }}
    .nav {{ display:flex; flex-wrap:wrap; gap:10px; align-items:center; margin: 12px 0 18px; }}
    button {{
      background:var(--accent); color:#fff; border:0; border-radius:10px;
      padding:10px 16px; font-weight:700; cursor:pointer;
    }}
    button.secondary {{ background:#e8eef8; color:var(--ink); }}
    .meta {{ color:var(--muted); font-size:.92rem; }}
    .pills {{ display:flex; flex-wrap:wrap; gap:8px; margin: 8px 0 14px; }}
    .pill {{ background:#eef3fa; border:1px solid var(--line); border-radius:999px; padding:4px 12px; font-weight:600; }}
    .pill span {{ color:var(--muted); font-weight:500; margin-right:6px; }}
    .grid {{ display:grid; grid-template-columns:minmax(0,1fr) 168px minmax(0,1fr); gap:14px; align-items:stretch; }}
    @media (max-width: 920px) {{
      .grid {{ grid-template-columns:1fr; }}
      .field-card {{ order: -1; }}
    }}
    .card {{ background:var(--card); border:1px solid var(--line); border-radius:12px; padding:12px;
             box-shadow:0 8px 24px rgba(20,32,51,.05); }}
    .card h2 {{ font-size:1rem; margin:0 0 4px; }}
    .vol {{ color:var(--muted); font-size:.8rem; margin-bottom:8px; }}
    .chart-box {{ width:100%; height:300px; }}
    .field-box {{ width:100%; height:300px; }}
    canvas {{ width:100%; height:300px; display:block; cursor:crosshair; touch-action:none; }}
    .hud {{ display:grid; grid-template-columns:1fr auto; gap:8px; border-top:1px solid var(--line);
            margin-top:8px; padding-top:10px; min-height:72px; }}
    .field-hud {{ border-top:1px solid var(--line); margin-top:8px; padding-top:10px; min-height:72px;
                  font-size:.85rem; color:var(--muted); }}
    .field-hud strong {{ color:var(--ink); }}
    .score {{ font-weight:700; margin:0 0 4px; }}
    .play {{ color:var(--muted); font-size:.9rem; margin:0; min-height:2.4em; }}
    .wp {{ font-size:1.8rem; font-weight:800; text-align:right; line-height:1; }}
    .wp-label {{ color:var(--muted); font-size:.75rem; text-align:right; }}
    .legend {{ display:flex; flex-wrap:wrap; gap:16px; color:var(--muted); font-size:.85rem; margin-top:10px; }}
    .swatch {{ display:inline-block; width:12px; height:12px; border-radius:2px; margin-right:6px; vertical-align:middle; }}
  </style>
</head>
<body>
  <div class="wrap">
    <h1>Deployed vs proposal win probability</h1>
    <div class="meta">Side-by-side on the same game timeline. Move across a chart to scrub throws.</div>
    <div class="nav">
      <button class="secondary" id="prevBtn" type="button">Prev game</button>
      <button id="nextBtn" type="button">Next game</button>
      <span class="meta" id="gameCounter"></span>
    </div>
    <div class="pills">
      <div class="pill"><span>Away</span><span id="awayCode">—</span></div>
      <div class="pill"><span>Home</span><span id="homeCode">—</span></div>
      <div class="pill" id="focusPill">Graph = —</div>
      <div class="pill" id="finalPill">Final —</div>
    </div>
    <div class="grid">
      <div class="card">
        <h2 style="color:var(--old)">Deployed (Game Center path)</h2>
        <div class="vol" id="oldVol"></div>
        <div class="chart-box"><canvas id="oldCanvas" width="520" height="300"></canvas></div>
        <div class="hud">
          <div>
            <p class="score" id="oldScore">—</p>
            <p class="play" id="oldPlay">—</p>
          </div>
          <div>
            <div class="wp" id="oldWp" style="color:var(--old)">—</div>
            <div class="wp-label" id="oldWpLabel">win %</div>
          </div>
        </div>
      </div>
      <div class="card field-card">
        <h2>Disc on field</h2>
        <div class="vol">Number line · start → catch</div>
        <div class="field-box"><canvas id="fieldCanvas" width="168" height="300"></canvas></div>
        <div class="field-hud" id="fieldHud">
          Move across either win chart to scrub.
        </div>
      </div>
      <div class="card">
        <h2 style="color:var(--new)">Proposal (smooth live)</h2>
        <div class="vol" id="newVol"></div>
        <div class="chart-box"><canvas id="newCanvas" width="520" height="300"></canvas></div>
        <div class="hud">
          <div>
            <p class="score" id="newScore">—</p>
            <p class="play" id="newPlay">—</p>
          </div>
          <div>
            <div class="wp" id="newWp" style="color:var(--new)">—</div>
            <div class="wp-label" id="newWpLabel">win %</div>
          </div>
        </div>
      </div>
    </div>
    <div class="legend">
      <span><i class="swatch" style="background:var(--old)"></i>Deployed win %</span>
      <span><i class="swatch" style="background:var(--new)"></i>Proposal win %</span>
      <span><i class="swatch" style="background:var(--disc)"></i>Middle = field number line (thrower → receiver)</span>
    </div>
  </div>
<script>
const DATA_URL = {json.dumps(data_url)};
let GAMES = [];
let idx = 0;
let scrub = {{ old: 0, neu: 0 }};
const layouts = {{}};

async function boot() {{
  const res = await fetch(DATA_URL);
  const payload = await res.json();
  GAMES = payload.games || [];
  if (!GAMES.length) {{
    document.body.insertAdjacentHTML('afterbegin', '<p style="padding:20px">No games in data/games.json</p>');
    return;
  }}
  document.getElementById('prevBtn').onclick = () => {{ idx = (idx - 1 + GAMES.length) % GAMES.length; render(true); }};
  document.getElementById('nextBtn').onclick = () => {{ idx = (idx + 1) % GAMES.length; render(true); }};
  window.addEventListener('keydown', (e) => {{
    if (e.key === 'ArrowRight') document.getElementById('nextBtn').click();
    if (e.key === 'ArrowLeft') document.getElementById('prevBtn').click();
  }});
  window.addEventListener('resize', () => {{ if (GAMES.length) render(true); }});
  render(true);
}}

function render(forceLayout) {{
  const g = GAMES[idx];
  document.getElementById('gameCounter').textContent = (idx + 1) + ' / ' + GAMES.length + ' · ' + g.gameId;
  document.getElementById('awayCode').textContent = g.away;
  document.getElementById('homeCode').textContent = g.home;
  document.getElementById('focusPill').textContent = 'Graph = ' + g.focusRole + ' (' + g.focusTeam + ') win probability';
  document.getElementById('finalPill').textContent =
    'Final ' + g.away + ' ' + (g.finalAwayScore ?? '—') + ' – ' + g.home + ' ' + (g.finalHomeScore ?? '—');
  document.getElementById('oldVol').textContent =
    'mean |step|=' + (g.deployed.volatility.mean_abs_step ?? '—') +
    ' · jumps>5pp=' + g.deployed.volatility.jumps_gt_5pp;
  document.getElementById('newVol').textContent =
    'mean |step|=' + (g.proposal.volatility.mean_abs_step ?? '—') +
    ' · jumps>5pp=' + g.proposal.volatility.jumps_gt_5pp;
  document.getElementById('oldWpLabel').textContent = g.focusTeam + ' win % (deployed)';
  document.getElementById('newWpLabel').textContent = g.focusTeam + ' win % (proposal)';
  scrub.old = Math.min(scrub.old, g.deployed.points.length - 1);
  scrub.neu = Math.min(scrub.neu, g.proposal.points.length - 1);
  if (forceLayout) {{
    scrub.old = Math.floor(g.deployed.points.length * 0.55);
    scrub.neu = Math.floor(g.proposal.points.length * 0.55);
    delete layouts.oldCanvas;
    delete layouts.newCanvas;
    delete layouts.fieldCanvas;
  }}
  drawPanel('oldCanvas', g, g.deployed, '#c45c26', 'old', !!forceLayout);
  drawPanel('newCanvas', g, g.proposal, '#1f4b99', 'neu', !!forceLayout);
  drawField(g, !!forceLayout);
  updateHud(g);
}}

function currentFieldPoint(g) {{
  // Prefer proposal series (same throws); fall back to deployed.
  return g.proposal.points[scrub.neu] || g.deployed.points[scrub.old];
}}

function updateHud(g) {{
  const o = g.deployed.points[scrub.old];
  const n = g.proposal.points[scrub.neu];
  document.getElementById('oldScore').textContent = 'Away ' + g.away + ' ' + o.away_score + ' – Home ' + g.home + ' ' + o.home_score;
  document.getElementById('newScore').textContent = 'Away ' + g.away + ' ' + n.away_score + ' – Home ' + g.home + ' ' + n.home_score;
  document.getElementById('oldPlay').textContent = (o.clock ? o.clock + ' · ' : '') + o.play;
  document.getElementById('newPlay').textContent = (n.clock ? n.clock + ' · ' : '') + n.play;
  document.getElementById('oldWp').textContent = (o.wp * 100).toFixed(1) + '%';
  document.getElementById('newWp').textContent = (n.wp * 100).toFixed(1) + '%';
  const p = currentFieldPoint(g);
  const start = p.start_y == null ? '—' : p.start_y;
  const end = p.end_y == null ? '—' : p.end_y;
  const dist = p.throw_dist == null ? '—' : p.throw_dist;
  document.getElementById('fieldHud').innerHTML =
    '<div><strong>Start</strong> Y ' + start + '</div>' +
    '<div><strong>Catch</strong> Y ' + end + '</div>' +
    '<div><strong>Throw</strong> ' + dist + ' yd</div>';
}}

function nearest(points, t) {{
  let lo = 0, hi = points.length - 1;
  while (lo < hi) {{
    const mid = (lo + hi) >> 1;
    if (points[mid].t < t) lo = mid + 1; else hi = mid;
  }}
  if (lo > 0 && Math.abs(points[lo-1].t - t) <= Math.abs(points[lo].t - t)) return lo - 1;
  return lo;
}}

function ensureLayout(canvasId, force, margins) {{
  if (layouts[canvasId] && !force) return layouts[canvasId];
  const canvas = document.getElementById(canvasId);
  const box = canvas.parentElement;
  const cssW = Math.max(120, Math.floor(box.clientWidth));
  const cssH = 300;
  const dpr = window.devicePixelRatio || 1;
  canvas.width = Math.round(cssW * dpr);
  canvas.height = Math.round(cssH * dpr);
  canvas.style.width = cssW + 'px';
  canvas.style.height = cssH + 'px';
  const ctx = canvas.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  const M = margins || {{ l: 44, r: 14, t: 18, b: 36 }};
  layouts[canvasId] = {{
    canvas, ctx, cssW, cssH, M,
    plotW: cssW - M.l - M.r,
    plotH: cssH - M.t - M.b,
  }};
  return layouts[canvasId];
}}

function scrubToTime(game, t) {{
  scrub.old = nearest(game.deployed.points, t);
  scrub.neu = nearest(game.proposal.points, t);
  drawPanel('oldCanvas', game, game.deployed, '#c45c26', 'old', false);
  drawPanel('newCanvas', game, game.proposal, '#1f4b99', 'neu', false);
  drawField(game, false);
  updateHud(game);
}}

function drawPanel(canvasId, game, series, color, key, forceLayout) {{
  const L = ensureLayout(canvasId, forceLayout, {{ l: 44, r: 14, t: 18, b: 36 }});
  const {{ ctx, cssW, cssH, M, plotW, plotH }} = L;
  const tMax = Math.max(game.tMax, 1);
  const pts = series.points;
  const xOf = t => M.l + (t / tMax) * plotW;
  const yOf = p => M.t + (1 - p) * plotH;

  ctx.clearRect(0, 0, cssW, cssH);

  ctx.strokeStyle = '#cfd7e2';
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(M.l, M.t); ctx.lineTo(M.l, M.t + plotH); ctx.lineTo(M.l + plotW, M.t + plotH);
  ctx.stroke();

  ctx.setLineDash([4,4]);
  ctx.strokeStyle = '#b7c2d0';
  const mid = yOf(0.5);
  ctx.beginPath(); ctx.moveTo(M.l, mid); ctx.lineTo(M.l + plotW, mid); ctx.stroke();
  (game.periods || []).forEach(per => {{
    const x = xOf(per.t);
    ctx.beginPath(); ctx.moveTo(x, M.t); ctx.lineTo(x, M.t + plotH); ctx.stroke();
  }});
  ctx.setLineDash([]);
  ctx.fillStyle = '#6b7c8f';
  ctx.font = '11px Segoe UI, sans-serif';
  (game.periods || []).forEach(per => ctx.fillText(per.label, xOf(per.t) + 3, M.t + plotH + 16));
  ctx.textAlign = 'right';
  ctx.fillText('100%', M.l - 6, M.t + 4);
  ctx.fillText('50%', M.l - 6, mid + 4);
  ctx.fillText('0%', M.l - 6, M.t + plotH);
  ctx.textAlign = 'left';

  ctx.beginPath();
  ctx.moveTo(xOf(pts[0].t), mid);
  pts.forEach(p => ctx.lineTo(xOf(p.t), yOf(p.wp)));
  ctx.lineTo(xOf(pts[pts.length-1].t), mid);
  ctx.closePath();
  ctx.fillStyle = color === '#c45c26' ? 'rgba(196,92,38,0.12)' : 'rgba(31,75,153,0.12)';
  ctx.fill();
  ctx.beginPath();
  pts.forEach((p,i) => {{ const x=xOf(p.t), y=yOf(p.wp); if(i===0) ctx.moveTo(x,y); else ctx.lineTo(x,y); }});
  ctx.strokeStyle = color; ctx.lineWidth = 2.1; ctx.lineJoin = 'round'; ctx.stroke();

  const cur = pts[scrub[key]];
  const sx = xOf(cur.t), sy = yOf(cur.wp);
  ctx.strokeStyle = '#8a97a8'; ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(sx, M.t); ctx.lineTo(sx, M.t + plotH); ctx.stroke();
  ctx.fillStyle = color; ctx.beginPath(); ctx.arc(sx, sy, 4, 0, Math.PI*2); ctx.fill();

  L.canvas.onpointermove = (e) => {{
    const rect = L.canvas.getBoundingClientRect();
    let x = Math.min(M.l + plotW, Math.max(M.l, e.clientX - rect.left));
    const t = ((x - M.l) / plotW) * tMax;
    scrubToTime(game, t);
  }};
}}

function drawField(game, forceLayout) {{
  const L = ensureLayout('fieldCanvas', forceLayout, {{ l: 42, r: 18, t: 22, b: 28 }});
  const {{ ctx, cssW, cssH, M, plotH }} = L;
  const p = currentFieldPoint(game);
  const fieldYOf = y => M.t + plotH - (Math.max(0, Math.min(120, y)) / 120) * plotH;
  const cx = M.l + 28;

  ctx.clearRect(0, 0, cssW, cssH);

  // End-zone bands (neutral slate, not green)
  const y120 = fieldYOf(120), y100 = fieldYOf(100), y0 = fieldYOf(0), y20 = fieldYOf(20);
  ctx.fillStyle = 'rgba(61,79,99,0.10)';
  ctx.fillRect(cx - 18, y120, 36, y100 - y120);
  ctx.fillRect(cx - 18, y20, 36, y0 - y20);

  // Number line
  ctx.strokeStyle = '#3d4f63';
  ctx.lineWidth = 2.2;
  ctx.beginPath();
  ctx.moveTo(cx, M.t);
  ctx.lineTo(cx, M.t + plotH);
  ctx.stroke();

  ctx.fillStyle = '#5b6b7c';
  ctx.font = '10px Segoe UI, sans-serif';
  ctx.textAlign = 'right';
  [[120, '120 EZ'], [100, '100'], [75, '75'], [50, '50'], [25, '25'], [0, '0']].forEach(([yy, lab]) => {{
    const y = fieldYOf(yy);
    ctx.strokeStyle = '#3d4f63';
    ctx.lineWidth = 1.4;
    ctx.beginPath();
    ctx.moveTo(cx - 10, y);
    ctx.lineTo(cx + 10, y);
    ctx.stroke();
    ctx.fillText(lab, cx - 14, y + 3);
  }});
  ctx.textAlign = 'left';
  ctx.fillStyle = '#8a97a8';
  ctx.fillText('goal', cx + 14, y100 + 3);
  ctx.fillText('back', cx + 14, y0);

  const start = p.start_y;
  const end = p.end_y;
  if (start != null && end != null) {{
    const ys = fieldYOf(start);
    const ye = fieldYOf(end);
    // Throw path
    ctx.strokeStyle = '#142033';
    ctx.lineWidth = 2;
    ctx.setLineDash([3, 3]);
    ctx.beginPath();
    ctx.moveTo(cx, ys);
    ctx.lineTo(cx, ye);
    ctx.stroke();
    ctx.setLineDash([]);

    // Start (thrower)
    ctx.fillStyle = '#fff';
    ctx.strokeStyle = '#142033';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.arc(cx, ys, 6, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();

    // End (disc / catch)
    ctx.fillStyle = '#3d4f63';
    ctx.beginPath();
    ctx.arc(cx, ye, 7, 0, Math.PI * 2);
    ctx.fill();
    ctx.fillStyle = '#fff';
    ctx.beginPath();
    ctx.arc(cx, ye, 2.5, 0, Math.PI * 2);
    ctx.fill();

    // Distance callout
    if (p.throw_dist != null) {{
      const midY = (ys + ye) / 2;
      ctx.fillStyle = '#142033';
      ctx.font = 'bold 11px Segoe UI, sans-serif';
      ctx.textAlign = 'left';
      ctx.fillText(p.throw_dist + ' yd', cx + 16, midY + 4);
    }}
  }} else if (end != null) {{
    const ye = fieldYOf(end);
    ctx.fillStyle = '#3d4f63';
    ctx.beginPath();
    ctx.arc(cx, ye, 7, 0, Math.PI * 2);
    ctx.fill();
  }}
}}

boot();
</script>
</body>
</html>
"""


if __name__ == "__main__":
    main()
