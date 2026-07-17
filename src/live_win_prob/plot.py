"""Win-probability charts: static SVG + interactive HTML scrubber."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from live_win_prob.features import REGULATION_SECONDS


def _period_boundaries(t_max: float) -> list[tuple[float, str]]:
    """Elapsed-second markers for Q1–Q4 and OT when the game reaches them."""
    marks = [(0.0, "Q1"), (720.0, "Q2"), (1440.0, "Q3"), (2160.0, "Q4")]
    if t_max > REGULATION_SECONDS + 5:
        marks.append((float(REGULATION_SECONDS), "OT"))
    return [(t, lab) for t, lab in marks if t <= t_max + 1]


def _format_clock(quarter: float | int | None, time_left: float | None) -> str:
    if quarter is None or (isinstance(quarter, float) and np.isnan(quarter)):
        return ""
    q = int(quarter)
    tl = 0.0 if time_left is None or (isinstance(time_left, float) and np.isnan(time_left)) else float(time_left)
    tl = max(0.0, tl)
    mins = int(tl // 60)
    secs = int(round(tl % 60))
    if secs == 60:
        mins += 1
        secs = 0
    label = "OT" if q >= 5 else f"Q{q}"
    return f"{label} {mins}:{secs:02d}"


def _play_description(row: pd.Series) -> str:
    thrower = str(row.get("Thrower") or "Unknown").strip()
    receiver = str(row.get("Receiver") or "Unknown").strip()
    # Prefer resolved display names when present
    if row.get("ThrowerName"):
        thrower = str(row["ThrowerName"]).strip() or thrower
    if row.get("ReceiverName"):
        receiver = str(row["ReceiverName"]).strip() or receiver
    dist = pd.to_numeric(row.get("throw_distance"), errors="coerce")
    if pd.isna(dist):
        ty = pd.to_numeric(row.get("ThrowerY"), errors="coerce")
        ry = pd.to_numeric(row.get("ReceiverY"), errors="coerce")
        if pd.notna(ty) and pd.notna(ry):
            dist = abs(float(ry) - float(ty))
    dist_s = f"{int(round(float(dist)))} yard" if pd.notna(dist) else "pass"

    turnover = float(pd.to_numeric(row.get("turnover"), errors="coerce") or 0)
    stall = float(pd.to_numeric(row.get("stall"), errors="coerce") or 0)
    dropped = float(pd.to_numeric(row.get("DroppedThrow"), errors="coerce") or 0)
    recv_y = float(pd.to_numeric(row.get("ReceiverY"), errors="coerce") or 0)
    is_goal = recv_y >= 100 and turnover == 0

    if stall:
        return f"{thrower} stalls"
    if dropped:
        return f"{receiver} drops a throw from {thrower}"
    if turnover:
        return f"{thrower} throwaway (turnover)"
    if is_goal:
        return f"{thrower} scores — goal to {receiver}"
    return f"{thrower} completes a {dist_s} pass to {receiver}"


def attach_player_display_names(frame: pd.DataFrame, players: pd.DataFrame | None) -> pd.DataFrame:
    """Map Thrower/Receiver IDs or usernames to full_name when possible."""
    out = frame.copy()
    if players is None or players.empty:
        out["ThrowerName"] = out.get("Thrower")
        out["ReceiverName"] = out.get("Receiver")
        return out

    p = players.copy()
    if "full_name" not in p.columns and {"FirstName", "LastName"}.issubset(p.columns):
        p["full_name"] = p["FirstName"].astype(str).str.strip() + " " + p["LastName"].astype(str).str.strip()
    if "full_name" not in p.columns:
        out["ThrowerName"] = out.get("Thrower")
        out["ReceiverName"] = out.get("Receiver")
        return out

    # Key by PlayerID and by lowercase username-ish tokens from full name
    by_id = {}
    if "PlayerID" in p.columns:
        by_id = (
            p.dropna(subset=["PlayerID"])
            .drop_duplicates("PlayerID")
            .set_index("PlayerID")["full_name"]
            .astype(str)
            .to_dict()
        )
    by_name = (
        p.dropna(subset=["full_name"])
        .drop_duplicates("full_name")
        .assign(_k=lambda d: d["full_name"].astype(str).str.strip().str.lower())
        .set_index("_k")["full_name"]
        .astype(str)
        .to_dict()
    )

    def _resolve(val: object) -> str:
        s = "" if val is None or (isinstance(val, float) and np.isnan(val)) else str(val).strip()
        if not s:
            return "Unknown"
        if s in by_id:
            return by_id[s]
        low = s.lower()
        if low in by_name:
            return by_name[low]
        # username heuristic: first initial + last name compressed, e.g. jmiller
        for full, pretty in by_name.items():
            parts = pretty.lower().split()
            if len(parts) >= 2:
                guess = parts[0][0] + parts[-1]
                if guess == low:
                    return pretty
        return s

    out["ThrowerName"] = out["Thrower"].map(_resolve) if "Thrower" in out.columns else "Unknown"
    out["ReceiverName"] = out["Receiver"].map(_resolve) if "Receiver" in out.columns else "Unknown"
    return out


def _team_names_from_game_id(game_id: str) -> tuple[str, str]:
    """UFA GameID like 2023-05-13-SLC-OAK → (away, home)."""
    parts = str(game_id).split("-")
    if len(parts) >= 2:
        return parts[-2], parts[-1]
    return "Away", "Home"


def _ordered_frame(play_df: pd.DataFrame, *, wp_col: str, time_col: str) -> pd.DataFrame:
    frame = play_df.copy()
    frame["_t"] = pd.to_numeric(frame[time_col], errors="coerce")
    frame["_wp"] = pd.to_numeric(frame[wp_col], errors="coerce")
    return frame.dropna(subset=["_t", "_wp"]).sort_values("_t").reset_index(drop=True)


def write_win_prob_svg(
    play_df: pd.DataFrame,
    output_path: str | Path,
    *,
    title: str = "",
    team_label: str = "Home",
    wp_col: str = "win_prob",
    time_col: str = "elapsed_seconds",
    team: str = "home",
) -> Path:
    """Static SVG with 50% line + vertical period markers (Q1–OT)."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    frame = _ordered_frame(play_df, wp_col=wp_col, time_col=time_col)
    t_arr = frame["_t"].to_numpy(dtype=float)
    wp_arr = frame["_wp"].to_numpy(dtype=float)

    width, height = 900, 440
    margin = {"l": 72, "r": 48, "t": 72, "b": 64}
    plot_w = width - margin["l"] - margin["r"]
    plot_h = height - margin["t"] - margin["b"]
    t_max = max(float(t_arr.max()) if len(t_arr) else 1.0, 1.0)

    def x_of(t: float) -> float:
        return margin["l"] + (t / t_max) * plot_w

    def y_of(p: float) -> float:
        return margin["t"] + (1.0 - p) * plot_h

    points = " ".join(f"{x_of(t):.1f},{y_of(p):.1f}" for t, p in zip(t_arr, wp_arr))
    mid_y = y_of(0.5)
    area = [f"{x_of(t_arr[0]):.1f},{mid_y:.1f}"] if len(t_arr) else []
    area += [f"{x_of(t):.1f},{y_of(p):.1f}" for t, p in zip(t_arr, wp_arr)]
    if len(t_arr):
        area += [f"{x_of(t_arr[-1]):.1f},{mid_y:.1f}"]
    area_pts = " ".join(area)

    steps = np.abs(np.diff(wp_arr)) if len(wp_arr) > 1 else np.array([])
    mean_step = float(steps.mean()) if len(steps) else 0.0
    jumps5 = int((steps > 0.05).sum()) if len(steps) else 0

    gid = str(frame["GameID"].iloc[0]) if "GameID" in frame.columns and len(frame) else ""
    away_code, home_code = _team_names_from_game_id(gid) if gid else ("Away", "Home")
    top_label = home_code if team == "home" else away_code
    bot_label = away_code if team == "home" else home_code
    title = title or f"Live win probability — {gid}"
    final_pct = f"{100 * wp_arr[-1]:.0f}%" if len(wp_arr) else "—"

    period_lines = []
    for t, lab in _period_boundaries(t_max):
        x = x_of(t)
        period_lines.append(
            f'  <line x1="{x:.1f}" y1="{margin["t"]}" x2="{x:.1f}" y2="{margin["t"] + plot_h}" '
            f'stroke="#2a3342" stroke-dasharray="4 4"/>'
        )
        period_lines.append(
            f'  <text x="{x + 4:.1f}" y="{margin["t"] + plot_h + 18}" fill="#9aa7b8" font-size="11">{lab}</text>'
        )

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <defs>
    <linearGradient id="bg" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0%" stop-color="#10141c"/>
      <stop offset="100%" stop-color="#1a2230"/>
    </linearGradient>
    <linearGradient id="fillAbove" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0%" stop-color="#5ec8ff" stop-opacity="0.22"/>
      <stop offset="100%" stop-color="#5ec8ff" stop-opacity="0.02"/>
    </linearGradient>
  </defs>
  <rect width="100%" height="100%" fill="url(#bg)"/>
  <text x="{margin['l']}" y="28" fill="#eef2f7" font-size="16" font-weight="700" font-family="Georgia, serif">{title}</text>
  <text x="{margin['l']}" y="48" fill="#9aa7b8" font-size="11" font-family="Segoe UI, sans-serif">{team_label} win % · final {final_pct} · mean |Δ|={mean_step:.3f} · jumps&gt;5pp={jumps5}</text>
  <line x1="{margin['l']}" y1="{margin['t'] + plot_h}" x2="{margin['l'] + plot_w}" y2="{margin['t'] + plot_h}" stroke="#3a4556"/>
  <line x1="{margin['l']}" y1="{margin['t']}" x2="{margin['l']}" y2="{margin['t'] + plot_h}" stroke="#3a4556"/>
  <line x1="{margin['l']}" y1="{mid_y:.1f}" x2="{margin['l'] + plot_w}" y2="{mid_y:.1f}" stroke="#2a3342" stroke-dasharray="4 4"/>
{chr(10).join(period_lines)}
  <text x="{margin['l'] - 10}" y="{margin['t'] + 4}" fill="#9aa7b8" font-size="10" text-anchor="end">100%</text>
  <text x="{margin['l'] - 10}" y="{mid_y + 4:.1f}" fill="#9aa7b8" font-size="10" text-anchor="end">50%</text>
  <text x="{margin['l'] - 10}" y="{margin['t'] + plot_h}" fill="#9aa7b8" font-size="10" text-anchor="end">0%</text>
  <text x="{margin['l'] + plot_w + 6}" y="{margin['t'] + 12}" fill="#c5d0dc" font-size="11">{top_label}</text>
  <text x="{margin['l'] + plot_w + 6}" y="{margin['t'] + plot_h}" fill="#c5d0dc" font-size="11">{bot_label}</text>
  <polygon fill="url(#fillAbove)" points="{area_pts}"/>
  <polyline fill="none" stroke="#5ec8ff" stroke-width="2.4" stroke-linejoin="round" stroke-linecap="round" points="{points}"/>
  <text x="{margin['l']}" y="{height - 14}" fill="#9aa7b8" font-size="11">0s</text>
  <text x="{margin['l'] + plot_w}" y="{height - 14}" fill="#9aa7b8" font-size="11" text-anchor="end">{t_max:.0f}s</text>
</svg>
"""
    output_path.write_text(svg, encoding="utf-8")
    return output_path


def _series_payload(frame: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for _, r in frame.iterrows():
        q = pd.to_numeric(r.get("game_quarter"), errors="coerce")
        tl = pd.to_numeric(r.get("time_left"), errors="coerce")
        home = pd.to_numeric(r.get("home_team_score"), errors="coerce")
        away = pd.to_numeric(r.get("away_team_score"), errors="coerce")
        turn = float(pd.to_numeric(r.get("turnover"), errors="coerce") or 0)
        recv_y = float(pd.to_numeric(r.get("ReceiverY"), errors="coerce") or 0)
        rows.append(
            {
                "t": round(float(r["_t"]), 3),
                "wp": round(float(r["_wp"]), 4),
                "clock": _format_clock(q, tl),
                "play": _play_description(r),
                "home_score": None if pd.isna(home) else int(home),
                "away_score": None if pd.isna(away) else int(away),
                "thrower": str(r.get("Thrower") or ""),
                "receiver": str(r.get("Receiver") or ""),
                "turnover": int(turn),
                "is_goal": int(recv_y >= 100 and turn == 0),
            }
        )
    return rows


def write_win_prob_html(
    play_df: pd.DataFrame,
    output_path: str | Path,
    *,
    title: str = "",
    team: str = "home",
    wp_col: str = "win_prob",
    time_col: str = "elapsed_seconds",
) -> Path:
    """Interactive Gamecast-style HTML: quarter lines + scrubber with throw + win %."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    frame = _ordered_frame(play_df, wp_col=wp_col, time_col=time_col)
    if frame.empty:
        raise ValueError("No points to plot")

    gid = str(frame["GameID"].iloc[0]) if "GameID" in frame.columns else ""
    away_code, home_code = _team_names_from_game_id(gid) if gid else ("Away", "Home")
    focus = home_code if team == "home" else away_code
    other = away_code if team == "home" else home_code
    title = title or f"{gid} — smooth live win probability"

    payload = {
        "title": title,
        "gameId": gid,
        "team": team,
        "focusTeam": focus,
        "otherTeam": other,
        "periods": [{"t": t, "label": lab} for t, lab in _period_boundaries(float(frame["_t"].max()))],
        "points": _series_payload(frame),
    }
    output_path.write_text(_interactive_html(payload, away_code, home_code), encoding="utf-8")
    return output_path


def _interactive_html(payload: dict[str, Any], away_code: str, home_code: str) -> str:
    data_json = json.dumps(payload, ensure_ascii=True)
    title = payload.get("title", "Win probability")
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>{title}</title>
  <style>
    :root {{
      --ink: #142033; --muted: #5b6b7c; --line: #d7dee7;
      --accent: #1f4b99; --bg: #f7f9fc; --card: #ffffff;
    }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; font-family: "Segoe UI", system-ui, sans-serif; background: var(--bg); color: var(--ink); }}
    .wrap {{ max-width: 980px; margin: 24px auto; padding: 0 16px 40px; }}
    h1 {{ font-family: Georgia, serif; font-size: 1.45rem; margin: 0 0 6px; }}
    .sub {{ color: var(--muted); font-size: 0.92rem; margin-bottom: 16px; }}
    .card {{ background: var(--card); border: 1px solid var(--line); border-radius: 12px;
             padding: 16px 16px 8px; box-shadow: 0 8px 24px rgba(20,32,51,.06); }}
    .chart-wrap {{ position: relative; width: 100%; user-select: none; cursor: crosshair; touch-action: none; }}
    canvas {{ width: 100%; height: auto; display: block; }}
    .hud {{ display: grid; grid-template-columns: 1fr auto; gap: 12px; align-items: start;
            padding: 14px 4px 8px; border-top: 1px solid var(--line); margin-top: 4px; }}
    .score {{ font-size: 1.05rem; font-weight: 700; margin: 0 0 4px; }}
    .play {{ color: var(--muted); font-size: 0.95rem; line-height: 1.35; margin: 0; min-height: 2.6em; }}
    .clock {{ color: var(--muted); font-size: 0.85rem; margin-top: 6px; }}
    .wp-big {{ font-size: 2.4rem; font-weight: 800; letter-spacing: -0.03em; color: var(--accent);
               line-height: 1; text-align: right; }}
    .wp-label {{ text-align: right; color: var(--muted); font-size: 0.8rem; margin-top: 4px; }}
    .hint {{ margin-top: 10px; color: var(--muted); font-size: 0.82rem; }}
    .legend {{ display: flex; justify-content: space-between; color: var(--muted); font-size: 0.8rem; margin: 2px 8px 0; }}
  </style>
</head>
<body>
  <div class="wrap">
    <h1 id="title"></h1>
    <div class="sub">Smooth live model · move or drag across the chart to scrub each throw</div>
    <div class="card">
      <div class="legend">
        <span id="topTeam"></span>
        <span>dashed = 50% and quarter starts</span>
        <span id="botTeam"></span>
      </div>
      <div class="chart-wrap" id="chartWrap">
        <canvas id="chart" width="900" height="420"></canvas>
      </div>
      <div class="hud">
        <div>
          <p class="score" id="score">—</p>
          <p class="play" id="play">Move your mouse across the chart</p>
          <div class="clock" id="clock"></div>
        </div>
        <div>
          <div class="wp-big" id="wpBig">—</div>
          <div class="wp-label" id="wpLabel">win probability</div>
        </div>
      </div>
    </div>
    <p class="hint">Vertical dashed lines mark Q1 / Q2 / Q3 / Q4 / OT. Horizontal dashed line is 50%.</p>
  </div>
<script>
const DATA = {data_json};
const AWAY = {json.dumps(away_code)};
const HOME = {json.dumps(home_code)};
const canvas = document.getElementById('chart');
const ctx = canvas.getContext('2d');
const wrap = document.getElementById('chartWrap');
document.getElementById('title').textContent = DATA.title;
document.getElementById('topTeam').textContent = DATA.focusTeam + ' @ 100%';
document.getElementById('botTeam').textContent = DATA.otherTeam + ' @ 0%';
document.getElementById('wpLabel').textContent = DATA.focusTeam + ' win probability';

const M = {{ l: 56, r: 28, t: 28, b: 42 }};
let scrubIndex = Math.floor(DATA.points.length * 0.55);
let size = {{ w: 900, h: 420, plotW: 816, plotH: 350 }};

function layout() {{
  const cssW = wrap.clientWidth;
  const cssH = Math.max(320, Math.round(cssW * 0.45));
  const dpr = window.devicePixelRatio || 1;
  canvas.width = Math.round(cssW * dpr);
  canvas.height = Math.round(cssH * dpr);
  canvas.style.width = cssW + 'px';
  canvas.style.height = cssH + 'px';
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  size = {{ w: cssW, h: cssH, plotW: cssW - M.l - M.r, plotH: cssH - M.t - M.b }};
}}
function tMax() {{ return Math.max(DATA.points[DATA.points.length - 1].t, 1); }}
function xOf(t) {{ return M.l + (t / tMax()) * size.plotW; }}
function yOf(p) {{ return M.t + (1 - p) * size.plotH; }}

function nearestIndex(clientX) {{
  const rect = canvas.getBoundingClientRect();
  let x = Math.min(M.l + size.plotW, Math.max(M.l, clientX - rect.left));
  const t = ((x - M.l) / size.plotW) * tMax();
  let lo = 0, hi = DATA.points.length - 1;
  while (lo < hi) {{
    const mid = (lo + hi) >> 1;
    if (DATA.points[mid].t < t) lo = mid + 1; else hi = mid;
  }}
  if (lo > 0 && Math.abs(DATA.points[lo - 1].t - t) <= Math.abs(DATA.points[lo].t - t)) return lo - 1;
  return lo;
}}

function updateHud() {{
  const cur = DATA.points[scrubIndex];
  document.getElementById('score').textContent =
    AWAY + ' ' + (cur.away_score ?? '—') + '  –  ' + HOME + ' ' + (cur.home_score ?? '—');
  document.getElementById('play').textContent = cur.play || '—';
  document.getElementById('clock').textContent = cur.clock || '';
  document.getElementById('wpBig').textContent = (cur.wp * 100).toFixed(1) + '%';
}}

function draw() {{
  layout();
  const {{ plotW, plotH }} = size;
  ctx.clearRect(0, 0, size.w, size.h);

  ctx.strokeStyle = '#cfd7e2';
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(M.l, M.t);
  ctx.lineTo(M.l, M.t + plotH);
  ctx.lineTo(M.l + plotW, M.t + plotH);
  ctx.stroke();

  const mid = yOf(0.5);
  ctx.setLineDash([4, 4]);
  ctx.strokeStyle = '#b7c2d0';
  ctx.beginPath();
  ctx.moveTo(M.l, mid);
  ctx.lineTo(M.l + plotW, mid);
  ctx.stroke();
  for (const per of DATA.periods) {{
    const x = xOf(per.t);
    ctx.beginPath();
    ctx.moveTo(x, M.t);
    ctx.lineTo(x, M.t + plotH);
    ctx.stroke();
  }}
  ctx.setLineDash([]);
  ctx.fillStyle = '#6b7c8f';
  ctx.font = '12px Segoe UI, sans-serif';
  for (const per of DATA.periods) ctx.fillText(per.label, xOf(per.t) + 4, M.t + plotH + 18);

  ctx.textAlign = 'right';
  ctx.font = '11px Segoe UI, sans-serif';
  ctx.fillText('100%', M.l - 8, M.t + 4);
  ctx.fillText('50%', M.l - 8, mid + 4);
  ctx.fillText('0%', M.l - 8, M.t + plotH);
  ctx.textAlign = 'left';
  ctx.fillText(DATA.focusTeam, M.l + plotW - 36, M.t + 12);
  ctx.fillText(DATA.otherTeam, M.l + plotW - 36, M.t + plotH);

  const pts = DATA.points;
  ctx.beginPath();
  ctx.moveTo(xOf(pts[0].t), mid);
  for (const p of pts) ctx.lineTo(xOf(p.t), yOf(p.wp));
  ctx.lineTo(xOf(pts[pts.length - 1].t), mid);
  ctx.closePath();
  ctx.fillStyle = 'rgba(31, 75, 153, 0.14)';
  ctx.fill();

  ctx.beginPath();
  pts.forEach((p, i) => {{
    const x = xOf(p.t), y = yOf(p.wp);
    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  }});
  ctx.strokeStyle = '#1f4b99';
  ctx.lineWidth = 2.25;
  ctx.lineJoin = 'round';
  ctx.stroke();

  const cur = pts[scrubIndex];
  const sx = xOf(cur.t), sy = yOf(cur.wp);
  ctx.strokeStyle = '#8a97a8';
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(sx, M.t);
  ctx.lineTo(sx, M.t + plotH);
  ctx.stroke();
  ctx.fillStyle = '#1f4b99';
  ctx.beginPath();
  ctx.arc(sx, sy, 4.5, 0, Math.PI * 2);
  ctx.fill();
  ctx.fillStyle = '#fff';
  ctx.beginPath();
  ctx.arc(sx, sy, 2, 0, Math.PI * 2);
  ctx.fill();

  const tip = (cur.clock ? cur.clock + ' · ' : '') + (cur.wp * 100).toFixed(1) + '%';
  ctx.font = '12px Segoe UI, sans-serif';
  const tw = ctx.measureText(tip).width + 14;
  let tx = Math.max(M.l, Math.min(M.l + plotW - tw, sx - tw / 2));
  const ty = Math.max(M.t + 4, sy - 28);
  ctx.fillStyle = '#142033';
  ctx.fillRect(tx, ty, tw, 20);
  ctx.fillStyle = '#fff';
  ctx.textAlign = 'center';
  ctx.fillText(tip, tx + tw / 2, ty + 14);
  ctx.textAlign = 'left';
}}

function onPointer(e) {{
  scrubIndex = nearestIndex(e.clientX);
  draw();
  updateHud();
}}
wrap.addEventListener('pointerdown', (e) => {{ wrap.setPointerCapture(e.pointerId); onPointer(e); }});
wrap.addEventListener('pointermove', onPointer);
window.addEventListener('resize', () => {{ draw(); updateHud(); }});

draw();
updateHud();
</script>
</body>
</html>
"""
