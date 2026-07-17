"""SVG chart: elapsed time (0→end) vs win probability %."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def write_win_prob_svg(
    play_df: pd.DataFrame,
    output_path: str | Path,
    *,
    title: str = "",
    team_label: str = "Home",
    wp_col: str = "win_prob",
    time_col: str = "elapsed_seconds",
) -> Path:
    """
    Write a dark-theme SVG.

    X = elapsed seconds from 0 to end of series.
    Y = win probability percent for ``team_label``.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    t = pd.to_numeric(play_df[time_col], errors="coerce")
    wp = pd.to_numeric(play_df[wp_col], errors="coerce")
    mask = t.notna() & wp.notna()
    t_arr = t[mask].to_numpy(dtype=float)
    wp_arr = wp[mask].to_numpy(dtype=float)
    order = np.argsort(t_arr)
    t_arr, wp_arr = t_arr[order], wp_arr[order]

    width, height = 840, 420
    margin = {"l": 64, "r": 28, "t": 64, "b": 56}
    plot_w = width - margin["l"] - margin["r"]
    plot_h = height - margin["t"] - margin["b"]
    t_max = max(float(t_arr.max()) if len(t_arr) else 1.0, 1.0)

    def xy() -> str:
        if len(t_arr) == 0:
            return ""
        xs = margin["l"] + (t_arr / t_max) * plot_w
        ys = margin["t"] + (1.0 - wp_arr) * plot_h
        return " ".join(f"{x:.1f},{y:.1f}" for x, y in zip(xs, ys))

    # volatility footnote
    steps = np.abs(np.diff(wp_arr)) if len(wp_arr) > 1 else np.array([])
    mean_step = float(steps.mean()) if len(steps) else 0.0
    jumps5 = int((steps > 0.05).sum()) if len(steps) else 0

    gid = ""
    if "GameID" in play_df.columns and len(play_df):
        gid = str(play_df["GameID"].iloc[0])
    title = title or f"Live win probability — {gid}"

    final_pct = f"{100 * wp_arr[-1]:.0f}%" if len(wp_arr) else "—"

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <defs>
    <linearGradient id="bg" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0%" stop-color="#10141c"/>
      <stop offset="100%" stop-color="#1a2230"/>
    </linearGradient>
  </defs>
  <rect width="100%" height="100%" fill="url(#bg)"/>
  <text x="{margin['l']}" y="28" fill="#eef2f7" font-size="16" font-weight="700" font-family="Georgia, serif">{title}</text>
  <text x="{margin['l']}" y="48" fill="#9aa7b8" font-size="11" font-family="Segoe UI, sans-serif">{team_label} win % — smooth live path · final {final_pct} · mean |Δ|={mean_step:.3f} · jumps&gt;5pp={jumps5}</text>
  <line x1="{margin['l']}" y1="{margin['t'] + plot_h}" x2="{margin['l'] + plot_w}" y2="{margin['t'] + plot_h}" stroke="#3a4556"/>
  <line x1="{margin['l']}" y1="{margin['t']}" x2="{margin['l']}" y2="{margin['t'] + plot_h}" stroke="#3a4556"/>
  <line x1="{margin['l']}" y1="{margin['t'] + plot_h/2}" x2="{margin['l'] + plot_w}" y2="{margin['t'] + plot_h/2}" stroke="#2a3342" stroke-dasharray="4 4"/>
  <text x="{margin['l'] - 10}" y="{margin['t'] + 4}" fill="#9aa7b8" font-size="10" text-anchor="end">100%</text>
  <text x="{margin['l'] - 10}" y="{margin['t'] + plot_h/2 + 4}" fill="#9aa7b8" font-size="10" text-anchor="end">50%</text>
  <text x="{margin['l'] - 10}" y="{margin['t'] + plot_h}" fill="#9aa7b8" font-size="10" text-anchor="end">0%</text>
  <text x="{margin['l']}" y="{height - 18}" fill="#9aa7b8" font-size="11">0s (start)</text>
  <text x="{margin['l'] + plot_w}" y="{height - 18}" fill="#9aa7b8" font-size="11" text-anchor="end">end ({t_max:.0f}s)</text>
  <text x="{margin['l'] + plot_w/2}" y="{height - 18}" fill="#9aa7b8" font-size="11" text-anchor="middle">Elapsed game time (seconds)</text>
  <polyline fill="none" stroke="#5ec8ff" stroke-width="2.4" stroke-linejoin="round" stroke-linecap="round" points="{xy()}"/>
</svg>
"""
    output_path.write_text(svg, encoding="utf-8")
    return output_path
