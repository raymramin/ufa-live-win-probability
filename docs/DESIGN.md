# Design: smooth live win probability

## Product constraints (from stakeholder)

1. Chart X = **elapsed time from 0 → end**, not “time remaining.”
2. Chart Y = **win probability % for one team**, ending at **0 or 100**.
3. Must respond to real context: sideline, deep field, weaker players.
4. Must be **usable live** — not a volatile throw-level scribble.
5. Must be grounded in the **actual DB schema**.

## Failure mode we are avoiding

Throw-level models that treat signed field value as a first-class feature reverse polarity every possession and move WP 10–20+ points on routine completions. That looks “alive” but is unreadable on TV / Game Center.

## Solution architecture

```text
new throw row
    → feature builder (score, clock, field, skill)
    → SmoothWinGAM.predict_raw()
    → event-gated EMA smoother
    → append (elapsed_seconds, win_pct)
    → SVG / API series
```

### 1. Backbone logit surface

Train:

```text
P(home win) = σ( te(score_diff, elapsed_frac) + s(poss) + s(field) + s(skill) )
```

with **large `lam`** so field/skill curves are nearly linear and small.

**Why GAM:** interpretable partial dependence; matches existing ShownSpace stack (pygam); good for sparse sports state spaces.

**Why tensor score×time:** being up 1 with 30s left ≠ up 1 with 20 minutes left.

### 2. Field features (contextual but weak)

- `yards_to_endzone_home`: distance remaining toward home’s attack endzone in a home-centric frame.
- `sideline_pressure`: `|x| / sideline_half_width` clipped to [0, 1].

Effects are intentionally **bounded** in live mode (`max_field_delta_pp`).

### 3. Skill features

Prior-year (or season-to-date excluding current game) thrower:

```text
skill = 0.6 * z(completion_percentage) + 0.4 * z(OE)
offense_skill_edge = skill * has_possession_home_sign
```

If prior missing → 0 (league average). This satisfies “less skilled players lower WP” without making the chart jump when a star checks in for one throw (EMA dampens that).

### 4. Event-gated EMA (the live product trick)

Let `p_raw` be the model output and `p_live` the displayed series.

```text
max_step = large  if goal or turnover else small
p_capped = clip(p_raw, p_live - max_step, p_live + max_step)
α = 1 - 0.5 ** (Δelapsed / half_life_seconds)
p_live ← (1-α) * p_live + α * p_capped
```

**Why this works in real life:** operators ingest throws every few seconds; viewers should see WP **drift** with field position and **jump** on goals/turns — the same mental model as football WP charts.

### 5. Axis conventions

| Axis | Definition |
|------|------------|
| X = 0 | First throw / pull of the game |
| X = T | Final recorded event elapsed second |
| Y | Team win probability in percent |
| End | If home won: home series → 100, away → 0 (and vice versa) |

### 6. Evaluation checklist (not accuracy alone)

1. **Corr / MAE vs** stored `advanced_stats.win_prob` (optional sanity).
2. **Path volatility:** mean |Δp|, count of jumps > 5 pp and > 10 pp.
3. **Visual review** on close games (e.g. 2023-05-13-SLC-OAK): stays near 50 early, moves on scores, ends at 0/100.
4. **Monotonic sanity:** with possession fixed, deeper field should not *decrease* conversion odds for the offense.

### 7. What we deliberately postponed

- Full line-up OE/DE averages (parse `offensive_line` arrays) — v1.1
- Weather / wind
- Explicit stall-count model
- Replacing production Game Center joblibs

## Promotion rule

This model ships in **this repo** as a proposal. Promote into ShownSpace `models/` only after the checklist above and product sign-off.
