# UFA Live Win Probability

A from-scratch, **live-friendly** win-probability model for Ultimate Frisbee (UFA), built against the real ShownSpace Postgres schema.

**Output:** a single graph per game  
- **X axis:** elapsed game time, from `0` seconds → end of game  
- **Y axis:** win probability for a chosen team (0% → 100%)  
- **Shape:** smooth / low-volatility, with meaningful moves on goals, turnovers, and sustained field/skill context — not throw-by-throw sawtooth.

---

## 0. Why this repo exists

Existing production win models in ShownSpace score **every throw** with a strong `field_position` signal. That is statistically informative but **too bumpy for a live product chart**. This project redesigns the problem for live broadcast / Game Center use:

1. Prefer **score + clock + possession** as the backbone.
2. Add **field context** (distance, sideline) and **player skill priors** as *small, regularized* adjustments.
3. Apply an **event-gated smoother** so live updates do not jitter every completion.
4. Force **0% or 100%** when the game ends.

---

## 1. Database schema — what we can use

Probed from the live ShownSpace schema (48 public tables). Only the tables below are used for this model.

### Primary event stream — `throws` (~172k rows)

| Column | Why it matters |
|--------|----------------|
| `GameID`, `throw_id`, `event_id` | Game / event identity and order |
| `home_team_score`, `away_team_score` | Score differential (largest WP driver) |
| `game_quarter`, `time_left` | Clock → elapsed seconds |
| `is_home_team` | Which team has the disc |
| `ThrowerX/Y`, `ReceiverX/Y` | Field position, distance to endzone, sideline |
| `throw_distance`, `throw_angle` | Difficulty context |
| `turnover`, `DroppedThrow`, `stall` | Possession-changing events |
| `Thrower`, `Receiver` | Join to player skill priors |
| `offensive_line`, `defensive_line` | Optional line-level skill (parsed IDs) |

### Game outcomes — `games` (~2k rows)

| Column | Why |
|--------|-----|
| `GameID`, `HomeScore`, `AwayScore`, `Status`, `Year` | Final score → training label `home_won` |

### Player skill priors — `player_stats` + `players`

| Column | Why |
|--------|-----|
| `PlayerID`, `full_name`, `Year`, `TeamID` | Identity join from thrower/receiver names |
| `OE`, `DE`, `completion_percentage`, `cpoe` | Prior skill: weaker throwers → lower offensive edge |

### Optional enrichment — `advanced_stats`

| Column | Why |
|--------|-----|
| `win_prob`, `fv_*`, `cp` | Baseline comparison only (not required to train) |

### Explicitly **not** used as live WP drivers

- `byu_*` simulation tables (dev/replay only)
- `manual_events` (sparse; future live-ingest path)
- UI tables (`profiles`, `link_clicks`, logos)

Full write-up: [`docs/SCHEMA_ANALYSIS.md`](docs/SCHEMA_ANALYSIS.md)  
Design choices: [`docs/DESIGN.md`](docs/DESIGN.md)

---

## 2. Model design (step-by-step)

### Step A — Define the prediction target

For each historical throw row, label = `1` if the **home team won the game**, else `0`.  
Live output is always **P(team wins | state so far)**. For away perspective: `1 - p_home`.

### Step B — Build a stable game clock

UFA regulation ≈ 4 × 720s. Elapsed time:

```text
elapsed = (quarter - 1) * 720 + (720 - time_left)   # quarters 1–4
```

OT continues past 2880s. The chart always starts at **x = 0** and ends at the final elapsed second.

### Step C — Features (intentional hierarchy)

| Tier | Features | Rationale |
|------|----------|-----------|
| 1 (backbone) | `score_diff`, `elapsed_frac`, `has_possession_home` | Dominate win probability in any sport; keep chart sane |
| 2 (field, weak) | `yards_to_endzone_home`, `sideline_pressure` | Sideline / deep field matter, but **small coefficients** so they don't sawtooth |
| 3 (skill, weak) | `offense_skill_edge` from prior `OE` / completion % | Less skilled handlers lower WP slightly while they have the disc |
| Live only | EMA + event gate | Completions barely move WP; goals/turnovers may |

### Step D — Estimator

`LogisticGAM` with **high smoothing (`lam`)** on all terms, plus tensor `te(score_diff, elapsed_frac)` so late-game score gaps matter more than early ones.

### Step E — Live smoother

Raw model output → **event-gated exponential moving average**:

- On **goal / turnover / pull**: allow up to a larger step (e.g. 8–12 pp).
- On ordinary throws: cap step (e.g. 1.5 pp) then EMA with half-life ~25s of elapsed time.
- At game end: snap to **0% or 100%**.

### Step F — Product graph

`scripts/plot_game.py` writes an SVG: elapsed seconds on X, win % on Y, one continuous smooth polyline.

---

## 3. Quick start

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
pip install -r requirements.txt

# Works offline (synthetic demo game + chart)
python scripts/demo_synthetic.py

# With DB credentials in .env (see .env.example)
python scripts/export_training_frame.py
python scripts/train_model.py
python scripts/plot_game.py --game-id 2023-05-13-SLC-OAK --team home
```

Outputs land in `figures/` and `models/`.

### Interactive Gamecast-style chart

```bash
python scripts/plot_interactive.py --game-id 2023-05-13-SLC-OAK --team home --open
```

Opens `figures/..._live.html` with:
- dashed **Q1 / Q2 / Q3 / Q4 / OT** vertical markers
- scrubber (move/drag) showing **clock, play text, score, and win %**
- area fill to 50%, team labels at 100%/0%

---

## 4. Repository layout

```text
docs/                 schema + design analysis
src/live_win_prob/    features, model, smoother, plotting
scripts/              CLI entry points
tests/                unit tests
models/               trained joblib
figures/              SVG charts
```

---

## 5. Relationship to ShownSpace production

| Artifact | Role |
|----------|------|
| ShownSpace `models/win_model.joblib` | Current Game Center (throw-level, bumpier) |
| This repo | **Proposal** live chart model — smooth, skill-aware, elapsed-time axis |

Do not overwrite production artifacts until holdout + visual review pass.
