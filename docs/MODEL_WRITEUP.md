# UFA live win probability — full write-up

**Repo:** [ufa-live-win-probability](https://github.com/raymramin/ufa-live-win-probability)  
**Live demo:** [Side-by-side compare](https://raymramin.github.io/ufa-live-win-probability/site/compare.html)  
**Short versions:** [1-page changes vs deployed](CHANGES_VS_DEPLOYED.md) · [Bayesian thought process](BAYESIAN_SMOOTHING.md)

---

## Part 1 — The short version (for anyone)

### What win probability is

Win probability is the live line that answers: *"Right now, how likely is this team to win?"* It starts near 50%, moves as the game happens, and ends at 0% or 100%.

### What was wrong with the current chart

The model running in Game Center today reacts to **every single throw**. A routine 5-yard completion can move the line several points, and a turnover can make it plunge, even when the score and clock say very little has really changed. On a typical game the line jumps more than 5 points **over 100 times**. That makes it look like a seismograph rather than a story of the game.

### What the new model does differently

The new model treats win probability as a **belief that updates with evidence**:

- **The scoreboard and the clock matter most.** A one-goal lead with two minutes left is worth much more than a one-goal lead in the first quarter, because there is less time to come back.
- **Big leads stop stacking forever.** Going from tied to up one is a big deal. Going from up four to up five barely changes anything — you were already very likely to win.
- **Routine throws barely move the line.** A completion is expected; it is weak evidence.
- **Goals move it more, especially late.** A goal is strong evidence, and stronger with less time left.
- **Turnovers hurt, but don't crater the line.** A turnover early in the game, or when you're already ahead, is bad news but not a disaster, so the line dips instead of plunging.

### What it looks like

On the same six games, the new (blue) line is about **10–30× calmer** than the deployed (orange) line, with **0–1** big jumps per game instead of **38–179**. It still tells the same story: the team that ends up winning generally trends up, and goals and turnovers are still visible.

### What has *not* changed

Nothing in production has changed. Game Center still runs the original model. This repo is a **proposal, training workspace, and shareable demo**. Moving the new model into production would be a separate, deliberate step.

---

## Part 2 — Technical write-up

### 2.1 Background: how this started

While comparing local charts to Game Center for **2023-05-13 SLC @ OAK**, the curves didn't match. The cause was **two different `win_model.joblib` files**:

| File | What it was | Behavior |
|------|-------------|----------|
| `shownspace_backend/models/win_model.joblib` | Production (Game Center) | Correlation ≈ 0.97 with Game Center values in the DB |
| `backend/model_training/models/win_model.joblib` | A local retrain | Sawtooth; mean step ≈ 0.062; 101 jumps > 10 pp |

Two data notes from that investigation:

- `throws.throw_id` does **not** join to `advanced_stats.throw_id`. Game Center values were matched to throws by a `game_time_left` fingerprint instead.
- The production feature builder had pandas dtype bugs (boolean `has_possession`, length mismatch on `field_position`). These were fixed in the ported copy here (`training/deployed/production_features.py`) using float casts and `np.where`.

A dedicated repo was then created so the win model could be rebuilt from the database schema up, without touching production.

### 2.2 Data used

From the ShownSpace Postgres database (read-only):

| Table | Used for |
|-------|----------|
| `throws` | Every throw: score, quarter, clock, thrower/receiver coordinates, turnover / drop / stall flags |
| `games` | Final scores → win/loss label per game |
| `players` | Display names for play-by-play text |
| `player_stats` | Thrower skill prior (completion % and OE) |

Credentials live in a local `.env` only (gitignored; template in `.env.example`).

### 2.3 The deployed model (orange, "before")

Ported to `training/deployed/` so it can be run side by side.

**Pipeline:** throw → FV (field value) model → signed `field_position` → `CombinedWinModel`.

**`CombinedWinModel`:**

- If more than 30 seconds remain: a **LogisticGAM** on `score_diff`, `time_remaining_in_game`, `field_position`, with very light regularization (`lam = 0.1`).
- If 30 seconds or fewer remain: a shallow **XGBoost** (75 trees, depth 2) on `score_diff`, `time_remaining_in_game`, `distance_to_endzone`, `has_possession`.

**Why it is jumpy:**

1. `field_position` comes from the FV model and changes sign on every possession flip, so each turnover swings the input from one side to the other.
2. Light regularization lets the GAM follow those swings closely.
3. There is no memory between throws — every throw is scored independently.

### 2.4 The proposal model (blue, "after")

Code: `src/live_win_prob/` (`features.py`, `model.py`, `smooth.py`).

The proposal has two layers: a **calm base model** that produces a raw probability for each throw, and a **live smoother** that decides how much of that raw value to actually show.

#### Layer 1 — Features (`features.py`)

| Feature | Definition | Why |
|---------|------------|-----|
| `score_diff` | `tanh((home − away) / 2)` | Soft score edge. ±1 goal ≈ ±0.46, ±2 ≈ ±0.76, ±4 ≈ ±0.96. Big leads saturate instead of growing linearly. |
| `elapsed_frac` | Elapsed seconds ÷ game length (regulation = 2,880 s), capped at 1.5 | Game progress, 0 → 1 (> 1 in overtime). |
| `has_possession_home` | 1 if home holds the disc after this throw; flips on goals, turnovers, stalls | Possession context. |
| `yards_to_endzone_home` | Home-centric yards to score, from `ReceiverY` | Field position (deliberately weak). |
| `sideline_pressure` | `|ThrowerX| / 27.5`, clipped 0–1 | How close to the sideline the throw was. |
| `offense_skill_edge` | Thrower skill z-score (60% completion %, 40% OE), signed by possession | Player prior (deliberately weak). |

The raw goal difference is kept separately as `score_diff_raw` for the smoother.

Event flags: **goal** = receiver at Y ≥ 100 with no turnover; **turnover event** = turnover, stall, or dropped throw.

#### Layer 2 — Base model (`model.py`)

A **LogisticGAM** (from `pygam`):

```
te(score_diff, elapsed_frac)        8 × 10 splines, lam = 5.6   ← backbone
+ s(has_possession_home)            4 splines,      lam = 14.4
+ s(yards_to_endzone_home)          5 splines,      lam = 20
+ s(sideline_pressure)              4 splines,      lam = 24
+ s(offense_skill_edge)             4 splines,      lam = 24
```

(base `gam_lam = 8.0`)

- The **tensor term** `te(score, time)` is the key change. It lets the model learn that the *same* score edge means more later in the game, which a purely additive model can't express.
- The context terms have **3–4× more regularization** than the backbone, so field position, sideline, and skill can only nudge the curve, not drive it.
- Trained on the most recent ~400 decided games in the `games` table (ties excluded), labeled by whether home won.
- Saved to `models/smooth_win_model.joblib` (gitignored).

#### Layer 3 — Context-aware live smoother (`smooth.py`)

For each throw, in time order:

1. **Cap the step.** The raw prediction is clamped to within `max_step` of the current live value. `max_step` depends on the event and context:

   | Event | Max step | Context scaling |
   |-------|----------|-----------------|
   | Ordinary throw | 1.2 pp | none |
   | Goal | 12 pp × (0.55 + 0.45 × progress) | 6.6 pp at kickoff → 12 pp at the end |
   | Turnover / stall / drop | 4.5 pp × early-damp × lead-damp | Early-damp: up to 35% smaller at kickoff. Lead-damp: up to 45% smaller late with a 3+ goal lead. |

2. **Blend (exponential moving average).** `live = (1 − α) · live + α · capped`, where `α = 1 − 0.5^(Δt / 30 s)`, clamped to [0.05, 1]. A longer gap between throws lets more of the new value through; rapid throws mostly keep the previous value.

3. **End snap.** On the final throw the line is set to 0% or 100% from the actual result.

This is what we mean by "Bayesian" here: score × time is the prior, each throw is evidence, and the step cap encodes how strong each kind of evidence should be. It is **Bayesian-inspired**, not a formal posterior computation.

### 2.5 Results (same six games)

Mean |step| is the average absolute change in win probability between consecutive throws. Jumps are steps larger than 5 percentage points.

| Game | Deployed mean \|step\| | Deployed jumps > 5 pp | Proposal mean \|step\| | Proposal jumps > 5 pp |
|------|------|------|------|------|
| 2023-05-13 SLC @ OAK | 0.0379 | 156 | 0.0026 | 1 |
| 2024-06-01 CHI @ IND | 0.0415 | 179 | 0.0012 | 1 |
| 2024-05-18 ATL @ IND | 0.0111 | 54 | 0.0010 | 0 |
| 2024-05-18 PHI @ DC | 0.0272 | 124 | 0.0024 | 1 |
| 2024-06-08 MIN @ IND | 0.0141 | 59 | 0.0010 | 0 |
| 2024-06-02 IND @ PIT | 0.0081 | 38 | 0.0009 | 1 |

The single proposal "jump" per game is typically the final snap to 0% / 100%.

### 2.6 The demo site

- **`site/compare.html`** — orange deployed chart, blue proposal chart, and a middle **"Disc on field"** panel. Scrubbing either chart syncs both by game time. The middle panel is a vertical field number line (0 → 100 goal line → 120 end zone) showing the throw start (hollow circle, `ThrowerY`), the catch (filled circle, `ReceiverY`), and the throw distance. Prev/Next and ← / → switch games. The canvas is sized once per game or window resize, which fixed the earlier "zoom on hover" bug.
- **`site/data/games.json`** — only the win-probability series, clock, score, play text, and disc coordinates. No credentials or model files.
- **`index.html`** — landing page, with the charts and the written summaries kept separate.

### 2.7 Repo layout

| Path | Contents |
|------|----------|
| `src/live_win_prob/` | Proposal: features, GAM, smoother, plotting, DB helpers |
| `training/deployed/` | Port of production `CombinedWinModel` + FV feature builder |
| `training/train_deployed_style.py` | Retrain a production-style model locally |
| `scripts/build_comparison_site.py` | Train/load proposal, score deployed + proposal, write the site |
| `docs/` | This write-up and the short summaries |
| `.env.example` | DB config template (real `.env` is gitignored) |

### 2.8 How to run

```powershell
copy .env.example .env      # fill in DB credentials; never commit .env
pip install -r requirements.txt

# Retrain the proposal and rebuild the site
python scripts/build_comparison_site.py --n-games 6 --prefer 2023-05-13-SLC-OAK

# Or reuse the saved model
python scripts/build_comparison_site.py --n-games 6 --skip-train --prefer 2023-05-13-SLC-OAK

# Preview locally
cd site; python -m http.server 8765    # then open http://127.0.0.1:8765/compare.html
```

Deployed models are loaded from `SHOWNSPACE_MODELS_DIR` or, by default, `Desktop/shownspace_backend/models`.

---

## Part 3 — Limitations and next steps

These matter before anyone considers promoting the proposal to production:

1. **Smoothness is measured; accuracy is not yet.** The proposal is clearly calmer, but it has not been checked on held-out games for calibration (does "70%" win 70% of the time?) or Brier score / log loss against the deployed model.
2. **The six showcase games are in the training set.** The build script adds them to the ~400 training games, so the demo is not an out-of-sample test.
3. **Two replay-only shortcuts.** `elapsed_frac` uses each game's actual length (only matters for overtime), and the end snap uses the known final result. A live version would use regulation length and snap only when the game ends.
4. **Smoother settings are hand-tuned.** Step caps (1.2 / 4.5 / 12 pp) and the 30-second half-life were chosen by judgment and visual review, not fit to data.
5. **Lag.** Any smoother trades jumpiness for lag: the blue line can trail a real momentum shift by a few throws.

**Suggested next steps:** hold out a season and compare calibration and Brier score; tune step caps and half-life on held-out data; switch the two replay shortcuts to live-safe versions; then review with the team before any production change.
