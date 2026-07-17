# Schema analysis (ShownSpace public tables)

Source: live Postgres information_schema probe (48 base tables).

## Tables selected for live win probability

### 1. `throws` — event grain (required)

~172,000 rows. One row per recorded throw.

**Identity / ordering:** `GameID`, `throw_id`, `event_id`, `game_quarter`, `quarter_point`, `possession_num`, `possession_throw`

**Score / clock:** `home_team_score`, `away_team_score`, `time_left`, `PointStartTime`

**Possession / sides:** `is_home_team`, `start_on_offense`, `offensive_line`, `defensive_line`

**Spatial:** `ThrowerX`, `ThrowerY`, `ReceiverX`, `ReceiverY`, `throw_distance`, `x_diff`, `y_diff`, `throw_angle`

**Outcomes of the throw:** `turnover`, `DroppedThrow`, `stall`

**Players:** `Thrower`, `Receiver` (names — join via `players.full_name`)

**Why chosen:** This is the only dense live stream that updates as the game progresses. Every live WP tick must be computable from a new `throws` row (plus static priors).

**Caveat:** `throws` does **not** store `win_prob`. That lives in `advanced_stats` under a different `throw_id` namespace (cannot naive-join). This project recomputes WP from features instead of trusting that join.

### 2. `games` — labels (required for training)

~2,048 rows. `HomeScore`, `AwayScore`, `Status`, `Year`.

**Why chosen:** Clean game-level winner label without scanning every throw.

### 3. `players` + `player_stats` — skill priors (required for skill tier)

`players`: `PlayerID`, `full_name`, `TeamID`, `Year`  
`player_stats`: `OE`, `DE`, `completion_percentage`, `cpoe`, `ThrowAttempts`, …

**Why chosen:** User requirement — weaker players should pull WP down when they control the disc. Season priors avoid leaking the current game’s outcomes into features if we use **prior-year** stats (or season-to-date excluding current `GameID`).

### 4. `advanced_stats` — evaluation only (optional)

Stores production `win_prob`, FV, CP. Useful to compare smoothness / correlation, not required to train.

### 5. Live / manual future path

`manual_events` already has `game_clock_seconds_remaining`, scores, field x/y, player IDs. Sparse today (dozens of rows) but is the natural schema for **manual live scoring** feeds. The feature builder accepts either `throws`-like or `manual_events`-like frames.

## Tables deliberately ignored for WP

| Table | Reason |
|-------|--------|
| `byu_simulated_*`, `byu_poller_*` | Simulation / poller plumbing |
| `*_Python_dev`, `*_R_dev` | Dev mirrors |
| `profiles`, `link_clicks`, `team_logos` | Product/UI |
| `blocks`, `pulls`, `penalties` | Useful later as event types; start with throws to keep v1 shippable |
| `team_stats*` | Redundant with score + player priors for v1 |

## Feature → column map

| Feature | Source columns | Intuition |
|---------|----------------|-----------|
| `score_diff` | home − away scores | Tied game ≈ 50%; lead grows WP |
| `elapsed_seconds` | quarter + `time_left` | X-axis of the product chart |
| `has_possession_home` | `is_home_team` + turnover/goal flips | Offense should be favored |
| `yards_to_endzone` | `100 - ReceiverY` (home frame) | Far from EZ → lower conversion odds |
| `sideline_pressure` | `abs(ThrowerX) / 27.5` | Sideline squeezes options |
| `offense_skill_edge` | thrower prior OE / completion % | Skilled handlers raise conversion |
| `is_turnover_event` | `turnover`, `stall`, `DroppedThrow` | Gate large WP moves |
| `is_goal_event` | `ReceiverY >= 100` & not turnover | Gate large WP moves + score change |

## Data quality notes

1. **Pre- vs post-goal scores:** DB often stores post-goal score on the goal throw. Training uses a consistent post-play or pre-play convention (documented in code); mixing them creates cliffs.
2. **Name joins:** `Thrower`/`Receiver` are names; fuzzy join to `players.full_name` needs normalization.
3. **OT clocks:** quarters ≥ 5 need special elapsed handling (`time_left` semantics differ).
4. **ID split:** `throws.throw_id` ≠ `advanced_stats.throw_id` ranges — never assume equality.
