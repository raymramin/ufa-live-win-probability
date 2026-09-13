# Changes vs deployed (1 page)

**Repo:** [ufa-live-win-probability](https://github.com/raymramin/ufa-live-win-probability)  
**Live compare:** [Side-by-side chart](https://raymramin.github.io/ufa-live-win-probability/site/compare.html)

## What is deployed today (Game Center)

| Piece | Behavior |
|-------|----------|
| Artifact | ShownSpace `models/win_model.joblib` (`CombinedWinModel`) + `fv_model.joblib` |
| Features | `score_diff`, `time_remaining_in_game`, signed **`field_position` (from FV)**, late-game XGB on distance/possession |
| Update grain | **Every throw** — field/possession flips move WP immediately |
| Chart feel | Informative but **bumpy** (many jumps >5pp on routine completions) |
| Clock axis (product) | Quarter labels; scrubber shows throw + % |

## What this proposal changes

| Piece | Behavior |
|-------|----------|
| Artifact | `models/smooth_win_model.joblib` (gitignored locally; retrain via scripts) |
| Features | Score × elapsed backbone + **weak** yards / sideline / player-skill priors |
| Live layer | **Event-gated EMA** — large steps on goals/turnovers; tiny steps on completions |
| Chart feel | **Streamlined** for broadcast / live UI; ends at 0% or 100% |
| Clock axis | Elapsed **0 → end** with Q1–OT markers; multi-game **Next / Prev** |

## Side-by-side (same games)

Open `site/compare.html`:

- **Left (orange):** deployed Game Center path (production FV → CombinedWinModel)
- **Right (blue):** proposal smooth live path
- **Next game →** cycles through exported games (keyboard ← → works too)
- Scrub either panel; throw text, score, and win % update together by time

## Why this is safer for live use

1. **Volatility** drops sharply (proposal mean \|Δp\| typically ≪ deployed on the same game).
2. **Context retained** — score, clock, field depth, sideline, skill priors still move the line; they just cannot dominate every completion.
3. **No credential leak** — GitHub only hosts static HTML + JSON win series; `.env` and `.joblib` stay local (see `.gitignore` / `.env.example`).

## What did *not* change in production

ShownSpace runtime still loads repo-root `models/` until an explicit promotion. This repo is the **training + evaluation + shareable demo** track.

## Reproduce locally

```bash
cp .env.example .env   # fill DB URL — never commit .env
pip install -r requirements.txt
python scripts/build_comparison_site.py --n-games 8
# open site/compare.html or push to GitHub Pages
```
