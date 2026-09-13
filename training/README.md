# Win-probability training (this repo)

Two model tracks live here:

| Folder / package | Role |
|------------------|------|
| `training/deployed/` | Port of ShownSpace **production** `CombinedWinModel` + FV→field_position features |
| `src/live_win_prob/` | **Proposal** smooth GAM + event-gated EMA for live charts |

## Train proposal (smooth) model

```bash
# from repo root, with .env configured (never commit .env)
python scripts/train_and_plot_real.py --game-id 2023-05-13-SLC-OAK
```

## Retrain a production-style CombinedWinModel locally

```bash
python training/train_deployed_style.py
```

Writes `models/deployed_win_model.joblib` (gitignored). Prefer comparing against
the real ShownSpace artifacts via `SHOWNSPACE_MODELS_DIR` when available.

## Build the public side-by-side site

```bash
python scripts/build_comparison_site.py --n-games 8
```

Writes `site/compare.html` + `site/data/games.json` (WP series only — no credentials).
