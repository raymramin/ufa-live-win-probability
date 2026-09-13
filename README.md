# UFA Live Win Probability

Public compare (deployed vs proposal, multi-game Next/Prev):

**https://raymramin.github.io/ufa-live-win-probability/site/compare.html**

1-page summary: [`docs/CHANGES_VS_DEPLOYED.md`](docs/CHANGES_VS_DEPLOYED.md)

---

## What’s in this repo

| Path | Purpose |
|------|---------|
| `training/deployed/` | Ported ShownSpace **production** `CombinedWinModel` + FV→field_position features |
| `src/live_win_prob/` | **Proposal** smooth GAM + event-gated EMA |
| `scripts/build_comparison_site.py` | Export multi-game JSON + side-by-side HTML |
| `site/compare.html` | Shareable UI (GitHub Pages) |
| `.env.example` | DB template — **never commit `.env`** |

## Setup (local)

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env   # fill DATABASE_URL — stays gitignored
```

Optional: `SHOWNSPACE_MODELS_DIR` pointing at ShownSpace `models/` (win + fv joblibs).

## Build the compare site

```bash
python scripts/build_comparison_site.py --n-games 8 --prefer 2023-05-13-SLC-OAK
```

Then open `site/compare.html` or push `main` (Pages serves `/`).

## Security

- `.gitignore` blocks `.env`, credentials, `*.joblib`, raw parquet/csv extracts
- Published `site/data/games.json` contains only win-% series + play text (no DB secrets)
