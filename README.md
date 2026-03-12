# march-ml 🏀

A machine-learning bracket predictor for the NCAA men's tournament. It scrapes historical game data, builds pre-tournament team efficiency profiles (including [BartTorvik T-Rank](https://barttorvik.com) ratings), trains a calibrated LR + XGBoost ensemble, and runs Monte Carlo simulations to estimate every team's championship odds.

[![Python 3.14](https://img.shields.io/badge/python-3.14-blue.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.8-orange.svg)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-3.2-red.svg)](https://xgboost.readthedocs.io/)

Data updates automatically every 6 hours via GitHub Actions. The bracket and championship odds below are live.

---

## 2026 Championship Odds

> ⚠️ **Pre-Selection-Sunday projection** — bracket generated from top-64 teams by efficiency rating. Odds will update automatically once the official bracket is announced on March 15.

![2026 Champion Probabilities](results/charts/champion_probs_2026.png)

| # | Team | Championship% |
|---|------|--------------|
| 1 | Duke Blue Devils | 24.7% |
| 2 | Michigan Wolverines | 23.5% |
| 3 | Arizona Wildcats | 19.2% |
| 4 | Florida Gators | 6.5% |
| 5 | Gonzaga Bulldogs | 2.8% |
| 6 | Michigan State Spartans | 2.7% |
| 7 | North Carolina Tar Heels | 2.0% |
| 8 | UConn Huskies | 1.7% |
| 9 | Nebraska Cornhuskers | 1.6% |
| 10 | Illinois Fighting Illini | 1.5% |
| 11 | Saint Mary's Gaels | 1.4% |
| 12 | Houston Cougars | 1.2% |
| 13 | St. John's Red Storm | 1.2% |
| 14 | Arkansas Razorbacks | 1.0% |
| 15 | Iowa State Cyclones | 1.0% |

Simulated across **5,000 bracket runs**. See [`results/sim_5000.json`](results/sim_5000.json) for full round-by-round probabilities for all 64 teams.

---

## How accurate is it?

The model is evaluated with leave-one-season-out (LOSO) cross-validation — each season is held out while training on all other seasons, simulating real deployment conditions.

| Metric | Value |
|--------|-------|
| **LOSO accuracy (2015–2025)** | **~79%** |
| Seasons of historical data | 2015–2026 |
| Tournament games in training | ~670 |
| Features | 16 |
| Key data sources | cbbpy + BartTorvik T-Rank |

> Note: LOSO numbers for historical seasons may be slightly optimistic (~2pp) because BartTorvik's year-end ratings include post-tournament game results. The 2026 production predictions have no such leakage — all data was collected before the tournament began.

### Feature importance

![SHAP importance](results/charts/shap_importance.png)

---

## How it works

The model computes a head-to-head **matchup diff** for each pair of teams across 16 features, then predicts win probability. Key signals:

| Category | Features |
|----------|---------|
| Efficiency | BartTorvik `adj_em` (net efficiency), `barthag` (Pythagorean win prob) |
| Schedule strength | `adj_margin` (Massey 2-iter), `sos_win_pct` |
| Season stats | `win_pct`, `luck` (deviation from expected wins) |
| Tempo | `adj_t` (possessions/40 min); `tempo_mismatch` |
| Tournament | `seed`, `seed_matchup_prior` (40-yr historical seed win rates) |
| Context | `neutral_site`, `is_tournament` |

Each feature in training is the **difference** between team A and team B. Matchups are oriented so team A always has the higher efficiency rating — this prevents label bias from cbbpy's home/away assignment.

---

## Pipeline

```mermaid
flowchart LR
    A[scrape_with_cbbpy.py] -->|games_YYYY.csv\nboxscores_YYYY.csv| B[prepare_features.py]
    B -->|tournament_teams.csv\nseeds_all.csv| C[train_baseline.py]
    C -->|lr_model.joblib\nxgb_model.joblib\n*_cal.joblib| D[simulate_bracket.py]
    D -->|sim_YYYY_5000.json| E[optimize_entries.py]
    E -->|optimizer_*.json| F[pool_scorer.py]

    style A fill:#1a73e8,color:#fff
    style B fill:#1a73e8,color:#fff
    style C fill:#f5a623,color:#000
    style D fill:#3fb950,color:#000
    style E fill:#3fb950,color:#000
    style F fill:#3fb950,color:#000
```

---

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install --no-user -r requirements.txt   # --no-user required inside venv

# Fast end-to-end smoke test
python scripts/run_pipeline.py --mode smoke

# Full pipeline: scrape → features → train → 5000 sims
python scripts/run_pipeline.py --mode full --sims 5000 --sim_out results/sim_5000.json
```

---

## Commands

### Scrape data
```bash
python scripts/run_pipeline.py --mode scrape
```

### Build features
```bash
python scripts/run_pipeline.py --mode features --seasons 2021 2022 2023 2024 2025 2026
```

### Train models

```bash
# Tournament games only (334 rows — fast, interpretable)
python scripts/run_pipeline.py --mode train

# Recommended: augment with regular-season games at reduced weight
python scripts/run_pipeline.py --mode train --include_regular_season

# Training flags
#   --include_regular_season        add ~33K regular-season rows at weight 0.3
#   --regular_season_weight FLOAT   sample weight for regular-season rows (default: 0.3)
#   --interactions                  add quadratic interaction features (off by default — overfits)
```

### Simulate bracket

```bash
# With official bracket (recommended after Selection Sunday)
python scripts/run_pipeline.py --mode simulate --sims 5000 \
  --bracket_file data/brackets/official_2026.csv --official_bracket \
  --sim_out results/sim_2026_official.json

# Projection using top-64 by adj_margin (pre-Selection Sunday)
python scripts/run_pipeline.py --mode simulate --sims 5000 \
  --sim_out results/sim_2026_5000.json
```

### Pool optimizer
```bash
python scripts/run_pipeline.py --mode optimize --sim_out results/sim_2026_official.json \
  --strategy balanced --num_entries 10

# Strategies: chalk | balanced | contrarian
```

### Generate charts
```bash
python scripts/generate_charts.py --sim results/sim_2026_5000.json
# Outputs: results/charts/loso_by_season.png
#          results/charts/model_vs_baselines.png
#          results/charts/champion_probs_2026.png
```

### View results
```bash
python show_results.py                          # auto-detects latest sim
python show_results.py results/sim_2026_5000.json
```

---

## Repo layout

```
data/
  processed/          scraped game CSVs and box scores (games_YYYY.csv, boxscores_YYYY.csv)
  processed/features/ engineered feature CSVs (tournament_teams.csv, seeds_all.csv, …)
  brackets/           bracket input files (official_2024.csv, template_64.csv)
  mappings/           D-I normalization and conference mapping inputs

models/               trained model artifacts + training_summary.json
results/              simulation outputs (sim_*.json) and charts/
scripts/              all pipeline scripts (see table below)
```

### Script reference

| Script | Purpose |
|--------|---------|
| `run_pipeline.py` | Orchestrator — chains all stages via `--mode` |
| `scrape_with_cbbpy.py` | Pulls game + box score data via `cbbpy` |
| `prepare_features.py` | Builds leakage-free pre-tournament team snapshots |
| `train_baseline.py` | Trains LR + XGBoost ensemble with LOSO evaluation |
| `cross_validate_models.py` | Standalone LOSO + rolling cross-validation |
| `simulate_bracket.py` | Monte Carlo bracket simulation |
| `optimize_entries.py` | Strategy-aware bracket entry portfolio generation |
| `pool_scorer.py` | Scores brackets; ESPN/CBS/simple scoring profiles |
| `parse_bracket.py` | Parses official NCAA bracket CSV/JSON |
| `validate_artifacts.py` | Validates simulation JSON schema |
| `generate_charts.py` | Generates accuracy, SHAP importance, and champion-probability charts |
| `optimize_ensemble_weights.py` | LOSO-based grid search for optimal LR/XGB blend; writes `models/ensemble_weights.json` |
| `train_seed_stratified_models.py` | Per-seed-stratum model variants |
| `fetch_official_bracket.py` | Polls ESPN API for official bracket on Selection Sunday |

---

## Automation (GitHub Actions)

Three workflows keep the repo self-updating:

| Workflow | Trigger | What it does |
|---|---|---|
| **update-data.yml** | Every 6 hours + manual | Scrape → features → train → 5000-sim → commit results |
| **bracket-watch.yml** | Every 15 min on Selection Sunday + manual | Polls ESPN for bracket; triggers 10 000-sim run when found |
| **scrape-and-save.yml** | Manual only (legacy) | Raw data artifact only, no retrain |

### How results get committed back

The `update-data.yml` job has `permissions: contents: write` and pushes three
artefacts back to `main` after each run (only when files actually changed):

```
results/sim_*.json        — latest Monte Carlo simulation
results/charts/           — champion-probability and accuracy charts
models/training_summary.json — LOSO metrics from the most recent train
```

Auto-commit messages include `[skip ci]` to prevent recursive workflow
triggers.

### Raw-data caching

`data/raw/` (cbbpy pickle files) is cached in GitHub Actions with a
weekly-rotating key. This means scraping all five seasons costs one network
round-trip per week; subsequent 6-hour runs skip the API calls entirely and
rebuild features from cache.

### Selection Sunday automation

`bracket-watch.yml` polls ESPN's public tournament API every 15 minutes from
March 14 evening through March 16 morning (UTC). When 68 real team names
appear in the response (not "TBD"), the script:

1. Writes `data/brackets/official_{year}.csv`
2. Commits it to `main`
3. Kicks off `update-data.yml` with `--bracket_file` and 10 000 sims

To test manually:
```bash
python scripts/fetch_official_bracket.py --year 2026 --dry_run
```

To force a re-fetch even if the file already exists:
```
GitHub Actions → bracket-watch → Run workflow → force: true
```

---

## Bracket file formats

### Official bracket (`--official_bracket` flag)

```csv
team,seed,region,slot
Duke Blue Devils,1,East,1
American University,16,East,2
```

### Legacy formats (no `--official_bracket` flag)

- **Plain text** — one team per line, bracket order
- **CSV** — `team` column, optional `slot`, `seed`, `region`
- **JSON** — list of `{team, seed, region, slot}` objects

Adjacent slots (1–2, 3–4, …) are paired in round 1.

---

## Known limitations & roadmap

- **No player-level data** — injuries, roster experience, and height aren't modeled yet. A manual `overrides.csv` mechanism is planned for Selection Sunday.
- **Small tournament sample** — ~670 historical tournament matchup rows across 10 seasons; per-season estimates have high variance.
- **cbbpy API fragility** — if ESPN changes their API schema during tournament week, scraping may fail. See [Troubleshooting](#troubleshooting) for the manual fallback.

---

## Troubleshooting

**Models not found**
```bash
python scripts/run_pipeline.py --mode train --include_regular_season
```

**`pip install` fails inside venv**
```bash
pip install --no-user -r requirements.txt
```
Global `pip.conf` may have `user = true`; `--no-user` overrides it inside a virtualenv.

**Features out of date**
```bash
python scripts/run_pipeline.py --mode features --seasons 2021 2022 2023 2024 2025 2026
```

**Smoke test to validate pipeline**
```bash
python scripts/run_pipeline.py --mode smoke
```

**cbbpy scraping fails during tournament week**

If the ESPN API is down or has changed schema, use the manual features fallback:

```bash
# 1. Copy the template
cp data/manual_features_template.csv data/processed/features/tournament_teams.csv

# 2. Fill in the 64 team rows using KenPom, Bart Torvik, or ESPN BPI for these 8 columns:
#    season, team, seed, adj_margin, win_pct, sos_win_pct, conf_strength_tier, form_rating
#    (see the header comments in the template for field definitions and typical ranges)

# 3. Train and simulate as normal
python scripts/run_pipeline.py --mode train --include_regular_season
python scripts/run_pipeline.py --mode simulate --sims 1000
```

The `seed_matchup_prior` feature is computed automatically from a hardcoded 40-year prior table — no data entry required.

---

