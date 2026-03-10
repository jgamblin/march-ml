# ncaa-bets

NCAA men's basketball tournament prediction pipeline — scrape game data, engineer pre-tournament features, train a calibrated LR + XGBoost ensemble, simulate bracket outcomes with Monte Carlo, and generate strategy-aware pool entries.

[![Python 3.14](https://img.shields.io/badge/python-3.14-blue.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.8-orange.svg)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-3.2-red.svg)](https://xgboost.readthedocs.io/)

---

## Current metrics (seasons 2021–2025)

| Metric | Value |
|--------|-------|
| LOSO accuracy | **64.7%** |
| 95% bootstrap CI | [59.3%, 70.1%] |
| Lower-seed baseline | 70.4% |
| Win-% baseline | 54.5% |
| Tournament games evaluated | 334 |
| Features | 27 |
| Seasons of data | 2021–2026 |

> Accuracy is measured with true **Leave-One-Season-Out** (LOSO) evaluation — each season held out completely while the model trains on remaining seasons. This prevents any data leakage across time.

### LOSO accuracy by season

![LOSO by season](results/charts/loso_by_season.png)

### Model vs baselines

![Model vs baselines](results/charts/model_vs_baselines.png)

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
| `generate_charts.py` | Generates accuracy and champion-probability charts |
| `train_seed_stratified_models.py` | Per-seed-stratum model variants |

---

## Methodology

### Features (27 total, all computed pre-tournament)

Each training example is a matchup `(team_A, team_B)` expressed as the **difference** in each team's features (`diff_*`), plus two context features:

| Category | Features |
|----------|---------|
| Season stats | `diff_games_played`, `diff_wins`, `diff_losses`, `diff_win_pct`, `diff_avg_points_for/against`, `diff_avg_margin` |
| Rolling form | `diff_last10_wins/losses`, `diff_offense_trend`, `diff_defense_trend`, `diff_last5_win_pct` |
| Schedule strength | `diff_sos_win_pct`, `diff_opp_avg_margin`, `diff_adj_margin` |
| Conference | `diff_conf_avg_adj_margin`, `diff_conf_avg_win_pct`, `diff_conf_strength_tier` |
| Momentum | `diff_weighted_last10_margin`, `diff_win_streak`, `diff_margin_trend_slope`, `diff_last5_momentum`, `diff_last10_momentum`, `diff_form_rating` |
| Tournament | `diff_seed`, `neutral_site`, `is_tournament` |

> `tournament_teams.csv` contains **pre-tournament snapshots** built from non-postseason games only. Never use `teams.csv` (full-season) for training — it leaks postseason results.

### Label orientation

cbbpy consistently assigns the better team as "home" in tournament game records (even though all games are played at neutral sites). This creates ~67% positive-label bias. **Fix:** each matchup row is reoriented so `team_A` = higher `adj_margin` team; the label is flipped accordingly.

### Seed extraction

cbbpy stores the tournament seed in `home_rank`/`away_rank` for NCAA tournament games. `prepare_features.py` extracts these automatically — no external seed file required. Seeds (1–16) are merged into `tournament_teams.csv` each time features are rebuilt.

### Model calibration

Both models are wrapped with `CalibratedClassifierCV`:
- Logistic Regression → sigmoid (Platt scaling)
- XGBoost → **isotonic regression** (sigmoid is designed for SVMs; wrong for tree ensembles)

### `build_match_dataset` return signature

```python
X, y, meta, weights = build_match_dataset(games_dir, features_df, game_scope, ...)
```

Returns **4 values**. `weights` is a float array (1.0 = tournament, `regular_season_weight` = regular season). All callers must unpack 4 values.

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

## Known limitations

- **Model vs seed baseline gap**: LOSO 64.7% vs lower-seed baseline 70.4%. Seeds are strong predictors; closing this gap requires richer features (player usage, injury data, coaching changes) or a deeper architecture.
- **Small tournament sample**: only 334 historical tournament matchup rows across 5 complete seasons; high variance in per-season estimates.
- **No automated regression tests**: manual smoke test only.
- **Conference strength partial**: `conf_strength_tier` is a rough 3-level mapping; full inter-conference win% matrix not yet implemented.

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

---

## 2026 season

A pre-Selection Sunday projection (top-64 by adj_margin) has been run:

![2026 Champion Probabilities](results/charts/champion_probs_2026.png)

> ⚠️ This does **not** reflect the actual tournament field. Re-run after Selection Sunday (March 15, 2026) with the official bracket:
> ```bash
> python scripts/simulate_bracket.py --sims 5000 --season 2026 \
>   --bracket_file data/brackets/official_2026.csv --official_bracket \
>   --out results/sim_2026_official.json
> ```

