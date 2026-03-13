# GitHub Copilot Instructions

## Project Overview

NCAA men's tournament modeling pipeline: scrape game data → engineer features → train matchup models → Monte Carlo bracket simulation → pool entry optimization.

## Environment Setup

```bash
python -m venv .venv
source .venv/bin/activate
# Override pip's global --user flag (required inside venv on this machine):
pip install --no-user -r requirements.txt
```

Virtual environment is at `.venv/`. Always run scripts from the **repo root**, not from within `scripts/`. Prefix commands with `PIP_NO_USER=1` if pip complains about `--user` inside the venv.

## Key Commands

All pipeline stages are orchestrated through `run_pipeline.py` with a `--mode` flag:

```bash
# Fast smoke test — validates core pipeline end-to-end
python scripts/run_pipeline.py --mode smoke

# Full pipeline: scrape → features → train → 5000 sims
python scripts/run_pipeline.py --mode full --sims 5000 --sim_out results/sim_5000.json

# Individual stages
python scripts/run_pipeline.py --mode scrape
python scripts/run_pipeline.py --mode features --seasons 2021 2022 2023 2024 2025 2026
python scripts/run_pipeline.py --mode train --game_scope ncaa_tourney
python scripts/run_pipeline.py --mode train --include_regular_season   # recommended
python scripts/run_pipeline.py --mode simulate --sims 5000 --sim_out results/sim_5000.json
python scripts/run_pipeline.py --mode validate --sim_out results/sim_5000.json

# Pool optimizer (strategies: chalk | balanced | contrarian)
python scripts/run_pipeline.py --mode optimize --sim_out results/sim_5000.json \
  --strategy balanced --num_entries 10
```

### New Training Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--include_regular_season` | off | Augment tournament training rows with regular-season games |
| `--regular_season_weight` | `0.3` | Sample weight applied to regular-season rows (vs 1.0 for tournament) |
| `--interactions` | off | Add quadratic interaction features (overfits with <1000 rows; not recommended) |

## Architecture

### Data Flow

```
scrape_with_cbbpy.py
  → data/raw/games_{season}.pkl
  → data/processed/games_{season}.csv
  → data/processed/boxscores_{season}.csv

prepare_features.py
  → data/processed/features/season_aggregates_{season}.csv   # full-season
  → data/processed/features/tournament_team_features_{season}.csv  # pre-tourney
  → data/processed/features/seeds_{season}.csv               # extracted from home_rank/away_rank
  → data/processed/features/seeds_all.csv                    # combined across seasons
  → data/processed/features/teams.csv          # combined full-season
  → data/processed/features/tournament_teams.csv  # combined pre-tourney (used for training)

train_baseline.py
  → models/lr_model.joblib
  → models/xgb_model.joblib
  → models/lr_cal.joblib / xgb_cal.joblib
  → models/model_features.joblib
  → models/training_summary.json

simulate_bracket.py → results/sim_*.json
optimize_entries.py → results/optimizer_*.json
```

### Model Directories

- `models/` — default production models
- `models_interactions/` — models trained with interaction features
- `models_seed_stratified/` — seed-stratified variant models (prefixed `chalk_`, `balanced_`)

### Script Roles

| Script | Purpose |
|--------|---------|
| `run_pipeline.py` | Orchestrator — chains all stages |
| `scrape_with_cbbpy.py` | Pulls game/boxscore data via `cbbpy` library |
| `prepare_features.py` | Builds leakage-resistant pre-tournament team snapshots |
| `train_baseline.py` | Season-aware model training (LR + XGBoost ensemble) |
| `simulate_bracket.py` | Monte Carlo bracket simulation using calibrated ensemble |
| `optimize_entries.py` | Generates strategy-aware bracket entry portfolios |
| `pool_scorer.py` | Scores brackets against simulation output; ESPN/CBS/custom profiles |
| `parse_bracket.py` | Parses official NCAA bracket CSV/JSON format |
| `validate_artifacts.py` | Validates simulation JSON schema and probability ranges |
| `cross_validate_models.py` | Leave-one-season-out cross-validation |

## Key Conventions

### Feature Engineering (leakage prevention)

`tournament_teams.csv` is the training target — it contains **pre-tournament snapshots** built from non-postseason games only. Never use `teams.csv` (full-season aggregates) for tournament modeling; it leaks postseason results.

Core features in `BASE_FEATURES` (defined in `train_baseline.py`):
- Season stats: `games_played`, `wins`, `losses`, `win_pct`, `avg_points_for/against`, `avg_margin`
- Rolling form: `last10_wins`, `last10_losses`, `offense_trend`, `defense_trend`, `last5_win_pct`
- Schedule strength: `sos_win_pct`, `opp_avg_margin`, `adj_margin` (2-iteration SOS to break circular dependency)
- Tournament seed: `seed` (extracted from `home_rank`/`away_rank` in cbbpy game CSVs for NCAA tourney games)

Optional features (used when present): `conf_avg_adj_margin`, `conf_avg_win_pct`, `conf_strength_tier`, `weighted_last10_margin`, `win_streak`, `margin_trend_slope`, `last5_momentum`

**Do not add `home_edge` to features** — it is perfectly collinear with `neutral_site` in tournament data (all games are neutral-site).

### Seed Extraction

cbbpy stores the NCAA tournament seed in `home_rank`/`away_rank` columns for tournament games. `prepare_features.py` extracts these via `extract_seeds_from_games()` and writes `seeds_{season}.csv`. Seeds range 1–16, with 68 teams per season (4 play-in teams get seeds 11 or 16). Seasons without completed tournaments (e.g., 2026) will have no seed files.

### Model Training

- **Label orientation**: each matchup row is oriented so `team_A` = higher `adj_margin` team. This prevents systematic label bias (cbbpy assigns better teams as "home" in tournament records).
- **Evaluation**: true Leave-One-Season-Out (LOSO) with 1000-sample bootstrap CI. **Also run rolling CV** (strictly forward-looking: train on seasons 1..k-1, test on k) — this is the deployment-relevant metric. Never use forward-chaining holdout without augmented data.
- **Current accuracy (March 2026)**: Rolling CV **74.6%** (95% CI: [69.0%, 79.5%]) · LOSO **78.4%** (95% CI: [74.3%, 82.6%]). LOSO is ~2pp optimistic for 2015–2025 because BartTorvik year-end JSON includes tournament game results; 2026 production predictions are clean (pre-tournament data only).
- **Baselines to beat**: `lower_seed` = 69.8%, `win_pct_sign` = 54.5%, `always_team_a` = 52.4%
- **Calibration**: XGBoost uses isotonic regression (≥3000 rows) or sigmoid/Platt scaling (fallback). LR always uses sigmoid. The 3000-row threshold means: tournament-only training (~670 rows) → sigmoid; `--include_regular_season` augmentation (~33K rows) → isotonic. Do **not** lower this threshold — isotonic regression on ≤1000 rows is prone to overfitting the calibration curve.
- `model_features.joblib` stores the feature list used at training time — simulation must use the same features
- `training_summary.json` records metadata including `loso_per_season`, `baselines`, and `feature_columns`

### `build_match_dataset` Return Signature

```python
X, y, meta, weights = build_match_dataset(games_dir, features_df, game_scope, ...)
```

Returns **4 values** (not 3). `weights` is a numpy array of sample weights (1.0 for tournament rows, `regular_season_weight` for augmented rows). All callers must unpack 4 values.

### Simulation Output Schema

All simulation JSONs include a `schema_version` field (date string) and `generated_at` (ISO timestamp). Required top-level keys: `schema_version`, `generated_at`, `season`, `sims`, `bracket_source`, `bracket`, `champion_probs`, `round_probs`, `model_metadata`.

### Bracket File Formats

Official bracket (use `--official_bracket` flag):
```csv
team,seed,region,slot
UConn,1,South,1
```

Legacy formats: plain text (one team per line), CSV with `team` + optional `slot`/`seed`/`region`, or JSON list. Adjacent slots pair in round 1 (slots 1–2, 3–4, …).

When no bracket file is provided, simulation auto-selects by `adj_margin` rank with a warning — this does **not** reflect real tournament selection criteria.

### NET Rank Imputation

`net_rank` (NCAA NET rankings) is only available for seasons where the scraper successfully fetched it (~2026 coverage). Historical seasons (2015–2025) have `NaN` for this column, which normally causes it to be excluded from the model due to the `_OPTIONAL_COVERAGE_THRESHOLD = 0.5` check.

`impute_net_rank_from_efficiency(features_df)` in `train_baseline.py` fills historical `net_rank` with a within-season `barthag` percentile rank (rank 1 = best team). Correlation between NET and barthag ≈ 0.85, making this a high-quality proxy. After imputation, `net_rank` has ~100% coverage and passes the threshold check, adding it as a training feature.

This function **must** be called in `main()` before `build_match_dataset` — the imputed column must be in the features DataFrame before matchup pairs are built.

### Manual Features Fallback

If the cbbpy API goes down and `scrape_with_cbbpy.py` cannot fetch game data, a fallback path exists:

1. Copy `data/manual_features_template.csv` to `data/processed/features/tournament_teams.csv`.
2. Populate one row per team with the required columns (all `BASE_FEATURES` plus optional efficiency columns).
3. Run `python scripts/run_pipeline.py --mode train --game_scope ncaa_tourney` — the training script will use the provided features directly without needing game logs.

During inference (simulation), `simulate_bracket.py` reads `data/processed/features/tournament_teams.csv` for 2026 team stats. If this file was manually populated, the simulation will run normally but accuracy depends entirely on the quality of the manually entered stats. Log a clear warning if `games_dir` contains no processed game CSVs for the current season so the user knows they're in fallback mode.



Defined in `pool_scorer.py` as `SCORING_PROFILES`. Built-in: `espn`, `cbs`, `simple`. Pass `--profile espn` (default) or `--profile cbs` to scoring/optimizer scripts.

## GitHub Actions

`.github/workflows/scrape-and-save.yml` — runs weekly (Monday 06:00 UTC) or on `workflow_dispatch`. Executes `--mode scrape` and uploads `data/processed/` as an artifact. Python 3.11.
