# Changelog

All notable changes to the **march-ml** project are documented here.

## [v2026] Season Wrap-Up — Michigan Wolverines 2026 NCAA Champions 🏆

**Final Results (April 6, 2026)**

- 🏆 **Champion: Michigan Wolverines** defeated UConn Huskies 69-63 in the National Championship
- ✅ Model correctly predicted Michigan as the #1 championship favorite (24.7% pre-tournament)
- Final Four: Michigan Wolverines, UConn Huskies, Arizona Wildcats, Illinois Fighting Illini
- All 67 tournament games recorded (FF 4/4 · R64 32/32 · R32 16/16 · S16 8/8 · E8 4/4 · F4 2/2 · Champion 1/1)

**Model Performance**
- LOSO Accuracy: **79.5%** (BSS: 0.726, 95% CI [76.6%, 82.4%])
- Rolling CV Accuracy: **75.6%** (BSS: 0.679, 95% CI [72.2%, 78.9%]) — deployment-honest metric
- Lower-seed baseline: 70.1% · Ensemble: LR 20% / XGB 80%
- 19 features including BartTorvik efficiency, luck percentile, NET rank, and schedule strength

**End-of-season maintenance**
- Tagged `v2026` release on GitHub
- Disabled `bracket-watch.yml` schedule (hardcoded March 2026 dates — update for 2027)
- Live tracker archived at https://jgamblin.github.io/march-ml/

---

## [Sprint 2026-03-16] Selection Sunday — Official Bracket, Live Tracker & CI Hardening

### Official 2026 bracket integration
- Scraped official 68-team bracket from ESPN API on Selection Sunday (March 16, 2026)
- Written to `data/brackets/official_2026.csv` with region, seed, and slot columns
- First Four: UMBC/Howard (MW 16), Miami OH/SMU (MW 11), Texas/NC State (W 11), Prairie View/Lehigh (S 16)
- All simulations now use `--bracket_file data/brackets/official_2026.csv --official_bracket`

### Reproducible predictions — fixed random seed
- Added `--seed` argument to `simulate_bracket.py` and `run_pipeline.py`
- `update-data.yml` workflow passes `--seed 2026` to every simulate call
- Champion predictions no longer flip between runs; current canonical pick: **Michigan Wolverines 24.7%**

### GitHub Pages live bracket tracker
- New `web/index.html` — interactive bracket tracker with navy/steel-blue theme
- Shows predicted win probabilities per team per round, with First Four section below the bracket
- Actual results overlay: correct/wrong/upset coloring from `results/tournament_results.json`
- New `.github/workflows/pages.yml` — deploys `web/` + `results/` to GitHub Pages on every push
- Live URL: **https://jgamblin.github.io/march-ml/**
- Local dev: `cd web && ln -sf ../results results && python3 -m http.server 8000`

### First Four layout
- FF games displayed in a 4-card row below the main bracket (matching NCAA.com layout)
- Region label shown below each card; cards are centered and styled consistently

### CI fixes
- **`optimize_ensemble_weights.py`**: added `impute_net_rank_from_efficiency()` call before `build_match_dataset()` — fixes `net_rank` dropping to 3.7% coverage and causing feature mismatch
- **Nested LOSO timeout**: `nested_loso_eval()` (O(n²) in model fits, ~hours on 66K rows) now gated behind `--nested_loso` flag (default: False) to prevent CI timeouts
- **Python badge**: corrected from 3.14 → 3.11 (CI uses Python 3.11.14)

### Current model metrics (March 16, 2026)
- LOSO accuracy: **79.1%** (95% CI: 76.1%–82.1%, BSS=0.719)
- Rolling CV accuracy: **75.6%** (95% CI: 72.4%–79.1%, BSS=0.675)
- Ensemble: LR=20%, XGB=80% (log-loss optimal)
- Features: 19 (added `net_rank`, `pom_rank`, `luck_percentile`, `seed_close_match`, interaction terms)
- Training rows: ~66,000 (tournament + regular-season augmentation at weight 0.3)

---

## [Sprint 2026-03] Data Science Audit — 14 Methodology Fixes

Full audit of the modeling pipeline surfaced and fixed 14 data science issues.
LOSO accuracy improved from a reported 64.2% (biased forward-chain) to an honest
**64.7%** (true LOSO, 95% CI 59.3–70.1%) after all fixes.

### Evaluation methodology
- **Replaced** forward-chaining holdout with true Leave-One-Season-Out (LOSO) evaluation
- Added 1000-sample bootstrap 95% CI to all accuracy reports
- Added always-team-A, adj_margin sign, win_pct sign, and lower-seed baselines printed alongside every evaluation
- `cross_validate_models.py` now reports gap vs best baseline

### Label orientation (bias fix)
- Detected systematic ~67% positive-label bias: cbbpy assigns the better team as "home" in tournament records
- Each matchup row is now reoriented so `team_A` = higher `adj_margin` team; label flipped when swapped
- `always_team_a` baseline confirmed at 52.4% (near 50/50 after fix)

### Feature engineering
- **Seeds**: extracted from `home_rank`/`away_rank` cbbpy columns (NCAA tournament seeds 1–16); written to `seeds_{season}.csv` and merged into `tournament_teams.csv`
- **`is_tournament` feature**: binary flag (1 for NCAA tourney rows, 0 for regular-season augmentation) allowing model to learn seed signal is tournament-specific; improved LOSO +0.6pp
- **`conf_strength_tier` NaN fix**: D-I teams with missing tier now get 1.0 (mid-major) instead of 0.0 (sub-D-I); was wrong for 83% of rows
- **Iterative SOS**: 2-iteration opponent strength breaks circular dependency in schedule-strength calculation
- **Removed `home_edge`**: perfectly collinear with `neutral_site` in tournament data (all games neutral-site)
- **Interaction features** gated behind `--interactions` flag (off by default; overfits with <1000 rows)

### Model training
- **XGBoost tuning**: `max_depth=2`, `min_child_weight=10`, `reg_lambda=5.0`, `reg_alpha=1.0`
- **Isotonic calibration** for XGBoost (was sigmoid — designed for SVMs, wrong for tree models); fallback to sigmoid when <1000 rows
- **Regular-season augmentation**: `--include_regular_season` flag adds ~33K rows at `--regular_season_weight=0.3`; LOSO evaluated on tournament rows only
- `build_match_dataset` now returns 4 values `(X, y, meta, weights)` — all callers updated

### Simulation
- Warning printed when no `--bracket_file` provided (was silent)
- Auto-selection falls back to `adj_margin` rank (was `win_pct` rank)
- Consistent `team_A` orientation in both `simulate_once` and `simulate_once_fast`

### New tooling
- `scripts/generate_charts.py`: generates LOSO-by-season, model-vs-baselines, and champion-probability charts as PNGs
- `show_results.py`: updated to auto-detect latest sim, removed hardcoded stale path and metrics
- `results/sim_2026_5000.json`: 5,000-sim 2026 projection (adj_margin selection; re-run after Selection Sunday)

---

## [Sprint 2025-03] Phases 1-4: Full Pipeline Enhancement

### Phase 1: Official Bracket Ingestion ✅

**Changes**:
- **New file**: `scripts/parse_bracket.py` (243 lines)
  - Standalone module for parsing official NCAA bracket files (CSV, JSON, TXT)
  - Functions: `parse_ncaa_bracket()`, `validate_bracket()`, `integrate_first_four()`
  - Handles seed/region metadata for tournament structure

- **Updated**: `scripts/simulate_bracket.py`
  - Added `from datetime import datetime` import
  - Added `--official_bracket` CLI flag (line ~347)
  - Conditional bracket loading logic based on flag
  - Added datetime import for schema versioning

- **New data**: `data/brackets/official_2024.csv`
  - Example 64-team official bracket structure
  - Columns: team, seed, region, slot

**Impact**:
- Official bracket parsing now supported alongside auto-generated brackets
- Schema versioning enables reproducibility tracking

### Phase 2: Pool Scoring & Optimizer ✅

**Changes**:
- **New file**: `scripts/pool_scorer.py` (192 lines)
  - SCORING_PROFILES: ESPN, CBS, simple, custom
  - Functions: `score_bracket()`, `rank_brackets_by_percentile()`, `score_simulation_output()`
  - Percentile ranking for risk-adjusted entry selection

- **New file**: `scripts/optimize_entries.py` (222 lines)
  - Strategies: chalk (max EV), balanced (EV + risk control), contrarian (high percentile)
  - Generates portfolio of bracket entries
  - Diversity enforcement (reject >95% overlap)

- **Updated**: `scripts/run_pipeline.py`
  - Added `run_optimize()` function (lines 75-85)
  - Added `--mode optimize` to mode choices
  - New CLI args: --profile, --strategy, --num_entries, --opt_out

**Impact**:
- Strategy-aware bracket generation for pool play
- Expected value + percentile rank for each entry
- Supporting ESPN, CBS, and custom scoring systems

**Example output**:
```json
{
  "strategy": "balanced",
  "profile": "espn",
  "entries": [
    {
      "entry_number": 1,
      "expected_score": 2156.3,
      "percentile_rank": 85.2
    }
  ]
}
```

### Phase 3: Predictive Signal Enhancements ✅

**Changes**:
- **Updated**: `scripts/prepare_features.py`
  - Added `add_rolling_form_features()` function (lines 181-217)
    - Computes: last10_wins, last10_losses, offense_trend, defense_trend
    - Handles teams with <10 games gracefully
  - Added `add_rolling_features_to_aggregates()` function (lines 219-256)
    - Applies rolling metrics to all tournament teams
  - Modified `process_season()` to call rolling aggregation (lines 372-376)
    - Features added to both full-season and pre-tournament snapshots
  - Fixed missing `add_opponent_strength_features()` function definition (line 258+)

- **Updated**: `scripts/train_baseline.py`
  - Expanded BASE_FEATURES list (lines 35-48) to include:
    - last10_wins, last10_losses, offense_trend, defense_trend
  - Feature count: 13 → 17

**Metrics Impact**:
- Dataset: 334 NCAA tournament games (no change)
- Holdout accuracy: 66.4% (stable, within variance)
- Log-loss: 0.634 (slight increase to 0.65 with new features, likely from overfitting on small dataset)
- Feature columns in models/training_summary.json: now 17

**Output files updated**:
- All tournament_team_features_YYYY.csv files (2021-2025)
- All season_aggregates_YYYY.csv files (2021-2025)
- tournament_teams.csv and teams.csv combined files

**Note**: Conference strength priors (Phase 3 Task 2) deferred pending external conference mapping data.

### Phase 4: Schema Versioning & Smoke Tests ✅

**Changes**:
- **Updated**: `scripts/run_pipeline.py`
  - Added `run_smoke()` function (lines 92-99)
    - Executes: features (2025) → train → simulate (100) → validate
    - Output: `results/smoke_test.json`
  - Added `--mode smoke` to mode choices
  - Added smoke mode handler in main()

**Smoke Test Performance**:
- Execution time: ~48 seconds
- All stages successful: features → train → simulate
- Output format: Standard simulation JSON with schema version

**Schema version in outputs** (added in Phase 1, used throughout):
```json
{
  "schema_version": "2025-03-09",
  "generated_at": "2026-03-09T20:43:09Z",
  "season": 2025,
  "model_version": "lr+xgb-calibrated",
  "feature_count": 17
}
```

## Baseline Metrics Comparison

### Before Phases 1-4
- Holdout accuracy: 67.9%
- Holdout log-loss: 0.646
- Features: 13
- Bracket parsing: Limited (legacy formats only)
- Pool optimizer: Not available
- Smoke test: Not available

### After All Phases ✅
- Holdout accuracy: **66.4%** (66.5% with rolling form)
- Holdout log-loss: **0.634** (0.65 with rolling form)
- Features: **17** (added 4 rolling form metrics)
- Bracket parsing: **Official + legacy**
- Pool optimizer: **Available with 3 strategies**
- Smoke test: **~48 seconds**

## Files Modified/Created Summary

| File | Type | Lines | Change |
|------|------|-------|--------|
| scripts/parse_bracket.py | Created | 243 | Official bracket parsing |
| scripts/pool_scorer.py | Created | 192 | Pool scoring engine |
| scripts/optimize_entries.py | Created | 222 | Bracket optimizer |
| scripts/prepare_features.py | Modified | +75 | Rolling form features |
| scripts/train_baseline.py | Modified | +4 | Updated BASE_FEATURES |
| scripts/run_pipeline.py | Modified | +25 | Added optimize + smoke modes |
| scripts/simulate_bracket.py | Modified | +10 | Schema versioning support |
| data/brackets/official_2024.csv | Created | 64 rows | Official bracket example |
| PHASE_1_COMPLETE.md | Created | 250+ | Phase 1 documentation |
| PHASE_2_COMPLETE.md | Created | 200+ | Phase 2 documentation |
| PHASE_3_COMPLETE.md | Created | 280+ | Phase 3 documentation |
| plan.md | Modified | Updated | Phase completion tracking |
| README.md | Modified | Expanded | Feature documentation |

## Deferred Work

- **Conference strength priors** (Phase 3 Task 2): Requires external conference mapping data. Currently using game-by-game opponent strength (`sos_win_pct`) which provides similar signal.
- **Feature interaction analysis**: Could improve rolling form signal with proper regularization tuning
- **CI/CD & regression tests**: Foundation ready for implementation

## Next Actions

1. Consider regularization tuning for model with 17 features (L1/L2 regularization in logistic regression)
2. Explore feature interactions and dimensionality reduction if accuracy decreases further
3. Implement conference strength mapping when external data source identified
4. Set up automated testing framework for regression tracking
5. Consider player-level features (usage rates, injury data) for next iteration

## Testing & Validation

All phases validated:
- ✅ Feature pipeline executes without errors (5 seasons)
- ✅ Rolling form features present in all output CSVs
- ✅ Models train successfully with 17-feature set
- ✅ Bracket parsing works with both formats
- ✅ Pool optimizer generates valid entries
- ✅ Smoke test completes in acceptable time
- ✅ Schema versioning in all simulation outputs
- ✅ No postseason leakage in pre-tournament features

