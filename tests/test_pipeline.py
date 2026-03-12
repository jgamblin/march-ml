"""Regression tests for the march-ml pipeline.

Run with:
    pytest tests/test_pipeline.py -v
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Allow imports from scripts/
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from train_baseline import (
    _lookup_seed_prior,
    build_match_dataset,
    compute_baselines,
    load_features,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

FEATURES_PATH = Path("data/processed/features/tournament_teams.csv")
GAMES_DIR = Path("data/processed")


@pytest.fixture(scope="session")
def features_df():
    if not FEATURES_PATH.exists():
        pytest.skip("tournament_teams.csv not found — run pipeline features step first")
    return load_features(FEATURES_PATH)


@pytest.fixture(scope="session")
def dataset(features_df):
    X, y, meta, weights = build_match_dataset(
        str(GAMES_DIR), features_df, "ncaa_tourney"
    )
    return X, y, meta, weights


# ---------------------------------------------------------------------------
# 1. adj_margin seed ordering
# ---------------------------------------------------------------------------

def test_adj_margin_seed_ordering(features_df):
    """Seed 1 teams should have higher adj_margin than Seed 12 and Seed 16."""
    tourney = features_df[features_df["seed"].notna() & (features_df["seed"] > 0)]
    if tourney.empty:
        pytest.skip("No seeded teams in features")

    seed1 = tourney[tourney["seed"] == 1]["adj_margin"].mean()
    seed12 = tourney[tourney["seed"] == 12]["adj_margin"].mean()
    seed16 = tourney[tourney["seed"] == 16]["adj_margin"].mean()

    assert seed1 > seed12, (
        f"Seed 1 adj_margin ({seed1:.2f}) should exceed Seed 12 ({seed12:.2f})"
    )
    assert seed12 > seed16, (
        f"Seed 12 adj_margin ({seed12:.2f}) should exceed Seed 16 ({seed16:.2f})"
    )
    assert seed16 >= -20, f"Seed 16 adj_margin ({seed16:.2f}) looks implausibly low"


# ---------------------------------------------------------------------------
# 2. build_match_dataset shape and no NaN in feature columns
# ---------------------------------------------------------------------------

def test_build_match_dataset_shape(dataset):
    """Dataset should have expected columns with no NaN in feature matrix."""
    X, y, meta, weights = dataset

    assert X.shape[0] > 0, "Dataset is empty"
    assert X.shape[1] >= 6, f"Expected at least 6 features, got {X.shape[1]}"
    assert len(y) == X.shape[0], "y length mismatch"
    assert set(y).issubset({0, 1}), "Labels must be binary"

    nan_counts = X.isna().sum()
    bad_cols = nan_counts[nan_counts > 0]
    assert bad_cols.empty, f"NaN found in feature columns: {bad_cols.to_dict()}"


def test_build_match_dataset_expected_columns(dataset):
    """Core feature columns must be present."""
    X, *_ = dataset
    required = {"diff_adj_margin", "diff_win_pct", "diff_seed", "seed_matchup_prior",
                "seed_close_match", "adj_when_close", "adj_when_far"}
    missing = required - set(X.columns)
    assert not missing, f"Missing expected columns: {missing}"


# ---------------------------------------------------------------------------
# 3. Historical seed matchup prior
# ---------------------------------------------------------------------------

def test_seed_prior_1_vs_16():
    """1 vs 16 matchup: prior win rate should be ≥ 0.95 (historically ~0.985)."""
    prior = _lookup_seed_prior(1, 16)
    assert prior >= 0.95, f"1 vs 16 prior too low: {prior}"


def test_seed_prior_8_vs_9():
    """8 vs 9 matchup should be near 50/50 (historically ~0.495)."""
    prior = _lookup_seed_prior(8, 9)
    assert 0.40 <= prior <= 0.60, f"8 vs 9 prior not near 50/50: {prior}"


def test_seed_prior_symmetry():
    """_lookup_seed_prior(a, b) + _lookup_seed_prior(b, a) should ≈ 1.0."""
    for a, b in [(1, 16), (5, 12), (8, 9), (3, 14)]:
        p_ab = _lookup_seed_prior(a, b)
        p_ba = _lookup_seed_prior(b, a)
        assert abs(p_ab + p_ba - 1.0) < 1e-9, (
            f"Symmetry failed for ({a},{b}): {p_ab} + {p_ba} ≠ 1.0"
        )


def test_seed_prior_range():
    """All priors should be in (0, 1)."""
    for seed_a in range(1, 17):
        for seed_b in range(seed_a + 1, 17):
            p = _lookup_seed_prior(seed_a, seed_b)
            assert 0.0 < p < 1.0, f"Prior out of (0,1) for ({seed_a},{seed_b}): {p}"


# ---------------------------------------------------------------------------
# 4. Predict probability bounds
# ---------------------------------------------------------------------------

def test_predict_prob_bounds(dataset, features_df):
    """predict_prob (used in simulation) should return values in [0, 1]."""
    try:
        import joblib
        from simulate_bracket import predict_prob, load_models
    except (ImportError, FileNotFoundError):
        pytest.skip("Models not trained or simulate_bracket unavailable")

    models_dir = Path("models")
    if not (models_dir / "lr_model.joblib").exists():
        pytest.skip("Models not found — run pipeline train step first")

    base_lr, base_xgb, lr_cal, xgb_cal, feat_names, feature_scaler = load_models(str(models_dir))
    X, y, meta, weights = dataset

    # Use first two tournament teams from a known season as home/away
    season = meta["season"].max()
    teams_in_season = features_df[features_df["season"] == season].head(2)
    if len(teams_in_season) < 2:
        pytest.skip("Not enough teams for prediction test")

    home_row = teams_in_season.iloc[0]
    away_row = teams_in_season.iloc[1]

    prob = predict_prob(base_lr, base_xgb, lr_cal, xgb_cal, feat_names, home_row, away_row)
    assert isinstance(prob, float), f"predict_prob should return float, got {type(prob)}"
    assert 0.0 <= prob <= 1.0, f"predict_prob out of bounds: {prob}"


# ---------------------------------------------------------------------------
# 5. Baselines — lower seed wins should be ≥ 65%
# ---------------------------------------------------------------------------

def test_lower_seed_baseline(dataset):
    """Lower-seed-wins baseline should be ≥ 65% on tournament games."""
    X, y, meta, weights = dataset
    tourney_mask = weights == 1.0
    X_t = X.loc[tourney_mask].reset_index(drop=True)
    y_t = y[tourney_mask]
    meta_t = meta.loc[tourney_mask].reset_index(drop=True)

    baselines = compute_baselines(X_t, y_t, meta_t)
    seed_baseline = baselines.get("lower_seed", 0.0)
    assert seed_baseline >= 0.65, (
        f"Lower-seed baseline ({seed_baseline:.3f}) below expected 65% — "
        "check adj_margin sign or team orientation in build_match_dataset"
    )


# ---------------------------------------------------------------------------
# 6. Training summary JSON is valid
# ---------------------------------------------------------------------------

def test_training_summary_valid():
    """models/training_summary.json should exist and have required keys."""
    summary_path = Path("models/training_summary.json")
    if not summary_path.exists():
        pytest.skip("training_summary.json not found — run train step first")

    summary = json.loads(summary_path.read_text())
    required_keys = {"loso_overall", "baselines", "feature_columns", "seasons"}
    missing = required_keys - set(summary.keys())
    assert not missing, f"training_summary.json missing keys: {missing}"

    loso = summary["loso_overall"]
    assert "accuracy" in loso, "loso_overall missing 'accuracy'"
    assert 0.0 < loso["accuracy"] < 1.0, f"LOSO accuracy out of range: {loso['accuracy']}"
