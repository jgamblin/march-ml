"""Train baseline tournament models with season-aware evaluation.

The default flow trains on pre-tournament team snapshots from
`data/processed/features/tournament_teams.csv` and evaluates against
historical NCAA Tournament games only.

Artifacts written under `models/`:
- `lr_model.joblib`
- `xgb_model.joblib` (when XGBoost is available)
- `lr_cal.joblib`
- `xgb_cal.joblib`
- `model_features.joblib`
- `training_summary.json`
"""
import argparse
import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss

try:
    from xgboost import XGBClassifier

    HAS_XGB = True
except Exception:
    HAS_XGB = False


# Core independent features — kept deliberately small to prevent overfitting.
# adj_margin: Massey-style schedule-adjusted margin (best quality metric).
# win_pct: season-long success rate (independent of margin).
# sos_win_pct: schedule quality (independent of own margin).
BASE_FEATURES = [
    "adj_margin",
    "win_pct",
    "sos_win_pct",
]

# Optional features used when present in the feature file.
# conf_strength_tier: ordinal conference tier (power-6=3, high-major=2, mid-major=1).
# form_rating: composite recent-form score (best single momentum feature).
OPTIONAL_NUMERIC_FEATURES = [
    "conf_strength_tier",
    "form_rating",
]

INTERACTION_FEATURES = [
    "seed_diff_abs",
    "seed_adj_margin_interaction",
    "seed_form_interaction",
    "seed_conf_interaction",
    "seed_sos_interaction",
    "adj_margin_form_interaction",
    "momentum_conf_interaction",
    "momentum_sos_interaction",
    "neutral_form_interaction",
    "neutral_seed_interaction",
]


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def coerce_bool(series):
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    lowered = series.astype(str).str.strip().str.lower()
    return lowered.isin({"true", "1", "yes", "y"})


def load_features(path):
    return pd.read_csv(path)


def coerce_numeric_feature(value):
    if pd.isna(value):
        return 0.0
    if isinstance(value, str):
        lowered = value.strip().lower()
        tier_map = {
            "power-6": 3.0,
            "high-major": 2.0,
            "mid-major": 1.0,
        }
        if lowered in tier_map:
            return tier_map[lowered]
    try:
        return float(value)
    except Exception:
        return 0.0


def filter_games(games_df, game_scope):
    games = games_df.copy()
    games["is_postseason"] = coerce_bool(games.get("is_postseason", False))
    tournament_series = games.get("tournament", pd.Series("", index=games.index)).fillna("").astype(str)
    ncaa_mask = tournament_series.str.contains("Men's Basketball Championship", case=False, na=False)

    if game_scope == "all":
        return games
    if game_scope == "postseason":
        return games[games["is_postseason"]].copy()
    return games[ncaa_mask].copy()


def build_feature_lookup(features_df):
    by_id = {}
    by_name = {}
    for _, row in features_df.iterrows():
        season = int(row["season"])
        team_id = str(row.get("team_id", "")).strip()
        team_name = str(row.get("team", "")).strip().lower()
        if team_id:
            by_id[(season, team_id)] = row
        if team_name:
            by_name[(season, team_name)] = row
    return by_id, by_name


def lookup_feature_row(by_id, by_name, season, team_id, team_name):
    if team_id and (season, team_id) in by_id:
        return by_id[(season, team_id)]
    key = (season, str(team_name).strip().lower())
    return by_name.get(key)


def feature_columns_from_df(features_df):
    cols = [col for col in BASE_FEATURES if col in features_df.columns]
    cols.extend([col for col in OPTIONAL_NUMERIC_FEATURES if col in features_df.columns])
    if "seed" in features_df.columns:
        cols.append("seed")
    return cols


def add_interaction_features(row_features):
    diff_seed = float(row_features.get("diff_seed", 0.0))
    diff_adj_margin = float(row_features.get("diff_adj_margin", 0.0))
    diff_form_rating = float(row_features.get("diff_form_rating", 0.0))
    diff_conf_avg_adj_margin = float(row_features.get("diff_conf_avg_adj_margin", 0.0))
    diff_sos_win_pct = float(row_features.get("diff_sos_win_pct", 0.0))
    diff_last10_momentum = float(row_features.get("diff_last10_momentum", 0.0))
    neutral_site = float(row_features.get("neutral_site", 1.0))

    row_features.update(
        {
            "seed_diff_abs": abs(diff_seed),
            "seed_adj_margin_interaction": diff_seed * diff_adj_margin,
            "seed_form_interaction": diff_seed * diff_form_rating,
            "seed_conf_interaction": diff_seed * diff_conf_avg_adj_margin,
            "seed_sos_interaction": diff_seed * diff_sos_win_pct,
            "adj_margin_form_interaction": diff_adj_margin * diff_form_rating,
            "momentum_conf_interaction": diff_last10_momentum * diff_conf_avg_adj_margin,
            "momentum_sos_interaction": diff_form_rating * diff_sos_win_pct,
            "neutral_form_interaction": neutral_site * diff_form_rating,
            "neutral_seed_interaction": neutral_site * diff_seed,
        }
    )
    return row_features


def build_match_dataset(games_dir, features_df, game_scope, include_interactions=False,
                        include_regular_season=False, regular_season_weight=0.3):
    feature_cols = feature_columns_from_df(features_df)
    by_id, by_name = build_feature_lookup(features_df)

    X_rows = []
    y_rows = []
    meta_rows = []
    weight_rows = []

    def _process_games_file(games_file, row_weight=1.0):
        season = int(games_file.stem.split("_")[1])
        games = pd.read_csv(games_file)
        games = filter_games(games, game_scope)
        if games.empty:
            return

        games["home_win"] = coerce_bool(games.get("home_win", False)).astype(int)
        games["is_neutral"] = coerce_bool(games.get("is_neutral", False)).astype(int)

        for _, game in games.iterrows():
            home_row = lookup_feature_row(
                by_id, by_name, season,
                str(game.get("home_id", "")).strip(),
                str(game.get("home_team", "")).strip(),
            )
            away_row = lookup_feature_row(
                by_id, by_name, season,
                str(game.get("away_id", "")).strip(),
                str(game.get("away_team", "")).strip(),
            )
            if home_row is None or away_row is None:
                continue
            if "is_d1" in features_df.columns:
                if not bool(home_row.get("is_d1", False)) or not bool(away_row.get("is_d1", False)):
                    continue

            # Orient matchup so team_A = better team by adj_margin to remove label bias.
            # This ensures the label consistently means "did the stronger team win?"
            home_adj = coerce_numeric_feature(home_row.get("adj_margin", 0.0))
            away_adj = coerce_numeric_feature(away_row.get("adj_margin", 0.0))
            if home_adj >= away_adj:
                team_a_row, team_b_row = home_row, away_row
                label = int(game["home_win"])
            else:
                team_a_row, team_b_row = away_row, home_row
                label = 1 - int(game["home_win"])

            row_features = {
                f"diff_{col}": coerce_numeric_feature(team_a_row.get(col, 0.0)) - coerce_numeric_feature(team_b_row.get(col, 0.0))
                for col in feature_cols
            }
            # neutral_site is always 1.0 when game_scope=ncaa_tourney
            neutral_site = int(game.get("is_neutral", 0))
            row_features["neutral_site"] = float(neutral_site)
            row_features["is_tournament"] = 1.0
            if include_interactions:
                row_features = add_interaction_features(row_features)

            X_rows.append(row_features)
            y_rows.append(label)
            weight_rows.append(row_weight)
            meta_rows.append({
                "season": season,
                "game_id": str(game.get("game_id", "")),
                "tournament": str(game.get("tournament", "")),
                "team_a": str(team_a_row.get("team", "")),
                "team_b": str(team_b_row.get("team", "")),
            })

    # Primary dataset: tournament games (full weight)
    for games_file in sorted(Path(games_dir).glob("games_*.csv")):
        _process_games_file(games_file, row_weight=1.0)

    if not X_rows:
        raise ValueError("No matchup rows were built. Check feature files and game scope.")

    # Optional: augment with regular-season games at reduced weight.
    # This expands from ~334 rows to ~30k rows, dramatically improving XGBoost stability.
    # Regular-season games use game_scope="all" so neutral_site varies and adds signal.
    if include_regular_season and game_scope == "ncaa_tourney":
        print(f"Augmenting with regular-season games (weight={regular_season_weight})...")
        orig_len = len(X_rows)
        for games_file in sorted(Path(games_dir).glob("games_*.csv")):
            season = int(games_file.stem.split("_")[1])
            games_all = pd.read_csv(games_file)
            # Regular season = not postseason
            regular_mask = ~coerce_bool(games_all.get("is_postseason", False))
            games_all = games_all[regular_mask].copy()
            games_all["home_win"] = coerce_bool(games_all.get("home_win", False)).astype(int)
            games_all["is_neutral"] = coerce_bool(games_all.get("is_neutral", False)).astype(int)
            # (iterates regular-season game rows inline)
            # Write to a temp path and re-use _process_games_file logic inline
            for _, game in games_all.iterrows():
                home_row = lookup_feature_row(
                    by_id, by_name, season,
                    str(game.get("home_id", "")).strip(),
                    str(game.get("home_team", "")).strip(),
                )
                away_row = lookup_feature_row(
                    by_id, by_name, season,
                    str(game.get("away_id", "")).strip(),
                    str(game.get("away_team", "")).strip(),
                )
                if home_row is None or away_row is None:
                    continue
                if "is_d1" in features_df.columns:
                    if not bool(home_row.get("is_d1", False)) or not bool(away_row.get("is_d1", False)):
                        continue
                home_adj = coerce_numeric_feature(home_row.get("adj_margin", 0.0))
                away_adj = coerce_numeric_feature(away_row.get("adj_margin", 0.0))
                if home_adj >= away_adj:
                    team_a_row, team_b_row = home_row, away_row
                    label = int(game["home_win"])
                else:
                    team_a_row, team_b_row = away_row, home_row
                    label = 1 - int(game["home_win"])
                row_features = {
                    f"diff_{col}": coerce_numeric_feature(team_a_row.get(col, 0.0)) - coerce_numeric_feature(team_b_row.get(col, 0.0))
                    for col in feature_cols
                }
                row_features["neutral_site"] = float(int(game.get("is_neutral", 0)))
                row_features["is_tournament"] = 0.0
                if include_interactions:
                    row_features = add_interaction_features(row_features)
                X_rows.append(row_features)
                y_rows.append(label)
                weight_rows.append(regular_season_weight)
                meta_rows.append({
                    "season": season,
                    "game_id": str(game.get("game_id", "")),
                    "tournament": "regular_season",
                    "team_a": str(team_a_row.get("team", "")),
                    "team_b": str(team_b_row.get("team", "")),
                })
        print(f"  Added {len(X_rows) - orig_len} regular-season rows (total: {len(X_rows)})")

    X = pd.DataFrame(X_rows).fillna(0.0)
    y = np.asarray(y_rows)
    weights = np.asarray(weight_rows, dtype=float)
    meta = pd.DataFrame(meta_rows)
    return X, y, meta, weights


def build_lr_model():
    return LogisticRegression(max_iter=2000, random_state=42)


def build_xgb_model():
    return XGBClassifier(
        n_estimators=100,
        max_depth=2,
        learning_rate=0.1,
        subsample=0.7,
        colsample_bytree=0.5,
        min_child_weight=10,
        reg_lambda=5.0,
        reg_alpha=1.0,
        random_state=42,
        eval_metric="logloss",
    )


def compute_metrics(y_true, probs):
    clipped = np.clip(np.asarray(probs), 1e-6, 1 - 1e-6)
    preds = (clipped >= 0.5).astype(int)
    return {
        "log_loss": float(log_loss(y_true, clipped, labels=[0, 1])),
        "brier": float(brier_score_loss(y_true, clipped)),
        "accuracy": float(accuracy_score(y_true, preds)),
        "games": int(len(y_true)),
    }


def _compute_baseline_accuracy(X, y, meta, col="diff_adj_margin"):
    """Baseline: always predict team_A wins (diff > 0 means team_A better)."""
    # Since team_A is always the better team by adj_margin, baseline = always predict 1
    # A smarter baseline uses diff_adj_margin sign
    if col in X.columns:
        preds = (X[col] >= 0).astype(int)
    else:
        preds = np.ones(len(y), dtype=int)
    return float(accuracy_score(y, preds))


def evaluate_loso(X, y, meta, lr_weight=0.65, xgb_weight=0.35):
    """Leave-one-season-out cross-validation."""
    seasons = sorted(meta["season"].unique().tolist())
    per_season = []
    all_probs = []
    all_y = []

    for holdout in seasons:
        train_mask = meta["season"] != holdout
        test_mask = meta["season"] == holdout
        if train_mask.sum() == 0 or test_mask.sum() == 0:
            continue

        X_train, X_test = X.loc[train_mask], X.loc[test_mask]
        y_train, y_test = y[train_mask.to_numpy()], y[test_mask.to_numpy()]

        lr = build_lr_model()
        lr.fit(X_train, y_train)
        p_lr = lr.predict_proba(X_test)[:, 1]

        model_probs = [p_lr]
        if HAS_XGB:
            xgb = build_xgb_model()
            xgb.fit(X_train, y_train)
            p_xgb = xgb.predict_proba(X_test)[:, 1]
            model_probs.append(p_xgb)

        total_w = max(1e-9, lr_weight + (xgb_weight if HAS_XGB else 0))
        weights = [lr_weight] + ([xgb_weight] if HAS_XGB else [])
        p_ens = sum(w * p for w, p in zip(weights, model_probs)) / total_w

        season_result = compute_metrics(y_test, p_ens)
        season_result["season"] = int(holdout)
        per_season.append(season_result)

        all_probs.extend(p_ens.tolist())
        all_y.extend(y_test.tolist())

    overall = compute_metrics(np.asarray(all_y), np.asarray(all_probs)) if all_y else {}

    # Bootstrap CI on overall accuracy
    if all_y:
        rng = np.random.default_rng(42)
        n = len(all_y)
        boot_scores = []
        for _ in range(1000):
            idx = rng.integers(0, n, n)
            preds = (np.asarray(all_probs)[idx] >= 0.5).astype(int)
            boot_scores.append(float(accuracy_score(np.asarray(all_y)[idx], preds)))
        overall["accuracy_ci_95"] = [float(np.percentile(boot_scores, 2.5)), float(np.percentile(boot_scores, 97.5))]

    return per_season, overall


def compute_baselines(X, y, meta):
    """Compute naive baseline accuracies for model comparison."""
    baselines = {}
    # Baseline 1: always predict team_A wins (team_A = better adj_margin by construction)
    baselines["always_team_a"] = float((y == 1).mean())
    # Baseline 2: predict by sign of diff_adj_margin
    if "diff_adj_margin" in X.columns:
        preds = (X["diff_adj_margin"] >= 0).astype(int)
        baselines["adj_margin_sign"] = float(accuracy_score(y, preds))
    # Baseline 3: predict by sign of diff_win_pct
    if "diff_win_pct" in X.columns:
        preds = (X["diff_win_pct"] >= 0).astype(int)
        baselines["win_pct_sign"] = float(accuracy_score(y, preds))
    # Baseline 4: predict by diff_seed (lower seed number = better team, so negative diff = team_A better)
    if "diff_seed" in X.columns:
        preds = (X["diff_seed"] <= 0).astype(int)
        baselines["lower_seed"] = float(accuracy_score(y, preds))
    return baselines


def fit_and_save_final_models(X, y, out_dir, sample_weight=None):
    lr = build_lr_model()
    lr.fit(X, y, sample_weight=sample_weight)
    joblib.dump(lr, Path(out_dir) / "lr_model.joblib")

    # LR: sigmoid (Platt scaling) is appropriate for linear models
    lr_cal = CalibratedClassifierCV(build_lr_model(), method="sigmoid", cv=5)
    lr_cal.fit(X, y, sample_weight=sample_weight)
    joblib.dump(lr_cal, Path(out_dir) / "lr_cal.joblib")

    if HAS_XGB:
        xgb = build_xgb_model()
        xgb.fit(X, y, sample_weight=sample_weight)
        joblib.dump(xgb, Path(out_dir) / "xgb_model.joblib")
        # XGBoost: isotonic regression is recommended over Platt scaling for tree ensembles.
        # Fall back to sigmoid when too few samples for isotonic (needs ~1000+).
        cal_method = "isotonic" if len(y) >= 1000 else "sigmoid"
        xgb_cal = CalibratedClassifierCV(build_xgb_model(), method=cal_method, cv=5)
        xgb_cal.fit(X, y, sample_weight=sample_weight)
        joblib.dump(xgb_cal, Path(out_dir) / "xgb_cal.joblib")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--features", default="data/processed/features/tournament_teams.csv")
    p.add_argument("--games_dir", default="data/processed")
    p.add_argument("--out_dir", default="models")
    p.add_argument("--game_scope", choices=["ncaa_tourney", "postseason", "all"], default="ncaa_tourney")
    p.add_argument("--interactions", action="store_true", default=False,
                   help="Include interaction features (off by default; requires large dataset)")
    p.add_argument("--include_regular_season", action="store_true", default=False,
                   help="Augment tournament training data with regular-season games at reduced weight (~100x more rows)")
    p.add_argument("--regular_season_weight", type=float, default=0.3,
                   help="Sample weight for regular-season rows when --include_regular_season is set (default: 0.3)")
    p.add_argument("--lr_weight", type=float, default=0.65)
    p.add_argument("--xgb_weight", type=float, default=0.35)
    args = p.parse_args()

    ensure_dir(args.out_dir)

    features_path = Path(args.features)
    if not features_path.exists() and features_path.name == "tournament_teams.csv":
        features_path = features_path.with_name("teams.csv")

    feats = load_features(features_path)
    X, y, meta, weights = build_match_dataset(
        args.games_dir, feats, args.game_scope,
        include_interactions=args.interactions,
        include_regular_season=args.include_regular_season,
        regular_season_weight=args.regular_season_weight,
    )
    print(f"Built dataset {X.shape} across seasons {sorted(meta['season'].unique().tolist())}")

    # LOSO evaluation runs only on tournament rows (weight==1.0) for fair evaluation
    tourney_mask = weights == 1.0
    X_tourney = X.loc[tourney_mask].reset_index(drop=True)
    y_tourney = y[tourney_mask]
    meta_tourney = meta.loc[tourney_mask].reset_index(drop=True)

    loso_per_season, loso_overall = evaluate_loso(X_tourney, y_tourney, meta_tourney,
                                                   lr_weight=args.lr_weight,
                                                   xgb_weight=args.xgb_weight)
    baselines = compute_baselines(X_tourney, y_tourney, meta_tourney)

    if loso_per_season:
        print("LOSO evaluation (leave-one-season-out):")
        for r in loso_per_season:
            print(f"  {r['season']}: accuracy={r['accuracy']:.4f} logloss={r['log_loss']:.4f} ({r['games']} games)")
    if loso_overall:
        ci = loso_overall.get("accuracy_ci_95", [None, None])
        print(f"LOSO overall: accuracy={loso_overall['accuracy']:.4f} 95%CI=[{ci[0]:.3f},{ci[1]:.3f}]")
        print(f"Baselines: {baselines}")

    fit_and_save_final_models(X, y, args.out_dir,
                              sample_weight=weights if args.include_regular_season else None)
    joblib.dump(list(X.columns), Path(args.out_dir) / "model_features.joblib")

    summary = {
        "features_path": str(features_path),
        "games_dir": args.games_dir,
        "game_scope": args.game_scope,
        "include_interactions": args.interactions,
        "rows": int(len(X)),
        "tourney_rows": int(tourney_mask.sum()),
        "feature_columns": list(X.columns),
        "seasons": sorted(int(season) for season in meta["season"].unique().tolist()),
        "baselines": baselines,
        "loso_per_season": loso_per_season,
        "loso_overall": loso_overall,
        # backward-compatible keys
        "holdout_results": loso_per_season,
        "overall_holdout_ensemble": loso_overall,
        "models": {
            "logistic_regression": True,
            "xgboost": bool(HAS_XGB),
            "calibration": "CalibratedClassifierCV(sigmoid/isotonic, cv=5)",
        },
    }
    with open(Path(args.out_dir) / "training_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(f"Saved models and training summary to {args.out_dir}")


if __name__ == "__main__":
    main()
