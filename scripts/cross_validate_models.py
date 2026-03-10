#!/usr/bin/env python3
"""Cross-validated evaluation for tournament models.

Adds:
- Leave-one-season-out validation
- Rolling time-split validation
- Bootstrap CI for holdout accuracy
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss

from train_baseline import (
    HAS_XGB,
    build_feature_lookup,
    build_lr_model,
    build_match_dataset,
    build_xgb_model,
    coerce_bool,
    feature_columns_from_df,
    filter_games,
    load_features,
)


def clipped_metrics(y_true, probs):
    p = np.clip(np.asarray(probs), 1e-6, 1 - 1e-6)
    pred = (p >= 0.5).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, pred)),
        "log_loss": float(log_loss(y_true, p, labels=[0, 1])),
        "brier": float(brier_score_loss(y_true, p)),
        "games": int(len(y_true)),
    }


def bootstrap_accuracy_ci(y_true, probs, n_boot=1000, seed=42):
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    probs = np.asarray(probs)
    n = len(y_true)
    if n == 0:
        return {"mean": 0.0, "ci_95_low": 0.0, "ci_95_high": 0.0}

    scores = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        y_b = y_true[idx]
        p_b = probs[idx]
        pred_b = (p_b >= 0.5).astype(int)
        scores.append(float(accuracy_score(y_b, pred_b)))

    scores = np.asarray(scores)
    return {
        "mean": float(scores.mean()),
        "ci_95_low": float(np.percentile(scores, 2.5)),
        "ci_95_high": float(np.percentile(scores, 97.5)),
    }


def compute_baselines(X, y, meta):
    """Compute naive baseline accuracies for LOSO comparison.

    Since team_A is oriented as the stronger team (by adj_margin),
    'always predict team_A wins' is the strongest naive baseline.
    """
    baselines = {}

    # Baseline: always predict team_A (the better adj_margin team) wins
    baselines["always_team_a"] = float((y == 1).mean())

    # Baseline: predict by sign of diff_adj_margin (team_A better adj margin = predict win)
    if "diff_adj_margin" in X.columns:
        preds_adj = (X["diff_adj_margin"] >= 0).astype(int)
        baselines["adj_margin_sign"] = float(accuracy_score(y, preds_adj))

    # Baseline: predict by sign of diff_win_pct
    if "diff_win_pct" in X.columns:
        preds_wp = (X["diff_win_pct"] >= 0).astype(int)
        baselines["win_pct_sign"] = float(accuracy_score(y, preds_wp))

    # Baseline: predict by seed differential (lower seed = better = more likely team_A)
    if "diff_seed" in X.columns:
        preds_seed = (X["diff_seed"] <= 0).astype(int)
        baselines["lower_seed_wins"] = float(accuracy_score(y, preds_seed))

    return baselines


def evaluate_loso(X, y, meta, lr_weight=0.65, xgb_weight=0.35):
    results = []
    all_probs = []
    all_y = []

    seasons = sorted(meta["season"].unique().tolist())
    for holdout in seasons:
        train_mask = meta["season"] != holdout
        test_mask = meta["season"] == holdout
        if train_mask.sum() == 0 or test_mask.sum() == 0:
            continue

        X_train = X.loc[train_mask]
        X_test = X.loc[test_mask]
        y_train = y[train_mask.to_numpy()]
        y_test = y[test_mask.to_numpy()]

        lr = build_lr_model()
        lr.fit(X_train, y_train)
        p_lr = lr.predict_proba(X_test)[:, 1]

        if HAS_XGB:
            xgb = build_xgb_model()
            xgb.fit(X_train, y_train)
            p_xgb = xgb.predict_proba(X_test)[:, 1]
            total_w = max(1e-9, lr_weight + xgb_weight)
            p_ens = (lr_weight * p_lr + xgb_weight * p_xgb) / total_w
        else:
            p_ens = p_lr

        m = clipped_metrics(y_test, p_ens)
        m["season"] = int(holdout)
        results.append(m)

        all_probs.extend(p_ens.tolist())
        all_y.extend(y_test.tolist())

    overall = clipped_metrics(np.asarray(all_y), np.asarray(all_probs)) if all_y else None
    ci = bootstrap_accuracy_ci(np.asarray(all_y), np.asarray(all_probs)) if all_y else None
    return results, overall, ci


def evaluate_rolling(X, y, meta, lr_weight=0.65, xgb_weight=0.35):
    results = []
    seasons = sorted(meta["season"].unique().tolist())

    for holdout in seasons[1:]:
        train_mask = meta["season"] < holdout
        test_mask = meta["season"] == holdout
        if train_mask.sum() == 0 or test_mask.sum() == 0:
            continue

        X_train = X.loc[train_mask]
        X_test = X.loc[test_mask]
        y_train = y[train_mask.to_numpy()]
        y_test = y[test_mask.to_numpy()]

        lr = build_lr_model()
        lr.fit(X_train, y_train)
        p_lr = lr.predict_proba(X_test)[:, 1]

        if HAS_XGB:
            xgb = build_xgb_model()
            xgb.fit(X_train, y_train)
            p_xgb = xgb.predict_proba(X_test)[:, 1]
            total_w = max(1e-9, lr_weight + xgb_weight)
            p_ens = (lr_weight * p_lr + xgb_weight * p_xgb) / total_w
        else:
            p_ens = p_lr

        m = clipped_metrics(y_test, p_ens)
        m["holdout_season"] = int(holdout)
        m["train_seasons"] = [int(s) for s in seasons if s < holdout]
        results.append(m)

    if not results:
        return results, None

    overall = {
        "accuracy_mean": float(np.mean([r["accuracy"] for r in results])),
        "accuracy_std": float(np.std([r["accuracy"] for r in results])),
        "log_loss_mean": float(np.mean([r["log_loss"] for r in results])),
        "brier_mean": float(np.mean([r["brier"] for r in results])),
        "splits": int(len(results)),
    }
    return results, overall


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", default="data/processed/features/tournament_teams.csv")
    parser.add_argument("--games_dir", default="data/processed")
    parser.add_argument("--game_scope", choices=["ncaa_tourney", "postseason", "all"], default="ncaa_tourney")
    parser.add_argument("--lr_weight", type=float, default=0.65)
    parser.add_argument("--xgb_weight", type=float, default=0.35)
    parser.add_argument("--out", default="results/cross_validation_summary.json")
    args = parser.parse_args()

    features_path = Path(args.features)
    if not features_path.exists() and features_path.name == "tournament_teams.csv":
        features_path = features_path.with_name("teams.csv")

    feats = load_features(features_path)
    X, y, meta, _weights = build_match_dataset(args.games_dir, feats, args.game_scope)

    print(f"Dataset: {X.shape[0]} games, {X.shape[1]} features")
    print(f"Seasons: {sorted(meta['season'].unique().tolist())}")

    baselines = compute_baselines(X, y, meta)
    print(f"Baselines: {baselines}")

    loso_results, loso_overall, loso_ci = evaluate_loso(
        X, y, meta, lr_weight=args.lr_weight, xgb_weight=args.xgb_weight
    )
    rolling_results, rolling_overall = evaluate_rolling(
        X, y, meta, lr_weight=args.lr_weight, xgb_weight=args.xgb_weight
    )

    print("\nLeave-One-Season-Out (LOSO):")
    for row in loso_results:
        print(
            f"  {row['season']}: acc={row['accuracy']:.4f} "
            f"logloss={row['log_loss']:.4f} brier={row['brier']:.4f} games={row['games']}"
        )
    if loso_overall:
        print(
            f"Overall LOSO: acc={loso_overall['accuracy']:.4f}, "
            f"logloss={loso_overall['log_loss']:.4f}, brier={loso_overall['brier']:.4f}"
        )
    if loso_ci:
        print(
            f"Accuracy 95% CI (bootstrap): "
            f"[{loso_ci['ci_95_low']:.4f}, {loso_ci['ci_95_high']:.4f}]"
        )

    best_baseline = max(baselines.values()) if baselines else 0.0
    model_acc = loso_overall.get("accuracy", 0.0) if loso_overall else 0.0
    gap = model_acc - best_baseline
    print(f"Model vs best baseline: {model_acc:.4f} vs {best_baseline:.4f} (gap: {gap:+.4f})")

    print("\nRolling Time Split:")
    for row in rolling_results:
        print(
            f"  holdout {row['holdout_season']}: acc={row['accuracy']:.4f} "
            f"logloss={row['log_loss']:.4f}"
        )
    if rolling_overall:
        print(
            f"Rolling mean acc={rolling_overall['accuracy_mean']:.4f} "
            f"(std={rolling_overall['accuracy_std']:.4f})"
        )

    summary = {
        "features_path": str(features_path),
        "game_scope": args.game_scope,
        "rows": int(X.shape[0]),
        "feature_columns": list(X.columns),
        "ensemble_weights": {"lr_weight": float(args.lr_weight), "xgb_weight": float(args.xgb_weight)},
        "loso": {
            "per_season": loso_results,
            "overall": loso_overall,
            "accuracy_bootstrap_ci": loso_ci,
        },
        "baselines": baselines,
        "rolling": {
            "per_split": rolling_results,
            "overall": rolling_overall,
        },
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved cross-validation summary to {out_path}")


if __name__ == "__main__":
    main()
