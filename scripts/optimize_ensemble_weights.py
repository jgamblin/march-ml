"""Optimize ensemble weights for LR + XGBoost model combination.

Uses LOSO (Leave-One-Season-Out) cross-validation to find optimal weights —
weights are chosen based on held-out predictions only, so there is no
train/test leakage.

Writes models/ensemble_weights.json with the best lr_weight / xgb_weight,
which simulate_bracket.py reads automatically.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

# Reuse the exact same dataset builder and model constructors as training
sys.path.insert(0, str(Path(__file__).parent))
from train_baseline import (
    build_match_dataset,
    build_lr_model,
    build_xgb_model,
    load_features,
    HAS_XGB,
)


def loso_per_model_probs(X, y, meta, weights=None):
    """Run LOSO and collect per-fold (y_true, p_lr, p_xgb) tuples.

    Trains on all data (including regular season) from k-1 seasons;
    evaluates on tournament rows from the holdout season.
    """
    tourney_mask = (weights == 1.0) if weights is not None else np.ones(len(y), dtype=bool)
    seasons = sorted(meta.loc[tourney_mask, "season"].unique().tolist())

    all_y, all_p_lr, all_p_xgb = [], [], []

    for holdout in seasons:
        train_mask = meta["season"] != holdout
        test_mask = (meta["season"] == holdout) & tourney_mask

        X_train = X.loc[train_mask]
        y_train = y[train_mask.to_numpy()]
        w_train = weights[train_mask.to_numpy()] if weights is not None else None

        X_test = X.loc[test_mask]
        y_test = y[test_mask.to_numpy()]

        if train_mask.sum() == 0 or test_mask.sum() == 0:
            continue

        lr = build_lr_model()
        lr.fit(X_train, y_train, sample_weight=w_train)
        p_lr = lr.predict_proba(X_test)[:, 1]

        if HAS_XGB:
            xgb = build_xgb_model()
            xgb.fit(X_train, y_train, sample_weight=w_train)
            p_xgb = xgb.predict_proba(X_test)[:, 1]
        else:
            p_xgb = p_lr  # fall back to LR-only

        all_y.extend(y_test.tolist())
        all_p_lr.extend(p_lr.tolist())
        all_p_xgb.extend(p_xgb.tolist())

    return np.array(all_y), np.array(all_p_lr), np.array(all_p_xgb)


def grid_search_weights(all_y, all_p_lr, all_p_xgb, step=0.05):
    """Grid search lr_weight in [0, step, …, 1] on stacked LOSO predictions."""
    from sklearn.metrics import accuracy_score, log_loss

    results = []
    for lr_w in np.arange(0.0, 1.0 + step / 2, step):
        xgb_w = 1.0 - lr_w
        p_ens = lr_w * all_p_lr + xgb_w * all_p_xgb
        preds = (p_ens >= 0.5).astype(int)
        results.append({
            "lr_weight": round(float(lr_w), 4),
            "xgb_weight": round(float(xgb_w), 4),
            "accuracy": float(accuracy_score(all_y, preds)),
            "log_loss": float(log_loss(all_y, np.clip(p_ens, 1e-6, 1 - 1e-6))),
        })

    best_acc = max(results, key=lambda r: r["accuracy"])
    best_ll = min(results, key=lambda r: r["log_loss"])
    baseline = next(r for r in results if abs(r["lr_weight"] - 0.5) < 1e-6)
    return results, best_acc, best_ll, baseline


def nested_loso_eval(X, y, meta, weights=None, step=0.05):
    """Nested LOSO: unbiased estimate of ensemble performance.

    For each test season t (starting from the 3rd so we have ≥2 prior seasons):
      - Inner loop: optimize lr_weight by log-loss on LOSO over prior seasons only.
      - Outer loop: train on all-but-t, blend with inner-optimal lr_weight, eval on t.

    Returns a dict with per_season list and overall accuracy/log_loss.
    This is diagnostic only — production weights still come from the naive LOSO.
    """
    from sklearn.metrics import accuracy_score, log_loss as sk_log_loss

    tourney_mask = (weights == 1.0) if weights is not None else np.ones(len(y), dtype=bool)
    seasons = sorted(meta.loc[tourney_mask, "season"].unique().tolist())

    if len(seasons) < 3:
        return {"per_season": [], "accuracy": None, "log_loss": None}

    per_season = []
    all_y_out: list = []
    all_p_out: list = []

    for idx in range(2, len(seasons)):
        test_season = seasons[idx]
        prior_seasons = seasons[:idx]

        # ── Inner loop: find optimal lr_weight using LOSO on prior seasons only ──
        inner_row_mask = np.isin(meta["season"].to_numpy(), prior_seasons)
        X_inner = X.loc[inner_row_mask]
        y_inner = y[inner_row_mask]
        meta_inner = meta.loc[inner_row_mask]
        w_inner = weights[inner_row_mask] if weights is not None else None

        inner_all_y, inner_p_lr, inner_p_xgb = loso_per_model_probs(
            X_inner, y_inner, meta_inner, weights=w_inner
        )
        if len(inner_all_y) == 0:
            continue
        _, _, inner_best_ll, _ = grid_search_weights(inner_all_y, inner_p_lr, inner_p_xgb, step=step)
        lr_w = inner_best_ll["lr_weight"]
        xgb_w = inner_best_ll["xgb_weight"]

        # ── Outer loop: train on all-but-test_season, evaluate on test_season ──
        train_mask = meta["season"] != test_season
        test_mask = (meta["season"] == test_season) & tourney_mask

        if train_mask.sum() == 0 or test_mask.sum() == 0:
            continue

        X_train = X.loc[train_mask]
        y_train = y[train_mask.to_numpy()]
        w_train = weights[train_mask.to_numpy()] if weights is not None else None
        X_test = X.loc[test_mask]
        y_test = y[test_mask.to_numpy()]

        lr = build_lr_model()
        lr.fit(X_train, y_train, sample_weight=w_train)
        p_lr = lr.predict_proba(X_test)[:, 1]

        if HAS_XGB:
            xgb = build_xgb_model()
            xgb.fit(X_train, y_train, sample_weight=w_train)
            p_xgb = xgb.predict_proba(X_test)[:, 1]
        else:
            p_xgb = p_lr

        p_ens = lr_w * p_lr + xgb_w * p_xgb
        preds = (p_ens >= 0.5).astype(int)

        per_season.append({
            "season": int(test_season),
            "lr_weight_used": round(float(lr_w), 4),
            "accuracy": float(accuracy_score(y_test, preds)),
            "log_loss": float(sk_log_loss(y_test, np.clip(p_ens, 1e-6, 1 - 1e-6))),
            "n": int(len(y_test)),
        })
        all_y_out.extend(y_test.tolist())
        all_p_out.extend(p_ens.tolist())

    if all_y_out:
        preds_all = (np.array(all_p_out) >= 0.5).astype(int)
        overall_acc = float(accuracy_score(all_y_out, preds_all))
        overall_ll = float(sk_log_loss(all_y_out, np.clip(all_p_out, 1e-6, 1 - 1e-6)))
    else:
        overall_acc = None
        overall_ll = None

    return {"per_season": per_season, "accuracy": overall_acc, "log_loss": overall_ll}


def main():
    parser = argparse.ArgumentParser(description="LOSO-based ensemble weight optimizer")
    parser.add_argument("--features", default="data/processed/features/tournament_teams.csv")
    parser.add_argument("--games_dir", default="data/processed")
    parser.add_argument("--models_dir", default="models")
    parser.add_argument("--include_regular_season", action="store_true", default=True)
    parser.add_argument("--regular_season_weight", type=float, default=0.3)
    parser.add_argument("--step", type=float, default=0.05, help="grid search step size")
    args = parser.parse_args()

    print("\n=== LOSO Ensemble Weight Optimization ===\n")

    features_path = Path(args.features)
    if not features_path.exists() and features_path.name == "tournament_teams.csv":
        features_path = features_path.with_name("teams.csv")

    feats = load_features(features_path)
    X, y, meta, weights = build_match_dataset(
        args.games_dir, feats, "ncaa_tourney",
        include_regular_season=args.include_regular_season,
        regular_season_weight=args.regular_season_weight,
    )
    print(f"Dataset: {X.shape[0]} rows, {X.shape[1]} features")

    print("Running LOSO to collect per-model predictions (this may take ~30s)...")
    all_y, all_p_lr, all_p_xgb = loso_per_model_probs(X, y, meta, weights=weights)
    print(f"Collected {len(all_y)} held-out tournament predictions\n")

    results, best_acc, best_ll, baseline = grid_search_weights(all_y, all_p_lr, all_p_xgb, args.step)

    print(f"{'LR':>6}  {'XGB':>6}  {'Accuracy':>10}  {'LogLoss':>10}")
    print("-" * 40)
    for r in results:
        marker = " ← best acc" if r == best_acc else (" ← best ll" if r == best_ll else "")
        print(f"{r['lr_weight']:>6.0%}  {r['xgb_weight']:>6.0%}  "
              f"{r['accuracy']:>10.4f}  {r['log_loss']:>10.4f}{marker}")

    print(f"\nBaseline (50/50):  acc={baseline['accuracy']:.4f}  ll={baseline['log_loss']:.4f}")
    print(f"Best by accuracy:  LR={best_acc['lr_weight']:.0%} / XGB={best_acc['xgb_weight']:.0%}"
          f"  acc={best_acc['accuracy']:.4f} ({best_acc['accuracy']-baseline['accuracy']:+.4f})")
    print(f"Best by log-loss:  LR={best_ll['lr_weight']:.0%} / XGB={best_ll['xgb_weight']:.0%}"
          f"  ll={best_ll['log_loss']:.4f} ({best_ll['log_loss']-baseline['log_loss']:+.4f})")

    # Save canonical weights file — simulate_bracket.py reads this automatically
    out_path = Path(args.models_dir) / "ensemble_weights.json"
    chosen = best_ll  # log-loss objective: better calibration for Monte Carlo sims
    output = {
        "lr_weight": chosen["lr_weight"],
        "xgb_weight": chosen["xgb_weight"],
        "accuracy": chosen["accuracy"],
        "log_loss": chosen["log_loss"],
        "baseline_5050_accuracy": baseline["accuracy"],
        "accuracy_improvement": round(chosen["accuracy"] - baseline["accuracy"], 4),
        "optimization_objective": "log_loss",
        "method": "loso_grid_search",
        "all_results": results,
    }
    out_path.write_text(json.dumps(output, indent=2))
    print(f"\n✓ Saved ensemble weights to {out_path}")
    print(f"  simulate_bracket.py will use LR={chosen['lr_weight']:.0%} / XGB={chosen['xgb_weight']:.0%} automatically")

    # ── Nested LOSO (unbiased diagnostic) ──────────────────────────────────────
    print("\n=== Nested LOSO (unbiased) — diagnostic only, does NOT change saved weights ===\n")
    print("Running nested LOSO (inner weight optimization + outer evaluation)...")
    nested = nested_loso_eval(X, y, meta, weights=weights, step=args.step)
    if nested["per_season"]:
        print(f"\n{'Season':>8}  {'LR Used':>8}  {'Accuracy':>10}  {'LogLoss':>10}  {'N':>5}")
        print("-" * 50)
        for r in nested["per_season"]:
            print(f"{r['season']:>8d}  {r['lr_weight_used']:>8.0%}  "
                  f"{r['accuracy']:>10.4f}  {r['log_loss']:>10.4f}  {r['n']:>5d}")
        print(f"\nNaive LOSO (optimistic):  acc={best_ll['accuracy']:.4f}  ll={best_ll['log_loss']:.4f}")
        print(f"Nested LOSO (unbiased):   acc={nested['accuracy']:.4f}  ll={nested['log_loss']:.4f}")
    else:
        print("Not enough seasons for nested LOSO (need ≥3 holdout-eligible seasons).")


if __name__ == "__main__":
    main()

