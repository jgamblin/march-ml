#!/usr/bin/env python3
"""Train and evaluate seed-stratified tournament models.

Routes historical tournament matchups into three strata:
- chalk: abs(seed_diff) >= 9
- competitive: abs(seed_diff) <= 4
- balanced: everything else

Outputs:
- models_seed_stratified/<stratum>_* artifacts
- results/seed_stratified_summary.json
"""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV

from train_baseline import (
    HAS_XGB,
    build_lr_model,
    build_match_dataset,
    build_xgb_model,
    compute_metrics,
    ensure_dir,
    load_features,
)


STRATA = {
    "chalk": lambda x: x >= 9,
    "competitive": lambda x: x <= 4,
    "balanced": lambda x: (x > 4) & (x < 9),
}


def fit_models(X_train, y_train):
    lr = build_lr_model()
    lr.fit(X_train, y_train)
    lr_cal = CalibratedClassifierCV(build_lr_model(), method="sigmoid", cv=min(5, max(2, int(np.bincount(y_train).min()))))
    lr_cal.fit(X_train, y_train)

    xgb = None
    xgb_cal = None
    if HAS_XGB:
        xgb = build_xgb_model()
        xgb.fit(X_train, y_train)
        xgb_cal = CalibratedClassifierCV(build_xgb_model(), method="sigmoid", cv=min(5, max(2, int(np.bincount(y_train).min()))))
        xgb_cal.fit(X_train, y_train)

    return lr, lr_cal, xgb, xgb_cal


def predict_ensemble(lr, xgb, X, lr_weight=0.65, xgb_weight=0.35):
    p_lr = lr.predict_proba(X)[:, 1]
    if xgb is None:
        return p_lr
    p_xgb = xgb.predict_proba(X)[:, 1]
    total = max(1e-9, lr_weight + xgb_weight)
    return (lr_weight * p_lr + xgb_weight * p_xgb) / total


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--features", default="data/processed/features/tournament_teams.csv")
    p.add_argument("--games_dir", default="data/processed")
    p.add_argument("--out_dir", default="models_seed_stratified")
    p.add_argument("--summary_out", default="results/seed_stratified_summary.json")
    p.add_argument("--game_scope", choices=["ncaa_tourney", "postseason", "all"], default="ncaa_tourney")
    args = p.parse_args()

    ensure_dir(args.out_dir)
    Path(args.summary_out).parent.mkdir(parents=True, exist_ok=True)

    feats = load_features(args.features)
    X, y, meta, _weights = build_match_dataset(args.games_dir, feats, args.game_scope)
    if "diff_seed" not in X.columns:
        raise ValueError("Seed-stratified training requires diff_seed in feature matrix")

    abs_seed_diff = X["diff_seed"].abs()
    summary = {
        "features": args.features,
        "rows": int(len(X)),
        "game_scope": args.game_scope,
        "strata": {},
    }

    for name, fn in STRATA.items():
        mask = fn(abs_seed_diff)
        X_s = X.loc[mask].copy()
        y_s = y[mask.to_numpy()]
        meta_s = meta.loc[mask].copy()
        if len(X_s) < 20 or len(np.unique(y_s)) < 2:
            summary["strata"][name] = {
                "rows": int(len(X_s)),
                "status": "skipped_insufficient_data",
            }
            continue

        seasons = sorted(meta_s["season"].unique().tolist())
        holdouts = []
        all_y = []
        all_p = []
        for holdout in seasons[1:]:
            train_mask = meta_s["season"] < holdout
            test_mask = meta_s["season"] == holdout
            if train_mask.sum() < 10 or test_mask.sum() == 0:
                continue
            X_train = X_s.loc[train_mask]
            X_test = X_s.loc[test_mask]
            y_train = y_s[train_mask.to_numpy()]
            y_test = y_s[test_mask.to_numpy()]
            if len(np.unique(y_train)) < 2:
                continue
            lr, lr_cal, xgb, xgb_cal = fit_models(X_train, y_train)
            probs = predict_ensemble(lr_cal, xgb_cal, X_test)
            metrics = compute_metrics(y_test, probs)
            metrics["season"] = int(holdout)
            holdouts.append(metrics)
            all_y.extend(y_test.tolist())
            all_p.extend(probs.tolist())

        lr, lr_cal, xgb, xgb_cal = fit_models(X_s, y_s)
        joblib.dump(lr, Path(args.out_dir) / f"{name}_lr_model.joblib")
        joblib.dump(lr_cal, Path(args.out_dir) / f"{name}_lr_cal.joblib")
        if xgb is not None:
            joblib.dump(xgb, Path(args.out_dir) / f"{name}_xgb_model.joblib")
            joblib.dump(xgb_cal, Path(args.out_dir) / f"{name}_xgb_cal.joblib")
        joblib.dump(list(X.columns), Path(args.out_dir) / f"{name}_model_features.joblib")

        summary["strata"][name] = {
            "rows": int(len(X_s)),
            "seasons": [int(s) for s in seasons],
            "holdout_results": holdouts,
            "overall": compute_metrics(np.asarray(all_y), np.asarray(all_p)) if all_y else None,
            "status": "trained",
        }
        print(f"{name}: rows={len(X_s)} overall={summary['strata'][name]['overall']}")

    with open(args.summary_out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved seed-stratified summary to {args.summary_out}")


if __name__ == "__main__":
    main()
