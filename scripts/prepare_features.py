"""Prepare season-level team features for modeling and simulation.

Outputs written under `data/processed/features/`:
- `season_aggregates_{season}.csv`: full-season aggregates using all games.
- `tournament_team_features_{season}.csv`: pre-tournament snapshot using only
  non-postseason games.
- `teams.csv`: combined full-season aggregates.
- `tournament_teams.csv`: combined pre-tournament snapshots.

Usage:
    python scripts/prepare_features.py --input data/processed --out data/processed/features
    python scripts/prepare_features.py --seasons 2021 2022
"""
import argparse
import glob
import os
from pathlib import Path

import numpy as np
import pandas as pd


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def coerce_bool(series):
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    lowered = series.astype(str).str.strip().str.lower()
    return lowered.isin({"true", "1", "yes", "y"})


def load_games_for_season(input_dir, season):
    path = Path(input_dir) / f"games_{season}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}")
    df = pd.read_csv(path)
    if "game_day" in df.columns:
        df["game_day"] = pd.to_datetime(df["game_day"], errors="coerce")
    for col in ["home_win", "is_neutral", "is_postseason"]:
        if col in df.columns:
            df[col] = coerce_bool(df[col])
        else:
            df[col] = False
    return df


def build_team_game_rows(games_df, season):
    home = games_df[
        [
            "game_id",
            "game_day",
            "home_team",
            "home_id",
            "home_score",
            "away_team",
            "away_id",
            "away_score",
            "home_win",
            "is_neutral",
            "is_postseason",
            "tournament",
        ]
    ].copy()
    home.columns = [
        "game_id",
        "game_day",
        "team",
        "team_id",
        "points",
        "opp",
        "opp_id",
        "opp_points",
        "win",
        "is_neutral",
        "is_postseason",
        "tournament",
    ]
    home["is_home"] = True

    away = games_df[
        [
            "game_id",
            "game_day",
            "away_team",
            "away_id",
            "away_score",
            "home_team",
            "home_id",
            "home_score",
            "home_win",
            "is_neutral",
            "is_postseason",
            "tournament",
        ]
    ].copy()
    away.columns = [
        "game_id",
        "game_day",
        "team",
        "team_id",
        "points",
        "opp",
        "opp_id",
        "opp_points",
        "home_win",
        "is_neutral",
        "is_postseason",
        "tournament",
    ]
    away["win"] = (~away["home_win"]).astype(int)
    away = away.drop(columns=["home_win"])
    away["is_home"] = False

    rows = pd.concat([home, away], ignore_index=True, sort=False)
    rows["season"] = season
    rows["points"] = pd.to_numeric(rows["points"], errors="coerce").fillna(0).astype(int)
    rows["opp_points"] = pd.to_numeric(rows["opp_points"], errors="coerce").fillna(0).astype(int)
    rows["win"] = pd.to_numeric(rows["win"], errors="coerce").fillna(0).astype(int)
    rows["margin"] = rows["points"] - rows["opp_points"]
    rows = rows.sort_values(["team", "game_day", "game_id"], na_position="last").reset_index(drop=True)
    return rows


def aggregate_team_season(rows):
    if rows.empty:
        return pd.DataFrame(
            columns=[
                "season",
                "team",
                "team_id",
                "games_played",
                "wins",
                "losses",
                "win_pct",
                "points_for",
                "points_against",
                "avg_points_for",
                "avg_points_against",
                "avg_margin",
                "last5_win_pct",
            ]
        )

    grouped = rows.groupby(["season", "team", "team_id"], dropna=False)
    agg = grouped.agg(
        games_played=("game_id", "size"),
        wins=("win", "sum"),
        points_for=("points", "sum"),
        points_against=("opp_points", "sum"),
        avg_points_for=("points", "mean"),
        avg_points_against=("opp_points", "mean"),
        avg_margin=("margin", "mean"),
    ).reset_index()
    agg["losses"] = agg["games_played"] - agg["wins"]
    agg["win_pct"] = agg["wins"] / agg["games_played"].clip(lower=1)

    last5 = grouped.apply(
        lambda df: float(df.sort_values(["game_day", "game_id"]).tail(5)["win"].mean()) if len(df) else 0.0,
        include_groups=False,
    ).reset_index(name="last5_win_pct")

    merged = agg.merge(last5, on=["season", "team", "team_id"], how="left")
    numeric_cols = [
        "games_played",
        "wins",
        "losses",
        "points_for",
        "points_against",
        "avg_points_for",
        "avg_points_against",
        "avg_margin",
        "win_pct",
        "last5_win_pct",
    ]
    for col in numeric_cols:
        merged[col] = pd.to_numeric(merged[col], errors="coerce").fillna(0.0)
    return merged.sort_values(["season", "team"]).reset_index(drop=True)


def add_rolling_form_features(team_games):
    """Add last-10 and trend features without postseason leakage.
    
    Args:
        team_games: All games for one team, sorted by game_day
    
    Returns:
        Dict with last10_wins, last10_losses, offense_trend, defense_trend
    """
    # Get games ordered by date (most recent last)
    sorted_games = team_games.sort_values("game_day").reset_index(drop=True)
    
    # Get last 10 games (or all if fewer than 10)
    last_10 = sorted_games.tail(10)
    last_10_wins = int(last_10["win"].sum())
    last_10_losses = len(last_10) - last_10_wins
    
    # Compute offense and defense trends
    if len(sorted_games) >= 10:
        season_ptsfor_avg = float(sorted_games["points"].mean())
        last10_ptsfor_avg = float(last_10["points"].mean())
        offense_trend = last10_ptsfor_avg - season_ptsfor_avg
        
        season_ptsagainst_avg = float(sorted_games["opp_points"].mean())
        last10_ptsagainst_avg = float(last_10["opp_points"].mean())
        defense_trend = season_ptsagainst_avg - last10_ptsagainst_avg  # lower is better
    else:
        offense_trend = 0.0
        defense_trend = 0.0
    
    return {
        'last10_wins': float(last_10_wins),
        'last10_losses': float(last_10_losses),
        'offense_trend': float(offense_trend),
        'defense_trend': float(defense_trend),
    }


def add_rolling_features_to_aggregates(rows, aggregated):
    """Add rolling form features to aggregated team stats.
    
    Computes last 10 game stats and trend direction for each team.
    """
    if rows.empty or aggregated.empty:
        for col in ['last10_wins', 'last10_losses', 'offense_trend', 'defense_trend']:
            if col not in aggregated.columns:
                aggregated[col] = 0.0
        return aggregated
    
    rolling_features = []
    for _, team_row in aggregated.iterrows():
        team_id = str(team_row.get("team_id", "")).strip()
        team_name = str(team_row.get("team", "")).strip()
        
        # Find all games for this team
        team_games = rows[
            (rows["team_id"].astype(str).str.strip() == team_id) |
            (rows["team"].astype(str).str.strip() == team_name)
        ]
        
        if len(team_games) == 0:
            rolling_features.append({
                'last10_wins': 0.0,
                'last10_losses': 0.0,
                'offense_trend': 0.0,
                'defense_trend': 0.0,
            })
        else:
            rolling_features.append(add_rolling_form_features(team_games))
    
    rolling_df = pd.DataFrame(rolling_features)
    enriched = pd.concat([aggregated.reset_index(drop=True), rolling_df], axis=1)
    return enriched


def add_opponent_strength_features(rows, aggregated, n_iter=2):
    """Compute SOS with n_iter iterations to reduce circular dependency.
    
    Iteration 1: use raw win_pct.
    Iteration 2: use SOS-adjusted win_pct (win_pct normalized by opponent quality).
    """
    if rows.empty or aggregated.empty:
        enriched = aggregated.copy()
        for col in ["sos_win_pct", "opp_avg_margin", "adj_margin"]:
            if col not in enriched.columns:
                enriched[col] = 0.0
        return enriched

    enriched = aggregated.copy()

    # Build fast lookup maps
    id_to_idx = {str(r.get("team_id", "")).strip(): i for i, (_, r) in enumerate(enriched.iterrows()) if str(r.get("team_id", "")).strip()}
    name_to_idx = {str(r.get("team", "")).strip().lower(): i for i, (_, r) in enumerate(enriched.iterrows()) if str(r.get("team", "")).strip()}

    def lookup_idx(opp_id, opp_name):
        oid = str(opp_id).strip()
        if oid and oid in id_to_idx:
            return id_to_idx[oid]
        oname = str(opp_name).strip().lower()
        if oname and oname in name_to_idx:
            return name_to_idx[oname]
        return None

    # Build per-team opponent index lists once
    team_opp_indices = []
    records = enriched.to_dict("records")
    for rec in records:
        tid = str(rec.get("team_id", "")).strip()
        tname = str(rec.get("team", "")).strip()
        team_rows = rows[
            (rows["team_id"].astype(str).str.strip() == tid) |
            (rows["team"].astype(str).str.strip() == tname)
        ]
        opp_indices = []
        opp_margins = []
        for _, gr in team_rows.iterrows():
            idx = lookup_idx(gr.get("opp_id", ""), gr.get("opp", ""))
            opp_indices.append(idx)
            opp_margins.append(float(gr.get("opp_points", 0) - gr.get("points", 0)))
        team_opp_indices.append((opp_indices, opp_margins))

    # Iterative SOS: start with raw win_pct
    current_win_pcts = [float(rec.get("win_pct", 0.5)) for rec in records]

    final_sos = [0.0] * len(records)
    for iteration in range(n_iter):
        new_sos = []
        for i, (opp_indices, _) in enumerate(team_opp_indices):
            valid_pcts = [current_win_pcts[j] for j in opp_indices if j is not None]
            new_sos.append(float(pd.Series(valid_pcts).mean()) if valid_pcts else 0.5)

        if iteration < n_iter - 1:
            # Build SOS-adjusted win_pcts for next iteration
            mean_sos = float(pd.Series(new_sos).mean()) if new_sos else 0.5
            mean_sos = max(mean_sos, 1e-6)
            adjusted = []
            for i, rec in enumerate(records):
                gp = max(int(rec.get("games_played", 1)), 1)
                wins = float(rec.get("wins", 0))
                # Adjust wins downward if SOS was easy, upward if SOS was hard
                sos_ratio = new_sos[i] / mean_sos
                adj_wins = wins / max(sos_ratio, 0.1)
                adjusted.append(float(np.clip(adj_wins / gp, 0.0, 1.0)))
            current_win_pcts = adjusted
        else:
            final_sos = new_sos

    enriched["sos_win_pct"] = final_sos

    # Compute adj_margin iteratively (Massey: adj_i ≈ avg_margin_i + avg_opp_adj_j, 2 iterations)
    # Iteration 0 uses raw avg_margin as the initial team quality estimate.
    # Each subsequent iteration uses the previous adj_margin as opponent quality.
    current_adj = [float(rec.get("avg_margin", 0.0)) for rec in records]
    for _ in range(2):
        new_adj = []
        for i, (opp_indices, _) in enumerate(team_opp_indices):
            valid = [current_adj[j] for j in opp_indices if j is not None]
            avg_opp = float(np.mean(valid)) if valid else 0.0
            new_adj.append(float(records[i].get("avg_margin", 0.0)) + avg_opp)
        current_adj = new_adj

    # opp_avg_margin: single-pass average of opponents' raw avg_margin (used as standalone feature)
    opp_margin_values = []
    for i, (opp_indices, _) in enumerate(team_opp_indices):
        valid = [float(records[j].get("avg_margin", 0.0)) for j in opp_indices if j is not None]
        opp_margin_values.append(float(np.mean(valid)) if valid else 0.0)

    enriched["opp_avg_margin"] = opp_margin_values
    enriched["adj_margin"] = current_adj
    return enriched


def merge_optional_mapping(df, mapping_path, value_col):
    if not mapping_path:
        return df
    path = Path(mapping_path)
    if not path.exists():
        return df
    mapping_df = pd.read_csv(path)
    join_cols = []
    if "season" in mapping_df.columns and "season" in df.columns:
        join_cols.append("season")
    if "team_id" in mapping_df.columns and "team_id" in df.columns:
        join_cols.append("team_id")
    elif "team" in mapping_df.columns and "team" in df.columns:
        join_cols.append("team")
    else:
        return df
    keep_cols = join_cols + [value_col]
    return df.merge(mapping_df[keep_cols].drop_duplicates(), on=join_cols, how="left")


def merge_optional_columns(df, source_path, value_cols):
    path = Path(source_path)
    if not path.exists():
        return df

    source = pd.read_csv(path)
    available_cols = [col for col in value_cols if col in source.columns and col not in df.columns]
    if not available_cols:
        return df

    join_cols = []
    if "season" in source.columns and "season" in df.columns:
        join_cols.append("season")
    if "team_id" in source.columns and "team_id" in df.columns:
        join_cols.append("team_id")
    elif "team" in source.columns and "team" in df.columns:
        join_cols.append("team")
    else:
        return df

    keep_cols = join_cols + available_cols
    return df.merge(source[keep_cols].drop_duplicates(), on=join_cols, how="left")


def apply_auto_enrichment(df, out_dir):
    enriched = df.copy()

    conference_path = Path(out_dir) / "tournament_teams_with_conferences.csv"
    conference_cols = ["conference", "conf_avg_adj_margin", "conf_avg_win_pct", "conf_strength_tier"]
    enriched = merge_optional_columns(enriched, conference_path, conference_cols)

    momentum_cols = [
        "weighted_last10_margin",
        "win_streak",
        "margin_trend_slope",
        "last5_momentum",
        "last10_momentum",
        "form_rating",
    ]
    momentum_sources = sorted(Path(out_dir).glob("season_aggregates_*_with_momentum.csv"))
    for source in momentum_sources:
        enriched = merge_optional_columns(enriched, source, momentum_cols)

    momentum_cols_fill = momentum_cols
    if "conf_avg_adj_margin" in enriched.columns:
        enriched["conf_avg_adj_margin"] = pd.to_numeric(enriched["conf_avg_adj_margin"], errors="coerce").fillna(0.0)
    if "conf_avg_win_pct" in enriched.columns:
        enriched["conf_avg_win_pct"] = pd.to_numeric(enriched["conf_avg_win_pct"], errors="coerce").fillna(0.5)
    if "conf_strength_tier" in enriched.columns:
        # Map string tiers to numbers, then fill unknown D-I teams with 1.0 (mid-major default)
        def tier_to_num(val):
            if pd.isna(val):
                return None
            v = str(val).strip().lower()
            return {"power-6": 3.0, "high-major": 2.0, "mid-major": 1.0}.get(v, None)
        enriched["conf_strength_tier"] = enriched["conf_strength_tier"].apply(tier_to_num)
        # For D-I teams missing tier, default to mid-major (1.0); non-D-I teams stay 0.0
        if "is_d1" in enriched.columns:
            d1_mask = enriched["is_d1"].astype(bool)
            enriched.loc[d1_mask & enriched["conf_strength_tier"].isna(), "conf_strength_tier"] = 1.0
            enriched.loc[~d1_mask & enriched["conf_strength_tier"].isna(), "conf_strength_tier"] = 0.0
        else:
            enriched["conf_strength_tier"] = enriched["conf_strength_tier"].fillna(1.0)
    for col in momentum_cols_fill:
        if col in enriched.columns:
            enriched[col] = pd.to_numeric(enriched[col], errors="coerce").fillna(0.0)

    return enriched


def apply_d1_tags(df, d1_list_path):
    tagged = df.copy()
    default_mask = ~tagged["team_id"].astype(str).str.startswith("nd-")
    if not d1_list_path:
        tagged["is_d1"] = default_mask
        return tagged

    path = Path(d1_list_path)
    if not path.exists():
        tagged["is_d1"] = default_mask
        return tagged

    try:
        d1_df = pd.read_csv(path)
        if "team_id" in d1_df.columns:
            d1_ids = set(d1_df["team_id"].astype(str).str.strip())
            tagged["is_d1"] = tagged["team_id"].astype(str).str.strip().isin(d1_ids)
        elif "team" in d1_df.columns:
            d1_names = set(d1_df["team"].astype(str).str.strip().str.lower())
            tagged["is_d1"] = tagged["team"].astype(str).str.strip().str.lower().isin(d1_names)
        else:
            tagged["is_d1"] = default_mask
    except Exception:
        lines = [line.strip().lower() for line in path.read_text().splitlines() if line.strip()]
        tagged["is_d1"] = tagged["team"].astype(str).str.strip().str.lower().isin(set(lines))

    # sanity-check coverage; if the custom file clearly under-matches, fall back
    # to the built-in heuristic based on ESPN non-D1 ids.
    if int(tagged["is_d1"].sum()) < 250:
        tagged["is_d1"] = default_mask
    return tagged


def extract_seeds_from_games(games_df, season, out_dir):
    """Extract tournament seeds from game files.

    cbbpy stores the tournament seed in home_rank/away_rank for NCAA tournament
    games (not AP poll rank). This gives us historical seeds without needing
    external bracket files.
    """
    tournament_mask = (
        games_df.get("tournament", pd.Series("", index=games_df.index))
        .fillna("").astype(str)
        .str.contains("Men's Basketball Championship", case=False, na=False)
    )
    tourney = games_df[tournament_mask].copy()
    if tourney.empty:
        return pd.DataFrame(columns=["season", "team", "team_id", "seed"])

    seeds = []
    for _, row in tourney.iterrows():
        h_seed = pd.to_numeric(row.get("home_rank", None), errors="coerce")
        a_seed = pd.to_numeric(row.get("away_rank", None), errors="coerce")
        if not pd.isna(h_seed) and h_seed > 0:
            seeds.append({
                "season": season,
                "team": str(row.get("home_team", "")).strip(),
                "team_id": str(row.get("home_id", "")).strip(),
                "seed": int(h_seed),
            })
        if not pd.isna(a_seed) and a_seed > 0:
            seeds.append({
                "season": season,
                "team": str(row.get("away_team", "")).strip(),
                "team_id": str(row.get("away_id", "")).strip(),
                "seed": int(a_seed),
            })

    if not seeds:
        return pd.DataFrame(columns=["season", "team", "team_id", "seed"])

    df = pd.DataFrame(seeds).drop_duplicates(subset=["team_id"]).reset_index(drop=True)
    path = Path(out_dir) / f"seeds_{season}.csv"
    ensure_dir(out_dir)
    df.to_csv(path, index=False)
    return df


def merge_seeds(df, seeds_df):
    """Left-join seeds onto a features DataFrame by team_id (fallback to team name)."""
    if seeds_df.empty or "seed" not in seeds_df.columns:
        return df
    keep = seeds_df[["season", "team_id", "seed"]].drop_duplicates()
    merged = df.merge(keep, on=["season", "team_id"], how="left")
    # Fallback: name-based join for teams where team_id didn't match
    if merged["seed"].isna().any() and "team" in seeds_df.columns:
        name_map = seeds_df.set_index(["season", "team"])["seed"].to_dict()
        missing = merged["seed"].isna()
        merged.loc[missing, "seed"] = merged.loc[missing].apply(
            lambda r: name_map.get((r["season"], r["team"]), float("nan")), axis=1
        )
    return merged


def process_season(season, input_dir, out_dir):
    print(f"Processing season {season}")
    games = load_games_for_season(input_dir, season)
    rows = build_team_game_rows(games, season)

    full_season = aggregate_team_season(rows)
    full_season = add_opponent_strength_features(rows, full_season)
    full_season = add_rolling_features_to_aggregates(rows, full_season)
    
    regular_rows = rows[~rows["is_postseason"]].copy()
    pre_tournament = aggregate_team_season(regular_rows)
    pre_tournament = add_opponent_strength_features(regular_rows, pre_tournament)
    pre_tournament = add_rolling_features_to_aggregates(regular_rows, pre_tournament)

    ensure_dir(out_dir)

    seeds_df = extract_seeds_from_games(games, season, out_dir)

    def _merge_seeds(df, seeds):
        if seeds.empty:
            return df
        seed_cols = seeds[["team_id", "seed"]].rename(columns={"seed": "seed"})
        merged = df.merge(seed_cols.drop_duplicates(subset=["team_id"]), on="team_id", how="left")
        # Fall back to name match for rows where team_id didn't match
        unmatched = merged["seed"].isna() & seeds["team"].notna().any()
        if unmatched.any():
            seed_by_name = seeds[["team", "seed"]].drop_duplicates(subset=["team"])
            for idx in merged[merged["seed"].isna()].index:
                tname = str(merged.at[idx, "team"]).strip()
                match = seed_by_name[seed_by_name["team"] == tname]
                if not match.empty:
                    merged.at[idx, "seed"] = match.iloc[0]["seed"]
        return merged

    full_season = _merge_seeds(full_season, seeds_df)
    pre_tournament = _merge_seeds(pre_tournament, seeds_df)

    full_path = Path(out_dir) / f"season_aggregates_{season}.csv"
    snap_path = Path(out_dir) / f"tournament_team_features_{season}.csv"
    full_season.to_csv(full_path, index=False)
    pre_tournament.to_csv(snap_path, index=False)
    print(f"Wrote {full_path} ({len(full_season)} teams)")
    print(f"Wrote {snap_path} ({len(pre_tournament)} teams)")
    return full_season, pre_tournament


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="data/processed", help="directory with games_*.csv")
    p.add_argument("--out", default="data/processed/features", help="output directory for feature CSVs")
    p.add_argument("--seasons", nargs="*", type=int, help="explicit seasons to process (e.g. 2021 2022)")
    p.add_argument("--seed_map", default=None, help="optional CSV with seed mappings")
    p.add_argument("--conf_map", default=None, help="optional CSV with conference mappings")
    p.add_argument("--d1_list", default=None, help="optional file listing Division-I teams")
    p.add_argument("--no_auto_enrich", action="store_true", help="disable automatic merge of conference/momentum enrichment files")
    args = p.parse_args()

    ensure_dir(args.out)

    if args.seasons:
        seasons = sorted(args.seasons)
    else:
        files = glob.glob(os.path.join(args.input, "games_*.csv"))
        seasons = sorted(int(Path(f).stem.split("_")[1]) for f in files)

    all_teams = []
    all_tournament_teams = []
    for season in seasons:
        try:
            full_df, snap_df = process_season(season, args.input, args.out)
            all_teams.append(full_df)
            all_tournament_teams.append(snap_df)
        except FileNotFoundError as exc:
            print(f"Skipping season {season}: {exc}")

    if not all_teams:
        return

    combined = pd.concat(all_teams, ignore_index=True, sort=False)
    tournament_combined = pd.concat(all_tournament_teams, ignore_index=True, sort=False)

    # Combine all per-season seed files into seeds_all.csv
    seed_files = sorted(f for f in Path(args.out).glob("seeds_*.csv") if f.name != "seeds_all.csv")
    if seed_files:
        all_seeds = pd.concat([pd.read_csv(f) for f in seed_files], ignore_index=True, sort=False)
        seeds_all_path = Path(args.out) / "seeds_all.csv"
        all_seeds.to_csv(seeds_all_path, index=False)
        print(f"Wrote {seeds_all_path} ({len(all_seeds)} rows)")

    for df_name, df in [("teams", combined), ("tournament_teams", tournament_combined)]:
        enriched = merge_optional_mapping(df, args.seed_map, "seed")
        enriched = merge_optional_mapping(enriched, args.conf_map, "conference")
        if not args.no_auto_enrich:
            enriched = apply_auto_enrichment(enriched, args.out)
        enriched = apply_d1_tags(enriched, args.d1_list)
        out_path = Path(args.out) / f"{df_name}.csv"
        enriched.to_csv(out_path, index=False)
        print(f"Wrote {out_path} ({len(enriched)} rows)")


if __name__ == "__main__":
    main()
