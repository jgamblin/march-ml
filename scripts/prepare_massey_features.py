"""Extract pre-tournament POM (KenPom proxy) rankings from MMasseyOrdinals.csv.

Reads the Kaggle March Mania MMasseyOrdinals.csv (122 MB, gitignored) and writes
small per-season feature files:

    data/processed/features/pom_{season}.csv

Each file has columns: team, pom_rank
where pom_rank is the KenPom ordinal rank on the last available day before
Selection Sunday (day ≤ 133 in Kaggle's day numbering).

Usage:
    python scripts/prepare_massey_features.py
    python scripts/prepare_massey_features.py --seasons 2024 2025 2026
    python scripts/prepare_massey_features.py --systems POM NET TRK
"""

import argparse
import difflib
import logging
import re
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

KAGGLE_DIR = Path("data/kaggle")
OUT_DIR = Path("data/processed/features")

# Kaggle day 133 ≈ Selection Sunday (day 1 = Nov 1, day 133 = mid-March).
# Using max available day ≤ 133 gives the pre-tournament snapshot.
PRE_TOURNEY_DAY = 133

# Mapping from system name to output column name
SYSTEM_COLUMN = {
    "POM": "pom_rank",
    "NET": "net_rank_kaggle",
    "TRK": "trk_rank",
}


def build_spelling_lookup(spellings_df: pd.DataFrame) -> dict[str, int]:
    """Return {normalized_spelling: TeamID} from MTeamSpellings.csv."""
    lookup = {}
    for _, row in spellings_df.iterrows():
        key = str(row["TeamNameSpelling"]).strip().lower()
        lookup[key] = int(row["TeamID"])
    return lookup


def match_team_name(full_name: str, lookup: dict[str, int]) -> int | None:
    """Map a full team name (e.g. 'Duke Blue Devils') to a Kaggle TeamID.

    Strategy:
    1. Try progressively shorter prefixes of the name (drop words from the right).
       'Duke Blue Devils' → 'duke blue devils' → 'duke blue' → 'duke'
    2. If no prefix matches, try difflib fuzzy match against the full spelling list.
    """
    words = re.sub(r"[^a-z0-9 ]", "", full_name.lower()).split()
    for n in range(len(words), 0, -1):
        candidate = " ".join(words[:n])
        if candidate in lookup:
            return lookup[candidate]

    # Fuzzy fallback (slower, for edge cases)
    best = difflib.get_close_matches(full_name.lower(), list(lookup.keys()), n=1, cutoff=0.75)
    if best:
        return lookup[best[0]]
    return None


def build_our_team_to_kaggle_id(our_teams: list[str], lookup: dict[str, int]) -> dict[str, int]:
    """Return {our_team_name: kaggle_team_id} with coverage stats."""
    mapping: dict[str, int] = {}
    unmatched = []
    for name in our_teams:
        tid = match_team_name(name, lookup)
        if tid is not None:
            mapping[name] = tid
        else:
            unmatched.append(name)
    pct = 100 * len(mapping) / max(len(our_teams), 1)
    logger.info("Team name matching: %d/%d matched (%.1f%%)", len(mapping), len(our_teams), pct)
    if unmatched:
        logger.warning("Unmatched teams (%d): %s", len(unmatched), unmatched[:20])
    return mapping


def extract_pre_tourney_ranks(
    ordinals: pd.DataFrame,
    system: str,
    season: int,
    pre_tourney_day: int = PRE_TOURNEY_DAY,
) -> pd.DataFrame:
    """Return DataFrame of {TeamID, OrdinalRank} for the latest day ≤ pre_tourney_day."""
    sub = ordinals[(ordinals["Season"] == season) & (ordinals["SystemName"] == system)]
    if sub.empty:
        return pd.DataFrame(columns=["TeamID", "OrdinalRank"])
    available_days = sub[sub["RankingDayNum"] <= pre_tourney_day]["RankingDayNum"]
    if available_days.empty:
        # If nothing before day 133, use the earliest available
        day = sub["RankingDayNum"].min()
    else:
        day = available_days.max()
    snapshot = sub[sub["RankingDayNum"] == day][["TeamID", "OrdinalRank"]].copy()
    return snapshot


def write_season_features(
    season: int,
    snapshot: pd.DataFrame,
    our_teams: list[str],
    kaggle_id_map: dict[str, int],
    col_name: str,
    out_dir: Path,
) -> Path:
    """Write pom_{season}.csv (or net_rank_kaggle_{season}.csv, etc.)."""
    # Invert kaggle_id_map: {kaggle_id: our_team_name}
    id_to_our = {v: k for k, v in kaggle_id_map.items()}

    rows = []
    for _, r in snapshot.iterrows():
        our_name = id_to_our.get(int(r["TeamID"]))
        if our_name:
            rows.append({"team": our_name, col_name: int(r["OrdinalRank"])})

    df = pd.DataFrame(rows, columns=["team", col_name])

    # Derive output filename from column name
    prefix = col_name.replace("_rank", "").replace("_kaggle", "")
    out_path = out_dir / f"{prefix}_{season}.csv"
    df.to_csv(out_path, index=False)
    logger.info("  Wrote %s (%d teams)", out_path, len(df))
    return out_path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--massey", default=str(KAGGLE_DIR / "MMasseyOrdinals.csv"))
    p.add_argument("--teams", default=str(KAGGLE_DIR / "MTeams.csv"))
    p.add_argument("--spellings", default=str(KAGGLE_DIR / "MTeamSpellings.csv"))
    p.add_argument("--features", default="data/processed/features/tournament_teams.csv")
    p.add_argument("--out", default=str(OUT_DIR))
    p.add_argument("--seasons", nargs="*", type=int, default=None,
                   help="seasons to process (default: all available in Massey file)")
    p.add_argument("--systems", nargs="*", default=["POM"],
                   help="systems to extract, e.g. POM NET TRK (default: POM)")
    args = p.parse_args()

    massey_path = Path(args.massey)
    if not massey_path.exists():
        logger.error("MMasseyOrdinals.csv not found at %s", massey_path)
        logger.error("Download from: kaggle.com/competitions/march-machine-learning-mania-2026/data")
        raise SystemExit(1)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading team spellings from %s ...", args.spellings)
    spellings_df = pd.read_csv(args.spellings)
    lookup = build_spelling_lookup(spellings_df)

    logger.info("Loading our team features from %s ...", args.features)
    our_feats = pd.read_csv(args.features)
    # Restrict to D1 teams only — non-D1 teams share prefixes with D1 schools
    # (e.g. "Florida Tech Panthers" collides with "Florida Gators" → same Kaggle TeamID)
    if "is_d1" in our_feats.columns:
        d1_mask = our_feats["is_d1"].astype(str).str.lower().isin(["true", "1", "yes"])
        all_our_teams = sorted(our_feats[d1_mask]["team"].unique())
        logger.info("Restricted to %d D1 teams for matching", len(all_our_teams))
    else:
        all_our_teams = sorted(our_feats["team"].unique())
    kaggle_id_map = build_our_team_to_kaggle_id(all_our_teams, lookup)

    logger.info("Loading MMasseyOrdinals.csv (%s) — this takes ~10s ...", massey_path)
    ordinals = pd.read_csv(massey_path)
    logger.info("Loaded %d rows, %d systems, seasons %d–%d",
                len(ordinals), ordinals["SystemName"].nunique(),
                ordinals["Season"].min(), ordinals["Season"].max())

    seasons = args.seasons or sorted(ordinals["Season"].unique().tolist())

    for system in args.systems:
        col_name = SYSTEM_COLUMN.get(system, f"{system.lower()}_rank")
        logger.info("Extracting %s (%s) for %d seasons ...", system, col_name, len(seasons))
        for season in seasons:
            # Only process seasons we have our own feature data for
            season_teams = our_feats[our_feats["season"] == season]["team"].unique().tolist()
            if not season_teams:
                logger.debug("  Skipping %d — no our-team data", season)
                continue
            snapshot = extract_pre_tourney_ranks(ordinals, system, season)
            if snapshot.empty:
                logger.warning("  %d: no %s data available", season, system)
                continue
            write_season_features(season, snapshot, season_teams, kaggle_id_map, col_name, out_dir)

    logger.info("Done. Run 'python scripts/run_pipeline.py --mode features' to merge into tournament_teams.csv")


if __name__ == "__main__":
    main()
