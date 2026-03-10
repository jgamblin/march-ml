"""Starter scraper using cbbpy to fetch the last five completed seasons.

By default this script fetches game metadata and boxscores only (no play-by-play)
to reduce run-time and storage. Use `--pbp` to enable play-by-play scraping.

Writes processed CSVs to `data/processed/` and caches raw outputs under `data/raw/`.

Usage:
    python scripts/scrape_with_cbbpy.py --out data/processed --raw data/raw
    python scripts/scrape_with_cbbpy.py --seasons 2023 --pbp
"""
import argparse
import os
import sys
from datetime import datetime

import pandas as pd

try:
    import cbbpy.mens_scraper as s
except Exception as e:
    print("cbbpy is required. Install with `pip install cbbpy==2.1.3`", file=sys.stderr)
    raise


def seasons_to_fetch(last_n=5):
    current_year = datetime.utcnow().year
    # NCAA season is named by the calendar year it ends (e.g., 2025 for 2024-25 season)
    last_completed = current_year - 1
    return list(range(last_completed - last_n + 1, last_completed + 1))


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def fetch_season(season, out_dir, raw_dir, fetch_box=False, fetch_pbp=False):
    print(f"Fetching season {season}...")
    # box scores are not used in the feature pipeline; skip by default to save time
    games_tuple = s.get_games_season(season, info=True, box=fetch_box, pbp=fetch_pbp)
    # cbbpy returns tuple: (games_info_df, boxscores_df, pbp_df)
    if not isinstance(games_tuple, (list, tuple)) or len(games_tuple) < 1:
        raise RuntimeError("Unexpected return from cbbpy.get_games_season")

    games_df = games_tuple[0]
    box_df = games_tuple[1] if len(games_tuple) > 1 else None
    pbp_df = games_tuple[2] if len(games_tuple) > 2 else None

    # Save processed
    games_path = os.path.join(out_dir, f"games_{season}.csv")
    games_df.to_csv(games_path, index=False)
    print(f"Wrote {games_path} ({len(games_df)} rows)")

    if box_df is not None:
        box_path = os.path.join(out_dir, f"boxscores_{season}.csv")
        box_df.to_csv(box_path, index=False)
        print(f"Wrote {box_path} ({len(box_df)} rows)")

    if fetch_pbp and pbp_df is not None and not pbp_df.empty:
        pbp_path = os.path.join(out_dir, f"pbp_{season}.csv")
        pbp_df.to_csv(pbp_path, index=False)
        print(f"Wrote {pbp_path} ({len(pbp_df)} rows)")

    # Minimal raw caching: pickle the DataFrames
    raw_games_path = os.path.join(raw_dir, f"games_{season}.pkl")
    games_df.to_pickle(raw_games_path)
    print(f"Cached raw games to {raw_games_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="data/processed", help="processed output dir")
    p.add_argument("--raw", default="data/raw", help="raw cache dir")
    p.add_argument("--seasons", nargs="*", type=int, help="explicit seasons to fetch (e.g. 2021 2022)")
    p.add_argument("--box", action="store_true", help="fetch box scores (slow, not needed for feature pipeline)")
    p.add_argument("--pbp", action="store_true", help="fetch play-by-play in addition to game info (very slow)")
    args = p.parse_args()

    ensure_dir(args.out)
    ensure_dir(args.raw)

    seasons = args.seasons or seasons_to_fetch(5)
    print(f"Seasons: {seasons}")

    for season in seasons:
        try:
            fetch_season(season, args.out, args.raw, fetch_box=args.box, fetch_pbp=args.pbp)
        except Exception as e:
            print(f"Error fetching season {season}: {e}", file=sys.stderr)

if __name__ == "__main__":
    main()
