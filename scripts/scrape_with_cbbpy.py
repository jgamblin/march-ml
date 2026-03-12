"""Starter scraper using cbbpy to fetch the last five completed seasons.

By default this script fetches game metadata and boxscores only (no play-by-play)
to reduce run-time and storage. Use `--pbp` to enable play-by-play scraping.

For the current season, use `--since YYYY-MM-DD` to fetch only games from that date
forward and merge them into the existing CSV. This dramatically speeds up daily CI runs:
instead of re-scraping ~3,500 regular-season games, only new tournament games are fetched.

    # Normal daily run (fetch only new games since Selection Sunday)
    python scripts/scrape_with_cbbpy.py --since 2026-03-15

    # Full historical scrape (one-time)
    python scripts/scrape_with_cbbpy.py --historical

Writes processed CSVs to `data/processed/` and caches raw outputs under `data/raw/`.
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
    # NCAA season is named by the calendar year it ends (e.g., 2025 for 2024-25 season).
    # Include the current in-progress season so it gets scraped on full runs.
    last_completed = current_year - 1
    seasons = list(range(last_completed - last_n + 1, last_completed + 1))
    seasons.append(current_year)  # add current season (e.g., 2026)
    return seasons


# 2020 had no NCAA tournament (COVID cancellation). Skip it for feature/training purposes.
_NO_TOURNAMENT_YEARS = {2020}


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


def fetch_season_since(season, since_date, out_dir, raw_dir, fetch_box=False, fetch_pbp=False):
    """Incremental fetch: only pull games from `since_date` through today, then
    merge with the existing season CSV. Much faster than a full re-scrape when
    the regular season is already cached and only new tournament games are needed.
    """
    today = datetime.utcnow().strftime("%Y-%m-%d")
    if since_date > today:
        print(f"Season {season}: since_date {since_date} is in the future, skipping.")
        return

    games_path = os.path.join(out_dir, f"games_{season}.csv")
    if not os.path.exists(games_path):
        # Cold start: no base file exists — fall back to full season scrape so we don't
        # silently drop all the regular-season games from the training data.
        print(f"  No existing games_{season}.csv; doing full season scrape as cold-start fallback...")
        fetch_season(season, out_dir, raw_dir, fetch_box=fetch_box, fetch_pbp=fetch_pbp)
        return

    print(f"Fetching season {season} games since {since_date} (incremental)...")
    games_tuple = s.get_games_range(since_date, today, info=True, box=fetch_box, pbp=fetch_pbp)
    if not isinstance(games_tuple, (list, tuple)) or len(games_tuple) < 1:
        raise RuntimeError("Unexpected return from cbbpy.get_games_range")

    new_df = games_tuple[0]
    print(f"  Fetched {len(new_df)} new rows from {since_date} to {today}")

    existing = pd.read_csv(games_path, low_memory=False)
    # Deduplicate by game_id if available, otherwise by all columns
    id_col = "game_id" if "game_id" in existing.columns else None
    combined = pd.concat([existing, new_df], ignore_index=True)
    if id_col:
        combined = combined.drop_duplicates(subset=[id_col], keep="last")
    else:
        combined = combined.drop_duplicates(keep="last")
    print(f"  Merged: {len(existing)} existing + {len(new_df)} new = {len(combined)} total rows")
    combined.to_csv(games_path, index=False)
    games_df = combined

    # Update raw cache
    raw_games_path = os.path.join(raw_dir, f"games_{season}.pkl")
    games_df.to_pickle(raw_games_path)
    print(f"  Updated raw cache: {raw_games_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="data/processed", help="processed output dir")
    p.add_argument("--raw", default="data/raw", help="raw cache dir")
    p.add_argument("--seasons", nargs="*", type=int, help="explicit seasons to fetch (e.g. 2021 2022)")
    p.add_argument("--historical", action="store_true",
                   help="Also fetch 2015–2019 historical seasons (appended before recent seasons). "
                        "2020 is skipped automatically (no tournament due to COVID).")
    p.add_argument("--since", default=None, metavar="YYYY-MM-DD",
                   help="Only fetch games from this date forward for the current season and merge "
                        "with the existing CSV. Mutually exclusive with --lookback-days.")
    p.add_argument("--lookback-days", type=int, default=None, metavar="N",
                   help="Incremental scrape: fetch the last N days for the current season and merge "
                        "with the existing CSV. 14 days covers conf tournaments + early NCAA rounds "
                        "without re-scraping the full regular season.")
    p.add_argument("--box", action="store_true", help="fetch box scores (slow, not needed for feature pipeline)")
    p.add_argument("--pbp", action="store_true", help="fetch play-by-play in addition to game info (very slow)")
    args = p.parse_args()

    ensure_dir(args.out)
    ensure_dir(args.raw)

    # Resolve --lookback-days to a since date
    since = args.since
    if args.lookback_days and not since:
        from datetime import timedelta
        since = (datetime.utcnow() - timedelta(days=args.lookback_days)).strftime("%Y-%m-%d")

    if since and not args.seasons and not args.historical:
        # Incremental mode: only update the current season from the resolved since date
        current_year = datetime.utcnow().year
        current_season = current_year  # season named by end year (2025-26 → 2026)
        print(f"Incremental mode: fetching season {current_season} games since {since}")
        try:
            fetch_season_since(current_season, since, args.out, args.raw,
                               fetch_box=args.box, fetch_pbp=args.pbp)
        except Exception as e:
            # cbbpy sometimes raises KeyError('game_day') or similar when scraped
            # data is missing expected columns (e.g. scheduled/future games).
            # Fall back to a full season scrape so CI always produces a valid CSV.
            print(f"Warning: incremental fetch failed ({e}); falling back to full season scrape...",
                  file=sys.stderr)
            try:
                fetch_season(current_season, args.out, args.raw,
                             fetch_box=args.box, fetch_pbp=args.pbp)
            except Exception as e2:
                print(f"Error in full-season fallback for {current_season}: {e2}", file=sys.stderr)
        return

    if args.seasons:
        seasons = args.seasons
    else:
        recent = seasons_to_fetch(5)
        if args.historical:
            historical = [y for y in range(2015, 2020) if y not in _NO_TOURNAMENT_YEARS]
            # Merge: historical first, then recent (deduped, preserving order)
            seen = set(historical)
            for y in recent:
                if y not in seen:
                    historical.append(y)
                    seen.add(y)
            seasons = historical
        else:
            seasons = recent
    print(f"Seasons: {seasons}")

    for season in seasons:
        try:
            fetch_season(season, args.out, args.raw, fetch_box=args.box, fetch_pbp=args.pbp)
        except Exception as e:
            print(f"Error fetching season {season}: {e}", file=sys.stderr)

if __name__ == "__main__":
    main()
