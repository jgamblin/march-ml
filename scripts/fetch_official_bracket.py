"""
fetch_official_bracket.py
=========================
Fetches the official NCAA tournament bracket from ESPN's public API and
writes it to a CSV file in a format compatible with simulate_bracket.py.

Exit codes:
  0  — bracket found and written successfully
  1  — bracket not yet available (teams listed as "TBD" or data absent)
  2  — network/parse error (check logs)

Usage:
  python scripts/fetch_official_bracket.py --year 2026 --out data/brackets/official_2026.csv
"""

import argparse
import csv
import json
import sys
from pathlib import Path

try:
    import requests
except ImportError:
    print("ERROR: 'requests' not installed. Run: pip install requests", file=sys.stderr)
    sys.exit(2)

# ---------------------------------------------------------------------------
# ESPN public API endpoints (no API key required)
# ---------------------------------------------------------------------------
ESPN_BRACKET_URL = (
    "https://site.api.espn.com/apis/site/v2/sports/basketball/"
    "mens-college-basketball/tournament/bracket"
)
ESPN_TEAMS_URL = (
    "https://sports.core.api.espn.com/v2/sports/basketball/"
    "leagues/mens-college-basketball/seasons/{year}/types/3/teams"
    "?limit=68"
)

REGIONS = ["South", "East", "West", "Midwest"]

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; ncaa-bets-bot/1.0; "
        "+https://github.com)"
    )
}


def fetch_bracket_data(year: int) -> dict | None:
    """
    Hit the ESPN bracket API and return the raw JSON payload, or None if
    the bracket is not yet available (pre-Selection Sunday).
    """
    params = {"seasontype": 3, "year": year}
    try:
        r = requests.get(ESPN_BRACKET_URL, params=params, headers=HEADERS, timeout=15)
        r.raise_for_status()
        data = r.json()
    except requests.RequestException as exc:
        print(f"Network error fetching bracket: {exc}", file=sys.stderr)
        return None
    except json.JSONDecodeError as exc:
        print(f"JSON parse error: {exc}", file=sys.stderr)
        return None

    return data


def extract_teams(data: dict) -> list[dict]:
    """
    Parse ESPN bracket JSON into a flat list of dicts:
      { team_name, seed, region, abbreviation, espn_id }

    Returns an empty list if teams are listed as TBD (bracket unreleased).
    """
    teams = []

    # ESPN bracket JSON structure (as of 2024):
    #   data["bracket"]["rounds"][0]["matchups"] -> first-round matchups
    #   Each matchup has "competitors": [{"team": {...}, "seed": N}, ...]
    bracket = data.get("bracket") or data.get("event", {}).get("bracket", {})
    if not bracket:
        # Try top-level "rounds" key
        bracket = data

    rounds = bracket.get("rounds", [])
    if not rounds:
        print("No 'rounds' key found in bracket response.", file=sys.stderr)
        return []

    # We only need the first round (64 teams, or 68 for play-in)
    first_round = rounds[0]
    matchups = first_round.get("matchups") or first_round.get("games", [])

    seen = set()
    for matchup in matchups:
        region = (matchup.get("region") or {}).get("displayName", "Unknown")
        for comp in matchup.get("competitors", []):
            team = comp.get("team", {})
            name = team.get("displayName") or team.get("location", "")
            abbrev = team.get("abbreviation", "")
            espn_id = str(team.get("id", ""))
            seed = comp.get("seed") or comp.get("curatedRank", {}).get("current", 0)

            if not name or name.upper() in {"TBD", ""}:
                continue  # bracket not yet released
            if name in seen:
                continue
            seen.add(name)
            teams.append(
                {
                    "team_name": name,
                    "abbreviation": abbrev,
                    "seed": int(seed) if seed else 0,
                    "region": region,
                    "espn_id": espn_id,
                }
            )

    return teams


def write_csv(teams: list[dict], out_path: Path) -> None:
    """Write bracket teams to CSV for use with --bracket_file."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["seed", "team_name", "abbreviation", "region", "espn_id"]
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        # Sort by region then seed for readability
        for row in sorted(teams, key=lambda t: (t["region"], t["seed"])):
            writer.writerow(row)
    print(f"Wrote {len(teams)} teams → {out_path}")


def main() -> int:
    p = argparse.ArgumentParser(description="Fetch official NCAA bracket from ESPN")
    p.add_argument("--year", type=int, default=None, help="Tournament year (default: current year)")
    p.add_argument("--out", default=None, help="Output CSV path (default: data/brackets/official_{year}.csv)")
    p.add_argument("--dry_run", action="store_true", help="Print teams but don't write CSV")
    args = p.parse_args()

    import datetime
    year = args.year or datetime.datetime.now().year
    out_path = Path(args.out) if args.out else Path(f"data/brackets/official_{year}.csv")

    print(f"Fetching {year} NCAA tournament bracket from ESPN...")
    data = fetch_bracket_data(year)

    if data is None:
        print("Failed to retrieve bracket data (network/parse error).", file=sys.stderr)
        return 2

    teams = extract_teams(data)

    if not teams:
        print(
            f"Bracket not yet available for {year} "
            "(teams listed as TBD or no matchups found).",
            file=sys.stderr,
        )
        return 1

    print(f"Found {len(teams)} teams in the {year} bracket:")
    for t in sorted(teams, key=lambda x: (x["region"], x["seed"])):
        print(f"  [{t['seed']:2d}] {t['team_name']}  ({t['region']})")

    if args.dry_run:
        print("(dry run — no file written)")
        return 0

    write_csv(teams, out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
