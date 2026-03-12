#!/usr/bin/env python3
"""
Scrape projected seeds from bracketmatrix.com and write seeds_2026.csv.

bracketmatrix.com aggregates bracket projections from 80-100 bracketologists
and publishes a consensus average seed for all projected NCAA tournament teams.
This gives us a much better seed estimate than null for pre-Selection-Sunday
simulation runs.

Usage:
    python scripts/fetch_bracketmatrix_seeds.py
    python scripts/fetch_bracketmatrix_seeds.py --out data/processed/features/seeds_2026.csv
    python scripts/fetch_bracketmatrix_seeds.py --season 2026 --dry-run
"""

import argparse
import re
import sys
import urllib.request
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup

URL = "http://www.bracketmatrix.com/"

# Map bracketmatrix team names → cbbpy team names where they differ.
# Add entries here as mismatches are discovered.
_NAME_MAP = {
    "Connecticut": "UConn Huskies",
    "UConn": "UConn Huskies",
    "Michigan": "Michigan Wolverines",
    "Duke": "Duke Blue Devils",
    "Arizona": "Arizona Wildcats",
    "Florida": "Florida Gators",
    "Houston": "Houston Cougars",
    "Michigan State": "Michigan State Spartans",
    "Illinois": "Illinois Fighting Illini",
    "Iowa State": "Iowa State Cyclones",
    "Nebraska": "Nebraska Cornhuskers",
    "Purdue": "Purdue Boilermakers",
    "Gonzaga": "Gonzaga Bulldogs",
    "Alabama": "Alabama Crimson Tide",
    "Virginia": "Virginia Cavaliers",
    "Kansas": "Kansas Jayhawks",
    "Texas Tech": "Texas Tech Red Raiders",
    "Vanderbilt": "Vanderbilt Commodores",
    "Arkansas": "Arkansas Razorbacks",
    "St. John's": "St. John's Red Storm",
    "North Carolina": "North Carolina Tar Heels",
    "Wisconsin": "Wisconsin Badgers",
    "Tennessee": "Tennessee Volunteers",
    "Louisville": "Louisville Cardinals",
    "BYU": "BYU Cougars",
    "Kentucky": "Kentucky Wildcats",
    "St. Mary's (CA)": "Saint Mary's Gaels",
    "Miami (FLA.)": "Miami Hurricanes",
    "Villanova": "Villanova Wildcats",
    "Georgia": "Georgia Bulldogs",
    "UCLA": "UCLA Bruins",
    "Clemson": "Clemson Tigers",
    "Utah State": "Utah State Aggies",
    "Iowa": "Iowa Hawkeyes",
    "TCU": "TCU Horned Frogs",
    "Saint Louis": "Saint Louis Billikens",
    "Ohio State": "Ohio State Buckeyes",
    "Texas A&M": "Texas A&M Aggies",
    "Central Florida": "UCF Knights",
    "North Carolina State": "NC State Wolfpack",
    "Miami (Ohio)": "Miami (OH) RedHawks",
    "Missouri": "Missouri Tigers",
    "Santa Clara": "Santa Clara Broncos",
    "Texas": "Texas Longhorns",
    "SMU": "SMU Mustangs",
    "VCU": "VCU Rams",
    "South Florida": "South Florida Bulls",
    "Akron": "Akron Zips",
    "Yale": "Yale Bulldogs",
    "McNeese State": "McNeese Cowboys",
    "High Point": "High Point Panthers",
    "Northern Iowa": "Northern Iowa Panthers",
    "Utah Valley": "Utah Valley Wolverines",
    "Hofstra": "Hofstra Pride",
    "Sam Houston State": "Sam Houston Bearkats",
    "UC Irvine": "UC Irvine Anteaters",
    "Troy": "Troy Trojans",
    "North Dakota State": "North Dakota State Bison",
    "Wright State": "Wright State Raiders",
    "Tennessee State": "Tennessee State Tigers",
    "Furman": "Furman Paladins",
    "Idaho": "Idaho Vandals",
    "Siena": "Siena Saints",
    "Queens": "Queens Royals",
    "UMBC": "UMBC Retrievers",
    "Long Island": "LIU Sharks",
    "Belmont": "Belmont Bruins",
    "Stephen F. Austin": "SFA Lumberjacks",
    "Liberty": "Liberty Flames",
    "Norfolk State": "Norfolk State Spartans",
    "Montana": "Montana Grizzlies",
    "Boston University": "Boston University Terriers",
    "Bethune-Cookman": "Bethune-Cookman Wildcats",
    "Prairie View A&M": "Prairie View A&M Panthers",
}


def _normalize_name(raw: str) -> str:
    """Apply name mapping then return cleaned string."""
    name = raw.strip()
    return _NAME_MAP.get(name, name)


def fetch_html(url: str = URL) -> str:
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 (compatible; ncaa-bets/1.0)"},
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        raw = resp.read()
    # bracketmatrix uses windows-1252 encoding
    try:
        return raw.decode("windows-1252")
    except Exception:
        return raw.decode("utf-8", errors="replace")


def parse_seeds(html: str) -> pd.DataFrame:
    """
    Parse the bracketmatrix HTML and return a DataFrame with columns:
        team, avg_seed, seed (rounded int), projections (count), source_name
    """
    soup = BeautifulSoup(html, "html.parser")
    rows = []

    # The page is an Excel-exported table.  Each team block looks like:
    #   <td>Duke</td>  <td>ACC</td>  <td>1.00</td>  <td>98</td>
    #   ...individual seeds...
    #   <td>Duke</td>  <td>1</td>
    #
    # Strategy: collect all <td> text values, then look for the pattern:
    #   [team_name, conference, avg_seed_float, n_brackets, ..., team_name, rounded_seed]
    # where avg_seed_float matches ^\d+\.\d+$ and n_brackets is an integer ≥ 1.

    cells = [td.get_text(strip=True) for td in soup.find_all("td")]

    # Known conference names to help distinguish them from team names
    known_conferences = {
        "ACC", "Big Ten", "Big 12", "Big East", "SEC", "Pac-12", "Pac-10",
        "West Coast", "Mountain West", "American", "Atlantic 10", "MAC",
        "Missouri Valley", "Southland", "Big South", "Ivy", "Coastal",
        "CUSA", "Big West", "Sun Belt", "Summit", "Horizon", "Ohio Valley",
        "Southern", "Big Sky", "Metro Atlantic", "Atlantic Sun",
        "America East", "Northeast", "SWAC", "Mid-Eastern", "Patriot",
        "WAC", "MEAC",
    }

    i = 0
    while i < len(cells) - 3:
        cell = cells[i]

        # Look for a float that looks like an average seed (1.00 – 16.99)
        if re.match(r"^\d+\.\d+$", cell):
            avg = float(cell)
            if 1.0 <= avg <= 16.99:
                # Walk backwards to find team name and conference
                # Pattern: cells[i-2] = team_name, cells[i-1] = conference
                # OR: cells[i-1] = team_name (no separate conference cell)
                if i >= 2:
                    conf_candidate = cells[i - 1]
                    team_candidate = cells[i - 2]
                    if conf_candidate in known_conferences and team_candidate:
                        n_proj = cells[i + 1] if i + 1 < len(cells) else ""
                        try:
                            n_proj = int(n_proj)
                        except ValueError:
                            n_proj = 0
                        rows.append({
                            "source_name": team_candidate.strip(),
                            "team": _normalize_name(team_candidate.strip()),
                            "conference": conf_candidate,
                            "avg_seed": avg,
                            "seed": round(avg),
                            "projections": n_proj,
                        })
                        i += 2
                        continue
        i += 1

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # De-duplicate: keep highest-projection-count entry per team
    df = (
        df.sort_values("projections", ascending=False)
        .drop_duplicates(subset=["source_name"])
        .reset_index(drop=True)
    )
    return df.sort_values("avg_seed").reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser(description="Fetch bracketmatrix.com projected seeds")
    parser.add_argument("--out", default="data/processed/features/seeds_2026.csv",
                        help="Output CSV path")
    parser.add_argument("--season", type=int, default=2026,
                        help="Season label written to the CSV (default: 2026)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print results without writing file")
    parser.add_argument("--url", default=URL,
                        help=f"Override source URL (default: {URL})")
    args = parser.parse_args()

    print(f"Fetching {args.url} ...", file=sys.stderr)
    html = fetch_html(args.url)

    print("Parsing seeds ...", file=sys.stderr)
    df = parse_seeds(html)

    if df.empty:
        print("ERROR: no seeds parsed — the page format may have changed.", file=sys.stderr)
        sys.exit(1)

    df.insert(0, "season", args.season)

    print(f"\nParsed {len(df)} teams:\n")
    print(df[["seed", "avg_seed", "team", "conference", "projections"]].to_string(index=False))

    if args.dry_run:
        print("\n[dry-run] Not writing file.", file=sys.stderr)
        return

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Write only the columns that seeds_*.csv normally contains
    out_df = df[["season", "team", "seed"]].copy()
    out_df.to_csv(out_path, index=False)
    print(f"\nWrote {len(out_df)} rows to {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
