#!/usr/bin/env python3
"""
Scrape NCAA NET rankings from ncaa.com and write net_{season}.csv.

The NCAA NET (Nitty Gritty Efficiency Tool) is the official ranking used by
the selection committee since the 2018-19 season. It incorporates win/loss
record, game results, strength of schedule, game location (home/road/neutral),
and scoring efficiency. Unlike AP poll (top-25 only), NET ranks all ~364 D-I
teams, making it a comprehensive quality signal for tournament modeling.

Output columns:
    season       — integer season year
    team         — normalized cbbpy team name
    source_name  — raw ncaa.com name (for debugging)
    net_rank     — NET rank (1 = best)
    quad1_wins   — wins vs Quad 1 opponents
    quad1_losses — losses vs Quad 1 opponents
    quad2_wins   — wins vs Quad 2 opponents
    quad2_losses — losses vs Quad 2 opponents

LIMITATION: ncaa.com only serves current-season data. Historical NET rankings
for past seasons (2019-2025) are not publicly available via simple HTTP.
This scraper collects data for CI weekly runs; LOSO improvement will accrue as
seasons accumulate. Once 2+ seasons have NET data, expect a meaningful signal
since NET correlates ~0.85+ with seed selection.

Usage:
    python scripts/fetch_net_rankings.py
    python scripts/fetch_net_rankings.py --season 2026 --out data/processed/features/net_2026.csv
    python scripts/fetch_net_rankings.py --dry-run
"""

import argparse
import sys
import urllib.request
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup

NCAA_NET_URL = "https://www.ncaa.com/rankings/basketball-men/d1/ncaa-mens-basketball-net-rankings"

# Map ncaa.com abbreviated names → cbbpy full team names.
# ncaa.com uses "Iowa St." instead of "Iowa State Cyclones" etc.
# Add entries here as mismatches are discovered.
_NAME_MAP = {
    # Drop the trailing St. → State + add nickname
    "Iowa St.": "Iowa State Cyclones",
    "Michigan St.": "Michigan State Spartans",
    "Ohio St.": "Ohio State Buckeyes",
    "Utah St.": "Utah State Aggies",
    "Kansas St.": "Kansas State Wildcats",
    "Arizona St.": "Arizona State Sun Devils",
    "Oklahoma St.": "Oklahoma State Cowboys",
    "Colorado St.": "Colorado State Rams",
    "Florida St.": "Florida State Seminoles",
    "Mississippi St.": "Mississippi State Bulldogs",
    "North Dakota St.": "North Dakota State Bison",
    "Wichita St.": "Wichita State Shockers",
    "San Diego St.": "San Diego State Aztecs",
    "Boise St.": "Boise State Broncos",
    "Illinois St.": "Illinois State Redbirds",
    "Murray St.": "Murray State Racers",
    # Special disambiguation needed
    "Saint Mary's (CA)": "Saint Mary's Gaels",
    "St. John's (NY)": "St. John's Red Storm",
    "St. Thomas (MN)": "St. Thomas-Minnesota Tommies",
    "Miami (FL)": "Miami Hurricanes",
    "Miami (OH)": "Miami (OH) RedHawks",
    # Abbreviations
    "UNI": "Northern Iowa Panthers",
    "UCF": "UCF Knights",
    "NC State": "NC State Wolfpack",
    "LSU": "LSU Tigers",
    "UNLV": "UNLV Rebels",
    "UNCW": "UNC Wilmington Seahawks",
    "VCU": "VCU Rams",
    "SMU": "SMU Mustangs",
    "BYU": "BYU Cougars",
    "TCU": "TCU Horned Frogs",
    "UConn": "UConn Huskies",
    "SFA": "SFA Lumberjacks",
    "UAB": "UAB Blazers",
    "UIC": "UIC Flames",
    # Partial abbreviations
    "South Fla.": "South Florida Bulls",
    "Southern Ill.": "Southern Illinois Salukis",
    "Fla. Atlantic": "Florida Atlantic Owls",
    # No nickname included on ncaa.com → add nickname
    "Duke": "Duke Blue Devils",
    "Michigan": "Michigan Wolverines",
    "Arizona": "Arizona Wildcats",
    "Florida": "Florida Gators",
    "Illinois": "Illinois Fighting Illini",
    "Gonzaga": "Gonzaga Bulldogs",
    "Houston": "Houston Cougars",
    "Purdue": "Purdue Boilermakers",
    "Nebraska": "Nebraska Cornhuskers",
    "Virginia": "Virginia Cavaliers",
    "Louisville": "Louisville Cardinals",
    "Alabama": "Alabama Crimson Tide",
    "Arkansas": "Arkansas Razorbacks",
    "Kansas": "Kansas Jayhawks",
    "Tennessee": "Tennessee Volunteers",
    "Wisconsin": "Wisconsin Badgers",
    "Kentucky": "Kentucky Wildcats",
    "Georgia": "Georgia Bulldogs",
    "UCLA": "UCLA Bruins",
    "Clemson": "Clemson Tigers",
    "Auburn": "Auburn Tigers",
    "Indiana": "Indiana Hoosiers",
    "Texas": "Texas Longhorns",
    "Missouri": "Missouri Tigers",
    "Oklahoma": "Oklahoma Sooners",
    "Baylor": "Baylor Bears",
    "Washington": "Washington Huskies",
    "Minnesota": "Minnesota Golden Gophers",
    "Oregon": "Oregon Ducks",
    "Stanford": "Stanford Cardinal",
    "Northwestern": "Northwestern Wildcats",
    "California": "California Golden Bears",
    "Nevada": "Nevada Wolf Pack",
    "Colorado": "Colorado Buffaloes",
    "Butler": "Butler Bulldogs",
    "Syracuse": "Syracuse Orange",
    "Providence": "Providence Friars",
    "Pittsburgh": "Pittsburgh Panthers",
    "Georgetown": "Georgetown Hoyas",
    "Marquette": "Marquette Golden Eagles",
    "Xavier": "Xavier Musketeers",
    "Villanova": "Villanova Wildcats",
    "Dayton": "Dayton Flyers",
    "Davidson": "Davidson Wildcats",
    "Liberty": "Liberty Flames",
    "Hawaii": "Hawaii Rainbow Warriors",
    "Bradley": "Bradley Braves",
    "Pacific": "Pacific Tigers",
    "Hofstra": "Hofstra Pride",
    "Belmont": "Belmont Bruins",
    "Yale": "Yale Bulldogs",
    "Akron": "Akron Zips",
    "Tulsa": "Tulsa Golden Hurricane",
    "Cincinnati": "Cincinnati Bearcats",
    "Ole Miss": "Ole Miss Rebels",
    "Texas A&M": "Texas A&M Aggies",
    "North Carolina": "North Carolina Tar Heels",
    "Iowa": "Iowa Hawkeyes",
    "South Carolina": "South Carolina Gamecocks",
    "New Mexico": "New Mexico Lobos",
    "West Virginia": "West Virginia Mountaineers",
    "Virginia Tech": "Virginia Tech Hokies",
    "Grand Canyon": "Grand Canyon Lopes",
    "Vanderbilt": "Vanderbilt Commodores",
    "Tennessee State": "Tennessee State Tigers",
    "George Mason": "George Mason Patriots",
    "George Washington": "George Washington Revolutionaries",
    "Mississippi": "Ole Miss Rebels",
    "Minnesota": "Minnesota Golden Gophers",
    "Saint Louis": "Saint Louis Billikens",
    "Seton Hall": "Seton Hall Pirates",
    "McNeese": "McNeese Cowboys",
    "Sam Houston": "Sam Houston Bearkats",
    "Santa Clara": "Santa Clara Broncos",
    "High Point": "High Point Panthers",
    "Utah Valley": "Utah Valley Wolverines",
    "Winthrop": "Winthrop Eagles",
    "California Baptist": "California Baptist Lancers",
    "Seattle U": "Seattle U Redhawks",
    "Washington St.": "Washington State Cougars",
    "Wake Forest": "Wake Forest Demon Deacons",
    "Notre Dame": "Notre Dame Fighting Irish",
    "DePaul": "DePaul Blue Demons",
    "Army": "Army Black Knights",
    "Navy": "Navy Midshipmen",
    "Creighton": "Creighton Bluejays",
    "Memphis": "Memphis Tigers",
    "Rutgers": "Rutgers Scarlet Knights",
    "Utah": "Utah Utes",
    "Maryland": "Maryland Terrapins",
    "Penn St.": "Penn State Nittany Lions",
    "Florida A&M": "Florida A&M Rattlers",
    "Southern California": "USC Trojans",
    "Ohio": "Ohio Bobcats",
    "ETSU": "East Tennessee State Buccaneers",
    "Tennessee St.": "Tennessee State Tigers",
    "Boston U.": "Boston University Terriers",
    "UNI": "Northern Iowa Panthers",
    "William & Mary": "William & Mary Tribe",
    "North Texas": "North Texas Mean Green",
    "Wyoming": "Wyoming Cowboys",
    "San Francisco": "San Francisco Dons",
    "Portland St.": "Portland State Vikings",
    "Montana St.": "Montana State Bobcats",
    "Idaho": "Idaho Vandals",
    "UC Irvine": "UC Irvine Anteaters",
    "UC San Diego": "UC San Diego Tritons",
    "UC Santa Barbara": "UC Santa Barbara Gauchos",
    "Cornell": "Cornell Big Red",
    "Troy": "Troy Trojans",
    "Duquesne": "Duquesne Dukes",
    "Saint Joseph's": "Saint Joseph's Hawks",
    "Wright St.": "Wright State Raiders",
    "Hofstra": "Hofstra Pride",
}


def _normalize_name(raw: str) -> str:
    """Apply name mapping; fall back to raw string if no mapping found."""
    name = raw.strip()
    return _NAME_MAP.get(name, name)


def _parse_wl(cell: str) -> tuple[int, int]:
    """Parse a 'W-L' cell like '15-2' into (wins, losses). Returns (0,0) on failure."""
    cell = cell.strip()
    if "-" in cell:
        parts = cell.split("-", 1)
        try:
            return int(parts[0]), int(parts[1])
        except ValueError:
            pass
    return 0, 0


def fetch_html(url: str = NCAA_NET_URL) -> str:
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                          "AppleWebKit/537.36 (KHTML, like Gecko) "
                          "Chrome/120.0.0.0 Safari/537.36"
        },
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        raw = resp.read()
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return raw.decode("latin-1", errors="replace")


def parse_net_rankings(html: str, season: int) -> pd.DataFrame:
    """
    Parse the ncaa.com NET rankings table.

    The page is server-side rendered Drupal HTML with a <table> containing 13
    columns: Rank, School, Record, Conf, Road, Neutral, Home, Non-Div I,
    Prev, Quad 1, Quad 2, Quad 3, Quad 4.
    """
    soup = BeautifulSoup(html, "html.parser")
    table = soup.find("table")
    if table is None:
        raise ValueError("No <table> found on the ncaa.com NET rankings page. "
                         "Page structure may have changed.")

    rows = []
    for tr in table.find_all("tr"):
        cells = [td.get_text(strip=True) for td in tr.find_all("td")]
        if len(cells) < 13:
            continue
        rank_str = cells[0]
        try:
            rank = int(rank_str)
        except ValueError:
            continue  # header or non-data row

        source_name = cells[1].strip()
        q1_wins, q1_losses = _parse_wl(cells[9])
        q2_wins, q2_losses = _parse_wl(cells[10])

        rows.append({
            "season": season,
            "team": _normalize_name(source_name),
            "source_name": source_name,
            "net_rank": rank,
            "quad1_wins": q1_wins,
            "quad1_losses": q1_losses,
            "quad2_wins": q2_wins,
            "quad2_losses": q2_losses,
        })

    if not rows:
        raise ValueError("Parsed 0 rows from the NET rankings table. "
                         "Check that the HTML contains the expected table structure.")

    df = pd.DataFrame(rows)
    df = df.sort_values("net_rank").reset_index(drop=True)
    return df


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--season", type=int, default=2026,
                   help="Season year to label the output (default: 2026)")
    p.add_argument("--out", default=None,
                   help="Output CSV path (default: data/processed/features/net_{season}.csv)")
    p.add_argument("--url", default=NCAA_NET_URL,
                   help="Override source URL (e.g. for Wayback Machine snapshots)")
    p.add_argument("--dry-run", action="store_true",
                   help="Print parsed data without writing to disk")
    args = p.parse_args()

    out_path = args.out or f"data/processed/features/net_{args.season}.csv"

    print(f"Fetching NET rankings for season {args.season} from:\n  {args.url}", file=sys.stderr)
    try:
        html = fetch_html(args.url)
    except Exception as exc:
        print(f"ERROR: Failed to fetch {args.url}: {exc}", file=sys.stderr)
        sys.exit(1)

    try:
        df = parse_net_rankings(html, args.season)
    except Exception as exc:
        print(f"ERROR: Failed to parse NET rankings: {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"Parsed {len(df)} teams. Top 10:", file=sys.stderr)
    print(df.head(10).to_string(index=False), file=sys.stderr)

    if args.dry_run:
        print("\n[dry-run] Not writing to disk.", file=sys.stderr)
        return

    out_dir = Path(out_path).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Wrote {out_path} ({len(df)} teams)", file=sys.stderr)


if __name__ == "__main__":
    main()
