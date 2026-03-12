#!/usr/bin/env python3
"""
Fetch BartTorvik T-Rank team efficiency data and write barttorvik_{season}.csv.

BartTorvik (barttorvik.com) publishes adjusted efficiency metrics for all ~365
D-I teams as a plain JSON endpoint — no JavaScript or authentication required.
The data is KenPom-equivalent (adjusted offensive/defensive efficiency, tempo,
SOS) and is available for every season going back to at least 2008.

Output columns:
    season      — integer season year
    team        — normalized cbbpy team name
    source_name — raw BartTorvik name (for debugging / auditing)
    rank        — T-Rank overall ranking (1 = best)
    adjoe       — Adjusted Offensive Efficiency (points per 100 poss, adj for SOS)
    adjde       — Adjusted Defensive Efficiency (points allowed per 100 poss, adj for SOS)
    adj_em      — Net efficiency margin = adjoe - adjde (like KenPom AdjEM)
    barthag     — Pythagorean win probability vs avg D-I team (0–1, higher=better)
    sos_adj     — Adjusted SOS (win probability scale)
    luck        — Luck factor (deviation from expected wins)
    wab         — Wins Above Bubble

JSON field layout (indices into each team array):
    [0]  rank        [1]  team       [2]  conf       [3]  record
    [4]  adjoe       [5]  adjoe_rank [6]  adjde      [7]  adjde_rank
    [8]  barthag     [9]  barthag_rk [10] wins        [11] losses
    [15] sos_adj     [33] luck       [41] wab

Usage:
    # Download all historical seasons at once (first-time setup):
    python scripts/fetch_barttorvik.py --seasons 2019 2020 2021 2022 2023 2024 2025 2026

    # Download just current season (CI / weekly update):
    python scripts/fetch_barttorvik.py --season 2026

    # Dry run: print table, don't write files:
    python scripts/fetch_barttorvik.py --season 2026 --dry-run
"""

import argparse
import sys
import time
import urllib.request
from pathlib import Path

import pandas as pd

# Reuse the same name mapping as the NET scraper — BartTorvik uses the same
# abbreviated institution names (no nicknames).
sys.path.insert(0, str(Path(__file__).parent))
from fetch_net_rankings import _NAME_MAP  # noqa: E402

# A handful of names that appear on BartTorvik but not on ncaa.com's NET page.
_BART_EXTRA: dict[str, str] = {
    "Connecticut": "UConn Huskies",
    "UConn": "UConn Huskies",           # seen in older seasons
    "N.C. State": "NC State Wolfpack",
    "NC St.": "NC State Wolfpack",
    "Ga. Tech": "Georgia Tech Yellow Jackets",
    "Georgia Tech": "Georgia Tech Yellow Jackets",
    "St. John's": "St. John's Red Storm",
    "Miami FL": "Miami Hurricanes",
    "Miami OH": "Miami (OH) RedHawks",
    "East Carolina": "East Carolina Pirates",
    "Southern Miss": "Southern Miss Golden Eagles",
    "Middle Tenn.": "Middle Tennessee Blue Raiders",
    "Western Ky.": "Western Kentucky Hilltoppers",
    "Western Kentucky": "Western Kentucky Hilltoppers",
    "Louisiana Tech": "Louisiana Tech Bulldogs",
    "La. Tech": "Louisiana Tech Bulldogs",
    "Fla. Intl": "FIU Panthers",
    "FIU": "FIU Panthers",
    "UTSA": "UTSA Roadrunners",
    "UTEP": "UTEP Miners",
    "Tex. A&M-CC": "Texas A&M-Corpus Christi Islanders",
    "UT Rio Grande Valley": "UTRGV Vaqueros",
    "New Mexico St.": "New Mexico State Aggies",
    "SE Missouri St.": "Southeast Missouri State Redhawks",
    "SIU Edwardsville": "SIU Edwardsville Cougars",
    "Tennessee Tech": "Tennessee Tech Golden Eagles",
    "UT Martin": "UT Martin Skyhawks",
    "Tx. Southern": "Texas Southern Tigers",
    "Alcorn St.": "Alcorn State Braves",
    "Bethune-Cookman": "Bethune-Cookman Wildcats",
    "Coppin St.": "Coppin State Eagles",
    "Delaware St.": "Delaware State Hornets",
    "Grambling St.": "Grambling State Tigers",
    "Hampton": "Hampton Pirates",
    "Howard": "Howard Bison",
    "Jackson St.": "Jackson State Tigers",
    "Maryland-Eastern Shore": "Maryland-Eastern Shore Hawks",
    "Morgan St.": "Morgan State Bears",
    "Norfolk St.": "Norfolk State Spartans",
    "North Carolina A&T": "North Carolina A&T Aggies",
    "North Carolina Central": "North Carolina Central Eagles",
    "Prairie View A&M": "Prairie View A&M Panthers",
    "Savannah St.": "Savannah State Tigers",
    "South Carolina St.": "South Carolina State Bulldogs",
    "Southern": "Southern Jaguars",
    "Winston-Salem St.": "Winston-Salem State Rams",
    "NJIT": "NJIT Highlanders",
    "UMBC": "UMBC Retrievers",
    "LIU": "LIU Sharks",
    "Stony Brook": "Stony Brook Seawolves",
    "Albany": "Albany Great Danes",
    "Binghamton": "Binghamton Bearcats",
    "Hartford": "Hartford Hawks",
    "Maine": "Maine Black Bears",
    "UMass Lowell": "UMass Lowell River Hawks",
    "New Hampshire": "New Hampshire Wildcats",
    "Vermont": "Vermont Catamounts",
    "La Salle": "La Salle Explorers",
    "Sacred Heart": "Sacred Heart Pioneers",
    "Quinnipiac": "Quinnipiac Bobcats",
    "Fairfield": "Fairfield Stags",
    "Canisius": "Canisius Golden Griffins",
    "Iona": "Iona Gaels",
    "Manhattan": "Manhattan Jaspers",
    "Marist": "Marist Red Foxes",
    "Niagara": "Niagara Purple Eagles",
    "Rider": "Rider Broncs",
    "Siena": "Siena Saints",
    "St. Peter's": "St. Peter's Peacocks",
    "Monmouth": "Monmouth Hawks",
    "Fairleigh Dickinson": "Fairleigh Dickinson Knights",
    "Wagner": "Wagner Seahawks",
    "Long Island": "LIU Sharks",
    "Colgate": "Colgate Raiders",
    "Holy Cross": "Holy Cross Crusaders",
    "Bucknell": "Bucknell Bison",
    "Loyola (MD)": "Loyola Maryland Greyhounds",
    "Loyola (Chi)": "Loyola Chicago Ramblers",
    "Loyola Chicago": "Loyola Chicago Ramblers",
    "Loyola MD": "Loyola Maryland Greyhounds",
    "American": "American Eagles",
    "Boston College": "Boston College Eagles",
    "Charleston": "Charleston Cougars",
    "Col. of Charleston": "Charleston Cougars",
    "College of Charleston": "Charleston Cougars",
    "Drexel": "Drexel Dragons",
    "Elon": "Elon Phoenix",
    "James Madison": "James Madison Dukes",
    "Towson": "Towson Tigers",
    "UNCW": "UNC Wilmington Seahawks",
    "UNC Greensboro": "UNC Greensboro Spartans",
    "W&M": "William & Mary Tribe",
    "William & Mary": "William & Mary Tribe",
    "Furman": "Furman Paladins",
    "Samford": "Samford Bulldogs",
    "Western Carolina": "Western Carolina Catamounts",
    "Wofford": "Wofford Terriers",
    "The Citadel": "Citadel Bulldogs",
    "VMI": "VMI Keydets",
    "East Tenn. St.": "East Tennessee State Buccaneers",
    "N. Kentucky": "Northern Kentucky Norse",
    "Northern Kentucky": "Northern Kentucky Norse",
    "Youngstown St.": "Youngstown State Penguins",
    "Cleveland St.": "Cleveland State Vikings",
    "Detroit Mercy": "Detroit Mercy Titans",
    "Green Bay": "Green Bay Phoenix",
    "IUPUI": "IUPUI Jaguars",
    "Milwaukee": "Milwaukee Panthers",
    "Oakland": "Oakland Golden Grizzlies",
    "Robert Morris": "Robert Morris Colonials",
    "Wright St.": "Wright State Raiders",
    "Ill.-Chicago": "UIC Flames",
    "Valparaiso": "Valparaiso Beacons",
    "Northern Illinois": "Northern Illinois Huskies",
    "Eastern Ill.": "Eastern Illinois Panthers",
    "Eastern Michigan": "Eastern Michigan Eagles",
    "Western Ill.": "Western Illinois Leathernecks",
    "Indiana St.": "Indiana State Sycamores",
    "Southern Ill.": "Southern Illinois Salukis",
    "Cal St. Fullerton": "Cal State Fullerton Titans",
    "Cal St. Northridge": "Cal State Northridge Matadors",
    "CS Fullerton": "Cal State Fullerton Titans",
    "UC Davis": "UC Davis Aggies",
    "UC Riverside": "UC Riverside Highlanders",
    "Long Beach St.": "Long Beach State Beach",
    "Cal Poly": "Cal Poly Mustangs",
    "CS Bakersfield": "Cal State Bakersfield Roadrunners",
    "Hawaii": "Hawaii Rainbow Warriors",
    "Portland": "Portland Pilots",
    "Gonzaga": "Gonzaga Bulldogs",
    "Pepperdine": "Pepperdine Waves",
    "Portland St.": "Portland State Vikings",
    "Sacramento St.": "Sacramento State Hornets",
    "San Jose St.": "San Jose State Spartans",
    "Idaho St.": "Idaho State Bengals",
    "Northern Ariz.": "Northern Arizona Lumberjacks",
    "Eastern Wash.": "Eastern Washington Eagles",
    "Montana": "Montana Grizzlies",
    "N. Colorado": "Northern Colorado Bears",
    "Southern Utah": "Southern Utah Thunderbirds",
    "Weber St.": "Weber State Wildcats",
    "S. Dakota St.": "South Dakota State Jackrabbits",
    "South Dakota": "South Dakota Coyotes",
    "Denver": "Denver Pioneers",
    "Oral Roberts": "Oral Roberts Golden Eagles",
    "W. Illinois": "Western Illinois Leathernecks",
    "S. Utah": "Southern Utah Thunderbirds",
    "Incarnate Word": "Incarnate Word Cardinals",
    "Abilene Christian": "Abilene Christian Wildcats",
    "McNeese St.": "McNeese Cowboys",
    "Northwestern St.": "Northwestern State Demons",
    "Nicholls St.": "Nicholls State Colonels",
    "SE Louisiana": "Southeastern Louisiana Lions",
    "New Orleans": "New Orleans Privateers",
    "Houston Baptist": "Houston Christian Huskies",
    "Houston Christian": "Houston Christian Huskies",
    "St. Francis (PA)": "St. Francis (PA) Red Flash",
    "St. Francis (BKN)": "St. Francis Brooklyn Terriers",
    "CCNY": "City College of New York",
    "Navy": "Navy Midshipmen",
    "Army": "Army Black Knights",
    "Air Force": "Air Force Falcons",
    "Lehigh": "Lehigh Mountain Hawks",
    "Lafayette": "Lafayette Leopards",
    "Fordham": "Fordham Rams",
    "Manhattan": "Manhattan Jaspers",
    "S.F. Austin": "SFA Lumberjacks",
    "SF Austin": "SFA Lumberjacks",
    "SC State": "South Carolina State Bulldogs",
    "MD-Eastern Shore": "Maryland-Eastern Shore Hawks",
    "Florida Gulf Coast": "FGCU Eagles",
    "FGCU": "FGCU Eagles",
    "Kennesaw St.": "Kennesaw State Owls",
    "Kennesaw State": "Kennesaw State Owls",
    "Fla. Gulf Coast": "FGCU Eagles",
    "Jax. St.": "Jacksonville State Gamecocks",
    "Jacksonville St.": "Jacksonville State Gamecocks",
}

# Merge the two maps; extra entries override base where there's a conflict
_FULL_NAME_MAP: dict[str, str] = {**_NAME_MAP, **_BART_EXTRA}

# JSON field indices (confirmed against multiple seasons)
_F_RANK = 0
_F_TEAM = 1
_F_ADJOE = 4
_F_ADJDE = 6
_F_BARTHAG = 8
_F_SOS = 15
_F_LUCK = 33
_F_WAB = 41
_F_ADJT = 44  # Adjusted Tempo (possessions per 40 min, pace-adjusted; ~62–72 for D-I)

BARTTORVIK_URL = "https://barttorvik.com/{year}_team_results.json"


def fetch_season(year: int, timeout: int = 30) -> list:
    """Fetch raw JSON list for a single season from barttorvik.com."""
    url = BARTTORVIK_URL.format(year=year)
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 (compatible; ncaa-bets-pipeline/1.0)"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        import json
        return json.loads(resp.read())


def parse_season(raw: list, season: int) -> pd.DataFrame:
    """Convert a raw JSON list into a tidy DataFrame."""
    rows = []
    for entry in raw:
        if len(entry) <= _F_ADJT:
            continue
        source_name = str(entry[_F_TEAM]).strip()
        team = _FULL_NAME_MAP.get(source_name, source_name)
        adjoe = float(entry[_F_ADJOE])
        adjde = float(entry[_F_ADJDE])
        rows.append({
            "season": season,
            "team": team,
            "source_name": source_name,
            "rank": int(entry[_F_RANK]),
            "adjoe": round(adjoe, 4),
            "adjde": round(adjde, 4),
            "adj_em": round(adjoe - adjde, 4),
            "barthag": round(float(entry[_F_BARTHAG]), 6),
            "sos_adj": round(float(entry[_F_SOS]), 6),
            "luck": round(float(entry[_F_LUCK]), 6),
            "wab": round(float(entry[_F_WAB]), 4),
            "adj_t": round(float(entry[_F_ADJT]), 4),
        })
    return pd.DataFrame(rows)


def fetch_and_save(seasons: list[int], out_dir: str, dry_run: bool = False,
                   delay: float = 0.5) -> None:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    for i, season in enumerate(seasons):
        print(f"Fetching BartTorvik {season}...", end=" ", flush=True)
        try:
            raw = fetch_season(season)
            df = parse_season(raw, season)
        except Exception as exc:
            print(f"ERROR: {exc}")
            continue

        if dry_run:
            print(f"{len(df)} teams (dry run — not saved)")
            print(df[["team", "rank", "adjoe", "adjde", "adj_em", "barthag", "wab"]].head(10).to_string(index=False))
        else:
            dest = out_path / f"barttorvik_{season}.csv"
            df.to_csv(dest, index=False)
            print(f"{len(df)} teams → {dest}")

        if i < len(seasons) - 1:
            time.sleep(delay)


def main() -> int:
    p = argparse.ArgumentParser(description="Fetch BartTorvik T-Rank efficiency data.")
    group = p.add_mutually_exclusive_group()
    group.add_argument("--season", type=int, help="single season to fetch (e.g. 2026)")
    group.add_argument("--seasons", nargs="+", type=int, help="multiple seasons")
    p.add_argument(
        "--out", default="data/processed/features",
        help="output directory (default: data/processed/features)",
    )
    p.add_argument("--dry-run", action="store_true", help="print data; don't write files")
    p.add_argument("--delay", type=float, default=0.5, help="seconds between requests (default: 0.5)")
    args = p.parse_args()

    if args.season:
        seasons = [args.season]
    elif args.seasons:
        seasons = sorted(args.seasons)
    else:
        # Default: fetch all seasons that BartTorvik has data for
        import datetime
        current = datetime.datetime.now().year
        seasons = list(range(2019, current + 1))

    fetch_and_save(seasons, args.out, dry_run=args.dry_run, delay=args.delay)
    return 0


if __name__ == "__main__":
    sys.exit(main())
