#!/usr/bin/env python3
"""
Fetch a canonical list of NCAA Division I men's basketball programs
from Wikipedia and write a plain `d1_list.txt` (one school per line).

Usage:
  python3 scripts/fetch_d1_list.py --out data/mappings/d1_list.txt

Dependencies: requests, beautifulsoup4
"""
import argparse
import re
import sys

try:
    import requests
    from bs4 import BeautifulSoup
except Exception as e:
    print("Missing dependency: install with `pip install requests beautifulsoup4`", file=sys.stderr)
    raise

WIKI_URL = "https://en.wikipedia.org/wiki/List_of_NCAA_Division_I_men%27s_basketball_programs"


def clean_name(s: str) -> str:
    s = s.strip()
    # Remove notes in parentheses like (UMass), keep main name
    s = re.sub(r"\s*\(.*?\)", "", s).strip()
    # Collapse whitespace
    s = re.sub(r"\s+", " ", s)
    return s


def fetch_wikipedia(url=WIKI_URL):
    r = requests.get(url, headers={"User-Agent": "ncaa-bets-fetcher/1.0 (github.com)"})
    r.raise_for_status()
    return r.text


def parse_wikipedia(html: str):
    soup = BeautifulSoup(html, "html.parser")
    teams = []
    # The page contains many tables; the team rows include a cell with the school name.
    # Find all table rows that have a pipe-delimited structure or first <td> with a link
    for table in soup.find_all("table"):
        # Heuristic: wikitable class and more than 3 rows
        classes = table.get("class") or []
        if "wikitable" not in classes and "sortable" not in classes:
            # still try other tables
            pass
        rows = table.find_all("tr")
        for tr in rows:
            tds = tr.find_all(["td","th"])
            if not tds:
                continue
            # look for first cell that looks like a school name (contains a link to an article)
            first = tds[0]
            a = first.find("a")
            if a and a.get_text(strip=True):
                name = a.get_text(strip=True)
                name = clean_name(name)
                # basic filtering: skip short headers
                if len(name) < 3:
                    continue
                if name not in teams:
                    teams.append(name)
    # Fallback: also try scanning for "| School Name |" patterns in page text
    if not teams:
        txt = soup.get_text("\n")
        for line in txt.splitlines():
            if "|" in line:
                parts = [p.strip() for p in line.split("|") if p.strip()]
                if parts and len(parts[0]) > 3:
                    candidate = clean_name(parts[0])
                    if candidate and candidate not in teams:
                        teams.append(candidate)
    return teams


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True, help="output file (e.g. data/mappings/d1_list.txt)")
    p.add_argument("--source", default=WIKI_URL, help="Wikipedia URL to fetch")
    args = p.parse_args()

    print(f"Fetching {args.source} ...")
    html = fetch_wikipedia(args.source)
    print("Parsing page ...")
    teams = parse_wikipedia(html)
    print(f"Found {len(teams)} candidate team names")

    # Write the list
    outpath = args.out
    with open(outpath, "w", encoding="utf-8") as f:
        for t in teams:
            f.write(t + "\n")
    print(f"Wrote {len(teams)} lines to {outpath}")


if __name__ == "__main__":
    main()
