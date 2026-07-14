"""Fetch real Urban Dictionary definitions for our slang terms.

Crowd-sourced = real, current usage. Writes data/dictionaries/urban_slang.csv,
which then feeds training like any other dictionary (see DICT_SOURCES in config).

- Resumable: skips terms already in the CSV, so you can stop/restart and re-run
  later to pick up NEW terms only.
- Reproducible after fetching: once the CSV exists, training is offline again.
- Needs internet. Polite rate limit (~4 requests/sec).

Usage:
    uv run python src/fetch_urban.py            # fetch all uncached terms
    uv run python src/fetch_urban.py --limit 50 # just the first 50 (quick test)
"""

from __future__ import annotations

import argparse
import csv
import sys
import time

import pandas as pd
import requests

from config import DICT_DIR, DICT_SOURCES, RAW_DIR, SOURCES

OUT = DICT_DIR / "urban_slang.csv"
API = "https://api.urbandictionary.com/v0/define"


def collect_terms() -> list[str]:
    """Unique slang terms from the dictionaries + the datasets' term columns."""
    terms: set[str] = set()
    for src in DICT_SOURCES:
        if src.get("emoji") or src["file"] == "urban_slang.csv":
            continue
        p = DICT_DIR / src["file"]
        if p.exists():
            df = pd.read_csv(p)
            if src["term_col"] in df.columns:
                terms |= {str(t).strip() for t in df[src["term_col"]].dropna()}
    for src in SOURCES:
        if src.get("term_col"):
            p = RAW_DIR / src["file"]
            if p.exists():
                df = pd.read_csv(p)
                if src["term_col"] in df.columns:
                    terms |= {str(t).strip() for t in df[src["term_col"]].dropna()}
    # keep short, real terms only
    return sorted({t for t in terms if t and 1 <= len(t) <= 40})


def clean(s: str) -> str:
    return (s or "").replace("[", "").replace("]", "").replace("\r", " ").replace("\n", " ").strip()


def load_done() -> set[str]:
    if not OUT.exists():
        return set()
    with open(OUT, encoding="utf-8") as f:
        return {r["term"] for r in csv.DictReader(f)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="fetch at most N new terms")
    args = ap.parse_args()

    DICT_DIR.mkdir(parents=True, exist_ok=True)
    terms = collect_terms()
    done = load_done()
    todo = [t for t in terms if t not in done]
    if args.limit:
        todo = todo[: args.limit]
    print(f"{len(terms)} unique terms | {len(done)} already cached | fetching {len(todo)}")

    session = requests.Session()
    new = 0
    with open(OUT, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if not done:
            w.writerow(["term", "definition", "example"])
        for i, term in enumerate(todo, 1):
            try:
                data = session.get(API, params={"term": term}, timeout=10).json()
                lst = data.get("list") or []
                if lst:
                    top = lst[0]
                    w.writerow([term, clean(top.get("definition", "")), clean(top.get("example", ""))])
                    f.flush()
                    new += 1
            except Exception as e:  # network hiccup, bad json, etc. — skip, keep going
                print(f"  skip {term!r}: {e}")
            time.sleep(0.25)
            if i % 50 == 0:
                print(f"  {i}/{len(todo)} …")

    print(f"\nWrote {new} new definitions -> {OUT}")
    print("Now re-run  uv run python src/prepare_data.py  and retrain to use them.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
