"""Score human 0/1 ratings on a grading sheet (primary project metric).

Usage (from project root):
    uv run python src/score_human.py
    uv run python src/score_human.py --csv results/grading_sheet_post_grok.csv

Rules (same as notebook §7):
  - Each cell is 1 (correct) or 0 (incorrect)
  - A row counts correct only if BOTH raters mark 1
  - Rows with missing ratings are skipped in that bucket
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import PROJECT_ROOT  # noqa: E402

RESULTS_DIR = PROJECT_ROOT / "results"
DEFAULT_CSV = RESULTS_DIR / "grading_sheet_post_grok.csv"
RATER_COLS = ["base_rater1", "base_rater2", "tuned_rater1", "tuned_rater2"]


def _to_bin(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    return s


def report(sub: pd.DataFrame, r1: str, r2: str, label: str) -> dict:
    a = _to_bin(sub[r1])
    b = _to_bin(sub[r2])
    graded = a.notna() & b.notna()
    n = int(graded.sum())
    if n == 0:
        print(f"  {label}: (no dual-rated rows yet)")
        return {"n": 0, "accuracy": None, "agreement": None}
    both_correct = (a[graded] == 1) & (b[graded] == 1)
    agree = a[graded] == b[graded]
    acc = float(both_correct.mean())
    agr = float(agree.mean())
    print(f"  {label}: accuracy={acc:.0%}  agreement={agr:.0%}  (n={n})")
    return {"n": n, "accuracy": acc, "agreement": agr}


def score(csv_path: Path) -> dict:
    df = pd.read_csv(csv_path, dtype=str).fillna("")
    missing = [c for c in RATER_COLS if c not in df.columns]
    if missing:
        raise SystemExit(f"CSV missing rater columns: {missing}")
    if "type" not in df.columns:
        df["type"] = "translate"

    print(f"Human scores from {csv_path}  ({len(df)} rows)\n")
    out: dict = {"source_csv": str(csv_path.relative_to(PROJECT_ROOT)), "buckets": {}}

    print("=== Translation (correct = BOTH raters marked 1) ===")
    tr = df[df["type"] == "translate"]
    out["buckets"]["translate_base"] = report(tr, "base_rater1", "base_rater2", "BASE ")
    out["buckets"]["translate_tuned"] = report(tr, "tuned_rater1", "tuned_rater2", "TUNED")

    print("\n=== Unanswerable / abstain ===")
    un = df[df["type"] == "unanswerable"]
    if len(un) == 0:
        print("  (no unanswerable rows)")
    else:
        out["buckets"]["unanswerable_base"] = report(un, "base_rater1", "base_rater2", "BASE ")
        out["buckets"]["unanswerable_tuned"] = report(un, "tuned_rater1", "tuned_rater2", "TUNED")

    # Overall inter-rater agreement across all dual-rated cells (base+tuned pooled)
    b = df[["base_rater1", "base_rater2"]].rename(columns={"base_rater1": "a", "base_rater2": "b"})
    t = df[["tuned_rater1", "tuned_rater2"]].rename(columns={"tuned_rater1": "a", "tuned_rater2": "b"})
    allg = pd.concat([b, t], ignore_index=True)
    allg["a"] = _to_bin(allg["a"])
    allg["b"] = _to_bin(allg["b"])
    mask = allg["a"].notna() & allg["b"].notna()
    if mask.any():
        agr = float((allg.loc[mask, "a"] == allg.loc[mask, "b"]).mean())
        print(f"\nInter-rater agreement (all dual-rated pairs): {agr:.0%}  (n={int(mask.sum())})")
        out["inter_rater_agreement"] = agr
        out["inter_rater_n"] = int(mask.sum())
    else:
        print("\nInter-rater agreement: (no dual-rated rows yet)")
        out["inter_rater_agreement"] = None
        out["inter_rater_n"] = 0

    RESULTS_DIR.mkdir(exist_ok=True)
    out_path = RESULTS_DIR / "human_metrics.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nWrote {out_path}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--csv",
        type=Path,
        default=DEFAULT_CSV,
        help="Grading sheet with rater columns (default: results/grading_sheet_post_grok.csv)",
    )
    args = ap.parse_args()
    path = args.csv if args.csv.is_absolute() else PROJECT_ROOT / args.csv
    if not path.exists():
        raise SystemExit(f"Missing {path}")
    score(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
