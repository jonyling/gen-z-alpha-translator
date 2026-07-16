"""Automatic metrics on a grading sheet (base vs tuned outputs already filled).

Metrics (proposal + METEOR):
  1. Accept / substring  — meaning or reference appears in the prediction
  2. Abstain rate        — unanswerable rows where the model declines
                           (hallucination = failing to abstain)
  3. BERTScore           — embedding semantic similarity vs reference
  4. METEOR              — flexible alignment + synonyms (good for translation)

Usage (from project root):
    uv run python src/evaluate.py
    uv run python src/evaluate.py --csv results/grading_sheet_retrained.csv
    uv run python src/evaluate.py --skip-bert   # faster, skips BERTScore download

Writes:
    results/auto_metrics.json
    results/auto_metrics_by_row.csv
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd

# Allow `uv run python src/evaluate.py` without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from abstain import is_abstention  # noqa: E402
from config import PROJECT_ROOT  # noqa: E402

RESULTS_DIR = PROJECT_ROOT / "results"
DEFAULT_CSV = RESULTS_DIR / "grading_sheet.csv"


def _norm(s: str) -> str:
    s = (s or "").lower().strip()
    s = re.sub(r"\s+", " ", s)
    return s


def accept_match(pred: str, reference: str, meaning: str) -> bool:
    """True if the prediction contains the key meaning and/or the reference.

    Proposal metric: 'accept' via substring matching. Meaning is the primary
    keyphrase (slang gloss); full reference is a secondary accept.
    """
    p = _norm(pred)
    if not p:
        return False
    m = _norm(meaning)
    # Skip the placeholder used on unanswerable rows.
    if m and not m.startswith("(unanswerable") and m in p:
        return True
    r = _norm(reference)
    if r and len(r) >= 4 and r in p:
        return True
    # Also accept if pred is contained in reference (model was terse but right).
    if r and len(p) >= 4 and p in r:
        return True
    return False


def _ensure_nltk():
    import nltk

    for pkg in ("wordnet", "omw-1.4", "punkt", "punkt_tab"):
        try:
            nltk.data.find(
                f"corpora/{pkg}" if pkg in ("wordnet", "omw-1.4") else f"tokenizers/{pkg}"
            )
        except LookupError:
            nltk.download(pkg, quiet=True)


def meteor_scores(preds: list[str], refs: list[str]) -> list[float]:
    from nltk.translate.meteor_score import meteor_score

    _ensure_nltk()
    out = []
    for pred, ref in zip(preds, refs):
        ref_toks = _norm(ref).split()
        pred_toks = _norm(pred).split()
        if not ref_toks or not pred_toks:
            out.append(0.0)
            continue
        out.append(float(meteor_score([ref_toks], pred_toks)))
    return out


def bert_scores(preds: list[str], refs: list[str], model_type: str) -> list[float]:
    """Return per-row BERTScore F1. Uses a small model so it fits a laptop GPU/CPU."""
    from bert_score import score as bert_score_fn

    # Empty strings crash some versions — pad with a space.
    preds_c = [p if (p or "").strip() else " " for p in preds]
    refs_c = [r if (r or "").strip() else " " for r in refs]
    _, _, f1 = bert_score_fn(
        preds_c, refs_c, model_type=model_type, verbose=False, lang="en"
    )
    return [float(x) for x in f1.tolist()]


def _mean(xs: list[float]) -> float | None:
    return sum(xs) / len(xs) if xs else None


def score_model(
    df: pd.DataFrame,
    output_col: str,
    skip_bert: bool,
    bert_model: str,
) -> tuple[dict, pd.DataFrame]:
    """Compute auto metrics for one model column (base_output or tuned_output)."""
    rows = []
    tr = df[df["type"] == "translate"].copy()
    un = df[df["type"] == "unanswerable"].copy()

    # --- translate: accept / METEOR / BERT ---
    accepts, meteors, berts = [], [], []
    preds = [str(x) if pd.notna(x) else "" for x in tr[output_col].tolist()]
    refs = [str(x) if pd.notna(x) else "" for x in tr["reference"].tolist()]
    meanings = [str(x) if pd.notna(x) else "" for x in tr["meaning"].tolist()]

    for pred, ref, meaning in zip(preds, refs, meanings):
        accepts.append(accept_match(pred, ref, meaning))

    if len(preds):
        meteors = meteor_scores(preds, refs)
        if not skip_bert:
            berts = bert_scores(preds, refs, bert_model)
        else:
            berts = [float("nan")] * len(preds)

    for i, (_, row) in enumerate(tr.iterrows()):
        rows.append({
            "id": row["id"],
            "type": "translate",
            "direction": row["direction"],
            "model": output_col.replace("_output", ""),
            "accept": int(accepts[i]),
            "meteor": meteors[i] if meteors else None,
            "bertscore_f1": berts[i] if berts else None,
            "abstained": int(is_abstention(preds[i])),
        })

    # --- unanswerable: abstain (correct) vs hallucinate ---
    abstain_hits = []
    for _, row in un.iterrows():
        pred = str(row[output_col]) if pd.notna(row[output_col]) else ""
        hit = is_abstention(pred)
        abstain_hits.append(hit)
        rows.append({
            "id": row["id"],
            "type": "unanswerable",
            "direction": "unanswerable",
            "model": output_col.replace("_output", ""),
            "accept": None,
            "meteor": None,
            "bertscore_f1": None,
            "abstained": int(hit),
        })

    # Aggregate by direction
    by_dir = {}
    for direction in ("to_english", "to_slang", "ALL"):
        mask = [True] * len(tr) if direction == "ALL" else (tr["direction"] == direction).tolist()
        idx = [i for i, m in enumerate(mask) if m]
        if not idx:
            continue
        by_dir[direction] = {
            "n": len(idx),
            "accept_rate": _mean([float(accepts[i]) for i in idx]),
            "meteor": _mean([meteors[i] for i in idx]) if meteors else None,
            "bertscore_f1": (
                _mean([berts[i] for i in idx if berts[i] == berts[i]])  # skip NaN
                if berts and not skip_bert
                else None
            ),
        }

    summary = {
        "n_translate": int(len(tr)),
        "n_unanswerable": int(len(un)),
        "accept_rate": _mean([float(a) for a in accepts]),
        "meteor": _mean(meteors) if meteors else None,
        "bertscore_f1": _mean([x for x in berts if x == x]) if berts and not skip_bert else None,
        "abstain_rate_unanswerable": _mean([float(a) for a in abstain_hits]),
        "hallucination_rate_unanswerable": (
            _mean([float(not a) for a in abstain_hits]) if abstain_hits else None
        ),
        "by_direction": by_dir,
    }
    return summary, pd.DataFrame(rows)


def _fmt(x: float | None) -> str:
    return "—" if x is None else f"{x:.1%}" if x <= 1.0 else f"{x:.3f}"


def _fmt_score(x: float | None) -> str:
    return "—" if x is None else f"{x:.3f}"


def print_report(name: str, s: dict) -> None:
    print(f"\n=== {name.upper()} ===")
    print(f"Translate n={s['n_translate']}  |  Unanswerable n={s['n_unanswerable']}")
    print(f"  Accept (substring) : {_fmt(s['accept_rate'])}")
    print(f"  METEOR             : {_fmt_score(s['meteor'])}")
    print(f"  BERTScore F1       : {_fmt_score(s['bertscore_f1'])}")
    print(f"  Abstain (unans.)   : {_fmt(s['abstain_rate_unanswerable'])}  "
          f"(hallucination = {_fmt(s['hallucination_rate_unanswerable'])})")
    for d, block in s.get("by_direction", {}).items():
        print(
            f"  [{d:11}] accept={_fmt(block['accept_rate'])}  "
            f"METEOR={_fmt_score(block['meteor'])}  "
            f"BERT={_fmt_score(block['bertscore_f1'])}  (n={block['n']})"
        )


def main() -> None:
    ap = argparse.ArgumentParser(description="Auto-eval accept / abstain / BERTScore / METEOR")
    ap.add_argument(
        "--csv",
        type=Path,
        default=DEFAULT_CSV,
        help="Grading sheet with base_output and tuned_output columns",
    )
    ap.add_argument("--skip-bert", action="store_true", help="Skip BERTScore (faster)")
    ap.add_argument(
        "--bert-model",
        default="bert-base-uncased",
        help="HF model for BERTScore (default: bert-base-uncased)",
    )
    args = ap.parse_args()

    csv_path = args.csv if args.csv.is_absolute() else PROJECT_ROOT / args.csv
    if not csv_path.exists():
        raise SystemExit(
            f"Missing {csv_path}. Run notebook §7 first to export the grading sheet "
            "(needs base_output + tuned_output filled)."
        )

    df = pd.read_csv(csv_path)
    for col in ("base_output", "tuned_output", "reference", "meaning", "type"):
        if col not in df.columns:
            raise SystemExit(f"CSV missing required column: {col}")
    if "type" not in df.columns:
        df["type"] = "translate"
    df["type"] = df["type"].fillna("translate")

    print(f"Scoring {csv_path}  ({len(df)} rows)")
    if args.skip_bert:
        print("(BERTScore skipped)")

    base_sum, base_rows = score_model(df, "base_output", args.skip_bert, args.bert_model)
    tuned_sum, tuned_rows = score_model(df, "tuned_output", args.skip_bert, args.bert_model)

    print_report("base", base_sum)
    print_report("tuned", tuned_sum)

    RESULTS_DIR.mkdir(exist_ok=True)
    metrics = {
        "source_csv": str(csv_path.relative_to(PROJECT_ROOT)),
        "bert_model": None if args.skip_bert else args.bert_model,
        "base": base_sum,
        "tuned": tuned_sum,
        "notes": {
            "accept": "substring: meaning or reference appears in prediction",
            "abstain": "unanswerable rows; hallucination = fail to abstain",
            "bertscore": "semantic F1 vs reference (translate rows only)",
            "meteor": "synonym-aware alignment vs reference (translate rows only)",
            "primary_metric": "human grading in grading_sheet.csv (not this script)",
        },
    }
    out_json = RESULTS_DIR / "auto_metrics.json"
    out_json.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    by_row = pd.concat([base_rows, tuned_rows], ignore_index=True)
    out_csv = RESULTS_DIR / "auto_metrics_by_row.csv"
    by_row.to_csv(out_csv, index=False)

    print(f"\nWrote {out_json}")
    print(f"Wrote {out_csv}")
    print("Human grading (primary) is still: fill rater columns in the grading sheet, "
          "then run notebook §7 scoring cell.")


if __name__ == "__main__":
    main()
