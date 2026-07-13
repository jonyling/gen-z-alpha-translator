"""Turn the raw slang datasets into clean, training-ready files.

Pipeline (see design spec section 3):

    data/raw/*  --[normalize per SOURCES]-->  records
    records     --[dedupe]-->                 unique pairs
    unique      --[freeze eval set]-->         eval.jsonl   (held out, NEVER trained on)
                --[expand both directions]-->  train.jsonl  (chat format)

Run it:  uv run python src/prepare_data.py
Output:  data/processed/train.jsonl and data/processed/eval.jsonl

The script is deterministic (fixed seed), so the frozen eval set is identical
every run. It refuses to leak eval rows into the training set.
"""

from __future__ import annotations

import json
import random
import sys
from collections import Counter

import pandas as pd

from config import (
    EVAL_PATH,
    EVAL_PER_DIRECTION,
    PROCESSED_DIR,
    RANDOM_SEED,
    RAW_DIR,
    SOURCES,
    TAG_TO_ENGLISH,
    TAG_TO_SLANG,
    TRAIN_PATH,
)


def _clean(text) -> str:
    """Normalise a cell to a stripped string, or '' if missing."""
    if text is None:
        return ""
    if isinstance(text, float) and pd.isna(text):
        return ""
    return str(text).strip()


def load_source(src: dict) -> list[dict]:
    """Read one source file and normalise it into a list of record dicts.

    A record always has: slang, english, source. It may also carry term,
    meaning, difficulty, strat (used to build a good eval set).
    """
    path = RAW_DIR / src["file"]
    if not path.exists():
        if src.get("use_for_eval"):
            # This is the file the eval set is built from — missing it is serious.
            print(f"  !!!! CRITICAL: eval source {src['file']} NOT FOUND in data/raw/ "
                  f"-- the eval set will be empty/degenerate. Fix the filename in config.py.")
        else:
            print(f"  !! SKIP: {src['file']} not found in data/raw/")
        return []

    df = pd.read_csv(path) if path.suffix == ".csv" else pd.read_excel(path)

    missing = [c for c in (src["slang_col"], src["english_col"]) if c not in df.columns]
    if missing:
        print(f"  !! SKIP: {src['file']} missing column(s) {missing}. "
              f"Available: {list(df.columns)}")
        return []

    records = []
    for _, row in df.iterrows():
        slang = _clean(row.get(src["slang_col"]))
        english = _clean(row.get(src["english_col"]))
        if not slang or not english or slang.lower() == english.lower():
            continue  # need a real, differing pair
        rec = {
            "slang": slang,
            "english": english,
            "source": src["file"],
            "term": _clean(row.get(src["term_col"])) if src.get("term_col") else "",
            "meaning": _clean(row.get(src["meaning_col"])) if src.get("meaning_col") else "",
            "difficulty": _clean(row.get(src["difficulty_col"])) if src.get("difficulty_col") else "",
            "strat": _clean(row.get(src["strat_col"])) if src.get("strat_col") else "",
            "use_for_eval": bool(src.get("use_for_eval", False)),
        }
        records.append(rec)

    print(f"  ok: {src['file']:<40} -> {len(records):>5} usable pairs")
    return records


def dedupe(records: list[dict]) -> list[dict]:
    """Drop exact-duplicate pairs (case-insensitive on slang+english)."""
    seen = set()
    out = []
    for r in records:
        key = (r["slang"].lower(), r["english"].lower())
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
    return out


def freeze_eval(records: list[dict], rng: random.Random) -> tuple[list[dict], list[dict]]:
    """Split off a frozen eval set from rows flagged use_for_eval.

    We stratify by the 'strat' column (e.g. trend_status) so the test set
    spans trending / real / generated, and prefer rows that have a written
    meaning (the answer key for graders). Returns (eval_rows, train_rows).
    """
    eligible = [r for r in records if r["use_for_eval"] and r["meaning"]]
    if len(eligible) < EVAL_PER_DIRECTION:
        print(f"  !! WARNING: only {len(eligible)} eval-eligible rows "
              f"(< {EVAL_PER_DIRECTION}). Using all of them.")

    # Stratified sample: spread picks across the strat buckets.
    buckets: dict[str, list[dict]] = {}
    for r in eligible:
        buckets.setdefault(r["strat"] or "unknown", []).append(r)
    for b in buckets.values():
        rng.shuffle(b)

    # Prefer real/human-authored slang over synthetically 'generated' rows: those
    # references read more naturally, which makes for a cleaner human-graded eval.
    # We round-robin across the PREFERRED buckets first (keeps direction/topic
    # diversity), and only dip into 'generated'/'unknown' if we still fall short.
    def is_preferred(name: str) -> bool:
        return name not in ("generated", "unknown", "")

    preferred = sorted(k for k in buckets if is_preferred(k))
    fallback = sorted(k for k in buckets if not is_preferred(k))

    picked: list[dict] = []

    def round_robin(keys: list[str]) -> None:
        i = 0
        while len(picked) < EVAL_PER_DIRECTION and any(buckets[k] for k in keys):
            k = keys[i % len(keys)]
            if buckets[k]:
                picked.append(buckets[k].pop())
            i += 1

    round_robin(preferred)
    if len(picked) < EVAL_PER_DIRECTION:
        round_robin(fallback)

    # Ban EVERY text string that appears in the eval set (both the slang side
    # and the english side). This drops not just the exact eval pairs but any
    # other-source row that reuses one of those sentences, preventing leakage.
    banned_texts = set()
    for r in picked:
        banned_texts.add(r["slang"].lower())
        banned_texts.add(r["english"].lower())
    train_rows = [r for r in records
                  if r["slang"].lower() not in banned_texts
                  and r["english"].lower() not in banned_texts]

    # Build eval items: each frozen pair becomes TWO test prompts (both directions).
    eval_items = []
    for idx, r in enumerate(picked):
        eval_items.append({
            "id": f"e2e_{idx:03d}",              # slang -> English
            "direction": "to_english",
            "tag": TAG_TO_ENGLISH,
            "input": r["slang"],
            "reference": r["english"],
            "term": r["term"],
            "meaning": r["meaning"],
            "strat": r["strat"],
        })
        eval_items.append({
            "id": f"e2s_{idx:03d}",              # English -> slang
            "direction": "to_slang",
            "tag": TAG_TO_SLANG,
            "input": r["english"],
            "reference": r["slang"],
            "term": r["term"],
            "meaning": r["meaning"],
            "strat": r["strat"],
        })
    return eval_items, train_rows


def to_chat(tag: str, source_text: str, target_text: str) -> dict:
    """One training example in HF chat format."""
    return {
        "messages": [
            {"role": "user", "content": f"{tag}\n{source_text}"},
            {"role": "assistant", "content": target_text},
        ]
    }


def build_train(train_rows: list[dict]) -> list[dict]:
    """Expand each pair into BOTH directions with the direction tag."""
    examples = []
    for r in train_rows:
        examples.append(to_chat(TAG_TO_ENGLISH, r["slang"], r["english"]))
        examples.append(to_chat(TAG_TO_SLANG, r["english"], r["slang"]))
    return examples


def write_jsonl(path, rows) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_existing_eval() -> list[dict] | None:
    """Return the already-frozen eval set if one exists, else None.

    Freezing the eval set on disk means adding/removing datasets later does NOT
    silently change which items are tested (which would misalign human grades).
    To deliberately re-freeze, delete data/processed/eval.jsonl and re-run.
    """
    if not EVAL_PATH.exists():
        return None
    with open(EVAL_PATH, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def banned_from_eval(eval_items: list[dict]) -> set:
    """Every text that appears in the eval set (input + reference), lowercased."""
    banned = set()
    for it in eval_items:
        banned.add(it["input"].lower())
        banned.add(it["reference"].lower())
    return banned


def main() -> int:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    rng = random.Random(RANDOM_SEED)

    print("Loading sources from data/raw/ ...")
    all_records: list[dict] = []
    for src in SOURCES:
        all_records.extend(load_source(src))

    if not all_records:
        print("\nERROR: no usable pairs loaded. Check SOURCES in src/config.py.")
        return 1

    before = len(all_records)
    all_records = dedupe(all_records)
    print(f"\nDeduped: {before} -> {len(all_records)} unique pairs")

    existing_eval = load_existing_eval()
    if existing_eval is not None:
        # Frozen eval already exists -> keep it, just rebuild training from the
        # current data while excluding everything that appears in the eval set.
        eval_items = existing_eval
        banned = banned_from_eval(eval_items)
        train_rows = [r for r in all_records
                      if r["slang"].lower() not in banned
                      and r["english"].lower() not in banned]
        print(f"\nUsing EXISTING frozen eval.jsonl ({len(eval_items)} items). "
              f"Delete it to re-freeze.")
    else:
        eval_items, train_rows = freeze_eval(all_records, rng)
        print("\nFroze a NEW eval set (no existing eval.jsonl found).")

    train_examples = build_train(train_rows)

    # Safety: eval inputs must NOT appear as training targets/inputs.
    eval_inputs = {it["input"].lower() for it in eval_items}
    leak = sum(
        1 for ex in train_examples
        if ex["messages"][0]["content"].split("\n", 1)[-1].lower() in eval_inputs
    )
    if leak:
        print(f"  !! WARNING: {leak} potential eval/train overlaps detected.")

    write_jsonl(TRAIN_PATH, train_examples)
    write_jsonl(EVAL_PATH, eval_items)

    print("\n--- SUMMARY ---")
    print(f"train.jsonl : {len(train_examples):>5} examples "
          f"(from {len(train_rows)} pairs x 2 directions)")
    print(f"eval.jsonl  : {len(eval_items):>5} items "
          f"({EVAL_PER_DIRECTION} pairs x 2 directions)")
    dirs = Counter(it["direction"] for it in eval_items)
    print(f"  eval by direction: {dict(dirs)}")
    strat = Counter(it["strat"] or "unknown" for it in eval_items)
    print(f"  eval by strat    : {dict(strat)}")
    print(f"\nWrote:\n  {TRAIN_PATH}\n  {EVAL_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
