"""Stage 2 — teacher generation. Given a Recipe, ask the NVIDIA teacher for a
natural slang/english pair fitting it. Resumable: appends to a raw jsonl.

Usage:
    uv run python -m sdg.generate --pilot          # 8 examples, printed, not saved
    uv run python -m sdg.generate --limit 1200     # full run, appends to raw jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
import time

from config import RAW_DIR, SDG_REQUEST_SLEEP, SDG_TARGET
from sdg.attributes import Recipe, sample_recipes
from teacher import chat, extract_json, get_client

RAW_OUT = RAW_DIR / "synthetic_slang.raw.jsonl"

_SYSTEM = ("detailed thinking off\n"
           "You generate training data for a Gen Z/Alpha slang <-> English "
           "translator. Output ONLY a JSON object. No preamble, no markdown.")


def build_prompt(r: Recipe) -> tuple[str, str]:
    if r.is_hard_negative:
        twist = ("Make this a HARD NEGATIVE: write a sentence where the word "
                 f"'{r.term}' is used in its LITERAL, non-slang sense (not slang), "
                 "so the English side is a plain literal reading.")
    else:
        twist = (f"The slang sentence must naturally use the slang term '{r.term}' "
                 f"in its Gen Z sense, with a {r.tone} tone.")
    user = (
        f"Write ONE realistic short message and its translation.\n"
        f"Context: {r.context}. Difficulty: {r.difficulty}.\n"
        f"{twist}\n"
        "Return STRICT JSON with exactly two keys:\n"
        '{"slang": "<the Gen Z slang sentence>", '
        '"english": "<faithful plain-English meaning>"}\n'
        "Both must be single sentences, 3-30 words, no emojis-only."
    )
    return _SYSTEM, user


def generate_one(client, r: Recipe) -> dict | None:
    system, user = build_prompt(r)
    out = chat(client, user, system=system, temperature=0.7, max_tokens=300)
    j = extract_json(out)
    if not j or "slang" not in j or "english" not in j:
        return None
    return {
        "slang": str(j["slang"]).strip(),
        "english": str(j["english"]).strip(),
        "term": r.term, "tone": r.tone, "difficulty": r.difficulty,
        "context": r.context, "is_hard_negative": r.is_hard_negative,
        "direction_focus": r.direction,
    }


def _done_count() -> int:
    if not RAW_OUT.exists():
        return 0
    with open(RAW_OUT, encoding="utf-8") as f:
        return sum(1 for _ in f)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=SDG_TARGET)
    ap.add_argument("--pilot", action="store_true", help="generate 8, print, don't save")
    args = ap.parse_args(argv)

    client = get_client()

    if args.pilot:
        for r in sample_recipes(8, seed=999):
            try:
                rec = generate_one(client, r)
            except Exception as e:  # transient API error: show it, keep going
                print(f"  skip (api error): {e}")
                rec = None
            print(json.dumps(rec, ensure_ascii=False, indent=2))
            time.sleep(0.3)
        print("\nPILOT: read these — do the pairs fit the recipe? Then run full.")
        return 0

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    already = _done_count()
    recipes = sample_recipes(args.limit)[already:]  # resume where we left off
    print(f"{args.limit} target | {already} already generated | generating {len(recipes)}")
    with open(RAW_OUT, "a", encoding="utf-8") as f:
        for i, r in enumerate(recipes, 1):
            try:
                rec = generate_one(client, r)
            except Exception as e:  # transient API error (429/timeout/5xx): skip, keep going
                print(f"  skip (api error): {e}")
                rec = None
            f.write(json.dumps(rec or {"_failed": True}, ensure_ascii=False) + "\n")
            f.flush()
            time.sleep(SDG_REQUEST_SLEEP)
            if i % 50 == 0:
                print(f"  {i}/{len(recipes)} ...")
    print(f"\nDone. Raw file: {RAW_OUT}. Next: uv run python -m sdg.validate")
    return 0


if __name__ == "__main__":
    sys.exit(main())
