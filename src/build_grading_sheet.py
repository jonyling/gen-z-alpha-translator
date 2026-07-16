"""Build a clean post-Grok eval set + grading sheet (base vs tuned outputs).

Replaces the old gloss-heavy frozen eval with natural slang↔English pairs, then
fills results/grading_sheet.csv for human raters.

Modes:
  --held-out-api   (default) Ask Grok for NEW pairs not in train (fair eval)
  --from-existing  Sample quality rows from genz_grok_synthetic + genz_dataset
                   (faster; may overlap current train — note on sheet)

Usage:
    uv run python src/build_grading_sheet.py
    uv run python src/build_grading_sheet.py --from-existing
    uv run python src/build_grading_sheet.py --n-pairs 30 --skip-inference
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from abstain import make_abstain_eval_items  # noqa: E402
from config import (  # noqa: E402
    EVAL_PATH,
    EVAL_PER_DIRECTION,
    PROJECT_ROOT,
    RANDOM_SEED,
    RAW_DIR,
    TAG_TO_ENGLISH,
    TAG_TO_SLANG,
)
from generate_pairs import (  # noqa: E402
    DEFAULT_MODEL,
    XAI_BASE_URL,
    _load_env,
    call_grok,
    is_bad_pair,
    load_seed_examples,
    load_terms,
    normalize_pair,
)
from openai import OpenAI
from translate_core import generate_translation  # noqa: E402

RESULTS_DIR = PROJECT_ROOT / "results"
GRADING_CSV = RESULTS_DIR / "grading_sheet.csv"
BASE_MODEL = "unsloth/Llama-3.2-3B-Instruct-bnb-4bit"
ADAPTER_DIR = PROJECT_ROOT / "genz_lora_adapter"
MAX_SEQ_LEN = 1024


def _backup(path: Path, suffix: str = ".pre_grok_eval.bak") -> None:
    if path.exists():
        dest = path.with_name(path.name + suffix)
        shutil.copy2(path, dest)
        print(f"  backed up {path.name} → {dest.name}")


def pairs_from_existing(rng: random.Random, n: int) -> list[dict]:
    """Quality-filter existing Grok + clean CSVs into n eval pairs."""
    candidates: list[dict] = []

    grok = RAW_DIR / "genz_grok_synthetic.csv"
    if grok.exists():
        df = pd.read_csv(grok)
        for _, r in df.iterrows():
            s = str(r.get("slang_sentence") or "").strip()
            e = str(r.get("normal_sentence") or "").strip()
            m = str(r.get("meaning") or "").strip()
            t = str(r.get("slang_term") or "").strip()
            if is_bad_pair(s, e, m):
                continue
            # Prefer rows where English actually differs a lot from slang
            if len(set(s.lower().split()) & set(e.lower().split())) / max(
                1, len(s.split())
            ) > 0.85:
                continue
            candidates.append(
                {
                    "slang": s,
                    "english": e,
                    "term": t,
                    "meaning": m or t,
                    "strat": "grok",
                }
            )

    clean = RAW_DIR / "genz_dataset.csv"
    if clean.exists():
        df = pd.read_csv(clean)
        for _, r in df.iterrows():
            s = str(r.get("gen_z") or "").strip()
            e = str(r.get("normal") or "").strip()
            if is_bad_pair(s, e):
                continue
            candidates.append(
                {
                    "slang": s,
                    "english": e,
                    "term": "",
                    "meaning": "(see reference — natural paraphrase)",
                    "strat": "clean",
                }
            )

    # Dedupe by slang sentence
    seen: set[str] = set()
    uniq = []
    for c in candidates:
        key = c["slang"].lower()
        if key in seen:
            continue
        seen.add(key)
        uniq.append(c)

    rng.shuffle(uniq)
    # Prefer grok (has meaning) then clean
    uniq.sort(key=lambda c: 0 if c["strat"] == "grok" else 1)
    picked = uniq[:n]
    if len(picked) < n:
        raise SystemExit(f"Only {len(picked)} quality pairs available (need {n}).")
    rng.shuffle(picked)
    return picked


def pairs_held_out_api(rng: random.Random, n: int, model: str) -> list[dict]:
    """Ask Grok for brand-new pairs (not in existing train CSVs)."""
    _load_env()
    import os

    key = os.getenv("XAI_API_KEY")
    if not key:
        raise SystemExit("XAI_API_KEY missing in .env — use --from-existing or set the key.")

    client = OpenAI(api_key=key, base_url=XAI_BASE_URL)
    seeds = load_seed_examples(rng, n=8)
    terms = load_terms()
    rng.shuffle(terms)

    # Avoid terms already heavily used in the synthetic file so sentences feel fresh
    used_terms: set[str] = set()
    grok_path = RAW_DIR / "genz_grok_synthetic.csv"
    if grok_path.exists():
        df = pd.read_csv(grok_path)
        used_terms = {str(t).strip().lower() for t in df.get("slang_term", []) if str(t).strip()}

    term_pool = [t for t in terms if t["term"].lower() not in used_terms] or terms
    kept: list[dict] = []
    batch_size = 10
    i = 0
    attempts = 0
    max_attempts = 12

    print(f">> generating ~{n} held-out pairs via {model}…")
    while len(kept) < n and attempts < max_attempts:
        attempts += 1
        batch = term_pool[i : i + batch_size]
        i += batch_size
        if not batch:
            rng.shuffle(term_pool)
            i = 0
            batch = term_pool[:batch_size]
        try:
            raw, items = call_grok(client, model, seeds, batch, temperature=0.9)
        except Exception as e:
            print(f"  batch failed: {e}")
            continue
        for j, item in enumerate(items):
            fb = batch[j]["term"] if j < len(batch) else ""
            pair = normalize_pair(item, fallback_term=fb)
            if pair is None:
                continue
            kept.append(
                {
                    "slang": pair["slang_sentence"],
                    "english": pair["normal_sentence"],
                    "term": pair["slang_term"],
                    "meaning": pair["meaning"] or pair["slang_term"],
                    "strat": "grok_heldout",
                }
            )
            if len(kept) >= n:
                break
        print(f"  kept {len(kept)}/{n} (attempt {attempts})")

    if len(kept) < n:
        raise SystemExit(f"Only got {len(kept)} held-out pairs (need {n}). Re-run or use --from-existing.")
    return kept[:n]


def pairs_to_eval_items(pairs: list[dict]) -> list[dict]:
    items = []
    for idx, r in enumerate(pairs):
        items.append(
            {
                "id": f"e2e_{idx:03d}",
                "direction": "to_english",
                "type": "translate",
                "tag": TAG_TO_ENGLISH,
                "input": r["slang"],
                "reference": r["english"],
                "term": r["term"],
                "meaning": r["meaning"],
                "strat": r["strat"],
            }
        )
        items.append(
            {
                "id": f"e2s_{idx:03d}",
                "direction": "to_slang",
                "type": "translate",
                "tag": TAG_TO_SLANG,
                "input": r["english"],
                "reference": r["slang"],
                "term": r["term"],
                "meaning": r["meaning"],
                "strat": r["strat"],
            }
        )
    items.extend(make_abstain_eval_items())
    return items


def write_eval(items: list[dict]) -> None:
    EVAL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(EVAL_PATH, "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    print(f"Wrote {EVAL_PATH} ({len(items)} items)")


def rebuild_train() -> None:
    """Re-run prepare_data so train excludes the new eval texts."""
    import prepare_data

    print(">> rebuilding train.jsonl (excluding new eval)…")
    rc = prepare_data.main()
    if rc != 0:
        raise SystemExit(f"prepare_data failed with code {rc}")


def _load_model(model_name: str):
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=MAX_SEQ_LEN,
        dtype=None,
        load_in_4bit=True,
    )
    if getattr(tokenizer, "eos_token", None) in (None, "<EOS_TOKEN>"):
        tokenizer.eos_token = "<|eot_id|>"
    FastLanguageModel.for_inference(model)
    return model, tokenizer


def fill_outputs(items: list[dict], which: str) -> None:
    """which: 'base' | 'tuned'. Writes into items[*][f'{which}_output']."""
    import gc

    import torch

    name = BASE_MODEL if which == "base" else str(ADAPTER_DIR)
    print(f">> loading {which} model ({name})…")
    model, tokenizer = _load_model(name)
    key = f"{which}_output"
    for i, it in enumerate(items):
        out = generate_translation(model, tokenizer, it["tag"], it["input"])
        it[key] = out
        if (i + 1) % 10 == 0 or i + 1 == len(items):
            print(f"  {which}: {i + 1}/{len(items)}")
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def write_grading_sheet(items: list[dict], note: str) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for it in items:
        rows.append(
            {
                "id": it["id"],
                "type": it.get("type", "translate"),
                "direction": it["direction"],
                "strat": it.get("strat", ""),
                "input": it["input"],
                "reference": it["reference"],
                "slang_term": it.get("term", ""),
                "meaning": it.get("meaning", ""),
                "base_output": it.get("base_output", ""),
                "tuned_output": it.get("tuned_output", ""),
                "base_rater1": "",
                "base_rater2": "",
                "tuned_rater1": "",
                "tuned_rater2": "",
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(GRADING_CSV, index=False)
    # Also keep a dated alias so the old atrocious sheet stays distinguishable
    alias = RESULTS_DIR / "grading_sheet_post_grok.csv"
    df.to_csv(alias, index=False)
    notes_path = RESULTS_DIR / "GRADING_SHEET_NOTES.txt"
    notes_path.write_text(
        note + "\n\nHuman grading: fill base_rater1/2 and tuned_rater1/2 with 1 or 0.\n"
        "Then run the notebook §7 scoring cell, or score by hand for slides.\n",
        encoding="utf-8",
    )
    print(f"Wrote {GRADING_CSV}")
    print(f"Wrote {alias}")
    return GRADING_CSV


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-pairs", type=int, default=EVAL_PER_DIRECTION, help="Pairs per direction (default 30)")
    ap.add_argument(
        "--from-existing",
        action="store_true",
        help="Sample from existing Grok/clean CSVs instead of calling the API",
    )
    ap.add_argument("--skip-inference", action="store_true", help="Only freeze eval + rebuild train")
    ap.add_argument("--model", default=DEFAULT_MODEL, help="Grok model for --held-out-api")
    args = ap.parse_args()

    rng = random.Random(RANDOM_SEED)
    print("Backing up previous eval / grading sheet…")
    _backup(EVAL_PATH)
    _backup(GRADING_CSV)
    old_retrained = RESULTS_DIR / "grading_sheet_retrained.csv"
    _backup(old_retrained)

    if args.from_existing:
        pairs = pairs_from_existing(rng, args.n_pairs)
        note = (
            "Eval sampled from quality-filtered genz_grok_synthetic.csv + genz_dataset.csv.\n"
            "WARNING: many of these sentences were in the previous train.jsonl, so the CURRENT "
            "adapter may have seen them. Human grading of output quality is still useful; "
            "for fully fair auto metrics, retrain after this rebuild (train.jsonl was rewritten "
            "to exclude these eval texts)."
        )
    else:
        pairs = pairs_held_out_api(rng, args.n_pairs, args.model)
        # Save held-out pairs for audit / reuse
        holdout_csv = RAW_DIR / "genz_eval_heldout_grok.csv"
        pd.DataFrame(
            [
                {
                    "slang_term": p["term"],
                    "meaning": p["meaning"],
                    "slang_sentence": p["slang"],
                    "normal_sentence": p["english"],
                    "source": "grok_heldout",
                }
                for p in pairs
            ]
        ).to_csv(holdout_csv, index=False)
        print(f"Wrote {holdout_csv}")
        note = (
            "Eval = fresh Grok held-out pairs (not taken from the old gloss freeze).\n"
            "Sentences were generated for this sheet; train.jsonl was rebuilt to exclude them.\n"
            "The CURRENT adapter was trained before this holdout — fair for base vs tuned "
            "comparison on these items (tuned has not been trained on these exact sentences)."
        )

    items = pairs_to_eval_items(pairs)
    write_eval(items)
    rebuild_train()

    if args.skip_inference:
        write_grading_sheet(items, note + "\n(Inference skipped — outputs empty.)")
        return 0

    if not ADAPTER_DIR.exists():
        raise SystemExit(f"Missing adapter at {ADAPTER_DIR}")

    fill_outputs(items, "base")
    fill_outputs(items, "tuned")
    write_grading_sheet(items, note)

    # Preview a few rows
    print("\n--- sample rows ---")
    for it in items[:4]:
        print(f"{it['id']} [{it['direction']}]")
        print(f"  IN : {it['input'][:100]}")
        print(f"  REF: {it['reference'][:100]}")
        print(f"  BASE : {str(it.get('base_output', ''))[:100]}")
        print(f"  TUNED: {str(it.get('tuned_output', ''))[:100]}")
    print("\nDone. Two teammates: fill rater columns in results/grading_sheet.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
