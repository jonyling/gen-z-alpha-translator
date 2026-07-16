"""Generate high-quality slang↔English pairs with xAI Grok, then auto-filter.

Pipeline:
  seed examples (clean + filtered) + slang terms (dicts/Urban)
    → Grok writes natural sentence pairs
    → gloss / quality filters
    → data/raw/genz_grok_synthetic.csv

Setup:
  1. Put your key in a project-root .env file:
       XAI_API_KEY=xai-...
  2. Install deps (once):
       uv sync
  3. Smoke test (1 cheap batch, ~10 pairs):
       uv run python src/generate_pairs.py --smoke
  4. Full run (default 3000 kept pairs):
       uv run python src/generate_pairs.py --target 3000

Safe to re-run: appends to the output CSV and skips terms already used.
Does NOT modify train.jsonl — add the new file to SOURCES in config.py later.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
from pathlib import Path

import pandas as pd
from openai import OpenAI

# Allow `uv run python src/generate_pairs.py` without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import DICT_DIR, DICT_SOURCES, PROJECT_ROOT, RAW_DIR, RANDOM_SEED  # noqa: E402

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    load_dotenv = None

XAI_BASE_URL = "https://api.x.ai/v1"
DEFAULT_MODEL = "grok-4.5"
DEFAULT_OUT = RAW_DIR / "genz_grok_synthetic.csv"
DEFAULT_RAW_OUT = RAW_DIR / "genz_grok_synthetic.raw.jsonl"  # every model reply, for audit

SYSTEM_PROMPT = """\
You write training pairs for a Gen Z / Alpha slang ↔ plain-English translator.

Hard rules:
- English must sound like a real person talking — NOT a dictionary gloss.
- NEVER write glossary English like: "extremely for real", "somewhat secretly",
  "amazing or excellent", "okay or agreement", "no lies", "visibly mad and frustrated".
- Do NOT paste a definition into a sentence template.
- Slang must be natural in context (not Mad Libs like "He is no cap af").
- Keep both sides roughly the same length and meaning.
- Output ONLY a JSON array (no markdown fences, no commentary).

Each item shape:
{
  "slang_term": "<term you were given>",
  "meaning": "<short plain meaning, a few words>",
  "slang_sentence": "<natural slang sentence using the term>",
  "english_sentence": "<natural plain-English paraphrase>"
}
"""

GLOSS_RE = re.compile(
    r"(?i)("
    r"\bextremely\b|"
    r"\bsomewhat secretly\b|"
    r"\bopenly and obviously\b|"
    r"\bend of discussion\b|"
    r"\bokay or agreement\b|"
    r"\bsituation or context\b|"
    r"\bamazing or excellent\b|"
    r"\baverage or mediocre\b|"
    r"\bbothered or upset\b|"
    r"\bstopped responding completely\b|"
    r"\ba very specific\b|"
    r"\bno lies\b|"
    r"\bvisibly mad\b|"
    r"\bmind absorbed\b|"
    r"\bone-sided emotional connection\b|"
    r"\bfor real energy\b"
    r")"
)

BAD_SLANG_FRAME = re.compile(
    r"(?i)("
    r"^that person has .+ today\.?$|"
    r"^he is [\w\'\- ]+ af\.?$|"
    r"^that fit is so [\w\'\- ]+\.?$|"
    r"^how do people stay this .+ all the time\?$|"
    r"^i can not believe how .+ that turned out\.?$|"
    r"^everyone on discord went full .+ when that happened\.?$"
    r")"
)


def _load_env() -> None:
    if load_dotenv is not None:
        load_dotenv(PROJECT_ROOT / ".env")
    # also allow a bare export
    if not os.getenv("XAI_API_KEY"):
        # tiny fallback parser if python-dotenv missing
        env_path = PROJECT_ROOT / ".env"
        if env_path.exists():
            for line in env_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                if k.strip() == "XAI_API_KEY" and v.strip():
                    os.environ.setdefault("XAI_API_KEY", v.strip().strip('"').strip("'"))


def _clean(text) -> str:
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return ""
    return str(text).strip()


def load_seed_examples(rng: random.Random, n: int = 8) -> list[dict]:
    """Few-shot examples from clean + filtered files."""
    seeds: list[dict] = []

    clean = RAW_DIR / "genz_dataset.csv"
    if clean.exists():
        df = pd.read_csv(clean)
        for _, r in df.iterrows():
            s, e = _clean(r.get("gen_z")), _clean(r.get("normal"))
            if s and e and s.lower() != e.lower():
                seeds.append(
                    {
                        "slang_term": "",
                        "meaning": "",
                        "slang_sentence": s,
                        "english_sentence": e,
                        "source": "clean",
                    }
                )

    filt = RAW_DIR / "genz_dataset_augmented_1500.filtered.csv"
    if filt.exists():
        df = pd.read_csv(filt)
        for _, r in df.iterrows():
            s, e = _clean(r.get("slang_sentence")), _clean(r.get("normal_sentence"))
            if not s or not e or s.lower() == e.lower():
                continue
            if is_bad_pair(s, e, _clean(r.get("meaning"))):
                continue
            seeds.append(
                {
                    "slang_term": _clean(r.get("slang_term")),
                    "meaning": _clean(r.get("meaning")),
                    "slang_sentence": s,
                    "english_sentence": e,
                    "source": "filtered",
                }
            )

    if not seeds:
        raise SystemExit(
            "No seed examples found. Need data/raw/genz_dataset.csv "
            "and/or genz_dataset_augmented_1500.filtered.csv"
        )
    rng.shuffle(seeds)
    return seeds[:n]


def load_terms() -> list[dict]:
    """Slang terms to prompt with (Urban + project dictionaries)."""
    terms: list[dict] = []
    seen: set[str] = set()

    def add(term: str, meaning: str, origin: str) -> None:
        t = _clean(term)
        m = _clean(meaning)
        if not t or len(t) < 2 or len(t) > 40:
            return
        # skip markup / junk Urban entries
        if re.search(r"[*_[\]{}<>]|^\W+$", t):
            return
        key = t.lower()
        if key in seen:
            return
        seen.add(key)
        terms.append({"term": t, "meaning": m, "origin": origin})

    for src in DICT_SOURCES:
        path = DICT_DIR / src["file"]
        if not path.exists():
            continue
        df = pd.read_csv(path)
        for _, r in df.iterrows():
            add(_clean(r.get(src["term_col"])), _clean(r.get(src["meaning_col"])), src["file"])

    # Prefer terms that look like real slang words for generation order:
    # longer multi-word phrases and modern-looking tokens first is optional;
    # we'll shuffle later.
    return terms


def is_bad_pair(slang: str, english: str, meaning: str = "") -> bool:
    if not slang or not english:
        return True
    if slang.lower() == english.lower():
        return True
    if len(slang.split()) < 3 or len(english.split()) < 3:
        return True
    if GLOSS_RE.search(english):
        return True
    if meaning and GLOSS_RE.search(meaning) and " or " in meaning.lower():
        return True
    if BAD_SLANG_FRAME.search(slang.strip()):
        return True
    if re.search(r"(?i)\baf\b", slang) and re.search(r"(?i)\bextremely\b", english):
        return True
    return False


def build_user_prompt(examples: list[dict], batch_terms: list[dict]) -> str:
    shots = []
    for ex in examples:
        shots.append(
            {
                "slang_term": ex.get("slang_term") or "(from sentence)",
                "meaning": ex.get("meaning") or "(implied)",
                "slang_sentence": ex["slang_sentence"],
                "english_sentence": ex["english_sentence"],
            }
        )
    term_lines = [
        f"- {t['term']}" + (f" (hint meaning: {t['meaning']})" if t["meaning"] else "")
        for t in batch_terms
    ]
    return (
        "Here are GOOD example pairs (style to imitate):\n"
        f"{json.dumps(shots, ensure_ascii=False, indent=2)}\n\n"
        "Write ONE natural pair for EACH of these slang terms "
        f"(exactly {len(batch_terms)} objects in a JSON array):\n"
        + "\n".join(term_lines)
        + "\n\nReminders: natural English, no dictionary-gloss phrasing, "
        "no markdown, JSON array only."
    )


def extract_json_array(text: str) -> list[dict]:
    text = (text or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    # direct parse
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return data
        if isinstance(data, dict) and "pairs" in data:
            return data["pairs"]
    except json.JSONDecodeError:
        pass
    # slice first [...] 
    m = re.search(r"\[.*\]", text, flags=re.S)
    if not m:
        raise ValueError("No JSON array in model response")
    data = json.loads(m.group(0))
    if not isinstance(data, list):
        raise ValueError("Parsed JSON is not a list")
    return data


def call_grok(
    client: OpenAI,
    model: str,
    examples: list[dict],
    batch_terms: list[dict],
    temperature: float,
) -> tuple[str, list[dict]]:
    resp = client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(examples, batch_terms)},
        ],
    )
    content = resp.choices[0].message.content or ""
    return content, extract_json_array(content)


def normalize_pair(item: dict, fallback_term: str = "") -> dict | None:
    slang = _clean(item.get("slang_sentence") or item.get("slang"))
    english = _clean(
        item.get("english_sentence")
        or item.get("normal_sentence")
        or item.get("english")
    )
    term = _clean(item.get("slang_term") or item.get("term") or fallback_term)
    meaning = _clean(item.get("meaning") or item.get("definition"))
    if is_bad_pair(slang, english, meaning):
        return None
    return {
        "slang_term": term,
        "meaning": meaning,
        "slang_sentence": slang,
        "normal_sentence": english,
        "source": "grok",
    }


def already_used_terms(out_path: Path) -> set[str]:
    if not out_path.exists():
        return set()
    df = pd.read_csv(out_path)
    if "slang_term" not in df.columns:
        return set()
    return {str(t).strip().lower() for t in df["slang_term"].dropna() if str(t).strip()}


def append_rows(out_path: Path, rows: list[dict]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    header = not out_path.exists()
    df.to_csv(out_path, mode="a", index=False, header=header)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--target", type=int, default=3000, help="Kept pairs to reach (default 3000)")
    parser.add_argument("--batch-size", type=int, default=25, help="Terms per Grok call")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="xAI model id")
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--seed-examples", type=int, default=8)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--raw-out", type=Path, default=DEFAULT_RAW_OUT)
    parser.add_argument("--smoke", action="store_true", help="One batch only (~batch-size pairs)")
    parser.add_argument("--sleep", type=float, default=0.4, help="Seconds between API calls")
    parser.add_argument("--max-batches", type=int, default=0, help="Optional hard cap on API calls")
    args = parser.parse_args()

    _load_env()
    api_key = os.getenv("XAI_API_KEY")
    if not api_key:
        print(
            "ERROR: XAI_API_KEY not set.\n"
            f"Create {PROJECT_ROOT / '.env'} with:\n"
            "  XAI_API_KEY=xai-your-key-here\n"
            "Get a key at https://console.x.ai/",
            file=sys.stderr,
        )
        return 1

    rng = random.Random(RANDOM_SEED)
    examples = load_seed_examples(rng, n=args.seed_examples)
    terms = load_terms()
    rng.shuffle(terms)

    used = already_used_terms(args.out)
    pending = [t for t in terms if t["term"].lower() not in used]
    existing = len(used)
    # existing file may have more rows than unique terms
    if args.out.exists():
        existing = len(pd.read_csv(args.out))

    print(f"Seeds for few-shot : {len(examples)}")
    print(f"Term pool          : {len(terms)} ({len(pending)} unused)")
    print(f"Output             : {args.out}")
    print(f"Already kept rows  : {existing}")
    print(f"Model              : {args.model}")
    target = args.batch_size if args.smoke else args.target
    if args.smoke:
        print("SMOKE MODE — one batch only")

    if existing >= target and not args.smoke:
        print(f"Already at target ({existing} >= {target}). Nothing to do.")
        return 0
    if not pending:
        print("No unused terms left in dictionaries. Add more terms or lower filters.")
        return 1

    client = OpenAI(api_key=api_key, base_url=XAI_BASE_URL)
    kept_total = existing
    batches = 0
    i = 0

    while kept_total < target and i < len(pending):
        if args.max_batches and batches >= args.max_batches:
            break
        batch = pending[i : i + args.batch_size]
        i += args.batch_size
        batches += 1
        print(f"\nBatch {batches}: {len(batch)} terms → Grok…")
        try:
            raw_text, items = call_grok(
                client, args.model, examples, batch, args.temperature
            )
        except Exception as e:
            print(f"  !! API/parse error: {e}")
            # still log raw if present
            args.raw_out.parent.mkdir(parents=True, exist_ok=True)
            with open(args.raw_out, "a", encoding="utf-8") as f:
                f.write(json.dumps({"error": str(e), "terms": [t["term"] for t in batch]}) + "\n")
            time.sleep(max(args.sleep, 1.0))
            if args.smoke:
                break
            continue

        with open(args.raw_out, "a", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    {
                        "terms": [t["term"] for t in batch],
                        "raw": raw_text,
                        "n_parsed": len(items),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

        kept_rows: list[dict] = []
        for idx, item in enumerate(items):
            if not isinstance(item, dict):
                continue
            fb = batch[idx]["term"] if idx < len(batch) else ""
            row = normalize_pair(item, fallback_term=fb)
            if row is None:
                continue
            kept_rows.append(row)

        if kept_rows:
            append_rows(args.out, kept_rows)
            kept_total += len(kept_rows)
        dropped = len(items) - len(kept_rows)
        print(f"  parsed={len(items)} kept={len(kept_rows)} filtered_out={max(0, dropped)}  total_kept={kept_total}")

        if args.smoke:
            break
        time.sleep(args.sleep)

    print("\n--- DONE ---")
    print(f"Kept pairs file : {args.out}")
    print(f"Raw audit log   : {args.raw_out}")
    print(f"Rows on disk    : {len(pd.read_csv(args.out)) if args.out.exists() else 0}")
    print(
        "\nNext:\n"
        "  1) Spot-check ~50 rows in the CSV.\n"
        "  2) Add to SOURCES in src/config.py, e.g.\n"
        '       {"file": "genz_grok_synthetic.csv",\n'
        '        "slang_col": "slang_sentence", "english_col": "normal_sentence",\n'
        '        "term_col": "slang_term", "meaning_col": "meaning"},\n'
        "  3) Prefer dropping/skipping the old gloss synthetic for retrain.\n"
        "  4) uv run python src/prepare_data.py   # rebuilds train.jsonl\n"
        "  5) Retrain in the notebook.\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
