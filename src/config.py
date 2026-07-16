"""Shared configuration: file paths, the direction tags, and the SOURCES mapping.

This is the ONE place teammates edit when adding a new dataset (see
data/README_BEFORE_UPLOAD.md and the design spec section 3a/3b).
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths. PROJECT_ROOT is the folder that contains this "src" directory.
# Works both locally and on Colab as long as the project folder is intact.
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
DICT_DIR = DATA_DIR / "dictionaries"
PROCESSED_DIR = DATA_DIR / "processed"

TRAIN_PATH = PROCESSED_DIR / "train.jsonl"
EVAL_PATH = PROCESSED_DIR / "eval.jsonl"

# ---------------------------------------------------------------------------
# The two directions. The tag is prepended to the input so ONE model can do
# both jobs. Keep these strings stable — training and inference must match.
# ---------------------------------------------------------------------------
TAG_TO_ENGLISH = "Translate to English:"
TAG_TO_SLANG = "Translate to Gen Z slang:"

# ---------------------------------------------------------------------------
# SOURCES: one entry per training file in data/raw/.
# To add a dataset: drop the file in data/raw/, add an entry here naming which
# column holds the slang text and which holds the plain-English text, then
# re-run prepare_data.py. Optional columns enrich the eval set / metadata.
# ---------------------------------------------------------------------------
SOURCES = [
    {
        "file": "genz_dataset.csv",
        "slang_col": "gen_z",
        "english_col": "normal",
    },
    {
        "file": "genz_grok_synthetic.csv",
        "slang_col": "slang_sentence",
        "english_col": "normal_sentence",
        "term_col": "slang_term",
        "meaning_col": "meaning",
    },
    {
        "file": "synthetic_slang.csv",
        "slang_col": "slang_sentence",
        "english_col": "normal_sentence",
        "hardneg_col": "is_hard_negative",
        # metadata columns (slang_term/tone/difficulty/is_hard_negative) are ignored
        # by load_source; kept in the CSV for provenance.
    },
    # Dropped gloss-heavy Kaggle sets (kept on disk for reference only).
    # {
    #     "file": "genz_synthetic_dataset.csv",
    #     "slang_col": "input_text",
    #     "english_col": "target_text",
    #     "difficulty_col": "difficulty_level",  # easy / medium / hard
    # },
]

# ---------------------------------------------------------------------------
# Dictionary sources (data/dictionaries/). Each term<->meaning entry becomes a
# short training example in BOTH directions, so the model learns vocabulary
# (helps the harder slang->English direction). Emoji entries are one-way
# (emoji -> meaning). These are curated/clean; we do NOT pull Urban Dictionary
# (crowd-sourced, often NSFW/joke, network-dependent).
# ---------------------------------------------------------------------------
DICT_SOURCES = [
    {"file": "all_slangs.csv",   "term_col": "Slang",       "meaning_col": "Description"},
    {"file": "gen_zz_words.csv", "term_col": "Word/Phrase", "meaning_col": "Definition"},
    {"file": "genz_slang.csv",   "term_col": "Word",        "meaning_col": "Meaning"},
    {"file": "genz_emojis.csv",  "term_col": "emoji",       "meaning_col": "Description", "emoji": True},
    # Real crowd-sourced definitions, fetched by src/fetch_urban.py (optional; only
    # used once the file exists). Run that script to (re)build it, then retrain.
    {"file": "urban_slang.csv",  "term_col": "term",        "meaning_col": "definition"},
]

# How many eval items to freeze PER DIRECTION (spec: ~30 each -> ~60 total).
EVAL_PER_DIRECTION = 30

# Reproducibility: fixed seed so the frozen eval set is identical every run.
RANDOM_SEED = 42

# ---------------------------------------------------------------------------
# Abstention: when the input isn't clear, the tool should say so instead of
# inventing a translation. (Matches the proposal's "unanswerable -> abstain;
# hallucination = failing to abstain" eval.)
# ---------------------------------------------------------------------------
ABSTAIN_MESSAGE = "I'm not sure how to translate that — the input isn't clear to me."

# Substrings that mark an output as an abstention (used to auto-detect it).
ABSTAIN_MARKERS = [
    "not sure", "isn't clear", "is not clear", "can't translate",
    "cannot translate", "don't understand", "do not understand", "unclear",
]

# How many synthetic 'unclear -> abstain' examples to add to TRAINING so a
# future retrain teaches the behavior. (Small vs the ~31k real examples.)
N_ABSTAIN_TRAIN = 300

# ---------------------------------------------------------------------------
# Teacher/judge provider (SDG stage 2 + DPO stage 4). Swappable so we're not
# locked to one API (NVIDIA's free tier was unresponsive). The key lives in
# .env under the provider's key env var — never hard-coded here.
#   groq   -> fast + free, hosts Llama 3.3 70B (get a key at console.groq.com)
#   nvidia -> build.nvidia.com
# ---------------------------------------------------------------------------
TEACHER_PROVIDER = "groq"   # "groq" | "nvidia"
_TEACHER_PROVIDERS = {
    "groq":   {"base_url": "https://api.groq.com/openai/v1",      "key_env": "GROQ_API_KEY",   "model": "llama-3.3-70b-versatile"},
    "nvidia": {"base_url": "https://integrate.api.nvidia.com/v1", "key_env": "NVIDIA_API_KEY", "model": "meta/llama-3.3-70b-instruct"},
}
_prov = _TEACHER_PROVIDERS[TEACHER_PROVIDER]
TEACHER_BASE_URL = _prov["base_url"]
TEACHER_KEY_ENV = _prov["key_env"]
TEACHER_MODEL = _prov["model"]
JUDGE_MODEL = TEACHER_MODEL

# Constraint-First SDG settings.
SDG_PATH = RAW_DIR / "synthetic_slang.csv"
SDG_TARGET = 1200                       # kept pairs to aim for
SDG_REQUEST_SLEEP = 2.1                  # seconds between teacher calls (Groq free tier ~30 req/min)
SDG_HARD_NEG_FRAC = 0.12                # fraction generated as "bait" hard negatives
SDG_DIRECTION_WEIGHTS = [("to_english", 0.65), ("to_slang", 0.35)]  # aim at weak dir
SDG_DIFFICULTY_WEIGHTS = [("clear", 0.5), ("ambiguous", 0.3), ("edge", 0.2)]
SDG_TONES = ["playful", "hype", "sarcastic", "deadpan", "annoyed",
             "affectionate", "dramatic", "chill"]
SDG_CONTEXTS = ["texting a friend", "group chat", "gaming voice chat",
                "social media caption", "replying to a post", "DM to a crush"]
