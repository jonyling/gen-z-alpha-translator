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
        "file": "genz_dataset_augmented_1500.csv",
        "slang_col": "slang_sentence",
        "english_col": "normal_sentence",
        # extra columns used to build a good, well-labelled eval set:
        "term_col": "slang_term",       # the slang word itself (answer key)
        "meaning_col": "meaning",       # plain-English meaning (answer key)
        "strat_col": "trend_status",    # trending / real / generated (for stratifying)
        "use_for_eval": True,           # hold some of these rows out as the test set
    },
    {
        "file": "genz_synthetic_dataset.csv",
        "slang_col": "input_text",
        "english_col": "target_text",
        "difficulty_col": "difficulty_level",  # easy / medium / hard
    },
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
