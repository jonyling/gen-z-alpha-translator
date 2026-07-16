# Gen Z / Alpha Slang ↔ English Translator

Fine-tunes **Llama 3.2 3B Instruct** with **QLoRA** into one model that translates *both*
directions, chosen by a tag in the prompt:

- `Translate to English:` → plain English
- `Translate to Gen Z slang:` → slang

Runs on a single **T4 GPU** (free Google Colab). Course mini-project, Group 2.

## Repo layout
```
slang prj/
├─ train_genz_translator.ipynb   ← the deliverable notebook (run on Colab T4)
├─ pyproject.toml                ← uv project (local dev)
├─ src/
│  ├─ config.py                  ← paths + SOURCES column mapping (edit to add data)
│  └─ prepare_data.py            ← builds train/eval from data/
├─ data/
│  ├─ raw/                       ← training-pair files (see README_BEFORE_UPLOAD.md)
│  ├─ dictionaries/              ← slang→meaning word lists (eval answer key)
│  ├─ processed/                 ← auto-generated train.jsonl + eval.jsonl
│  ├─ unused/                    ← files we deliberately skip (reasons in source info.csv)
│  └─ README_BEFORE_UPLOAD.md    ← checklist before adding a dataset
├─ results/                      ← grading_sheet.csv + metrics (created by the notebook)
└─ docs/superpowers/specs/       ← design spec
```

## Quick start (local dev, with uv)
```bash
uv sync                        # creates .venv + installs core deps (data prep, no GPU)
uv run python src/prepare_data.py
```
This writes `data/processed/train.jsonl` (~31k examples) and the frozen
`data/processed/eval.jsonl` (60 items). Deterministic — same output every run.

Training needs a GPU, so do it in the notebook on Colab (see `USER_MANUAL.md`).

## Run the demo (needs the trained adapter in `genz_lora_adapter/`)
```bash
uv run python serve.py    # Slangify two-phone chat mockup → http://127.0.0.1:8010
uv run python app.py      # plain Gradio chat box (makes a public share link)
```
`serve.py` prints its URL when ready (~30–60 s to load the model). If port 8010
is busy it auto-picks the next free port. Run ONE of the two at a time — each
loads its own copy of the model onto the GPU.

## Human grading (primary metric)
Teammates fill `1`/`0` in `results/grading_sheet_post_grok.csv`, then:
```bash
uv run python src/score_human.py
uv run python src/evaluate.py --skip-bert   # optional auto metrics
```

## Adding more data later
Drop a CSV/XLSX in `data/raw/`, add one entry to `SOURCES` in `src/config.py`,
re-run `prepare_data.py`. Full rules: `data/README_BEFORE_UPLOAD.md`.
