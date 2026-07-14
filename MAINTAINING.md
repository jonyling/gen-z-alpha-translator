# Keeping the translator smart on new slang

Slang changes fast. The pipeline is built to refresh easily — here's the loop.

## The refresh loop
1. **Add new data** (any mix):
   - **Urban Dictionary refresh:** `uv run python src/fetch_urban.py`
     (resumable — only fetches terms not already in `data/dictionaries/urban_slang.csv`).
   - **New sentence-pair dataset:** drop the CSV in `data/raw/`, add one entry to `SOURCES` in `src/config.py`.
   - **New dictionary / emoji / trending words:** drop the CSV in `data/dictionaries/`, add one entry to `DICT_SOURCES` (or append rows to an existing dict).
2. **Rebuild the training data:** `uv run python src/prepare_data.py` (dedupes + mixes + shuffles).
3. **Retrain:** open `train_genz_translator.ipynb` → Run All (~15–25 min on an NVIDIA GPU).
   - Always retrain **from the base model** (the notebook does this) — do NOT stack a new
     LoRA on top of the old adapter, or quality drifts.
4. **Check for regressions:** the frozen `data/processed/eval.jsonl` gives a before/after number.
   For genuinely new slang, add fresh eval items periodically: delete `eval.jsonl` and re-run
   `prepare_data.py` to re-freeze (or extend the eval set), then re-grade.

## Data sources today
| Source | Type | Feeds |
|---|---|---|
| `data/raw/*.csv` (via `SOURCES`) | slang↔English sentence pairs | both-direction sentence training |
| `data/dictionaries/*.csv` (via `DICT_SOURCES`) | term→meaning + emoji→meaning | vocabulary training (both dirs; emoji one-way) |
| `urban_slang.csv` (via `fetch_urban.py`) | real crowd-sourced definitions | vocabulary training |

## For the commercial future (the flywheel)
- **Log misses:** capture inputs the model got wrong or abstained on → review → add as training
  data. That's the real "learns from usage" loop.
- **Automate:** schedule fetch → rebuild → retrain → eval, and deploy a new adapter **only if the
  eval score holds or improves**.
- **Version** each model + its eval results so a bad retrain can be rolled back.
- As usage grows, consider a larger base model and more data.

## Notes
- No content filtering on Urban Dictionary data (project decision — ship real usage as-is).
- Everything is reproducible: fixed seed in `prepare_data.py`, frozen eval, pinned deps in `uv.lock`.
