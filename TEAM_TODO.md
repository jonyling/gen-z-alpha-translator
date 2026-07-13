# Team TODO — Gen Z Slang Translator (due Fri 11:30 PM)

Repo: https://github.com/hm-base/gen-z-alpha-translator

## Where we are ✅
- Model **trained and working** (Llama 3.2 3B + QLoRA), both directions (slang↔English).
- Chat **app works**: `uv run python app.py` → open http://127.0.0.1:7860
- Data pipeline, notebook, eval set (60 items), and grading sheet are all done and in the repo.

## Get set up (everyone, once)
```bash
git clone https://github.com/hm-base/gen-z-alpha-translator.git
cd gen-z-alpha-translator
uv sync
uv run python app.py        # try the translator
```
(To train yourself you need an NVIDIA GPU, or use Google Colab with a T4 — see `USER_MANUAL.md`.
The trained model is already in the repo, so you can run the app without training.)

## Who does what

**Data & eval (grading — this is our main grade):**
- [ ] Two people open `results/grading_sheet.csv`.
- [ ] For every row put **1 (correct)** or **0 (incorrect)**:
  - BASE answer → `base_rater1`, `base_rater2`
  - TUNED answer → `tuned_rater1`, `tuned_rater2`
- [ ] Judge on *meaning* vs the `reference`/`meaning` columns (wording doesn't matter).
- [ ] Re-run the **scoring cell** (section 7) in the notebook → gives base-vs-tuned accuracy.

**Modelling:**
- [x] Training done (6,000-sample run). Optional: retrain on full data (won't change the story much).

**Analysis & write-up:**
- [ ] From the base-vs-tuned output: pick 1 clear BASE failure (the "gap") + 1–2 TUNED mistakes (error analysis).
- [ ] Note the key finding: **slang→English works well; English→slang is weaker** (open-ended task + data does word-substitution).
- [ ] Draft the recommendation: **ITERATE** (improve data quality), with reasons.

**Coordinator:**
- [ ] Build slides from `docs/slides_outline.md` (5–6 slides, everyone speaks).
- [ ] Drop in the accuracy numbers (slide 4) + the failure examples.
- [ ] Time the talk (5 min) — practice once.

## Submit (Friday, to xsite)
Zip: `train_genz_translator.ipynb` + `data/` + `results/` (filled grading sheet) + slides.
