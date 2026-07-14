# User Manual — Slang ↔ English Translator

Two audiences:
- **A. Run it on Colab** (what your teammates do) — no local setup.
- **B. Develop locally** (with uv) — for editing data/prep before handing off.

---

## A. Run it on Google Colab (T4)

### One-time prep (do this early — takes 2 minutes)
1. **Hugging Face token** (free): sign up at https://huggingface.co →
   Settings → Access Tokens → create a token (read scope).
   > We use Unsloth's open mirror of Llama 3.2, so this usually isn't even
   > required — but having it avoids surprises. If you'd rather use Meta's
   > official model, also click "Agree" on https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct

### Steps
1. Upload the project. Either:
   - **Zip method:** zip the whole `slang prj` folder, upload it in Colab's file panel,
     then in a cell: `!unzip -q "slang prj.zip" -d /content/genz_slang_prj`
     (make sure the folder ends up containing `data/` and `src/`), **or**
   - **Drive method:** put the folder in Google Drive; the notebook has a commented
     `drive.mount` block — uncomment it and set `PROJECT_DIR`.
2. Open `train_genz_translator.ipynb` in Colab.
3. `Runtime → Change runtime type → T4 GPU`.
4. In cell 1, set `PROJECT_DIR` to where you put the folder.
5. (Optional) Add your token: left sidebar → 🔑 Secrets → name it `HF_TOKEN`.
6. `Runtime → Run all`. Go top to bottom.

### What each section does
| Section | What happens | Time (T4) |
|---|---|---|
| 1 Setup | installs packages, finds project, HF login | ~3–5 min |
| 2 Data | builds train/eval from `data/` | <1 min |
| 3 Model | loads Llama 3.2 3B (4-bit) | ~2 min |
| 4 Baseline | base model answers the 60 eval items ("before") | ~2 min |
| 5 Train | QLoRA fine-tune (`TRAIN_SAMPLE=6000` default) | ~15–25 min |
| 6 Tuned | tuned model answers the same 60 items ("after") | ~2 min |
| 7 Grading | exports `results/grading_sheet.csv` | instant |
| 8 Judge | *(optional)* 8B model auto-grades — skip if short on time | ~10 min |
| 9 Try it | type your own sentence | instant |
| 10 Chat app | *(optional)* Gradio chat box + public share link for a live demo | ~1 min |

> **First run tip:** keep `TRAIN_SAMPLE = 6000` for a fast end-to-end pass. Once you've
> seen it work, raise it (or set `None` for all ~31k) and retrain for the final numbers.

---

## B. Develop AND train locally (uv + your own NVIDIA GPU)

The uv environment installs the full GPU stack (CUDA torch, unsloth, trl, etc.), so if you
have an NVIDIA GPU you can train locally — no Colab needed for dev.

```bash
uv sync                             # creates .venv + installs everything
uv run python src/prepare_data.py   # rebuild data/processed/
uv run python -c "import torch; print('GPU:', torch.cuda.get_device_name(0))"   # sanity check
```

**One-time Hugging Face login** (from a terminal):
```bash
uv run hf auth login                # paste your token; saved permanently
```
(Often not even required — we use unsloth's open mirror of Llama 3.2 — but this avoids surprises.)

**Run the notebook in VS Code:**
1. Open `train_genz_translator.ipynb`.
2. Kernel picker (top-right) → **Python Environments** → pick **`.venv`**.
3. Run cells top to bottom. It auto-detects "local", skips the Colab install, and uses your GPU.
   On a modern GPU it trains faster than the T4 and uses **bf16** automatically.

> **GPU note:** this project is pinned to the CUDA 12.8 (`cu128`) torch build for an
> RTX 5070 (Blackwell). If you set this up on a *different* machine with an older GPU/driver,
> you may need a different CUDA index — see `[tool.uv.sources]` in `pyproject.toml`.

If you don't have an NVIDIA GPU, do data work here and train on Colab (section A).

---

## Using the app later (WITHOUT re-running the notebook)

Once training has run once, an adapter is saved to `genz_lora_adapter/`. After that, launch
the chat app directly — no training, no Run All:
```bash
uv run python app.py
```
It loads the base model + your saved adapter and opens the Gradio chat box in your browser.
(Edit the last line of `app.py` to `app.launch(share=True)` if you want a public link to show others.)

### Nicer demo: the messaging-app mockup (real model)
For the presentation there's a chat-app style demo (a "Slangify" sender phone + auto-translate
receiver phone) backed by the real model:
```bash
uv run python serve.py           # loads the model, serves the page
# then open http://127.0.0.1:8010
```
Type in the sender box → the model slangifies it → the receiver auto-translates it back. Unclear
input makes it abstain. (Uses port 8010; set `PORT=xxxx` if that's taken. Same look/feel as the
plain `app.py`, just a fuller demo.) The page also opens offline as a static mockup — it just shows
"offline demo mode" and uses a placeholder transform until you start `serve.py`.

> Teammates need the `genz_lora_adapter/` folder to run `app.py` / `serve.py`. It is **not** in GitHub by
> default (it's a model file). Either: (a) each person trains once, or (b) share the adapter
> folder via Google Drive / Hugging Face, or (c) commit it (it's small, ~50 MB — ask and we can
> un-ignore it).

## How to grade (this is our PRIMARY metric)

1. After the notebook runs, open `results/grading_sheet.csv`.
2. **Two teammates** independently fill these columns with `1` (correct) or `0` (wrong):
   - `base_rater1`, `base_rater2` — judging the `base_output`
   - `tuned_rater1`, `tuned_rater2` — judging the `tuned_output`
   Judge against `reference` + `meaning`. "Correct" = conveys the right meaning
   (exact wording doesn't matter).
   > Tip: grade *blind* if you can — don't look at which column is base vs tuned
   > until after scoring, so you're not biased.
3. Save the CSV, re-run the **scoring cell** (section 7). It prints:
   - accuracy **base vs tuned**, per direction and overall
   - inter-rater agreement.

---

## For the Friday presentation (5–6 slides, everyone speaks)
1. **Problem** — bidirectional slang↔English; show the input/output tags.
2. **Gap** — one concrete base-model failure from section 4.
3. **What we did** — SFT / QLoRA on Llama 3.2 3B, and why SFT fits a knowledge+behaviour gap.
4. **Results** — base vs tuned accuracy table (from grading). Be honest about misses.
5. **Error analysis** — 1–2 things the tuned model still gets wrong, and why.
6. **Recommendation** — deploy / iterate / hold, defended.

## Submission (xsite, Friday 11:30 PM)
Zip: `train_genz_translator.ipynb` + `data/` + `results/` (filled grading sheet) + slides.

---

## Troubleshooting
- **CUDA out of memory** → lower `TRAIN_SAMPLE`, keep batch size 2, or lower `MAX_SEQ_LEN`.
- **`data/ not found`** → `PROJECT_DIR` is wrong; point it at the folder containing `data/` and `src/`.
- **Gated model / 401** → add your `HF_TOKEN` secret, or accept the Llama license on HF.
- **Slow training** → confirm `Runtime type = T4 GPU` (not CPU).
- **`trl` API errors** → the notebook pins `trl<0.9.0` for Unsloth compatibility; re-run the install cell.
