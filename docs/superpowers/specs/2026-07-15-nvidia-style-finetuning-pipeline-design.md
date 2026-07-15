# NVIDIA-Style Fine-Tuning Pipeline — Design Spec

**Date:** 2026-07-15
**Project:** Gen Z/Alpha slang ↔ English translator (Group 2)
**Goal:** Upgrade the existing translator by applying NVIDIA's 4-stage customization
methodology (Eval → SDG → SFT → DPO), aimed at the model's proven weak spot:
the **slang→English** direction.

---

## 1. Context & Decisions

This bolts onto the **existing** project — it does not replace it. Decisions locked
during brainstorming:

| Decision | Choice |
|---|---|
| Scope | **Upgrade the real project** (stays the course submission) |
| Teacher model | **NVIDIA API free tier** (build.nvidia.com, Nemotron-class) — course-native |
| Timeline | **Days** → lean, YAGNI; highest value / lowest risk first |
| Student model | **Keep Llama 3.2 3B** (Colab-T4-friendly for teammates) |
| DPO | **Built as a labeled experiment** (QLoRA-DPO on 3B), not a core deliverable |

**Why these:** the model's grading + demo already showed slang→English is weak and
rambly. Constraint-First synthetic data aimed at that direction is the highest-value,
lowest-risk lift. DPO is the *right* tool for the "rambly phrasing" gap in principle,
but NVIDIA found DPO fails to generalize below ~7–8B and their method needs
full-parameter training (~30 GB VRAM). This machine has **12 GB VRAM** (32 GB is
*system RAM*, which can't substitute for VRAM in training). So DPO runs as a QLoRA
experiment: a positive result is a bonus, a null result is honest error-analysis that
demonstrates methodology judgment.

**The core principle we honor:** eval-first → aim synthetic data at the *proven* gap →
retrain → re-eval on the frozen 70-item set. Every stage is gated by held-out eval
evidence, never by internal training loss.

---

## 2. Architecture

Existing backbone (unchanged, reused):
- `data/processed/eval.jsonl` — frozen 70-item scorecard for every stage.
- `src/config.py` + `src/prepare_data.py` — drop-in data folding (register a source, rebuild).
- Training notebook + `serve.py` — reused for retrains and testing.

New stage modules:

```
src/
  teacher.py           # shared NVIDIA API client (OpenAI-compatible) + LLM-as-judge helper
  eval_icl.py          # Stage 1: ICL ceiling test (glossary-in-prompt vs not)
  sdg/
    __init__.py
    attributes.py      # Constraint-First samplers (direction, term, tone, difficulty, context, hard-neg)
    generate.py        # call teacher -> {slang, english} pairs fitting sampled attrs (resumable, cached)
    validate.py        # parse/quality/leak/dedup checks -> data/raw/synthetic_slang.csv
  dpo/
    build_pairs.py     # on-policy rejected -> teacher-rewritten chosen -> judge -> delta gate
data/
  raw/synthetic_slang.csv     # Stage 2 output, committed, registered in SOURCES
  dpo/style_pairs.jsonl       # Stage 4 output, committed
train_dpo.ipynb               # Stage 4 QLoRA-DPO training notebook
.env                          # NVIDIA_API_KEY (gitignored, user-filled)
```

Data flow:

```
Stage 1  eval_icl.py ──proves──▶ "slang→English needs data / or prompting is enough"
Stage 2  sdg/ ──writes──▶ data/raw/synthetic_slang.csv ──(SOURCES)──▶ prepare_data.py
Stage 3  prepare_data.py rebuilds train.jsonl (synthetic folded in) ──▶ retrain ──▶ re-grade eval.jsonl
Stage 4  dpo/build_pairs.py (SFT model + teacher + judge) ──▶ style_pairs.jsonl ──▶ train_dpo.ipynb ──▶ re-eval
```

New dependencies: `openai` (NVIDIA endpoint is OpenAI-compatible), `python-dotenv`.

---

## 3. Stage 1 — ICL Ceiling Test (`src/eval_icl.py`)

**Purpose:** before spending training compute, prove whether the slang→English gap
needs *data* or can be closed by *prompting* — the cheaper intervention.

- Build a compact **slang glossary** (~40 `term = meaning` lines) from the dictionaries.
- Run **base Llama 3.2 3B** (no adapter) on the frozen 70-item eval **twice**:
  (a) normal prompt, (b) prompt **+ glossary block**.
- Score with the **NVIDIA LLM-as-judge**, broken out **per direction** (to_english / to_slang).
- **Output:** a small before/after table (avg + per-direction) + interpretation note.
- **Gate:** big to_english lift → prompting helps (document as cheap option); little lift
  → confirms training data is required (justifies Stage 2). Either result is slide material.

---

## 4. Stage 2 — Constraint-First SDG (`src/sdg/`)

**Principle:** sample the *recipe* (attributes) first; the teacher only writes content
that fits it. Labels come from us (the spec), so the teacher can't poison them.

### `attributes.py` — samplers (fixed seed), weighted at the gap
- **direction:** 65% slang→English, 35% English→slang
- **slang term:** drawn from the ~2,900-term dictionary pool
- **tone/register:** playful, hype, sarcastic, deadpan, annoyed, affectionate, …
- **difficulty:** 50% clear · 30% ambiguous · 20% edge
- **context:** texting a friend, group chat, gaming, social caption, reply, …
- **example type:** ~12% **hard negatives** ("bait" — looks like slang but used literally,
  or plain English that should not be over-slangified)

### `generate.py` — teacher call
- Uses `src/teacher.py` (NVIDIA API, OpenAI-compatible client), "detailed thinking off".
- Prompt embeds sampled attributes; requests **strict JSON** `{"slang": "...", "english": "..."}`
  where the slang side naturally uses the featured term with the given tone/context and the
  english side is a faithful plain-English translation.
- Temperature ~0.7 for variety; bounded `max_tokens`.
- **Resumable + cached** (skip already-generated), rate-limited for the free tier — modeled on `fetch_urban.py`.
- **Pilot gate:** generate **8 examples first, read them by hand**; only scale to ~1,000–1,500 after they look right.

### `validate.py` — quality mechanics
Reject a row unless it passes ALL:
- JSON parses; both fields non-empty; within length bounds (~3–200 chars).
- Featured slang term actually appears on the slang side (for term-featured rows).
- No preamble leak ("Sure, here is…").
- **Eval-leak guard:** reject if slang or english overlaps any frozen eval item (reuse `prepare_data`'s `banned_texts`).
- **Dedup:** exact + fuzzy (normalized string set; TF-IDF optional later).

Kept rows → `data/raw/synthetic_slang.csv` with columns matching an existing `SOURCES`
schema (`slang_sentence`, `normal_sentence`) plus metadata (`slang_term`, `tone`,
`difficulty`, `is_hard_negative`). Print a `generated → kept → rejected(by reason)` summary.

**Target:** ~1,000–1,500 clean pairs.

---

## 5. Stage 3 — Re-SFT + Re-Eval

- Register `synthetic_slang.csv` in `SOURCES` (drop-in design already supports this).
- `prepare_data.py` rebuilds `train.jsonl` (dedupe + both-direction expansion + leak guard covering synthetic).
- **Retrain** via the existing notebook (TRAIN_SAMPLE tuned to time budget).
- **Re-eval:** auto-judge for the fast read; **human grade** for the headline before/after.
- **Guardrail:** slang→English should rise **without** regressing the slang direction or the
  abstention behavior. Field-level (per-direction) comparison travels beside the aggregate.

---

## 6. Stage 4 — DPO Experiment (`src/dpo/build_pairs.py` + `train_dpo.ipynb`)

**Clearly labeled as an experiment.** Fixes "rambly/robotic" phrasing — a preference
between valid outputs, which is what DPO is for.

### Preference pairs (`build_pairs.py`)
- ~300 slang→English prompts.
- **rejected** = current SFT model's own on-policy output (must be the model's real failure).
- **chosen** = NVIDIA **teacher rewrite**: same meaning, natural + concise, no rambling.
- **judge** (NVIDIA) scores both on a 4-dim rubric (naturalness, conciseness, fluency, faithfulness; 1–5 each, max 20).
- **Gate:** keep a pair only if `chosen − rejected ≥ 3` **and** meaning preserved
  (faithfulness check). Print the funnel (candidates → dropped → kept).
- Output `data/dpo/style_pairs.jsonl` = `{prompt, chosen, rejected}`.

### Training (`train_dpo.ipynb`)
- TRL `DPOTrainer` + **QLoRA** (4-bit, fits 12 GB), starting from the SFT adapter.
- Reference model via PEFT (adapter-disabled base, no separate copy). `beta=0.1`, low LR, few steps.

### Eval
- Style rubric (judge) **and** translation-accuracy *drift* on the frozen eval.
- Report movement or null honestly. Expectation set low (3B is below where DPO reliably generalizes).

---

## 7. Cross-Cutting

- **Secrets:** `.env` (gitignored) holds `NVIDIA_API_KEY`; `teacher.py` reads via env; missing key → clear error. Key never enters chat or git.
- **Reproducibility:** fixed sampler seed; frozen eval; **commit** `synthetic_slang.csv` + `style_pairs.jsonl` so results reproduce without re-calling the API; log teacher model id/temperature + judge-prompt version.
- **Eval-first gating:** ICL result shapes SDG; re-grade decides whether DPO is worth attempting.
- **Human grading** stays the primary metric for headline numbers; auto-judge is for fast internal iteration only.
- **Cost:** free-tier rate limits handled by rate-limiting + resumability.

---

## 8. Slides Mapping

| Slide | Fed by |
|---|---|
| Problem | existing |
| Gap | Stage 1 ICL ceiling test |
| Method | Constraint-First SDG + retrain |
| Results (before→after) | baseline grading vs post-SDG grading on frozen eval |
| Error analysis | remaining failures + DPO experiment (positive or honest null) |
| Future work | 8B student for DPO, ongoing SDG refresh |

---

## 9. Manual Steps (user-only)

1. **Grade the current model** (2 raters, 1/0) — baseline for the before/after.
2. **Add the NVIDIA API key** to `.env` (obtained; not pasted into chat).
3. **Close Excel + rename** `grading_sheet_retrained.csv` over the original.
4. **Kick off / approve** retrains (or let them run headless).

Everything else (all stage scripts, SDG generation, data folding, DPO pair building,
auto-judge wiring, driving retrains) is automated.

---

## 10. Out of Scope

- Full-parameter DPO (needs ~30 GB VRAM; not feasible on 12 GB).
- Moving to an 8B student (documented as future work).
- NeMo Data Designer / Curator tooling (concepts applied; heavy tooling not required for this scale).
- Any change to the frozen eval set (kept stable for comparability).
