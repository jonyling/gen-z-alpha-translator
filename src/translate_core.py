"""One shared translation function used by the notebook, app.py, and serve.py.

Keeping generation in ONE place means the eval, the chat app, and the demo server
all decode identically — the number you measure matches what the demo shows, and
the anti-repetition settings apply everywhere (not just one surface).
"""

from __future__ import annotations

from config import TAG_TO_ENGLISH, TAG_TO_SLANG, ABSTAIN_MESSAGE
from abstain import looks_unclear

# Shared decoding settings. repetition_penalty + no_repeat_ngram_size stop the
# "letaletaleta…" loops the slang->English direction is prone to.
# Cap length so the model can be a bit verbose without launching into a lecture.
GEN_KWARGS = dict(
    max_new_tokens=60,
    use_cache=True,
    do_sample=False,
    repetition_penalty=1.3,
    no_repeat_ngram_size=3,
)

# Appended to every user turn: allow multi-sentence translations, but no meta-talk.
TRANSLATE_ONLY = "No explanation."


def generate_translation(model, tokenizer, tag: str, text: str) -> str:
    """Run the model for one (tag, text) prompt and return the decoded reply."""
    enc = tokenizer.apply_chat_template(
        [{"role": "user", "content": f"{tag}\n{text}\n\n{TRANSLATE_ONLY}"}],
        tokenize=True, add_generation_prompt=True,
        return_tensors="pt", return_dict=True,
    )
    # Most tokenizers return a dict-like (input_ids + attention_mask); older ones
    # may return the ids tensor directly. Handle both so a version bump can't break
    # all three surfaces at once.
    if hasattr(enc, "keys"):
        input_ids = enc["input_ids"].to("cuda")
        attention_mask = enc["attention_mask"].to("cuda") if "attention_mask" in enc else None
    else:
        input_ids = enc.to("cuda")
        attention_mask = None
    out = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        pad_token_id=tokenizer.eos_token_id,
        **GEN_KWARGS,
    )
    return tokenizer.decode(out[0][input_ids.shape[1]:], skip_special_tokens=True).strip()


def translate(model, tokenizer, text: str, direction: str, use_guard: bool = False) -> str:
    """High-level translate. direction: 'to_slang' | 'to_english'.

    use_guard=True (app/server) declines clearly-unclear input via looks_unclear
    instead of hallucinating. The eval calls with use_guard=False to measure the
    raw model.
    """
    if use_guard and looks_unclear(text):
        return ABSTAIN_MESSAGE
    tag = TAG_TO_SLANG if direction == "to_slang" else TAG_TO_ENGLISH
    return generate_translation(model, tokenizer, tag, text)
