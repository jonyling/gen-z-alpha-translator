"""Shared Unsloth load helper for app.py / serve.py / grading refill."""

from __future__ import annotations

from pathlib import Path


def load_infer_model(model_name: str | Path, max_seq_length: int = 1024):
    """Load a 4-bit causal LM (base id or LoRA adapter folder) ready for generate()."""
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(model_name),
        max_seq_length=max_seq_length,
        dtype=None,
        load_in_4bit=True,
    )
    # Unsloth chat-template helpers sometimes leave a placeholder EOS that
    # breaks stopping; pin Llama-3 end-of-turn.
    if getattr(tokenizer, "eos_token", None) in (None, "<EOS_TOKEN>"):
        tokenizer.eos_token = "<|eot_id|>"
    if getattr(tokenizer, "pad_token", None) in (None, "<EOS_TOKEN>"):
        tokenizer.pad_token = tokenizer.eos_token
    FastLanguageModel.for_inference(model)
    return model, tokenizer
