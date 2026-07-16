"""QLoRA retrain — same recipe as train_genz_translator.ipynb §5.

Loads the 4-bit Llama 3.2 3B base, trains LoRA adapters on
data/processed/train.jsonl, saves to genz_lora_adapter/.

Usage:
    uv run python src/train_lora.py
    uv run python src/train_lora.py --sample 6000   # faster smoke
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import PROJECT_ROOT, TRAIN_PATH  # noqa: E402

MAX_SEQ_LEN = 1024
BASE_MODEL = "unsloth/Llama-3.2-3B-Instruct-bnb-4bit"
ADAPTER_DIR = PROJECT_ROOT / "genz_lora_adapter"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Use only first N shuffled train rows (default: full train.jsonl)",
    )
    parser.add_argument(
        "--backup",
        action="store_true",
        default=True,
        help="Backup existing adapter to genz_lora_adapter.bak before overwrite",
    )
    args = parser.parse_args()

    if not TRAIN_PATH.exists():
        print(f"ERROR: {TRAIN_PATH} missing. Run: uv run python src/prepare_data.py")
        return 1

    import torch
    from datasets import Dataset

    # Import Unsloth BEFORE trl so SFTTrainer/SFTConfig are the patched classes.
    from unsloth import FastLanguageModel
    from trl import SFTConfig, SFTTrainer

    print(">> loading base model (4-bit)…")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL,
        max_seq_length=MAX_SEQ_LEN,
        dtype=None,
        load_in_4bit=True,
    )
    # Keep the Instruct model's native chat template. Avoid Unsloth's
    # get_chat_template('llama-3.1') — it sets eos_token to a placeholder
    # '<EOS_TOKEN>' that trl>=0.24 rejects.
    if getattr(tokenizer, "eos_token", None) in (None, "<EOS_TOKEN>"):
        tokenizer.eos_token = "<|eot_id|>"
    if getattr(tokenizer, "pad_token", None) in (None, "<EOS_TOKEN>"):
        tokenizer.pad_token = tokenizer.eos_token
    print(">> eos_token:", repr(tokenizer.eos_token), "id=", tokenizer.eos_token_id)
    print(">> SFTTrainer:", SFTTrainer.__module__)

    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        lora_alpha=16,
        lora_dropout=0,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )

    with open(TRAIN_PATH, encoding="utf-8") as f:
        train_rows = [json.loads(line) for line in f if line.strip()]
    rows = train_rows if args.sample is None else train_rows[: args.sample]

    def to_text(ex):
        return {
            "text": tokenizer.apply_chat_template(
                ex["messages"], tokenize=False, add_generation_prompt=False
            )
        }

    ds = Dataset.from_list(rows).map(to_text)
    print(f">> training rows: {len(ds)}")

    use_bf16 = torch.cuda.is_bf16_supported()
    print(">> precision:", "bf16" if use_bf16 else "fp16")

    sft_args = SFTConfig(
        dataset_text_field="text",
        max_length=MAX_SEQ_LEN,  # trl>=0.24 renamed max_seq_length → max_length
        packing=False,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=50,
        num_train_epochs=1,
        learning_rate=2e-4,
        fp16=not use_bf16,
        bf16=use_bf16,
        logging_steps=25,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=42,
        output_dir=str(PROJECT_ROOT / "outputs"),
        report_to="none",
        eos_token="<|eot_id|>",
    )
    # Belt-and-suspenders: some Unsloth paths rewrite this to a placeholder.
    sft_args.eos_token = "<|eot_id|>"
    print(">> SFTConfig.eos_token:", repr(sft_args.eos_token))

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=ds,
        args=sft_args,
    )
    print(">> training…")
    trainer.train()

    if args.backup and ADAPTER_DIR.exists():
        bak = PROJECT_ROOT / "genz_lora_adapter.bak"
        if bak.exists():
            shutil.rmtree(bak)
        print(f">> backing up previous adapter → {bak}")
        shutil.move(str(ADAPTER_DIR), str(bak))

    ADAPTER_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(ADAPTER_DIR))
    tokenizer.save_pretrained(str(ADAPTER_DIR))
    print(">> saved adapter to", ADAPTER_DIR)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
