"""Local demo server: the Slangify chat mockup, backed by the REAL model.

Loads the base model + your trained LoRA adapter once, serves the chat UI at
http://127.0.0.1:8000, and exposes a /api/translate endpoint the page calls so
you can type anything and see the actual fine-tuned model translate BOTH ways.

Usage:
    uv run python serve.py
Then open http://127.0.0.1:8000 in your browser.

Requires that training has been run once (so genz_lora_adapter/ exists).
"""

from pathlib import Path
import sys

from unsloth import FastLanguageModel
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
import uvicorn

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from config import TAG_TO_ENGLISH, TAG_TO_SLANG, ABSTAIN_MESSAGE  # noqa: E402
from abstain import looks_unclear  # noqa: E402

ADAPTER_DIR = ROOT / "genz_lora_adapter"
HTML_PATH = ROOT / "docs" / "slangify_mockup.html"
MAX_SEQ_LEN = 1024

if not ADAPTER_DIR.exists():
    sys.exit(f"No trained adapter at {ADAPTER_DIR}. Run the training notebook once first.")

print(">> loading model (base + LoRA adapter)…")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=str(ADAPTER_DIR), max_seq_length=MAX_SEQ_LEN, dtype=None, load_in_4bit=True,
)
FastLanguageModel.for_inference(model)
print(">> model ready")


def run_model(tag: str, text: str) -> str:
    msgs = [{"role": "user", "content": f"{tag}\n{text}"}]
    ids = tokenizer.apply_chat_template(
        msgs, tokenize=True, add_generation_prompt=True, return_tensors="pt"
    ).to("cuda")
    out = model.generate(input_ids=ids, max_new_tokens=80, use_cache=True, do_sample=False)
    return tokenizer.decode(out[0][ids.shape[1]:], skip_special_tokens=True).strip()


app = FastAPI()


class Req(BaseModel):
    text: str
    direction: str  # "to_slang" | "to_english"


@app.get("/", response_class=HTMLResponse)
def index():
    return HTML_PATH.read_text(encoding="utf-8")


@app.get("/api/health")
def health():
    return {"ok": True}


@app.post("/api/translate")
def translate(req: Req):
    text = (req.text or "").strip()
    # Abstention guard: decline clearly-unclear input instead of hallucinating.
    if looks_unclear(text):
        return JSONResponse({"output": ABSTAIN_MESSAGE, "abstained": True})
    tag = TAG_TO_SLANG if req.direction == "to_slang" else TAG_TO_ENGLISH
    return JSONResponse({"output": run_model(tag, text)})


if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", "8010"))
    print(f">> open http://127.0.0.1:{port}")
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")
