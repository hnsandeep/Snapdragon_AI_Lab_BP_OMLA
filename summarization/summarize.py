"""summarization/summarize.py — GenieX local LLM summarization on Snapdragon NPU.

Model: Qwen/Qwen3-0.6B-GGUF  @ Q4_0  (runs on Hexagon NPU via llama.cpp backend)
Fallback (non-Snapdragon): same model on CPU via device_map="cpu"

Install (choose ONE — they are mutually exclusive packages):
    On Snapdragon X Elite:   pip install geniex-llama-cpp
    Off-device dev/CI:       pip install geniex-llama-cpp   (still works on CPU)

Set GENIEX_MODEL in .env to override the default model alias.
Set GENIEX_DEVICE_MAP to "cpu" / "gpu" / "npu" / "auto" (default: "npu").
"""
from __future__ import annotations
import os, sys, threading
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:
    pass

# ── System prompt (embedded exactly as specified) ─────────────────────────────
SYSTEM_PROMPT = (
    "You are summarizing a lecture or meeting transcript for someone who could not "
    "attend. Be faithful to the source — do not invent facts, names, or numbers not "
    "present in the transcript. If a section is unclear or incomplete, say so rather "
    "than guessing. Output in this exact format: "
    "## Key Points (max 5 bullets) ## Action Items (or 'None mentioned') ## Summary "
    "(120 words max, plain language)."
)

_DEFAULT_MODEL   = os.getenv("GENIEX_MODEL",       "Qwen/Qwen3-0.6B-GGUF")
_DEFAULT_DEVICE  = os.getenv("GENIEX_DEVICE_MAP",  "npu")
_DEFAULT_PREC    = os.getenv("GENIEX_PRECISION",   "Q4_0")
_MAX_NEW_TOKENS  = int(os.getenv("GENIEX_MAX_TOKENS", "512"))

# ── singleton ─────────────────────────────────────────────────────────────────
_llm       = None
_llm_lock  = threading.Lock()


def _load_model() -> None:
    global _llm
    if _llm is not None:
        return
    try:
        from geniex import AutoModelForCausalLM
    except ImportError as e:
        raise ImportError(
            "GenieX is not installed.\n"
            "  On Snapdragon X Elite:  pip install geniex-llama-cpp\n"
            "  Off-device / CI:        pip install geniex-llama-cpp\n"
            f"  Original error: {e}"
        ) from e

    print(f"[GenieX] Loading {_DEFAULT_MODEL} @ {_DEFAULT_PREC} on {_DEFAULT_DEVICE} …")
    _llm = AutoModelForCausalLM.from_pretrained(
        _DEFAULT_MODEL,
        device_map=_DEFAULT_DEVICE,
        precision=_DEFAULT_PREC,
    )
    print("[GenieX] Model ready.")


# ── public API ────────────────────────────────────────────────────────────────

def summarize(transcript: str, stream: bool = False,
              on_token: "Callable[[str], None] | None" = None) -> str:
    """Run the transcript through the local GenieX LLM and return structured output.

    Args:
        transcript: Full translated (or source-language) transcript text.
        stream:     If True, tokens are streamed to *on_token* callback as they
                    arrive; the full text is still returned at the end.
        on_token:   Called with each decoded token string when stream=True.

    Returns:
        Structured string containing ## Key Points, ## Action Items, ## Summary.
        All inference runs on-device (Hexagon NPU via llama.cpp Q4_0).
        Zero network calls are made.
    """
    if not transcript.strip():
        return "## Key Points\n- (no transcript)\n## Action Items\nNone mentioned\n## Summary\nNo transcript was provided."

    with _llm_lock:
        _load_model()

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": f"Transcript:\n\n{transcript.strip()}"},
    ]
    prompt = _llm.tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, enable_thinking=False
    )

    if stream and on_token is not None:
        streamer = _llm.generate(prompt, max_new_tokens=_MAX_NEW_TOKENS,
                                 temperature=0.1, top_p=0.9,
                                 repetition_penalty=1.1, stream=True)
        chunks: list[str] = []
        for tok in streamer:
            on_token(tok)
            chunks.append(tok)
        result = streamer.output.text if streamer.output else "".join(chunks)
    else:
        output = _llm.generate(
            prompt,
            max_new_tokens=_MAX_NEW_TOKENS,
            temperature=0.1,
            top_p=0.9,
            repetition_penalty=1.1,
        )
        result = output.text

    return result.strip()


def close() -> None:
    """Release the GenieX model handle and free NPU resources."""
    global _llm
    with _llm_lock:
        if _llm is not None:
            _llm.close()
            _llm = None
