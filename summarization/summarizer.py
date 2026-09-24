"""
summarization/summarizer.py
────────────────────────────
Abstractive text summarization for the Offline Multilingual Lecture Assistant.

Two backends
────────────
  "onnx"         (default — on-device NPU path)
    Loads encoder + decoder ONNX files via ONNX Runtime.
    Uses QNN HTP Execution Provider when onnxruntime-qnn is installed and
    QnnHtp.dll is on PATH.  Falls back silently to CPU EP otherwise.
    Requires: scripts/export_summarizer.py to have run first.

  "transformers" (off-device dev/test fallback)
    Uses HuggingFace transformers pipeline directly.  No ONNX files needed.
    Downloads weights on first use; cached locally after that.

Model selection
───────────────
  "bart"  → facebook/bart-large-cnn            (English)
  "mt5"   → csebuetnlp/mT5_multilingual_XLSum  (45 languages)

  Auto-selected based on language: "en" → bart, anything else → mt5.
  Override via SUMMARY_MODEL env var or constructor argument.

Long-input handling
───────────────────
  The encoder is fixed at 512 tokens.  Long transcripts are split into
  word-count-based chunks, each chunk summarized independently, then the
  partial summaries are recursively reduced to one final summary.

ONNX model paths
────────────────
  Set in .env after running scripts/export_summarizer.py:
    SUMMARY_MODEL=bart
    SUMMARY_ENCODER_ONNX=models/bart/int8/encoder_model_compiled.onnx
    SUMMARY_DECODER_ONNX=models/bart/int8/decoder_model_compiled.onnx
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

from config.settings import MODELS_DIR


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

_HF_MODELS = {
    "bart": "facebook/bart-large-cnn",
    "mt5":  "csebuetnlp/mT5_multilingual_XLSum",
}

# BART uses BOS token 2 (</s>); mT5 uses 0 (<pad>)
_DECODER_START_TOKEN = {
    "bart": 2,
    "mt5":  0,
}
# EOS token IDs
_EOS_TOKEN = {
    "bart": 2,
    "mt5":  1,
}

# Encoder hidden dim (needed for decoder input shape)
_HIDDEN_DIM = {
    "bart": 1024,
    "mt5":  512,
}

MAX_INPUT_TOKENS   = 512    # encoder fixed length
MAX_SUMMARY_TOKENS = 150
MIN_SUMMARY_TOKENS = 40
WORDS_PER_CHUNK    = 400    # ~300–400 tokens; safe headroom under 512


# ─────────────────────────────────────────────────────────────────────────────
# ORT session factory (shared pattern with transcriber / translator)
# ─────────────────────────────────────────────────────────────────────────────

def _make_ort_session(onnx_path: str) -> "onnxruntime.InferenceSession":
    import onnxruntime as ort
    available = ort.get_available_providers()
    use_qnn   = "QNNExecutionProvider" in available

    if use_qnn:
        providers       = ["QNNExecutionProvider"]
        provider_options = [{
            "backend_path":                            "QnnHtp.dll",
            "htp_performance_mode":                    "burst",
            "enable_htp_fp16_precision":               "1",
            "htp_graph_finalization_optimization_mode": "3",
        }]
        print(f"  [ORT] QNN HTP EP  → {Path(onnx_path).name}")
    else:
        providers        = ["CPUExecutionProvider"]
        provider_options = [{}]
        print(f"  [ORT] CPU EP  → {Path(onnx_path).name}")

    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.intra_op_num_threads = 4
    return ort.InferenceSession(
        onnx_path,
        sess_options=opts,
        providers=providers,
        provider_options=provider_options,
    )


# ─────────────────────────────────────────────────────────────────────────────
# ONNX backend
# ─────────────────────────────────────────────────────────────────────────────

class _ONNXSummarizer:
    """
    Encoder-decoder summarization via ONNX Runtime + optimum's
    ORTModelForSeq2SeqLM.

    Why ORTModelForSeq2SeqLM instead of raw ORT sessions?
      The three-graph seq2seq pattern (encoder, decoder_first_step,
      decoder_with_past) requires carefully threading KV-cache tensors between
      steps.  optimum already implements this correctly and is tested against
      the HuggingFace generate() API.  The QNN EP is activated through the
      provider_options at session creation inside ORTModel.
    """

    def __init__(
        self,
        model_key:       str,
        encoder_path:    Optional[str],
        decoder_path:    Optional[str],
        max_new_tokens:  int,
        min_new_tokens:  int,
        num_beams:       int,
    ) -> None:
        self.model_key      = model_key
        self.max_new_tokens = max_new_tokens
        self.min_new_tokens = min_new_tokens
        self.num_beams      = num_beams
        hf_id = _HF_MODELS[model_key]

        try:
            from optimum.onnxruntime import ORTModelForSeq2SeqLM
            from transformers import AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "optimum[onnxruntime] required:\n"
                f"  pip install optimum[onnxruntime]==1.19.2\n"
                f"  Original: {exc}"
            ) from exc

        # Resolve model directory from the encoder path
        model_dir = Path(encoder_path).parent if encoder_path else None

        if model_dir and model_dir.exists() and (model_dir / "encoder_model.onnx").exists():
            print(f"[Summarizer ONNX] Loading {model_key.upper()} from {model_dir} …")
            self._model = ORTModelForSeq2SeqLM.from_pretrained(
                str(model_dir), use_cache=True
            )
            # Tokenizer lives one level up (the fp32 export dir, not int8/)
            tok_dir = model_dir.parent if model_dir.name == "int8" else model_dir
            self._tokenizer = AutoTokenizer.from_pretrained(str(tok_dir))
        else:
            print(f"[Summarizer ONNX] Local files not found — downloading {hf_id} …")
            self._model = ORTModelForSeq2SeqLM.from_pretrained(
                hf_id, export=True, use_cache=True
            )
            self._tokenizer = AutoTokenizer.from_pretrained(hf_id)

        print(f"[Summarizer ONNX] Ready.")

    def summarize_chunk(self, text: str) -> str:
        import torch
        enc = self._tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_INPUT_TOKENS,
            padding="max_length",
        )
        with torch.inference_mode():
            out = self._model.generate(
                **enc,
                num_beams=self.num_beams,
                max_new_tokens=self.max_new_tokens,
                min_new_tokens=self.min_new_tokens,
                no_repeat_ngram_size=3,
                early_stopping=True,
            )
        return self._tokenizer.decode(out[0], skip_special_tokens=True).strip()


# ─────────────────────────────────────────────────────────────────────────────
# Transformers backend
# ─────────────────────────────────────────────────────────────────────────────

class _TransformersSummarizer:
    """HuggingFace pipeline — off-device dev/test fallback."""

    def __init__(
        self,
        model_key:      str,
        max_new_tokens: int,
        min_new_tokens: int,
    ) -> None:
        from transformers import pipeline
        hf_id = _HF_MODELS[model_key]
        print(f"[Summarizer] Loading {hf_id} …")
        self._pipeline = pipeline(
            "summarization",
            model=hf_id,
            tokenizer=hf_id,
            device=-1,           # CPU; change to 0 for GPU
        )
        self.max_new_tokens = max_new_tokens
        self.min_new_tokens = min_new_tokens
        print("[Summarizer] Ready.")

    def summarize_chunk(self, text: str) -> str:
        out = self._pipeline(
            text,
            max_length=self.max_new_tokens,
            min_length=self.min_new_tokens,
            truncation=True,
            do_sample=False,
        )
        return out[0]["summary_text"].strip()


# ─────────────────────────────────────────────────────────────────────────────
# Public Summarizer class
# ─────────────────────────────────────────────────────────────────────────────

class Summarizer:
    """
    Condenses a long lecture transcript into a concise summary.

    backend = "onnx"          → QNN NPU → CPU EP fallback (on-device)
    backend = "transformers"  → HuggingFace pipeline (off-device dev/test)

    language determines which model is loaded:
      "en"  → BART-large-cnn
      other → mT5_multilingual_XLSum
    Override model selection with SUMMARY_MODEL env var or model_key argument.

    Example
    ───────
        s = Summarizer(backend="onnx", language="en")
        print(s.summarize(long_transcript))

        # Multilingual
        s2 = Summarizer(backend="transformers", language="hi")
        print(s2.summarize(hindi_transcript))
    """

    def __init__(
        self,
        language:        str = "en",
        backend:         str = "onnx",
        model_key:       Optional[str] = None,
        max_summary_tokens: int = MAX_SUMMARY_TOKENS,
        min_summary_tokens: int = MIN_SUMMARY_TOKENS,
        num_beams:       int = 4,
        encoder_path:    Optional[str] = None,
        decoder_path:    Optional[str] = None,
    ) -> None:
        self.language  = language
        self.backend   = backend

        # Auto-select model based on language
        env_model  = os.getenv("SUMMARY_MODEL", "")
        if model_key:
            self.model_key = model_key
        elif env_model in _HF_MODELS:
            self.model_key = env_model
        else:
            self.model_key = "bart" if language == "en" else "mt5"

        # Resolve ONNX paths from env or defaults
        _enc = encoder_path or os.getenv(
            "SUMMARY_ENCODER_ONNX",
            str(MODELS_DIR / self.model_key / "int8" / "encoder_model_compiled.onnx")
        )
        _dec = decoder_path or os.getenv(
            "SUMMARY_DECODER_ONNX",
            str(MODELS_DIR / self.model_key / "int8" / "decoder_model_compiled.onnx")
        )

        if backend == "onnx":
            # Fall back to transformers if compiled ONNX files don't exist yet
            enc_exists = Path(_enc).exists()
            # Also accept non-compiled fp32 export
            fp32_enc = MODELS_DIR / self.model_key / "encoder_model.onnx"
            if not enc_exists and fp32_enc.exists():
                _enc = str(fp32_enc)
                _dec = str(MODELS_DIR / self.model_key / "decoder_model.onnx")
                enc_exists = True

            if enc_exists:
                self._impl = _ONNXSummarizer(
                    self.model_key, _enc, _dec,
                    max_summary_tokens, min_summary_tokens, num_beams,
                )
            else:
                print(
                    f"[Summarizer] ONNX files not found at {_enc}\n"
                    f"  Falling back to transformers backend.\n"
                    f"  Run: python scripts/export_summarizer.py --export-only"
                )
                self.backend = "transformers"
                self._impl = _TransformersSummarizer(
                    self.model_key, max_summary_tokens, min_summary_tokens
                )

        elif backend == "transformers":
            self._impl = _TransformersSummarizer(
                self.model_key, max_summary_tokens, min_summary_tokens
            )
        else:
            raise ValueError(
                f"Unknown backend {backend!r}. Choose 'onnx' or 'transformers'."
            )

    # ── Public API ────────────────────────────────────────────────────────────

    def summarize(self, text: str) -> str:
        """
        Summarize an arbitrarily long transcript.

        Long texts are split into WORDS_PER_CHUNK-word chunks, each summarized
        independently, then the partial summaries are merged and re-summarized
        until only one summary remains (hierarchical reduction).

        Returns a single summary string.
        """
        text = text.strip()
        if not text:
            return ""
        chunks = self._chunk_text(text)
        if len(chunks) == 1:
            return self._impl.summarize_chunk(chunks[0])

        # First pass — summarize each chunk
        partials = [self._impl.summarize_chunk(c) for c in chunks]
        combined = " ".join(partials)

        # Second pass — if the combined partials still exceed chunk size, recurse
        if len(combined.split()) > WORDS_PER_CHUNK:
            return self.summarize(combined)
        return self._impl.summarize_chunk(combined)

    # ── Internal ──────────────────────────────────────────────────────────────

    @staticmethod
    def _chunk_text(text: str, words_per_chunk: int = WORDS_PER_CHUNK) -> list[str]:
        words = text.split()
        return [
            " ".join(words[i : i + words_per_chunk])
            for i in range(0, max(1, len(words)), words_per_chunk)
        ]
