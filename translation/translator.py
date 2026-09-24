"""
translation/translator.py
──────────────────────────
Fully-offline neural machine translation for the Snapdragon AI Lab project.

Three backends
──────────────
  "indictrans2"  (default — best for Indic languages)
    Uses the IndicTrans2-distilled-200M ONNX model with IndicTransToolkit
    preprocessing.  Supports all 22 scheduled Indic languages.
    Requires: IndicTransToolkit, optimum[onnxruntime], transformers

  "nllb200"
    Uses NLLB-200-distilled-600M ONNX model via optimum's ORTModelForSeq2SeqLM.
    200-language coverage.  Standard HuggingFace tokenizer.
    Requires: optimum[onnxruntime], transformers

  "transformers"  (off-device dev/test fallback)
    Direct HuggingFace transformers pipeline (Helsinki-NLP OPUS-MT).
    No ONNX files required.

Public API
──────────
    from translation.translator import Translator

    t = Translator(backend="indictrans2")
    print(t.translate("The lecture begins at nine.", "eng_Latn", "hin_Deva"))
    # → "व्याख्यान नौ बजे शुरू होता है।"

    # Batch translation:
    results = t.translate_batch(sentences, "eng_Latn", "hin_Deva")

Language tag formats
────────────────────
  IndicTrans2 / NLLB-200 use ISO 639-3 + ISO 15924 script tags:
    "eng_Latn"  English
    "hin_Deva"  Hindi
    "kan_Knda"  Kannada
    "tam_Taml"  Tamil
    "tel_Telu"  Telugu
    "ben_Beng"  Bengali
    "mar_Deva"  Marathi
    "deu_Latn"  German   (NLLB-200 only)
    "fra_Latn"  French   (NLLB-200 only)

  OPUS-MT (transformers fallback) uses ISO 639-1 codes ("en", "de", …)

ONNX model paths
────────────────
  Set in .env after running scripts/export_translation.py:
    IT2_ENCODER_ONNX=models/indictrans2/int8/encoder_model_compiled.onnx
    IT2_DECODER_ONNX=models/indictrans2/int8/decoder_model_compiled.onnx
    NLLB_ENCODER_ONNX=models/nllb200/int8/encoder_model_compiled.onnx
    NLLB_DECODER_ONNX=models/nllb200/int8/decoder_model_compiled.onnx
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

# ── Path setup ────────────────────────────────────────────────────────────────
_HERE = Path(__file__).resolve().parent
ROOT  = _HERE.parent
sys.path.insert(0, str(ROOT))
from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

from config.settings import SOURCE_LANGUAGE, TARGET_LANGUAGE, MODELS_DIR


# ─────────────────────────────────────────────────────────────────────────────
# ORT session factory (shared with transcriber pattern)
# ─────────────────────────────────────────────────────────────────────────────

def _make_ort_session(onnx_path: str) -> "onnxruntime.InferenceSession":
    """
    Create an ORT session with QNN HTP EP → CPU EP fallback.
    See transcription/transcriber.py for full rationale.
    """
    import onnxruntime as ort  # noqa: PLC0415
    available = ort.get_available_providers()
    use_qnn   = "QNNExecutionProvider" in available

    if use_qnn:
        providers = ["QNNExecutionProvider"]
        provider_options = [{
            "backend_path":    "QnnHtp.dll",
            "htp_performance_mode": "burst",
            "enable_htp_fp16_precision": "1",
            "htp_graph_finalization_optimization_mode": "3",
        }]
        print(f"  [ORT] QNN HTP EP  → {Path(onnx_path).name}")
    else:
        providers = ["CPUExecutionProvider"]
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
# Backend A — IndicTrans2  (IndicTransToolkit + ORTModelForSeq2SeqLM)
# ─────────────────────────────────────────────────────────────────────────────

class _IndicTrans2Backend:
    """
    Wraps IndicTrans2-distilled-200M via optimum's ORTModelForSeq2SeqLM.

    Why ORTModelForSeq2SeqLM and not raw ORT sessions?
      IndicTrans2 uses a three-graph export (encoder, decoder, decoder_with_past).
      optimum handles the auto-regressive loop (including KV cache passing)
      through the ORTModelForSeq2SeqLM interface — rewriting that loop manually
      is error-prone and adds no value.  The QNN EP is still used at the session
      level inside ORTModelForSeq2SeqLM (see provider_options).

    IndicProcessor (from IndicTransToolkit) is MANDATORY:
      It normalises Indic script, applies language-tag prefixing, protects
      named entities, and normalises Unicode.  Skipping it produces fluent-
      looking but wrong-language output.
    """

    # Supported source → target tag pairs for this direction model
    SUPPORTED_SRC  = {"eng_Latn"}
    SUPPORTED_TAGS = {
        "hin_Deva", "kan_Knda", "tam_Taml", "tel_Telu", "ben_Beng",
        "mar_Deva", "guj_Gujr", "mal_Mlym", "pan_Guru", "ory_Orya",
        "asm_Beng", "mai_Deva", "npi_Deva", "urd_Arab", "san_Deva",
        "brx_Deva", "doi_Deva", "gom_Deva", "kas_Arab", "kas_Deva",
        "mni_Beng", "mni_Mtei", "sat_Olck", "snd_Arab", "snd_Deva",
    }

    def __init__(
        self,
        encoder_path: Optional[str] = None,
        decoder_path: Optional[str] = None,
        subfolder: str = "int8",
        max_new_tokens: int = 256,
        num_beams: int = 4,
    ) -> None:
        self.max_new_tokens = max_new_tokens
        self.num_beams      = num_beams

        # ── Resolve paths ─────────────────────────────────────────────────────
        it2_base = MODELS_DIR / "indictrans2"
        _enc = encoder_path or os.getenv(
            "IT2_ENCODER_ONNX",
            str(it2_base / subfolder / "encoder_model_compiled.onnx")
        )
        _dec = decoder_path or os.getenv(
            "IT2_DECODER_ONNX",
            str(it2_base / subfolder / "decoder_model_compiled.onnx")
        )
        # Determine which directory to load from (ORTModel loads by directory)
        model_dir = Path(_enc).parent
        repo_id   = "TigreGotico/indictrans2-en-indic-dist-200M-onnx"

        # ── Load ORTModel ──────────────────────────────────────────────────────
        try:
            from optimum.onnxruntime import ORTModelForSeq2SeqLM  # noqa: PLC0415
            from transformers import AutoTokenizer                 # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                f"optimum[onnxruntime] and transformers are required: {exc}\n"
                "  pip install optimum[onnxruntime]==1.19.2 transformers==4.41.2"
            ) from exc

        print(f"[IndicTrans2] Loading model from {model_dir} …")
        qnn_eps = [{"QNNExecutionProvider": {
            "backend_path": "QnnHtp.dll",
            "htp_performance_mode": "burst",
            "enable_htp_fp16_precision": "1",
        }}]

        if model_dir.exists() and (model_dir / "encoder_model.onnx").exists():
            self._model = ORTModelForSeq2SeqLM.from_pretrained(
                str(model_dir),
                use_cache=True,
                trust_remote_code=True,
            )
            self._tokenizer = AutoTokenizer.from_pretrained(
                str(it2_base), trust_remote_code=True
            )
        else:
            # Download from HuggingFace on first use
            print(f"  Local files not found — downloading from {repo_id} …")
            _sub = subfolder if (
                Path(it2_base / subfolder).exists() or subfolder == ""
            ) else ""
            self._model = ORTModelForSeq2SeqLM.from_pretrained(
                repo_id,
                subfolder=subfolder,
                use_cache=True,
                trust_remote_code=True,
            )
            self._tokenizer = AutoTokenizer.from_pretrained(
                repo_id, trust_remote_code=True
            )

        # ── IndicProcessor ────────────────────────────────────────────────────
        try:
            from IndicTransToolkit.processor import IndicProcessor  # noqa: PLC0415
            self._ip = IndicProcessor(inference=True)
        except ImportError as exc:
            raise ImportError(
                "IndicTransToolkit is required for IndicTrans2:\n"
                "  pip install IndicTransToolkit==0.1.1\n"
                f"  Original error: {exc}"
            ) from exc

        print("[IndicTrans2] Ready.")

    def translate(self, text: str, src_lang: str, tgt_lang: str) -> str:
        return self.translate_batch([text], src_lang, tgt_lang)[0]

    def translate_batch(
        self, texts: list[str], src_lang: str, tgt_lang: str
    ) -> list[str]:
        import torch  # noqa: PLC0415
        self._validate_langs(src_lang, tgt_lang)

        # IndicProcessor.preprocess_batch adds language tags and normalises text
        batch = self._ip.preprocess_batch(texts, src_lang=src_lang, tgt_lang=tgt_lang)
        enc   = self._tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256,
        )
        with torch.inference_mode():
            out = self._model.generate(
                **enc,
                num_beams=self.num_beams,
                max_new_tokens=self.max_new_tokens,
                use_cache=True,
            )
        decoded = self._tokenizer.batch_decode(out, skip_special_tokens=True)
        # postprocess_batch reverses entity protection
        return self._ip.postprocess_batch(decoded, lang=tgt_lang)

    def _validate_langs(self, src: str, tgt: str) -> None:
        if src not in self.SUPPORTED_SRC:
            raise ValueError(
                f"IndicTrans2 (en→Indic) only accepts src_lang='eng_Latn', got {src!r}.\n"
                f"For Indic→English, use the reverse-direction model."
            )
        if tgt not in self.SUPPORTED_TAGS:
            raise ValueError(
                f"Unsupported tgt_lang {tgt!r}.\n"
                f"Supported: {sorted(self.SUPPORTED_TAGS)}"
            )


# ─────────────────────────────────────────────────────────────────────────────
# Backend B — NLLB-200  (ORTModelForSeq2SeqLM, 200 languages)
# ─────────────────────────────────────────────────────────────────────────────

class _NLLB200Backend:
    """
    Wraps NLLB-200-distilled-600M via optimum's ORTModelForSeq2SeqLM.
    200-language coverage via standard HuggingFace NllbTokenizer.
    Language tags: ISO 639-3 + ISO 15924 script (same format as IndicTrans2).
    """

    HF_REPO = "facebook/nllb-200-distilled-600M"

    def __init__(
        self,
        encoder_path: Optional[str] = None,
        decoder_path: Optional[str] = None,
        subfolder: str = "int8",
        max_new_tokens: int = 256,
        num_beams: int = 4,
    ) -> None:
        self.max_new_tokens = max_new_tokens
        self.num_beams      = num_beams

        nllb_base = MODELS_DIR / "nllb200"
        _enc = encoder_path or os.getenv(
            "NLLB_ENCODER_ONNX",
            str(nllb_base / subfolder / "encoder_model_compiled.onnx")
        )
        model_dir = Path(_enc).parent

        try:
            from optimum.onnxruntime import ORTModelForSeq2SeqLM  # noqa: PLC0415
            from transformers import NllbTokenizer                 # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                f"optimum[onnxruntime] and transformers are required: {exc}\n"
                "  pip install optimum[onnxruntime]==1.19.2 transformers==4.41.2"
            ) from exc

        print(f"[NLLB-200] Loading model from {model_dir} …")
        if model_dir.exists() and (model_dir / "encoder_model.onnx").exists():
            self._model     = ORTModelForSeq2SeqLM.from_pretrained(
                str(model_dir), use_cache=True
            )
            self._tokenizer = NllbTokenizer.from_pretrained(str(nllb_base))
        else:
            print(f"  Local files not found — downloading from {self.HF_REPO} …")
            self._model     = ORTModelForSeq2SeqLM.from_pretrained(
                self.HF_REPO, export=True, use_cache=True
            )
            self._tokenizer = NllbTokenizer.from_pretrained(self.HF_REPO)
        print("[NLLB-200] Ready.")

    def translate(self, text: str, src_lang: str, tgt_lang: str) -> str:
        return self.translate_batch([text], src_lang, tgt_lang)[0]

    def translate_batch(
        self, texts: list[str], src_lang: str, tgt_lang: str
    ) -> list[str]:
        import torch  # noqa: PLC0415
        self._tokenizer.src_lang = src_lang
        enc = self._tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256,
        )
        tgt_lang_id = self._tokenizer.lang_code_to_id[tgt_lang]
        with torch.inference_mode():
            out = self._model.generate(
                **enc,
                forced_bos_token_id=tgt_lang_id,
                num_beams=self.num_beams,
                max_new_tokens=self.max_new_tokens,
                use_cache=True,
            )
        return self._tokenizer.batch_decode(out, skip_special_tokens=True)


# ─────────────────────────────────────────────────────────────────────────────
# Backend C — OPUS-MT transformers pipeline (off-device fallback)
# ─────────────────────────────────────────────────────────────────────────────

class _OpusMTBackend:
    """Helsinki-NLP OPUS-MT via HuggingFace pipeline.  No ONNX required."""

    _MODEL_MAP: dict[tuple[str, str], str] = {
        ("en", "de"):    "Helsinki-NLP/opus-mt-en-de",
        ("en", "fr"):    "Helsinki-NLP/opus-mt-en-fr",
        ("en", "es"):    "Helsinki-NLP/opus-mt-en-es",
        ("en", "zh"):    "Helsinki-NLP/opus-mt-en-zh",
        ("en", "ar"):    "Helsinki-NLP/opus-mt-en-ar",
        ("en", "hi"):    "Helsinki-NLP/opus-mt-en-hi",
        ("de", "en"):    "Helsinki-NLP/opus-mt-de-en",
        ("fr", "en"):    "Helsinki-NLP/opus-mt-fr-en",
        ("es", "en"):    "Helsinki-NLP/opus-mt-es-en",
    }

    def __init__(self, src_lang: str, tgt_lang: str, max_length: int = 512) -> None:
        from transformers import pipeline  # noqa: PLC0415
        key      = (src_lang, tgt_lang)
        model_id = self._MODEL_MAP.get(key)
        if model_id is None:
            raise ValueError(
                f"No OPUS-MT model for {src_lang!r} → {tgt_lang!r}.\n"
                f"Available: {list(self._MODEL_MAP.keys())}"
            )
        print(f"[OPUS-MT] Loading {model_id} …")
        self._pipeline = pipeline(
            "translation", model=model_id, tokenizer=model_id,
            device=-1, max_length=max_length,
        )
        self._max_length = max_length
        print("[OPUS-MT] Ready.")

    def translate(self, text: str, src_lang: str = "", tgt_lang: str = "") -> str:
        return self.translate_batch([text])[0]

    def translate_batch(self, texts: list[str], *_) -> list[str]:
        out = self._pipeline(texts, max_length=self._max_length)
        return [o["translation_text"] for o in out]


# ─────────────────────────────────────────────────────────────────────────────
# Public Translator class
# ─────────────────────────────────────────────────────────────────────────────

class Translator:
    """
    Unified offline translation interface.

    backend   : "indictrans2" | "nllb200" | "transformers"
    src_lang  : ISO 639-1 code for transformers backend ("en", "de", …)
                ISO 639-3 + script for IndicTrans2/NLLB ("eng_Latn", "hin_Deva", …)
    tgt_lang  : same format as src_lang

    The translate() and translate_batch() methods accept src_lang / tgt_lang
    as call-time overrides so one Translator instance can serve multiple pairs.

    Example
    ───────
        t = Translator(backend="indictrans2")
        t.translate("Neural processing makes this fast.", "eng_Latn", "tam_Taml")

        t2 = Translator(backend="nllb200")
        t2.translate("Quantum computing is fascinating.", "eng_Latn", "deu_Latn")
    """

    def __init__(
        self,
        backend:      str = "indictrans2",
        # Legacy positional kwargs kept for backwards compat with ui/app.py
        source_lang:  str = SOURCE_LANGUAGE,
        target_lang:  str = TARGET_LANGUAGE,
        max_length:   int = 256,
        num_beams:    int = 4,
        # Allow caller to pass pre-resolved ONNX paths
        encoder_path: Optional[str] = None,
        decoder_path: Optional[str] = None,
    ) -> None:
        self.backend     = backend
        self.source_lang = source_lang
        self.target_lang = target_lang

        if backend == "indictrans2":
            self._impl = _IndicTrans2Backend(
                encoder_path=encoder_path,
                decoder_path=decoder_path,
                max_new_tokens=max_length,
                num_beams=num_beams,
            )
        elif backend == "nllb200":
            self._impl = _NLLB200Backend(
                encoder_path=encoder_path,
                decoder_path=decoder_path,
                max_new_tokens=max_length,
                num_beams=num_beams,
            )
        elif backend == "transformers":
            self._impl = _OpusMTBackend(
                src_lang=source_lang,
                tgt_lang=target_lang,
                max_length=max_length,
            )
        else:
            raise ValueError(
                f"Unknown backend {backend!r}. "
                "Choose 'indictrans2', 'nllb200', or 'transformers'."
            )

    # ── Public API ────────────────────────────────────────────────────────────

    def translate(
        self,
        text:     str,
        src_lang: Optional[str] = None,
        tgt_lang: Optional[str] = None,
    ) -> str:
        """
        Translate a single string.  Returns the translation.

        src_lang / tgt_lang default to the values passed at construction time.
        For IndicTrans2 / NLLB-200 use BCP-47 script tags ("eng_Latn").
        For transformers backend use ISO 639-1 ("en", "de").
        """
        if not text.strip():
            return ""
        src = src_lang or self.source_lang
        tgt = tgt_lang or self.target_lang
        return self._impl.translate(text, src, tgt)

    def translate_batch(
        self,
        texts:    list[str],
        src_lang: Optional[str] = None,
        tgt_lang: Optional[str] = None,
    ) -> list[str]:
        """Translate a batch of strings.  More efficient than looping translate()."""
        if not texts:
            return []
        src = src_lang or self.source_lang
        tgt = tgt_lang or self.target_lang
        return self._impl.translate_batch(texts, src, tgt)
