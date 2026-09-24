"""translation/translate.py — translate(text, src, tgt) using IndicTrans2 or NLLB-200.

Language tags  (ISO 639-3 + ISO 15924):
    eng_Latn  hin_Deva  kan_Knda  tam_Taml  tel_Telu  ben_Beng
    mar_Deva  guj_Gujr  mal_Mlym  pan_Guru  urd_Arab
    deu_Latn  fra_Latn  spa_Latn  arb_Arab  zho_Hans

Set TRANSLATE_MODEL=indictrans2 (default) or nllb200 in .env.
Set IT2_ENCODER_ONNX / IT2_DECODER_ONNX  or  NLLB_ENCODER_ONNX / NLLB_DECODER_ONNX
to the compiled .onnx paths after running scripts/export_translation.py.
"""
from __future__ import annotations
import os, sys
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:
    pass

MODELS_DIR = ROOT / "models"
_BACKEND   = os.getenv("TRANSLATE_MODEL", "indictrans2").lower()

# ── singleton ─────────────────────────────────────────────────────────────────
_model     = None
_tokenizer = None
_ip        = None          # IndicProcessor
_loaded_be = None


def _load(backend: str) -> None:
    global _model, _tokenizer, _ip, _loaded_be
    if _loaded_be == backend:
        return

    from optimum.onnxruntime import ORTModelForSeq2SeqLM
    from transformers import AutoTokenizer

    if backend == "indictrans2":
        repo = "TigreGotico/indictrans2-en-indic-dist-200M-onnx"
        enc  = os.getenv("IT2_ENCODER_ONNX",
               str(MODELS_DIR / "indictrans2" / "int8" / "encoder_model_compiled.onnx"))
        mdir = Path(enc).parent
        if mdir.exists() and (mdir / "encoder_model.onnx").exists():
            _model     = ORTModelForSeq2SeqLM.from_pretrained(str(mdir), use_cache=True,
                                                               trust_remote_code=True)
            _tokenizer = AutoTokenizer.from_pretrained(
                str(mdir.parent), trust_remote_code=True)
        else:
            _model     = ORTModelForSeq2SeqLM.from_pretrained(
                repo, use_cache=True, trust_remote_code=True)
            _tokenizer = AutoTokenizer.from_pretrained(repo, trust_remote_code=True)

        from IndicTransToolkit.processor import IndicProcessor
        _ip = IndicProcessor(inference=True)

    else:  # nllb200
        repo = "facebook/nllb-200-distilled-600M"
        enc  = os.getenv("NLLB_ENCODER_ONNX",
               str(MODELS_DIR / "nllb200" / "int8" / "encoder_model_compiled.onnx"))
        mdir = Path(enc).parent
        from transformers import NllbTokenizer
        if mdir.exists() and (mdir / "encoder_model.onnx").exists():
            _model     = ORTModelForSeq2SeqLM.from_pretrained(str(mdir), use_cache=True)
            _tokenizer = NllbTokenizer.from_pretrained(str(mdir.parent))
        else:
            _model     = ORTModelForSeq2SeqLM.from_pretrained(repo, export=True)
            _tokenizer = NllbTokenizer.from_pretrained(repo)

    _loaded_be = backend


# ── public API ────────────────────────────────────────────────────────────────

def translate(text: str, src: str = "eng_Latn", tgt: str = "hin_Deva",
              backend: str | None = None, num_beams: int = 4) -> str:
    """Translate *text* from *src* to *tgt* fully offline.

    No external API calls — inference runs through ORTModelForSeq2SeqLM on the
    locally loaded ONNX model (QNN EP when onnxruntime-qnn is installed).
    """
    if not text.strip():
        return ""

    be = (backend or _BACKEND).lower()
    _load(be)

    import torch

    if be == "indictrans2":
        batch  = _ip.preprocess_batch([text], src_lang=src, tgt_lang=tgt)
        enc_in = _tokenizer(batch, return_tensors="pt", padding=True,
                            truncation=True, max_length=256)
        with torch.inference_mode():
            out = _model.generate(**enc_in, num_beams=num_beams,
                                  max_new_tokens=256, use_cache=True)
        decoded = _tokenizer.batch_decode(out, skip_special_tokens=True)
        return _ip.postprocess_batch(decoded, lang=tgt)[0]

    else:  # nllb200
        _tokenizer.src_lang = src
        enc_in = _tokenizer([text], return_tensors="pt", padding=True,
                             truncation=True, max_length=256)
        tgt_id = _tokenizer.lang_code_to_id[tgt]
        with torch.inference_mode():
            out = _model.generate(**enc_in, forced_bos_token_id=tgt_id,
                                  num_beams=num_beams, max_new_tokens=256)
        return _tokenizer.decode(out[0], skip_special_tokens=True)


def translate_batch(texts: list[str], src: str = "eng_Latn", tgt: str = "hin_Deva",
                    backend: str | None = None) -> list[str]:
    return [translate(t, src, tgt, backend) for t in texts]
