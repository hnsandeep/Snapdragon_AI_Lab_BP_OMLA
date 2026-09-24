"""
config/settings.py
Central configuration for the Offline Multilingual Lecture Assistant.
Sensitive values (API token) are loaded from .env — never hard-coded here.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# ── Resolve project root regardless of where Python is invoked from ──────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")          # reads .env if present

# ── AI Hub ────────────────────────────────────────────────────────────────────
# Set QAI_HUB_API_TOKEN in your .env file or as an environment variable.
# Never commit the token to version control.
QAI_HUB_API_TOKEN: str = os.getenv("QAI_HUB_API_TOKEN", "")

# Target device for compile / profile jobs on AI Hub.
# Run `qai-hub list-devices` to see all available hosted devices.
QAI_HUB_DEVICE: str = os.getenv("QAI_HUB_DEVICE", "Snapdragon X Elite CRD")

# ── Model paths ───────────────────────────────────────────────────────────────
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)   # ensure directory exists at import time

# ── Whisper ONNX paths ────────────────────────────────────────────────────────
# After running  scripts/export_whisper.py  set these in .env:
#   WHISPER_ENCODER_ONNX=models/WhisperEncoder_compiled.onnx
#   WHISPER_DECODER_ONNX=models/WhisperDecoder_compiled.onnx
WHISPER_ENCODER_ONNX: str = os.getenv(
    "WHISPER_ENCODER_ONNX", str(MODELS_DIR / "WhisperEncoder.onnx")
)
WHISPER_DECODER_ONNX: str = os.getenv(
    "WHISPER_DECODER_ONNX", str(MODELS_DIR / "WhisperDecoder.onnx")
)
# Legacy single-file alias kept for any code that imports WHISPER_ONNX
WHISPER_ONNX = WHISPER_ENCODER_ONNX

# ── Translation ONNX paths ────────────────────────────────────────────────────
# After running  scripts/export_translation.py  set ONE of these blocks in .env:
#
#   IndicTrans2 (recommended for Indic languages):
#     TRANSLATE_MODEL=indictrans2
#     IT2_ENCODER_ONNX=models/indictrans2/int8/encoder_model_compiled.onnx
#     IT2_DECODER_ONNX=models/indictrans2/int8/decoder_model_compiled.onnx
#
#   NLLB-200 (200 languages):
#     TRANSLATE_MODEL=nllb200
#     NLLB_ENCODER_ONNX=models/nllb200/int8/encoder_model_compiled.onnx
#     NLLB_DECODER_ONNX=models/nllb200/int8/decoder_model_compiled.onnx
#
TRANSLATE_MODEL: str   = os.getenv("TRANSLATE_MODEL", "indictrans2")

IT2_ENCODER_ONNX: str  = os.getenv(
    "IT2_ENCODER_ONNX",
    str(MODELS_DIR / "indictrans2" / "int8" / "encoder_model_compiled.onnx")
)
IT2_DECODER_ONNX: str  = os.getenv(
    "IT2_DECODER_ONNX",
    str(MODELS_DIR / "indictrans2" / "int8" / "decoder_model_compiled.onnx")
)
NLLB_ENCODER_ONNX: str = os.getenv(
    "NLLB_ENCODER_ONNX",
    str(MODELS_DIR / "nllb200" / "int8" / "encoder_model_compiled.onnx")
)
NLLB_DECODER_ONNX: str = os.getenv(
    "NLLB_DECODER_ONNX",
    str(MODELS_DIR / "nllb200" / "int8" / "decoder_model_compiled.onnx")
)

# Legacy alias
TRANSLATE_ONNX = IT2_ENCODER_ONNX
SUMMARY_ONNX   = str(MODELS_DIR / "bart_summary.onnx")

# ── Audio ─────────────────────────────────────────────────────────────────────
AUDIO_SAMPLE_RATE: int    = 16_000    # Hz — required by Whisper
AUDIO_CHANNELS: int       = 1         # mono
AUDIO_CHUNK_SECONDS: float = 30.0     # seconds per processing chunk

# ── Language defaults ─────────────────────────────────────────────────────────
# ISO 639-1 for transformers backend; ISO 639-3+script for IndicTrans2/NLLB
SOURCE_LANGUAGE: str = os.getenv("SOURCE_LANGUAGE", "en")
TARGET_LANGUAGE: str = os.getenv("TARGET_LANGUAGE", "hin_Deva")  # Hindi by default

# ── UI ────────────────────────────────────────────────────────────────────────
UI_HOST: str = "127.0.0.1"
UI_PORT: int = 7860
