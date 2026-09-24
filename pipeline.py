"""pipeline.py — wires audio_capture → transcription → translation → summarization.

Zero network calls at runtime.  All inference runs on-device (QNN/NPU or CPU fallback).

Usage
─────
    from pipeline import Pipeline, PipelineConfig

    cfg = PipelineConfig(src_lang="en", tgt_lang="hin_Deva")
    p   = Pipeline(cfg)

    p.on_segment = lambda seg: print(seg)   # called for every 30-second chunk
    p.start()
    ...
    p.stop()
    summary = p.summarize()
"""
from __future__ import annotations
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Optional
import numpy as np

# ── lazy imports so each module's heavy deps load only once ───────────────────
from audio_capture.capture     import AudioCapture, SAMPLE_RATE
from transcription.whisper_stt import WhisperSTT
from translation.translate     import translate
from summarization.summarize   import summarize as geniex_summarize, close as geniex_close


@dataclass
class Segment:
    start_s: float
    end_s:   float
    text_src: str          # source-language transcript
    text_tgt: str = ""     # translated text (filled after translate step)

    def __str__(self) -> str:
        t = self.text_tgt or self.text_src
        return f"[{self.start_s:6.1f}s–{self.end_s:5.1f}s]  {t}"


@dataclass
class PipelineConfig:
    src_lang:           str   = "en"
    tgt_lang:           str   = "hin_Deva"
    translate_backend:  str   = "indictrans2"
    whisper_backend:    str   = "onnx"          # "onnx" or "whisper" (PyTorch)
    translate_enabled:  bool  = True
    chunk_seconds:      float = 30.0
    mic_device:         Optional[int] = None


class Pipeline:
    """End-to-end offline pipeline: mic → STT → translate → LLM summary."""

    def __init__(self, config: PipelineConfig | None = None) -> None:
        self.cfg      = config or PipelineConfig()
        self._stt     = WhisperSTT(language=self.cfg.src_lang) \
                        if self.cfg.whisper_backend == "onnx" else None
        self._capture = AudioCapture(
            on_chunk=self._on_audio_chunk,
            chunk_seconds=self.cfg.chunk_seconds,
            device=self.cfg.mic_device,
        )
        self._segments:  list[Segment]          = []
        self._seg_lock   = threading.Lock()
        self._elapsed_s  = 0.0
        self.on_segment: Optional[Callable[[Segment], None]] = None

    # ── public ────────────────────────────────────────────────────────────────

    def start(self) -> None:
        """Open microphone and start processing."""
        self._capture.start()

    def stop(self) -> None:
        """Stop microphone and flush remaining audio."""
        self._capture.stop()

    def summarize(self, on_token: Optional[Callable[[str], None]] = None) -> str:
        """Run GenieX LLM over all accumulated translated transcript text.

        Entirely local — no network calls.
        """
        with self._seg_lock:
            full = " ".join(
                s.text_tgt or s.text_src for s in self._segments
            ).strip()
        return geniex_summarize(full, stream=(on_token is not None),
                                on_token=on_token)

    def get_segments(self) -> list[Segment]:
        with self._seg_lock:
            return list(self._segments)

    def get_full_transcript(self, translated: bool = True) -> str:
        with self._seg_lock:
            return " ".join(
                (s.text_tgt if translated and s.text_tgt else s.text_src)
                for s in self._segments
            ).strip()

    def close(self) -> None:
        self.stop()
        geniex_close()

    # ── internal ──────────────────────────────────────────────────────────────

    def _on_audio_chunk(self, audio: np.ndarray) -> None:
        t0 = self._elapsed_s
        t1 = t0 + len(audio) / SAMPLE_RATE
        self._elapsed_s = t1

        text_src = self._transcribe(audio)
        if not text_src:
            return

        text_tgt = ""
        if self.cfg.translate_enabled and self.cfg.src_lang != self.cfg.tgt_lang:
            try:
                src_tag = _iso1_to_tag(self.cfg.src_lang)
                text_tgt = translate(
                    text_src, src=src_tag, tgt=self.cfg.tgt_lang,
                    backend=self.cfg.translate_backend,
                )
            except Exception as exc:
                text_tgt = f"[translation error: {exc}]"

        seg = Segment(start_s=t0, end_s=t1,
                      text_src=text_src, text_tgt=text_tgt)
        with self._seg_lock:
            self._segments.append(seg)

        if self.on_segment is not None:
            try:
                self.on_segment(seg)
            except Exception:
                pass

    def _transcribe(self, audio: np.ndarray) -> str:
        if self.cfg.whisper_backend == "onnx" and self._stt is not None:
            return self._stt.transcribe(audio)
        # PyTorch whisper fallback
        import whisper as _w
        if not hasattr(self, "_pt_model"):
            self._pt_model = _w.load_model("base")
        result = self._pt_model.transcribe(audio.astype(np.float32),
                                           language=self.cfg.src_lang, fp16=False)
        return result["text"].strip()


# ISO 639-1 → ISO 639-3+script map (subset for common languages)
_ISO1_MAP = {
    "en": "eng_Latn", "hi": "hin_Deva", "kn": "kan_Knda",
    "ta": "tam_Taml", "te": "tel_Telu", "bn": "ben_Beng",
    "mr": "mar_Deva", "gu": "guj_Gujr", "ml": "mal_Mlym",
    "pa": "pan_Guru", "ur": "urd_Arab", "de": "deu_Latn",
    "fr": "fra_Latn", "es": "spa_Latn", "ar": "arb_Arab",
    "zh": "zho_Hans",
}

def _iso1_to_tag(code: str) -> str:
    return _ISO1_MAP.get(code, code)
