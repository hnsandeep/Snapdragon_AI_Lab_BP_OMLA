"""
transcription/transcriber.py
─────────────────────────────
Full Whisper-base-en inference pipeline for the Snapdragon X Elite NPU.

Two backends
────────────
  "onnx"    (default on-device)
    Loads WhisperEncoder.onnx + WhisperDecoder.onnx via ONNX Runtime.
    On a Snapdragon X Elite the compiled PRECOMPILED_QNN_ONNX variants run on
    the Hexagon NPU through the QNN Execution Provider.
    Falls back silently to CPU EP if onnxruntime-qnn / QnnHtp.dll is absent.

  "whisper" (PyTorch — for off-device dev/testing)
    Uses openai-whisper directly.  No ONNX files required.

Rolling 30-second capture
─────────────────────────
  WhisperPipeline wraps both the AudioCapture and the Transcriber into a
  single object.  Audio is collected in a thread-safe ring buffer; once a
  full chunk is ready, it is processed and the result appended to a list of
  TimedSegment objects (start_s, end_s, text).

  Usage:
    from transcription.transcriber import WhisperPipeline, TimedSegment

    def on_segment(seg: TimedSegment):
        print(f"[{seg.start_s:.1f}s – {seg.end_s:.1f}s]  {seg.text}")

    pipe = WhisperPipeline(on_segment=on_segment)
    pipe.start()
    time.sleep(120)
    pipe.stop()

  Or call transcribe_file(path) for offline file processing.

ONNX model paths
────────────────
  Default: models/WhisperEncoder.onnx and models/WhisperDecoder.onnx
  After AI Hub compile: set env vars (or .env):
    WHISPER_ENCODER_ONNX=models/WhisperEncoder_compiled.onnx
    WHISPER_DECODER_ONNX=models/WhisperDecoder_compiled.onnx
"""

from __future__ import annotations

import os
import queue
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterator, List, Optional

import numpy as np

# ── Project root resolution ───────────────────────────────────────────────────
_HERE = Path(__file__).resolve().parent
ROOT  = _HERE.parent
import sys
sys.path.insert(0, str(ROOT))
from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

from config.settings import (
    AUDIO_SAMPLE_RATE,
    AUDIO_CHUNK_SECONDS,
    MODELS_DIR,
    SOURCE_LANGUAGE,
)


# ─────────────────────────────────────────────────────────────────────────────
# Constants (Whisper-base)
# ─────────────────────────────────────────────────────────────────────────────
SAMPLE_RATE     = 16_000
CHUNK_SECONDS   = 30
N_SAMPLES       = CHUNK_SECONDS * SAMPLE_RATE      # 480 000
N_FFT           = 400
HOP_LENGTH      = 160
N_MELS          = 80
MELS_LEN        = N_SAMPLES // HOP_LENGTH          # 3000 — encoder input width
AUDIO_EMB_LEN   = MELS_LEN // 2                    # 1500 — cross-attention len

# Whisper-base decoder topology
NUM_DECODER_BLOCKS = 6
NUM_DECODER_HEADS  = 8
ATTENTION_DIM      = 512
HEAD_DIM           = ATTENTION_DIM // NUM_DECODER_HEADS   # 64
MAX_DECODE_TOKENS  = 224   # kv-cache capacity; Whisper limit for 30 s

# Special tokens (multilingual Whisper vocabulary)
TOKEN_SOT            = 50258   # <|startoftranscript|>
TOKEN_EOT            = 50256   # <|endoftext|>
TOKEN_BLANK          = 220     # " "
TOKEN_NO_TIMESTAMP   = 50362
TOKEN_TIMESTAMP_BEGIN = 50363
TOKEN_NO_SPEECH      = 50361
TOKEN_LANGUAGE_EN    = 50259   # <|en|>
TOKEN_TASK_TRANSCRIBE = 50359  # <|transcribe|>
NO_SPEECH_THR        = 0.6

# Tokens the decoder should never emit
_NON_SPEECH = frozenset([
    1, 2, 7, 8, 9, 10, 14, 25, 26, 27, 28, 29, 31, 58, 59, 60,
    61, 62, 63, 90, 91, 92, 93, 357, 366, 438, 532, 685, 705, 796,
    930, 1058, 1220, 1267, 1279, 1303, 1343, 1377, 1391, 1635,
    1782, 1875, 2162, 2361, 2488, 3467, 4008, 4211, 4600, 4808,
    5299, 5855, 6329, 7203, 9609, 9959, 10563, 10786, 11420, 11709,
    11907, 13163, 13697, 13700, 14808, 15306, 16410, 16791, 17992,
    19203, 19510, 20724, 22305, 22935, 27007, 30109, 30420, 33409,
    34949, 40283, 40493, 40549, 47282, 49146, 50257, 50357, 50358,
    50359, 50360, 50361,
])


# ─────────────────────────────────────────────────────────────────────────────
# Data types
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class TimedSegment:
    """A transcript fragment with wall-clock timestamps."""
    start_s: float
    end_s:   float
    text:    str

    def __str__(self) -> str:
        return f"[{self.start_s:6.1f}s – {self.end_s:6.1f}s]  {self.text}"


# ─────────────────────────────────────────────────────────────────────────────
# QNN / CPU session factory
# ─────────────────────────────────────────────────────────────────────────────

def _make_ort_session(onnx_path: str) -> "onnxruntime.InferenceSession":
    """
    Create an ONNX Runtime session.

    Provider priority:
      1. QNNExecutionProvider with HTP backend (Snapdragon NPU, FP16)
         — requires onnxruntime-qnn and QnnHtp.dll on PATH
      2. CPUExecutionProvider (fallback for off-device dev/CI)

    onnxruntime and onnxruntime-qnn are mutually exclusive packages.
    Install only one:
      On-device  : pip install onnxruntime-qnn==2.6.0
      Off-device : pip install onnxruntime==1.18.1
    """
    import onnxruntime as ort  # noqa: PLC0415

    # Probe for QNN EP availability
    available_eps = ort.get_available_providers()
    use_qnn = "QNNExecutionProvider" in available_eps

    if use_qnn:
        provider_options = [{
            "backend_path":                        "QnnHtp.dll",
            "htp_performance_mode":                "burst",
            "htp_graph_finalization_optimization_mode": "3",
            "enable_htp_fp16_precision":           "1",
            "profiling_level":                     "off",
        }]
        providers = ["QNNExecutionProvider"]
        print(f"  [ORT] QNN HTP EP  → {Path(onnx_path).name}")
    else:
        provider_options = [{}]
        providers = ["CPUExecutionProvider"]
        print(f"  [ORT] CPU EP (QNN not available)  → {Path(onnx_path).name}")

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
# Mel-spectrogram helper
# ─────────────────────────────────────────────────────────────────────────────

def _log_mel_spectrogram(audio: np.ndarray) -> np.ndarray:
    """
    Compute Whisper-compatible log-mel spectrogram.
    Returns shape (1, N_MELS, MELS_LEN) = (1, 80, 3000).
    """
    import torch  # noqa: PLC0415

    audio_t = torch.from_numpy(audio.astype(np.float32))
    # Pad or trim to exactly N_SAMPLES
    if len(audio_t) < N_SAMPLES:
        audio_t = torch.nn.functional.pad(audio_t, (0, N_SAMPLES - len(audio_t)))
    else:
        audio_t = audio_t[:N_SAMPLES]

    window  = torch.hann_window(N_FFT)
    stft    = torch.stft(audio_t, N_FFT, HOP_LENGTH, window=window, return_complex=True)
    magnitudes = stft[..., :-1].abs() ** 2

    # Load Whisper's exact mel filter bank (saved by extract_mel_filters.py or
    # from the whisper package directly)
    mel_filters = _load_mel_filters()
    mel_spec = torch.from_numpy(mel_filters) @ magnitudes

    log_spec = torch.clamp(mel_spec, min=1e-10).log10()
    log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
    log_spec = (log_spec + 4.0) / 4.0
    return log_spec.unsqueeze(0).detach().numpy().astype(np.float32)


def _load_mel_filters() -> np.ndarray:
    """
    Load the 80-band mel filter bank (shape 80×201).
    Priority:
      1. models/mel_filters.npz   (extracted once from openai-whisper)
      2. whisper.audio.mel_filters()  (requires openai-whisper installed)
      3. Scipy hand-rolled fallback (less accurate — use only for CI)
    """
    npz_path = MODELS_DIR / "mel_filters.npz"
    if npz_path.exists():
        data = np.load(str(npz_path))
        key = "mel_80" if "mel_80" in data else list(data.keys())[0]
        return data[key].astype(np.float32)
    try:
        import whisper  # noqa: PLC0415
        filters = whisper.audio.mel_filters(
            whisper.audio.torch.device("cpu"), N_MELS
        )
        return filters.numpy().astype(np.float32)
    except Exception:  # noqa: BLE001
        pass
    # Scipy fallback
    from scipy.signal import windows as W  # noqa: PLC0415
    freqs = np.linspace(0, SAMPLE_RATE / 2, N_FFT // 2 + 1)
    mel_min, mel_max = 2595 * np.log10(1 + np.array([0, SAMPLE_RATE / 2]) / 700)
    mel_pts = np.linspace(mel_min, mel_max, N_MELS + 2)
    hz_pts  = 700 * (10 ** (mel_pts / 2595) - 1)
    fbank   = np.zeros((N_MELS, N_FFT // 2 + 1), dtype=np.float32)
    for j in range(N_MELS):
        lo, ctr, hi = hz_pts[j], hz_pts[j + 1], hz_pts[j + 2]
        for i, f in enumerate(freqs):
            if lo <= f <= ctr:
                fbank[j, i] = (f - lo) / (ctr - lo)
            elif ctr < f <= hi:
                fbank[j, i] = (hi - f) / (hi - ctr)
    return fbank


def save_mel_filters_from_whisper() -> None:
    """
    One-time helper: extract mel filters from the whisper package and cache
    them to models/mel_filters.npz so the ONNX path does not need PyTorch
    installed at inference time.

    Run once after `pip install openai-whisper`:
        python -c "from transcription.transcriber import save_mel_filters_from_whisper; save_mel_filters_from_whisper()"
    """
    import whisper  # noqa: PLC0415
    import torch
    filters = whisper.audio.mel_filters(torch.device("cpu"), N_MELS)
    out = MODELS_DIR / "mel_filters.npz"
    np.savez(str(out), mel_80=filters.numpy())
    print(f"Saved mel filters → {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Tokenizer
# ─────────────────────────────────────────────────────────────────────────────

def _get_tokenizer(language: str = "en"):
    """Return the Whisper tokenizer (multilingual)."""
    try:
        import whisper  # noqa: PLC0415
        return whisper.tokenizer.get_tokenizer(
            multilingual=True, language=language, task="transcribe"
        )
    except Exception:  # noqa: BLE001
        pass
    # Fallback: tiktoken GPT-2 encoding (loses timestamp tokens but
    # still works for plain English transcription)
    import tiktoken  # noqa: PLC0415
    return tiktoken.get_encoding("gpt2")


# ─────────────────────────────────────────────────────────────────────────────
# Core ONNX inference engine
# ─────────────────────────────────────────────────────────────────────────────

class WhisperONNX:
    """
    Encoder-decoder inference using ONNX Runtime.

    The encoder processes a fixed-length mel spectrogram (30 s) and produces
    cross-attention key/value caches.  The decoder runs auto-regressively,
    consuming those caches on every step while maintaining its own self-
    attention KV cache.
    """

    def __init__(self, encoder_path: str, decoder_path: str, language: str = "en") -> None:
        print("[WhisperONNX] Loading encoder …")
        self._enc = _make_ort_session(encoder_path)
        print("[WhisperONNX] Loading decoder …")
        self._dec = _make_ort_session(decoder_path)
        self._tokenizer = _get_tokenizer(language)
        self._language  = language
        print("[WhisperONNX] Ready.")

    # ── Public ────────────────────────────────────────────────────────────────

    def transcribe(self, audio: np.ndarray) -> str:
        """Transcribe a mono float32 array of any length (split into 30 s chunks)."""
        chunks = self._split_audio(audio)
        texts  = [self._transcribe_chunk(c) for c in chunks]
        return " ".join(t for t in texts if t).strip()

    def transcribe_with_timestamps(
        self, audio: np.ndarray, start_offset_s: float = 0.0
    ) -> list[TimedSegment]:
        """
        Transcribe and return a list of TimedSegment objects.
        Each 30 s chunk becomes one segment.
        """
        segments: list[TimedSegment] = []
        chunk_size = N_SAMPLES
        for i, chunk in enumerate(self._split_audio(audio)):
            t0 = start_offset_s + i * CHUNK_SECONDS
            t1 = t0 + CHUNK_SECONDS
            text = self._transcribe_chunk(chunk)
            if text:
                segments.append(TimedSegment(start_s=t0, end_s=t1, text=text))
        return segments

    # ── Encoder ───────────────────────────────────────────────────────────────

    def _run_encoder(self, mel: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Run the encoder. Returns (k_cache_cross, v_cache_cross)."""
        outputs = self._enc.run(None, {"audio": mel})
        # qai_hub_models encoder outputs: [k_cache_cross, v_cache_cross]
        return outputs[0], outputs[1]

    # ── Decoder ───────────────────────────────────────────────────────────────

    def _run_decoder_step(
        self,
        x: np.ndarray,
        index: np.ndarray,
        k_cache_cross: np.ndarray,
        v_cache_cross: np.ndarray,
        k_cache_self:  np.ndarray,
        v_cache_self:  np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        One auto-regressive decoder step.
        Returns (logits, k_cache_self_updated, v_cache_self_updated).
        """
        outputs = self._dec.run(None, {
            "x":             x.astype(np.int32),
            "index":         index.astype(np.int32),
            "k_cache_cross": k_cache_cross,
            "v_cache_cross": v_cache_cross,
            "k_cache_self":  k_cache_self,
            "v_cache_self":  v_cache_self,
        })
        # qai_hub_models decoder outputs: [logits, k_cache_self, v_cache_self]
        return outputs[0], outputs[1], outputs[2]

    # ── Chunk transcription ───────────────────────────────────────────────────

    def _transcribe_chunk(self, audio: np.ndarray) -> str:
        """Transcribe a single ≤30 s audio chunk."""
        mel = _log_mel_spectrogram(audio)                    # (1, 80, 3000)
        k_cross, v_cross = self._run_encoder(mel)

        # Initialise self-attention KV caches to zero
        # k_cache_self shape: (blocks, heads, head_dim, max_tokens)
        # v_cache_self shape: (blocks, heads, max_tokens, head_dim)
        k_self = np.zeros(
            (NUM_DECODER_BLOCKS, NUM_DECODER_HEADS, HEAD_DIM, MAX_DECODE_TOKENS),
            dtype=np.float32,
        )
        v_self = np.zeros(
            (NUM_DECODER_BLOCKS, NUM_DECODER_HEADS, MAX_DECODE_TOKENS, HEAD_DIM),
            dtype=np.float32,
        )

        # Whisper prompt: SOT + language + task
        prompt_tokens = [TOKEN_SOT, TOKEN_LANGUAGE_EN, TOKEN_TASK_TRANSCRIBE]
        x = np.array([[prompt_tokens[-1]]], dtype=np.int32)

        decoded_tokens: list[int] = list(prompt_tokens)
        no_speech_detected = False

        for i in range(MAX_DECODE_TOKENS):
            idx = np.array([[len(decoded_tokens) - 1]], dtype=np.int32)
            logits, k_self, v_self = self._run_decoder_step(
                x, idx, k_cross, v_cross, k_self, v_self
            )
            logits = logits[0, -1].astype(np.float64)       # (vocab,)

            # ── Suppress non-speech tokens ────────────────────────────────────
            for t in _NON_SPEECH:
                logits[t] = -np.inf
            # Suppress timestamps except at natural breaks
            logits[TOKEN_TIMESTAMP_BEGIN:] = -np.inf
            logits[TOKEN_NO_TIMESTAMP] = -np.inf

            if i == 0:
                # Suppress EOT and blank on the first real token
                logits[TOKEN_EOT]   = -np.inf
                logits[TOKEN_BLANK] = -np.inf
                # No-speech detection
                log_probs       = logits - _logsumexp(logits)
                no_speech_prob  = float(np.exp(log_probs[TOKEN_NO_SPEECH]))
                if no_speech_prob > NO_SPEECH_THR:
                    no_speech_detected = True
                    break

            next_token = int(np.argmax(logits))
            if next_token == TOKEN_EOT:
                break

            decoded_tokens.append(next_token)
            x = np.array([[next_token]], dtype=np.int32)

        if no_speech_detected or len(decoded_tokens) <= len(prompt_tokens):
            return ""

        # Decode — strip the prompt prefix
        payload = decoded_tokens[len(prompt_tokens):]
        try:
            return self._tokenizer.decode(payload).strip()
        except Exception:  # noqa: BLE001
            # tiktoken fallback: ignore unmappable tokens
            return "".join(
                self._tokenizer.decode([t]) for t in payload
                if t < 50257
            ).strip()

    # ── Audio chunking ────────────────────────────────────────────────────────

    @staticmethod
    def _split_audio(audio: np.ndarray) -> list[np.ndarray]:
        """Split a long audio array into ≤30 s chunks."""
        if len(audio) == 0:
            return []
        n_chunks = max(1, int(np.ceil(len(audio) / N_SAMPLES)))
        chunks = []
        for i in range(n_chunks):
            chunk = audio[i * N_SAMPLES : (i + 1) * N_SAMPLES]
            if len(chunk) > 0:
                chunks.append(chunk)
        return chunks


def _logsumexp(x: np.ndarray) -> float:
    """Numerically stable log-sum-exp."""
    m = x.max()
    return float(m + np.log(np.sum(np.exp(x - m))))


# ─────────────────────────────────────────────────────────────────────────────
# PyTorch Whisper backend (off-device dev/test)
# ─────────────────────────────────────────────────────────────────────────────

class WhisperPyTorch:
    """Thin wrapper around openai-whisper for off-device development."""

    def __init__(self, model_size: str = "base", language: str = "en") -> None:
        import whisper  # noqa: PLC0415
        print(f"[WhisperPyTorch] Loading '{model_size}' …")
        self._model    = whisper.load_model(model_size)
        self._language = language
        print("[WhisperPyTorch] Ready.")

    def transcribe(self, audio: np.ndarray) -> str:
        result = self._model.transcribe(
            audio.astype(np.float32),
            language=self._language,
            fp16=False,
            verbose=False,
        )
        return result["text"].strip()

    def transcribe_with_timestamps(
        self, audio: np.ndarray, start_offset_s: float = 0.0
    ) -> list[TimedSegment]:
        result = self._model.transcribe(
            audio.astype(np.float32),
            language=self._language,
            fp16=False,
            verbose=False,
            word_timestamps=False,
        )
        segments: list[TimedSegment] = []
        for seg in result.get("segments", []):
            segments.append(TimedSegment(
                start_s=start_offset_s + seg["start"],
                end_s=start_offset_s + seg["end"],
                text=seg["text"].strip(),
            ))
        return segments


# ─────────────────────────────────────────────────────────────────────────────
# High-level pipeline: rolling mic capture → timestamped segments
# ─────────────────────────────────────────────────────────────────────────────

class WhisperPipeline:
    """
    Combines sounddevice mic capture with Whisper inference.

    Audio arrives from the microphone in 30-second rolling chunks.
    Each chunk is transcribed and delivered to `on_segment` callback
    from a dedicated worker thread — the mic capture thread is never blocked.

    Example
    ───────
        segments = []

        def handle(seg):
            segments.append(seg)
            print(seg)

        pipe = WhisperPipeline(backend="onnx", on_segment=handle)
        pipe.start()
        # ... record lecture ...
        pipe.stop()

        for s in pipe.get_all_segments():
            print(s)
    """

    def __init__(
        self,
        backend: str = "onnx",
        on_segment: Optional[Callable[[TimedSegment], None]] = None,
        language: str = SOURCE_LANGUAGE,
        model_size: str = "base",             # used only for PyTorch backend
        encoder_path: Optional[str] = None,
        decoder_path: Optional[str] = None,
        sample_rate: int = SAMPLE_RATE,
        chunk_seconds: float = CHUNK_SECONDS,
    ) -> None:
        self._backend       = backend
        self._on_segment    = on_segment
        self._language      = language
        self._sample_rate   = sample_rate
        self._chunk_samples = int(sample_rate * chunk_seconds)
        self._chunk_seconds = chunk_seconds

        # ── Resolve model paths ───────────────────────────────────────────────
        enc = encoder_path or os.getenv(
            "WHISPER_ENCODER_ONNX",
            str(MODELS_DIR / "WhisperEncoder.onnx")
        )
        dec = decoder_path or os.getenv(
            "WHISPER_DECODER_ONNX",
            str(MODELS_DIR / "WhisperDecoder.onnx")
        )

        # ── Load model ────────────────────────────────────────────────────────
        if backend == "onnx":
            if not Path(enc).exists() or not Path(dec).exists():
                raise FileNotFoundError(
                    f"ONNX model files not found:\n  {enc}\n  {dec}\n"
                    "Run: python scripts/export_whisper.py --export-only"
                )
            self._engine: WhisperONNX | WhisperPyTorch = WhisperONNX(enc, dec, language)
        elif backend == "whisper":
            self._engine = WhisperPyTorch(model_size, language)
        else:
            raise ValueError(f"Unknown backend {backend!r}. Choose 'onnx' or 'whisper'.")

        # ── State ─────────────────────────────────────────────────────────────
        self._audio_queue: queue.Queue[np.ndarray] = queue.Queue()
        self._segments:    List[TimedSegment]       = []
        self._segments_lock = threading.Lock()
        self._running       = False
        self._elapsed_s     = 0.0              # wall-clock audio time so far
        self._capture_buf   = np.empty((0,), dtype=np.float32)
        self._buf_lock      = threading.Lock()
        self._stream        = None
        self._worker_thread: Optional[threading.Thread] = None

    # ── Public API ────────────────────────────────────────────────────────────

    def start(self) -> None:
        """Open the microphone and start the transcription worker."""
        if self._running:
            return
        self._running = True
        # Start inference worker first so queue is ready
        self._worker_thread = threading.Thread(
            target=self._worker_loop, daemon=True, name="whisper-worker"
        )
        self._worker_thread.start()
        # Open mic stream
        import sounddevice as sd  # noqa: PLC0415
        self._stream = sd.InputStream(
            samplerate=self._sample_rate,
            channels=1,
            dtype="float32",
            callback=self._audio_callback,
        )
        self._stream.start()
        print("[WhisperPipeline] Recording …  (Ctrl-C to stop)")

    def stop(self) -> None:
        """Stop the mic stream and drain the worker queue."""
        self._running = False
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        # Flush remaining buffer
        with self._buf_lock:
            tail = self._capture_buf.copy()
        if len(tail) > 0:
            self._audio_queue.put(tail)
        # Sentinel to stop worker
        self._audio_queue.put(None)
        if self._worker_thread is not None:
            self._worker_thread.join(timeout=120)
        print("[WhisperPipeline] Stopped.")

    def get_all_segments(self) -> list[TimedSegment]:
        """Return all collected segments (thread-safe copy)."""
        with self._segments_lock:
            return list(self._segments)

    def transcribe_file(self, path: str) -> list[TimedSegment]:
        """
        Transcribe an audio file offline (no mic).
        Returns timestamped segments.
        """
        import soundfile as sf  # noqa: PLC0415
        audio, sr = sf.read(path, dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sr != SAMPLE_RATE:
            audio = _resample(audio, sr, SAMPLE_RATE)
        return self._engine.transcribe_with_timestamps(audio)

    # ── Internal ──────────────────────────────────────────────────────────────

    def _audio_callback(
        self,
        indata:    np.ndarray,
        frames:    int,
        time_info,
        status,
    ) -> None:
        """sounddevice callback — runs in a high-priority audio thread."""
        mono = indata[:, 0]
        with self._buf_lock:
            self._capture_buf = np.concatenate([self._capture_buf, mono])
            while len(self._capture_buf) >= self._chunk_samples:
                chunk = self._capture_buf[: self._chunk_samples].copy()
                self._capture_buf = self._capture_buf[self._chunk_samples :]
                self._audio_queue.put(chunk)

    def _worker_loop(self) -> None:
        """Drain the audio queue and transcribe each chunk."""
        while True:
            try:
                chunk = self._audio_queue.get(timeout=0.5)
            except queue.Empty:
                if not self._running:
                    break
                continue

            if chunk is None:           # sentinel
                break

            t0 = self._elapsed_s
            t1 = t0 + len(chunk) / self._sample_rate
            self._elapsed_s = t1

            try:
                segs = self._engine.transcribe_with_timestamps(chunk, start_offset_s=t0)
                for seg in segs:
                    with self._segments_lock:
                        self._segments.append(seg)
                    if self._on_segment is not None:
                        try:
                            self._on_segment(seg)
                        except Exception:  # noqa: BLE001
                            pass
            except Exception as exc:  # noqa: BLE001
                print(f"[WhisperPipeline] Transcription error: {exc}")

            self._audio_queue.task_done()


# ─────────────────────────────────────────────────────────────────────────────
# Resampling helper
# ─────────────────────────────────────────────────────────────────────────────

def _resample(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Simple linear-interpolation resampler (no extra deps required)."""
    if orig_sr == target_sr:
        return audio
    ratio  = target_sr / orig_sr
    n_out  = int(len(audio) * ratio)
    x_orig = np.linspace(0, len(audio) - 1, len(audio))
    x_new  = np.linspace(0, len(audio) - 1, n_out)
    return np.interp(x_new, x_orig, audio).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Backwards-compatible Transcriber class (used by ui/app.py)
# ─────────────────────────────────────────────────────────────────────────────

class Transcriber:
    """
    Simple interface kept for backwards compatibility with ui/app.py.

    backend="onnx"    — uses WhisperONNX (QNN EP → CPU EP fallback)
    backend="whisper" — uses openai-whisper PyTorch
    """

    def __init__(
        self,
        backend: str = "onnx",
        model_size: str = "base",
        language: str = SOURCE_LANGUAGE,
        encoder_path: Optional[str] = None,
        decoder_path: Optional[str] = None,
    ) -> None:
        self._pipeline = WhisperPipeline(
            backend=backend,
            language=language,
            model_size=model_size,
            encoder_path=encoder_path,
            decoder_path=decoder_path,
        )

    def transcribe(self, audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> str:
        if sample_rate != SAMPLE_RATE:
            audio = _resample(audio, sample_rate, SAMPLE_RATE)
        return self._pipeline._engine.transcribe(audio)

    def transcribe_with_timestamps(
        self, audio: np.ndarray, sample_rate: int = SAMPLE_RATE
    ) -> list[TimedSegment]:
        if sample_rate != SAMPLE_RATE:
            audio = _resample(audio, sample_rate, SAMPLE_RATE)
        return self._pipeline._engine.transcribe_with_timestamps(audio)
