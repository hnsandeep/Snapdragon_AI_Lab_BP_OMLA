"""transcription/whisper_stt.py — Whisper-base-en via ONNX Runtime (QNN EP → CPU fallback).

Export once:
    python scripts/export_whisper.py --export-only
    # produces models/WhisperEncoder.onnx + models/WhisperDecoder.onnx

Compile for NPU on AI Hub:
    python scripts/export_whisper.py
    # produces models/WhisperEncoder_compiled.onnx + models/WhisperDecoder_compiled.onnx
    # set WHISPER_ENCODER_ONNX / WHISPER_DECODER_ONNX in .env
"""
from __future__ import annotations
import os, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# ── constants (Whisper-base) ──────────────────────────────────────────────────
SAMPLE_RATE  = 16_000
N_SAMPLES    = 480_000   # 30 s
N_FFT        = 400
HOP_LENGTH   = 160
N_MELS       = 80
MELS_LEN     = N_SAMPLES // HOP_LENGTH   # 3000
CROSS_LEN    = MELS_LEN  // 2            # 1500
N_BLOCKS     = 6
N_HEADS      = 8
HEAD_DIM     = 64        # attention_dim / n_heads = 512/8
MAX_TOKENS   = 224

TOKEN_SOT          = 50258
TOKEN_EOT          = 50256
TOKEN_BLANK        = 220
TOKEN_NO_SPEECH    = 50361
TOKEN_NO_TIMESTAMP = 50362
TOKEN_TS_BEGIN     = 50363
TOKEN_LANG_EN      = 50259
TOKEN_TRANSCRIBE   = 50359
NO_SPEECH_THR      = 0.6

_NON_SPEECH = frozenset([
    1,2,7,8,9,10,14,25,26,27,28,29,31,58,59,60,61,62,63,90,91,92,93,
    357,366,438,532,685,705,796,930,1058,1220,1267,1279,1303,1343,1377,
    1391,1635,1782,1875,2162,2361,2488,3467,4008,4211,4600,4808,5299,
    5855,6329,7203,9609,9959,10563,10786,11420,11709,11907,13163,13697,
    13700,14808,15306,16410,16791,17992,19203,19510,20724,22305,22935,
    27007,30109,30420,33409,34949,40283,40493,40549,47282,49146,
    50257,50357,50358,50359,50360,50361,
])

MODELS_DIR = ROOT / "models"


# ── ORT session with QNN → CPU fallback ──────────────────────────────────────

def _ort_session(path: str):
    import onnxruntime as ort
    use_qnn = "QNNExecutionProvider" in ort.get_available_providers()
    if use_qnn:
        providers = ["QNNExecutionProvider"]
        provider_opts = [{"backend_path": "QnnHtp.dll",
                          "htp_performance_mode": "burst",
                          "enable_htp_fp16_precision": "1",
                          "htp_graph_finalization_optimization_mode": "3"}]
    else:
        providers, provider_opts = ["CPUExecutionProvider"], [{}]
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.intra_op_num_threads = 4
    return ort.InferenceSession(path, sess_options=opts,
                                providers=providers, provider_options=provider_opts)


# ── mel spectrogram ───────────────────────────────────────────────────────────

def _mel_filters() -> np.ndarray:
    npz = MODELS_DIR / "mel_filters.npz"
    if npz.exists():
        d = np.load(str(npz))
        return d[list(d.keys())[0]].astype(np.float32)
    import whisper, torch
    return whisper.audio.mel_filters(torch.device("cpu"), N_MELS).numpy().astype(np.float32)


_MEL_CACHE: np.ndarray | None = None

def _log_mel(audio: np.ndarray) -> np.ndarray:
    global _MEL_CACHE
    import torch
    if _MEL_CACHE is None:
        _MEL_CACHE = _mel_filters()
    a = torch.from_numpy(audio.astype(np.float32))
    if len(a) < N_SAMPLES:
        a = torch.nn.functional.pad(a, (0, N_SAMPLES - len(a)))
    else:
        a = a[:N_SAMPLES]
    w   = torch.hann_window(N_FFT)
    mag = torch.stft(a, N_FFT, HOP_LENGTH, window=w, return_complex=True)[..., :-1].abs() ** 2
    ms  = torch.from_numpy(_MEL_CACHE) @ mag
    ls  = torch.clamp(ms, min=1e-10).log10()
    ls  = torch.maximum(ls, ls.max() - 8.0)
    ls  = (ls + 4.0) / 4.0
    return ls.unsqueeze(0).numpy().astype(np.float32)


# ── tokenizer ─────────────────────────────────────────────────────────────────

def _tokenizer(language: str = "en"):
    try:
        import whisper
        return whisper.tokenizer.get_tokenizer(multilingual=True, language=language, task="transcribe")
    except Exception:
        import tiktoken
        return tiktoken.get_encoding("gpt2")


# ── WhisperSTT ────────────────────────────────────────────────────────────────

class WhisperSTT:
    """Encoder-decoder Whisper inference over ONNX Runtime."""

    def __init__(self, language: str = "en",
                 encoder_path: str | None = None,
                 decoder_path: str | None = None) -> None:
        self.language = language
        enc = encoder_path or os.getenv(
            "WHISPER_ENCODER_ONNX", str(MODELS_DIR / "WhisperEncoder.onnx"))
        dec = decoder_path or os.getenv(
            "WHISPER_DECODER_ONNX", str(MODELS_DIR / "WhisperDecoder.onnx"))
        if not Path(enc).exists():
            raise FileNotFoundError(
                f"Encoder not found: {enc}\n"
                "Run: python scripts/export_whisper.py --export-only")
        self._enc  = _ort_session(enc)
        self._dec  = _ort_session(dec)
        self._tok  = _tokenizer(language)

    # ── public ──────────────────────────────────────────────────────────────

    def transcribe(self, audio: np.ndarray) -> str:
        chunks = self._split(audio)
        return " ".join(filter(None, (self._chunk(c) for c in chunks))).strip()

    def transcribe_timed(self, audio: np.ndarray, offset: float = 0.0) -> list[dict]:
        """Returns list of {start, end, text} dicts."""
        results = []
        for i, chunk in enumerate(self._split(audio)):
            t0 = offset + i * 30.0
            t  = self._chunk(chunk)
            if t:
                results.append({"start": t0, "end": t0 + 30.0, "text": t})
        return results

    # ── internals ────────────────────────────────────────────────────────────

    def _split(self, audio: np.ndarray) -> list[np.ndarray]:
        n = max(1, int(np.ceil(len(audio) / N_SAMPLES)))
        return [audio[i * N_SAMPLES:(i + 1) * N_SAMPLES] for i in range(n)
                if len(audio[i * N_SAMPLES:(i + 1) * N_SAMPLES]) > 0]

    def _chunk(self, audio: np.ndarray) -> str:
        mel   = _log_mel(audio)
        kc, vc = self._enc.run(None, {"audio": mel})

        k_self = np.zeros((N_BLOCKS, N_HEADS, HEAD_DIM, MAX_TOKENS), np.float32)
        v_self = np.zeros((N_BLOCKS, N_HEADS, MAX_TOKENS, HEAD_DIM), np.float32)
        prompt = [TOKEN_SOT, TOKEN_LANG_EN, TOKEN_TRANSCRIBE]
        decoded = list(prompt)
        x = np.array([[prompt[-1]]], np.int32)

        for i in range(MAX_TOKENS):
            idx     = np.array([[len(decoded) - 1]], np.int32)
            logits, k_self, v_self = self._dec.run(
                None, {"x": x.astype(np.int32), "index": idx,
                       "k_cache_cross": kc, "v_cache_cross": vc,
                       "k_cache_self": k_self, "v_cache_self": v_self})
            lg = logits[0, -1].astype(np.float64)
            for t in _NON_SPEECH:
                lg[t] = -np.inf
            lg[TOKEN_NO_TIMESTAMP] = -np.inf
            lg[TOKEN_TS_BEGIN:]    = -np.inf
            if i == 0:
                lg[TOKEN_EOT] = lg[TOKEN_BLANK] = -np.inf
                m   = lg.max()
                lp  = lg - (m + np.log(np.sum(np.exp(lg - m))))
                if np.exp(lp[TOKEN_NO_SPEECH]) > NO_SPEECH_THR:
                    return ""
            nxt = int(np.argmax(lg))
            if nxt == TOKEN_EOT:
                break
            decoded.append(nxt)
            x = np.array([[nxt]], np.int32)

        payload = decoded[len(prompt):]
        if not payload:
            return ""
        try:
            return self._tok.decode(payload).strip()
        except Exception:
            return "".join(self._tok.decode([t]) for t in payload if t < 50257).strip()
