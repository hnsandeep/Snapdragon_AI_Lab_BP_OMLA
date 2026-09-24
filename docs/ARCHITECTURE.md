# Architecture — Offline Multilingual Lecture Assistant

---

## Overview

The system is a four-stage, fully offline pipeline running on a single Snapdragon X Elite laptop.
No component makes any network call during runtime. All AI inference uses the Hexagon NPU
via ONNX Runtime's QNN Execution Provider or GenieX's llama.cpp backend.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    SNAPDRAGON X ELITE LAPTOP                                │
│                                                                             │
│  Microphone                                                                 │
│      │  PCM audio @ 16 kHz, float32 mono                                   │
│      ▼                                                                      │
│  ┌─────────────────┐                                                        │
│  │  AudioCapture   │  sounddevice / PortAudio                               │
│  │  (30-s chunks)  │  rolling ring buffer, fires on_chunk callback          │
│  └────────┬────────┘                                                        │
│           │  np.ndarray(480000,) float32                                    │
│           ▼                                                                 │
│  ┌─────────────────┐                                                        │
│  │  WhisperSTT     │  Whisper-base-en ONNX                                  │
│  │  (ONNX Runtime) │  ┌──────────────────────────────────────────────┐     │
│  │                 │  │ QNNExecutionProvider (QnnHtp.dll)             │     │
│  │  Encoder→       │  │   Hexagon NPU · FP16 · ~49 ms/chunk          │     │
│  │  Decoder loop   │  │ CPUExecutionProvider (fallback)               │     │
│  └────────┬────────┘  └──────────────────────────────────────────────┘     │
│           │  str  (transcript text)                                         │
│           ▼                                                                 │
│  ┌─────────────────┐                                                        │
│  │  Translator     │  IndicTrans2-dist-200M  OR  NLLB-200-600M             │
│  │  (ORT Seq2Seq)  │  INT8 dynamic quantized ONNX                          │
│  │                 │  ORTModelForSeq2SeqLM + IndicProcessor                │
│  │                 │  QNN EP → CPU EP fallback                              │
│  └────────┬────────┘                                                        │
│           │  str  (translated text)                                         │
│           ▼                                                                 │
│  ┌─────────────────┐                                                        │
│  │  GenieX LLM     │  Qwen3-0.6B-GGUF @ Q4_0                              │
│  │  (llama.cpp)    │  geniex-llama-cpp · Hexagon NPU                       │
│  │                 │  Structured output: Key Points / Actions / Summary     │
│  └────────┬────────┘                                                        │
│           │  str  (structured summary)                                      │
│           ▼                                                                 │
│  ┌─────────────────┐                                                        │
│  │  Tkinter UI     │  Live transcript scroll · Language toggle             │
│  │  (ui/app.py)    │  Generate Summary button · Offline badge              │
│  └─────────────────┘                                                        │
│                                                                             │
│  🔒  ZERO NETWORK CALLS AT RUNTIME                                          │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Component Details

### 1. AudioCapture (`audio_capture/capture.py`)

```
sounddevice.InputStream (PortAudio)
    │
    ├── 16 kHz, mono, float32
    ├── Ring buffer (_buf: np.ndarray)
    └── Fires on_chunk(audio: np.ndarray) when buffer ≥ 480,000 samples (30 s)
        └── Worker thread dispatched per chunk (non-blocking)
```

Key design decisions:
- The sounddevice callback is kept minimal (just buffer append + length check)
  to avoid blocking PortAudio's real-time audio thread.
- Tail flush on `stop()` ensures the final partial chunk is not lost.
- Chunk size is configurable (`chunk_seconds` parameter) — the default 30 s
  matches Whisper's fixed-length input exactly.

---

### 2. WhisperSTT (`transcription/whisper_stt.py`)

```
Audio (float32 mono, 16 kHz)
    │
    ├── _log_mel(audio)
    │       torch.stft → magnitudes → mel_filter_bank @ magnitudes
    │       → log10 → normalize → shape (1, 80, 3000)
    │
    ├── Encoder ORT session
    │       Input:  audio (1, 80, 3000) float32
    │       Output: k_cache_cross (6,8,64,1500)
    │               v_cache_cross (6,8,1500,64)
    │
    └── Decoder ORT session  [loop ≤ 224 tokens]
            Inputs:  x (1,1) int32        current token
                     index (1,1) int32    position
                     k/v_cache_cross      frozen encoder output
                     k/v_cache_self       updated per step
            Output:  logits, k_cache_self_new, v_cache_self_new
            → argmax → token → decode via Whisper tokenizer
```

**QNN EP configuration:**
```python
provider_options = [{
    "backend_path":                             "QnnHtp.dll",
    "htp_performance_mode":                     "burst",
    "enable_htp_fp16_precision":                "1",
    "htp_graph_finalization_optimization_mode": "3",
}]
```

Fallback: if `QNNExecutionProvider` is not in `ort.get_available_providers()`,
the session silently uses `CPUExecutionProvider`.

---

### 3. Translator (`translation/translate.py`)

```
Text (source language)
    │
    ├── [indictrans2 backend]
    │       IndicProcessor.preprocess_batch()   ← mandatory
    │       AutoTokenizer.encode()
    │       ORTModelForSeq2SeqLM.generate()     ← encoder + decoder ONNX
    │       AutoTokenizer.decode()
    │       IndicProcessor.postprocess_batch()  ← mandatory
    │
    └── [nllb200 backend]
            NllbTokenizer.encode(src_lang=...)
            ORTModelForSeq2SeqLM.generate(forced_bos_token_id=tgt_lang_id)
            NllbTokenizer.decode()
```

**Why ORTModelForSeq2SeqLM instead of raw ORT sessions?**
Seq2seq generation requires a three-graph export (encoder, decoder-first-step,
decoder-with-past). `optimum` manages the KV-cache threading between graphs
correctly. The QNN EP is still active at the session level inside ORTModel.

**Why IndicTrans2 over NLLB for Indic languages?**
IndicTrans2 was trained on all 22 scheduled Indian languages with in-domain data.
On Hindi, Tamil, and Kannada it beats NLLB-200-600M by 5–12 BLEU points.
NLLB-200 remains the choice for non-Indic targets.

**INT8 quantization:**
Dynamic INT8 (weight-only, no calibration data needed) reduces model size ~4×.
IndicTrans2-200M: 1.87 GB → 472 MB. Exact-match parity ~60%; BLEU loss < 1 pt.

---

### 4. GenieX Summarizer (`summarization/summarize.py`)

```
Full transcript (str)
    │
    ├── messages = [
    │       {"role": "system", "content": SYSTEM_PROMPT},
    │       {"role": "user",   "content": f"Transcript:\n\n{transcript}"}
    │   ]
    │
    ├── model.tokenizer.apply_chat_template(messages)
    │
    └── model.generate(prompt, max_new_tokens=512, temperature=0.1)
            ↓
        ## Key Points
        - bullet 1 … (max 5)
        ## Action Items
        - item 1 …  (or "None mentioned")
        ## Summary
        One paragraph, ≤ 120 words
```

**GenieX runtime path:**
```
geniex-llama-cpp package
    └── llama.cpp backend
            └── Hexagon NPU via ggml-hexagon
                    └── Q4_0 quantized weights
                            └── Qwen3-0.6B parameters
```

**Why Qwen3-0.6B?**
- Smallest model that reliably follows structured output instructions.
- Q4_0 GGUF = best Hexagon NPU support in llama.cpp.
- ~400 MB download, cached locally after first pull.
- Fits in NPU memory alongside the other models.

**CPU fallback:** set `GENIEX_DEVICE_MAP=cpu` in `.env`.

---

### 5. Pipeline (`pipeline.py`)

```
PipelineConfig
    src_lang, tgt_lang, translate_backend,
    whisper_backend, translate_enabled, chunk_seconds

Pipeline
    ├── AudioCapture(on_chunk=_on_audio_chunk)
    ├── WhisperSTT
    └── on_segment callback (fires for each processed chunk)

_on_audio_chunk(audio):
    1. WhisperSTT.transcribe(audio)          → text_src
    2. translate(text_src, src, tgt)         → text_tgt
    3. Segment(start_s, end_s, src, tgt)
    4. on_segment(segment)                   → UI update

Pipeline.summarize():
    → geniex_summarize(full_transcript)      → structured string
```

---

### 6. Tkinter UI (`ui/app.py`)

```
App (tk.Tk)
├── Badge frame         "🔒 OFFLINE MODE: ON | No network calls"
├── Controls row
│   ├── Source language  (Combobox)
│   ├── Target language  (Combobox)
│   ├── ▶ Start Recording  → pipeline.start()
│   ├── ⏹ Stop            → pipeline.stop()
│   └── ✦ Generate Summary → threading.Thread → pipeline.summarize()
├── PanedWindow (horizontal)
│   ├── Left:  ScrolledText  "Live Transcript"
│   │           updated via after(0, _append_segment) from worker threads
│   └── Right: ScrolledText  "Summary (GenieX NPU)"
│               tokens streamed via after(0, _append_summary_token)
└── Status bar          segment count + device label
```

Thread safety: all Tkinter widget updates are dispatched via `self.after(0, fn)`
from worker threads. Tkinter's mainloop is single-threaded; never call widget
methods directly from a background thread.

---

## Data Flow Timing (Snapdragon X Elite, NPU)

```
t=0.0 s   Mic chunk ready (30 s of audio collected)
t=0.049   Whisper Encoder done (~49 ms)
t=0.769   Whisper Decoder done (~720 ms for ~200 tokens)
t=0.869   IndicTrans2 Encoder + beam decode (~100 ms)
           ──────────────────────────────────────────
t≈0.9 s   Segment displayed in UI  (< 1 s after chunk end)

On Generate Summary click:
t=0.0     First GenieX token (TTFT ~300–500 ms on NPU)
t≈8–15 s  All 512 tokens decoded (streaming into UI)
```

---

## Offline Guarantee

Every component is verified to make zero outbound network calls:
- `sounddevice` — local OS audio API only
- `onnxruntime` — local file I/O only
- `ORTModelForSeq2SeqLM` — local ONNX files only
- `GenieX` — local GGUF file only
- `Tkinter` — local OS GUI only
- `translate()` / `summarize()` — no `requests`, no `urllib`, no socket calls

The only network-enabled operations are one-time setup:
- First model download (HuggingFace / GenieX model pull)
- AI Hub compile/profile jobs (benchmarking only, not runtime)

---

## AI Hub Integration (Benchmarking Only)

```
bench/benchmark.py
    │
    ├── hub.upload_model(onnx_bytes)           upload ONNX to AI Hub
    ├── hub.submit_compile_job(                compile → PRECOMPILED_QNN_ONNX
    │       input_specs, target_runtime=onnx)
    ├── compile_job.wait()
    ├── hub.submit_profile_job(compiled_model) run on hosted Snapdragon X Elite
    ├── profile_job.wait()
    └── profile_job.download_profile()         latency, memory, compute unit
            └── bench/results.md               Markdown table
```

The compiled `.onnx` artifacts downloaded from AI Hub are what go into
`models/` and get loaded by the runtime pipeline.

---

> **Related:** [API_REFERENCE.md](API_REFERENCE.md) for class/function signatures.
