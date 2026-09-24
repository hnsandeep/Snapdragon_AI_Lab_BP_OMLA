# Complete Technical Documentation
## Offline Multilingual Lecture Assistant — Snapdragon AI Lab Build & Present Challenge

---

## Table of Contents

1. [Project Purpose](#1-project-purpose)
2. [Stack Overview](#2-stack-overview)
3. [Directory Reference](#3-directory-reference)
4. [Environment Variables](#4-environment-variables)
5. [Model Inventory](#5-model-inventory)
6. [Stage 1 — Audio Capture](#6-stage-1--audio-capture)
7. [Stage 2 — Speech-to-Text (Whisper)](#7-stage-2--speech-to-text-whisper)
8. [Stage 3 — Translation](#8-stage-3--translation)
9. [Stage 4 — LLM Summarization (GenieX)](#9-stage-4--llm-summarization-geniex)
10. [Pipeline Orchestration](#10-pipeline-orchestration)
11. [Desktop UI (Tkinter)](#11-desktop-ui-tkinter)
12. [AI Hub Export & Compile Scripts](#12-ai-hub-export--compile-scripts)
13. [Benchmarking](#13-benchmarking)
14. [Offline Guarantee](#14-offline-guarantee)
15. [Security & Privacy](#15-security--privacy)
16. [Performance Tuning](#16-performance-tuning)
17. [Known Limitations](#17-known-limitations)

---

## 1. Project Purpose

This project demonstrates a complete, production-quality AI pipeline that:

1. **Captures** microphone audio in real time from a live lecture or meeting
2. **Transcribes** speech to text using Whisper-base-en running on the Snapdragon X Elite NPU
3. **Translates** the transcript into a target Indic or European language using IndicTrans2 or NLLB-200
4. **Summarizes** the full session using a local Qwen3-0.6B LLM via GenieX, producing structured output (key points, action items, paragraph summary)

**Key constraint:** every byte of AI inference happens on the local device (Hexagon NPU / ONNX Runtime QNN EP). No audio, text, or transcript is ever sent to a remote server.

---

## 2. Stack Overview

| Layer | Technology | Version |
|---|---|---|
| Speech-to-text | Whisper-base-en ONNX | openai/whisper (20231117) |
| ORT runtime | onnxruntime-qnn | 2.6.0 (on-device) / onnxruntime 1.18.1 (dev) |
| Translation | IndicTrans2-dist-200M INT8 | TigreGotico/indictrans2-en-indic-dist-200M-onnx |
| Translation alt | NLLB-200-distilled-600M INT8 | facebook/nllb-200-distilled-600M |
| Model export | optimum[onnxruntime] | 1.19.2 |
| LLM | Qwen3-0.6B-GGUF Q4_0 | Qwen/Qwen3-0.6B-GGUF |
| LLM runtime | GenieX (geniex-llama-cpp) | latest |
| Model compile/profile | Qualcomm AI Hub | qai-hub 0.55.0 |
| Model catalog | qai-hub-models | 0.62.0 |
| Audio I/O | sounddevice | 0.4.7 |
| ML framework | PyTorch | 2.3.1 |
| UI | Tkinter (stdlib) | Python 3.11 built-in |
| Language | Python | 3.11 AMD x64 |

---

## 3. Directory Reference

```
Snapdragon_AI_lab/
│
├── audio_capture/
│   └── capture.py          AudioCapture class — mic → 30-s float32 chunks
│
├── transcription/
│   └── whisper_stt.py      WhisperSTT — ONNX encoder+decoder, greedy decode
│
├── translation/
│   └── translate.py        translate() — IndicTrans2 or NLLB-200 via ORT
│
├── summarization/
│   └── summarize.py        summarize() — GenieX LLM, SYSTEM_PROMPT embedded
│
├── pipeline.py             Pipeline + PipelineConfig + Segment dataclasses
│
├── ui/
│   └── app.py              Tkinter App — live transcript, summary, offline badge
│
├── bench/
│   └── benchmark.py        AI Hub compile + profile for all 8 model components
│
├── scripts/
│   ├── setup_aihub.py      Configure API token, list devices
│   ├── verify_hub_job.py   Smoke-test compile+profile pipeline
│   ├── export_whisper.py   Export Whisper ONNX → compile → profile
│   ├── export_translation.py  Export IndicTrans2/NLLB → INT8 → compile
│   └── export_summarizer.py   Export BART/mT5 → INT8 → compile
│
├── config/
│   └── settings.py         Centralised env-var config (all model paths, langs)
│
├── models/                 Model weights (git-ignored, ~8 GB total)
│   ├── WhisperEncoder.onnx
│   ├── WhisperDecoder.onnx
│   ├── mel_filters.npz
│   ├── indictrans2/
│   ├── nllb200/
│   ├── bart/
│   └── mt5/
│
├── docs/                   Documentation suite (this file and siblings)
├── .env                    Your secrets — never commit
├── .env.example            Template
├── .gitignore
├── .python-version         Pins 3.11 for pyenv/mise
├── requirements.txt
├── README.md
└── CONTRIBUTING.md
```

---

## 4. Environment Variables

All configuration is driven by `.env`. Copy `.env.example` to `.env` and fill in.

| Variable | Required | Default | Description |
|---|---|---|---|
| `QAI_HUB_API_TOKEN` | 🔑 Yes | — | AI Hub API token for compile/profile jobs |
| `QAI_HUB_DEVICE` | No | `Snapdragon X Elite CRD` | Target device name on AI Hub |
| `SOURCE_LANGUAGE` | No | `en` | ISO 639-1 source language for Whisper |
| `TARGET_LANGUAGE` | No | `hin_Deva` | ISO 639-3+script target language for translation |
| `TRANSLATE_MODEL` | No | `indictrans2` | `indictrans2` \| `nllb200` \| `transformers` |
| `WHISPER_ENCODER_ONNX` | No | `models/WhisperEncoder.onnx` | Path to Whisper encoder ONNX |
| `WHISPER_DECODER_ONNX` | No | `models/WhisperDecoder.onnx` | Path to Whisper decoder ONNX |
| `IT2_ENCODER_ONNX` | No | `models/indictrans2/int8/encoder_model_compiled.onnx` | IndicTrans2 encoder path |
| `IT2_DECODER_ONNX` | No | `models/indictrans2/int8/decoder_model_compiled.onnx` | IndicTrans2 decoder path |
| `NLLB_ENCODER_ONNX` | No | `models/nllb200/int8/encoder_model_compiled.onnx` | NLLB encoder path |
| `NLLB_DECODER_ONNX` | No | `models/nllb200/int8/decoder_model_compiled.onnx` | NLLB decoder path |
| `SUMMARY_MODEL` | No | `bart` | `bart` \| `mt5` |
| `SUMMARY_ENCODER_ONNX` | No | `models/bart/int8/encoder_model_compiled.onnx` | Summarizer encoder path |
| `SUMMARY_DECODER_ONNX` | No | `models/bart/int8/decoder_model_compiled.onnx` | Summarizer decoder path |
| `GENIEX_MODEL` | No | `Qwen/Qwen3-0.6B-GGUF` | GenieX model repo or alias |
| `GENIEX_DEVICE_MAP` | No | `npu` | `npu` \| `cpu` \| `gpu` \| `auto` |
| `GENIEX_PRECISION` | No | `Q4_0` | GGUF quantization (Q4_0 = best NPU support) |
| `GENIEX_MAX_TOKENS` | No | `512` | Max tokens GenieX generates per summary |

---

## 5. Model Inventory

| Model | Size (INT8) | Parameters | Task | Runtime |
|---|---|---|---|---|
| Whisper-base-en Encoder | 90.7 MB | 23.7M | Mel → cross-attention KV | QNN EP |
| Whisper-base-en Decoder | 187 MB | 48.9M | Autoregressive token generation | QNN EP |
| IndicTrans2-dist-200M (enc+dec) | 472 MB | 200M | EN→22 Indic languages | QNN EP |
| NLLB-200-distilled-600M (enc+dec) | ~630 MB | 600M | 200 language pairs | QNN EP |
| BART-large-cnn (enc+dec) | ~410 MB | 400M | English summarization | QNN EP |
| mT5-XLSum (enc+dec) | ~560 MB | 300M | Multilingual summarization | QNN EP |
| Qwen3-0.6B-GGUF Q4_0 | ~400 MB | 600M | Structured LLM summarization | GenieX NPU |

**Total disk usage (all models):** ~3.2 GB (INT8 + GGUF variants)

---

## 6. Stage 1 — Audio Capture

**File:** `audio_capture/capture.py`
**Class:** `AudioCapture`

Captures microphone input via `sounddevice.InputStream` (PortAudio). Audio is
buffered in a thread-safe `np.ndarray` and dispatched in 30-second chunks.

### Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `on_chunk` | `Callable[[np.ndarray], None]` | required | Called with each 30-s float32 chunk |
| `sample_rate` | `int` | `16000` | Sample rate in Hz — must be 16 kHz for Whisper |
| `chunk_seconds` | `float` | `30.0` | Chunk length in seconds |
| `device` | `int \| None` | `None` | sounddevice device index (None = default) |

### Design Notes

- The PortAudio callback (`_cb`) appends samples to a ring buffer and immediately spawns a daemon thread per completed chunk. This keeps the audio callback latency under 1 ms.
- `stop()` flushes the remaining buffer as a final partial chunk.
- Call `AudioCapture.list_devices()` to see device indices.

---

## 7. Stage 2 — Speech-to-Text (Whisper)

**File:** `transcription/whisper_stt.py`
**Class:** `WhisperSTT`

Implements full Whisper-base-en encoder-decoder inference in ONNX Runtime.

### Mel Spectrogram

Input audio (float32, 16 kHz) is converted to a log-mel spectrogram:
- STFT: 400-sample window, 160-sample hop
- 80 mel bands, 3000 time frames (= 30 seconds)
- Output shape: `(1, 80, 3000)` float32

### Encoder

- Input: mel spectrogram `(1, 80, 3000)`
- Output: cross-attention KV caches `k_cross (6,8,64,1500)` and `v_cross (6,8,1500,64)`
- Latency on X Elite NPU: **~49 ms**

### Decoder (auto-regressive loop)

- Per-step inputs: current token `x (1,1)`, position `index (1,1)`, both KV caches
- Per-step output: logits over 51,865-token vocabulary, updated self-attention caches
- Maximum 224 auto-regressive steps
- Latency: **~3.6 ms/step** on X Elite NPU
- Full 30-second chunk: **~770 ms total** (~39× real-time)

### Non-Speech Detection

On the first decoder step, if the no-speech token probability exceeds 0.6,
the chunk returns an empty string without running the full decode loop.

### Mel Filter Cache

`models/mel_filters.npz` stores the 80-band mel filter matrix (shape `80×201`).
Generated once by `save_mel_filters_from_whisper()`. If not present, extracted
live from the `openai-whisper` package; final fallback is a scipy-computed approximation.

---

## 8. Stage 3 — Translation

**File:** `translation/translate.py`
**Function:** `translate(text, src, tgt, backend, num_beams)`

### IndicTrans2 Backend (default)

- Model: `TigreGotico/indictrans2-en-indic-dist-200M-onnx` (INT8 sub-folder)
- Mandatory: `IndicTransToolkit.IndicProcessor` for pre/post-processing
  - `preprocess_batch()` normalises script, adds language tags, protects entities
  - `postprocess_batch()` reverses entity protection
- Skipping `IndicProcessor` produces fluent-looking but wrong-language output
- Supports all 22 scheduled Indian languages (eng_Latn → any Indic tag)

### NLLB-200 Backend

- Model: `facebook/nllb-200-distilled-600M`
- Standard `NllbTokenizer` from HuggingFace transformers
- 200 language pairs via ISO 639-3+script tags
- Use for non-Indic targets (German, French, Spanish, Arabic, Chinese, etc.)

### OPUS-MT Backend (transformers)

- Helsinki-NLP OPUS-MT models, HuggingFace pipeline
- No ONNX files required — for off-device development only
- Limited language pairs

### Language Tag Format

Both IndicTrans2 and NLLB-200 use the format `<iso639-3>_<ISO15924-script>`:
- `eng_Latn` — English (Latin script)
- `hin_Deva` — Hindi (Devanagari)
- `kan_Knda` — Kannada
- `tam_Taml` — Tamil
- `deu_Latn` — German

---

## 9. Stage 4 — LLM Summarization (GenieX)

**File:** `summarization/summarize.py`
**Function:** `summarize(transcript, stream, on_token)`

### GenieX Runtime

GenieX is Qualcomm's community on-device Gen AI inference runtime.
Three mutually-exclusive Python packages share the same `geniex` namespace:

| Package | Backend | Best for |
|---|---|---|
| `geniex-llama-cpp` | llama.cpp + ggml-hexagon | GGUF models, Hexagon NPU Q4_0 |
| `geniex-qairt` | Qualcomm AI Engine Direct | AI Hub pre-compiled bundles |
| `geniex` | Meta-package | Lets pip choose best variant |

This project uses `geniex-llama-cpp` with `Qwen/Qwen3-0.6B-GGUF` at Q4_0 precision,
which delivers the best Hexagon NPU acceleration in llama.cpp.

### System Prompt (exact, as required)

```
You are summarizing a lecture or meeting transcript for someone who could not
attend. Be faithful to the source — do not invent facts, names, or numbers not
present in the transcript. If a section is unclear or incomplete, say so rather
than guessing. Output in this exact format:
## Key Points (max 5 bullets) ## Action Items (or 'None mentioned') ## Summary
(120 words max, plain language).
```

### Output Format

```markdown
## Key Points
- Bullet 1
- Bullet 2
- (max 5 bullets)

## Action Items
- Action 1
(or "None mentioned" if no action items detected)

## Summary
One paragraph, maximum 120 words, plain language.
```

### Streaming Mode

When `stream=True` is passed, GenieX returns a `TextIteratorStreamer` whose tokens
are forwarded to the `on_token` callback as they are generated. The Tkinter UI uses
this to render tokens progressively without blocking the main thread.

### Token Budget

- Input: full translated transcript (up to ~4,000 tokens for a 30-minute lecture)
- Output: max 512 tokens (set via `GENIEX_MAX_TOKENS`)
- Temperature: 0.1 (near-deterministic — appropriate for factual summarization)

---

## 10. Pipeline Orchestration

**File:** `pipeline.py`
**Classes:** `Pipeline`, `PipelineConfig`, `Segment`

### PipelineConfig

```python
@dataclass
class PipelineConfig:
    src_lang:           str   = "en"
    tgt_lang:           str   = "hin_Deva"
    translate_backend:  str   = "indictrans2"
    whisper_backend:    str   = "onnx"
    translate_enabled:  bool  = True
    chunk_seconds:      float = 30.0
    mic_device:         Optional[int] = None
```

### Segment

```python
@dataclass
class Segment:
    start_s:  float   # wall-clock start of this chunk
    end_s:    float   # wall-clock end
    text_src: str     # source-language transcript
    text_tgt: str     # translated text (empty if translation disabled)
```

### Processing Flow

```
AudioCapture.on_chunk(audio)
    └── Pipeline._on_audio_chunk(audio)
            1. WhisperSTT.transcribe(audio)       → text_src
            2. translate(text_src, src, tgt)       → text_tgt
            3. Segment(t0, t1, text_src, text_tgt)
            4. self.on_segment(segment)             → UI callback
            5. self._segments.append(segment)
```

### Summary Call

```python
pipeline.summarize(on_token=callback)
    └── full_text = " ".join(s.text_tgt or s.text_src for s in segments)
    └── geniex_summarize(full_text, stream=True, on_token=callback)
```

### Thread Safety

All segment list access is protected by `self._seg_lock` (threading.Lock).
Summarization is non-blocking — called in its own daemon thread by the UI.

---

## 11. Desktop UI (Tkinter)

**File:** `ui/app.py`
**Class:** `App(tk.Tk)`

### Layout

```
┌─────────────────────────────────────────────────────────────────────┐
│  🔒 OFFLINE MODE: ON  |  No network calls  |  All inference on-device │
├─────────────────────────────────────────────────────────────────────┤
│  Title: Offline Multilingual Lecture Assistant                      │
├─────────────────────────────────────────────────────────────────────┤
│  [Source ▼] → [Target ▼]  [▶ Start]  [⏹ Stop]  [✦ Summary]  Status │
├───────────────────────────────┬─────────────────────────────────────┤
│  Live Transcript              │  Summary (GenieX NPU)               │
│  (ScrolledText, auto-scroll)  │  (ScrolledText, token-streaming)    │
│                               │                                     │
│  [0.0s – 30.0s]               │  ## Key Points                      │
│  Transcript text here…        │  - bullet 1                         │
│                               │  …                                  │
├───────────────────────────────┴─────────────────────────────────────┤
│  Segments: 12                    Snapdragon X Elite  |  QNN/NPU     │
└─────────────────────────────────────────────────────────────────────┘
```

### Thread Safety Pattern

All widget updates from background threads use `self.after(0, fn, arg)`:
```python
# In worker thread:
self.after(0, self._append_segment, segment)

# In main thread:
def _append_segment(self, seg):
    self._transcript_box.config(state="normal")
    self._transcript_box.insert("end", text)
    self._transcript_box.config(state="disabled")
```

### Offline Badge

The green badge at the top is the centrepiece of the demo pitch.
It uses a distinct `BADGE_BG` / `BADGE_FG` colour pair (dark green / bright green)
to draw immediate attention. It is permanently visible and cannot be hidden.

---

## 12. AI Hub Export & Compile Scripts

### scripts/export_whisper.py

| Flag | Description |
|---|---|
| `--export-only` | ONNX export only, skip AI Hub |
| `--compile-only` | Use existing ONNX files, just compile |
| `--model` | `whisper_base_en` (default) or `whisper_small_en` |
| `--device` | AI Hub device name |
| `--runtime` | `onnx` (default), `tflite`, `qnn` |

### scripts/export_translation.py

| Flag | Description |
|---|---|
| `--model` | `indictrans2`, `nllb200`, or `both` |
| `--export-only` | Download/export ONNX, skip AI Hub |
| `--fp32` | Skip INT8 quantization |
| `--profile-only` | Profile already-compiled models |

### scripts/export_summarizer.py

| Flag | Description |
|---|---|
| `--model` | `bart` (default) or `mt5` |
| `--export-only` | Export ONNX + quantize, skip AI Hub |
| `--fp32` | Skip INT8 quantization |

### AI Hub Compile Options

All compile jobs use `--target_runtime onnx` which produces a
**PRECOMPILED_QNN_ONNX** artifact — an ONNX file with an embedded QNN context
binary loadable by ONNX Runtime via the QNN Execution Provider.

---

## 13. Benchmarking

**File:** `bench/benchmark.py`

Profiles all 8 model components on a cloud-hosted Snapdragon X Elite CRD.
Outputs `bench/results.md` — a Markdown table with compile time, latency,
peak memory, and primary compute unit.

```powershell
python bench/benchmark.py                        # all models
python bench/benchmark.py --models whisper_encoder whisper_decoder
python bench/benchmark.py --device "Snapdragon X Elite CRD"
```

See [BENCHMARK_GUIDE.md](BENCHMARK_GUIDE.md) for full instructions.

---

## 14. Offline Guarantee

The following table lists every component and confirms no outbound network call:

| Component | Network call? | Verification |
|---|---|---|
| `sounddevice.InputStream` | No | OS audio API only |
| `onnxruntime.InferenceSession` | No | Local .onnx file I/O |
| `ORTModelForSeq2SeqLM.generate()` | No | Local ONNX + in-memory |
| `IndicProcessor.preprocess/postprocess` | No | Pure Python string processing |
| `geniex AutoModelForCausalLM.generate()` | No | Local .gguf file, no callbacks |
| `tkinter` | No | OS GUI only |
| `numpy`, `scipy`, `torch` | No | Compute only |

**First-time setup** (downloads, one-time only):
- HuggingFace model downloads (IndicTrans2, NLLB-200, BART ONNX)
- GenieX model pull (Qwen3-0.6B GGUF)
- AI Hub compile jobs (benchmarking only)

After setup, the application runs indefinitely with no internet access.

---

## 15. Security & Privacy

- No audio leaves the device. Ever.
- No transcript is logged to disk unless you add logging explicitly.
- The AI Hub API token is stored in `~/.qai_hub/client.ini` (not in the repo).
- `.env` is in `.gitignore` — never committed.
- Model weights are in `models/` which is also `.gitignore`d.

---

## 16. Performance Tuning

### Whisper

- Use `WhisperEncoder_compiled.onnx` (AI Hub compiled) for maximum NPU throughput.
- Set `htp_performance_mode=burst` (already default in the code).
- For lower latency at cost of accuracy, switch to `whisper_tiny_en`.

### Translation

- IndicTrans2 INT8 (472 MB) is significantly faster than FP32 (1.87 GB) at minimal BLEU cost.
- Set `num_beams=1` in `translate()` for greedy decoding (fastest, lower quality).

### GenieX / LLM

- Q4_0 is the only GGUF precision with first-class Hexagon NPU support in llama.cpp.
- Reduce `GENIEX_MAX_TOKENS` to `256` for faster summary generation.
- Use `GENIEX_DEVICE_MAP=cpu` on non-Snapdragon machines for development.

### General

- Keep only one `onnxruntime` variant installed (either `onnxruntime` or `onnxruntime-qnn`, never both).
- ORT session `graph_optimization_level = ORT_ENABLE_ALL` is set globally.
- `intra_op_num_threads = 4` is a safe default for the X Elite; increase to 8 for CPU-only inference.

---

## 17. Known Limitations

| Limitation | Workaround |
|---|---|
| Whisper-base accuracy degrades on heavy accents | Switch to `whisper_small_en` (higher latency) |
| IndicTrans2 only supports English → Indic direction | Use NLLB-200 for Indic→English or other pairs |
| GenieX model requires ~400 MB disk on first pull | Pre-pull with the Python snippet in HOW_TO_RUN.md |
| ONNX Runtime and onnxruntime-qnn conflict | Install only one; see requirements.txt comments |
| Gradio UI not used (replaced by Tkinter) | Tkinter is more reliable for live offline demos |
| Bench requires AI Hub token | Use `--export-only` flags for local-only operation |
| Python 3.14 not supported | Install Python 3.11 AMD x64 |
