# API Reference — Offline Multilingual Lecture Assistant

All public classes and functions, with signatures, parameters, and examples.

---

## audio_capture.capture

### `class AudioCapture`

Continuous microphone capture with 30-second chunk callbacks.

```python
from audio_capture.capture import AudioCapture

cap = AudioCapture(on_chunk=my_callback, chunk_seconds=30.0)
cap.start()
# ... record ...
cap.stop()
```

#### `__init__(on_chunk, sample_rate=16000, chunk_seconds=30.0, device=None)`

| Parameter | Type | Description |
|---|---|---|
| `on_chunk` | `Callable[[np.ndarray], None]` | Called with each chunk. Array is float32, shape `(N,)`, 16 kHz mono. |
| `sample_rate` | `int` | Microphone sample rate. Must be 16000 for Whisper compatibility. |
| `chunk_seconds` | `float` | Chunk length in seconds. Default 30.0 matches Whisper's fixed input. |
| `device` | `int \| None` | sounddevice device index. `None` uses the OS default. |

#### `start() → None`
Open the microphone stream and begin buffering. Idempotent.

#### `stop() → None`
Stop recording, flush remaining buffered audio as a final partial chunk, close the stream.

#### `list_devices() → None` (static)
Print all available audio input/output devices to stdout.

---

## transcription.whisper_stt

### `class WhisperSTT`

ONNX-based Whisper-base-en inference. QNN HTP EP → CPU EP fallback.

```python
from transcription.whisper_stt import WhisperSTT

stt = WhisperSTT(language="en")
text = stt.transcribe(audio_np)           # str
segs = stt.transcribe_timed(audio_np)     # list[dict]
```

#### `__init__(language="en", encoder_path=None, decoder_path=None)`

| Parameter | Type | Description |
|---|---|---|
| `language` | `str` | ISO 639-1 language code (default `"en"`). Used to set the language token in the decoder prompt. |
| `encoder_path` | `str \| None` | Path to encoder `.onnx`. Defaults to `WHISPER_ENCODER_ONNX` env var or `models/WhisperEncoder.onnx`. |
| `decoder_path` | `str \| None` | Path to decoder `.onnx`. Defaults to `WHISPER_DECODER_ONNX` env var or `models/WhisperDecoder.onnx`. |

Raises `FileNotFoundError` if encoder is not found.

#### `transcribe(audio: np.ndarray) → str`

Transcribe a mono float32 audio array of any length (auto-split into 30-second chunks).

| Parameter | Type | Description |
|---|---|---|
| `audio` | `np.ndarray` | float32, shape `(N,)`, 16 kHz mono |

Returns the full transcript as a single string. Returns `""` if no speech is detected.

#### `transcribe_timed(audio: np.ndarray, offset: float = 0.0) → list[dict]`

Returns a list of dicts, one per 30-second chunk:
```python
[{"start": 0.0, "end": 30.0, "text": "Hello world"}]
```

| Parameter | Type | Description |
|---|---|---|
| `audio` | `np.ndarray` | float32, shape `(N,)`, 16 kHz |
| `offset` | `float` | Wall-clock offset in seconds to add to `start`/`end` |

### `save_mel_filters_from_whisper() → None` (module-level)

One-time helper. Extracts the 80-band mel filter matrix from the `openai-whisper`
package and saves it to `models/mel_filters.npz`. Run once after installing openai-whisper:

```powershell
python -c "from transcription.whisper_stt import save_mel_filters_from_whisper; save_mel_filters_from_whisper()"
```

---

## translation.translate

### `translate(text, src="eng_Latn", tgt="hin_Deva", backend=None, num_beams=4) → str`

Translate a string offline. All inference is local (QNN EP → CPU fallback).

```python
from translation.translate import translate

result = translate("The lecture starts at 9 AM.", "eng_Latn", "hin_Deva")
print(result)  # "व्याख्यान सुबह 9 बजे शुरू होती है।"
```

| Parameter | Type | Description |
|---|---|---|
| `text` | `str` | Source text to translate |
| `src` | `str` | Source language tag (ISO 639-3+script). Default `"eng_Latn"` |
| `tgt` | `str` | Target language tag. Default `"hin_Deva"` |
| `backend` | `str \| None` | `"indictrans2"` \| `"nllb200"` \| `"transformers"`. `None` reads from `TRANSLATE_MODEL` env var. |
| `num_beams` | `int` | Beam search width. `1` = greedy (fast), `4` = default, `5+` = higher quality |

Returns translated string. Returns `""` for empty input.

The model is loaded lazily on first call and cached as a module-level singleton.

### `translate_batch(texts, src, tgt, backend=None) → list[str]`

Translate a list of strings. Currently calls `translate()` sequentially.

---

## summarization.summarize

### `SYSTEM_PROMPT: str` (module-level constant)

The exact system prompt embedded in every GenieX call:
```
"You are summarizing a lecture or meeting transcript for someone who could not
attend. Be faithful to the source — do not invent facts, names, or numbers not
present in the transcript. If a section is unclear or incomplete, say so rather
than guessing. Output in this exact format:
## Key Points (max 5 bullets) ## Action Items (or 'None mentioned') ## Summary
(120 words max, plain language)."
```

### `summarize(transcript, stream=False, on_token=None) → str`

Run the full transcript through the local GenieX LLM and return structured output.
Zero network calls. All inference on-device (Hexagon NPU via llama.cpp Q4_0).

```python
from summarization.summarize import summarize

# One-shot
result = summarize(full_transcript)
print(result)

# Streaming (tokens delivered as generated)
def on_tok(token: str):
    print(token, end="", flush=True)

result = summarize(full_transcript, stream=True, on_token=on_tok)
```

| Parameter | Type | Description |
|---|---|---|
| `transcript` | `str` | Full translated transcript text |
| `stream` | `bool` | If `True`, tokens are passed to `on_token` as generated |
| `on_token` | `Callable[[str], None] \| None` | Token callback for streaming mode |

Returns the complete generated text (all three sections). When `stream=True`,
also streams tokens to `on_token` before returning.

### `close() → None`

Release the GenieX model handle and free NPU resources. Call when the application exits.

---

## pipeline

### `class PipelineConfig`

```python
from pipeline import PipelineConfig

cfg = PipelineConfig(
    src_lang="en",
    tgt_lang="hin_Deva",
    translate_backend="indictrans2",
    whisper_backend="onnx",
    translate_enabled=True,
    chunk_seconds=30.0,
    mic_device=None,
)
```

| Field | Type | Default | Description |
|---|---|---|---|
| `src_lang` | `str` | `"en"` | ISO 639-1 source language for Whisper |
| `tgt_lang` | `str` | `"hin_Deva"` | ISO 639-3+script target for translation |
| `translate_backend` | `str` | `"indictrans2"` | Translation backend |
| `whisper_backend` | `str` | `"onnx"` | `"onnx"` or `"whisper"` (PyTorch) |
| `translate_enabled` | `bool` | `True` | Set `False` to skip translation |
| `chunk_seconds` | `float` | `30.0` | Mic chunk length |
| `mic_device` | `int \| None` | `None` | sounddevice device index |

### `class Segment`

```python
@dataclass
class Segment:
    start_s:  float   # seconds from recording start
    end_s:    float
    text_src: str     # original transcript
    text_tgt: str     # translated text (empty if translation disabled)
```

`str(segment)` → `"[ 0.0s– 30.0s]  Translated text here"`

### `class Pipeline`

```python
from pipeline import Pipeline, PipelineConfig

cfg  = PipelineConfig(src_lang="en", tgt_lang="kan_Knda")
pipe = Pipeline(cfg)

# Optional: callback for each completed segment
pipe.on_segment = lambda seg: print(seg)

pipe.start()
# ... recording in progress ...
pipe.stop()

# Get all segments
segs = pipe.get_segments()          # list[Segment]
text = pipe.get_full_transcript()   # str (translated)

# Run LLM summarization (blocking, unless on_token callback provided)
summary = pipe.summarize()
# or streaming:
summary = pipe.summarize(on_token=lambda tok: print(tok, end="", flush=True))

pipe.close()    # releases GenieX NPU resources
```

#### Methods

| Method | Returns | Description |
|---|---|---|
| `start()` | `None` | Open mic, begin capture+transcription+translation |
| `stop()` | `None` | Stop mic, flush remaining audio |
| `summarize(on_token=None)` | `str` | Run GenieX over full transcript |
| `get_segments()` | `list[Segment]` | Thread-safe copy of all segments |
| `get_full_transcript(translated=True)` | `str` | Joined text of all segments |
| `close()` | `None` | Stop pipeline and release all resources |

---

## bench.benchmark

### `_compile_and_profile(hub, key, spec, device) → dict`

Internal — used by `main()`. Returns a result dict with keys:
`label`, `compile_s`, `latency_ms`, `peak_mem_mb`, `compute_unit`, `compile_url`, `profile_url`

### CLI

```powershell
python bench/benchmark.py [--device DEVICE] [--models MODEL [MODEL ...]]
```

| Flag | Default | Description |
|---|---|---|
| `--device` | `Snapdragon X Elite CRD` | AI Hub hosted device |
| `--models` | all | Subset: `whisper_encoder`, `whisper_decoder`, `it2_encoder`, `it2_decoder`, `nllb_encoder`, `nllb_decoder`, `bart_encoder`, `bart_decoder` |

Output: `bench/results.md`

---

## config.settings

All values are read from environment variables (`.env`). Import the module to access them:

```python
from config.settings import (
    MODELS_DIR,
    WHISPER_ENCODER_ONNX, WHISPER_DECODER_ONNX,
    IT2_ENCODER_ONNX, IT2_DECODER_ONNX,
    NLLB_ENCODER_ONNX, NLLB_DECODER_ONNX,
    SUMMARY_ENCODER_ONNX, SUMMARY_DECODER_ONNX,
    TRANSLATE_MODEL, SUMMARY_MODEL,
    SOURCE_LANGUAGE, TARGET_LANGUAGE,
    QAI_HUB_DEVICE, QAI_HUB_API_TOKEN,
    UI_HOST, UI_PORT,
)
```

`MODELS_DIR.mkdir(exist_ok=True)` is called at import time to ensure the directory exists.
