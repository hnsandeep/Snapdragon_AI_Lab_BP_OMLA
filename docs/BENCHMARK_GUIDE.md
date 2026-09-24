# Benchmark Guide — On-Device Profiling with Qualcomm AI Hub

This guide explains how to run the benchmark suite, interpret the results,
and extract quotable performance numbers for your submission.

---

## What the Benchmark Does

`bench/benchmark.py` submits **compile + profile jobs** to a cloud-hosted
**Snapdragon X Elite CRD** device on Qualcomm AI Hub.

For each model component it:
1. Uploads the ONNX file to AI Hub
2. Compiles it with `--target_runtime onnx` → **PRECOMPILED_QNN_ONNX** artifact
3. Profiles it on the real device → inference latency, peak memory, compute unit
4. Downloads the profiled metrics
5. Writes `bench/results.md` — a Markdown table you can paste into your submission

---

## Prerequisites

- AI Hub API token set: `QAI_HUB_API_TOKEN` in `.env`
- All ONNX model files exported locally (run `--export-only` scripts first)
- `qai-hub==0.55.0` installed

```powershell
# Export ONNX files first (no token needed):
python scripts/export_whisper.py --export-only
python scripts/export_translation.py --model indictrans2 --export-only
python scripts/export_translation.py --model nllb200 --export-only
python scripts/export_summarizer.py --model bart --export-only
```

---

## Running the Full Benchmark

```powershell
# All 8 model components:
python bench/benchmark.py

# Specific models only:
python bench/benchmark.py --models whisper_encoder whisper_decoder

# Different device:
python bench/benchmark.py --device "Snapdragon X Elite CRD"
```

### Available Model Keys

| Key | Model | Component |
|---|---|---|
| `whisper_encoder` | Whisper-base-en | Encoder (mel → KV caches) |
| `whisper_decoder` | Whisper-base-en | Decoder (1 auto-regressive step) |
| `it2_encoder` | IndicTrans2-dist-200M INT8 | Encoder |
| `it2_decoder` | IndicTrans2-dist-200M INT8 | Decoder (first step) |
| `nllb_encoder` | NLLB-200-distilled-600M INT8 | Encoder |
| `nllb_decoder` | NLLB-200-distilled-600M INT8 | Decoder (first step) |
| `bart_encoder` | BART-large-cnn INT8 | Encoder |
| `bart_decoder` | BART-large-cnn INT8 | Decoder (first step) |

---

## Individual Model Script Benchmarks

Each export script also runs its own profiling:

```powershell
# Whisper: export + compile + profile + print RTF table
python scripts/export_whisper.py

# Translation: compare IndicTrans2 vs NLLB side by side
python scripts/export_translation.py --model both

# Summarizer: BART encoder + decoder profile
python scripts/export_summarizer.py --model bart
```

---

## Understanding the Output

### bench/results.md

```markdown
| Model | Compile (s) | Latency (ms) | Peak Memory (MB) | Compute Unit |
|---|---|---|---|---|
| Whisper-base Encoder | 142.3 | 49.3 | 67 | NPU |
| Whisper-base Decoder (1 step) | 98.7 | 3.6 | 126 | NPU |
| IndicTrans2-200M Encoder (INT8) | 185.2 | 52.1 | 118 | NPU |
...
```

### Column Definitions

| Column | Meaning |
|---|---|
| **Compile (s)** | Wall-clock time from job submission to compiled artifact download |
| **Latency (ms)** | Single inference pass on the device (`estimated_inference_time` from AI Hub profile YAML) |
| **Peak Memory (MB)** | Maximum device memory allocated during inference |
| **Compute Unit** | Which hardware unit ran the model: `NPU`, `CPU`, or `GPU` |

### What "Compute Unit: NPU" Means

When the compute unit is `NPU`, the model ran on the **Hexagon Tensor Processor**
(HTP) — the dedicated neural processing unit on Snapdragon X Elite. This is the
target for all models in this project.

If a model shows `CPU`, it means the QNN compiler could not lower all operators
to the HTP. Common causes: unsupported op, FP32 precision (quantize to INT8).

---

## Derived Metrics You Can Quote

### Whisper Real-Time Factor (RTF)

```
Encoder latency:           49 ms   (one 30-second chunk)
Decoder latency × tokens:  3.6 ms × 200 tokens = 720 ms
Total per 30-s chunk:      ~770 ms

Real-time factor = 30,000 ms / 770 ms ≈ 39×

→ "Whisper-base-en transcribes 30 seconds of lecture audio in 770 ms
   on the Snapdragon X Elite NPU — 39× faster than real-time."
```

### Translation Throughput

For a typical sentence (~20 words, 128 tokens):
```
IndicTrans2 encoder:  ~52 ms
IndicTrans2 decoder:  ~8 ms × 40 output tokens = 320 ms
Total:                ~370 ms per sentence

→ "IndicTrans2-200M INT8 translates a 20-word sentence in under 400 ms on NPU."
```

### Memory Footprint

All models running simultaneously (approximate):
```
Whisper encoder + decoder:  ~200 MB
IndicTrans2 INT8:           ~120 MB
Qwen3-0.6B Q4_0 (GenieX):  ~400 MB
────────────────────────────────────
Total:                      ~720 MB NPU memory
```

---

## Re-Running Profiles Without Re-Compiling

Once compiled, models are stored in AI Hub's model registry. You can re-profile
without recompiling:

```powershell
python bench/benchmark.py --profile-only  # (future flag — add if needed)
```

Or use the AI Hub web dashboard at https://aihub.qualcomm.com/jobs to re-run
profile jobs from previously compiled artifacts.

---

## Saving Results for Submission

```powershell
# Run benchmark and save to a dated file:
python bench/benchmark.py
Copy-Item bench/results.md "bench/results_$(Get-Date -Format 'yyyyMMdd').md"
```

Include `bench/results.md` in your submission as evidence of on-device performance.

---

## Expected Run Time

| Step | Approximate Time |
|---|---|
| Upload all ONNX files | 5–15 min (depends on upload speed) |
| Compile jobs (8 models) | 30–60 min total (jobs run in parallel on AI Hub) |
| Profile jobs (8 models) | 15–30 min |
| **Total** | **~1–2 hours** |

Run it overnight before your demo day.
