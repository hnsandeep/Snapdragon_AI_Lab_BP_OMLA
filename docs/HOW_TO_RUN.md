# How to Run — Offline Multilingual Lecture Assistant

> Complete step-by-step guide from a fresh Windows machine to a running live demo.

---

## Prerequisites Checklist

Before you begin, confirm these are available:

- [ ] Windows 11 on Snapdragon X Elite (or any x64 Windows for off-device dev)
- [ ] Python 3.11 AMD x64 installed (`py -3.11 --version` → `3.11.x`)
- [ ] Git installed
- [ ] Qualcomm AI Hub account + API token (https://aihub.qualcomm.com)
- [ ] At least **20 GB free disk** (model weights: ~8 GB total)
- [ ] Microphone connected (for live capture)
- [ ] Internet access for first-time model downloads only

---

## Step 0 — Install Python 3.11 AMD x64

> **Do not use the Python 3.14 that may already be on your system.**
> `qai-hub-models` requires Python ≥ 3.10 and < 3.14, AMD x64 only.

1. Go to https://www.python.org/downloads/release/python-3119/
2. Download **"Windows installer (64-bit)"** — the one labelled `amd64.exe`
3. Install with "Add to PATH" checked
4. Verify: open a new PowerShell and run:
   ```powershell
   py -3.11 --version
   # Expected: Python 3.11.x
   ```

---

## Step 1 — Get the Code

```powershell
git clone <your-repo-url> Snapdragon_AI_lab
cd Snapdragon_AI_lab
```

Or if you already have the folder:
```powershell
cd d:\Snapdragon_AI_lab
```

---

## Step 2 — Create Virtual Environment

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1

# Confirm you are in the venv
python --version   # Should show 3.11.x
```

If PowerShell blocks execution:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

---

## Step 3 — Install Dependencies

```powershell
pip install --upgrade pip
pip install -r requirements.txt
```

This installs approximately 4 GB of packages. Allow 10–20 minutes on first run.

### On-Device vs Off-Device Runtime

| Scenario | Action |
|---|---|
| **Off-device development** (x64 PC, CI) | Nothing extra — `onnxruntime` is already in requirements.txt |
| **On Snapdragon X Elite** (NPU inference) | Swap the ORT package: `pip uninstall onnxruntime -y && pip install onnxruntime-qnn==2.6.0` |

---

## Step 4 — Configure Environment

```powershell
Copy-Item .env.example .env
```

Open `.env` in any text editor and fill in:

```ini
# Required
QAI_HUB_API_TOKEN=your_token_here     # from https://aihub.qualcomm.com

# Defaults you may want to change
TARGET_LANGUAGE=hin_Deva              # Hindi; see language tag table below
TRANSLATE_MODEL=indictrans2           # or: nllb200
GENIEX_DEVICE_MAP=npu                 # or: cpu (off-device dev)
```

### Language Tag Reference

| Language | Tag |
|---|---|
| Hindi | `hin_Deva` |
| Kannada | `kan_Knda` |
| Tamil | `tam_Taml` |
| Telugu | `tel_Telu` |
| Bengali | `ben_Beng` |
| Marathi | `mar_Deva` |
| German | `deu_Latn` |
| French | `fra_Latn` |
| Spanish | `spa_Latn` |

---

## Step 5 — Configure AI Hub

```powershell
python scripts/setup_aihub.py --token YOUR_TOKEN_HERE
```

This writes your token to `~/.qai_hub/client.ini` and lists available devices.
Expected output:
```
[1/3] Configuring AI Hub API token …
  ✓  Token written to ~/.qai_hub/client.ini
[2/3] Verifying connectivity …
  ✓  Connected. 47 device(s) available.
[3/3] Searching for Snapdragon X Elite / X2 devices …
  ✓  Found device: 'Snapdragon X Elite CRD'
```

---

## Step 6 — Export ONNX Models (One-Time)

Run each export script. These download model weights from HuggingFace,
export to ONNX, and (optionally) compile on AI Hub. 
Pass `--export-only` to skip AI Hub if you just want local ONNX files.

### 6a. Whisper Speech-to-Text

```powershell
# Local ONNX only (no AI Hub token needed):
python scripts/export_whisper.py --export-only

# Full pipeline — export + compile on AI Hub + benchmark:  🔑 TOKEN
python scripts/export_whisper.py
```

Output files:
```
models/WhisperEncoder.onnx          (~91 MB)
models/WhisperDecoder.onnx          (~187 MB)
models/WhisperEncoder_compiled.onnx (after AI Hub compile)
models/WhisperDecoder_compiled.onnx
models/mel_filters.npz              (mel filter bank, extracted once)
```

After compile, add to `.env`:
```ini
WHISPER_ENCODER_ONNX=models/WhisperEncoder_compiled.onnx
WHISPER_DECODER_ONNX=models/WhisperDecoder_compiled.onnx
```

### 6b. Translation Model

```powershell
# IndicTrans2 (best for Indic languages):
python scripts/export_translation.py --model indictrans2 --export-only

# NLLB-200 (200-language coverage):
python scripts/export_translation.py --model nllb200 --export-only

# Compare both on AI Hub:  🔑 TOKEN
python scripts/export_translation.py --model both
```

Output:
```
models/indictrans2/         fp32 ONNX + int8/ subfolder (~472 MB INT8)
models/nllb200/             fp32 ONNX + int8/ subfolder (~630 MB INT8)
```

### 6c. Summarization Model

```powershell
# BART (English lectures):
python scripts/export_summarizer.py --model bart --export-only

# mT5 (multilingual lectures, 45 languages):
python scripts/export_summarizer.py --model mt5 --export-only
```

### 6d. Extract Mel Filters (Whisper helper)

```powershell
python -c "from transcription.transcriber import save_mel_filters_from_whisper; save_mel_filters_from_whisper()"
```

This saves `models/mel_filters.npz` so the ONNX inference path doesn't need PyTorch at runtime.

---

## Step 7 — GenieX First-Run Model Pull

GenieX downloads the Qwen3-0.6B GGUF model on first use (~400 MB). This happens automatically when `summarize()` is called for the first time. To pre-pull manually:

```powershell
python -c "
from geniex import AutoModelForCausalLM
m = AutoModelForCausalLM.from_pretrained('Qwen/Qwen3-0.6B-GGUF', device_map='cpu', precision='Q4_0')
m.close()
print('GenieX model cached.')
"
```

After this, GenieX runs fully offline.

---

## Step 8 — Launch the UI

```powershell
python ui/app.py
```

The Tkinter window opens immediately. You will see:
- Green **"🔒 OFFLINE MODE: ON | No network calls | All inference on-device"** badge at the top
- Language dropdowns (Source / Target)
- **▶ Start Recording** button
- Live transcript scrolling area
- **✦ Generate Summary** button → streams GenieX output

---

## Step 9 — Run a Quick Pipeline Test (Optional)

```powershell
python -c "
import sys, numpy as np
sys.path.insert(0, '.')
from transcription.whisper_stt import WhisperSTT

stt = WhisperSTT()
# feed 2 seconds of silence — should return empty string (no-speech detected)
audio = np.zeros(32000, dtype=np.float32)
print(repr(stt.transcribe(audio)))
print('Pipeline import test passed.')
"
```

---

## Step 10 — Run Benchmarks (Optional, needs AI Hub)

```powershell
python bench/benchmark.py
# Results written to bench/results.md
```

See [BENCHMARK_GUIDE.md](BENCHMARK_GUIDE.md) for full details.

---

## Common Run Modes

| Mode | Command | Notes |
|---|---|---|
| Live demo (full pipeline) | `python ui/app.py` | Main demo entry point |
| File transcription only | See API_REFERENCE.md WhisperSTT | No mic needed |
| Headless pipeline | `python pipeline.py` (import) | Programmatic use |
| Benchmark only | `python bench/benchmark.py` | Needs AI Hub token |
| Export models | `python scripts/export_whisper.py` | One-time setup |

---

## Updating the `.env` After Compile

After running AI Hub compile jobs, update `.env` with the compiled model paths:

```ini
WHISPER_ENCODER_ONNX=models/WhisperEncoder_compiled.onnx
WHISPER_DECODER_ONNX=models/WhisperDecoder_compiled.onnx
TRANSLATE_MODEL=indictrans2
IT2_ENCODER_ONNX=models/indictrans2/int8/encoder_model_compiled.onnx
IT2_DECODER_ONNX=models/indictrans2/int8/decoder_model_compiled.onnx
SUMMARY_MODEL=bart
SUMMARY_ENCODER_ONNX=models/bart/int8/encoder_model_compiled.onnx
SUMMARY_DECODER_ONNX=models/bart/int8/decoder_model_compiled.onnx
```

---

## Shutting Down

- Click the ✕ window button — the pipeline calls `close()` which releases NPU resources and frees GenieX model handles.
- Or `Ctrl+C` in the terminal.

---

> **Next:** Read [ARCHITECTURE.md](ARCHITECTURE.md) to understand how the pipeline works,
> or [TROUBLESHOOTING.md](TROUBLESHOOTING.md) if something goes wrong.
