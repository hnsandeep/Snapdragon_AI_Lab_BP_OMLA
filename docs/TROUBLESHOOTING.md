# Troubleshooting Guide

Common errors, their root causes, and exact fixes.

---

## Python & Environment

### `ModuleNotFoundError: No module named 'qai_hub_models'`

**Cause:** You are running Python 3.14 or the wrong Python.

**Fix:**
```powershell
python --version    # Must be 3.11.x
# If not 3.11, activate the correct venv:
.venv\Scripts\Activate.ps1
python --version
```

If you haven't created a 3.11 venv yet:
```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

---

### `ERROR: Could not find a version that satisfies the requirement qai-hub-models==0.62.0`

**Cause:** Python 3.14 or ARM64 Python. `qai-hub-models` requires `<3.14` and AMD x64.

**Fix:** Install Python 3.11 AMD x64 from https://www.python.org/downloads/release/python-3119/
Choose `Windows installer (64-bit)` — not the ARM64 variant.

---

### `ImportError: DLL load failed while importing onnxruntime`

**Cause:** Both `onnxruntime` and `onnxruntime-qnn` are installed. They conflict.

**Fix:**
```powershell
pip uninstall onnxruntime onnxruntime-qnn -y
# Then install exactly one:
pip install onnxruntime==1.18.1          # off-device dev
# OR
pip install onnxruntime-qnn==2.6.0      # on Snapdragon X Elite
```

---

### `Set-ExecutionPolicy` error when activating venv

**Fix:**
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
.venv\Scripts\Activate.ps1
```

---

## Model Files

### `FileNotFoundError: Encoder not found: models/WhisperEncoder.onnx`

**Cause:** ONNX models have not been exported yet.

**Fix:**
```powershell
python scripts/export_whisper.py --export-only
```

---

### `FileNotFoundError: models/mel_filters.npz`

**Cause:** Mel filter bank not extracted yet.

**Fix:**
```powershell
python -c "from transcription.whisper_stt import save_mel_filters_from_whisper; save_mel_filters_from_whisper()"
```

---

### `ORTModelForSeq2SeqLM` loads slowly (5–10 minutes on first call)

**Cause:** HuggingFace is downloading IndicTrans2 or NLLB-200 weights (~1.87 GB) on first use.

**Fix:** This is expected on first run. Subsequent loads are instant (cached in `~/.cache/huggingface`).
To pre-download:
```powershell
python scripts/export_translation.py --model indictrans2 --export-only
```

---

### `optimum-cli export onnx` fails with OOM (Out of Memory)

**Cause:** FP32 BART encoder (~1 GB) + onnxruntime in memory simultaneously exceeds available RAM.

**Fix:** Run on a machine with ≥ 16 GB RAM. Or use the pre-exported INT8 variant from HuggingFace for IndicTrans2 (already handled automatically by the export script).

---

## AI Hub

### `ERROR: QAI_HUB_API_TOKEN not set`

**Fix:**
```powershell
# Option 1 — via script:
python scripts/setup_aihub.py --token YOUR_TOKEN_HERE

# Option 2 — via .env:
# Add this line to .env:  QAI_HUB_API_TOKEN=your_token_here
```

Get your token at https://aihub.qualcomm.com → Account → API Token.

---

### `Device 'Snapdragon X Elite CRD' not found`

**Cause:** The device name is wrong or your account doesn't have access to that device.

**Fix:**
```powershell
# List all available devices:
python -c "import qai_hub as hub; [print(d.name) for d in hub.get_devices()]"
```

Update `QAI_HUB_DEVICE` in `.env` to match an available device name exactly.

---

### Compile job `FAILED` on AI Hub

**Cause:** Usually input_specs mismatch or unsupported op.

**Fix:**
1. Check the compile job URL printed during the run — it links to the AI Hub job log.
2. Verify the ONNX file is valid: `python -c "import onnx; onnx.checker.check_model('models/WhisperEncoder.onnx')"`
3. Try with FP32 first (`--fp32` flag) to rule out quantization issues.

---

## GenieX / LLM

### `ImportError: No module named 'geniex'`

**Fix:**
```powershell
pip install geniex-llama-cpp
```

---

### GenieX first inference is slow (~30 s)

**Cause:** First run compiles GGUF kernels for the Hexagon NPU. This is a one-time cost.

**Fix:** Expected behaviour. Subsequent calls are fast (~300–500 ms TTFT). The compilation result is cached by the llama.cpp runtime.

---

### GenieX output is truncated or cuts off mid-sentence

**Cause:** `GENIEX_MAX_TOKENS` is too low for a long transcript.

**Fix:**
```ini
# In .env:
GENIEX_MAX_TOKENS=768
```

---

### `device_map="npu"` falls back to CPU silently

**Cause:** Running on a non-Snapdragon machine.

**Expected:** This is the correct fallback behaviour. Set `GENIEX_DEVICE_MAP=cpu` explicitly for off-device development to suppress warnings.

---

## Translation

### `IndicTransToolkit import error`

**Fix:**
```powershell
pip install IndicTransToolkit==0.1.1
```

If that fails with a network error on some regions:
```powershell
pip install IndicTransToolkit==0.1.1 --index-url https://pypi.org/simple
```

---

### Translation output is in the wrong language or garbled

**Cause:** `IndicProcessor.preprocess_batch()` was skipped, or wrong language tags.

**Fix:** Always pass ISO 639-3+script tags: `eng_Latn`, `hin_Deva`, `kan_Knda`, etc.
The `translate()` function handles this automatically when called through the Pipeline.

---

### `KeyError: 'hin_Deva'` in `NllbTokenizer.lang_code_to_id`

**Cause:** NLLB uses slightly different tag format for some languages.

**Fix:** Use `hin_Deva` for NLLB-200 as well. If a specific language fails, check the NLLB tokenizer's language list:
```python
from transformers import NllbTokenizer
t = NllbTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")
print([k for k in t.lang_code_to_id if "hin" in k])
```

---

## Audio Capture

### `sounddevice.PortAudioError: [Errno -9996] Invalid input device`

**Cause:** No microphone connected, or wrong device index.

**Fix:**
```python
from audio_capture.capture import AudioCapture
AudioCapture.list_devices()   # shows device index numbers
# Then in .env or PipelineConfig: mic_device=2  (use your index)
```

---

### Transcript is empty even though you spoke clearly

**Cause 1:** No-speech detection threshold triggered. Check that microphone is working.
**Cause 2:** ONNX files are the uncompiled versions (not AI Hub compiled). CPU inference is slower.

**Fix:**
```powershell
# Test microphone level:
python -c "
import sounddevice as sd, numpy as np
rec = sd.rec(16000, samplerate=16000, channels=1, dtype='float32')
sd.wait()
print(f'Max amplitude: {np.abs(rec).max():.4f}  (should be > 0.01 if speaking)')
"
```

---

## UI

### Tkinter window doesn't open or crashes immediately

**Cause:** Display server not available (e.g., running in a headless environment).

**Fix:** Run on a machine with a display. Tkinter requires a display. For headless testing, use the pipeline directly without the UI:
```python
from pipeline import Pipeline, PipelineConfig
p = Pipeline(PipelineConfig())
p.start()
```

---

### `AttributeError: 'App' object has no attribute '_poll_id'`

**Cause:** Tkinter's `after_cancel` called before `_poll_id` was set.

**Fix:** This is a race condition fixed in `_stop_recording()` — ensure you are running the latest version of `ui/app.py`.

---

### Summary pane shows `[Error: ...]`

**Cause:** GenieX model not loaded, or transcript is empty.

**Fix:** 
1. Confirm GenieX is installed: `python -c "from geniex import AutoModelForCausalLM; print('ok')"`
2. Ensure you recorded something before clicking Generate Summary.

---

## Common Quick Fixes

```powershell
# Full clean reinstall
.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt --force-reinstall

# Re-export all ONNX models
python scripts/export_whisper.py --export-only
python scripts/export_translation.py --model indictrans2 --export-only
python scripts/export_summarizer.py --model bart --export-only

# Verify pipeline imports
python -c "from pipeline import Pipeline; print('imports ok')"

# Verify GenieX
python -c "from geniex import AutoModelForCausalLM; print('geniex ok')"

# Verify ONNX Runtime + QNN
python -c "import onnxruntime as ort; print(ort.get_available_providers())"
```
