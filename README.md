# 🎓 Offline Multilingual Lecture Assistant

> **Snapdragon AI Lab — Build & Present Challenge**
> Real-time lecture transcription, translation, and AI summarization — running **100 % on-device** with zero internet required.

---

## 🔒 Core Demo Pitch

| Capability | Technology | Where it runs |
|---|---|---|
| Speech-to-text | Whisper-base-en ONNX | Snapdragon X Elite **NPU** (QNN EP) |
| Translation | IndicTrans2-200M INT8 / NLLB-200-600M INT8 | Snapdragon X Elite **NPU** (QNN EP) |
| LLM Summarization | Qwen3-0.6B GGUF Q4_0 via GenieX | Snapdragon X Elite **NPU** (Hexagon, llama.cpp) |
| UI | Tkinter desktop | Local, no server needed |
| Network calls at runtime | **Zero** | — |

---

## ⚠️ Python Version — Critical

`qai-hub-models` requires **Python 3.11 AMD x64** on Windows.
Python 3.14 (system default) **will not work**.

```powershell
# Download Python 3.11 AMD x64 from:
# https://www.python.org/downloads/release/python-3119/
# Choose "Windows installer (64-bit)" — NOT the ARM64 build.
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
```

---

## 🚀 Quickstart (5 steps)

```powershell
# 1. Clone and create venv
git clone <repo-url>  cd Snapdragon_AI_lab
py -3.11 -m venv .venv && .venv\Scripts\Activate.ps1

# 2. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 3. Configure AI Hub token
cp .env.example .env
# Edit .env → set QAI_HUB_API_TOKEN=your_token_here
python scripts/setup_aihub.py --token YOUR_TOKEN

# 4. Export ONNX models (one-time, ~10 min)
python scripts/export_whisper.py --export-only
python scripts/export_translation.py --model indictrans2 --export-only
python scripts/export_summarizer.py --model bart --export-only

# 5. Launch the UI
python ui/app.py
```

> For full step-by-step instructions see **[docs/HOW_TO_RUN.md](docs/HOW_TO_RUN.md)**

---

## 📁 Project Structure

```
Snapdragon_AI_lab/
├── audio_capture/capture.py        Rolling 30-s mic chunks (sounddevice)
├── transcription/whisper_stt.py    Whisper-base ONNX STT, QNN → CPU fallback
├── translation/translate.py        translate(text, src, tgt) — IndicTrans2 / NLLB
├── summarization/summarize.py      GenieX LLM: 3-section structured summary
├── pipeline.py                     End-to-end wiring, zero network calls
├── ui/app.py                       Tkinter desktop UI — live transcript + summary
├── bench/benchmark.py              AI Hub profiling → bench/results.md
├── scripts/
│   ├── setup_aihub.py              Configure AI Hub token
│   ├── export_whisper.py           Export + compile + profile Whisper
│   ├── export_translation.py       Export + compile + profile IndicTrans2/NLLB
│   └── export_summarizer.py        Export + compile + profile BART/mT5
├── config/settings.py              Central env-driven config
├── docs/                           Full documentation suite
│   ├── COMPLETE_DOCUMENTATION.md
│   ├── HOW_TO_RUN.md
│   ├── ARCHITECTURE.md
│   ├── API_REFERENCE.md
│   ├── TROUBLESHOOTING.md
│   ├── BENCHMARK_GUIDE.md
│   └── DEMO_SCRIPT.md
├── models/                         ONNX weights (git-ignored)
├── .env.example                    Template — copy to .env
├── requirements.txt
└── CONTRIBUTING.md
```

---

## 📊 Benchmark Numbers (Snapdragon X Elite CRD)

| Model | Latency | Memory | Compute |
|---|---|---|---|
| Whisper-base Encoder | **49 ms** | 67 MB | NPU |
| Whisper-base Decoder (1 step) | **3.6 ms** | 126 MB | NPU |
| **Full 30-second chunk** | **~770 ms** | — | NPU |
| **Real-time factor** | **~39×** | — | NPU |
| IndicTrans2-200M Encoder INT8 | ~50 ms | ~120 MB | NPU |
| NLLB-200 Encoder INT8 | ~95 ms | ~200 MB | NPU |
| BART Encoder INT8 | ~110 ms | ~250 MB | NPU |

> Run `python bench/benchmark.py` to regenerate these on your device.

---

## 📚 Documentation Index

| Document | Purpose |
|---|---|
| [HOW_TO_RUN.md](docs/HOW_TO_RUN.md) | Step-by-step setup and launch guide |
| [COMPLETE_DOCUMENTATION.md](docs/COMPLETE_DOCUMENTATION.md) | Full technical reference |
| [ARCHITECTURE.md](docs/ARCHITECTURE.md) | System design and data flow diagrams |
| [API_REFERENCE.md](docs/API_REFERENCE.md) | Every public class and function |
| [TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md) | Common errors and fixes |
| [BENCHMARK_GUIDE.md](docs/BENCHMARK_GUIDE.md) | How to run and interpret benchmarks |
| [DEMO_SCRIPT.md](docs/DEMO_SCRIPT.md) | Live demo script for judges |
| [CONTRIBUTING.md](CONTRIBUTING.md) | Dev setup and contribution guide |

---

## 🔑 What Needs Your Input

| Item | Where | Default |
|---|---|---|
| AI Hub API token | `.env` → `QAI_HUB_API_TOKEN` | *(required)* |
| Target language | `.env` → `TARGET_LANGUAGE` | `hin_Deva` (Hindi) |
| Translation model | `.env` → `TRANSLATE_MODEL` | `indictrans2` |
| LLM model | `.env` → `GENIEX_MODEL` | `Qwen/Qwen3-0.6B-GGUF` |
| Device name | `.env` → `QAI_HUB_DEVICE` | `Snapdragon X Elite CRD` |

---

## License

MIT — see [LICENSE](LICENSE) for details.
All models are subject to their respective upstream licenses
(OpenAI Whisper MIT, IndicTrans2 MIT, NLLB-200 CC-BY-NC, BART MIT, Qwen3 Apache 2.0).
