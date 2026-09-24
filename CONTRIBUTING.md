# Contributing Guide

---

## Development Setup

### 1. Clone and create venv

```powershell
git clone <repo-url>
cd Snapdragon_AI_lab
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Install dev extras

```powershell
pip install pytest==8.2.2 black==24.4.2 ruff==0.4.9 mypy==1.10.0
```

### 3. Configure .env

```powershell
Copy-Item .env.example .env
# Add QAI_HUB_API_TOKEN and set GENIEX_DEVICE_MAP=cpu for off-device dev
```

---

## Project Conventions

### Python Style

- **Formatter:** `black` with default settings (line length 88)
- **Linter:** `ruff` — run `ruff check .` before committing
- **Type hints:** Required on all public function signatures
- **Docstrings:** Module-level docstring required. Class/function docstrings only for non-obvious logic.
- **Imports:** stdlib → third-party → local, separated by blank lines

```powershell
# Format
black .

# Lint
ruff check .

# Type check
mypy pipeline.py transcription/ translation/ summarization/ audio_capture/
```

### Commit Messages

```
feat: add streaming summary token callback
fix: QNN EP fallback when QnnHtp.dll not found
docs: update benchmark numbers in README
refactor: extract _iso1_to_tag helper into pipeline
test: add WhisperSTT no-speech detection test
```

Format: `<type>: <short description>` (imperative, lowercase, ≤ 72 chars)

Types: `feat`, `fix`, `docs`, `refactor`, `test`, `chore`, `perf`

---

## Architecture Rules

These must not be broken by any contribution:

1. **Zero network calls at runtime.** No `requests`, `urllib`, `socket`, or any outbound call in `pipeline.py`, `audio_capture/`, `transcription/`, `translation/`, `summarization/`, or `ui/`.

2. **Lazy model loading.** Models are loaded once on first use (module-level singletons). Never load a model at import time.

3. **Thread safety.** All Tkinter widget mutations must go through `self.after(0, fn)`. All shared state in `Pipeline` must be protected by `self._seg_lock`.

4. **No global mutable state outside module-level singletons.** Each module may have one `_model` singleton and one `_lock`.

5. **Env-driven config.** All paths, device names, and model choices go through `config/settings.py`. Never hardcode a path in a module other than `settings.py`.

---

## Adding a New Translation Language

1. Verify the language tag exists in IndicTrans2 or NLLB-200 vocabulary.
2. Add it to the `LANGUAGES` table in `ui/app.py`.
3. If using IndicTrans2, confirm the tag is in `_IndicTrans2Backend.SUPPORTED_TAGS` in `translation/translator.py`.
4. Test: `python -c "from translation.translate import translate; print(translate('Hello', 'eng_Latn', 'YOUR_TAG'))"`

---

## Adding a New Model

To add a new speech-to-text or translation model:

1. Create the export script in `scripts/export_<model>.py` following the pattern of `export_whisper.py`.
2. Add the model's ONNX path to `config/settings.py`.
3. Add the model to the benchmark spec in `bench/benchmark.py`.
4. Update `docs/COMPLETE_DOCUMENTATION.md` → Model Inventory table.
5. Update `requirements.txt` if new deps are needed.

---

## Testing

```powershell
# Run all tests
pytest

# Run a specific module
pytest tests/test_whisper_stt.py -v

# Quick pipeline import smoke test
python -c "from pipeline import Pipeline; print('ok')"
```

Test files go in `tests/`. Name them `test_<module>.py`.

Tests must not require AI Hub credentials or internet access.
Use `models/test_fixtures/` for small test audio files (< 100 KB .wav).

---

## Pull Request Checklist

Before opening a PR:

- [ ] `black .` passes
- [ ] `ruff check .` passes (no errors)
- [ ] `python -c "from pipeline import Pipeline; print('ok')"` passes
- [ ] No hardcoded paths or credentials
- [ ] No `requests`/`urllib`/`socket` calls in runtime modules
- [ ] Relevant docs updated (API_REFERENCE.md if public API changed)
- [ ] Commit messages follow the format above

---

## File Ownership

| Area | Primary owner |
|---|---|
| Audio capture | `audio_capture/` |
| Whisper STT | `transcription/` |
| Translation | `translation/` |
| LLM summarization | `summarization/` |
| End-to-end pipeline | `pipeline.py` |
| Desktop UI | `ui/` |
| AI Hub scripts | `scripts/` |
| Benchmarking | `bench/` |
| Config | `config/` |
| Documentation | `docs/` |

---

## Licensing

All contributions must be compatible with the MIT License.
Model weights are subject to their upstream licenses:

| Model | License |
|---|---|
| OpenAI Whisper | MIT |
| IndicTrans2 | MIT |
| NLLB-200 | CC-BY-NC 4.0 |
| BART-large-cnn | MIT |
| mT5-XLSum | Apache 2.0 |
| Qwen3-0.6B | Apache 2.0 |

**Note:** NLLB-200 is CC-BY-NC (non-commercial). For commercial use, substitute
IndicTrans2 (Indic) or an MIT/Apache-licensed translation model.
