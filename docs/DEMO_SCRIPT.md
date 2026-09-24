# Live Demo Script — Snapdragon AI Lab Build & Present Challenge

This is your judge-facing talking script. Each section maps to a UI element or
live action. Times are approximate for a 5-minute demo slot.

---

## Opening (30 seconds)

> "What you're about to see is a complete AI assistant for lectures and meetings
> that runs **entirely on this laptop** — no cloud, no API keys, no internet.
> Every byte of AI inference happens right here, on the Snapdragon X Elite NPU."

**Action:** Point to the green badge at the top of the UI:
> "This badge — Offline Mode ON — is not a marketing claim. It's enforced in code.
> I can pull the ethernet cable and turn off Wi-Fi right now and nothing changes."

---

## The Stack (60 seconds)

> "The pipeline has four stages:"

Walk through while pointing at the UI:

1. **Microphone** → "Sound device captures 30-second rolling chunks at 16 kHz."

2. **Whisper** → "Whisper-base runs on the Hexagon NPU via ONNX Runtime's QNN Execution Provider.
   The encoder processes a full 30-second chunk in **49 milliseconds**.
   The decoder generates about 200 tokens in **720 milliseconds**.
   That's the entire 30-second chunk transcribed in **under 800 milliseconds** —
   **39 times faster than real-time** on the NPU."

3. **IndicTrans2** → "The transcript is immediately translated using IndicTrans2,
   a model trained specifically on all 22 scheduled Indian languages.
   It runs INT8 quantized — 472 MB instead of 1.87 GB — still on the NPU.
   Pick any target language from the dropdown."

4. **GenieX LLM** → "When you click Generate Summary, a Qwen3 0.6B model
   runs through Qualcomm's GenieX runtime — llama.cpp backend,
   Q4_0 quantized GGUF, Hexagon NPU.
   It follows a strict system prompt: key points, action items, 120-word summary.
   Watch the tokens stream in real time — all on-chip."

---

## Live Demo (2 minutes)

### Start Recording
1. Set Source: **English**, Target: **Hindi** (or your chosen language)
2. Click **▶ Start Recording**
3. Speak clearly for 30–60 seconds:

> *Suggested demo speech (read naturally):*
> "Welcome to today's session on machine learning deployment.
> We will cover three key topics: model quantization, on-device inference,
> and the Qualcomm AI Hub pipeline.
> Action items for this week: export the Whisper model to ONNX by Thursday,
> run the benchmark suite on the Snapdragon device, and prepare the demo slides.
> The main takeaway is that modern language models can run fully on-device
> at real-time speeds without any cloud dependency."

4. Watch the transcript appear in the left panel in ~30 seconds.
5. Point to the Hindi (or target language) translation appearing below each segment.

### Generate Summary
1. Click **⏹ Stop**
2. Click **✦ Generate Summary**
3. Watch tokens stream into the right panel.
4. Show the three sections:

> "Notice the three sections the LLM produces — exactly as instructed:
> Key Points, Action Items, and a plain-language summary under 120 words.
> The LLM was given a strict system prompt that says 'do not invent facts.'
> That's important for a lecture assistant — accuracy over creativity."

---

## Benchmark Numbers (30 seconds)

Open `bench/results.md` or point to the About tab in the UI:

> "These numbers are from actual AI Hub profile jobs on a cloud-hosted Snapdragon X Elite CRD.
> Not estimated. Not simulated."

| What to say | Number |
|---|---|
| Whisper encoder latency | "49 milliseconds per 30-second chunk" |
| Full transcription RTF | "39 times faster than real-time" |
| Translation (IndicTrans2 encoder) | "~52 milliseconds per sentence" |
| LLM model size on disk | "400 megabytes, Q4_0 GGUF" |
| Network calls at runtime | "Zero. None. Verified in code." |

---

## Language Toggle Demo (30 seconds)

1. Stop recording if still running.
2. Change target language to **Tamil** (or Kannada, Telugu — pick an Indian language).
3. Start recording again, speak the same ~30 seconds.
4. Show the Tamil translation appearing.

> "Same pipeline, different language — just a dropdown change.
> IndicTrans2 supports all 22 scheduled Indian languages.
> For non-Indic languages like German or French, we fall back to NLLB-200
> which covers 200 language pairs."

---

## Closing Pitch (30 seconds)

> "Every lecture or meeting you attend leaves students and colleagues behind —
> people who missed it, couldn't attend, or are learning in a different language.
> This assistant gives them a complete record: what was said, in their language,
> with the key points and action items highlighted.
>
> And because it runs entirely on-device, it works in hospitals, government buildings,
> corporate boardrooms, and classrooms where cloud AI is prohibited.
>
> Snapdragon X Elite makes this possible — 39× real-time transcription,
> instant translation, and a local LLM — in a laptop, with no internet."

---

## Q&A Prep

| Question | Answer |
|---|---|
| "Is it really offline?" | "Yes — pull the network cable right now. The app keeps running." |
| "What's the accuracy of Whisper-base?" | "~95% WER on clean English speech. Switch to Whisper-small for accented speech." |
| "Why Q4_0 for the LLM?" | "It's the only GGUF quantization with first-class Hexagon NPU support in llama.cpp. Other quants fall back to CPU." |
| "Why IndicTrans2 and not Google Translate?" | "IndicTrans2 is trained specifically on all 22 Indian languages. It outperforms generic models by 5–12 BLEU points on Indic pairs. And it runs offline." |
| "What's the total latency per 30-second chunk?" | "~770ms for transcription, ~400ms for translation = under 1.2 seconds per chunk from audio to translated text." |
| "Can it summarize in Hindi?" | "Yes — switch the summarizer to mT5-XLSum which covers 45 languages including Hindi, Tamil, and Bengali." |
| "How much disk space does it need?" | "~3.2 GB for all models combined. About the size of two movies." |
| "Could this run on a phone?" | "Yes — the same QNN stack runs on Snapdragon 8 Elite mobile. The ONNX models are identical; only the device target changes in AI Hub." |

---

## Demo Day Checklist

- [ ] Python 3.11 venv activated: `.venv\Scripts\Activate.ps1`
- [ ] All ONNX files exported: `models/WhisperEncoder.onnx` etc. exist
- [ ] GenieX model pre-downloaded (first-run cache warm)
- [ ] `.env` configured with correct paths and language
- [ ] `onnxruntime-qnn==2.6.0` installed (on Snapdragon device)
- [ ] Microphone tested: `python -c "from audio_capture.capture import AudioCapture; AudioCapture.list_devices()"`
- [ ] Wi-Fi turned off (to prove offline claim)
- [ ] `bench/results.md` ready to show benchmark numbers
- [ ] UI launched and language dropdowns set: `python ui/app.py`
