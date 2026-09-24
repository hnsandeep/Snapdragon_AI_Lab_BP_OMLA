# /models — ONNX Model Weights

This directory stores compiled `.onnx` (and `.bin`) model artifacts.
These are large binary files and are excluded from git (see `.gitignore`).

## Expected files

| File | Source model | How to generate |
|---|---|---|
| `whisper_encoder.onnx` | openai/whisper-base | See export guide below |
| `opus_mt_en_de.onnx` | Helsinki-NLP/opus-mt-en-de | `optimum-cli export onnx` |
| `bart_summary.onnx` | facebook/bart-large-cnn | `optimum-cli export onnx` |

## Export Whisper to ONNX

```bash
pip install openai-whisper
python - <<'EOF'
import whisper, torch, io
model = whisper.load_model("base").eval()
# Export encoder only (the NPU-friendly part)
dummy = torch.randn(1, 80, 3000)
torch.onnx.export(model.encoder, dummy, "models/whisper_encoder.onnx",
                  opset_version=17, input_names=["mel"],
                  output_names=["features"])
print("Done.")
EOF
```

## Export OPUS-MT to ONNX

```bash
pip install optimum[exporters]
optimum-cli export onnx \
  --model Helsinki-NLP/opus-mt-en-de \
  models/opus_mt_en_de/
```

## Compile for Snapdragon via AI Hub

After exporting, use `scripts/verify_hub_job.py` as a template to submit
a compile job to `Snapdragon X Elite CRD` targeting the `onnx` runtime.
The compiled `.onnx` returned by AI Hub is what you deploy on-device.
