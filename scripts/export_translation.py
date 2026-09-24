"""
scripts/export_translation.py
──────────────────────────────
Exports, quantizes (INT8), compiles, and profiles translation models for
Snapdragon X Elite.  Two model families are supported side-by-side so you
can pick whichever benchmarks faster on your device.

┌─────────────────────────────────────────────────────────────────────────────┐
│  Option A — IndicTrans2-distilled-200M  (AI4Bharat)                        │
│    • Best accuracy on all 22 Indic languages (Hindi, Kannada, Tamil, …)    │
│    • Uses pre-exported ONNX from TigreGotico/indictrans2-en-indic-dist-200M-onnx │
│    • Requires IndicTransToolkit for mandatory pre/post-processing           │
│    • FP32: 1.87 GB   INT8: 472 MB  (int8 exact-match ≈60 %)                │
│                                                                              │
│  Option B — NLLB-200-distilled-600M  (Meta)                                │
│    • 200 languages, simpler deps (standard HuggingFace tokenizer)          │
│    • Export via optimum-cli, quantize via OnnxRuntime                      │
│    • FP32 ~2.5 GB   INT8 ~630 MB                                           │
└─────────────────────────────────────────────────────────────────────────────┘

AI Hub compile options (both models):
  --target_runtime onnx          → PRECOMPILED_QNN_ONNX (use with QNN EP on-device)
  --quantize_full_type w8a8      → INT8 weights + INT8 activations via AI Hub AIMET

The AIMET INT8 path on AI Hub is the cleanest way to get NPU-runnable INT8
for seq2seq encoders on Snapdragon.  The optimum dynamic-INT8 ONNX artifacts
can also be loaded via CPU EP while you wait for AI Hub results.

Usage
─────
  # Download INT8 IndicTrans2 ONNX + compile on AI Hub:  🔑 TOKEN
  python scripts/export_translation.py --model indictrans2

  # Export NLLB-200, quantize locally, compile on AI Hub:  🔑 TOKEN
  python scripts/export_translation.py --model nllb200

  # Export both locally (no token):
  python scripts/export_translation.py --model both --export-only

  # Profile pre-compiled models (models/ already populated):  🔑 TOKEN
  python scripts/export_translation.py --model indictrans2 --profile-only

Outputs
───────
  models/indictrans2/                encoder_model.onnx, decoder_model.onnx,
                                     decoder_with_past_model.onnx, int8/,
                                     *_compiled.onnx (after AI Hub compile)
  models/nllb200/                    encoder_model.onnx, decoder_model.onnx,
                                     decoder_with_past_model.onnx,
                                     int8/ (local quantization),
                                     *_compiled.onnx (after AI Hub compile)
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

MODELS_DIR   = ROOT / "models"
IT2_DIR      = MODELS_DIR / "indictrans2"
NLLB_DIR     = MODELS_DIR / "nllb200"
DEVICE_DEFAULT = os.getenv("QAI_HUB_DEVICE", "Snapdragon X Elite CRD")

# HuggingFace repos
IT2_ONNX_REPO  = "TigreGotico/indictrans2-en-indic-dist-200M-onnx"
NLLB_HF_REPO   = "facebook/nllb-200-distilled-600M"

# Encoder input specs for AI Hub compile (seq2seq encoders take token IDs)
# max_length=256 matches the sinusoidal position table frozen into the ONNX
ENCODER_SEQ_LEN = 128       # typical sentence; compile at 128, dynamic at inference
IT2_ENCODER_SPECS   = {"input_ids": (1, ENCODER_SEQ_LEN),
                       "attention_mask": (1, ENCODER_SEQ_LEN)}
NLLB_ENCODER_SPECS  = {"input_ids": (1, ENCODER_SEQ_LEN),
                       "attention_mask": (1, ENCODER_SEQ_LEN)}

# Decoder (first step, no KV cache) — single token generation
DECODER_SPECS = {
    "input_ids":           (1, 1),
    "encoder_hidden_states": (1, ENCODER_SEQ_LEN, 512),
    "encoder_attention_mask": (1, ENCODER_SEQ_LEN),
}


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _run(cmd: list[str], cwd: str | None = None) -> None:
    """Run a subprocess; exit on failure."""
    print("  $", " ".join(cmd))
    r = subprocess.run(cmd, cwd=cwd)
    if r.returncode != 0:
        print(f"  ERROR: command exited {r.returncode}")
        sys.exit(1)


def _check_hub() -> object:
    """Import qai_hub and abort with a friendly message if missing / no token."""
    token = os.getenv("QAI_HUB_API_TOKEN", "")
    if not token:
        print("ERROR: QAI_HUB_API_TOKEN not set.")
        print("  Run: python scripts/setup_aihub.py --token YOUR_TOKEN")
        sys.exit(1)
    try:
        import qai_hub as hub   # noqa: PLC0415
        return hub
    except ImportError:
        print("ERROR: qai-hub not installed.  pip install qai-hub==0.55.0")
        sys.exit(1)


def _wait_compile(job, label: str, out_path: Path) -> object:
    """Block until compile job finishes, download, return compiled model handle."""
    print(f"  Waiting for {label} compile …", end="", flush=True)
    status = job.wait()
    print(f" {status}")
    if "FAIL" in str(status).upper():
        print(f"  ✗  {label} compile FAILED — see {job.url}")
        sys.exit(1)
    compiled = job.get_target_model()
    compiled.download(str(out_path))
    size_mb = out_path.stat().st_size / 1e6
    print(f"  ✓  {label} downloaded → {out_path.name}  ({size_mb:.0f} MB)")
    return compiled


def _profile(hub, model, device: str, job_name: str, label: str) -> dict:
    """Submit profile job, wait, return metrics dict."""
    print(f"\n  Profiling {label} …")
    pjob = hub.submit_profile_job(model=model, name=job_name,
                                  device=hub.Device(device))
    print(f"  Profile job: {pjob.url}")
    pjob.wait()
    try:
        data    = pjob.download_profile()
        summary = data.get("execution_summary", {})
        inf_us  = (summary.get("estimated_inference_time")
                   or summary.get("inference_time_us", None))
        mem_raw = summary.get("peak_memory_bytes", {})
        mem_b   = mem_raw.get("total", 0) if isinstance(mem_raw, dict) else mem_raw
        cu      = summary.get("primary_compute_unit", "unknown")
        return {
            "label":             label,
            "inference_time_ms": round(inf_us / 1000, 2) if inf_us else "N/A",
            "peak_memory_mb":    round(mem_b / 1e6, 1) if mem_b else "N/A",
            "primary_compute":   cu,
        }
    except Exception as exc:   # noqa: BLE001
        print(f"  ⚠  Could not parse profile: {exc}")
        return {"label": label}


def _print_comparison(results: list[dict], device: str) -> None:
    SEP = "─" * 64
    print(f"\n{SEP}")
    print(f"  TRANSLATION BENCHMARK  —  {device}")
    print(SEP)
    for m in results:
        if not m:
            continue
        print(f"\n  Model          : {m.get('label')}")
        print(f"  Inference time : {m.get('inference_time_ms', 'N/A')} ms")
        print(f"  Peak memory    : {m.get('peak_memory_mb', 'N/A')} MB")
        print(f"  Compute unit   : {m.get('primary_compute', 'N/A')}")
    print(f"\n{SEP}")
    print("  Recommendation: choose the model with lower inference_time_ms")
    print("  and NPU as primary compute unit for your submission claim.")
    print(SEP)


# ─────────────────────────────────────────────────────────────────────────────
# Option A — IndicTrans2  (download pre-exported ONNX from HuggingFace)
# ─────────────────────────────────────────────────────────────────────────────

def export_indictrans2(output_dir: Path) -> dict[str, Path]:
    """
    Download the pre-exported IndicTrans2-distilled-200M ONNX files from
    TigreGotico/indictrans2-en-indic-dist-200M-onnx on HuggingFace.

    Why we do NOT run optimum-cli here:
      IndicTrans2 uses a non-standard architecture (IndicTrans, not M2M100).
      The ONNX export requires a custom OnnxConfig that patches the field-name
      mismatch between IndicTrans and HuggingFace's M2M100OnnxConfig.
      TigreGotico has already done this work and published validated exports.

    Returns dict with keys: encoder, decoder, decoder_with_past, int8_encoder,
                              int8_decoder, int8_decoder_with_past
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    int8_dir = output_dir / "int8"
    int8_dir.mkdir(exist_ok=True)

    print(f"\n[IndicTrans2] Downloading ONNX from {IT2_ONNX_REPO} …")
    print("  (fp32: ~1.87 GB  |  int8: ~472 MB — be patient on first run)")

    try:
        from huggingface_hub import snapshot_download  # noqa: PLC0415
    except ImportError:
        print("  ERROR: huggingface_hub not installed.")
        print("         pip install huggingface-hub==0.23.4")
        sys.exit(1)

    # Download entire repo (includes int8/ sub-folder)
    local_repo = snapshot_download(
        repo_id=IT2_ONNX_REPO,
        local_dir=str(output_dir),
        ignore_patterns=["*.md", "*.txt", "*.json.bak"],
    )
    print(f"  ✓  Repo cached at {local_repo}")

    paths = {
        "encoder":             output_dir / "encoder_model.onnx",
        "decoder":             output_dir / "decoder_model.onnx",
        "decoder_with_past":   output_dir / "decoder_with_past_model.onnx",
        "int8_encoder":        output_dir / "int8" / "encoder_model.onnx",
        "int8_decoder":        output_dir / "int8" / "decoder_model.onnx",
        "int8_decoder_with_past": output_dir / "int8" / "decoder_with_past_model.onnx",
    }
    for key, p in paths.items():
        if p.exists():
            print(f"  ✓  {key:30s} {p.stat().st_size / 1e6:7.1f} MB")
        else:
            print(f"  ⚠  {key:30s} NOT FOUND at {p}")
    return paths


def compile_indictrans2(hub, paths: dict[str, Path], device: str,
                        use_int8: bool = True) -> dict[str, object]:
    """
    Compile the IndicTrans2 encoder + decoder for Snapdragon X Elite.

    AI Hub compile command equivalent (for your reference):
    ┌─────────────────────────────────────────────────────────────────────────┐
    │  qai-hub submit-compile-job                                             │
    │    --model models/indictrans2/int8/encoder_model.onnx                  │
    │    --input_specs "input_ids:(1,128) attention_mask:(1,128)"             │
    │    --device "Snapdragon X Elite CRD"                                    │
    │    --compile_options "--target_runtime onnx"                            │
    │    --name "indictrans2-encoder-int8-compile"                            │
    │                                                                          │
    │  qai-hub submit-compile-job                                             │
    │    --model models/indictrans2/int8/decoder_model.onnx                  │
    │    --input_specs "input_ids:(1,1) encoder_hidden_states:(1,128,512)     │
    │                   encoder_attention_mask:(1,128)"                        │
    │    --device "Snapdragon X Elite CRD"                                    │
    │    --compile_options "--target_runtime onnx"                            │
    │    --name "indictrans2-decoder-int8-compile"                            │
    └─────────────────────────────────────────────────────────────────────────┘
    """
    suffix = "int8" if use_int8 else "fp32"
    base   = paths["int8_encoder"].parent if use_int8 else paths["encoder"].parent

    enc_src = base / "encoder_model.onnx"
    dec_src = base / "decoder_model.onnx"

    print(f"\n[IndicTrans2] Compiling {suffix} encoder …")
    with open(enc_src, "rb") as fh:
        enc_bytes = fh.read()
    enc_uploaded = hub.upload_model(enc_bytes)

    enc_job = hub.submit_compile_job(
        model=enc_uploaded,
        name=f"indictrans2-encoder-{suffix}-compile",
        device=hub.Device(device),
        input_specs=IT2_ENCODER_SPECS,
        options="--target_runtime onnx",
    )
    print(f"  ✓  Encoder compile job → {enc_job.url}")

    print(f"[IndicTrans2] Compiling {suffix} decoder …")
    with open(dec_src, "rb") as fh:
        dec_bytes = fh.read()
    dec_uploaded = hub.upload_model(dec_bytes)

    dec_job = hub.submit_compile_job(
        model=dec_uploaded,
        name=f"indictrans2-decoder-{suffix}-compile",
        device=hub.Device(device),
        input_specs=DECODER_SPECS,
        options="--target_runtime onnx",
    )
    print(f"  ✓  Decoder compile job → {dec_job.url}")

    enc_compiled = _wait_compile(
        enc_job, "IndicTrans2 encoder",
        base / "encoder_model_compiled.onnx"
    )
    dec_compiled = _wait_compile(
        dec_job, "IndicTrans2 decoder",
        base / "decoder_model_compiled.onnx"
    )
    return {"encoder": enc_compiled, "decoder": dec_compiled,
            "enc_path": base / "encoder_model_compiled.onnx",
            "dec_path": base / "decoder_model_compiled.onnx"}


# ─────────────────────────────────────────────────────────────────────────────
# Option B — NLLB-200-distilled-600M
# ─────────────────────────────────────────────────────────────────────────────

def export_nllb200(output_dir: Path) -> dict[str, Path]:
    """
    Export facebook/nllb-200-distilled-600M to ONNX via optimum-cli, then
    apply dynamic INT8 quantization with onnxruntime.quantization.

    Equivalent shell commands (for your reference):
    ┌─────────────────────────────────────────────────────────────────────────┐
    │  # FP32 export                                                          │
    │  optimum-cli export onnx \\                                             │
    │    --model facebook/nllb-200-distilled-600M \\                         │
    │    --task seq2seq-lm \\                                                 │
    │    --opset 17 \\                                                        │
    │    models/nllb200/                                                      │
    │                                                                          │
    │  # Dynamic INT8 quantization (x86 host, not ARM)                       │
    │  python -c "                                                             │
    │    from onnxruntime.quantization import quantize_dynamic, QuantType     │
    │    quantize_dynamic('models/nllb200/encoder_model.onnx',                │
    │                     'models/nllb200/int8/encoder_model.onnx',           │
    │                     weight_type=QuantType.QInt8)"                       │
    └─────────────────────────────────────────────────────────────────────────┘
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    int8_dir = output_dir / "int8"
    int8_dir.mkdir(exist_ok=True)

    fp32_enc = output_dir / "encoder_model.onnx"
    fp32_dec = output_dir / "decoder_model.onnx"
    fp32_dwp = output_dir / "decoder_with_past_model.onnx"

    # ── Step 1: FP32 ONNX export ─────────────────────────────────────────────
    if fp32_enc.exists():
        print(f"\n[NLLB-200] FP32 ONNX already exists at {output_dir} — skipping export.")
    else:
        print(f"\n[NLLB-200] Exporting {NLLB_HF_REPO} to ONNX …")
        print("  (first run downloads ~2.5 GB weights from HuggingFace)")
        _run([
            sys.executable, "-m", "optimum.exporters.onnx",
            "--model",   NLLB_HF_REPO,
            "--task",    "seq2seq-lm",
            "--opset",   "17",
            str(output_dir),
        ])
        print("  ✓  FP32 ONNX export complete.")

    # ── Step 2: Dynamic INT8 quantization ────────────────────────────────────
    int8_enc = int8_dir / "encoder_model.onnx"
    int8_dec = int8_dir / "decoder_model.onnx"
    int8_dwp = int8_dir / "decoder_with_past_model.onnx"

    if int8_enc.exists():
        print(f"[NLLB-200] INT8 models already exist at {int8_dir} — skipping quant.")
    else:
        print("[NLLB-200] Quantizing to INT8 (dynamic, weight-only) …")
        try:
            from onnxruntime.quantization import quantize_dynamic, QuantType  # noqa: PLC0415
            for src, dst, label in [
                (fp32_enc, int8_enc, "encoder"),
                (fp32_dec, int8_dec, "decoder"),
                (fp32_dwp, int8_dwp, "decoder_with_past"),
            ]:
                if not src.exists():
                    print(f"  ⚠  {label}: {src} not found — skipping")
                    continue
                print(f"  Quantizing {label} …", end="", flush=True)
                quantize_dynamic(
                    str(src), str(dst),
                    weight_type=QuantType.QInt8,
                    extra_options={"ActivationSymmetric": True},
                )
                ratio = dst.stat().st_size / src.stat().st_size
                print(f" ✓  {ratio:.1%} of FP32 size ({dst.stat().st_size / 1e6:.0f} MB)")
        except ImportError:
            print("  ERROR: onnxruntime not installed or wrong version.")
            print("         pip install onnxruntime==1.18.1")
            sys.exit(1)

    paths = {
        "encoder":                fp32_enc,
        "decoder":                fp32_dec,
        "decoder_with_past":      fp32_dwp,
        "int8_encoder":           int8_enc,
        "int8_decoder":           int8_dec,
        "int8_decoder_with_past": int8_dwp,
    }
    for key, p in paths.items():
        if p.exists():
            print(f"  ✓  {key:30s} {p.stat().st_size / 1e6:7.1f} MB")
    return paths


def compile_nllb200(hub, paths: dict[str, Path], device: str,
                    use_int8: bool = True) -> dict[str, object]:
    """
    Compile NLLB-200 encoder + decoder for Snapdragon X Elite.

    AI Hub compile commands (for your reference):
    ┌─────────────────────────────────────────────────────────────────────────┐
    │  qai-hub submit-compile-job                                             │
    │    --model models/nllb200/int8/encoder_model.onnx                      │
    │    --input_specs "input_ids:(1,128) attention_mask:(1,128)"             │
    │    --device "Snapdragon X Elite CRD"                                    │
    │    --compile_options "--target_runtime onnx"                            │
    │    --name "nllb200-encoder-int8-compile"                                │
    │                                                                          │
    │  qai-hub submit-compile-job                                             │
    │    --model models/nllb200/int8/decoder_model.onnx                      │
    │    --input_specs "input_ids:(1,1) encoder_hidden_states:(1,128,1024)    │
    │                   encoder_attention_mask:(1,128)"                        │
    │    --device "Snapdragon X Elite CRD"                                    │
    │    --compile_options "--target_runtime onnx"                            │
    │    --name "nllb200-decoder-int8-compile"                                │
    └─────────────────────────────────────────────────────────────────────────┘

    NOTE: NLLB-200-600M uses d_model=1024; IndicTrans2-200M uses d_model=512.
    """
    suffix = "int8" if use_int8 else "fp32"
    base   = paths["int8_encoder"].parent if use_int8 else paths["encoder"].parent

    # NLLB-600M hidden size is 1024
    nllb_dec_specs = {
        "input_ids":               (1, 1),
        "encoder_hidden_states":   (1, ENCODER_SEQ_LEN, 1024),
        "encoder_attention_mask":  (1, ENCODER_SEQ_LEN),
    }

    enc_src = base / "encoder_model.onnx"
    dec_src = base / "decoder_model.onnx"

    print(f"\n[NLLB-200] Compiling {suffix} encoder …")
    with open(enc_src, "rb") as fh:
        enc_uploaded = hub.upload_model(fh.read())
    enc_job = hub.submit_compile_job(
        model=enc_uploaded,
        name=f"nllb200-encoder-{suffix}-compile",
        device=hub.Device(device),
        input_specs=NLLB_ENCODER_SPECS,
        options="--target_runtime onnx",
    )
    print(f"  ✓  Encoder compile job → {enc_job.url}")

    print(f"[NLLB-200] Compiling {suffix} decoder …")
    with open(dec_src, "rb") as fh:
        dec_uploaded = hub.upload_model(fh.read())
    dec_job = hub.submit_compile_job(
        model=dec_uploaded,
        name=f"nllb200-decoder-{suffix}-compile",
        device=hub.Device(device),
        input_specs=nllb_dec_specs,
        options="--target_runtime onnx",
    )
    print(f"  ✓  Decoder compile job → {dec_job.url}")

    enc_compiled = _wait_compile(
        enc_job, "NLLB-200 encoder",
        base / "encoder_model_compiled.onnx"
    )
    dec_compiled = _wait_compile(
        dec_job, "NLLB-200 decoder",
        base / "decoder_model_compiled.onnx"
    )
    return {"encoder": enc_compiled, "decoder": dec_compiled,
            "enc_path": base / "encoder_model_compiled.onnx",
            "dec_path": base / "decoder_model_compiled.onnx"}


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(
        description="Export, quantize, compile, and profile translation models."
    )
    p.add_argument("--model", default="both",
                   choices=["indictrans2", "nllb200", "both"],
                   help="Which model family to process (default: both)")
    p.add_argument("--device", default=DEVICE_DEFAULT,
                   help=f"AI Hub device (default: {DEVICE_DEFAULT})")
    p.add_argument("--fp32", action="store_true",
                   help="Compile FP32 instead of INT8")
    p.add_argument("--export-only", action="store_true",
                   help="Export/download ONNX files only; skip AI Hub")
    p.add_argument("--profile-only", action="store_true",
                   help="Profile already-compiled models only")
    args = p.parse_args()

    use_int8   = not args.fp32
    do_it2     = args.model in ("indictrans2", "both")
    do_nllb    = args.model in ("nllb200", "both")

    it2_paths  = {}
    nllb_paths = {}
    it2_result = {}
    nllb_result= {}

    # ── Export / download ─────────────────────────────────────────────────────
    if not args.profile_only:
        if do_it2:
            it2_paths = export_indictrans2(IT2_DIR)
        if do_nllb:
            nllb_paths = export_nllb200(NLLB_DIR)

    if args.export_only:
        print("\n✅  Export complete.  Run without --export-only to compile on AI Hub.")
        _print_env_hints(it2_paths, nllb_paths, use_int8)
        return

    # ── AI Hub compile + profile ──────────────────────────────────────────────
    hub = _check_hub()
    results: list[dict] = []

    if do_it2:
        if args.profile_only:
            suffix  = "int8" if use_int8 else "fp32"
            base    = IT2_DIR / ("int8" if use_int8 else "")
            enc_compiled = hub.upload_model(
                str(base / "encoder_model_compiled.onnx")
            )
            dec_compiled = hub.upload_model(
                str(base / "decoder_model_compiled.onnx")
            )
        else:
            compiled = compile_indictrans2(hub, it2_paths, args.device, use_int8)
            enc_compiled = compiled["encoder"]
            dec_compiled = compiled["decoder"]

        suffix = "int8" if use_int8 else "fp32"
        enc_m  = _profile(hub, enc_compiled, args.device,
                          f"it2-enc-{suffix}-profile",
                          f"IndicTrans2 Encoder ({suffix})")
        dec_m  = _profile(hub, dec_compiled, args.device,
                          f"it2-dec-{suffix}-profile",
                          f"IndicTrans2 Decoder ({suffix})")
        results.extend([enc_m, dec_m])

    if do_nllb:
        if args.profile_only:
            suffix  = "int8" if use_int8 else "fp32"
            base    = NLLB_DIR / ("int8" if use_int8 else "")
            enc_compiled = hub.upload_model(
                str(base / "encoder_model_compiled.onnx")
            )
            dec_compiled = hub.upload_model(
                str(base / "decoder_model_compiled.onnx")
            )
        else:
            compiled = compile_nllb200(hub, nllb_paths, args.device, use_int8)
            enc_compiled = compiled["encoder"]
            dec_compiled = compiled["decoder"]

        suffix = "int8" if use_int8 else "fp32"
        enc_m  = _profile(hub, enc_compiled, args.device,
                          f"nllb-enc-{suffix}-profile",
                          f"NLLB-200 Encoder ({suffix})")
        dec_m  = _profile(hub, dec_compiled, args.device,
                          f"nllb-dec-{suffix}-profile",
                          f"NLLB-200 Decoder ({suffix})")
        results.extend([enc_m, dec_m])

    _print_comparison(results, args.device)
    _print_env_hints(it2_paths, nllb_paths, use_int8)
    print("\n✅  Done.")


def _print_env_hints(it2_paths: dict, nllb_paths: dict, use_int8: bool) -> None:
    """Tell the user which env vars to set in .env based on what was exported."""
    suffix = "int8" if use_int8 else ""
    print("\n  Add whichever model you chose to your .env:")
    if it2_paths:
        base = "models/indictrans2/int8" if use_int8 else "models/indictrans2"
        print(f"    # IndicTrans2")
        print(f"    TRANSLATE_MODEL=indictrans2")
        print(f"    IT2_ENCODER_ONNX={base}/encoder_model_compiled.onnx")
        print(f"    IT2_DECODER_ONNX={base}/decoder_model_compiled.onnx")
    if nllb_paths:
        base = "models/nllb200/int8" if use_int8 else "models/nllb200"
        print(f"    # NLLB-200")
        print(f"    TRANSLATE_MODEL=nllb200")
        print(f"    NLLB_ENCODER_ONNX={base}/encoder_model_compiled.onnx")
        print(f"    NLLB_DECODER_ONNX={base}/decoder_model_compiled.onnx")


if __name__ == "__main__":
    main()
