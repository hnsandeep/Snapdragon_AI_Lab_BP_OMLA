"""
scripts/export_summarizer.py
─────────────────────────────
Export, compile, and profile the summarization model for Snapdragon X Elite.

Model choice
────────────
  facebook/bart-large-cnn   (English, ~1.63 GB FP32, ~410 MB INT8)
    Best quality for English lecture transcripts.  Standard CNN/DailyMail
    fine-tune; produces tight extractive-style abstractive summaries.

  csebuetnlp/mT5_multilingual_XLSum  (45 languages, ~2.2 GB FP32, ~560 MB INT8)
    Use when the source lecture is in a non-English language or when you want
    the summary in the same language as the transcript.

Export route
────────────
  Both models export cleanly via optimum-cli (seq2seq-lm task).
  Dynamic INT8 weight quantization applied with onnxruntime.quantization.

AI Hub compile target
─────────────────────
  --target_runtime onnx  →  PRECOMPILED_QNN_ONNX
  The encoder is the compute-heavy part (most MACs); the decoder is lighter
  and auto-regressive.  Both are compiled separately for maximum flexibility.

Usage
─────
  # Export ONNX + quantize locally (no token needed):
  python scripts/export_summarizer.py --export-only

  # Export, compile on AI Hub, profile:  🔑 TOKEN REQUIRED
  python scripts/export_summarizer.py

  # Use multilingual mT5 instead of BART:
  python scripts/export_summarizer.py --model mt5

  # Compile FP32 (no INT8 quant):
  python scripts/export_summarizer.py --fp32

Outputs
───────
  models/bart/                  encoder_model.onnx, decoder_model.onnx,
                                decoder_with_past_model.onnx,
                                int8/  (after quantization),
                                *_compiled.onnx (after AI Hub compile)
  models/mt5/                   same structure
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

MODELS_DIR     = ROOT / "models"
DEVICE_DEFAULT = os.getenv("QAI_HUB_DEVICE", "Snapdragon X Elite CRD")

# HuggingFace model IDs
_MODELS = {
    "bart": "facebook/bart-large-cnn",
    "mt5":  "csebuetnlp/mT5_multilingual_XLSum",
}

# Input specs for AI Hub compile jobs
# Encoder: token IDs + attention mask (fixed-length for NPU tile efficiency)
_ENC_SPECS = {
    "input_ids":      (1, 512),
    "attention_mask": (1, 512),
}
# Decoder first step (no KV cache)
# BART hidden size = 1024, mT5 = 512
_DEC_SPECS = {
    "bart": {
        "input_ids":               (1, 1),
        "encoder_hidden_states":   (1, 512, 1024),
        "encoder_attention_mask":  (1, 512),
    },
    "mt5": {
        "input_ids":               (1, 1),
        "encoder_hidden_states":   (1, 512, 512),
        "encoder_attention_mask":  (1, 512),
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# Helpers (same pattern as export_translation.py)
# ─────────────────────────────────────────────────────────────────────────────

def _run(cmd: list[str], cwd: str | None = None) -> None:
    import subprocess
    print("  $", " ".join(cmd))
    r = subprocess.run(cmd, cwd=cwd)
    if r.returncode != 0:
        print(f"  ERROR: command exited {r.returncode}")
        sys.exit(1)


def _check_hub():
    token = os.getenv("QAI_HUB_API_TOKEN", "")
    if not token:
        print("ERROR: QAI_HUB_API_TOKEN not set.")
        print("  Run: python scripts/setup_aihub.py --token YOUR_TOKEN")
        sys.exit(1)
    try:
        import qai_hub as hub
        return hub
    except ImportError:
        print("ERROR: pip install qai-hub==0.55.0")
        sys.exit(1)


def _wait_compile(job, label: str, out_path: Path):
    print(f"  Waiting for {label} …", end="", flush=True)
    status = job.wait()
    print(f" {status}")
    if "FAIL" in str(status).upper():
        print(f"  ✗  FAILED — {job.url}")
        sys.exit(1)
    compiled = job.get_target_model()
    compiled.download(str(out_path))
    print(f"  ✓  {out_path.name}  ({out_path.stat().st_size / 1e6:.0f} MB)")
    return compiled


def _extract_metrics(profile_data: dict, label: str) -> dict:
    summary = profile_data.get("execution_summary", {})
    inf_us  = (summary.get("estimated_inference_time")
               or summary.get("inference_time_us", None))
    mem_raw = summary.get("peak_memory_bytes", {})
    mem_b   = mem_raw.get("total", 0) if isinstance(mem_raw, dict) else (mem_raw or 0)
    cu      = summary.get("primary_compute_unit", "unknown")
    return {
        "label":             label,
        "inference_time_ms": round(inf_us / 1000, 2) if inf_us else "N/A",
        "peak_memory_mb":    round(mem_b / 1e6, 1) if mem_b else "N/A",
        "primary_compute":   cu,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 — ONNX export via optimum-cli
# ─────────────────────────────────────────────────────────────────────────────

def export_onnx(model_key: str, output_dir: Path) -> dict[str, Path]:
    """
    Export a seq2seq model to ONNX using optimum-cli.

    Equivalent shell command (for reference):
    ┌──────────────────────────────────────────────────────────────────────────┐
    │  # BART                                                                  │
    │  optimum-cli export onnx \\                                              │
    │    --model facebook/bart-large-cnn \\                                   │
    │    --task seq2seq-lm \\                                                  │
    │    --opset 17 \\                                                         │
    │    models/bart/                                                          │
    │                                                                          │
    │  # mT5                                                                   │
    │  optimum-cli export onnx \\                                              │
    │    --model csebuetnlp/mT5_multilingual_XLSum \\                         │
    │    --task seq2seq-lm \\                                                  │
    │    --opset 17 \\                                                         │
    │    models/mt5/                                                           │
    └──────────────────────────────────────────────────────────────────────────┘
    """
    hf_id = _MODELS[model_key]
    output_dir.mkdir(parents=True, exist_ok=True)
    enc_path = output_dir / "encoder_model.onnx"

    if enc_path.exists():
        print(f"[{model_key.upper()}] FP32 ONNX already at {output_dir} — skipping export.")
    else:
        print(f"\n[{model_key.upper()}] Exporting {hf_id} to ONNX …")
        print("  (first run downloads weights from HuggingFace)")
        _run([
            sys.executable, "-m", "optimum.exporters.onnx",
            "--model", hf_id,
            "--task",  "seq2seq-lm",
            "--opset", "17",
            str(output_dir),
        ])
        print(f"  ✓  FP32 export complete.")

    paths = {
        "encoder":            output_dir / "encoder_model.onnx",
        "decoder":            output_dir / "decoder_model.onnx",
        "decoder_with_past":  output_dir / "decoder_with_past_model.onnx",
    }
    for k, p in paths.items():
        if p.exists():
            print(f"  ✓  {k:25s}  {p.stat().st_size / 1e6:7.1f} MB")
        else:
            print(f"  ⚠  {k:25s}  NOT FOUND")
    return paths


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 — Dynamic INT8 quantization
# ─────────────────────────────────────────────────────────────────────────────

def quantize_int8(paths: dict[str, Path], output_dir: Path) -> dict[str, Path]:
    """
    Apply dynamic INT8 weight quantization with onnxruntime.quantization.

    Dynamic quantization:
      • Weights → INT8 at export time (no calibration data needed)
      • Activations → quantized at runtime
      • BART encoder: ~2.5× smaller, minimal accuracy loss on CNN/DM
      • Run on AMD x64 host only — not on ARM64

    Equivalent Python snippet (for reference):
    ┌──────────────────────────────────────────────────────────────────────────┐
    │  from onnxruntime.quantization import quantize_dynamic, QuantType        │
    │  quantize_dynamic(                                                       │
    │      "models/bart/encoder_model.onnx",                                  │
    │      "models/bart/int8/encoder_model.onnx",                             │
    │      weight_type=QuantType.QInt8,                                       │
    │      extra_options={"ActivationSymmetric": True},                       │
    │  )                                                                       │
    └──────────────────────────────────────────────────────────────────────────┘
    """
    int8_dir = output_dir / "int8"
    int8_dir.mkdir(exist_ok=True)

    enc_int8 = int8_dir / "encoder_model.onnx"
    if enc_int8.exists():
        print(f"  INT8 models already at {int8_dir} — skipping quantization.")
    else:
        print(f"\n  Quantizing to INT8 …")
        try:
            from onnxruntime.quantization import quantize_dynamic, QuantType
        except ImportError:
            print("  ERROR: pip install onnxruntime==1.18.1")
            sys.exit(1)

        for key in ("encoder", "decoder", "decoder_with_past"):
            src = paths.get(key)
            if src is None or not src.exists():
                print(f"  ⚠  {key}: not found, skipping")
                continue
            dst = int8_dir / src.name
            print(f"  Quantizing {key} …", end="", flush=True)
            quantize_dynamic(
                str(src), str(dst),
                weight_type=QuantType.QInt8,
                extra_options={"ActivationSymmetric": True},
            )
            ratio = dst.stat().st_size / src.stat().st_size
            print(f" ✓  {ratio:.1%} of FP32  ({dst.stat().st_size / 1e6:.0f} MB)")

    return {
        "encoder":            int8_dir / "encoder_model.onnx",
        "decoder":            int8_dir / "decoder_model.onnx",
        "decoder_with_past":  int8_dir / "decoder_with_past_model.onnx",
    }


# ─────────────────────────────────────────────────────────────────────────────
# Step 3 — AI Hub compile
# ─────────────────────────────────────────────────────────────────────────────

def compile_model(
    hub, paths: dict[str, Path], model_key: str,
    device: str, use_int8: bool,
) -> dict[str, object]:
    """
    Compile encoder + decoder for Snapdragon X Elite.

    AI Hub compile commands (for reference):
    ┌──────────────────────────────────────────────────────────────────────────┐
    │  # BART encoder INT8                                                     │
    │  qai-hub submit-compile-job \\                                           │
    │    --model models/bart/int8/encoder_model.onnx \\                       │
    │    --input_specs "input_ids:(1,512) attention_mask:(1,512)" \\           │
    │    --device "Snapdragon X Elite CRD" \\                                  │
    │    --compile_options "--target_runtime onnx" \\                         │
    │    --name "bart-encoder-int8-compile"                                   │
    │                                                                          │
    │  # BART decoder INT8                                                     │
    │  qai-hub submit-compile-job \\                                           │
    │    --model models/bart/int8/decoder_model.onnx \\                       │
    │    --input_specs "input_ids:(1,1)                                        │
    │                   encoder_hidden_states:(1,512,1024)                     │
    │                   encoder_attention_mask:(1,512)" \\                     │
    │    --device "Snapdragon X Elite CRD" \\                                  │
    │    --compile_options "--target_runtime onnx" \\                         │
    │    --name "bart-decoder-int8-compile"                                   │
    └──────────────────────────────────────────────────────────────────────────┘
    """
    src_paths = paths  # already int8 or fp32 depending on caller
    suffix    = "int8" if use_int8 else "fp32"
    dec_specs = _DEC_SPECS[model_key]
    label     = model_key.upper()

    results = {}
    jobs    = {}

    for component, spec, name_suffix in [
        ("encoder", _ENC_SPECS,  "encoder"),
        ("decoder", dec_specs,   "decoder"),
    ]:
        src = src_paths.get(component)
        if src is None or not src.exists():
            print(f"  ⚠  {component} ONNX not found, skipping compile.")
            continue
        print(f"\n[{label}] Uploading {component} ({src.stat().st_size / 1e6:.0f} MB) …")
        with open(src, "rb") as fh:
            uploaded = hub.upload_model(fh.read())
        job = hub.submit_compile_job(
            model=uploaded,
            name=f"{model_key}-{name_suffix}-{suffix}-compile",
            device=hub.Device(device),
            input_specs=spec,
            options="--target_runtime onnx",
        )
        print(f"  ✓  Compile job submitted → {job.url}")
        jobs[component] = job

    # Wait + download
    out_base = src_paths["encoder"].parent
    for component, job in jobs.items():
        out_path = out_base / f"{component}_model_compiled.onnx"
        compiled = _wait_compile(job, f"{label} {component}", out_path)
        results[component] = compiled
        results[f"{component}_path"] = out_path

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Step 4 — Profile + print benchmark numbers
# ─────────────────────────────────────────────────────────────────────────────

def profile_and_print(
    hub, compiled: dict, model_key: str, device: str, use_int8: bool
) -> None:
    suffix = "int8" if use_int8 else "fp32"
    label  = model_key.upper()
    metrics_list = []

    for component in ("encoder", "decoder"):
        model_handle = compiled.get(component)
        if model_handle is None:
            continue
        job_name = f"{model_key}-{component}-{suffix}-profile"
        print(f"\n  Profiling {label} {component} …")
        pjob = hub.submit_profile_job(
            model=model_handle,
            name=job_name,
            device=hub.Device(device),
        )
        print(f"  Profile job: {pjob.url}")
        pjob.wait()
        try:
            data    = pjob.download_profile()
            metrics = _extract_metrics(data, f"{label} {component} ({suffix})")
        except Exception as exc:
            metrics = {"label": f"{label} {component}", "error": str(exc)}
        metrics_list.append(metrics)

    SEP = "─" * 62
    print(f"\n{SEP}")
    print(f"  SUMMARIZER BENCHMARK  —  {device}")
    print(SEP)
    for m in metrics_list:
        print(f"\n  Component        : {m.get('label')}")
        print(f"  Inference time   : {m.get('inference_time_ms', 'N/A')} ms")
        print(f"  Peak memory      : {m.get('peak_memory_mb',   'N/A')} MB")
        print(f"  Primary compute  : {m.get('primary_compute',  'N/A')}")
    print(SEP)

    # Env var hints
    base = compiled.get("encoder_path")
    if base:
        out_dir = base.parent
        key_enc = f"SUMMARY_ENCODER_ONNX"
        key_dec = f"SUMMARY_DECODER_ONNX"
        print(f"\n  Add to .env:")
        print(f"    SUMMARY_MODEL={model_key}")
        print(f"    {key_enc}={out_dir}/encoder_model_compiled.onnx")
        print(f"    {key_dec}={out_dir}/decoder_model_compiled.onnx")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(
        description="Export, quantize (INT8), compile, and profile summarization model."
    )
    p.add_argument("--model",  default="bart", choices=["bart", "mt5"],
                   help="Model to export (default: bart)")
    p.add_argument("--device", default=DEVICE_DEFAULT)
    p.add_argument("--fp32",   action="store_true",
                   help="Use FP32 (skip INT8 quantization)")
    p.add_argument("--export-only", action="store_true",
                   help="Export + quantize only; skip AI Hub")
    args = p.parse_args()

    use_int8  = not args.fp32
    out_dir   = MODELS_DIR / args.model
    suffix    = "int8" if use_int8 else "fp32"

    # ── Export ────────────────────────────────────────────────────────────────
    fp32_paths = export_onnx(args.model, out_dir)

    # ── Quantize ──────────────────────────────────────────────────────────────
    if use_int8:
        src_paths = quantize_int8(fp32_paths, out_dir)
    else:
        src_paths = fp32_paths

    if args.export_only:
        print(f"\n✅  Export complete ({suffix}).")
        print(f"    Files in: {out_dir}" + (f"/int8" if use_int8 else ""))
        print("    Run without --export-only to compile on AI Hub.")
        return

    # ── Compile + profile ─────────────────────────────────────────────────────
    hub      = _check_hub()
    compiled = compile_model(hub, src_paths, args.model, args.device, use_int8)
    profile_and_print(hub, compiled, args.model, args.device, use_int8)
    print("\n✅  Done.")


if __name__ == "__main__":
    main()
