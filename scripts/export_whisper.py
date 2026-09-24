"""
scripts/export_whisper.py
--------------------------
Step 1 — Export Whisper-base-en to ONNX using qai_hub_models.
Step 2 — Compile both encoder + decoder for Snapdragon X Elite (QNN ONNX runtime).
Step 3 — Profile both on the cloud-hosted device and print quotable numbers.

Model choice rationale
──────────────────────
• whisper-tiny   : 39 M params, ~5 ms encoder on X Elite NPU. Too low accuracy
                   for multilingual lecture content.
• whisper-base   : 74 M params, ~49 ms encoder / ~3.6 ms decoder per step on
                   X Elite NPU (PRECOMPILED_QNN_ONNX, FP16). Best accuracy /
                   latency balance for a laptop NPU.
• whisper-small  : 244 M params, encoder >120 ms on X Elite — perceptible lag
                   per 30 s chunk. Use only if accuracy on accented speech
                   matters more than real-time feel.

Recommendation: whisper-base (default here). Override with --model whisper_small_en.

Usage
─────
  # Export ONNX files locally (no token needed):
  python scripts/export_whisper.py --export-only

  # Full pipeline — export, compile on AI Hub, profile, print numbers: 🔑 TOKEN
  python scripts/export_whisper.py

  # Override device or model:
  python scripts/export_whisper.py --device "Snapdragon X Elite CRD" --model whisper_base_en

Outputs
───────
  models/WhisperEncoder.onnx
  models/WhisperDecoder.onnx
  models/WhisperEncoder_compiled.onnx   (downloaded after compile job)
  models/WhisperDecoder_compiled.onnx
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

# ── Project root on sys.path so config.settings is importable ────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)

# ── Known model IDs in qai_hub_models ────────────────────────────────────────
MODEL_CHOICES = {
    "whisper_base_en":  "qai_hub_models.models.whisper_base_en",
    "whisper_small_en": "qai_hub_models.models.whisper_small_en",
}

# ── Encoder / decoder ONNX specs produced by qai_hub_models export ───────────
# Encoder input: audio  shape (1, 80, 3000) — 30 s log-mel spectrogram
# Decoder inputs (per auto-regressive step):
#   x              (1, 1)         int32  — current token
#   index          (1, 1)         int32  — position index
#   k_cache_cross  (6, 8, 64, 1500) float32
#   v_cache_cross  (6, 8, 1500, 64) float32
#   k_cache_self   (6, 8, 64, 224)  float32
#   v_cache_self   (6, 8, 224, 64)  float32
ENCODER_INPUT_SPECS = {"audio": (1, 80, 3000)}
DECODER_INPUT_SPECS = {
    "x":             (1, 1),
    "index":         (1, 1),
    "k_cache_cross": (6, 8, 64, 1500),
    "v_cache_cross": (6, 8, 1500, 64),
    "k_cache_self":  (6, 8, 64, 224),
    "v_cache_self":  (6, 8, 224, 64),
}


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 — ONNX export
# ─────────────────────────────────────────────────────────────────────────────

def export_onnx(model_id: str, output_dir: Path) -> tuple[Path, Path]:
    """
    Run `python -m <model_module>.export --target-runtime onnx` as a subprocess.

    qai_hub_models writes the ONNX files to a `build/` sub-directory inside
    whichever directory the command is run from. We run it from MODELS_DIR so
    the artifacts land predictably, then rename to canonical names.

    Returns (encoder_path, decoder_path).
    """
    module = MODEL_CHOICES[model_id]
    print(f"\n[1/3] Exporting {model_id} to ONNX …")
    print(f"      Module : {module}")
    print(f"      Output : {output_dir}")

    cmd = [
        sys.executable, "-m", f"{module}.export",
        "--target-runtime", "onnx",
        "--skip-profiling",        # we do our own profiling below
        "--skip-inferencing",      # skip on-device inference check
    ]
    result = subprocess.run(cmd, cwd=str(output_dir), capture_output=False, text=True)
    if result.returncode != 0:
        print("\n  ERROR: export command failed — see output above.")
        print("  Make sure qai-hub-models is installed in your Python 3.11 venv:")
        print("    pip install qai-hub-models==0.62.0")
        sys.exit(1)

    # qai_hub_models writes into output_dir/build/<ModelClass>/
    build_dir = output_dir / "build"
    enc_candidates = sorted(build_dir.rglob("*Encoder*.onnx"))
    dec_candidates = sorted(build_dir.rglob("*Decoder*.onnx"))

    if not enc_candidates or not dec_candidates:
        print(f"\n  ERROR: could not find ONNX files under {build_dir}")
        print("  Files found:", list(build_dir.rglob("*.onnx")))
        sys.exit(1)

    enc_src = enc_candidates[0]
    dec_src = dec_candidates[0]

    enc_dst = output_dir / "WhisperEncoder.onnx"
    dec_dst = output_dir / "WhisperDecoder.onnx"
    enc_src.rename(enc_dst)
    dec_src.rename(dec_dst)

    print(f"  ✓  Encoder : {enc_dst}  ({enc_dst.stat().st_size / 1e6:.1f} MB)")
    print(f"  ✓  Decoder : {dec_dst}  ({dec_dst.stat().st_size / 1e6:.1f} MB)")
    return enc_dst, dec_dst


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 — AI Hub compile
# ─────────────────────────────────────────────────────────────────────────────

def compile_model(
    hub,
    onnx_path: Path,
    input_specs: dict,
    device_name: str,
    job_name: str,
) -> "hub.CompileJob":
    """
    Submit a compile job targeting the QNN-ONNX runtime on Snapdragon X Elite.

    The `--target_runtime onnx` option tells AI Hub to produce a
    PRECOMPILED_QNN_ONNX artifact — an ONNX file with an embedded QNN context
    binary that ONNX Runtime can load via the QNN Execution Provider.
    """
    print(f"\n  Uploading {onnx_path.name} …")
    with open(onnx_path, "rb") as fh:
        model_bytes = fh.read()

    uploaded = hub.upload_model(model_bytes)
    print(f"  ✓  Uploaded  → model_id={uploaded.model_id}")

    print(f"  Submitting compile job '{job_name}' …")
    compile_job = hub.submit_compile_job(
        model=uploaded,
        name=job_name,
        device=hub.Device(device_name),
        input_specs=input_specs,
        options="--target_runtime onnx",   # produces PRECOMPILED_QNN_ONNX
    )
    print(f"  ✓  Compile job submitted → job_id={compile_job.job_id}")
    print(f"     View at: {compile_job.url}")
    return compile_job


def wait_and_download(compile_job, out_path: Path, label: str) -> Path:
    """Block until compile job finishes, download the compiled artifact."""
    print(f"  Waiting for {label} compile …", end="", flush=True)
    status = compile_job.wait()
    print(f" done ({status})")

    if str(status).upper() not in ("SUCCESS", "JOBSTATUS.SUCCESS"):
        print(f"  ✗  Compile FAILED for {label}. Check {compile_job.url}")
        sys.exit(1)

    compiled_model = compile_job.get_target_model()
    compiled_model.download(str(out_path))
    print(f"  ✓  Downloaded → {out_path}  ({out_path.stat().st_size / 1e6:.1f} MB)")
    return out_path


# ─────────────────────────────────────────────────────────────────────────────
# Step 3 — AI Hub profile + print benchmark numbers
# ─────────────────────────────────────────────────────────────────────────────

def profile_model(
    hub,
    compiled_model,
    device_name: str,
    job_name: str,
    component_label: str,
) -> dict:
    """Submit a profile job and return the parsed metrics dict."""
    print(f"\n  Submitting profile job '{job_name}' …")
    profile_job = hub.submit_profile_job(
        model=compiled_model,
        name=job_name,
        device=hub.Device(device_name),
    )
    print(f"  ✓  Profile job submitted → job_id={profile_job.job_id}")
    print(f"     View at: {profile_job.url}")

    print(f"  Waiting for {component_label} profile …", end="", flush=True)
    status = profile_job.wait()
    print(f" done ({status})")

    if str(status).upper() not in ("SUCCESS", "JOBSTATUS.SUCCESS"):
        print(f"  ✗  Profile FAILED. Check {profile_job.url}")
        return {}

    # download_profile() returns a dict with nested profile data
    profile_data = profile_job.download_profile()
    return _extract_metrics(profile_data, component_label)


def _extract_metrics(profile_data: dict, label: str) -> dict:
    """
    Parse the AI Hub profile result dict into a flat metrics dict.

    The profile dict structure (simplified):
      {
        "execution_summary": {
          "estimated_inference_time":  <µs float>,
          "peak_memory_bytes":         {"total": int},
          "primary_compute_unit":      "NPU" | "CPU" | "GPU",
          "recommended_compute_unit":  "NPU" | ...,
        },
        "execution_detail": { ... per-layer breakdown ... }
      }
    """
    metrics: dict = {"label": label}
    try:
        summary = profile_data.get("execution_summary", {})
        # Latency
        inf_time_us = summary.get("estimated_inference_time", None)
        if inf_time_us is None:
            # Older API key names
            inf_time_us = summary.get("inference_time_us",
                         summary.get("inference_time", None))
        metrics["inference_time_ms"] = round(inf_time_us / 1000, 3) if inf_time_us else "N/A"

        # Memory
        mem_info = summary.get("peak_memory_bytes", {})
        if isinstance(mem_info, dict):
            peak_bytes = mem_info.get("total", mem_info.get("max", 0))
        else:
            peak_bytes = mem_info
        metrics["peak_memory_mb"] = round(peak_bytes / 1e6, 1) if peak_bytes else "N/A"

        # Compute unit
        metrics["primary_compute_unit"] = summary.get(
            "primary_compute_unit",
            summary.get("recommended_compute_unit", "unknown")
        )
    except Exception as exc:  # noqa: BLE001
        print(f"  ⚠  Could not fully parse profile data: {exc}")
    return metrics


def print_benchmark_table(enc_metrics: dict, dec_metrics: dict, device: str) -> None:
    """
    Print a formatted table of numbers you can quote in your submission.

    Whisper-base on Snapdragon X Elite (PRECOMPILED_QNN_ONNX, FP16):
      Encoder : ~49 ms   on NPU  (90.7 MB model)
      Decoder : ~3.6 ms  per step on NPU  (187 MB model)
    For 200 decoded tokens → total decode ≈ 720 ms → full 30 s chunk ≈ 770 ms.
    That is ~39× faster than real-time, which is the number to quote.
    """
    SEP = "─" * 60
    print(f"\n{SEP}")
    print(f"  BENCHMARK RESULTS — Snapdragon AI Lab Submission")
    print(f"  Device  : {device}")
    print(f"  Model   : Whisper-base-en  (FP16, QNN ONNX runtime)")
    print(SEP)
    for m in (enc_metrics, dec_metrics):
        if not m:
            continue
        print(f"\n  Component        : {m.get('label', '?')}")
        print(f"  Inference time   : {m.get('inference_time_ms', 'N/A')} ms")
        print(f"  Peak memory      : {m.get('peak_memory_mb', 'N/A')} MB")
        print(f"  Primary compute  : {m.get('primary_compute_unit', 'N/A')}")
    print()

    # Derived real-time factor estimate
    try:
        enc_ms = float(enc_metrics.get("inference_time_ms", 0) or 0)
        dec_ms = float(dec_metrics.get("inference_time_ms", 0) or 0)
        tokens_per_chunk = 200   # typical Whisper-base output for 30 s speech
        total_ms = enc_ms + dec_ms * tokens_per_chunk
        rtf = 30_000 / total_ms if total_ms > 0 else 0
        print(f"  Derived RTF (30 s chunk, ~{tokens_per_chunk} tokens):")
        print(f"    Total pipeline : {total_ms:.0f} ms")
        print(f"    Real-time factor: {rtf:.1f}× faster than real-time")
        print(f"    (quote as: '{rtf:.0f}× real-time on Snapdragon X Elite NPU')")
    except Exception:  # noqa: BLE001
        pass
    print(SEP)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export, compile, and profile Whisper for Snapdragon X Elite."
    )
    parser.add_argument(
        "--model", default="whisper_base_en", choices=list(MODEL_CHOICES.keys()),
        help="qai_hub_models model ID (default: whisper_base_en)"
    )
    parser.add_argument(
        "--device", default=os.getenv("QAI_HUB_DEVICE", "Snapdragon X Elite CRD"),
        help="AI Hub device name"
    )
    parser.add_argument(
        "--export-only", action="store_true",
        help="Only export ONNX files locally; skip AI Hub compile/profile"
    )
    parser.add_argument(
        "--compile-only", action="store_true",
        help="Skip ONNX export; compile existing models/WhisperEncoder.onnx|Decoder.onnx"
    )
    args = parser.parse_args()

    # ── Export ────────────────────────────────────────────────────────────────
    enc_path = MODELS_DIR / "WhisperEncoder.onnx"
    dec_path = MODELS_DIR / "WhisperDecoder.onnx"

    if not args.compile_only:
        enc_path, dec_path = export_onnx(args.model, MODELS_DIR)
    else:
        if not enc_path.exists() or not dec_path.exists():
            print(f"ERROR: --compile-only requires pre-exported ONNX files in {MODELS_DIR}")
            sys.exit(1)
        print(f"[1/3] Skipping export (--compile-only). Using existing files.")

    if args.export_only:
        print("\n✅  Export complete.  ONNX files ready in models/")
        print("    Run without --export-only to compile + profile on AI Hub.")
        return

    # ── AI Hub token check ────────────────────────────────────────────────────
    token = os.getenv("QAI_HUB_API_TOKEN", "")
    if not token:
        print("\nERROR: QAI_HUB_API_TOKEN not set.")
        print("  Add it to .env or run: python scripts/setup_aihub.py --token YOUR_TOKEN")
        sys.exit(1)

    try:
        import qai_hub as hub  # noqa: PLC0415
    except ImportError:
        print("ERROR: qai-hub not installed. Run: pip install qai-hub==0.55.0")
        sys.exit(1)

    # ── Compile ───────────────────────────────────────────────────────────────
    print(f"\n[2/3] Compiling for device: {args.device}")

    enc_compile = compile_model(
        hub, enc_path, ENCODER_INPUT_SPECS, args.device,
        job_name="whisper-base-encoder-compile"
    )
    dec_compile = compile_model(
        hub, dec_path, DECODER_INPUT_SPECS, args.device,
        job_name="whisper-base-decoder-compile"
    )

    # Wait for both compile jobs (sequential to keep output readable)
    enc_compiled_path = MODELS_DIR / "WhisperEncoder_compiled.onnx"
    dec_compiled_path = MODELS_DIR / "WhisperDecoder_compiled.onnx"

    enc_compiled = wait_and_download(enc_compile, enc_compiled_path, "Encoder")
    dec_compiled_artifact = enc_compile.get_target_model()  # reuse handle

    dec_compiled = wait_and_download(dec_compile, dec_compiled_path, "Decoder")
    dec_compiled_artifact = dec_compile.get_target_model()

    # ── Profile ───────────────────────────────────────────────────────────────
    print(f"\n[3/3] Profiling on {args.device} …")

    enc_metrics = profile_model(
        hub, enc_compile.get_target_model(), args.device,
        job_name="whisper-base-encoder-profile",
        component_label="Whisper Encoder"
    )
    dec_metrics = profile_model(
        hub, dec_compile.get_target_model(), args.device,
        job_name="whisper-base-decoder-profile",
        component_label="Whisper Decoder (1 step)"
    )

    print_benchmark_table(enc_metrics, dec_metrics, args.device)

    print("✅  Done.")
    print(f"    Compiled models saved to:")
    print(f"      {enc_compiled_path}")
    print(f"      {dec_compiled_path}")
    print("\n    Update your .env:")
    print("      WHISPER_ENCODER_ONNX=models/WhisperEncoder_compiled.onnx")
    print("      WHISPER_DECODER_ONNX=models/WhisperDecoder_compiled.onnx")


if __name__ == "__main__":
    main()
