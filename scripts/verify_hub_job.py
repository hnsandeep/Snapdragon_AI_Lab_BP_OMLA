"""
scripts/verify_hub_job.py
--------------------------
Submits a minimal compile + profile job to a hosted Snapdragon X Elite device
on Qualcomm AI Hub, then polls until both jobs finish and prints the results.

This verifies your token, device access, and the full compile→profile pipeline
end-to-end before you invest time exporting real models.

The test model is a tiny 2-layer MobileNetV2-style bottleneck (< 0.1 MB) so
the job queues and completes quickly.

Usage:
    python scripts/verify_hub_job.py
    python scripts/verify_hub_job.py --device "Snapdragon X Elite CRD"
    python scripts/verify_hub_job.py --runtime onnx

Requires:
    QAI_HUB_API_TOKEN in .env or environment
    pip install qai-hub==0.55.0 torch==2.3.1 onnx==1.16.1
"""

import argparse
import sys
import os
import io
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

# ── Guard: token must exist before importing hub ──────────────────────────────
_token = os.getenv("QAI_HUB_API_TOKEN", "")
if not _token:
    print("ERROR: QAI_HUB_API_TOKEN is not set.")
    print("  Run setup first:  python scripts/setup_aihub.py --token YOUR_TOKEN")
    sys.exit(1)


def build_tiny_model():
    """Return a tiny PyTorch model as a traced torch.jit.ScriptModule."""
    import torch
    import torch.nn as nn

    class TinyBlock(nn.Module):
        """Minimal depthwise-separable block — representative of NPU workloads."""
        def __init__(self):
            super().__init__()
            self.dw = nn.Conv2d(16, 16, 3, padding=1, groups=16, bias=False)
            self.pw = nn.Conv2d(16, 32, 1, bias=False)
            self.bn = nn.BatchNorm2d(32)
            self.act = nn.ReLU6()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.act(self.bn(self.pw(self.dw(x))))

    model = TinyBlock().eval()
    dummy = torch.randn(1, 16, 64, 64)
    traced = torch.jit.trace(model, dummy)
    print(f"  ✓  Tiny model traced. Input shape: {list(dummy.shape)}")
    return traced, dummy


def export_to_onnx(traced_model, dummy_input) -> bytes:
    """Export the traced model to ONNX and return raw bytes (no file on disk)."""
    import torch
    buf = io.BytesIO()
    torch.onnx.export(
        traced_model,
        dummy_input,
        buf,
        opset_version=17,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
    )
    buf.seek(0)
    onnx_bytes = buf.read()
    print(f"  ✓  ONNX export complete ({len(onnx_bytes) / 1024:.1f} KB)")
    return onnx_bytes


def submit_compile_job(hub, onnx_bytes: bytes, device_name: str, runtime: str):
    """Upload model and submit a compile job."""
    print(f"\n[2/4] Uploading model to AI Hub …")
    model = hub.upload_model(onnx_bytes)
    print(f"  ✓  Model uploaded  → ID: {model.model_id}")

    print(f"[3/4] Submitting compile job (device={device_name!r}, runtime={runtime!r}) …")
    # ⚠ TOKEN NEEDED: compile_model calls AI Hub — requires a valid API token
    compile_job = hub.submit_compile_job(
        model=model,
        device=hub.Device(device_name),
        name="lecture-assistant-verify-compile",
        options=f"--target_runtime {runtime}",
    )
    print(f"  ✓  Compile job submitted → Job ID: {compile_job.job_id}")
    return model, compile_job


def submit_profile_job(hub, compiled_model, device_name: str):
    """Submit a profile job using the compiled model."""
    print(f"[4/4] Submitting profile job …")
    # ⚠ TOKEN NEEDED: submit_profile_job calls AI Hub — requires a valid API token
    profile_job = hub.submit_profile_job(
        model=compiled_model,
        device=hub.Device(device_name),
        name="lecture-assistant-verify-profile",
    )
    print(f"  ✓  Profile job submitted → Job ID: {profile_job.job_id}")
    return profile_job


def poll_job(job, label: str, poll_interval: int = 10):
    """Block until a job reaches a terminal state; print progress dots."""
    print(f"\n  Waiting for {label} to finish", end="", flush=True)
    while True:
        status = job.get_status()
        if status.finished:
            break
        if status.failed:
            print(f"\n  ✗  {label} FAILED: {status.message}")
            return False
        print(".", end="", flush=True)
        time.sleep(poll_interval)
    print(f"\n  ✓  {label} finished: {status}")
    return True


def print_profile_results(profile_job) -> None:
    """Extract and display key latency/memory metrics."""
    try:
        results = profile_job.download_results()
        inference_time = results.get("inference_time_us", "N/A")
        peak_memory    = results.get("peak_memory_bytes", "N/A")
        compute_units  = results.get("primary_compute_unit", "N/A")
        print("\n── Profile Results ──────────────────────────────────")
        print(f"  Inference latency : {inference_time} µs")
        print(f"  Peak memory       : {peak_memory} bytes")
        print(f"  Primary compute   : {compute_units}")
        print("─────────────────────────────────────────────────────")
    except Exception as exc:  # noqa: BLE001
        print(f"  (Could not parse profile results: {exc})")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Verify AI Hub compile+profile pipeline.")
    parser.add_argument("--device",  default=os.getenv("QAI_HUB_DEVICE", "Snapdragon X Elite CRD"))
    parser.add_argument("--runtime", default="onnx", choices=["onnx", "tflite", "qnn"])
    args = parser.parse_args()

    print("=" * 60)
    print("  AI Hub Compile + Profile Verification")
    print(f"  Device  : {args.device}")
    print(f"  Runtime : {args.runtime}")
    print("=" * 60)

    # Step 1 — build a tiny model locally
    print("\n[1/4] Building tiny test model …")
    try:
        traced, dummy = build_tiny_model()
        onnx_bytes = export_to_onnx(traced, dummy)
    except ImportError as exc:
        print(f"  ERROR: {exc}")
        print("  Make sure torch and onnx are installed in your venv.")
        sys.exit(1)

    # Step 2-4 — submit jobs to AI Hub
    try:
        import qai_hub as hub  # noqa: PLC0415
    except ImportError:
        print("  ERROR: qai-hub is not installed. Run: pip install qai-hub==0.55.0")
        sys.exit(1)

    try:
        _, compile_job = submit_compile_job(hub, onnx_bytes, args.device, args.runtime)
    except Exception as exc:  # noqa: BLE001
        print(f"\n  ERROR submitting compile job: {exc}")
        print("  Check your API token and device name.")
        sys.exit(1)

    # Poll compile job
    if not poll_job(compile_job, "compile job"):
        sys.exit(1)

    # Get compiled model artifact
    compiled_model = compile_job.get_target_model()

    # Submit and poll profile job
    try:
        profile_job = submit_profile_job(hub, compiled_model, args.device)
    except Exception as exc:  # noqa: BLE001
        print(f"\n  ERROR submitting profile job: {exc}")
        sys.exit(1)

    if not poll_job(profile_job, "profile job"):
        sys.exit(1)

    print_profile_results(profile_job)
    print("\n✅  Verification complete — AI Hub pipeline is working correctly.")
    print("   You can now export real Whisper / OPUS-MT / BART models and submit them.")


if __name__ == "__main__":
    main()
