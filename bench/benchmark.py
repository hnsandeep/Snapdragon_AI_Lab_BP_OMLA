"""bench/benchmark.py — AI Hub on-device profiling for all pipeline models.

Submits compile + profile jobs to a cloud-hosted Snapdragon X Elite CRD and
prints a Markdown table with: compile time, inference latency, peak memory,
and primary compute unit (NPU / CPU / GPU).

Usage
─────
    python bench/benchmark.py                      # profile all models
    python bench/benchmark.py --models whisper     # whisper only
    python bench/benchmark.py --device "Snapdragon X Elite CRD"

Outputs
───────
    bench/results.md   — Markdown table (overwritten on each run)

Requires: QAI_HUB_API_TOKEN in .env   🔑
"""
from __future__ import annotations
import argparse
import io
import os
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:
    pass

MODELS_DIR     = ROOT / "models"
RESULTS_MD     = Path(__file__).parent / "results.md"
DEVICE_DEFAULT = os.getenv("QAI_HUB_DEVICE", "Snapdragon X Elite CRD")

# ── Model specs for compile jobs ──────────────────────────────────────────────
MODEL_SPECS = {
    "whisper_encoder": {
        "path":        MODELS_DIR / "WhisperEncoder.onnx",
        "input_specs": {"audio": (1, 80, 3000)},
        "label":       "Whisper-base Encoder",
    },
    "whisper_decoder": {
        "path":        MODELS_DIR / "WhisperDecoder.onnx",
        "input_specs": {
            "x":             (1, 1),
            "index":         (1, 1),
            "k_cache_cross": (6, 8, 64, 1500),
            "v_cache_cross": (6, 8, 1500, 64),
            "k_cache_self":  (6, 8, 64, 224),
            "v_cache_self":  (6, 8, 224, 64),
        },
        "label": "Whisper-base Decoder (1 step)",
    },
    "it2_encoder": {
        "path":        MODELS_DIR / "indictrans2" / "int8" / "encoder_model.onnx",
        "input_specs": {"input_ids": (1, 128), "attention_mask": (1, 128)},
        "label":       "IndicTrans2-200M Encoder (INT8)",
    },
    "it2_decoder": {
        "path":        MODELS_DIR / "indictrans2" / "int8" / "decoder_model.onnx",
        "input_specs": {
            "input_ids":               (1, 1),
            "encoder_hidden_states":   (1, 128, 512),
            "encoder_attention_mask":  (1, 128),
        },
        "label": "IndicTrans2-200M Decoder (INT8)",
    },
    "nllb_encoder": {
        "path":        MODELS_DIR / "nllb200" / "int8" / "encoder_model.onnx",
        "input_specs": {"input_ids": (1, 128), "attention_mask": (1, 128)},
        "label":       "NLLB-200-600M Encoder (INT8)",
    },
    "nllb_decoder": {
        "path":        MODELS_DIR / "nllb200" / "int8" / "decoder_model.onnx",
        "input_specs": {
            "input_ids":               (1, 1),
            "encoder_hidden_states":   (1, 128, 1024),
            "encoder_attention_mask":  (1, 128),
        },
        "label": "NLLB-200-600M Decoder (INT8)",
    },
    "bart_encoder": {
        "path":        MODELS_DIR / "bart" / "int8" / "encoder_model.onnx",
        "input_specs": {"input_ids": (1, 512), "attention_mask": (1, 512)},
        "label":       "BART-large-cnn Encoder (INT8)",
    },
    "bart_decoder": {
        "path":        MODELS_DIR / "bart" / "int8" / "decoder_model.onnx",
        "input_specs": {
            "input_ids":               (1, 1),
            "encoder_hidden_states":   (1, 512, 1024),
            "encoder_attention_mask":  (1, 512),
        },
        "label": "BART-large-cnn Decoder (INT8)",
    },
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _check_hub():
    token = os.getenv("QAI_HUB_API_TOKEN", "")
    if not token:
        sys.exit("ERROR: QAI_HUB_API_TOKEN not set. Run scripts/setup_aihub.py --token YOUR_TOKEN")
    try:
        import qai_hub as hub
        return hub
    except ImportError:
        sys.exit("ERROR: pip install qai-hub==0.55.0")


def _compile_and_profile(hub, key: str, spec: dict, device: str) -> dict:
    path = spec["path"]
    if not Path(path).exists():
        return {"label": spec["label"], "skipped": True,
                "reason": f"ONNX not found: {path}"}

    t_start = time.time()
    with open(path, "rb") as fh:
        uploaded = hub.upload_model(fh.read())

    cjob = hub.submit_compile_job(
        model=uploaded,
        name=f"bench-{key}-compile",
        device=hub.Device(device),
        input_specs=spec["input_specs"],
        options="--target_runtime onnx",
    )
    print(f"  [{spec['label']}] compile submitted → {cjob.url}")
    cjob.wait()
    compile_s = time.time() - t_start

    if "FAIL" in str(cjob.get_status()).upper():
        return {"label": spec["label"], "error": "compile FAILED", "url": cjob.url}

    pjob = hub.submit_profile_job(
        model=cjob.get_target_model(),
        name=f"bench-{key}-profile",
        device=hub.Device(device),
    )
    print(f"  [{spec['label']}] profile  submitted → {pjob.url}")
    pjob.wait()

    try:
        data    = pjob.download_profile()
        summary = data.get("execution_summary", {})
        inf_us  = (summary.get("estimated_inference_time")
                   or summary.get("inference_time_us") or 0)
        mem_raw = summary.get("peak_memory_bytes", {})
        mem_b   = mem_raw.get("total", 0) if isinstance(mem_raw, dict) else (mem_raw or 0)
        cu      = summary.get("primary_compute_unit", "unknown")
    except Exception as exc:
        return {"label": spec["label"], "error": str(exc), "profile_url": pjob.url}

    return {
        "label":           spec["label"],
        "compile_s":       round(compile_s, 1),
        "latency_ms":      round(inf_us / 1000, 2) if inf_us else "N/A",
        "peak_mem_mb":     round(mem_b / 1e6, 1) if mem_b else "N/A",
        "compute_unit":    cu,
        "compile_url":     cjob.url,
        "profile_url":     pjob.url,
    }


def _render_markdown(rows: list[dict], device: str) -> str:
    lines = [
        f"# Benchmark Results — {device}",
        f"_Generated {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}_\n",
        "| Model | Compile (s) | Latency (ms) | Peak Memory (MB) | Compute Unit |",
        "|---|---|---|---|---|",
    ]
    for r in rows:
        if r.get("skipped"):
            lines.append(f"| {r['label']} | — | — | — | ⚠ {r['reason']} |")
        elif r.get("error"):
            lines.append(f"| {r['label']} | — | — | — | ✗ {r['error']} |")
        else:
            lines.append(
                f"| {r['label']} "
                f"| {r.get('compile_s','?')} "
                f"| {r.get('latency_ms','?')} "
                f"| {r.get('peak_mem_mb','?')} "
                f"| {r.get('compute_unit','?')} |"
            )
    lines += [
        "",
        "## Notes",
        "- Latency = single inference pass on the compiled QNN ONNX model.",
        "- Whisper Decoder latency is per auto-regressive step (~200 steps per 30 s chunk).",
        "- INT8 models quantized with `onnxruntime.quantization.quantize_dynamic`.",
        "- Compile jobs run once; profile jobs re-runnable from AI Hub dashboard.",
    ]
    return "\n".join(lines)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(description="AI Hub on-device benchmark for all pipeline models.")
    p.add_argument("--device",  default=DEVICE_DEFAULT)
    p.add_argument("--models",  nargs="*", choices=list(MODEL_SPECS.keys()),
                   help="Subset of models to profile (default: all)")
    args = p.parse_args()

    hub  = _check_hub()
    keys = args.models or list(MODEL_SPECS.keys())
    rows: list[dict] = []

    print(f"\nBenchmarking {len(keys)} model(s) on {args.device}\n")
    for key in keys:
        spec = MODEL_SPECS[key]
        print(f"→ {spec['label']}")
        row = _compile_and_profile(hub, key, spec, args.device)
        rows.append(row)
        _print_row(row)

    md = _render_markdown(rows, args.device)
    RESULTS_MD.parent.mkdir(exist_ok=True)
    RESULTS_MD.write_text(md, encoding="utf-8")
    print(f"\n✅  Results written to {RESULTS_MD}\n")
    print(md)


def _print_row(r: dict) -> None:
    if r.get("skipped"):
        print(f"  ⚠ Skipped — {r['reason']}")
    elif r.get("error"):
        print(f"  ✗ Error   — {r['error']}")
    else:
        print(f"  ✓ {r['latency_ms']} ms | {r['peak_mem_mb']} MB | {r['compute_unit']}")


if __name__ == "__main__":
    main()
