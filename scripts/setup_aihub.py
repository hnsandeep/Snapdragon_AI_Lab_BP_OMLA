"""
scripts/setup_aihub.py
----------------------
Configures Qualcomm AI Hub with your API token and verifies connectivity.

Usage:
    python scripts/setup_aihub.py --token YOUR_TOKEN_HERE
    # or set QAI_HUB_API_TOKEN in .env and run without --token

What it does:
  1. Writes the token via `qai-hub configure` (stores in ~/.qai_hub/client.ini).
  2. Calls qai_hub.get_devices() to confirm the token is valid and lists
     all hosted Snapdragon devices available for jobs.
"""

import argparse
import subprocess
import sys
import os
from pathlib import Path

# Allow running from the project root without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env")


def configure_hub(token: str) -> None:
    """Write the API token to ~/.qai_hub/client.ini via the CLI helper."""
    print("[1/3] Configuring AI Hub API token …")
    result = subprocess.run(
        [sys.executable, "-m", "qai_hub", "configure", "--api_token", token],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print("  ERROR:", result.stderr.strip())
        sys.exit(1)
    print("  ✓  Token written to ~/.qai_hub/client.ini")


def verify_connectivity() -> None:
    """List available devices — confirms the token works end-to-end."""
    print("[2/3] Verifying connectivity (fetching device list) …")
    try:
        import qai_hub as hub  # noqa: PLC0415
    except ImportError:
        print("  ERROR: qai-hub is not installed.")
        print("         Run:  pip install qai-hub==0.55.0")
        sys.exit(1)

    try:
        devices = hub.get_devices()
    except Exception as exc:  # noqa: BLE001
        print(f"  ERROR: Could not reach AI Hub — {exc}")
        sys.exit(1)

    print(f"  ✓  Connected. {len(devices)} device(s) available.")


def find_snapdragon_devices() -> None:
    """Print devices that match our target chipset for quick reference."""
    print("[3/3] Searching for Snapdragon X Elite / X2 devices …")
    import qai_hub as hub  # noqa: PLC0415

    keywords = ("snapdragon x elite", "snapdragon x2", "crd")
    matches = [d for d in hub.get_devices()
               if any(k in d.name.lower() for k in keywords)]

    if matches:
        for dev in matches:
            print(f"  ✓  Found device: {dev.name!r}")
    else:
        print("  ⚠  No Snapdragon X Elite / X2 devices found in your account's")
        print("     device pool. Check device availability at https://aihub.qualcomm.com")

    print("\nSetup complete. You can now run:  python scripts/verify_hub_job.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Configure and verify Qualcomm AI Hub access.")
    parser.add_argument(
        "--token",
        default=os.getenv("QAI_HUB_API_TOKEN", ""),
        help="Your AI Hub API token (or set QAI_HUB_API_TOKEN in .env)",
    )
    args = parser.parse_args()

    if not args.token:
        print("ERROR: No API token provided.")
        print("  Option 1: python scripts/setup_aihub.py --token YOUR_TOKEN")
        print("  Option 2: Add QAI_HUB_API_TOKEN=... to your .env file")
        sys.exit(1)

    configure_hub(args.token)
    verify_connectivity()
    find_snapdragon_devices()
