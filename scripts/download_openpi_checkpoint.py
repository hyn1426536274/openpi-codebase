"""Download an official OpenPI checkpoint with OPENPI_DATA_HOME.

Examples:
  python scripts/download_openpi_checkpoint.py pi05_libero --output_dir /workspace/data/pi_official_models
  python scripts/download_openpi_checkpoint.py gs://openpi-assets/checkpoints/pi05_droid --output_dir /workspace/data/pi_official_models

The checkpoint will be placed under:
  $OPENPI_DATA_HOME/openpi-assets/checkpoints/<checkpoint_name>
"""

from __future__ import annotations

import argparse
import logging
import os
import pathlib
import sys

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
_SRC_DIR = _REPO_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))


OFFICIAL_CHECKPOINTS: dict[str, str] = {
    "pi0_base": "gs://openpi-assets/checkpoints/pi0_base",
    "pi0_fast_base": "gs://openpi-assets/checkpoints/pi0_fast_base",
    "pi05_base": "gs://openpi-assets/checkpoints/pi05_base",
    "pi0_fast_droid": "gs://openpi-assets/checkpoints/pi0_fast_droid",
    "pi0_droid": "gs://openpi-assets/checkpoints/pi0_droid",
    "pi0_aloha_towel": "gs://openpi-assets/checkpoints/pi0_aloha_towel",
    "pi0_aloha_tupperware": "gs://openpi-assets/checkpoints/pi0_aloha_tupperware",
    "pi0_aloha_pen_uncap": "gs://openpi-assets/checkpoints/pi0_aloha_pen_uncap",
    "pi05_libero": "gs://openpi-assets/checkpoints/pi05_libero",
    "pi05_droid": "gs://openpi-assets/checkpoints/pi05_droid",
}


def _resolve_checkpoint(checkpoint: str) -> str:
    if checkpoint.startswith("gs://"):
        return checkpoint.rstrip("/")
    if checkpoint in OFFICIAL_CHECKPOINTS:
        return OFFICIAL_CHECKPOINTS[checkpoint]
    choices = ", ".join(sorted(OFFICIAL_CHECKPOINTS))
    raise ValueError(f"Unknown checkpoint '{checkpoint}'. Supported names: {choices}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "checkpoint",
        help="Official checkpoint name, e.g. pi05_libero, or a full gs:// checkpoint path.",
    )
    parser.add_argument(
        "--output_dir",
        "--output-dir",
        required=True,
        type=pathlib.Path,
        help="Directory to use as OPENPI_DATA_HOME.",
    )
    parser.add_argument(
        "--force_download",
        "--force-download",
        default=False,
        action="store_true",
        help="Force re-download from the remote checkpoint even if the OpenPI cache exists.",
    )
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    checkpoint_url = _resolve_checkpoint(args.checkpoint)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Official OpenPI behavior: maybe_download() caches under OPENPI_DATA_HOME.
    # This must be set before importing openpi.shared.download.
    os.environ["OPENPI_DATA_HOME"] = str(output_dir)

    from openpi.shared import download as _download

    logging.info("Resolving checkpoint: %s", checkpoint_url)
    logging.info("Using OPENPI_DATA_HOME: %s", output_dir)
    local_cache_path = _download.maybe_download(checkpoint_url, force_download=args.force_download)
    logging.info("Checkpoint ready at: %s", local_cache_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(parse_args())
