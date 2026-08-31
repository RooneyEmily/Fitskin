#!/usr/bin/env python3
"""Pack calibration + pipeline files for the ring-light best-stack Colab notebook."""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

INCLUDE = [
    "pipeline/__init__.py",
    "pipeline/skin_roi.py",
    "pipeline/d65_fairface7_roi.py",
    "pipeline/illuminant_estimation.py",
    "pipeline/post_corrections.py",
    "vendor/flash_align/__init__.py",
    "vendor/flash_align/align_pair.py",
    "vendor/flash_align/color_linear.py",
    "vendor/flash_align/lu2006_ambient.py",
    "calibration/tier3_affine/camera_rgb_to_xyz_affine.npy",
    "calibration/tier3_affine/camera_rgb_to_xyz.npy",
    "calibration/tier3_affine/lu_sharpening_M.npy",
    "calibration/tier3_affine/exposure_anchor_by_participant.json",
    "calibration/tier3_affine/iphone_calibration_bundle.json",
    "calibration/multi_illuminant_lab_affine/multi_illuminant_lab_affine.json",
    "models/__init__.py",
    "models/fairface_race.py",
    "scripts/evaluate_pansor20_chartfree_d65.py",
    "scripts/run_d65_fairface7_roi.py",
    "flash_no_flash_spectral.py",
    "delta_e_2000.py",
]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--out",
        type=Path,
        default=ROOT / "colab_assets" / "ringlight_best_stack.zip",
    )
    args = ap.parse_args()
    out = Path(args.out).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    missing = [rel for rel in INCLUDE if not (ROOT / rel).is_file()]
    if missing:
        raise SystemExit("Missing files:\n  " + "\n  ".join(missing))

    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for rel in INCLUDE:
            zf.write(ROOT / rel, rel)
    print(f"Wrote {out}  ({len(INCLUDE)} files)")


if __name__ == "__main__":
    main()
