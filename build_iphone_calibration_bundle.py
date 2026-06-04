#!/usr/bin/env python3
"""Build iPhone calibration bundle from CameraColorProject (Lab 3 style)."""

from __future__ import annotations

import argparse
from pathlib import Path

from iphone_camera_calibration import (
    DEFAULT_CAMERA_COLOR_ROOT,
    build_calibration_bundle,
    plot_lab3_style_figures,
)

ROOT = Path(__file__).resolve().parent


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--camera-color-root",
        type=Path,
        default=DEFAULT_CAMERA_COLOR_ROOT,
        help="CameraColorProject root (monochromator + flash folders).",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "calibration" / "iphone17pro_camera_color",
    )
    ap.add_argument(
        "--dng-start-index",
        type=int,
        default=0,
        help="First DNG in sorted list for 31×3 monochromator block (default 0 = IMG_1302…).",
    )
    ap.add_argument("--repeats-per-wl", type=int, default=3)
    ap.add_argument("--raw-half-size", type=int, default=1)
    ap.add_argument("--raw-camera-wb", action="store_true")
    ap.add_argument("--no-plots", action="store_true")
    ap.add_argument("--dpi", type=int, default=160)
    args = ap.parse_args()

    bundle = build_calibration_bundle(
        args.camera_color_root,
        dng_start_index=args.dng_start_index,
        repeats_per_wl=args.repeats_per_wl,
        raw_half_size=args.raw_half_size,
        raw_camera_wb=args.raw_camera_wb,
    )
    path = bundle.save(args.out_dir)
    print(f"Wrote {path}")
    print(f"  flash CCT ≈ {bundle.flash_cct_k:.1f} K  Duv ≈ {bundle.flash_duv:.4f}")
    print(f"  flash RGB (linear) = {bundle.flash_rgb_linear}")
    if not args.no_plots:
        plot_lab3_style_figures(bundle, args.out_dir, dpi=args.dpi)
        print(f"Wrote lab3_*.png and iphone_flash_spd.png in {args.out_dir}")


if __name__ == "__main__":
    main()
