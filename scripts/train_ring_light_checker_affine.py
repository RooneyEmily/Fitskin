#!/usr/bin/env python3
"""Train MCC24-supervised RGB→XYZ affines for ring D65 and F12 torch captures.

The torch zip DNGs include an in-frame ColorChecker (face + chart). Patches are
extracted from **flash/no-flash reflectance** R₀ = √(A₀·B₀) to match the
chart-free cheek inference path.

Example::

  python3 scripts/train_ring_light_checker_affine.py
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

BUILD = ROOT / "scripts" / "build_ring_light_cc_manifest.py"
MONO = ROOT / "calibration" / "iphone17pro_camera_color"

from exposure_anchor import aggregate_exposure_anchors  # noqa: E402
from flash_no_flash_skin_lab import (  # noqa: E402
    _resize_linear_max_width,
    align_flash_to_noflash_linear,
)
from mcc24_canonical_d65 import WHITE_PATCH_INDEX, load_canonical_xyz_d65  # noqa: E402
import physio_skin_lab_raw_pr250 as pr250  # noqa: E402
from train_flash_noflash_checker_calibration import (  # noqa: E402
    _chart_patches_camera_linear,
    _log_chroma_rows,
    _read_linear_rgb,
    fit_lu_sharpening_matrix,
)


def _run(cmd: list[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(ROOT), check=True)


def _load_manifest(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def reflectance_patches_from_pair(
    nf_path: Path,
    fl_path: Path,
    *,
    half_size: int,
    max_align_width: int,
) -> np.ndarray | None:
    nf_lin = _read_linear_rgb(nf_path, half_size=half_size, camera_wb=False)
    fl_lin = _read_linear_rgb(fl_path, half_size=half_size, camera_wb=False)
    nf_work = _resize_linear_max_width(nf_lin, max_align_width)
    fl_work = _resize_linear_max_width(fl_lin, max_align_width)
    align = align_flash_to_noflash_linear(nf_work, fl_work, motion_ecc="euclidean")
    r_lin = np.sqrt(
        np.maximum(align.noflash_linear, 0) * np.maximum(align.flash_aligned_linear, 0) + 1e-8
    )
    preview = pr250.linear_rgb_to_preview_bgr(r_lin)
    got = pr250.patch_linear_rgb_24(r_lin, preview, use_median=True)
    if got is None:
        return None
    patches, _ = got
    return patches


def train_reflectance_cc_affine(
    rows: List[Dict[str, str]],
    out_dir: Path,
    *,
    half_size: int = 1,
    max_align_width: int = 1600,
    monochromator_bundle: Path = MONO,
) -> Dict[str, Any]:
    xyz_ref = load_canonical_xyz_d65() / 100.0
    xyz_white = xyz_ref[WHITE_PATCH_INDEX]
    all_patches: List[np.ndarray] = []
    neutral_logs: List[np.ndarray] = []
    trial_log: List[Dict[str, Any]] = []

    for row in rows:
        patches = reflectance_patches_from_pair(
            Path(row["path_noflash"]),
            Path(row["path_flash"]),
            half_size=half_size,
            max_align_width=max_align_width,
        )
        if patches is None:
            print(f"skip {row['subject_id']}: no CC on reflectance", file=sys.stderr)
            continue
        pw = patches[WHITE_PATCH_INDEX]
        y_cam = 0.2126 * pw[0] + 0.7152 * pw[1] + 0.0722 * pw[2]
        white_scale = float(xyz_white[1] / max(y_cam, 1e-12))
        all_patches.append(patches)
        gray_idx = list(range(18, 24))
        neutral_logs.append(_log_chroma_rows(patches[gray_idx]))
        trial_log.append(
            {
                "subject_id": row["subject_id"],
                "person": row.get("person", ""),
                "illuminant": row.get("illuminant", ""),
                "n_patches": 24,
                "white_patch_scale": white_scale,
            }
        )

    if not all_patches:
        raise SystemExit("No reflectance CC patches — cannot train")

    stacked = np.concatenate(all_patches, axis=0)
    ref_stacked = np.tile(xyz_ref, (len(all_patches), 1))
    row_w_24 = pr250.build_patch_lstsq_row_weights(anchor_weight=2.5, skin_weight=1.0)
    row_w = np.tile(row_w_24, len(all_patches))
    M_affine = pr250.fit_rgb_to_xyz_lstsq(
        stacked, ref_stacked, with_intercept=True, row_weights=row_w
    )
    M_lu = fit_lu_sharpening_matrix(np.concatenate(neutral_logs, axis=0))
    exposure_by_participant = aggregate_exposure_anchors(trial_log)

    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "camera_rgb_to_xyz_affine.npy", M_affine)
    M_lin = pr250.fit_rgb_to_xyz_lstsq(stacked, ref_stacked, with_intercept=False, row_weights=row_w)
    np.save(out_dir / "camera_rgb_to_xyz.npy", M_lin)
    np.save(out_dir / "lu_sharpening_M.npy", M_lu)

    matrix_method = "affine lstsq R0 reflectance MCC24→D65 XYZ (ring torch CC in frame)"
    bundle: Dict[str, Any] = {
        "device_label": "iPhone ring-light torch CC (R0 patches)",
        "matrix_fit": matrix_method,
        "matrix_affine": True,
        "patch_domain": "reflectance_r0",
        "camera_rgb_to_xyz_affine": M_affine.tolist(),
        "camera_rgb_to_xyz": M_lin.tolist(),
        "lu_sharpening_M": M_lu.tolist(),
        "n_training_trials": len(all_patches),
        "training_trials": trial_log,
        "exposure_anchor_by_participant": exposure_by_participant,
    }
    mono_json = monochromator_bundle / "iphone_calibration_bundle.json"
    if mono_json.is_file():
        with mono_json.open(encoding="utf-8") as f:
            mono = json.load(f)
        for k in (
            "flash_cct_k",
            "flash_duv",
            "flash_rgb_linear",
            "flash_xyz",
            "flash_spd_wl_nm",
            "flash_spd_power",
            "spectral_sensitivity_rgb",
            "wavelengths_nm",
        ):
            if k in mono:
                bundle[k] = mono[k]

    (out_dir / "iphone_calibration_bundle.json").write_text(
        json.dumps(bundle, indent=2) + "\n", encoding="utf-8"
    )
    (out_dir / "exposure_anchor_by_participant.json").write_text(
        json.dumps(exposure_by_participant, indent=2) + "\n", encoding="utf-8"
    )
    return bundle


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", type=Path, default=None)
    ap.add_argument("--manifest-dir", type=Path, default=ROOT / "data" / "ring_light")
    ap.add_argument("--out-warm", type=Path, default=ROOT / "calibration" / "tier3_affine_warm_cc")
    ap.add_argument("--out-d65", type=Path, default=ROOT / "calibration" / "tier3_affine_d65_ring_cc")
    ap.add_argument("--skip-manifest", action="store_true")
    args = ap.parse_args()

    manifest_dir = Path(args.manifest_dir).expanduser().resolve()
    if not args.skip_manifest:
        cmd = [sys.executable, str(BUILD), "--out-dir", str(manifest_dir)]
        if args.data_root is not None:
            cmd.extend(["--data-root", str(args.data_root.expanduser().resolve())])
        _run(cmd)

    summary_path = manifest_dir / "manifest_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    print(json.dumps(summary, indent=2))

    mono = MONO if MONO.is_dir() else ROOT / "calibration" / "tier3_affine"
    meta: Dict[str, Any] = {"manifest_summary": summary, "bundles": {}}
    for manifest_name, out_dir, label in (
        ("manifest_ring_cc_f12.csv", args.out_warm, "F12 warm ring CC R0"),
        ("manifest_ring_cc_d65.csv", args.out_d65, "D65 ring CC R0"),
    ):
        rows = _load_manifest(manifest_dir / manifest_name)
        bundle = train_reflectance_cc_affine(
            rows, Path(out_dir).expanduser().resolve(), monochromator_bundle=mono
        )
        meta["bundles"][label] = {
            "out_dir": str(out_dir),
            "n_training_trials": bundle["n_training_trials"],
        }
        print(f"Trained {label} → {out_dir}  (n={bundle['n_training_trials']})")

    routed = {
        "method": "mcc24_in_frame_ring_torch_r0",
        **meta,
        "paths": {
            "default_cool": str(ROOT / "calibration" / "tier3_affine"),
            "warm_f12": str(args.out_warm),
            "d65_ring": str(args.out_d65),
        },
        "note": "MCC24 in torch frame; affine fit on R0=sqrt(nf*flash) patches.",
    }
    routed_path = ROOT / "calibration" / "tier3_affine_illuminant_routed.json"
    routed_path.write_text(json.dumps(routed, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {routed_path}")


if __name__ == "__main__":
    main()
