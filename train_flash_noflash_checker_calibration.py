#!/usr/bin/env python3
"""
Train **offline** calibration from flash/no-flash pairs that include a ColorChecker.

**Inference stays chart-free** (``flash_no_flash_skin_lab.py`` does not require a checker in frame).
This script only builds artifacts used at run time:

  - ``camera_rgb_to_xyz.npy`` — linear camera RGB → XYZ (D65), from MCC24 patches on **no-flash**
  - ``lu_sharpening_M.npy`` — optional 3×3 Lu spectral-sharpening matrix (neutral-patch log-chroma LS)
  - ``iphone_calibration_bundle.json`` — merges monochromator/MK350 bundle if present

Matches the original plan (Lu Phase 0 + PR-250/MCC evaluation bridge): train with checker,
deploy illuminant + reflectance without checker in the capture.

Example::

    python3 train_flash_noflash_checker_calibration.py \\
        --data-root "/path/to/RAW Dataset" \\
        --monochromator-bundle ./calibration/iphone17pro_camera_color \\
        --out-dir ./calibration/iphone17pro_trained
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parent

try:
    import cv2
except ImportError as e:
    raise SystemExit("pip install opencv-python") from e

from mcc24_canonical_d65 import WHITE_PATCH_INDEX, load_canonical_xyz_d65
from mcc24_classic import decode_all_patches_srgb_255
from srgb_eotf import srgb_255_to_linear

import physio_skin_lab_raw_pr250 as pr250
from exposure_anchor import aggregate_exposure_anchors, save_exposure_anchors

_RAW_EXTS = {".dng", ".DNG", ".cr2", ".CR2", ".jpg", ".jpeg", ".JPG", ".JPEG"}


def _discover_pairs(data_root: Path) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for part_dir in sorted(data_root.glob("Participant*")):
        for trial_dir in sorted(part_dir.glob("Trial*")):
            nf = fl = None
            for p in trial_dir.iterdir():
                if not p.is_file():
                    continue
                name = p.name.lower()
                if "noflash" in name and p.suffix in _RAW_EXTS:
                    nf = p
                elif "flash" in name and "noflash" not in name and p.suffix in _RAW_EXTS:
                    fl = p
            if nf and fl:
                m = re.search(r"(\d+)", part_dir.name)
                pnum = m.group(1) if m else part_dir.name
                tm = re.search(r"(\d+)", trial_dir.name)
                tnum = tm.group(1) if tm else trial_dir.name
                rows.append(
                    {
                        "subject_id": f"P{pnum}_T{tnum}",
                        "path_noflash": str(nf),
                        "path_flash": str(fl),
                    }
                )
    return rows


def _read_linear_rgb(path: Path, *, half_size: int, camera_wb: bool) -> np.ndarray:
    if path.suffix.lower() in (".dng", ".cr2", ".cr3", ".nef", ".arw"):
        rgb_lin = pr250.read_raw_linear_rgb(
            path, half_size=half_size, use_camera_wb=camera_wb
        )
        scale = float(np.percentile(rgb_lin, 99.5))
        return np.clip(rgb_lin / max(scale, 1e-6), 0.0, 1.0).astype(np.float64)
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"imread failed: {path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float64) / 255.0
    return srgb_255_to_linear(rgb)


def _chart_patches_camera_linear(
    noflash_linear: np.ndarray,
) -> Optional[np.ndarray]:
    """24×3 linear **camera** RGB patch means (mcc quads on stretched preview), or None."""
    preview_bgr = pr250.linear_rgb_to_preview_bgr(noflash_linear)
    got = pr250.patch_linear_rgb_24(noflash_linear, preview_bgr, use_median=True)
    if got is None:
        return None
    patches, _quads = got
    return patches


def fit_rgb_to_xyz_lstsq(patches_lin: np.ndarray, xyz_ref: np.ndarray) -> np.ndarray:
    """Least squares: xyz_ref ≈ patches_lin @ M.T  → return M (3×3) for rgb @ M.T."""
    A = np.asarray(patches_lin, dtype=np.float64)
    B = np.asarray(xyz_ref, dtype=np.float64)
    M_T, _, _, _ = np.linalg.lstsq(A, B, rcond=None)
    return M_T.T


def fit_lu_sharpening_matrix(neutral_log_chroma: np.ndarray) -> np.ndarray:
    """
    Simple Lu-style sharpening: find 3×3 ``M`` so neutral log-chroma vectors cluster.

    ``neutral_log_chroma`` shape (N, 3) — log RGB minus mean log per sample.
    """
    X = np.asarray(neutral_log_chroma, dtype=np.float64)
    if X.shape[0] < 6:
        return np.eye(3, dtype=np.float64)
    cov = X.T @ X / max(X.shape[0], 1)
    w, v = np.linalg.eigh(cov)
    # Whitening-like: emphasize directions with spread
    M = v @ np.diag(1.0 / np.sqrt(np.maximum(w, 1e-6))) @ v.T
    M = M / max(float(np.linalg.norm(M)), 1e-6) * np.sqrt(3.0)
    if np.linalg.det(M) < 0:
        M[:, 0] *= -1.0
    return M.astype(np.float64)


def _log_chroma_rows(rgb: np.ndarray) -> np.ndarray:
    d = np.log(np.maximum(rgb, 1e-6))
    return d - d.mean(axis=1, keepdims=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, default=ROOT / "calibration" / "iphone17pro_trained")
    ap.add_argument(
        "--monochromator-bundle",
        type=Path,
        default=ROOT / "calibration" / "iphone17pro_camera_color",
        help="Optional existing MK350/monochromator bundle to merge (flash CCT/RGB).",
    )
    ap.add_argument("--raw-half-size", type=int, default=1)
    ap.add_argument("--raw-camera-wb", action="store_true")
    ap.add_argument("--max-align-width", type=int, default=1600)
    ap.add_argument(
        "--huber-matrix",
        action="store_true",
        help=(
            "Per-trial Huber IRLS RGB→XYZ with skin/neutral row weights; "
            "median 3×3 across trials (experimental)."
        ),
    )
    ap.add_argument(
        "--huber-matrix-stacked",
        action="store_true",
        help="Huber IRLS on all stacked MCC patches (24 × n_trials rows).",
    )
    ap.add_argument(
        "--matrix-affine",
        action="store_true",
        help="Affine RGB→XYZ (4×3); saves camera_rgb_to_xyz_affine.npy for inference.",
    )
    ap.add_argument(
        "--issa-skin-rows",
        type=str,
        default="",
        help="Comma-separated ISSA prior names as extra training rows (e.g. issa_median_caucasian,issa_median_african).",
    )
    ap.add_argument(
        "--anchor-weight",
        type=float,
        default=2.5,
        help="Row weight for PR-250 skin/neutral anchor patches in matrix lstsq.",
    )
    ap.add_argument(
        "--skin-patch-weight",
        type=float,
        default=1.0,
        help="Extra row weight for MCC patches 0–1 (skin-like) in matrix lstsq.",
    )
    ap.add_argument(
        "--weighted-matrix",
        action="store_true",
        help="Weighted lstsq on stacked MCC (+ISSA rows if set).",
    )
    ap.add_argument(
        "--issa-row-weight",
        type=float,
        default=3.0,
        help="Row weight for each ISSA synthetic skin row.",
    )
    args = ap.parse_args()

    rows = _discover_pairs(args.data_root)
    if not rows:
        raise SystemExit(f"No Flash/NoFlash pairs under {args.data_root}")

    from flash_no_flash_skin_lab import (
        _resize_linear_max_width,
        align_flash_to_noflash_linear,
    )

    # Canonical XYZ is Y≈0–100; camera linear u01 pairs with Y≈0–1 (D65 white ≈ 1).
    xyz_ref = load_canonical_xyz_d65() / 100.0
    xyz_white = xyz_ref[WHITE_PATCH_INDEX]
    all_patches: List[np.ndarray] = []
    neutral_logs: List[np.ndarray] = []
    trial_log: List[Dict[str, Any]] = []

    for row in rows:
        nf_path = Path(row["path_noflash"])
        fl_path = Path(row["path_flash"])
        nf_lin = _read_linear_rgb(
            nf_path, half_size=args.raw_half_size, camera_wb=args.raw_camera_wb
        )
        fl_lin = _read_linear_rgb(
            fl_path, half_size=args.raw_half_size, camera_wb=args.raw_camera_wb
        )
        nf_work = _resize_linear_max_width(nf_lin, args.max_align_width)
        fl_work = _resize_linear_max_width(fl_lin, args.max_align_width)
        align = align_flash_to_noflash_linear(nf_work, fl_work, motion_ecc="euclidean")
        patches = _chart_patches_camera_linear(align.noflash_linear)
        if patches is None:
            print(f"skip {row['subject_id']}: no ColorChecker on no-flash", file=sys.stderr)
            continue
        pw = patches[WHITE_PATCH_INDEX]
        y_cam = 0.2126 * pw[0] + 0.7152 * pw[1] + 0.0722 * pw[2]
        white_scale = float(xyz_white[1] / max(y_cam, 1e-12))
        # Fit matrix on unscaled camera-linear patches; white_scale is for inference exposure anchor only.
        all_patches.append(patches)
        gray_idx = list(range(18, 24))  # white + neutrals + black
        neutral_logs.append(_log_chroma_rows(patches[gray_idx]))
        trial_log.append(
            {
                "subject_id": row["subject_id"],
                "n_patches": 24,
                "white_patch_scale": white_scale,
            }
        )

    if not all_patches:
        raise SystemExit("No trials with detectable ColorChecker — cannot train.")

    stacked = np.concatenate(all_patches, axis=0)
    ref_stacked = np.tile(xyz_ref, (len(all_patches), 1))
    n_mcc_rows = int(stacked.shape[0])
    row_w_24 = pr250.build_patch_lstsq_row_weights(
        anchor_weight=float(args.anchor_weight),
        skin_weight=float(args.skin_patch_weight),
    )

    issa_names = [s.strip() for s in args.issa_skin_rows.split(",") if s.strip()]
    extra_rgb = np.zeros((0, 3), dtype=np.float64)
    if issa_names:
        mono_json = args.monochromator_bundle / "iphone_calibration_bundle.json"
        if not mono_json.is_file():
            raise SystemExit(f"--issa-skin-rows requires {mono_json}")
        with mono_json.open(encoding="utf-8") as f:
            mono = json.load(f)
        from flash_noflash_spectral import issa_skin_calibration_rows

        s_arr = np.asarray(mono["spectral_sensitivity_rgb"], dtype=np.float64)
        wl_arr = np.asarray(mono["wavelengths_nm"], dtype=np.float64)
        extra_rgb, extra_xyz = issa_skin_calibration_rows(s_arr, wl_arr, issa_names)
        stacked = np.concatenate([stacked, extra_rgb], axis=0)
        ref_stacked = np.concatenate([ref_stacked, extra_xyz], axis=0)
        print(f"ISSA skin calibration rows: +{extra_rgb.shape[0]} from {issa_names}", file=sys.stderr)

    mcc_row_w = np.tile(row_w_24, len(all_patches)) if row_w_24 is not None else None
    if extra_rgb.shape[0] > 0:
        issa_w = np.full(extra_rgb.shape[0], float(args.issa_row_weight), dtype=np.float64)
        row_w = (
            np.concatenate([mcc_row_w, issa_w])
            if mcc_row_w is not None
            else issa_w
        )
    else:
        row_w = mcc_row_w

    if args.huber_matrix_stacked:
        M_cam, _, _ = pr250.fit_rgb_to_xyz_lstsq_huber_irls(
            stacked,
            ref_stacked,
            with_intercept=bool(args.matrix_affine),
            row_weights=row_w,
        )
        matrix_method = "stacked huber_irls MCC24→D65 XYZ (+ISSA rows if set)"
    elif args.huber_matrix:
        m_list: List[np.ndarray] = []
        for patches in all_patches:
            M_t, _, _ = pr250.fit_rgb_to_xyz_lstsq_huber_irls(
                patches,
                xyz_ref,
                with_intercept=False,
                row_weights=row_w_24,
            )
            m_list.append(M_t)
        M_cam = np.median(np.stack(m_list, axis=0), axis=0)
        matrix_method = (
            "per-trial huber_irls MCC24→D65 XYZ, median 3×3 "
            "(skin/neutral row weights)"
        )
    elif args.matrix_affine:
        M_affine = pr250.fit_rgb_to_xyz_lstsq(
            stacked, ref_stacked, with_intercept=True, row_weights=row_w
        )
        M_cam = M_affine
        matrix_method = "affine lstsq [R,G,B,1]→XYZ (stacked; use camera_rgb_to_xyz_affine.npy)"
    elif args.weighted_matrix:
        M_cam = pr250.fit_rgb_to_xyz_lstsq(
            stacked, ref_stacked, with_intercept=False, row_weights=row_w
        )
        matrix_method = "weighted lstsq MCC24→D65 XYZ (stacked)"
    else:
        M_cam = fit_rgb_to_xyz_lstsq(stacked, ref_stacked)
        matrix_method = "lstsq MCC24→D65 XYZ (stacked trials, unweighted)"
    M_lu = fit_lu_sharpening_matrix(np.concatenate(neutral_logs, axis=0))
    exposure_by_participant = aggregate_exposure_anchors(trial_log)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    M_bundle = np.asarray(M_cam, dtype=np.float64)
    if args.matrix_affine and M_bundle.shape == (4, 3):
        np.save(args.out_dir / "camera_rgb_to_xyz_affine.npy", M_bundle)
        M_lin = fit_rgb_to_xyz_lstsq(stacked, ref_stacked)
        np.save(args.out_dir / "camera_rgb_to_xyz.npy", M_lin)
        M_bundle = M_lin
    else:
        np.save(args.out_dir / "camera_rgb_to_xyz.npy", M_bundle)
    np.save(args.out_dir / "lu_sharpening_M.npy", M_lu)

    bundle: Dict[str, Any] = {
        "source_root": str(args.data_root),
        "device_label": "iPhone trained (MCC24 no-flash + optional monochromator flash)",
        "lab3_method": f"train: {matrix_method} on aligned no-flash; infer: chart-free",
        "matrix_fit": matrix_method,
        "huber_matrix": bool(args.huber_matrix),
        "huber_matrix_stacked": bool(args.huber_matrix_stacked),
        "weighted_matrix": bool(args.weighted_matrix),
        "matrix_affine": bool(args.matrix_affine),
        "issa_skin_rows": issa_names if issa_names else None,
        "patch_row_weights": (
            {"anchor_weight": args.anchor_weight, "skin_patch_weight": args.skin_patch_weight}
            if row_w_24 is not None
            else None
        ),
        "notes": (
            "Trained offline with ColorChecker visible. "
            "flash_no_flash_skin_lab.py does not detect chart at inference."
        ),
        "camera_rgb_to_xyz": M_bundle.tolist(),
        "camera_rgb_to_xyz_affine": (
            np.load(args.out_dir / "camera_rgb_to_xyz_affine.npy").tolist()
            if (args.out_dir / "camera_rgb_to_xyz_affine.npy").is_file()
            else None
        ),
        "lu_sharpening_M": M_lu.tolist(),
        "n_training_trials": len(all_patches),
        "training_trials": trial_log,
        "exposure_anchor_by_participant": exposure_by_participant,
    }

    mono_json = args.monochromator_bundle / "iphone_calibration_bundle.json"
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
            "monochromator_spd_scalar",
        ):
            if k in mono:
                bundle[k] = mono[k]
        bundle["monochromator_bundle_merged"] = str(args.monochromator_bundle)

    with (args.out_dir / "iphone_calibration_bundle.json").open("w", encoding="utf-8") as f:
        json.dump(bundle, f, indent=2)
    anchor_path = save_exposure_anchors(args.out_dir, exposure_by_participant, trial_log)

    print(f"Trained on {len(all_patches)} trial(s) with ColorChecker")
    print(f"Exposure anchors (median white-patch scale): {exposure_by_participant}")
    print(f"Wrote {anchor_path}")
    print(f"Wrote {args.out_dir}/camera_rgb_to_xyz.npy")
    print(f"Wrote {args.out_dir}/lu_sharpening_M.npy")
    print(f"Wrote {args.out_dir}/iphone_calibration_bundle.json")
    print("\nRun inference (no checker in frame):")
    print(
        f"  IPHONE_CALIBRATION={args.out_dir} \\\n"
        f"  python3 flash_no_flash_skin_lab.py --iphone-calibration {args.out_dir} ..."
    )
    print(
        f"  python3 flash_no_flash_skin_lab.py --iphone-calibration {args.out_dir} "
        f"--lu-sharpening-matrix {args.out_dir}/lu_sharpening_M.npy ..."
    )


if __name__ == "__main__":
    main()
