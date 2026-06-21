#!/usr/bin/env python3
"""
Evaluate Sephora bag white stripes as a chart-free exposure anchor.

This runs the same flash/no-flash skin extraction used in Phase 4, but replaces
or compares the reflectance exposure scale with an in-frame bag white reference:

    scale_bag = NIX_white_Y / camera_white_stripe_Y

The newer XYZ modes compute the bag white after the calibrated RGB->XYZ
transform, then either match Y only or choose the scalar exposure scale that
best matches the full NIX white XYZ.

Outputs one row per trial and anchor mode, with FitSkin cheek Delta E 2000.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
import tempfile
from pathlib import Path
from statistics import median
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import flash_no_flash_skin_lab as fnf
import physio_skin_lab_raw_pr250 as pr250
from delta_e_2000 import delta_e_2000
from exposure_anchor import load_exposure_anchors, participant_key
from flash_no_flash_skin_lab import (
    AlignResult,
    _resize_linear_max_width,
    align_flash_to_noflash_linear,
    estimate_exposure_scale_masked,
    estimate_reflectance_linear,
    load_fitskin_cheek_lookup,
    load_lu_sharpening_matrix,
    mean_lab_reflectance_linear,
    process_flash_pair,
    raw_to_linear_u01,
    skin_mask_from_bgr,
)
from sephora_bag_reference import (
    exposure_scale_from_bag_white,
    load_nix_bag_reference,
    segment_sephora_bag,
)


DEFAULT_DATA_ROOT = Path(
    "/media/mabl-main/Data/Bag image pairs/flash_noflash images/sephorabag_target"
)
DEFAULT_CALIBRATION = ROOT / "calibration" / "tier3_affine"


def _discover_pairs(data_root: Path) -> List[Dict[str, Any]]:
    pairs: List[Dict[str, Any]] = []
    for nf in sorted(data_root.rglob("*_NoFlash.DNG")):
        fl_list = sorted(nf.parent.glob("*_Flash.DNG"))
        if not fl_list:
            continue
        nf_num = int("".join(c for c in nf.stem if c.isdigit()) or "0")
        fl = min(
            fl_list,
            key=lambda p: abs(int("".join(c for c in p.stem if c.isdigit()) or "0") - nf_num),
        )
        trial_name = str(nf.parent.relative_to(data_root)).replace(" ", "_").replace("/", "_")
        pairs.append({"trial_id": trial_name, "noflash": nf, "flash": fl})
    return pairs


def _participant_trial_from_bag_id(
    trial_id: str,
    *,
    emily_participant: str,
    liki_participant: str,
) -> Tuple[str, str, str]:
    name, _, trial_s = trial_id.partition("_Trial_")
    if not trial_s:
        raise ValueError(f"Unexpected bag trial id: {trial_id}")
    participant = emily_participant if name.lower() == "emily" else liki_participant
    subject = f"P{participant.split()[-1]}_T{trial_s}"
    return subject, participant, trial_s


def _read_dng_u01(path: Path, half_size: int) -> np.ndarray:
    try:
        return raw_to_linear_u01(path, half_size=half_size, use_camera_wb=False)
    except OSError:
        tmp = Path(tempfile.gettempdir()) / f"sephora_eval_{path.name}"
        shutil.copy2(path, tmp)
        return raw_to_linear_u01(tmp, half_size=half_size, use_camera_wb=False)


def _face_landmarks_from_bgr(bgr: np.ndarray, face_mesh: Any) -> list:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb.flags.writeable = False
    res = face_mesh.process(rgb)
    if not res.multi_face_landmarks:
        return []
    return res.multi_face_landmarks[0].landmark


def _hands_from_bgr(bgr: np.ndarray, hands_detector: Any) -> list:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb.flags.writeable = False
    res = hands_detector.process(rgb)
    if not res.multi_hand_landmarks:
        return []
    return res.multi_hand_landmarks


def _de00(lab: Tuple[float, float, float], ref: Tuple[float, float, float]) -> float:
    a = np.array([[[lab[0], lab[1], lab[2]]]], dtype=np.float64)
    b = np.array([[[ref[0], ref[1], ref[2]]]], dtype=np.float64)
    return float(delta_e_2000(a, b)[0, 0])


def _load_iphone_calibration(calibration_dir: Path) -> Tuple[Optional[float], Optional[np.ndarray]]:
    from iphone_camera_calibration import load_calibration_bundle

    cal = load_calibration_bundle(calibration_dir)
    fnf._CAMERA_RGB_TO_XYZ = np.asarray(cal.camera_rgb_to_xyz, dtype=np.float64)
    fnf._CAMERA_RGB_TO_XYZ_AFFINE = None
    aff_path = calibration_dir / "camera_rgb_to_xyz_affine.npy"
    if aff_path.is_file():
        fnf._CAMERA_RGB_TO_XYZ_AFFINE = np.load(aff_path)
    fnf._IPHONE_FLASH_CCT_K = float(cal.flash_cct_k)
    fnf._IPHONE_FLASH_RGB = np.asarray(cal.flash_rgb_linear, dtype=np.float64)
    return float(cal.flash_cct_k), np.asarray(cal.flash_rgb_linear, dtype=np.float64)


def _xyz_lstsq_exposure_scale(camera_xyz: np.ndarray, target_xyz: np.ndarray) -> float:
    camera = np.asarray(camera_xyz, dtype=np.float64).reshape(3)
    target = np.asarray(target_xyz, dtype=np.float64).reshape(3)
    denom = float(np.dot(camera, camera))
    if denom <= 1e-12:
        return float("nan")
    return float(np.dot(camera, target) / denom)


def _get_m33() -> np.ndarray:
    """Return the 3×3 linear part of the current camera RGB→XYZ matrix."""
    M = fnf._CAMERA_RGB_TO_XYZ
    return np.asarray(M, dtype=np.float64).reshape(3, 3)


def _vonkries_rgb_scales(
    camera_white_rgb: np.ndarray,
    nix_white_xyz: np.ndarray,
) -> Optional[np.ndarray]:
    """
    Per-channel RGB scales for von Kries chromatic adaptation.

    Finds [sR, sG, sB] such that M @ (diag(s) @ camera_white_rgb) = nix_white_xyz,
    i.e. the bag white maps to the NIX D65 white after the camera calibration matrix.
    """
    try:
        M = _get_m33()
        M_inv = np.linalg.inv(M)
        target_rgb = M_inv @ np.asarray(nix_white_xyz, dtype=np.float64).reshape(3)
        scales = target_rgb / np.maximum(np.asarray(camera_white_rgb, dtype=np.float64), 1e-8)
        if not np.all(np.isfinite(scales)) or np.any(scales <= 0) or np.any(scales > 20):
            return None
        return scales
    except np.linalg.LinAlgError:
        return None


def _twopoint_affine_rgb(
    camera_white_rgb: np.ndarray,
    camera_black_rgb: np.ndarray,
    nix_white_xyz: np.ndarray,
    nix_black_xyz: np.ndarray,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Per-channel affine correction: corrected_c = a_c * raw_c + b_c.

    Maps camera_black → NIX-equivalent black and camera_white → NIX-equivalent white
    in camera-RGB space, giving a two-point density-style calibration per channel.
    """
    try:
        M = _get_m33()
        M_inv = np.linalg.inv(M)
        target_w = M_inv @ np.asarray(nix_white_xyz, dtype=np.float64).reshape(3)
        target_b = M_inv @ np.asarray(nix_black_xyz, dtype=np.float64).reshape(3)
        cam_w = np.asarray(camera_white_rgb, dtype=np.float64)
        cam_b = np.asarray(camera_black_rgb, dtype=np.float64)
        denom = cam_w - cam_b
        if np.any(np.abs(denom) < 1e-5):
            return None
        a = (target_w - target_b) / denom
        b = target_w - a * cam_w
        if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
            return None
        return a, b
    except np.linalg.LinAlgError:
        return None


# CAT02 forward matrix (XYZ → LMS)
_M_CAT02 = np.array([
    [ 0.7328,  0.4296, -0.1624],
    [-0.7036,  1.6975,  0.0061],
    [ 0.0030,  0.0136,  0.9834],
], dtype=np.float64)
_M_CAT02_INV = np.linalg.inv(_M_CAT02)


def _cat02_rgb_correction_matrix(
    camera_white_xyz: np.ndarray,
    nix_white_xyz: np.ndarray,
) -> Optional[np.ndarray]:
    """
    3×3 matrix in camera-RGB space that applies a CAT02 chromatic adaptation
    from the scene illuminant (estimated from bag white) to D65 (NIX reference).

    Returns P such that:  corrected_rgb = P @ raw_rgb
    """
    try:
        M = _get_m33()
        M_inv = np.linalg.inv(M)
        scene_xyz = np.asarray(camera_white_xyz, dtype=np.float64).reshape(3)
        target_xyz = np.asarray(nix_white_xyz, dtype=np.float64).reshape(3)
        scene_lms = _M_CAT02 @ scene_xyz
        target_lms = _M_CAT02 @ target_xyz
        if np.any(np.abs(scene_lms) < 1e-10):
            return None
        cat_diag = target_lms / scene_lms
        M_xyz_adapt = _M_CAT02_INV @ np.diag(cat_diag) @ _M_CAT02
        # Pre-correction in camera-RGB space: P = M_inv @ M_adapt @ M
        P = M_inv @ M_xyz_adapt @ M
        if not np.all(np.isfinite(P)):
            return None
        return P
    except np.linalg.LinAlgError:
        return None


def _xyz_to_cct(xyz: np.ndarray) -> Optional[float]:
    """McCamy CCT estimate from XYZ chromaticity."""
    X, Y, Z = float(xyz[0]), float(xyz[1]), float(xyz[2])
    s = X + Y + Z
    if s < 1e-8:
        return None
    x, y = X / s, Y / s
    denom = 0.1858 - y
    if abs(denom) < 1e-6:
        return None
    n = (x - 0.3320) / denom
    cct = 449.0 * n**3 + 3525.0 * n**2 + 6823.3 * n + 5520.33
    return float(cct) if 1500 < cct < 20000 else None


def _apply_rgb_correction(
    img: np.ndarray,
    *,
    diag: Optional[np.ndarray] = None,
    affine_a: Optional[np.ndarray] = None,
    affine_b: Optional[np.ndarray] = None,
    matrix: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Apply a per-channel correction (diagonal, affine, or 3×3 matrix) to a linear RGB image."""
    h, w = img.shape[:2]
    out = img.astype(np.float64)
    if diag is not None:
        out *= diag.reshape(1, 1, 3)
    elif affine_a is not None and affine_b is not None:
        out = out * affine_a.reshape(1, 1, 3) + affine_b.reshape(1, 1, 3)
    elif matrix is not None:
        out = (matrix @ out.reshape(-1, 3).T).T.reshape(h, w, 3)
    return np.clip(out, 0.0, None).astype(np.float32)


@dataclass
class _TrialCache:
    """Expensive per-trial computations that are shared across all anchor modes."""
    nf_work: np.ndarray        # resized no-flash linear
    fl_orig_work: np.ndarray   # resized flash linear (BEFORE warp, for re-warping corrected versions)
    align: AlignResult         # ECC alignment (nf_work, fl_warped_and_skin_scaled)
    lab_mask: np.ndarray       # cheek mask for lab extraction
    mask_nf: np.ndarray        # full mesh mask
    albedo_base: np.ndarray    # sqrt(nf * fl_skin_exp), unscaled


def _build_trial_cache(
    nf_lin: np.ndarray,
    fl_lin: np.ndarray,
    face_mesh: Any,
    *,
    max_align_width: int,
    skin_exclusion_dilate_iod_fraction: float,
) -> _TrialCache:
    """Run ECC alignment + face mesh + skin mask ONCE per trial."""
    import physio_skin_lab_raw_pr250 as pr250_mod

    nf_work = _resize_linear_max_width(nf_lin, max_align_width)
    fl_work = _resize_linear_max_width(fl_lin, max_align_width)

    # ECC alignment with skip_exposure so we can apply the skin-mask exposure later
    align_raw = align_flash_to_noflash_linear(nf_work, fl_work, skip_exposure=True)
    nf_bgr = pr250_mod.linear_rgb_to_preview_bgr(align_raw.noflash_linear)

    # Face mesh + skin mask (cheek mask used both for exposure and Lab extraction)
    mask_nf, _oval, _kept, _excl, _mesh_xy, cheek_mask = skin_mask_from_bgr(
        nf_bgr,
        face_mesh,
        skin_triangulation="tessellation",
        skin_exclusion_dilate_iod_fraction=skin_exclusion_dilate_iod_fraction,
        build_cheek_mask=True,
    )
    lab_mask = cheek_mask if cheek_mask is not None else mask_nf

    # Skin-mask-guided exposure matching (same as process_flash_pair with exposure_scale_skin_mask=True)
    exp_scale = estimate_exposure_scale_masked(
        align_raw.noflash_linear,
        align_raw.flash_aligned_linear,
        lab_mask,
    )
    fl_skin_exp = np.clip(align_raw.flash_aligned_linear * exp_scale, 0.0, None)
    align = AlignResult(
        flash_aligned_linear=fl_skin_exp,
        noflash_linear=align_raw.noflash_linear,
        warp_matrix=align_raw.warp_matrix,
        exposure_scale=float(exp_scale),
        ecc_cc=align_raw.ecc_cc,
    )

    albedo_base = estimate_reflectance_linear(align.noflash_linear, align.flash_aligned_linear)
    return _TrialCache(
        nf_work=nf_work,
        fl_orig_work=fl_work,
        align=align,
        lab_mask=lab_mask,
        mask_nf=mask_nf,
        albedo_base=albedo_base,
    )


_TRIM_KWARGS = dict(
    l_star_trim_lo=0.05, l_star_trim_hi=0.05,
    a_star_trim_lo=0.05, a_star_trim_hi=0.05,
    b_star_trim_lo=0.05, b_star_trim_hi=0.05,
    skin_min_chroma_ab=2.0,
)


def _lab_from_scale(
    cache: _TrialCache,
    scale: Optional[float],
    xyz_scene_white: Optional[np.ndarray] = None,
) -> Tuple[float, float, float]:
    """Fast path: multiply cached albedo_base by scale, extract Lab."""
    albedo = cache.albedo_base if scale is None else np.clip(cache.albedo_base * scale, 0.0, None)
    stats = mean_lab_reflectance_linear(albedo, cache.lab_mask, xyz_scene_white=xyz_scene_white, **_TRIM_KWARGS)
    return float(stats.L), float(stats.a), float(stats.b)


def _lab_from_corrected(
    cache: _TrialCache,
    *,
    diag: Optional[np.ndarray] = None,
    affine_a: Optional[np.ndarray] = None,
    affine_b: Optional[np.ndarray] = None,
    matrix: Optional[np.ndarray] = None,
    scale: Optional[float] = None,
    xyz_scene_white: Optional[np.ndarray] = None,
) -> Tuple[float, float, float]:
    """Apply per-channel correction to cached arrays, re-warp flash, recompute albedo, extract Lab."""
    nf_c = _apply_rgb_correction(cache.nf_work, diag=diag, affine_a=affine_a, affine_b=affine_b, matrix=matrix)
    fl_c = _apply_rgb_correction(cache.fl_orig_work, diag=diag, affine_a=affine_a, affine_b=affine_b, matrix=matrix)

    # Re-warp the corrected flash using the cached warp (skip expensive ECC re-compute)
    h, w = nf_c.shape[:2]
    fl_c_warp = cv2.warpAffine(
        fl_c.astype(np.float32),
        cache.align.warp_matrix,
        (w, h),
        flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_REPLICATE,
    ).astype(np.float64)

    exp_scale_c = estimate_exposure_scale_masked(nf_c, fl_c_warp, cache.lab_mask)
    fl_c_exp = np.clip(fl_c_warp * exp_scale_c, 0.0, None)
    albedo_c = estimate_reflectance_linear(nf_c, fl_c_exp)
    if scale is not None:
        albedo_c = np.clip(albedo_c * scale, 0.0, None)

    stats = mean_lab_reflectance_linear(albedo_c, cache.lab_mask, xyz_scene_white=xyz_scene_white, **_TRIM_KWARGS)
    return float(stats.L), float(stats.a), float(stats.b)


def _bag_white_scales(
    nf_lin: np.ndarray,
    fl_lin: np.ndarray,
    *,
    face_mesh: Any,
    hands_detector: Any,
    nix_white_y: float,
    nix_white_xyz: np.ndarray,
    nix_ref_black_xyz: np.ndarray,
    max_align_width: int,
    sam2_segmenter: Any = None,
) -> Tuple[Dict[str, float], Dict[str, Any]]:
    nf_work = _resize_linear_max_width(nf_lin, max_align_width)
    fl_work = _resize_linear_max_width(fl_lin, max_align_width)
    align = align_flash_to_noflash_linear(nf_work, fl_work)

    sources = {
        "bag_white_noflash": align.noflash_linear,
        "bag_white_flash_aligned": align.flash_aligned_linear,
        "bag_white_reflectance": np.sqrt(
            np.maximum(align.noflash_linear * align.flash_aligned_linear, 1e-12)
        ),
    }

    # Landmarks/prompts come from the reflectance preview because it has the
    # most visible bag stripes after registration.
    bgr = pr250.linear_rgb_to_preview_bgr(sources["bag_white_reflectance"])
    lm = _face_landmarks_from_bgr(bgr, face_mesh)
    if not lm:
        raise ValueError("No face detected for bag white reference")
    hand_lm = _hands_from_bgr(bgr, hands_detector)
    if not hand_lm:
        print("WARN: no hands detected; stripe-scan will run hand-free", file=sys.stderr)

    rgb_u8 = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    seg_prompt = segment_sephora_bag(
        sources["bag_white_reflectance"],
        lm,
        hand_lm,
        sam2_segmenter=sam2_segmenter,
        sam2_rgb_uint8=rgb_u8 if sam2_segmenter is not None else None,
    )
    if seg_prompt is None:
        raise ValueError("Bag stripe segmentation failed")

    scales: Dict[str, float] = {}
    info: Dict[str, Any] = {
        "bag_detection_mode": seg_prompt.detection_mode,
        "bag_bbox": [int(v) for v in seg_prompt.bag_bbox],
        "bag_align_ecc_cc": float(align.ecc_cc),
        "bag_align_exposure_scale": float(align.exposure_scale),
    }
    for source_name, img in sources.items():
        seg = segment_sephora_bag(img, lm, hand_lm)
        if seg is None:
            continue
        scale = exposure_scale_from_bag_white(seg, nix_white_y)
        scales[source_name] = scale

        white_rgb = np.asarray(seg.white_rgb_mean, dtype=np.float64)
        black_rgb = np.asarray(seg.black_rgb_mean, dtype=np.float64)
        white_xyz = fnf.linear_rgb_to_xyz_d65(white_rgb.reshape(1, 3))[0]
        black_xyz = fnf.linear_rgb_to_xyz_d65(black_rgb.reshape(1, 3))[0]
        xyz_y = float(white_xyz[1])

        if xyz_y > 1e-8:
            scales[f"{source_name}_xyz_y"] = float(nix_white_y / xyz_y)
        xyz_lstsq_scale = _xyz_lstsq_exposure_scale(white_xyz, nix_white_xyz)
        if np.isfinite(xyz_lstsq_scale) and xyz_lstsq_scale > 0:
            scales[f"{source_name}_xyz_lstsq"] = xyz_lstsq_scale

        # Von Kries per-channel scales
        vk = _vonkries_rgb_scales(white_rgb, nix_white_xyz)
        # Two-point per-channel affine
        tp = _twopoint_affine_rgb(white_rgb, black_rgb, nix_white_xyz, nix_ref_black_xyz)
        # CAT02 3×3 pre-correction matrix in camera-RGB space
        cat02 = _cat02_rgb_correction_matrix(white_xyz, nix_white_xyz)
        # Per-trial CCT from bag white chromaticity
        cct_est = _xyz_to_cct(white_xyz)

        prefix = source_name
        info[f"{prefix}_white_y"] = float(seg.white_y)
        info[f"{prefix}_white_xyz_x"] = float(white_xyz[0])
        info[f"{prefix}_white_xyz_y"] = xyz_y
        info[f"{prefix}_white_xyz_z"] = float(white_xyz[2])
        info[f"{prefix}_black_y"] = float(seg.black_y)
        info[f"{prefix}_wb_ratio"] = float(seg.white_y / max(seg.black_y, 1e-6))
        info[f"{prefix}_n_white"] = int(seg.n_white)
        info[f"{prefix}_n_black"] = int(seg.n_black)
        # Store RGB vectors and correction matrices for advanced modes
        info[f"{prefix}_white_rgb"] = white_rgb.tolist()
        info[f"{prefix}_black_rgb"] = black_rgb.tolist()
        info[f"{prefix}_black_xyz"] = black_xyz.tolist()
        info[f"{prefix}_vonkries_scales"] = vk.tolist() if vk is not None else None
        info[f"{prefix}_twopoint_a"] = tp[0].tolist() if tp is not None else None
        info[f"{prefix}_twopoint_b"] = tp[1].tolist() if tp is not None else None
        info[f"{prefix}_cat02_P"] = cat02.tolist() if cat02 is not None else None
        info[f"{prefix}_cct_estimate"] = cct_est
    return scales, info


def _summarize(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for mode in sorted({str(r["anchor_mode"]) for r in rows}):
        vals = [float(r["reflectance_cheek_de00"]) for r in rows if r["anchor_mode"] == mode]
        vals = [v for v in vals if np.isfinite(v)]
        if not vals:
            continue
        out[mode] = {
            "n": len(vals),
            "mean_de00": float(np.mean(vals)),
            "median_de00": float(median(vals)),
            "std_de00": float(np.std(vals)),
            "min_de00": float(np.min(vals)),
            "max_de00": float(np.max(vals)),
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    ap.add_argument("--nix-csv", type=Path, default=None)
    ap.add_argument("--iphone-calibration", type=Path, default=DEFAULT_CALIBRATION)
    ap.add_argument("--out-dir", type=Path, default=ROOT / "sephora_bag_white_reference_eval")
    ap.add_argument("--max-align-width", type=int, default=1600)
    ap.add_argument("--raw-half-size", type=int, default=1)
    ap.add_argument("--known-ambient-cct-k", type=float, default=6546.0)
    ap.add_argument("--known-ambient-duv", type=float, default=0.0017)
    ap.add_argument("--emily-participant", default="Participant 1")
    ap.add_argument("--liki-participant", default="Participant 2")
    ap.add_argument("--include-p2-t1", action="store_true")
    ap.add_argument("--sam2", action="store_true")
    ap.add_argument("--sam2-model", default="facebook/sam2.1-hiera-small")
    ap.add_argument("--mobile-sam", action="store_true", help="Use MobileSAM (recommended)")
    ap.add_argument("--mobile-sam-ckpt", type=Path, default=ROOT / "mobile_sam.pt")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    nix_csv = args.nix_csv or (args.data_root / "Sephorabag_measurement.csv")
    nix_ref = load_nix_bag_reference(nix_csv)
    measured_flash_cct_k, flash_rgb_measured = _load_iphone_calibration(args.iphone_calibration)
    lu_sharpening = load_lu_sharpening_matrix(args.iphone_calibration / "lu_sharpening_M.npy")
    training_anchors = load_exposure_anchors(args.iphone_calibration)

    fitskin_lookup = load_fitskin_cheek_lookup(
        fnf._DEFAULT_FITSKIN_SCAN_CSV,
        fnf._DEFAULT_FITSKIN_MAPPING_CSV,
    )

    sam2_segmenter = None
    if args.mobile_sam:
        from sephora_bag_mobile_sam import MobileSamBagSegmenter, mobile_sam_available
        if not mobile_sam_available():
            raise SystemExit(f"MobileSAM not available (ckpt: {args.mobile_sam_ckpt})")
        print(f"Loading MobileSAM from {args.mobile_sam_ckpt}...")
        sam2_segmenter = MobileSamBagSegmenter(checkpoint=args.mobile_sam_ckpt)
    elif args.sam2:
        from sephora_bag_sam2 import Sam2BagSegmenter, sam2_available
        if not sam2_available():
            raise SystemExit("SAM2 requested but unavailable")
        sam2_segmenter = Sam2BagSegmenter(model_id=args.sam2_model)

    pairs = _discover_pairs(args.data_root)
    if not pairs:
        raise SystemExit(f"No bag DNG pairs under {args.data_root}")

    rows: List[Dict[str, Any]] = []
    trial_records: List[Dict[str, Any]] = []
    mp_fm = mp.solutions.face_mesh
    mp_hands = mp.solutions.hands
    with mp_fm.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
    ) as face_mesh, mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=2,
        min_detection_confidence=0.4,
    ) as hands_detector:
        for pair in pairs:
            subject_id, participant, trial = _participant_trial_from_bag_id(
                str(pair["trial_id"]),
                emily_participant=args.emily_participant,
                liki_participant=args.liki_participant,
            )
            if subject_id == "P2_T1" and not args.include_p2_t1:
                continue
            fit = fitskin_lookup.get((participant, trial))
            if fit is None:
                print(f"WARN no FitSkin for {pair['trial_id']} -> {participant} trial {trial}", file=sys.stderr)
                continue

            nf_lin = _read_dng_u01(Path(pair["noflash"]), args.raw_half_size)
            fl_lin = _read_dng_u01(Path(pair["flash"]), args.raw_half_size)
            bag_scales, bag_info = _bag_white_scales(
                nf_lin,
                fl_lin,
                face_mesh=face_mesh,
                hands_detector=hands_detector,
                nix_white_y=float(nix_ref.white_y),
                nix_white_xyz=nix_ref.white_xyz,
                nix_ref_black_xyz=nix_ref.black_xyz,
                max_align_width=args.max_align_width,
                sam2_segmenter=sam2_segmenter,
            )

            pk = participant_key(subject_id, participant)
            fit_lab = (
                float(fit["fitskin_cheek_L"]),
                float(fit["fitskin_cheek_a"]),
                float(fit["fitskin_cheek_b"]),
            )
            trial_records.append(
                {
                    "pair": pair,
                    "subject_id": subject_id,
                    "participant": participant,
                    "trial": trial,
                    "participant_key": pk,
                    "fit_lab": fit_lab,
                    "bag_scales": bag_scales,
                    "bag_info": bag_info,
                    "nf_lin": nf_lin,
                    "fl_lin": fl_lin,
                }
            )

        participant_medians: Dict[str, Dict[str, float]] = {}
        for pk in sorted({str(r["participant_key"]) for r in trial_records}):
            participant_medians[pk] = {}
            subset = [r for r in trial_records if r["participant_key"] == pk]
            for key in (
                "bag_white_reflectance_xyz_y",
                "bag_white_flash_aligned_xyz_y",
                "bag_white_noflash_xyz_y",
                "bag_white_reflectance_xyz_lstsq",
                "bag_white_flash_aligned_xyz_lstsq",
                "bag_white_noflash_xyz_lstsq",
            ):
                vals = [
                    float(r["bag_scales"][key])
                    for r in subset
                    if key in r["bag_scales"] and np.isfinite(float(r["bag_scales"][key]))
                ]
                if vals:
                    participant_medians[pk][key] = float(np.median(vals))

        for rec0 in trial_records:
            pair = rec0["pair"]
            subject_id = str(rec0["subject_id"])
            participant = str(rec0["participant"])
            trial = str(rec0["trial"])
            pk = str(rec0["participant_key"])
            fit_lab = rec0["fit_lab"]
            bag_scales = rec0["bag_scales"]
            bag_info = rec0["bag_info"]
            nf_lin = rec0["nf_lin"]
            fl_lin = rec0["fl_lin"]

            training_scale = training_anchors.get(pk)

            # ── Scalar anchor modes (scale, lstar_correction) ──────────────────
            trial_modes: Dict[str, Tuple[Optional[float], Optional[float]]] = {
                "none": (None, None),
                "training_anchor": (training_scale, None),
            }
            for name, scale in bag_scales.items():
                trial_modes[name] = (scale, None)

            # Hybrid/lstar cross-session corrections
            for key in (
                "bag_white_reflectance_xyz_y",
                "bag_white_flash_aligned_xyz_y",
                "bag_white_noflash_xyz_y",
                "bag_white_reflectance_xyz_lstsq",
                "bag_white_flash_aligned_xyz_lstsq",
                "bag_white_noflash_xyz_lstsq",
            ):
                med = participant_medians.get(pk, {}).get(key)
                if training_scale is None or med is None or key not in bag_scales or med <= 0:
                    continue
                rel = float(bag_scales[key]) / float(med)
                trial_modes[f"hybrid_training_x_{key}_rel"] = (float(training_scale) * rel, None)
                trial_modes[f"lstar_training_x_{key}_rel"] = (training_scale, rel)

            # ── Pre-correction modes (applied to raw linear arrays) ─────────────
            # Use flash_aligned as the primary chromatic reference source.
            _src = "bag_white_flash_aligned"
            _vk   = bag_info.get(f"{_src}_vonkries_scales")
            _tp_a = bag_info.get(f"{_src}_twopoint_a")
            _tp_b = bag_info.get(f"{_src}_twopoint_b")
            _cat  = bag_info.get(f"{_src}_cat02_P")
            _cct  = bag_info.get(f"{_src}_cct_estimate")

            precorrected_variants: Dict[str, Tuple[np.ndarray, np.ndarray, Optional[float], Optional[float]]] = {}

            if _vk is not None:
                vk = np.array(_vk, dtype=np.float64)
                precorrected_variants["vonkries_bag"] = (
                    _apply_rgb_correction(nf_lin, diag=vk),
                    _apply_rgb_correction(fl_lin, diag=vk),
                    None, None,
                )
                if training_scale is not None:
                    precorrected_variants["vonkries_bag_x_training"] = (
                        _apply_rgb_correction(nf_lin, diag=vk),
                        _apply_rgb_correction(fl_lin, diag=vk),
                        training_scale, None,
                    )

            if _tp_a is not None and _tp_b is not None:
                tp_a = np.array(_tp_a, dtype=np.float64)
                tp_b = np.array(_tp_b, dtype=np.float64)
                precorrected_variants["twopoint_bag"] = (
                    _apply_rgb_correction(nf_lin, affine_a=tp_a, affine_b=tp_b),
                    _apply_rgb_correction(fl_lin, affine_a=tp_a, affine_b=tp_b),
                    None, None,
                )
                if training_scale is not None:
                    precorrected_variants["twopoint_bag_x_training"] = (
                        _apply_rgb_correction(nf_lin, affine_a=tp_a, affine_b=tp_b),
                        _apply_rgb_correction(fl_lin, affine_a=tp_a, affine_b=tp_b),
                        training_scale, None,
                    )

            if _cat is not None:
                cat_P = np.array(_cat, dtype=np.float64)
                precorrected_variants["cat02_bag"] = (
                    _apply_rgb_correction(nf_lin, matrix=cat_P),
                    _apply_rgb_correction(fl_lin, matrix=cat_P),
                    None, None,
                )
                if training_scale is not None:
                    precorrected_variants["cat02_bag_x_training"] = (
                        _apply_rgb_correction(nf_lin, matrix=cat_P),
                        _apply_rgb_correction(fl_lin, matrix=cat_P),
                        training_scale, None,
                    )

            # Build the per-trial cache (alignment + mask) — runs ONCE per trial
            try:
                tcache = _build_trial_cache(
                    nf_lin, fl_lin, face_mesh,
                    max_align_width=args.max_align_width,
                    skin_exclusion_dilate_iod_fraction=0.12,
                )
            except Exception as exc:
                print(f"WARN cache build failed for {pair['trial_id']}: {exc}", file=sys.stderr)
                continue

            bag_info_scalar = {k: v for k, v in bag_info.items()
                               if not isinstance(v, (list, np.ndarray)) or k == "bag_bbox"}

            def _record(mode_name: str, lab: Tuple[float, float, float], scale: Optional[float], lstar_corr: Optional[float]) -> None:
                if lstar_corr is not None and lstar_corr > 0:
                    lab = (float(np.clip(lab[0] * lstar_corr, 0.0, 100.0)), lab[1], lab[2])
                row: Dict[str, Any] = {
                    "trial_id": pair["trial_id"],
                    "subject_id": subject_id,
                    "participant": participant,
                    "trial": trial,
                    "anchor_mode": mode_name,
                    "reflectance_exposure_scale": scale,
                    "lstar_correction": lstar_corr,
                    "fitskin_cheek_L": fit_lab[0],
                    "fitskin_cheek_a": fit_lab[1],
                    "fitskin_cheek_b": fit_lab[2],
                    "reflectance_L": lab[0],
                    "reflectance_a": lab[1],
                    "reflectance_b": lab[2],
                    "delta_L": lab[0] - fit_lab[0],
                    "delta_a": lab[1] - fit_lab[1],
                    "delta_b": lab[2] - fit_lab[2],
                    "reflectance_cheek_de00": _de00(lab, fit_lab),
                    **bag_info_scalar,
                }
                rows.append(row)

            # ── Fast scalar modes: reuse cached albedo_base ─────────────────────
            for mode, (scale, lstar_correction) in trial_modes.items():
                try:
                    lab = _lab_from_scale(tcache, scale)
                except Exception as exc:
                    print(f"WARN {mode} failed: {exc}", file=sys.stderr)
                    continue
                _record(mode, lab, scale, lstar_correction)

            # ── Per-trial CCT from bag white: pass xyz_scene_white to Lab ────────
            if _cct is not None:
                cct_xyz_raw = [
                    bag_info.get("bag_white_flash_aligned_white_xyz_x"),
                    bag_info.get("bag_white_flash_aligned_white_xyz_y"),
                    bag_info.get("bag_white_flash_aligned_white_xyz_z"),
                ]
                if all(v is not None for v in cct_xyz_raw):
                    bag_xyz_w = np.array(cct_xyz_raw, dtype=np.float64)
                    bag_xyz_w_norm = bag_xyz_w / max(float(bag_xyz_w[1]), 1e-8)
                    try:
                        lab = _lab_from_scale(tcache, training_scale, xyz_scene_white=bag_xyz_w_norm)
                        _record("cct_from_bag", lab, training_scale, None)
                    except Exception as exc:
                        print(f"WARN cct_from_bag failed: {exc}", file=sys.stderr)

            # ── Pre-corrected chromatic modes: reuse warp, recompute albedo ──────
            if _vk is not None:
                vk = np.array(_vk, dtype=np.float64)
                for mode_name, extra_scale in [("vonkries_bag", None), ("vonkries_bag_x_training", training_scale)]:
                    if mode_name == "vonkries_bag_x_training" and training_scale is None:
                        continue
                    try:
                        lab = _lab_from_corrected(tcache, diag=vk, scale=extra_scale)
                        _record(mode_name, lab, extra_scale, None)
                    except Exception as exc:
                        print(f"WARN {mode_name} failed: {exc}", file=sys.stderr)

            if _tp_a is not None and _tp_b is not None:
                tp_a = np.array(_tp_a, dtype=np.float64)
                tp_b = np.array(_tp_b, dtype=np.float64)
                for mode_name, extra_scale in [("twopoint_bag", None), ("twopoint_bag_x_training", training_scale)]:
                    if mode_name == "twopoint_bag_x_training" and training_scale is None:
                        continue
                    try:
                        lab = _lab_from_corrected(tcache, affine_a=tp_a, affine_b=tp_b, scale=extra_scale)
                        _record(mode_name, lab, extra_scale, None)
                    except Exception as exc:
                        print(f"WARN {mode_name} failed: {exc}", file=sys.stderr)

            if _cat is not None:
                cat_P = np.array(_cat, dtype=np.float64)
                for mode_name, extra_scale in [("cat02_bag", None), ("cat02_bag_x_training", training_scale)]:
                    if mode_name == "cat02_bag_x_training" and training_scale is None:
                        continue
                    try:
                        lab = _lab_from_corrected(tcache, matrix=cat_P, scale=extra_scale)
                        _record(mode_name, lab, extra_scale, None)
                    except Exception as exc:
                        print(f"WARN {mode_name} failed: {exc}", file=sys.stderr)

            print(f"  {pair['trial_id']}: {len([r for r in rows if r['trial_id'] == pair['trial_id']])} modes done")

    csv_path = args.out_dir / "sephora_bag_white_reference_ablation.csv"
    fields = [
        "trial_id",
        "subject_id",
        "participant",
        "trial",
        "anchor_mode",
        "reflectance_exposure_scale",
        "lstar_correction",
        "bag_white_noflash_white_y",
        "bag_white_noflash_white_xyz_x",
        "bag_white_noflash_white_xyz_y",
        "bag_white_noflash_white_xyz_z",
        "bag_white_noflash_black_y",
        "bag_white_noflash_wb_ratio",
        "bag_white_flash_aligned_white_y",
        "bag_white_flash_aligned_white_xyz_x",
        "bag_white_flash_aligned_white_xyz_y",
        "bag_white_flash_aligned_white_xyz_z",
        "bag_white_flash_aligned_black_y",
        "bag_white_flash_aligned_wb_ratio",
        "bag_white_reflectance_white_y",
        "bag_white_reflectance_white_xyz_x",
        "bag_white_reflectance_white_xyz_y",
        "bag_white_reflectance_white_xyz_z",
        "bag_white_reflectance_black_y",
        "bag_white_reflectance_wb_ratio",
        "bag_detection_mode",
        "bag_bbox",
        "fitskin_cheek_L",
        "fitskin_cheek_a",
        "fitskin_cheek_b",
        "reflectance_L",
        "reflectance_a",
        "reflectance_b",
        "delta_L",
        "delta_a",
        "delta_b",
        "reflectance_cheek_de00",
        "bag_align_ecc_cc",
        "bag_align_exposure_scale",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            serial = dict(row)
            serial["bag_bbox"] = json.dumps(serial["bag_bbox"])
            writer.writerow({k: serial.get(k, "") for k in fields})

    summary_by_anchor = _summarize(rows)
    ranked = sorted(
        summary_by_anchor.items(),
        key=lambda item: (float(item[1]["mean_de00"]), float(item[1]["median_de00"])),
    )
    summary = {
        "nix_white_y": float(nix_ref.white_y),
        "nix_black_y": float(nix_ref.black_y),
        "nix_white_xyz": [float(v) for v in nix_ref.white_xyz],
        "nix_black_xyz": [float(v) for v in nix_ref.black_xyz],
        "participant_mapping": {
            "Emily": args.emily_participant,
            "Liki": args.liki_participant,
        },
        "best_anchor_by_mean_de00": ranked[0][0] if ranked else None,
        "top_anchors_by_mean_de00": [
            {"anchor_mode": mode, **stats} for mode, stats in ranked[:8]
        ],
        "summary_by_anchor": summary_by_anchor,
    }
    summary_path = args.out_dir / "sephora_bag_white_reference_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Wrote {csv_path}")
    print(f"Wrote {summary_path}")
    print(json.dumps(summary["summary_by_anchor"], indent=2))


if __name__ == "__main__":
    main()
