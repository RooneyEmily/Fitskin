"""
In-scene Sephora bag CAT02 chromatic correction for flash/no-flash reflectance.

Production mode: ``cat02_bag`` — estimate scene white from bag stripe on flash-aligned
linear RGB, build a CAT02 adaptation matrix in camera-RGB space, apply to both frames
before re-warp and skin-mask exposure matching.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np

from sephora_bag_reference import NixBagReference, segment_sephora_bag

# Best in-scene chromatic correction on Pansor (see evaluate_pansor_ablation.py).
PRODUCTION_BAG_MODE = "cat02_bag"


def is_sephora_bag_trial_row(row: dict) -> bool | None:
    """
    True = Sephora Bag session, False = Color Checker / non-bag, None = unknown manifest.
    """
    code = str(row.get("condition_code", "")).strip().upper()
    cond = str(row.get("condition", "")).strip().lower()
    if code == "BAG" or "sephora bag" in cond:
        return True
    if code == "CC" or "color checker" in cond:
        return False
    return None


_M_CAT02 = np.array(
    [
        [0.7328, 0.4296, -0.1624],
        [-0.7036, 1.6975, 0.0061],
        [0.0030, 0.0136, 0.9834],
    ],
    dtype=np.float64,
)
_M_CAT02_INV = np.linalg.inv(_M_CAT02)


def get_camera_m33() -> np.ndarray:
    import flash_no_flash_skin_lab as fnf

    if fnf._CAMERA_RGB_TO_XYZ_AFFINE is not None:
        return np.asarray(fnf._CAMERA_RGB_TO_XYZ_AFFINE[:3, :3], dtype=np.float64)
    M = fnf._CAMERA_RGB_TO_XYZ if fnf._CAMERA_RGB_TO_XYZ is not None else fnf._SRGB_D65_XYZ
    return np.asarray(M, dtype=np.float64).reshape(3, 3)


def apply_rgb_matrix_correction(img: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Apply 3×3 matrix in camera-RGB space: corrected = P @ rgb."""
    h, w = img.shape[:2]
    P = np.asarray(matrix, dtype=np.float64).reshape(3, 3)
    out = (P @ np.asarray(img, dtype=np.float64).reshape(-1, 3).T).T.reshape(h, w, 3)
    return np.clip(out, 0.0, None).astype(np.float64)


def cat02_rgb_correction_matrix(
    camera_white_xyz: np.ndarray,
    nix_white_xyz: np.ndarray,
) -> Optional[np.ndarray]:
    """
    3×3 matrix P in camera-RGB space: CAT02 from scene white (bag) to D65 (NIX).
    """
    try:
        M = get_camera_m33()
        M_inv = np.linalg.inv(M)
        scene_xyz = np.asarray(camera_white_xyz, dtype=np.float64).reshape(3)
        target_xyz = np.asarray(nix_white_xyz, dtype=np.float64).reshape(3)
        scene_lms = _M_CAT02 @ scene_xyz
        target_lms = _M_CAT02 @ target_xyz
        if np.any(np.abs(scene_lms) < 1e-10):
            return None
        cat_diag = target_lms / scene_lms
        M_xyz_adapt = _M_CAT02_INV @ np.diag(cat_diag) @ _M_CAT02
        P = M_inv @ M_xyz_adapt @ M
        if not np.all(np.isfinite(P)):
            return None
        return P
    except np.linalg.LinAlgError:
        return None


def _landmarks_from_bgr(bgr: np.ndarray, face_mesh: Any) -> list:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb.flags.writeable = False
    res = face_mesh.process(rgb)
    if not res.multi_face_landmarks:
        return []
    return res.multi_face_landmarks[0].landmark


def _hands_from_bgr(bgr: np.ndarray, hands_detector: Any) -> list:
    if hands_detector is None:
        return []
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb.flags.writeable = False
    res = hands_detector.process(rgb)
    if not res.multi_hand_landmarks:
        return []
    return res.multi_hand_landmarks


def estimate_bag_cat02_matrix(
    flash_aligned_lin: np.ndarray,
    preview_bgr: np.ndarray,
    face_mesh: Any,
    nix_ref: NixBagReference,
    *,
    hands_detector: Any = None,
    sam_segmenter: Any = None,
    require_hands_in_segment: bool = False,
) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
    """
    Segment bag on flash-aligned linear RGB; return CAT02 matrix P or None.

    When ``require_hands_in_segment`` is set, reject stripe-only hits (e.g. ColorChecker
    whites mistaken for bag stripes).
    """
    import flash_no_flash_skin_lab as fnf

    info: Dict[str, Any] = {"bag_cat02_status": "not_attempted"}
    lm = _landmarks_from_bgr(preview_bgr, face_mesh)
    if not lm:
        info["bag_cat02_status"] = "no_face"
        return None, info

    hand_lm = _hands_from_bgr(preview_bgr, hands_detector)
    rgb_u8 = cv2.cvtColor(preview_bgr, cv2.COLOR_BGR2RGB)
    seg = segment_sephora_bag(
        flash_aligned_lin,
        lm,
        hand_lm or None,
        sam2_segmenter=sam_segmenter,
        sam2_rgb_uint8=rgb_u8 if sam_segmenter is not None else None,
    )
    if seg is None:
        info["bag_cat02_status"] = "segmentation_failed"
        return None, info

    if require_hands_in_segment and "hands" not in str(seg.detection_mode):
        info["bag_cat02_status"] = "skipped_stripe_only"
        info["bag_detection_mode"] = seg.detection_mode
        return None, info

    white_rgb = np.asarray(seg.white_rgb_mean, dtype=np.float64)
    white_xyz = fnf.linear_rgb_to_xyz_d65(white_rgb.reshape(1, 3))[0]
    cat02 = cat02_rgb_correction_matrix(white_xyz, nix_ref.white_xyz)
    info.update(
        {
            "bag_cat02_status": "applied" if cat02 is not None else "matrix_failed",
            "bag_detection_mode": seg.detection_mode,
            "bag_n_white": int(seg.n_white),
            "bag_n_black": int(seg.n_black),
            "bag_white_y": float(seg.white_y),
            "bag_wb_ratio": float(seg.white_y / max(seg.black_y, 1e-6)),
            "bag_bbox": [int(v) for v in seg.bag_bbox],
        }
    )
    if cat02 is not None:
        info["bag_cat02_P"] = cat02.tolist()
    return cat02, info


def load_nix_bag_reference_json(path: Path) -> NixBagReference:
    import json
    from pathlib import Path as _Path

    p = _Path(path)
    d = json.loads(p.read_text(encoding="utf-8"))
    return NixBagReference(
        white_lab=np.asarray(d["white_lab"], dtype=np.float64),
        black_lab=np.asarray(d["black_lab"], dtype=np.float64),
        white_xyz=np.asarray(d["white_xyz"], dtype=np.float64),
        black_xyz=np.asarray(d["black_xyz"], dtype=np.float64),
        white_y=float(d["white_y"]),
        black_y=float(d["black_y"]),
    )


def apply_cat02_reflectance_pair(
    nf_work: np.ndarray,
    fl_orig_work: np.ndarray,
    warp_matrix: np.ndarray,
    lab_mask: np.ndarray,
    cat02_P: np.ndarray,
    *,
    use_skin_exposure: bool,
    ecc_cc: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, "AlignResult", Optional[float]]:
    """
    Apply CAT02 to both frames, re-warp flash, optionally skin-mask exposure scale.
    Returns (noflash_linear, flash_aligned_linear, align, skin_exposure_scale).
    """
    from flash_no_flash_skin_lab import AlignResult, estimate_exposure_scale_masked

    nf_c = apply_rgb_matrix_correction(nf_work, cat02_P)
    fl_c = apply_rgb_matrix_correction(fl_orig_work, cat02_P)

    h, w = nf_c.shape[:2]
    fl_warp = cv2.warpAffine(
        fl_c.astype(np.float32),
        warp_matrix,
        (w, h),
        flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_REPLICATE,
    ).astype(np.float64)

    align_exposure_scale_skin: Optional[float] = None
    if use_skin_exposure:
        align_exposure_scale_skin = estimate_exposure_scale_masked(nf_c, fl_warp, lab_mask)
        fl_exp = np.clip(fl_warp * align_exposure_scale_skin, 0.0, None)
    else:
        fl_exp = fl_warp

    align = AlignResult(
        flash_aligned_linear=fl_exp,
        noflash_linear=nf_c,
        warp_matrix=warp_matrix,
        exposure_scale=float(align_exposure_scale_skin if align_exposure_scale_skin is not None else 1.0),
        ecc_cc=float(ecc_cc),
    )
    return nf_c, fl_exp, align, align_exposure_scale_skin
