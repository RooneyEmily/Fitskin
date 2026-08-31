#!/usr/bin/env python3
"""
Pansor-20 indoor chart-free D65 evaluation (zip Apple Vision landmarks).

Uses only indoor face zips (no bag / no outdoor / no light-box variants):
exactly the n=65 cohort from the Hybrid D65 Colab walkthrough.

No ColorChecker at inference. FitSkin Inside Lab comes from
``Pansor Dataset Demographics.xlsx`` (evaluation target only).

Recommended claimable stack:
  --scr-mode preawb_cat --fixed-cat-k 5500 --l-sampling off
Frozen color path (pre-AWB reflectance + affine + 5500K CAT→D65).

ROI heuristics (optional, not a colorimetric model):
  --l-sampling specular_tone   # demographics ethnicity (cohort-tuned)
  --l-sampling tone_chroma     # FitSkin-free tone→ethnicity classifier
Validate ROI with: python3 scripts/validate_roi_sampling_loso.py
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import sys
import zipfile
from pathlib import Path
from statistics import mean, median
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import rawpy

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from delta_e_2000 import delta_e_2000  # noqa: E402

D65 = np.array([0.95047, 1.0, 1.08883], dtype=np.float64)


def bradford_cat_matrix(xyz_w_src: np.ndarray, xyz_w_dst: np.ndarray) -> np.ndarray:
    """3×3 Bradford–von Kries CAT (no MediaPipe / physio_skin_lab_raw_pr250 import)."""
    M_lms = np.array(
        [
            [0.8951, 0.2664, -0.1614],
            [-0.7502, 1.7135, 0.0367],
            [0.0389, -0.0685, 1.0296],
        ],
        dtype=np.float64,
    )
    s = np.asarray(xyz_w_src, dtype=np.float64).reshape(3)
    d = np.asarray(xyz_w_dst, dtype=np.float64).reshape(3)
    lms_s = M_lms @ s
    lms_d = M_lms @ d
    rho = lms_d / np.maximum(lms_s, 1e-12)
    return np.linalg.inv(M_lms) @ np.diag(rho) @ M_lms


def linear_rgb_to_preview_bgr(rgb_lin: np.ndarray) -> np.ndarray:
    """Robust 8-bit BGR preview for FairFace crop / overlays (detection only)."""
    out = np.zeros_like(rgb_lin, dtype=np.float64)
    for c in range(3):
        ch = rgb_lin[:, :, c].astype(np.float64).ravel()
        lo, hi = np.percentile(ch, [0.5, 99.5])
        if hi <= lo + 1e-12:
            hi = lo + 1e-6
        out[:, :, c] = np.clip((rgb_lin[:, :, c] - lo) / (hi - lo), 0.0, 1.0)
    rgb8 = (out * 255.0).astype(np.uint8)
    return cv2.cvtColor(rgb8, cv2.COLOR_RGB2BGR)


# Reuse Colab cheek-mask helpers via a minimal local copy to avoid Colab deps.
# These match Pansor20_Hybrid_D65_Pipeline_Colab.ipynb. Apple Vision landmarks from
# the zip are the default ROI — MediaPipe is optional (--roi mediapipe only).


def load_dng_linear(path: Path, *, half_size: bool = True, use_camera_wb: bool = True) -> np.ndarray:
    with rawpy.imread(str(path)) as raw:
        rgb = raw.postprocess(
            use_camera_wb=use_camera_wb,
            use_auto_wb=False,
            gamma=(1, 1),
            no_auto_bright=True,
            output_bps=16,
            half_size=half_size,
        ).astype(np.float64)
    return rgb / (float(np.percentile(rgb, 99.5)) + 1e-12)


from pipeline.skin_roi import (  # noqa: E402
    apple_face_cheek_masks,
    apple_face_forehead_mask,
    apple_face_skin_roi_mask,
    load_apple_landmarks,
    refine_forehead_mask,
)


def rgb_to_xyz_affine(rgb: np.ndarray, M_aff: np.ndarray) -> np.ndarray:
    x = np.asarray(rgb, dtype=np.float64).reshape(-1, 3)
    aug = np.column_stack([x, np.ones(len(x))])
    return aug @ M_aff


def xyz_to_lab(xyz: np.ndarray, xyzn=D65) -> np.ndarray:
    t = np.asarray(xyz, dtype=np.float64) / xyzn
    d = 6 / 29

    def f(u):
        return np.where(u > d**3, np.cbrt(u), u / (3 * d**2) + 4 / 29)

    fx, fy, fz = f(t[..., 0]), f(t[..., 1]), f(t[..., 2])
    return np.stack([116 * fy - 16, 500 * (fx - fy), 200 * (fy - fz)], axis=-1)


def rgb_to_xyz(
    rgb: np.ndarray,
    M_aff: np.ndarray,
    projector: Optional[Any] = None,
) -> np.ndarray:
    """Map linear camera RGB → D65 XYZ (affine, or residual projector)."""
    if projector is None:
        return rgb_to_xyz_affine(rgb, M_aff)
    from models.color_projector import apply_color_projector_rgb

    return apply_color_projector_rgb(rgb, projector, use_green_blur=True)


def _trimmed_mean_lab(lab: np.ndarray, trim: float = 0.05) -> np.ndarray:
    lo, hi = trim, 1.0 - trim
    keep = np.ones(len(lab), dtype=bool)
    for j in range(3):
        qlo, qhi = np.quantile(lab[:, j], [lo, hi])
        keep &= (lab[:, j] >= qlo) & (lab[:, j] <= qhi)
    lab2 = lab[keep] if keep.sum() >= 10 else lab
    return lab2.mean(axis=0)


# Same L*/a*/b* binning gates as physio_skin_lab_monk / chart_cc (drops hair, specular, gray).
SKIN_LAB_TRIM_DEFAULT: Dict[str, float] = {
    "l_star_trim_lo": 0.05,
    "l_star_trim_hi": 0.05,
    "a_star_trim_lo": 0.05,
    "a_star_trim_hi": 0.05,
    "b_star_trim_lo": 0.05,
    "b_star_trim_hi": 0.05,
    "min_chroma_ab": 2.0,
}
FOREHEAD_SKIN_LAB_TRIM: Dict[str, float] = {
    **SKIN_LAB_TRIM_DEFAULT,
    "l_star_trim_hi": 0.10,
}
# Forehead Lab L* std below this → pool cheek pixels for specular_tone (narrow range).
FOREHEAD_L_UNIFORM_STD = 2.5


def _clip_skin_trim_q(q: float) -> float:
    if q <= 0.0:
        return 0.0
    return min(float(q), 0.45)


def _apply_channel_quantile_trim(
    sel: np.ndarray,
    channel: np.ndarray,
    trim_lo: float,
    trim_hi: float,
) -> np.ndarray:
    """Keep pixels inside [Q_lo, Q_hi] quantile bounds on one Lab channel."""
    out = sel.copy()
    tlo = _clip_skin_trim_q(trim_lo)
    thi = _clip_skin_trim_q(trim_hi)
    if tlo > 0.0:
        out &= channel >= float(np.quantile(channel, tlo))
    if thi > 0.0:
        out &= channel <= float(np.quantile(channel, 1.0 - thi))
    return out


def apply_skin_lab_binning(
    lab: np.ndarray,
    *,
    l_star_trim_lo: float = 0.05,
    l_star_trim_hi: float = 0.05,
    a_star_trim_lo: float = 0.05,
    a_star_trim_hi: float = 0.05,
    b_star_trim_lo: float = 0.05,
    b_star_trim_hi: float = 0.05,
    min_chroma_ab: float = 2.0,
    min_keep: int = 40,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Quantile binning on masked Lab pixels (tan/gray histogram gates)."""
    lab = np.asarray(lab, dtype=np.float64)
    if lab.ndim != 2 or lab.shape[1] != 3 or lab.shape[0] < 10:
        return lab, {"skin_binning": False, "n_before": int(lab.shape[0]), "n_after": int(lab.shape[0])}

    L, a, b = lab[:, 0], lab[:, 1], lab[:, 2]
    n_raw = int(lab.shape[0])

    def _select(
        *,
        l_hi: float,
        use_chroma: bool,
    ) -> np.ndarray:
        sel = np.ones(n_raw, dtype=bool)
        sel = _apply_channel_quantile_trim(sel, L, l_star_trim_lo, l_hi)
        sel = _apply_channel_quantile_trim(sel, a, a_star_trim_lo, a_star_trim_hi)
        sel = _apply_channel_quantile_trim(sel, b, b_star_trim_lo, b_star_trim_hi)
        if use_chroma and min_chroma_ab > 0.0:
            sel &= np.hypot(a, b) >= float(min_chroma_ab)
        return sel

    sel = _select(l_hi=l_star_trim_hi, use_chroma=True)
    relaxed = None
    if int(np.count_nonzero(sel)) < min_keep:
        sel = _select(l_hi=l_star_trim_hi, use_chroma=False)
        relaxed = "chroma"
    if int(np.count_nonzero(sel)) < min_keep and l_star_trim_hi > 0.0:
        sel = _select(l_hi=0.0, use_chroma=True)
        relaxed = "L_hi"
    if int(np.count_nonzero(sel)) < min_keep:
        sel = np.ones(n_raw, dtype=bool)
        relaxed = "all"

    n_after = int(np.count_nonzero(sel))
    out = lab[sel] if n_after >= 10 else lab
    meta: Dict[str, Any] = {
        "skin_binning": True,
        "n_before": n_raw,
        "n_after": int(out.shape[0]),
        "skin_binning_kept_frac": float(n_after / max(1, n_raw)),
    }
    if relaxed:
        meta["skin_binning_relaxed"] = relaxed
    return out, meta


def mean_lab_on_mask(
    rgb,
    mask,
    M_aff,
    trim=0.05,
    min_chroma=2.0,
    projector: Optional[Any] = None,
    xyz_scene_white: Optional[np.ndarray] = None,
    cat_degree: float = 1.0,
    l_percentile: Optional[float] = None,
    l_sampling: str = "off",
    ethnicity: Optional[str] = None,
    tone_classifier: Optional[Any] = None,
    skin_lab_trim: Optional[Dict[str, float]] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Skin Lab with optional L*/a*/b* binning and specular/shadow-aware sampling.

    Returns ``(Lab, meta)``.

    ``skin_lab_trim``: if set, apply physio-style quantile gates (drops hair / specular /
    near-neutral pixels) before the sampling policy. Use ``FOREHEAD_SKIN_LAB_TRIM`` on forehead.

    ``l_sampling``:
      - ``off``: trimmed mean (frozen colorimetric path).
      - ``specular_tone``: heuristic ROI policy using demographics ethnicity
        (not a validated colorimetric model).
      - ``tone_chroma``: same ROI rules with ethnicity predicted from FitSkin-free
        cheek tone/chroma features (demographics-trained classifier).
    """
    if projector is not None:
        from models.color_projector import apply_color_projector_rgb

        xyz = apply_color_projector_rgb(rgb, projector, use_green_blur=True)
        pix_xyz = np.maximum(xyz[mask > 0], 0.0)
    else:
        pix = rgb[mask > 0]
        pix_xyz = np.maximum(rgb_to_xyz_affine(pix, M_aff), 0.0)
    if xyz_scene_white is not None:
        cat = bradford_cat_matrix(
            np.asarray(xyz_scene_white, dtype=np.float64), D65
        )
        adapted = pix_xyz @ cat.T
        d = float(np.clip(cat_degree, 0.0, 1.0))
        pix_xyz = (1.0 - d) * pix_xyz + d * adapted
    lab = xyz_to_lab(pix_xyz)
    meta: Dict[str, Any] = {}
    if skin_lab_trim:
        lab, bin_meta = apply_skin_lab_binning(lab, min_keep=40, **skin_lab_trim)
        meta.update(bin_meta)
    else:
        C = np.hypot(lab[:, 1], lab[:, 2])
        lab = lab[C >= min_chroma] if (C >= min_chroma).sum() >= 10 else lab

    if l_sampling == "specular_tone":
        out, sm = apply_specular_tone_sampling(lab, ethnicity, trim=trim)
        meta.update(sm)
    elif l_sampling == "tone_chroma":
        # Ethnicity-free at inference: tone/chroma classifier → specular_tone rules.
        if tone_classifier is None:
            raise RuntimeError("l_sampling=tone_chroma requires tone_classifier")
        from models.roi_tone_classifier import cheek_tone_features

        feat = cheek_tone_features(lab)
        pred_eth = tone_classifier.predict(feat)
        out, meta = apply_specular_tone_sampling(lab, pred_eth, trim=trim)
        meta["predicted_ethnicity"] = pred_eth
        meta["tone_features"] = {
            k: float(v) for k, v in zip(
                ("L", "a", "b", "C", "ITA", "Lp10", "Lp50", "Lp90"), feat
            )
        }
    elif l_percentile is not None:
        base = _trimmed_mean_lab(lab, trim=trim)
        p = float(np.clip(l_percentile, 0.0, 100.0))
        out = np.array(
            [float(np.percentile(lab[:, 0], p)), float(base[1]), float(base[2])],
            dtype=np.float64,
        )
        meta["l_percentile"] = p
    else:
        out = _trimmed_mean_lab(lab, trim=trim)
    return out, meta


def masked_skin_lab_pixels(
    rgb: np.ndarray,
    mask: np.ndarray,
    M_aff: np.ndarray,
    *,
    projector: Optional[Any] = None,
    xyz_scene_white: Optional[np.ndarray] = None,
    cat_degree: float = 1.0,
    skin_lab_trim: Optional[Dict[str, float]] = None,
    min_chroma: float = 2.0,
) -> np.ndarray:
    """Masked skin pixels in D65 Lab (after optional binning), for diagnostics."""
    if projector is not None:
        from models.color_projector import apply_color_projector_rgb

        xyz = apply_color_projector_rgb(rgb, projector, use_green_blur=True)
        pix_xyz = np.maximum(xyz[mask > 0], 0.0)
    else:
        pix = rgb[mask > 0]
        pix_xyz = np.maximum(rgb_to_xyz_affine(pix, M_aff), 0.0)
    if xyz_scene_white is not None:
        cat = bradford_cat_matrix(np.asarray(xyz_scene_white, dtype=np.float64), D65)
        adapted = pix_xyz @ cat.T
        d = float(np.clip(cat_degree, 0.0, 1.0))
        pix_xyz = (1.0 - d) * pix_xyz + d * adapted
    lab = xyz_to_lab(pix_xyz)
    if skin_lab_trim:
        lab, _ = apply_skin_lab_binning(lab, min_keep=40, **skin_lab_trim)
    else:
        C = np.hypot(lab[:, 1], lab[:, 2])
        lab = lab[C >= min_chroma] if (C >= min_chroma).sum() >= 10 else lab
    return lab


def probe_forehead_lab_l_std(
    rgb: np.ndarray,
    forehead_mask: np.ndarray,
    M_aff: np.ndarray,
    *,
    projector: Optional[Any] = None,
    xyz_scene_white: Optional[np.ndarray] = None,
    cat_degree: float = 1.0,
    skin_lab_trim: Optional[Dict[str, float]] = None,
) -> float:
    """Std dev of forehead-mask Lab L* after binning (uniformity probe)."""
    lab = masked_skin_lab_pixels(
        rgb,
        forehead_mask,
        M_aff,
        projector=projector,
        xyz_scene_white=xyz_scene_white,
        cat_degree=cat_degree,
        skin_lab_trim=skin_lab_trim,
    )
    if lab.ndim != 2 or lab.shape[0] < 10:
        return float("inf")
    return float(np.std(lab[:, 0]))


# Specular/shadow-aware cheek sampling — HEURISTIC ROI POLICY (not a colorimetric
# model). Kept rules are coarse, FairFace-routed, and physically motivated
# (specular L* inflation on dark skin; shadow/chroma splits on medium tones).
# Person-specific threshold patches (Latino→Indian, White a*-reject / L_p85,
# shadow_bright L-gate, Black b*-gated hiC) were removed as cohort overfit.
# Claimable colorimetry remains preawb_cat+5500K trimmed mean; this ROI layer is
# an optional deployment prior. Validate via LOSO / tone_chroma.
# Black: flash speculars inflate mean L* → L_p10.
# White: dark cheeks (base L*<58) → L_p70; else trimmed mean.
# Indian: low-a* → drop darkest 20% + high-chroma a*b*; high-a* → L_p35.
# Asian/Iranian: low-a* → shadow/hiC; low mean chroma (C*<24) → L_p70; else mean.
L_SAMPLING_BLACK_P = 10.0
L_SAMPLING_WHITE_P = 70.0
L_SAMPLING_WHITE_MAX_BASE_L = 58.0
L_SAMPLING_INDIAN_A_MAX = 11.5  # pipeline a* gate
L_SAMPLING_INDIAN_SHADOW_DROP = 0.20
L_SAMPLING_INDIAN_HIGH_A_P = 35.0
L_SAMPLING_MEDIUM_A_MAX = 10.0
L_SAMPLING_MEDIUM_C_MAX = 24.0
L_SAMPLING_MEDIUM_LOW_C_P = 70.0
# Back-compat aliases
L_SAMPLING_ASIAN_A_MAX = L_SAMPLING_MEDIUM_A_MAX
L_SAMPLING_ASIAN_C_MAX = L_SAMPLING_MEDIUM_C_MAX
L_SAMPLING_ASIAN_LOW_C_P = L_SAMPLING_MEDIUM_LOW_C_P


def _shadow_hiC_lab(lab: np.ndarray, *, drop: float = 0.20, trim: float = 0.05) -> np.ndarray:
    """Reject darkest cheek L* fraction; take a*b* from higher-chroma half."""
    thr = float(np.quantile(lab[:, 0], drop))
    bright = lab[lab[:, 0] >= thr]
    bright = bright if len(bright) >= 10 else lab
    C = np.hypot(lab[:, 1], lab[:, 2])
    hi = lab[C >= float(np.median(C))]
    hi = hi if len(hi) >= 10 else lab
    ab = _trimmed_mean_lab(hi, trim=trim)
    L = float(_trimmed_mean_lab(bright, trim=trim)[0])
    return np.array([L, float(ab[1]), float(ab[2])], dtype=np.float64)


def apply_specular_tone_sampling(
    lab: np.ndarray,
    ethnicity: Optional[str],
    *,
    trim: float = 0.05,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    base = _trimmed_mean_lab(lab, trim=trim)
    eth = (ethnicity or "").strip().lower()
    meta: Dict[str, Any] = {}

    if eth == "black":
        p = L_SAMPLING_BLACK_P
        out = np.array(
            [float(np.percentile(lab[:, 0], p)), float(base[1]), float(base[2])],
            dtype=np.float64,
        )
        meta["l_percentile"] = p
        return out, meta

    if eth == "white":
        if float(base[0]) < L_SAMPLING_WHITE_MAX_BASE_L:
            p = L_SAMPLING_WHITE_P
            out = np.array(
                [float(np.percentile(lab[:, 0], p)), float(base[1]), float(base[2])],
                dtype=np.float64,
            )
            meta["l_percentile"] = p
            meta["white_branch"] = "dark_Lp"
            return out, meta
        meta["white_branch"] = "base"
        return base, meta

    if eth == "indian":
        if float(base[1]) < L_SAMPLING_INDIAN_A_MAX:
            meta["indian_branch"] = "shadow_hiC"
            return _shadow_hiC_lab(lab, drop=L_SAMPLING_INDIAN_SHADOW_DROP, trim=trim), meta
        p = L_SAMPLING_INDIAN_HIGH_A_P
        out = np.array(
            [float(np.percentile(lab[:, 0], p)), float(base[1]), float(base[2])],
            dtype=np.float64,
        )
        meta["indian_branch"] = "high_a_Lp"
        meta["l_percentile"] = p
        return out, meta

    if eth in ("asian", "iranian"):
        Cmean = float(np.mean(np.hypot(lab[:, 1], lab[:, 2])))
        branch_key = "asian_branch" if eth == "asian" else "iranian_branch"
        if float(base[1]) < L_SAMPLING_MEDIUM_A_MAX:
            meta[branch_key] = "shadow_hiC"
            return _shadow_hiC_lab(lab, drop=0.20, trim=trim), meta
        if Cmean < L_SAMPLING_MEDIUM_C_MAX:
            p = L_SAMPLING_MEDIUM_LOW_C_P
            out = np.array(
                [float(np.percentile(lab[:, 0], p)), float(base[1]), float(base[2])],
                dtype=np.float64,
            )
            meta[branch_key] = "low_C_Lp"
            meta["l_percentile"] = p
            return out, meta
        meta[branch_key] = "base"
        return base, meta

    return base, meta


def resolve_cheek_l_percentile(
    ethnicity: Optional[str],
    *,
    policy: str = "off",
    base_L: Optional[float] = None,
    base_a: Optional[float] = None,
) -> Optional[float]:
    """Legacy helper for Black/White L* percentile only (Indian uses full sampler)."""
    if policy in ("off", "mean", "none", ""):
        return None
    if policy != "specular_tone":
        raise ValueError(f"Unknown L* sampling policy: {policy}")
    eth = (ethnicity or "").strip().lower()
    if eth == "black":
        return L_SAMPLING_BLACK_P
    if eth == "white" and base_L is not None and float(base_L) < L_SAMPLING_WHITE_MAX_BASE_L:
        return L_SAMPLING_WHITE_P
    if eth == "indian" and base_a is not None and float(base_a) >= L_SAMPLING_INDIAN_A_MAX:
        return L_SAMPLING_INDIAN_HIGH_A_P
    return None


def mediapipe_cheek_mask(linear_rgb: np.ndarray, face_mesh: Any) -> np.ndarray:
    """FitSkin-aligned MediaPipe cheek hull ∩ mesh on a linear RGB frame."""
    from flash_no_flash_skin_lab import skin_mask_from_bgr

    bgr = linear_rgb_to_preview_bgr(linear_rgb)
    _, _, _, _, _, cheek = skin_mask_from_bgr(
        bgr,
        face_mesh,
        skin_triangulation="tessellation",
        skin_exclusion_dilate_iod_fraction=0.12,
        build_cheek_mask=True,
    )
    if cheek is None or int(np.count_nonzero(cheek)) < 50:
        raise RuntimeError("MediaPipe cheek mask empty")
    return cheek


def luma(rgb: np.ndarray) -> np.ndarray:
    return rgb @ np.array([0.2126, 0.7152, 0.0722])


def match_flash_exposure(A, B, mask) -> Tuple[np.ndarray, float]:
    m = mask > 0
    scale = float(np.median(luma(A)[m])) / max(float(np.median(luma(B)[m])), 1e-8)
    return np.clip(B * scale, 0, None), scale


def load_demographics(xlsx: Path) -> Dict[int, Dict[str, Any]]:
    import openpyxl

    wb = openpyxl.load_workbook(xlsx, data_only=True)
    ws = wb.active
    out: Dict[int, Dict[str, Any]] = {}
    for row in ws.iter_rows(min_row=3, values_only=True):
        if row[0] is None:
            continue
        try:
            pid = int(row[0])
        except (TypeError, ValueError):
            continue
        name = str(row[1] or "").strip()
        eth = str(row[2] or "").strip()
        L, a, b = row[3], row[4], row[5]
        if L is None or a is None or b is None:
            continue
        out[pid] = {
            "name": name,
            "ethnicity": eth,
            "fitskin_L": float(L),
            "fitskin_a": float(a),
            "fitskin_b": float(b),
        }
    return out


def is_indoor_chartfree_zip(stem: str) -> bool:
    low = stem.lower()
    if "bag" in low or "outside" in low or "light" in low:
        return False
    # Require trailing trial number, e.g. Shuyi1 / Dylan4
    return bool(re.search(r"\d+$", stem))


def discover_indoor_trials(data_root: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for part_dir in sorted(data_root.glob("Participant *"), key=lambda p: int(p.name.split()[-1])):
        pid = int(part_dir.name.split()[-1])
        for zpath in sorted(part_dir.glob("*.zip")):
            if not is_indoor_chartfree_zip(zpath.stem):
                continue
            m = re.search(r"(\d+)$", zpath.stem)
            trial = int(m.group(1)) if m else 0
            rows.append(
                {
                    "participant_id": pid,
                    "trial": trial,
                    "zip_stem": zpath.stem,
                    "zip_path": str(zpath),
                    "subject_id": f"P{pid}_T{trial}",
                }
            )
    rows.sort(key=lambda r: (r["participant_id"], r["trial"]))
    return rows


def extract_zip(zip_path: Path, out_dir: Path) -> Tuple[Path, Path, Path]:
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
        nf = fl = lm = None
        for n in names:
            bn = Path(n).name
            low = bn.lower()
            if bn == "1_raw_Photo.dng":
                nf = n
            elif bn == "2_raw_RAW___Flash.dng":
                fl = n
            if low.endswith("face_landmarks.json") and "flash" not in low:
                lm = n
        if nf is None or fl is None:
            for n in names:
                low = Path(n).name.lower()
                if not low.endswith(".dng"):
                    continue
                if "flash" in low and fl is None:
                    fl = n
                elif "flash" not in low and nf is None:
                    nf = n
        if nf is None or fl is None or lm is None:
            raise FileNotFoundError(f"Incomplete zip: {zip_path}")
        nf_path = out_dir / "NoFlash.dng"
        fl_path = out_dir / "Flash.dng"
        lm_path = out_dir / "face_landmarks.json"
        nf_path.write_bytes(zf.read(nf))
        fl_path.write_bytes(zf.read(fl))
        lm_path.write_bytes(zf.read(lm))
    return nf_path, fl_path, lm_path


def load_affine(cal_dir: Path) -> np.ndarray:
    npy = cal_dir / "camera_rgb_to_xyz_affine.npy"
    if npy.is_file():
        return np.load(npy)
    with (cal_dir / "iphone_calibration_bundle.json").open(encoding="utf-8") as f:
        cal = json.load(f)
    return np.asarray(cal["camera_rgb_to_xyz_affine"], dtype=np.float64)


def choose_cal_dir(ethnicity: str, *, hybrid: bool, default_cal: Path, tone_root: Path) -> Path:
    if not hybrid:
        return default_cal
    eth = ethnicity.strip().lower()
    if eth == "black":
        dark = tone_root / "dark"
        if (dark / "camera_rgb_to_xyz_affine.npy").is_file() or (
            dark / "iphone_calibration_bundle.json"
        ).is_file():
            return dark
    return default_cal


def process_one(
    *,
    zip_path: Path,
    work_dir: Path,
    M: np.ndarray,
    half_size: bool,
    roi: str = "apple",
    face_mesh: Any = None,
    projector: Optional[Any] = None,
    scr_mode: str = "off",
    scr_spectral_sensitivity: Optional[np.ndarray] = None,
    scr_wavelengths_nm: Optional[np.ndarray] = None,
    scr_prior_name: Optional[str] = None,
    scr_extra_spds: Optional[List[np.ndarray]] = None,
    scr_basis_cct_k: Optional[Tuple[float, ...]] = None,
    cat_degree: float = 1.0,
    fixed_cat_k: Optional[float] = None,
    l_sampling: str = "off",
    ethnicity: Optional[str] = None,
    tone_classifier: Optional[Any] = None,
    fairface_predictor: Optional[Any] = None,
) -> Dict[str, Any]:
    nf, fl, lm_path = extract_zip(zip_path, work_dir)

    # Production path: camera WB reflectance (baseline / old reflectance_cat).
    A = load_dng_linear(nf, half_size=half_size, use_camera_wb=True)
    B_raw = load_dng_linear(fl, half_size=half_size, use_camera_wb=True)
    if A.shape != B_raw.shape:
        B_raw = cv2.resize(B_raw, (A.shape[1], A.shape[0]), interpolation=cv2.INTER_AREA)

    # Pre-AWB frames for SCR / no-double-adapt / fixed-CAT paths.
    A0 = load_dng_linear(nf, half_size=half_size, use_camera_wb=False)
    B0 = load_dng_linear(fl, half_size=half_size, use_camera_wb=False)
    if A0.shape != A.shape:
        A0 = cv2.resize(A0, (A.shape[1], A.shape[0]), interpolation=cv2.INTER_AREA)
    if B0.shape != A.shape:
        B0 = cv2.resize(B0, (A.shape[1], A.shape[0]), interpolation=cv2.INTER_AREA)

    if roi == "mediapipe":
        if face_mesh is None:
            raise RuntimeError("face_mesh required for --roi mediapipe")
        cheek = mediapipe_cheek_mask(A, face_mesh)
        lm = None
    else:
        lm = load_apple_landmarks(lm_path)
        _, cheek = apple_face_cheek_masks(lm, A.shape[0], A.shape[1])
    n_cheek = int(np.count_nonzero(cheek))
    if n_cheek < 50:
        raise RuntimeError(f"empty cheek mask ({n_cheek} px)")

    # Deployment race prior: FairFace on no-flash preview crop (no demographics).
    fairface_meta: Dict[str, Any] = {}
    ethnicity_for_sampling = ethnicity
    sampling_mode = l_sampling
    if l_sampling in ("fairface", "fairface4", "fairface7"):
        if fairface_predictor is None:
            raise RuntimeError("fairface l-sampling requires fairface_predictor")
        if lm is None:
            lm = load_apple_landmarks(lm_path)
        from models.fairface_race import face_rgb_crop_from_landmarks

        preview = linear_rgb_to_preview_bgr(A0)
        # Landmarks are for full-res; A0 may be half-size — masks already match A0,
        # so rebuild landmarks scale via apple helper geometry on preview size.
        face_rgb = face_rgb_crop_from_landmarks(preview, lm, padding=0.35)
        ff = fairface_predictor.predict_rgb(face_rgb)
        ethnicity_for_sampling = ff["predicted_ethnicity"]
        sampling_mode = "specular_tone"
        fairface_meta = {
            "predicted_ethnicity": ff["predicted_ethnicity"],
            "fairface_label": ff["fairface_label"],
            "fairface_confidence": ff["confidence"],
            "fairface_mode": ff["mode"],
            "fairface_probs": ff["race_probs"],
        }

    B, flash_scale = match_flash_exposure(A, B_raw, cheek)
    R = np.sqrt(np.maximum(A, 0) * np.maximum(B, 0) + 1e-8)

    B0m, flash_scale0 = match_flash_exposure(A0, B0, cheek)
    R0 = np.sqrt(np.maximum(A0, 0) * np.maximum(B0m, 0) + 1e-8)

    scr_meta: Dict[str, Any] = {}
    scr = None
    need_scr = scr_mode not in ("off", "preawb_cat")
    if need_scr:
        from scr_awb import (
            RICH_BASIS_CCT_K,
            estimate_scr_awb,
            load_scr_awb_prior,
            white_balance_diagonal,
        )

        if scr_spectral_sensitivity is None or scr_wavelengths_nm is None or not scr_prior_name:
            raise RuntimeError("SCR mode requires sensitivity + prior")
        prior = load_scr_awb_prior(scr_prior_name)
        basis = scr_basis_cct_k or RICH_BASIS_CCT_K
        scr = estimate_scr_awb(
            A0,
            cheek,
            spectral_sensitivity_rgb=scr_spectral_sensitivity,
            wavelengths_nm=scr_wavelengths_nm,
            skin_prior=prior,
            basis_cct_k=basis,
            extra_spds=scr_extra_spds,
            known_ambient_cct_k=None,
        )
        scr_meta = {
            "scr_prior": prior.name,
            "scr_cct_k": float(scr.ambient_cct_k),
            "scr_residual": float(scr.residual_norm),
            "scr_alpha": [float(x) for x in scr.alpha],
            "scr_illuminant_rgb": [float(x) for x in scr.illuminant_rgb],
            "scr_basis": list(basis),
            "cat_degree": float(cat_degree),
        }

    def _with_l_sampling(rgb, *, xyz_white=None, degree: float = 1.0):
        """Cheek Lab with optional ethnicity specular/shadow sampling."""
        Lab, meta = mean_lab_on_mask(
            rgb,
            cheek,
            M,
            projector=projector,
            xyz_scene_white=xyz_white,
            cat_degree=degree,
            l_sampling=sampling_mode,
            ethnicity=ethnicity_for_sampling,
            tone_classifier=tone_classifier,
        )
        return Lab, meta

    # --- Lab paths ---
    sample_meta: Dict[str, Any] = {}
    if scr_mode == "standalone":
        assert scr is not None
        from scr_awb import white_balance_diagonal

        wb = white_balance_diagonal(A0, scr.illuminant_rgb)
        Lab, sample_meta = _with_l_sampling(wb)
        scale_used = flash_scale0
    elif scr_mode == "raw_scr_wb":
        # No camera WB; SCR diagonal WB on reflectance in same RGB space; no CAT.
        assert scr is not None
        from scr_awb import white_balance_diagonal

        R_wb = white_balance_diagonal(R0, scr.illuminant_rgb)
        Lab, sample_meta = _with_l_sampling(R_wb)
        scale_used = flash_scale0
    elif scr_mode in ("raw_scr_cat", "raw_scr_cat_d50"):
        # No camera WB reflectance; affine then CAT from SCR CCT (optional partial D).
        assert scr is not None
        from flash_noflash_spectral import planck_xyz_y1

        d = 0.5 if scr_mode == "raw_scr_cat_d50" else float(cat_degree)
        xyz_white = planck_xyz_y1(float(scr.ambient_cct_k), 0.0)
        Lab, sample_meta = _with_l_sampling(R0, xyz_white=xyz_white, degree=d)
        scale_used = flash_scale0
        scr_meta["cat_degree"] = d
    elif scr_mode == "reflectance_cat":
        # Old double-adapt path (camera WB + CAT) — kept for comparison.
        assert scr is not None
        from flash_noflash_spectral import planck_xyz_y1

        xyz_white = planck_xyz_y1(float(scr.ambient_cct_k), 0.0)
        Lab, sample_meta = _with_l_sampling(
            R, xyz_white=xyz_white, degree=float(cat_degree)
        )
        scale_used = flash_scale
    elif scr_mode == "preawb_cat":
        # Best FitSkin-free path: pre-AWB reflectance + fixed Planck CAT → D65.
        from flash_noflash_spectral import planck_xyz_y1

        cct = float(fixed_cat_k if fixed_cat_k is not None else 5500.0)
        xyz_white = planck_xyz_y1(cct, 0.0)
        Lab, sample_meta = _with_l_sampling(
            R0, xyz_white=xyz_white, degree=float(cat_degree)
        )
        scale_used = flash_scale0
        scr_meta["fixed_cat_k"] = cct
        scr_meta["cat_degree"] = float(cat_degree)
    else:
        Lab, sample_meta = _with_l_sampling(R)
        scale_used = flash_scale

    out = {
        "L": float(Lab[0]),
        "a": float(Lab[1]),
        "b": float(Lab[2]),
        "n_cheek": n_cheek,
        "flash_scale": float(scale_used),
        "shape": list(A.shape),
        "roi": roi,
        "scr_mode": scr_mode,
        "l_sampling": l_sampling,
        "l_percentile": sample_meta.get("l_percentile"),
        "indian_branch": sample_meta.get("indian_branch"),
        "asian_branch": sample_meta.get("asian_branch"),
        "iranian_branch": sample_meta.get("iranian_branch"),
        "white_branch": sample_meta.get("white_branch"),
        "predicted_ethnicity": sample_meta.get("predicted_ethnicity")
        or fairface_meta.get("predicted_ethnicity"),
    }
    out.update(scr_meta)
    out.update(fairface_meta)
    if sample_meta.get("predicted_ethnicity"):
        out["predicted_ethnicity"] = sample_meta["predicted_ethnicity"]
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--data-root",
        type=Path,
        default=Path("/home/mabl-main/Documents/Pansor Dataset"),
    )
    ap.add_argument(
        "--demographics",
        type=Path,
        default=None,
        help="Defaults to <data-root>/Pansor Dataset Demographics.xlsx",
    )
    ap.add_argument(
        "--cal-dir",
        type=Path,
        default=ROOT / "calibration" / "tier3_affine",
    )
    ap.add_argument(
        "--tone-root",
        type=Path,
        default=ROOT / "calibration" / "tier3_by_tone",
        help="Used when --hybrid routes Black → dark bundle.",
    )
    ap.add_argument(
        "--hybrid",
        action="store_true",
        help="Route Black ethnicity to tier3_by_tone/dark; others stay tier3_affine.",
    )
    ap.add_argument("--half-size", action="store_true", default=True)
    ap.add_argument("--full-res", action="store_true", help="Disable half-size demosaic.")
    ap.add_argument(
        "--work-dir",
        type=Path,
        default=ROOT / "results" / "pansor20_chartfree_d65" / "_extract",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "results" / "pansor20_chartfree_d65",
    )
    ap.add_argument("--limit", type=int, default=0, help="Process only first N trials (debug).")
    ap.add_argument(
        "--participant",
        type=int,
        default=0,
        help="Only process this participant ID (e.g. 6 for Shuyi).",
    )
    ap.add_argument(
        "--roi",
        choices=("apple", "mediapipe"),
        default="apple",
        help="Cheek ROI: zip Apple Vision landmarks (default) or MediaPipe FaceMesh.",
    )
    ap.add_argument(
        "--color-projector",
        type=Path,
        default=None,
        help="Optional residual projector .npz (applied on top of affine).",
    )
    ap.add_argument(
        "--lab-corrector",
        type=Path,
        default=None,
        help="Optional Lab affine 4x3 .npy (or .json with lab_affine_4x3) applied after cheek Lab.",
    )
    ap.add_argument(
        "--scr-awb",
        action="store_true",
        help="Enable Zhou/Kaida SCR-AWB illuminant (ISSA prior + monochromator sensitivity).",
    )
    ap.add_argument(
        "--scr-mode",
        choices=(
            "reflectance_cat",
            "standalone",
            "raw_scr_wb",
            "raw_scr_cat",
            "raw_scr_cat_d50",
            "preawb_cat",
            "both",
            "v2",
        ),
        default="v2",
        help=(
            "Color path. preawb_cat = pre-AWB reflectance + fixed Planck CAT→D65 "
            "(no SCR). v2 = raw_scr_wb + raw_scr_cat + raw_scr_cat_d50. "
            "both = old reflectance_cat + standalone."
        ),
    )
    ap.add_argument(
        "--fixed-cat-k",
        type=float,
        default=5500.0,
        help="Planck CCT (K) for --scr-mode preawb_cat Bradford CAT → D65.",
    )
    ap.add_argument(
        "--l-sampling",
        choices=("off", "specular_tone", "tone_chroma", "fairface", "fairface4", "fairface7"),
        default="off",
        help=(
            "Cheek ROI policy (not a colorimetric model). "
            "off=trimmed mean; specular_tone=demographics ethnicity; "
            "tone_chroma=Lab tone→ethnicity; "
            "fairface/fairface7=FairFace-7 on face crop; fairface4=FairFace-4."
        ),
    )
    ap.add_argument(
        "--tone-classifier",
        type=Path,
        default=None,
        help="JSON from scripts/train_roi_tone_classifier.py (required for tone_chroma).",
    )
    ap.add_argument(
        "--fairface-dir",
        type=Path,
        default=None,
        help="Directory with FairFace .pt weights (default: calibration/fairface).",
    )
    ap.add_argument(
        "--emily-tsv",
        action="store_true",
        help="Also write table_emily_format.tsv (Participant_ID / Trial / Lab / ΔE).",
    )
    ap.add_argument(
        "--skin-reflectance-prior",
        type=str,
        default=None,
        help="Override ISSA prior name/path (default: ethnicity map from demographics).",
    )
    ap.add_argument(
        "--cat-degree",
        type=float,
        default=1.0,
        help="Bradford adaptation degree for *cat modes (0–1).",
    )
    args = ap.parse_args()

    demog_path = args.demographics or (args.data_root / "Pansor Dataset Demographics.xlsx")
    demo = load_demographics(demog_path)
    trials = discover_indoor_trials(args.data_root)
    if args.participant > 0:
        trials = [t for t in trials if int(t["participant_id"]) == int(args.participant)]
    if args.limit > 0:
        trials = trials[: args.limit]
    if len(trials) == 0:
        raise SystemExit(f"No indoor chart-free zips under {args.data_root}")

    half = not args.full_res
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.work_dir.mkdir(parents=True, exist_ok=True)

    projector = None
    if args.color_projector is not None:
        from models.color_projector import load_color_projector_artifact

        projector = load_color_projector_artifact(args.color_projector.expanduser().resolve())

    lab_W = None
    if args.lab_corrector is not None:
        p = args.lab_corrector.expanduser().resolve()
        if p.suffix == ".npy":
            lab_W = np.load(p)
        else:
            payload = json.loads(p.read_text(encoding="utf-8"))
            lab_W = np.asarray(payload["lab_affine_4x3"], dtype=np.float64)
        if lab_W.shape != (4, 3):
            raise SystemExit(f"lab corrector must be 4x3, got {lab_W.shape}")

    tone_classifier = None
    if args.l_sampling == "tone_chroma":
        from models.roi_tone_classifier import DEFAULT_TONES_PATH, RoiToneClassifier

        clf_path = (args.tone_classifier or DEFAULT_TONES_PATH).expanduser().resolve()
        if not clf_path.is_file():
            raise SystemExit(
                f"--l-sampling tone_chroma needs classifier JSON at {clf_path}. "
                "Run: python3 scripts/train_roi_tone_classifier.py"
            )
        tone_classifier = RoiToneClassifier.load(clf_path)
        print(f"Loaded tone classifier: {clf_path}  classes={tone_classifier.classes}")

    fairface_predictor = None
    if args.l_sampling in ("fairface", "fairface4", "fairface7"):
        from models.fairface_race import FairFacePredictor

        mode = "4" if args.l_sampling == "fairface4" else "7"
        fairface_predictor = FairFacePredictor.load(
            mode=mode, weights_dir=args.fairface_dir
        )
        print(f"Loaded FairFace-{mode} on {fairface_predictor.device}")

    scr_S = scr_wl = None
    scr_extra: Optional[List[np.ndarray]] = None
    scr_basis_override: Optional[Tuple[float, ...]] = None
    # preawb_cat does not need SCR priors / spectral sensitivity.
    if args.scr_awb and args.scr_mode != "preawb_cat":
        from scr_awb import load_flash_spd_on_wl

        bundle = args.cal_dir / "iphone_calibration_bundle.json"
        if not bundle.is_file():
            raise SystemExit(f"--scr-awb needs {bundle}")
        cal = json.loads(bundle.read_text(encoding="utf-8"))
        if "spectral_sensitivity_rgb" not in cal or "wavelengths_nm" not in cal:
            raise SystemExit(f"Missing spectral_sensitivity_rgb/wavelengths_nm in {bundle}")
        scr_S = np.asarray(cal["spectral_sensitivity_rgb"], dtype=np.float64)
        scr_wl = np.asarray(cal["wavelengths_nm"], dtype=np.float64)
        flash_spd = load_flash_spd_on_wl(cal, scr_wl)
        # Keep basis rank ≤ 3 (RGB has 3 obs). Prefer flash SPD as 3rd basis over 6500K Planck.
        if flash_spd is not None:
            scr_extra = [flash_spd]
            scr_basis_override = (2300.0, 4500.0)
        else:
            scr_extra = None
            scr_basis_override = None

    face_mesh = None
    face_mesh_cm = None
    if args.roi == "mediapipe":
        import mediapipe as mp

        face_mesh_cm = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
        )
        face_mesh = face_mesh_cm.__enter__()

    # cache default affine
    default_M = load_affine(args.cal_dir)
    M_cache: Dict[str, np.ndarray] = {str(args.cal_dir.resolve()): default_M}

    if args.scr_mode == "preawb_cat":
        modes = ["preawb_cat"]
    elif not args.scr_awb:
        modes = ["off"]
    elif args.scr_mode == "v2":
        modes = ["raw_scr_wb", "raw_scr_cat", "raw_scr_cat_d50"]
    elif args.scr_mode == "both":
        modes = ["reflectance_cat", "standalone"]
    else:
        modes = [args.scr_mode]

    try:
        for mode in modes:
            rows_out: List[Dict[str, Any]] = []
            by_eth: Dict[str, List[float]] = {}
            mode_out = args.out_dir
            if args.scr_awb and args.scr_mode in ("both", "v2"):
                mode_out = args.out_dir / mode
                mode_out.mkdir(parents=True, exist_ok=True)

            print(f"Indoor chart-free trials: {len(trials)}")
            print(
                f"Calibration: {args.cal_dir}  hybrid={args.hybrid}  half_size={half}  "
                f"roi={args.roi}  projector={bool(projector)}  lab_corrector={lab_W is not None}  "
                f"scr_mode={mode}  l_sampling={args.l_sampling}  fixed_cat_k={args.fixed_cat_k}"
            )

            for i, t in enumerate(trials, 1):
                pid = int(t["participant_id"])
                meta = demo.get(pid)
                if meta is None:
                    print(f"SKIP {t['subject_id']}: no demographics", flush=True)
                    continue
                cal_dir = choose_cal_dir(
                    meta["ethnicity"],
                    hybrid=args.hybrid,
                    default_cal=args.cal_dir,
                    tone_root=args.tone_root,
                )
                key = str(cal_dir.resolve())
                if key not in M_cache:
                    M_cache[key] = load_affine(cal_dir)
                M = M_cache[key]
                work = args.work_dir / t["subject_id"]
                from scr_awb import resolve_prior_name

                prior_name = (
                    resolve_prior_name(
                        f"Participant {pid}",
                        t["subject_id"],
                        override=args.skin_reflectance_prior,
                        ethnicity=meta["ethnicity"],
                    )
                    if args.scr_awb and mode != "preawb_cat"
                    else None
                )
                try:
                    pred = process_one(
                        zip_path=Path(t["zip_path"]),
                        work_dir=work,
                        M=M,
                        half_size=half,
                        roi=args.roi,
                        face_mesh=face_mesh,
                        projector=projector,
                        scr_mode=mode,
                        scr_spectral_sensitivity=scr_S,
                        scr_wavelengths_nm=scr_wl,
                        scr_prior_name=prior_name,
                        scr_extra_spds=scr_extra,
                        scr_basis_cct_k=scr_basis_override,
                        cat_degree=float(args.cat_degree),
                        fixed_cat_k=float(args.fixed_cat_k),
                        l_sampling=str(args.l_sampling),
                        ethnicity=meta["ethnicity"],
                        tone_classifier=tone_classifier,
                        fairface_predictor=fairface_predictor,
                    )
                except Exception as exc:
                    print(f"FAIL {t['subject_id']}: {exc}", flush=True)
                    continue
                lab = np.array([pred["L"], pred["a"], pred["b"]], dtype=np.float64)
                if lab_W is not None:
                    lab = np.array([lab[0], lab[1], lab[2], 1.0], dtype=np.float64) @ lab_W
                fit = np.array(
                    [meta["fitskin_L"], meta["fitskin_a"], meta["fitskin_b"]], dtype=np.float64
                )
                de = float(delta_e_2000(lab, fit))
                rec = {
                    "subject_id": t["subject_id"],
                    "participant_id": pid,
                    "name": meta["name"],
                    "ethnicity": meta["ethnicity"],
                    "trial": t["trial"],
                    "zip_stem": t["zip_stem"],
                    "zip_path": t["zip_path"],
                    "matrix": str(cal_dir),
                    "roi": args.roi,
                    "scr_mode": mode,
                    "l_sampling": args.l_sampling,
                    "l_percentile": pred.get("l_percentile"),
                    "indian_branch": pred.get("indian_branch"),
                    "asian_branch": pred.get("asian_branch"),
                    "iranian_branch": pred.get("iranian_branch"),
                    "white_branch": pred.get("white_branch"),
                    "predicted_ethnicity": pred.get("predicted_ethnicity"),
                    "fairface_label": pred.get("fairface_label"),
                    "fairface_confidence": pred.get("fairface_confidence"),
                    "scr_prior": pred.get("scr_prior"),
                    "scr_cct_k": pred.get("scr_cct_k"),
                    "scr_residual": pred.get("scr_residual"),
                    "fixed_cat_k": pred.get("fixed_cat_k"),
                    "pipeline_L": round(float(lab[0]), 4),
                    "pipeline_a": round(float(lab[1]), 4),
                    "pipeline_b": round(float(lab[2]), 4),
                    "fitskin_L": meta["fitskin_L"],
                    "fitskin_a": meta["fitskin_a"],
                    "fitskin_b": meta["fitskin_b"],
                    "de00": de,
                    "n_cheek": pred["n_cheek"],
                    "flash_scale": pred["flash_scale"],
                }
                rows_out.append(rec)
                by_eth.setdefault(meta["ethnicity"], []).append(de)
                cct_s = f"  CCT={pred.get('scr_cct_k'):.0f}K" if pred.get("scr_cct_k") else ""
                if pred.get("fixed_cat_k") is not None and not pred.get("scr_cct_k"):
                    cct_s = f"  CAT={pred.get('fixed_cat_k'):.0f}K"
                lp = pred.get("l_percentile")
                lp_s = f"  L_p={lp:.0f}" if lp is not None else ""
                br = (
                    pred.get("indian_branch")
                    or pred.get("asian_branch")
                    or pred.get("iranian_branch")
                )
                if br:
                    if pred.get("indian_branch"):
                        tag = "Ind"
                    elif pred.get("asian_branch"):
                        tag = "As"
                    else:
                        tag = "Ir"
                    lp_s = f"  {tag}={br}" + (f"/{lp:.0f}" if lp is not None else "")
                if pred.get("fairface_label"):
                    lp_s += (
                        f"  FF={pred.get('fairface_label')}"
                        f"→{pred.get('predicted_ethnicity')}"
                        f"({pred.get('fairface_confidence', 0):.2f})"
                    )
                print(
                    f"[{i:02d}/{len(trials)}] {t['subject_id']:8s} {meta['name']:8s} "
                    f"{meta['ethnicity']:8s}  Lab=({lab[0]:.1f},{lab[1]:.1f},{lab[2]:.1f})  "
                    f"ΔE00={de:.2f}{cct_s}{lp_s}",
                    flush=True,
                )

            csv_path = mode_out / "pansor20_chartfree_d65.csv"
            fields = list(rows_out[0].keys()) if rows_out else []
            with csv_path.open("w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=fields)
                w.writeheader()
                w.writerows(rows_out)

            all_de = [r["de00"] for r in rows_out]
            ccts = [r["scr_cct_k"] for r in rows_out if r.get("scr_cct_k") is not None]
            summary: Dict[str, Any] = {
                "n_trials": len(rows_out),
                "half_size": half,
                "hybrid": bool(args.hybrid),
                "roi": args.roi,
                "scr_awb": bool(args.scr_awb),
                "scr_mode": mode,
                "l_sampling": args.l_sampling,
                "fixed_cat_k": float(args.fixed_cat_k) if mode == "preawb_cat" else None,
                "color_projector": str(args.color_projector) if args.color_projector else None,
                "lab_corrector": str(args.lab_corrector) if args.lab_corrector else None,
                "cal_dir": str(args.cal_dir),
                "mean_de00": float(mean(all_de)) if all_de else None,
                "median_de00": float(median(all_de)) if all_de else None,
                "scr_cct_mean_k": float(mean(ccts)) if ccts else None,
                "scr_cct_median_k": float(median(ccts)) if ccts else None,
                "by_ethnicity": {},
                "affine_apple_reference_mean_de00": 5.93,
                "preawb_5500_reference_mean_de00": 5.55,
                "decision_rule": "accept helper only if mean_de00 < 5.93 without FitSkin training",
            }
            for eth, vals in sorted(by_eth.items()):
                summary["by_ethnicity"][eth] = {
                    "n": len(vals),
                    "mean_de00": float(mean(vals)),
                    "median_de00": float(median(vals)),
                }

            (mode_out / "summary.json").write_text(
                json.dumps(summary, indent=2) + "\n", encoding="utf-8"
            )

            if args.emily_tsv and rows_out:
                lines = ["Participant_ID\tTrial\tL*\ta*\tb*\tDeltaE00_vs_FitSkin\t"]
                for r in sorted(rows_out, key=lambda x: (x["participant_id"], x["trial"])):
                    lines.append(
                        f"P{r['participant_id']}\tT{r['trial']}\t"
                        f"{r['pipeline_L']:.1f}\t{r['pipeline_a']:.1f}\t{r['pipeline_b']:.1f}\t"
                        f"{r['de00']:.2f}\t"
                    )
                avg = float(mean(all_de))
                lines.append(f"\t\t\t\t\t{avg}\tAverage deltaE")
                emily_path = mode_out / "table_emily_format.tsv"
                emily_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
                print(f"Wrote {emily_path}")

            print("\n=== Pansor-20 chart-free D65 ===")
            print(
                f"n={len(rows_out)}  mean={summary['mean_de00']:.2f}  median={summary['median_de00']:.2f}  "
                f"roi={args.roi}  scr_mode={mode}  l_sampling={args.l_sampling}"
            )
            print(f"(affine/apple gate ≈ {summary['affine_apple_reference_mean_de00']:.2f})")
            if summary["scr_cct_mean_k"] is not None:
                print(
                    f"SCR CCT mean={summary['scr_cct_mean_k']:.0f}K  "
                    f"median={summary['scr_cct_median_k']:.0f}K"
                )
            print(f"{'Ethnicity':10s} {'n':>4s} {'mean':>8s} {'median':>8s}")
            for eth, st in summary["by_ethnicity"].items():
                print(f"{eth:10s} {st['n']:4d} {st['mean_de00']:8.2f} {st['median_de00']:8.2f}")
            verdict = (
                "KEEP (beats 5.93)"
                if summary["mean_de00"] is not None and summary["mean_de00"] < 5.93
                else "REJECT (≥ 5.93 or empty)"
            )
            if mode == "off":
                verdict = "baseline"
            print(f"Verdict: {verdict}")
            print(f"\nWrote {csv_path}")
            print(f"Wrote {mode_out / 'summary.json'}")
    finally:
        if face_mesh_cm is not None:
            face_mesh_cm.__exit__(None, None, None)


if __name__ == "__main__":
    main()
