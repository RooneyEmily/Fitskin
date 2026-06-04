#!/usr/bin/env python3
"""
Flash / no-flash skin Lab using **physio_skin_lab_monk** tessellation masks (same as
``physio_skin_lab_raw_pr250.py``).

Pipeline:
  1. Load pair; align flash → no-flash (ECC + exposure scaling so flash matches no-flash).
  2. **Primary skin metric (chart-free):** per-channel geometric mean of aligned linear RGB,
     R = sqrt(A ⊙ B), with A = no-flash (linear) and B = exposure-matched aligned flash.
     This proxy is **not** from Lu & Drew (2006); it follows a practical flash/no-flash recipe
     used in the project notes. Lu & Drew is cited only for illuminant estimation (below).
  3. **Secondary (Lu & Drew 2006):** pure-flash F = max(B − A, 0), log-difference χ, nearest
     Planckian ambient CCT; optional Lu-style WB on no-flash. Optional booth illuminant
     (measured CCT/Duv) for a comparison WB path when ``--known-ambient-cct-k`` is set.
  4. Face mesh skin mask; mean cheek L*a*b* (D65); ΔE₀₀ vs FitSkin.

See ``docs/FLASH_NOFLASH_SKIN_METHODS.md`` for paper-ready wording and citations.

Example (JPEG manifest)::

    python flash_no_flash_skin_lab.py \\
        --manifest ../mabl-flash-illumination/data/manifest_noflash_pairs_fitskin.csv \\
        --out-dir ./flash_noflash_skin_output \\
        --write-overlays

Example (iPhone DNG + monochromator/MK350 calibration bundle)::

    python flash_no_flash_skin_lab.py \\
        --data-root "/path/to/RAW Dataset" \\
        --input-mode dng \\
        --out-dir ./flash_noflash_dng_output \\
        --iphone-calibration ./calibration/iphone17pro_camera_color \\
        --exclude-trials P2_T1 \\
        --write-overlays

With SCR-AWB (Zhou 2025) + ISSA skin priors::

    python flash_no_flash_skin_lab.py \\
        --data-root "/path/to/RAW Dataset" \\
        --input-mode dng \\
        --out-dir ./flash_noflash_dng_output \\
        --iphone-calibration ./calibration/iphone17pro_camera_color \\
        --scr-awb \\
        --exclude-trials P2_T1
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np

import physio_skin_lab_monk as psl
import physio_skin_lab_raw_pr250 as pr250
from delta_e_2000 import delta_e_2000
from flash_noflash_face_roi import cheek_mask_from_landmarks

ROOT = Path(__file__).resolve().parent
FLASH_REPO = ROOT.parent / "mabl-flash-illumination"
if FLASH_REPO.is_dir():
    sys.path.insert(0, str(FLASH_REPO))
try:
    from src.align_pair import (  # type: ignore
        AlignResult,
        align_flash_to_noflash,
        align_result_to_bgr_preview,
        estimate_exposure_scale,
    )
    from src.color_linear import linear_rgb_to_bgr_uint8  # type: ignore
    from src.lu2006_ambient import (  # type: ignore
        Lu2006Result,
        estimate_ambient_lu2006,
        planck_rgb_from_cct_duv,
    )
except ImportError as e:
    raise SystemExit(
        f"Need mabl-flash-illumination at {FLASH_REPO} (align_pair, lu2006_ambient). {e}"
    ) from e

# sRGB (D65) linear RGB → XYZ, same matrix as IEC chart scripts
_SRGB_D65_XYZ = np.array(
    [
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ],
    dtype=np.float64,
)
D65_XYZN = np.array([0.95047, 1.0, 1.08883], dtype=np.float64)

# Optional iPhone calibration (monochromator + MK350 flash); set in main() from --iphone-calibration
_CAMERA_RGB_TO_XYZ: Optional[np.ndarray] = None
_CAMERA_RGB_TO_XYZ_AFFINE: Optional[np.ndarray] = None  # (4, 3): [R,G,B,1] @ M → XYZ
_IPHONE_FLASH_CCT_K: Optional[float] = None
_IPHONE_FLASH_RGB: Optional[np.ndarray] = None

_DEFAULT_RAW_DATA_ROOT = Path(
    "/home/mabl-main/Documents/RAW Dataset-20260531T233644Z-3-001/RAW Dataset"
)

_RAW_EXTS = {".dng", ".DNG", ".cr2", ".CR2", ".cr3", ".CR3", ".nef", ".NEF", ".arw", ".ARW"}

_DEFAULT_FITSKIN_SCAN_CSV = Path(
    "/home/mabl-main/Downloads/NoFlash Pairs Dataset-20260524T210745Z-3-001/"
    "Flash-NoFlash Pairs Dataset/scan-sessions-2026-05-20 (1).csv"
)
_DEFAULT_FITSKIN_MAPPING_CSV = FLASH_REPO / "data" / "noflash_pairs_fitskin_mapping.csv"

_MANIFEST_FIELDS = [
    "subject_id",
    "participant",
    "trial",
    "path_noflash",
    "path_flash",
    "scan_session_id",
    "fitskin_cheek_L",
    "fitskin_cheek_a",
    "fitskin_cheek_b",
    "fitskin_forehead_L",
    "fitskin_forehead_a",
    "fitskin_forehead_b",
]

_DEFAULT_MANIFEST = FLASH_REPO / "data" / "manifest_noflash_pairs_fitskin.csv"


def load_fitskin_scan_session_mapping(mapping_csv: Path) -> Dict[str, Tuple[str, str]]:
    """scan_session_id -> (participant label, trial str)."""
    out: Dict[str, Tuple[str, str]] = {}
    with mapping_csv.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            sid = str(row.get("scan_session_id", "")).strip()
            if not sid:
                continue
            out[sid] = (str(row["participant"]).strip(), str(row["trial"]).strip())
    if not out:
        raise SystemExit(f"No scan_session_id rows in {mapping_csv}")
    return out


def load_fitskin_cheek_lookup(
    scan_csv: Path,
    mapping_csv: Path,
) -> Dict[Tuple[str, str], Dict[str, str]]:
    """
    FitSkin cheek Lab keyed by ``(participant, trial)`` — same keys as RAW discovery rows.
    """
    sid_to_pt = load_fitskin_scan_session_mapping(mapping_csv)
    by_session: Dict[str, Dict[str, str]] = {}
    with scan_csv.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            sid = str(row.get("id", "")).strip()
            if sid not in sid_to_pt:
                continue
            cheek_L = row.get("cheek_lab_d65_l_1", "")
            cheek_a = row.get("cheek_lab_d65_a_1", "")
            cheek_b = row.get("cheek_lab_d65_b_1", "")
            if not all(str(x).strip() for x in (cheek_L, cheek_a, cheek_b)):
                continue
            by_session[sid] = {
                "scan_session_id": sid,
                "fitskin_cheek_L": cheek_L,
                "fitskin_cheek_a": cheek_a,
                "fitskin_cheek_b": cheek_b,
                "fitskin_forehead_L": row.get("forehead_lab_d65_l_1", ""),
                "fitskin_forehead_a": row.get("forehead_lab_d65_a_1", ""),
                "fitskin_forehead_b": row.get("forehead_lab_d65_b_1", ""),
            }

    out: Dict[Tuple[str, str], Dict[str, str]] = {}
    for sid, (participant, trial) in sid_to_pt.items():
        if sid in by_session:
            out[(participant, trial)] = by_session[sid]
    if not out:
        raise SystemExit(
            f"No overlapping FitSkin rows between {scan_csv} and {mapping_csv}"
        )
    return out


def attach_fitskin_to_manifest_rows(
    rows: List[Dict[str, str]],
    lookup: Dict[Tuple[str, str], Dict[str, str]],
) -> List[Dict[str, str]]:
    """Merge FitSkin cheek Lab into rows by participant + trial."""
    merged: List[Dict[str, str]] = []
    for row in rows:
        key = (str(row["participant"]).strip(), str(row["trial"]).strip())
        rec = dict(row)
        if key in lookup:
            rec.update(lookup[key])
        merged.append(rec)
    return merged


def _to_gray01(linear_rgb: np.ndarray) -> np.ndarray:
    g = 0.2126 * linear_rgb[..., 0] + 0.7152 * linear_rgb[..., 1] + 0.0722 * linear_rgb[..., 2]
    return np.clip(g, 0.0, 1.0).astype(np.float32)


def estimate_exposure_scale_masked(
    noflash: np.ndarray,
    flash: np.ndarray,
    mask: np.ndarray,
    *,
    max_sat: float = 0.98,
    min_luma: float = 0.05,
) -> float:
    """Median luma ratio on ``mask`` (falls back to global if too few pixels)."""
    g0 = _to_gray01(noflash)
    g1 = _to_gray01(flash)
    m = (
        (np.asarray(mask) > 0)
        & (g0 > min_luma)
        & (g0 < max_sat)
        & (g1 > min_luma)
        & (g1 < max_sat)
    )
    if int(m.sum()) < 100:
        return estimate_exposure_scale(noflash, flash, max_sat=max_sat, min_luma=min_luma)
    r = np.median(g0[m] / np.maximum(g1[m], 1e-6))
    return float(np.clip(r, 0.25, 4.0))


def divide_by_illuminant_linear(
    linear_rgb: np.ndarray,
    illuminant_rgb: np.ndarray,
) -> np.ndarray:
    """Diagonal WB (Lu-style): divide by illuminant normalized to unit median."""
    e = np.maximum(np.asarray(illuminant_rgb, dtype=np.float64).reshape(1, 1, 3), 1e-8)
    e = e / np.median(e)
    return np.asarray(linear_rgb, dtype=np.float64) / e


def align_flash_to_noflash_linear(
    noflash_lin: np.ndarray,
    flash_lin: np.ndarray,
    *,
    motion_ecc: str = "euclidean",
    exposure_mask: Optional[np.ndarray] = None,
    skip_exposure: bool = False,
) -> AlignResult:
    """ECC align flash → no-flash on linear RGB ~[0, 1] (normalized camera RAW)."""
    nf_lin = np.asarray(noflash_lin, dtype=np.float64)
    fl_lin = np.asarray(flash_lin, dtype=np.float64)
    g0 = _to_gray01(nf_lin)
    g1 = _to_gray01(fl_lin)

    if motion_ecc == "affine":
        warp_mode = cv2.MOTION_AFFINE
        warp = np.eye(2, 3, dtype=np.float32)
    else:
        warp_mode = cv2.MOTION_EUCLIDEAN
        warp = np.eye(2, 3, dtype=np.float32)

    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 500, 1e-6)
    try:
        cc, warp = cv2.findTransformECC(g0, g1, warp, warp_mode, criteria, None, 5)
    except cv2.error:
        cc, warp = 0.0, np.eye(2, 3, dtype=np.float32)

    h, w = g0.shape
    fl_warp = cv2.warpAffine(
        fl_lin.astype(np.float32),
        warp,
        (w, h),
        flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_REPLICATE,
    ).astype(np.float64)

    if skip_exposure:
        scale = 1.0
        fl_scaled = fl_warp
    elif exposure_mask is not None:
        scale = estimate_exposure_scale_masked(nf_lin, fl_warp, exposure_mask)
        fl_scaled = np.clip(fl_warp * scale, 0.0, None)
    else:
        scale = estimate_exposure_scale(nf_lin, fl_warp)
        fl_scaled = np.clip(fl_warp * scale, 0.0, None)

    return AlignResult(
        flash_aligned_linear=fl_scaled,
        noflash_linear=nf_lin,
        warp_matrix=warp,
        exposure_scale=scale,
        ecc_cc=float(cc),
    )


def raw_to_linear_u01(path: Path, *, half_size: int, use_camera_wb: bool) -> np.ndarray:
    """Demosaic DNG/CR2 → linear camera RGB, percentile-normalized to ~[0, 1]."""
    rgb_lin = pr250.read_raw_linear_rgb(path, half_size=half_size, use_camera_wb=use_camera_wb)
    scale = float(np.percentile(rgb_lin, 99.5))
    return np.clip(rgb_lin / max(scale, 1e-6), 0.0, 1.0)


def raw_to_linear_u01_skin_percentile(
    path: Path,
    face_mesh: Any,
    *,
    half_size: int,
    use_camera_wb: bool,
    skin_triangulation: str,
    skin_exclusion_dilate_iod_fraction: float,
    percentile: float = 99.5,
) -> np.ndarray:
    """u01 scale from high percentile of linear RGB inside face mesh (not full frame)."""
    rgb_lin = pr250.read_raw_linear_rgb(path, half_size=half_size, use_camera_wb=use_camera_wb)
    bgr = pr250.linear_rgb_to_preview_bgr(rgb_lin)
    mask, *_rest = skin_mask_from_bgr(
        bgr,
        face_mesh,
        skin_triangulation=skin_triangulation,
        skin_exclusion_dilate_iod_fraction=skin_exclusion_dilate_iod_fraction,
        build_cheek_mask=False,
    )
    m = mask > 0
    if int(m.sum()) < 50:
        return raw_to_linear_u01(path, half_size=half_size, use_camera_wb=use_camera_wb)
    vals = np.asarray(rgb_lin[m], dtype=np.float64)
    scale = float(np.percentile(vals, percentile))
    return np.clip(rgb_lin / max(scale, 1e-6), 0.0, 1.0)


def discover_raw_pairs(data_root: Path) -> List[Dict[str, str]]:
    """Find *NoFlash* + *Flash* RAW files in the same trial folder."""
    root = data_root.expanduser().resolve()
    if not root.is_dir():
        raise SystemExit(f"--data-root is not a directory: {root}")

    rows: List[Dict[str, str]] = []
    for nf_path in sorted(root.rglob("*")):
        if not nf_path.is_file() or nf_path.suffix not in _RAW_EXTS:
            continue
        if "noflash" not in nf_path.stem.lower():
            continue
        trial_dir = nf_path.parent
        flash_paths = [
            p
            for p in trial_dir.iterdir()
            if p.is_file()
            and p.suffix in _RAW_EXTS
            and "flash" in p.stem.lower()
            and "noflash" not in p.stem.lower()
        ]
        if not flash_paths:
            continue
        fl_path = sorted(flash_paths)[0]
        participant = trial_dir.parent.name
        trial_name = trial_dir.name
        pm = re.search(r"(\d+)", participant)
        tm = re.search(r"(\d+)", trial_name)
        pid = pm.group(1) if pm else participant.replace(" ", "_")
        trial_n = tm.group(1) if tm else "0"
        sid = f"P{pid}_T{trial_n}"
        rows.append(
            {
                "subject_id": sid,
                "participant": participant,
                "trial": trial_n,
                "path_noflash": str(nf_path.resolve()),
                "path_flash": str(fl_path.resolve()),
            }
        )

    if not rows:
        raise SystemExit(f"No Flash/NoFlash RAW pairs under {root}")
    rows.sort(key=lambda r: (r["subject_id"], r["path_noflash"]))
    return rows


def write_manifest_csv(rows: List[Dict[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=_MANIFEST_FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


@contextlib.contextmanager
def _silence_stderr() -> Iterator[None]:
    stderr_fd = sys.stderr.fileno()
    saved_fd = os.dup(stderr_fd)
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull_fd, stderr_fd)
        yield
    finally:
        os.dup2(saved_fd, stderr_fd)
        os.close(saved_fd)
        os.close(devnull_fd)


def _subject_key(row: dict) -> str:
    if row.get("subject_id"):
        return str(row["subject_id"])
    part = row.get("participant", "unknown")
    m = re.search(r"(\d+)", part)
    p = f"P{m.group(1)}" if m else part.replace(" ", "_")
    return f"{p}_T{row.get('trial', 0)}"


def linear_rgb_to_xyz_d65(rgb_lin: np.ndarray) -> np.ndarray:
    """(H,W,3) linear RGB → XYZ (D65 white; matrix from calibration bundle or sRGB default)."""
    x = np.asarray(rgb_lin, dtype=np.float64)
    if _CAMERA_RGB_TO_XYZ_AFFINE is not None:
        shp = x.shape
        flat = x.reshape(-1, 3)
        aug = np.column_stack([flat, np.ones((flat.shape[0], 1), dtype=np.float64)])
        return (aug @ _CAMERA_RGB_TO_XYZ_AFFINE).reshape(shp)
    M = _CAMERA_RGB_TO_XYZ if _CAMERA_RGB_TO_XYZ is not None else _SRGB_D65_XYZ
    return x @ M.T


def _f_xyz_to_lab_ratio(t: np.ndarray) -> np.ndarray:
    t = np.asarray(t, dtype=np.float64)
    out = np.empty_like(t, dtype=np.float64)
    m = t <= 216.0 / 24389.0
    out[m] = ((24389.0 / 27.0) * t[m] + 16.0) / 116.0
    out[~m] = np.cbrt(t[~m])
    return out


def xyz_to_lab_batch(xyz: np.ndarray, xyzn: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    w = np.asarray(xyzn, dtype=np.float64).reshape(1, 3)
    r = np.asarray(xyz, dtype=np.float64) / np.maximum(w, 1e-12)
    fx = _f_xyz_to_lab_ratio(r[:, 0])
    fy = _f_xyz_to_lab_ratio(r[:, 1])
    fz = _f_xyz_to_lab_ratio(r[:, 2])
    L = 116.0 * fy - 16.0
    a = 500.0 * (fx - fy)
    b = 200.0 * (fy - fz)
    return L, a, b


def estimate_reflectance_linear(
    noflash_lin: np.ndarray,
    flash_aligned_lin: np.ndarray,
    *,
    fusion: str = "geometric",
) -> np.ndarray:
    """
    Chart-free reflectance proxy on aligned linear RGB (not Lu & Drew 2006).

    ``geometric``: sqrt(A ⊙ B); ``log``: exp(0.5*(log A + log B)); ``arithmetic``: (A+B)/2.
    """
    a = np.clip(np.asarray(noflash_lin, dtype=np.float64), 1e-8, None)
    b = np.clip(np.asarray(flash_aligned_lin, dtype=np.float64), 1e-8, None)
    mode = (fusion or "geometric").strip().lower()
    if mode == "geometric":
        return np.sqrt(a * b)
    if mode == "log":
        return np.exp(0.5 * (np.log(a) + np.log(b)))
    if mode == "arithmetic":
        return 0.5 * (a + b)
    raise ValueError(f"Unknown reflectance fusion: {fusion!r}")


@dataclass
class SkinLabStats:
    L: float
    a: float
    b: float
    C: float
    h_deg: float
    n_pixels: int
    n_mesh_mask: int


def mean_lab_reflectance_linear(
    albedo_lin: np.ndarray,
    mask: np.ndarray,
    *,
    l_star_trim_lo: float,
    l_star_trim_hi: float,
    a_star_trim_lo: float,
    a_star_trim_hi: float,
    b_star_trim_lo: float,
    b_star_trim_hi: float,
    skin_min_chroma_ab: float,
    xyz_scene_white: Optional[np.ndarray] = None,
) -> SkinLabStats:
    """Lab (D65) from linear reflectance map + binary mask; optional Bradford CAT before Lab."""
    m = mask > 0
    n_mesh = int(np.count_nonzero(m))
    if not np.any(m):
        return SkinLabStats(float("nan"), float("nan"), float("nan"), float("nan"), float("nan"), 0, n_mesh)
    xyz = linear_rgb_to_xyz_d65(albedo_lin[m])
    if xyz_scene_white is not None:
        cat = pr250.bradford_cat_matrix(
            np.asarray(xyz_scene_white, dtype=np.float64), D65_XYZN
        )
        xyz = xyz @ cat.T
    L, a, b = xyz_to_lab_batch(xyz, D65_XYZN)
    keep, _sel_ch, _cr, _ctr, _ce, _chte = psl.skin_lab_trim_selection(
        L,
        a,
        b,
        l_star_trim_lo=l_star_trim_lo,
        l_star_trim_hi=l_star_trim_hi,
        a_star_trim_lo=a_star_trim_lo,
        a_star_trim_hi=a_star_trim_hi,
        b_star_trim_lo=b_star_trim_lo,
        b_star_trim_hi=b_star_trim_hi,
        min_chroma_ab=skin_min_chroma_ab,
    )
    if not np.any(keep):
        return SkinLabStats(float("nan"), float("nan"), float("nan"), float("nan"), float("nan"), 0, n_mesh)
    Lk, ak, bk = L[keep], a[keep], b[keep]
    mean_L = float(np.mean(Lk))
    mean_a = float(np.mean(ak))
    mean_b = float(np.mean(bk))
    mean_C = float(np.hypot(mean_a, mean_b))
    mean_h = float(np.degrees(np.arctan2(mean_b, mean_a)))
    return SkinLabStats(mean_L, mean_a, mean_b, mean_C, mean_h, int(keep.sum()), n_mesh)


def skin_mask_from_bgr(
    bgr: np.ndarray,
    face_mesh: Any,
    *,
    skin_triangulation: str,
    skin_exclusion_dilate_iod_fraction: float,
    build_cheek_mask: bool = False,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    Optional[np.ndarray],
    np.ndarray,
    Optional[np.ndarray],
    Optional[np.ndarray],
]:
    """Same call pattern as ``physio_skin_lab_raw_pr250.process_one_raw``."""
    h, w = bgr.shape[:2]
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb.flags.writeable = False
    res = face_mesh.process(rgb)
    if not res.multi_face_landmarks:
        raise ValueError("No face detected")
    lm = res.multi_face_landmarks[0].landmark
    mesh_mask, oval, kept, excl, mesh_xy = psl.build_skin_mask_from_mesh(
        h,
        w,
        lm,
        skin_triangulation=skin_triangulation,
        exclusion_dilate_iod_fraction=skin_exclusion_dilate_iod_fraction,
    )
    cheek_mask = (
        cheek_mask_from_landmarks(h, w, lm, mesh_mask) if build_cheek_mask else None
    )
    return mesh_mask, oval, kept, excl, mesh_xy, cheek_mask


def mean_lab_on_bgr(
    bgr: np.ndarray,
    mask: np.ndarray,
    *,
    l_star_trim_lo: float,
    l_star_trim_hi: float,
    a_star_trim_lo: float,
    a_star_trim_hi: float,
    b_star_trim_lo: float,
    b_star_trim_hi: float,
    skin_min_chroma_ab: float,
) -> SkinLabStats:
    L, a, b, npx, _, _, _, _, _, _, _, _, _, _ = psl.mean_lab_masked(
        bgr,
        mask,
        l_star_trim_lo=l_star_trim_lo,
        l_star_trim_hi=l_star_trim_hi,
        a_star_trim_lo=a_star_trim_lo,
        a_star_trim_hi=a_star_trim_hi,
        b_star_trim_lo=b_star_trim_lo,
        b_star_trim_hi=b_star_trim_hi,
        min_chroma_ab=skin_min_chroma_ab,
    )
    n_mesh = int(np.count_nonzero(mask > 0))
    if npx <= 0:
        return SkinLabStats(float("nan"), float("nan"), float("nan"), float("nan"), float("nan"), 0, n_mesh)
    C = float(np.hypot(a, b))
    h_deg = float(np.degrees(np.arctan2(b, a)))
    return SkinLabStats(L, a, b, C, h_deg, int(npx), n_mesh)


def _de00_cheek(
    L: float, a: float, b: float, Lf: float, af: float, bf: float
) -> float:
    if not all(np.isfinite([L, a, b, Lf, af, bf])):
        return float("nan")
    lab_p = np.array([[[L, a, b]]], dtype=np.float64)
    lab_f = np.array([[[Lf, af, bf]]], dtype=np.float64)
    return float(delta_e_2000(lab_p, lab_f)[0, 0])


def _resize_bgr_max_width(bgr: np.ndarray, max_width: int) -> np.ndarray:
    """Downscale one frame if wider than ``max_width`` (0 = no resize)."""
    if max_width <= 0:
        return bgr
    h, w = bgr.shape[:2]
    if w <= max_width:
        return bgr
    scale = max_width / float(w)
    return cv2.resize(bgr, (max_width, int(round(h * scale))), interpolation=cv2.INTER_AREA)


def _resize_linear_max_width(rgb_lin: np.ndarray, max_width: int) -> np.ndarray:
    if max_width <= 0:
        return rgb_lin
    h, w = rgb_lin.shape[:2]
    if w <= max_width:
        return rgb_lin
    scale = max_width / float(w)
    nh, nw = int(round(h * scale)), max_width
    return cv2.resize(rgb_lin, (nw, nh), interpolation=cv2.INTER_AREA)


_METHODS_VS_FITSKIN = (
    ("reflectance", "reflectance_cheek_de00"),
    ("scr_awb_wb", "scr_awb_wb_cheek_de00"),
    ("lu_booth_wb", "lu_booth_wb_cheek_de00"),
    ("lu_wb", "lu_wb_cheek_de00"),
    ("noflash", "noflash_cheek_de00"),
    ("flash_aligned", "flash_aligned_cheek_de00"),
)


def _estimate_fitskin_lightness_gains(pass1_rows: List[Dict[str, Any]]) -> Dict[str, float]:
    """Per-participant linear gain so median reflectance L* matches FitSkin (pilot calibration)."""
    from exposure_anchor import participant_key

    lr: Dict[str, List[float]] = {}
    lf: Dict[str, List[float]] = {}
    for row in pass1_rows:
        Lr = float(row.get("reflectance_L", np.nan))
        Lf = float(row.get("fitskin_cheek_L", np.nan))
        if not (np.isfinite(Lr) and np.isfinite(Lf) and Lr > 1.0):
            continue
        pk = participant_key(str(row.get("subject_id", "")), str(row.get("participant", "")))
        lr.setdefault(pk, []).append(Lr)
        lf.setdefault(pk, []).append(Lf)
    gains: Dict[str, float] = {}
    for pk, r_vals in lr.items():
        med_r = float(np.median(r_vals))
        med_f = float(np.median(lf[pk]))
        ratio = med_f / med_r
        # Boost only when reflectance L* is below FitSkin (never darken toward scanner).
        gains[pk] = float(np.clip(ratio, 1.0, 2.0)) if ratio > 1.02 else 1.0
    return gains


def _annotate_best_method_vs_fitskin(rows: List[Dict[str, Any]]) -> None:
    for row in rows:
        best_name = ""
        best_de = float("inf")
        for name, col in _METHODS_VS_FITSKIN:
            try:
                de = float(row[col])
            except (KeyError, TypeError, ValueError):
                continue
            if np.isfinite(de) and de < best_de:
                best_de = de
                best_name = name
        row["best_method_vs_fitskin"] = best_name or ""
        row["best_method_de00"] = best_de if np.isfinite(best_de) else float("nan")


def load_lu_sharpening_matrix(path: Optional[Path]) -> Optional[np.ndarray]:
    """Optional 3×3 Lu spectral-sharpening matrix (``None`` = identity / no sharpening)."""
    if path is None:
        return None
    p = path.expanduser()
    if not p.is_file():
        raise SystemExit(f"--lu-sharpening-matrix not found: {p}")
    M = np.load(p)
    if M.shape != (3, 3):
        raise SystemExit(f"--lu-sharpening-matrix must be 3×3, got {M.shape}")
    return np.asarray(M, dtype=np.float64)


def process_flash_pair(
    face_mesh: Any,
    *,
    noflash_bgr: Optional[np.ndarray] = None,
    flash_bgr: Optional[np.ndarray] = None,
    noflash_lin_u01: Optional[np.ndarray] = None,
    flash_lin_u01: Optional[np.ndarray] = None,
    max_align_width: int = 1600,
    motion_ecc: str,
    skin_triangulation: str,
    skin_exclusion_dilate_iod_fraction: float,
    l_star_trim_lo: float,
    l_star_trim_hi: float,
    a_star_trim_lo: float,
    a_star_trim_hi: float,
    b_star_trim_lo: float,
    b_star_trim_hi: float,
    skin_min_chroma_ab: float,
    input_mode: str = "jpeg",
    flash_cct_k: Optional[float] = None,
    lu_sharpening_matrix: Optional[np.ndarray] = None,
    known_ambient_cct_k: Optional[float] = None,
    known_ambient_duv: float = 0.0,
    measured_flash_cct_k: Optional[float] = None,
    flash_rgb_measured: Optional[np.ndarray] = None,
    scr_awb_prior_name: Optional[str] = None,
    scr_awb_spectral_sensitivity: Optional[np.ndarray] = None,
    scr_awb_wavelengths_nm: Optional[np.ndarray] = None,
    reflectance_exposure_scale: Optional[float] = None,
    reflectance_fitskin_lightness_gain: Optional[float] = None,
    cheek_roi: bool = False,
    exposure_scale_skin_mask: bool = False,
    reflectance_pre_wb: str = "none",
    reflectance_fusion: str = "geometric",
    reflectance_cat: str = "none",
) -> Dict[str, Any]:
    use_skin_exposure = bool(exposure_scale_skin_mask)
    if noflash_lin_u01 is not None and flash_lin_u01 is not None:
        nf_work = _resize_linear_max_width(noflash_lin_u01, max_align_width)
        fl_work = _resize_linear_max_width(flash_lin_u01, max_align_width)
        align = align_flash_to_noflash_linear(
            nf_work,
            fl_work,
            motion_ecc=motion_ecc,
            skip_exposure=use_skin_exposure,
        )
        nf_bgr = pr250.linear_rgb_to_preview_bgr(align.noflash_linear)
        fl_bgr = pr250.linear_rgb_to_preview_bgr(align.flash_aligned_linear)
    elif noflash_bgr is not None and flash_bgr is not None:
        nf_work = _resize_bgr_max_width(noflash_bgr, max_align_width)
        fl_work = _resize_bgr_max_width(flash_bgr, max_align_width)
        align = align_flash_to_noflash(nf_work, fl_work, motion_ecc=motion_ecc)
        nf_bgr, fl_bgr = align_result_to_bgr_preview(align)
        use_skin_exposure = False
    else:
        raise ValueError("Provide either BGR pair or linear-u01 RAW pair")

    mask_nf, oval, kept, excl, mesh_xy, cheek_mask = skin_mask_from_bgr(
        nf_bgr,
        face_mesh,
        skin_triangulation=skin_triangulation,
        skin_exclusion_dilate_iod_fraction=skin_exclusion_dilate_iod_fraction,
        build_cheek_mask=cheek_roi or use_skin_exposure,
    )

    align_exposure_scale_skin: Optional[float] = None
    if use_skin_exposure and noflash_lin_u01 is not None:
        exp_mask = cheek_mask if cheek_mask is not None else mask_nf
        align_exposure_scale_skin = estimate_exposure_scale_masked(
            align.noflash_linear,
            align.flash_aligned_linear,
            exp_mask,
        )
        fl_scaled = np.clip(
            align.flash_aligned_linear * align_exposure_scale_skin, 0.0, None
        )
        align = AlignResult(
            flash_aligned_linear=fl_scaled,
            noflash_linear=align.noflash_linear,
            warp_matrix=align.warp_matrix,
            exposure_scale=float(align_exposure_scale_skin),
            ecc_cc=align.ecc_cc,
        )
        fl_bgr = pr250.linear_rgb_to_preview_bgr(align.flash_aligned_linear)

    lab_mask = cheek_mask if (cheek_roi and cheek_mask is not None) else mask_nf

    nf_refl = align.noflash_linear
    fl_refl = align.flash_aligned_linear
    if reflectance_pre_wb == "booth":
        if known_ambient_cct_k is None or float(known_ambient_cct_k) <= 0.0:
            raise ValueError("reflectance_pre_wb=booth requires known_ambient_cct_k > 0")
        e_booth = planck_rgb_from_cct_duv(float(known_ambient_cct_k), float(known_ambient_duv))
        nf_refl = divide_by_illuminant_linear(nf_refl, e_booth)
        fl_refl = divide_by_illuminant_linear(fl_refl, e_booth)

    # --- Primary: chart-free reflectance; B = exposure-matched aligned flash ---
    albedo = estimate_reflectance_linear(
        nf_refl, fl_refl, fusion=reflectance_fusion
    )
    total_reflectance_scale = 1.0
    if reflectance_exposure_scale is not None and reflectance_exposure_scale > 0.0:
        total_reflectance_scale *= float(reflectance_exposure_scale)
    if reflectance_fitskin_lightness_gain is not None and reflectance_fitskin_lightness_gain > 0.0:
        total_reflectance_scale *= float(reflectance_fitskin_lightness_gain)
    if total_reflectance_scale != 1.0:
        albedo = np.clip(albedo * total_reflectance_scale, 0.0, 1.0).astype(np.float64)
    albedo_bgr = linear_rgb_to_bgr_uint8(
        np.clip(albedo / (np.percentile(albedo, 99.0) + 1e-6), 0.0, 1.0)
    )

    # --- Secondary: Lu & Drew 2006 ambient CCT (+ optional booth-known illuminant WB) ---
    lu: Lu2006Result = estimate_ambient_lu2006(
        align,
        cct_flash_k=flash_cct_k,
        measured_flash_cct_k=measured_flash_cct_k,
        flash_rgb_measured=flash_rgb_measured,
        sharpening_matrix=lu_sharpening_matrix,
        known_ambient_cct_k=known_ambient_cct_k,
        known_ambient_duv=known_ambient_duv,
    )
    wb_bgr = linear_rgb_to_bgr_uint8(lu.white_balanced_lu)
    wb_booth_bgr: Optional[np.ndarray] = None
    if lu.white_balanced_booth is not None:
        wb_booth_bgr = linear_rgb_to_bgr_uint8(lu.white_balanced_booth)

    xyz_scene_white: Optional[np.ndarray] = None
    cat_mode = (reflectance_cat or "none").strip().lower()
    if cat_mode == "booth":
        if known_ambient_cct_k is None or float(known_ambient_cct_k) <= 0.0:
            raise ValueError("reflectance_cat=booth requires known_ambient_cct_k > 0")
        from flash_noflash_spectral import planck_xyz_y1

        xyz_scene_white = planck_xyz_y1(float(known_ambient_cct_k), float(known_ambient_duv))
    elif cat_mode == "lu":
        rgb_p = np.maximum(np.asarray(lu.ambient_rgb_planck, dtype=np.float64), 1e-8)
        rgb_p = rgb_p / np.median(rgb_p)
        xyz_scene_white = rgb_p @ _SRGB_D65_XYZ.T
        xyz_scene_white = xyz_scene_white / max(float(xyz_scene_white[1]), 1e-8)

    stats: Dict[str, Any] = {
        "input_mode": input_mode,
        "primary_skin_metric": "reflectance",
        "lab_roi": "cheek" if cheek_roi else "mesh",
        "ambient_cct_k": float(lu.ambient_cct_k),
        "ambient_cct_estimated_k": float(lu.ambient_cct_estimated_k),
        "ambient_cct_source": str(lu.ambient_cct_source),
        "ambient_duv": float(lu.ambient_duv),
        "flash_cct_k": float(lu.flash_cct_k),
        "flash_cct_source": str(lu.flash_cct_source),
        "lu_spectral_sharpening": str(lu.spectral_sharpening),
        "lu_chi_distance": float(lu.chi_distance),
        "align_ecc_cc": float(align.ecc_cc),
        "align_exposure_scale": float(align.exposure_scale),
        "align_exposure_scale_skin": (
            float(align_exposure_scale_skin)
            if align_exposure_scale_skin is not None
            else None
        ),
        "reflectance_pre_wb": reflectance_pre_wb if reflectance_pre_wb != "none" else None,
        "reflectance_fusion": reflectance_fusion,
        "reflectance_cat": cat_mode if cat_mode != "none" else None,
        "n_lu_valid_pixels": int(lu.n_valid_pixels),
        "reflectance_exposure_scale": (
            float(reflectance_exposure_scale)
            if reflectance_exposure_scale is not None
            else None
        ),
        "reflectance_fitskin_lightness_gain": (
            float(reflectance_fitskin_lightness_gain)
            if reflectance_fitskin_lightness_gain is not None
            else None
        ),
        "reflectance_total_scale": (
            float(total_reflectance_scale) if total_reflectance_scale != 1.0 else None
        ),
    }

    method_list: List[Tuple[str, Optional[np.ndarray], Optional[np.ndarray]]] = [
        ("reflectance", albedo_bgr, albedo),
        ("noflash", nf_bgr, None),
        ("flash_aligned", fl_bgr, None),
        ("lu_wb", wb_bgr, None),
    ]
    if wb_booth_bgr is not None:
        method_list.append(("lu_booth_wb", wb_booth_bgr, None))

    scr_wb_bgr: Optional[np.ndarray] = None
    if (
        scr_awb_prior_name
        and scr_awb_spectral_sensitivity is not None
        and scr_awb_wavelengths_nm is not None
    ):
        from scr_awb import estimate_scr_awb, load_scr_awb_prior

        prior = load_scr_awb_prior(scr_awb_prior_name)
        scr = estimate_scr_awb(
            align.noflash_linear,
            mask_nf,
            spectral_sensitivity_rgb=scr_awb_spectral_sensitivity,
            wavelengths_nm=scr_awb_wavelengths_nm,
            skin_prior=prior,
            known_ambient_cct_k=known_ambient_cct_k,
            known_ambient_duv=known_ambient_duv,
        )
        if known_ambient_cct_k is not None:
            stats["scr_awb_wb_illuminant_source"] = "booth_known_cct"
        else:
            stats["scr_awb_wb_illuminant_source"] = "solved_basis"
        scr_wb_bgr = linear_rgb_to_bgr_uint8(scr.white_balanced_linear)
        stats["scr_awb_prior"] = prior.name
        stats["scr_awb_ambient_cct_k"] = float(scr.ambient_cct_k)
        stats["scr_awb_residual"] = float(scr.residual_norm)
        stats["scr_awb_alpha"] = scr.alpha.tolist()
        stats["scr_awb_illuminant_rgb"] = scr.illuminant_rgb.tolist()
        stats["scr_awb_skin_rgb_median"] = scr.skin_rgb_median.tolist()
        method_list.append(("scr_awb_wb", scr_wb_bgr, None))

    for key, bgr, lin_src in method_list:
        roi_mask = lab_mask if (lin_src is not None and key == "reflectance") else mask_nf
        if lin_src is not None:
            lab = mean_lab_reflectance_linear(
                lin_src,
                roi_mask,
                l_star_trim_lo=l_star_trim_lo,
                l_star_trim_hi=l_star_trim_hi,
                a_star_trim_lo=a_star_trim_lo,
                a_star_trim_hi=a_star_trim_hi,
                b_star_trim_lo=b_star_trim_lo,
                b_star_trim_hi=b_star_trim_hi,
                skin_min_chroma_ab=skin_min_chroma_ab,
                xyz_scene_white=xyz_scene_white if key == "reflectance" else None,
            )
        else:
            lab = mean_lab_on_bgr(
                bgr,
                mask_nf,
                l_star_trim_lo=l_star_trim_lo,
                l_star_trim_hi=l_star_trim_hi,
                a_star_trim_lo=a_star_trim_lo,
                a_star_trim_hi=a_star_trim_hi,
                b_star_trim_lo=b_star_trim_lo,
                b_star_trim_hi=b_star_trim_hi,
                skin_min_chroma_ab=skin_min_chroma_ab,
            )
        stats[f"{key}_L"] = lab.L
        stats[f"{key}_a"] = lab.a
        stats[f"{key}_b"] = lab.b
        stats[f"{key}_C"] = lab.C
        stats[f"{key}_h_deg"] = lab.h_deg
        stats[f"{key}_n_pixels"] = lab.n_pixels
        stats[f"{key}_n_mesh_mask"] = lab.n_mesh_mask

    stats["_debug"] = {
        "mask": mask_nf,
        "cheek_mask": cheek_mask,
        "lab_mask": lab_mask,
        "oval_pts": oval,
        "kept_tris": kept,
        "excl_dil": excl,
        "mesh_xy": mesh_xy,
        "noflash_bgr": nf_bgr,
        "flash_bgr": fl_bgr,
        "wb_bgr": wb_bgr,
        "albedo_bgr": albedo_bgr,
        "scr_wb_bgr": scr_wb_bgr,
    }
    return stats


def main() -> None:
    ap = argparse.ArgumentParser(description="Flash/no-flash skin Lab (physio tessellation mask).")
    ap.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="CSV with path_noflash, path_flash (JPEG pilot). Omit when using --data-root.",
    )
    ap.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help=f"Scan for *Flash*/*NoFlash* RAW pairs (DNG, CR2, …). Example: {_DEFAULT_RAW_DATA_ROOT}",
    )
    ap.add_argument(
        "--input-mode",
        choices=("jpeg", "dng", "auto"),
        default="auto",
        help="jpeg=8-bit files; dng=RAW via rawpy; auto=by file extension.",
    )
    ap.add_argument(
        "--raw-half-size",
        type=int,
        default=1,
        help="rawpy half_size for DNG/CR2 demosaic (0=full; 1=default).",
    )
    ap.add_argument(
        "--raw-camera-wb",
        action="store_true",
        help="Use embedded camera white balance for RAW (default: unity WB).",
    )
    ap.add_argument(
        "--fitskin-scan-csv",
        type=Path,
        default=_DEFAULT_FITSKIN_SCAN_CSV,
        help="FitSkin scan-sessions export (cheek_lab_d65_* columns).",
    )
    ap.add_argument(
        "--fitskin-mapping-csv",
        type=Path,
        default=_DEFAULT_FITSKIN_MAPPING_CSV,
        help="Maps scan_session_id -> participant + trial (noflash_pairs_fitskin_mapping.csv).",
    )
    ap.add_argument(
        "--no-fitskin",
        action="store_true",
        help="Skip FitSkin merge (no ΔE00 vs scanner).",
    )
    ap.add_argument("--out-dir", type=Path, default=ROOT / "flash_noflash_skin_output")
    ap.add_argument("--motion", choices=("euclidean", "affine"), default="euclidean")
    ap.add_argument(
        "--max-align-width",
        type=int,
        default=1600,
        help="Downscale both frames before ECC/Lu (0 = full resolution; 1600 is fast on iPhone 12MP).",
    )
    ap.add_argument("--skin-exclusion-dilate-iod", type=float, default=0.12)
    ap.add_argument(
        "--skin-triangulation",
        choices=("tessellation", "oval_delaunay"),
        default="tessellation",
        help="Same as physio_skin_lab_raw_pr250 (default: tessellation).",
    )
    ap.add_argument("--skin-l-star-trim-lo", type=float, default=0.05)
    ap.add_argument("--skin-l-star-trim-hi", type=float, default=0.05)
    ap.add_argument(
        "--skin-a-star-trim-lo",
        type=float,
        default=0.05,
        help="Drop skin pixels with a* below this quantile (0=off). Default 0.05 matches L* trim.",
    )
    ap.add_argument("--skin-a-star-trim-hi", type=float, default=0.05)
    ap.add_argument("--skin-b-star-trim-lo", type=float, default=0.05)
    ap.add_argument("--skin-b-star-trim-hi", type=float, default=0.05)
    ap.add_argument("--skin-min-chroma-ab", type=float, default=2.0)
    ap.add_argument("--write-overlays", action="store_true")
    ap.add_argument("--overlay-max-width", type=int, default=1600)
    ap.add_argument("--exclude-trials", type=str, default="P2_T1", help="Comma-separated subject_id keys to skip.")
    ap.add_argument(
        "--flash-cct-k",
        type=float,
        default=0.0,
        help="Fixed flash Planck CCT (K) for Lu χ locus. <=0 = auto-estimate from pure-flash neutrals "
        f"(fallback {5500.0:.0f} K if auto fails).",
    )
    ap.add_argument(
        "--lu-sharpening-matrix",
        type=Path,
        default=None,
        help="Optional 3×3 .npy spectral-sharpening matrix (Lu et al.). Omit = no sharpening (identity).",
    )
    ap.add_argument(
        "--known-ambient-cct-k",
        type=float,
        default=0.0,
        help="Booth/measured ambient CCT (K) for comparison WB (e.g. 6546). <=0 = skip.",
    )
    ap.add_argument(
        "--known-ambient-duv",
        type=float,
        default=0.0,
        help="Duv for --known-ambient-cct-k (e.g. 0.0017).",
    )
    ap.add_argument(
        "--iphone-calibration",
        type=Path,
        default=None,
        help=(
            "Dir with iphone_calibration_bundle.json from build_iphone_calibration_bundle.py "
            "(monochromator camera matrix + MK350 flash SPD)."
        ),
    )
    ap.add_argument(
        "--scr-awb",
        action="store_true",
        help="Zhou 2025 SCR-AWB on no-flash linear RAW + ISSA skin prior (requires --iphone-calibration).",
    )
    ap.add_argument(
        "--skin-reflectance-prior",
        type=str,
        default=None,
        help="Prior name or .json path for all trials (default: P1→issa_median_caucasian, P2→issa_median_african).",
    )
    ap.add_argument(
        "--exposure-anchor-from-training",
        action="store_true",
        help=(
            "Multiply reflectance linear RGB by per-participant white-patch scale from "
            "checker training bundle (chart-free at inference; requires --iphone-calibration)."
        ),
    )
    ap.add_argument(
        "--fitskin-lightness-calibration",
        action="store_true",
        help=(
            "Two-pass pilot calibration: scale reflectance per participant so median L* "
            "matches FitSkin cheek (requires FitSkin CSV merged; for reporting alignment)."
        ),
    )
    ap.add_argument(
        "--cheek-roi",
        action="store_true",
        help="Mean reflectance Lab inside MediaPipe cheek hull ∩ mesh (FitSkin-aligned ROI).",
    )
    ap.add_argument(
        "--exposure-scale-skin-mask",
        action="store_true",
        help=(
            "After ECC, scale flash to no-flash using median luma on skin mask "
            "(cheek mask when --cheek-roi, else full mesh)."
        ),
    )
    ap.add_argument(
        "--reflectance-pre-wb",
        choices=("none", "booth"),
        default="none",
        help=(
            "Deprecated: RGB division before sqrt(A⊙B) (wrong with D65 matrix). "
            "Prefer --reflectance-cat booth."
        ),
    )
    ap.add_argument(
        "--reflectance-fusion",
        choices=("geometric", "log", "arithmetic"),
        default="geometric",
        help="How to combine aligned no-flash A and flash B in linear RGB.",
    )
    ap.add_argument(
        "--reflectance-cat",
        choices=("none", "booth", "lu"),
        default="none",
        help="Bradford CAT on reflectance XYZ: scene white = booth Planck or Lu-estimated ambient.",
    )
    ap.add_argument(
        "--raw-u01-percentile-skin",
        action="store_true",
        help="Scale RAW u01 from skin-mask 99.5th percentile (not full-frame).",
    )
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    global _CAMERA_RGB_TO_XYZ, _CAMERA_RGB_TO_XYZ_AFFINE, _IPHONE_FLASH_CCT_K, _IPHONE_FLASH_RGB
    measured_flash_cct_k: Optional[float] = None
    flash_rgb_measured: Optional[np.ndarray] = None
    cal = None
    scr_spectral_sensitivity: Optional[np.ndarray] = None
    scr_wavelengths_nm: Optional[np.ndarray] = None
    if args.iphone_calibration is not None:
        from iphone_camera_calibration import load_calibration_bundle

        cal = load_calibration_bundle(args.iphone_calibration)
        _CAMERA_RGB_TO_XYZ = np.asarray(cal.camera_rgb_to_xyz, dtype=np.float64)
        _CAMERA_RGB_TO_XYZ_AFFINE = None
        aff_path = Path(args.iphone_calibration) / "camera_rgb_to_xyz_affine.npy"
        if aff_path.is_file():
            _CAMERA_RGB_TO_XYZ_AFFINE = np.load(aff_path)
            print(f"Using affine camera_rgb_to_xyz from {aff_path}", file=sys.stderr)
        _IPHONE_FLASH_CCT_K = float(cal.flash_cct_k)
        _IPHONE_FLASH_RGB = np.asarray(cal.flash_rgb_linear, dtype=np.float64)
        measured_flash_cct_k = _IPHONE_FLASH_CCT_K
        flash_rgb_measured = _IPHONE_FLASH_RGB
        scr_spectral_sensitivity = np.asarray(cal.spectral_sensitivity_rgb, dtype=np.float64)
        scr_wavelengths_nm = np.asarray(cal.wavelengths_nm, dtype=np.float64)
        print(
            f"iPhone calibration: {cal.device_label}  flash CCT≈{cal.flash_cct_k:.0f} K  "
            f"(MK350 SPD)",
            file=sys.stderr,
        )

    exposure_anchors: Optional[Dict[str, float]] = None
    if args.fitskin_lightness_calibration and args.no_fitskin:
        raise SystemExit("--fitskin-lightness-calibration requires FitSkin columns (omit --no-fitskin).")
    if args.reflectance_pre_wb == "booth" and (
        args.known_ambient_cct_k is None or float(args.known_ambient_cct_k) <= 0.0
    ):
        raise SystemExit("--reflectance-pre-wb booth requires --known-ambient-cct-k > 0.")
    if args.reflectance_cat == "booth" and (
        args.known_ambient_cct_k is None or float(args.known_ambient_cct_k) <= 0.0
    ):
        raise SystemExit("--reflectance-cat booth requires --known-ambient-cct-k > 0.")

    if args.exposure_anchor_from_training:
        if args.iphone_calibration is None:
            raise SystemExit(
                "--exposure-anchor-from-training requires --iphone-calibration "
                "(trained bundle with training_trials[].white_patch_scale)."
            )
        from exposure_anchor import load_exposure_anchors

        exposure_anchors = load_exposure_anchors(args.iphone_calibration)
        print(f"Reflectance exposure anchors: {exposure_anchors}", file=sys.stderr)

    if args.scr_awb and cal is None:
        raise SystemExit("--scr-awb requires --iphone-calibration (monochromator S_j(lambda)).")
    if args.scr_awb:
        print(
            "SCR-AWB: Zhou 2025 skin prior + monochromator sensitivity "
            f"(override prior: {args.skin_reflectance_prior or 'per-participant'})",
            file=sys.stderr,
        )

    lu_sharpen_path = args.lu_sharpening_matrix
    if lu_sharpen_path is None and args.iphone_calibration is not None:
        auto_m = Path(args.iphone_calibration) / "lu_sharpening_M.npy"
        if auto_m.is_file():
            lu_sharpen_path = auto_m
    lu_sharpening = load_lu_sharpening_matrix(lu_sharpen_path)
    flash_cct_k: Optional[float] = float(args.flash_cct_k) if args.flash_cct_k > 0.0 else None
    known_ambient_cct_k: Optional[float] = (
        float(args.known_ambient_cct_k) if args.known_ambient_cct_k > 0.0 else None
    )

    for optname, val in (
        ("skin-l-star-trim-lo", args.skin_l_star_trim_lo),
        ("skin-l-star-trim-hi", args.skin_l_star_trim_hi),
        ("skin-a-star-trim-lo", args.skin_a_star_trim_lo),
        ("skin-a-star-trim-hi", args.skin_a_star_trim_hi),
        ("skin-b-star-trim-lo", args.skin_b_star_trim_lo),
        ("skin-b-star-trim-hi", args.skin_b_star_trim_hi),
    ):
        if val < 0.0 or val >= 0.5:
            raise SystemExit(f"--{optname} must be in [0, 0.5), got {val}")
    if args.skin_min_chroma_ab < 0.0:
        raise SystemExit("--skin-min-chroma-ab must be >= 0")
    if args.raw_half_size < 0:
        raise SystemExit("--raw-half-size must be >= 0")

    manifest_rows: List[Dict[str, str]]
    if args.data_root is not None:
        manifest_rows = discover_raw_pairs(args.data_root)
        if args.input_mode == "auto":
            args.input_mode = "dng"
    elif args.manifest is not None and args.manifest.is_file():
        with args.manifest.open(newline="", encoding="utf-8") as f:
            manifest_rows = list(csv.DictReader(f))
        if args.input_mode == "auto":
            args.input_mode = "jpeg"
    elif (_DEFAULT_MANIFEST.is_file() and args.data_root is None):
        with _DEFAULT_MANIFEST.open(newline="", encoding="utf-8") as f:
            manifest_rows = list(csv.DictReader(f))
        if args.input_mode == "auto":
            args.input_mode = "jpeg"
    else:
        raise SystemExit("Provide --data-root for RAW pairs or --manifest for JPEG CSV.")

    if not manifest_rows:
        raise SystemExit("Manifest has no rows.")

    fitskin_lookup: Optional[Dict[Tuple[str, str], Dict[str, str]]] = None
    if not args.no_fitskin:
        scan_csv = args.fitskin_scan_csv.expanduser()
        map_csv = args.fitskin_mapping_csv.expanduser()
        if scan_csv.is_file() and map_csv.is_file():
            fitskin_lookup = load_fitskin_cheek_lookup(scan_csv, map_csv)
            manifest_rows = attach_fitskin_to_manifest_rows(manifest_rows, fitskin_lookup)
            n_ok = sum(
                1
                for r in manifest_rows
                if str(r.get("fitskin_cheek_L", "")).strip()
            )
            print(
                f"FitSkin: merged cheek Lab for {n_ok}/{len(manifest_rows)} row(s) "
                f"from {scan_csv.name} + {map_csv.name}",
                flush=True,
            )
        else:
            print(
                f"Warning: FitSkin CSV missing ({scan_csv} or {map_csv}); "
                "ΔE00 vs FitSkin will be n/a. Use --no-fitskin to silence.",
                file=sys.stderr,
            )

    if args.data_root is not None:
        manifest_path = args.out_dir / "manifest_raw_discovered.csv"
        args.out_dir.mkdir(parents=True, exist_ok=True)
        write_manifest_csv(manifest_rows, manifest_path)
        print(f"Discovered {len(manifest_rows)} RAW pair(s); wrote {manifest_path}", flush=True)

    exclude = {x.strip() for x in args.exclude_trials.split(",") if x.strip()}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir = args.out_dir / "skin_mask_overlays"
    if args.write_overlays:
        overlay_dir.mkdir(parents=True, exist_ok=True)

    rows_out: List[Dict[str, Any]] = []
    pass1_rows: List[Dict[str, Any]] = []
    fitskin_l_gains: Dict[str, float] = {}
    pass_schedule = (0, 1) if args.fitskin_lightness_calibration else (1,)

    mp_fm = mp.solutions.face_mesh
    with _silence_stderr():
        with mp_fm.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
        ) as face_mesh:
            for pass_idx in pass_schedule:
                if pass_idx == 1 and args.fitskin_lightness_calibration:
                    fitskin_l_gains = _estimate_fitskin_lightness_gains(pass1_rows)
                    print(
                        f"FitSkin lightness gains (median L* ratio): {fitskin_l_gains}",
                        file=sys.stderr,
                    )
                for row in manifest_rows:
                    sid = row.get("subject_id") or _subject_key(row)
                    if sid in exclude:
                        if args.debug:
                            print(f"skip excluded {sid}", file=sys.stderr)
                        continue
                    nf_path = Path(row["path_noflash"])
                    fl_path = Path(row["path_flash"])
                    if not nf_path.is_file() or not fl_path.is_file():
                        print(f"skip {sid}: missing image", file=sys.stderr)
                        continue

                    use_raw = args.input_mode == "dng" or nf_path.suffix in _RAW_EXTS
                    scr_prior_name: Optional[str] = None
                    if args.scr_awb:
                        from scr_awb import resolve_prior_name

                        scr_prior_name = resolve_prior_name(
                            str(row.get("participant", "")),
                            sid,
                            override=args.skin_reflectance_prior,
                        )
                    from exposure_anchor import participant_key

                    pk = participant_key(sid, str(row.get("participant", "")))
                    reflectance_scale: Optional[float] = None
                    if exposure_anchors is not None:
                        reflectance_scale = exposure_anchors.get(pk)
                        if reflectance_scale is None:
                            print(
                                f"Warning: no exposure anchor for {pk}; reflectance unscaled",
                                file=sys.stderr,
                            )
                    fitskin_l_gain: Optional[float] = None
                    if pass_idx == 1 and fitskin_l_gains:
                        fitskin_l_gain = fitskin_l_gains.get(pk)
                    try:
                        if use_raw:
                            if args.debug:
                                print(f"RAW {sid}: {nf_path.name} / {fl_path.name}", file=sys.stderr)
                            if args.raw_u01_percentile_skin:
                                nf_lin = raw_to_linear_u01_skin_percentile(
                                    nf_path,
                                    face_mesh,
                                    half_size=args.raw_half_size,
                                    use_camera_wb=args.raw_camera_wb,
                                    skin_triangulation=args.skin_triangulation,
                                    skin_exclusion_dilate_iod_fraction=args.skin_exclusion_dilate_iod,
                                )
                                fl_lin = raw_to_linear_u01_skin_percentile(
                                    fl_path,
                                    face_mesh,
                                    half_size=args.raw_half_size,
                                    use_camera_wb=args.raw_camera_wb,
                                    skin_triangulation=args.skin_triangulation,
                                    skin_exclusion_dilate_iod_fraction=args.skin_exclusion_dilate_iod,
                                )
                            else:
                                nf_lin = raw_to_linear_u01(
                                    nf_path,
                                    half_size=args.raw_half_size,
                                    use_camera_wb=args.raw_camera_wb,
                                )
                                fl_lin = raw_to_linear_u01(
                                    fl_path,
                                    half_size=args.raw_half_size,
                                    use_camera_wb=args.raw_camera_wb,
                                )
                            stats = process_flash_pair(
                                face_mesh,
                                noflash_lin_u01=nf_lin,
                                flash_lin_u01=fl_lin,
                                max_align_width=args.max_align_width,
                                motion_ecc=args.motion,
                                skin_triangulation=args.skin_triangulation,
                                skin_exclusion_dilate_iod_fraction=args.skin_exclusion_dilate_iod,
                                l_star_trim_lo=args.skin_l_star_trim_lo,
                                l_star_trim_hi=args.skin_l_star_trim_hi,
                                a_star_trim_lo=args.skin_a_star_trim_lo,
                                a_star_trim_hi=args.skin_a_star_trim_hi,
                                b_star_trim_lo=args.skin_b_star_trim_lo,
                                b_star_trim_hi=args.skin_b_star_trim_hi,
                                skin_min_chroma_ab=args.skin_min_chroma_ab,
                                input_mode="dng",
                                flash_cct_k=flash_cct_k,
                                lu_sharpening_matrix=lu_sharpening,
                                known_ambient_cct_k=known_ambient_cct_k,
                                known_ambient_duv=float(args.known_ambient_duv),
                                measured_flash_cct_k=measured_flash_cct_k,
                                flash_rgb_measured=flash_rgb_measured,
                                scr_awb_prior_name=scr_prior_name,
                                scr_awb_spectral_sensitivity=scr_spectral_sensitivity,
                                scr_awb_wavelengths_nm=scr_wavelengths_nm,
                                reflectance_exposure_scale=reflectance_scale,
                                reflectance_fitskin_lightness_gain=fitskin_l_gain,
                                cheek_roi=args.cheek_roi,
                                exposure_scale_skin_mask=args.exposure_scale_skin_mask,
                                reflectance_pre_wb=args.reflectance_pre_wb,
                                reflectance_fusion=args.reflectance_fusion,
                                reflectance_cat=args.reflectance_cat,
                            )
                        else:
                            nf_bgr = cv2.imread(str(nf_path), cv2.IMREAD_COLOR)
                            fl_bgr = cv2.imread(str(fl_path), cv2.IMREAD_COLOR)
                            if nf_bgr is None or fl_bgr is None:
                                print(f"skip {sid}: imread failed", file=sys.stderr)
                                continue
                            stats = process_flash_pair(
                                face_mesh,
                                noflash_bgr=nf_bgr,
                                flash_bgr=fl_bgr,
                                max_align_width=args.max_align_width,
                                motion_ecc=args.motion,
                                skin_triangulation=args.skin_triangulation,
                                skin_exclusion_dilate_iod_fraction=args.skin_exclusion_dilate_iod,
                                l_star_trim_lo=args.skin_l_star_trim_lo,
                                l_star_trim_hi=args.skin_l_star_trim_hi,
                                a_star_trim_lo=args.skin_a_star_trim_lo,
                                a_star_trim_hi=args.skin_a_star_trim_hi,
                                b_star_trim_lo=args.skin_b_star_trim_lo,
                                b_star_trim_hi=args.skin_b_star_trim_hi,
                                skin_min_chroma_ab=args.skin_min_chroma_ab,
                                input_mode="jpeg",
                                flash_cct_k=flash_cct_k,
                                lu_sharpening_matrix=lu_sharpening,
                                known_ambient_cct_k=known_ambient_cct_k,
                                known_ambient_duv=float(args.known_ambient_duv),
                                measured_flash_cct_k=measured_flash_cct_k,
                                flash_rgb_measured=flash_rgb_measured,
                                scr_awb_prior_name=scr_prior_name if args.scr_awb else None,
                                scr_awb_spectral_sensitivity=scr_spectral_sensitivity,
                                scr_awb_wavelengths_nm=scr_wavelengths_nm,
                                reflectance_exposure_scale=reflectance_scale,
                                reflectance_fitskin_lightness_gain=fitskin_l_gain,
                                cheek_roi=args.cheek_roi,
                                exposure_scale_skin_mask=args.exposure_scale_skin_mask,
                                reflectance_pre_wb=args.reflectance_pre_wb,
                                reflectance_fusion=args.reflectance_fusion,
                                reflectance_cat=args.reflectance_cat,
                            )
                    except ValueError as ex:
                        print(f"skip {sid}: {ex}", file=sys.stderr)
                        continue
                    except Exception as ex:
                        print(f"skip {sid}: {ex}", file=sys.stderr)
                        if args.debug:
                            raise
                        continue

                    dbg = stats.pop("_debug")
                    rec: Dict[str, Any] = {
                        "subject_id": sid,
                        "participant": row.get("participant", ""),
                        "trial": row.get("trial", ""),
                        "path_noflash": str(nf_path),
                        "path_flash": str(fl_path),
                        "fitskin_cheek_L": float(row.get("fitskin_cheek_L", np.nan)),
                        "fitskin_cheek_a": float(row.get("fitskin_cheek_a", np.nan)),
                        "fitskin_cheek_b": float(row.get("fitskin_cheek_b", np.nan)),
                    }
                    rec.update({k: v for k, v in stats.items() if not k.startswith("_")})

                    if pass_idx == 0 and args.fitskin_lightness_calibration:
                        lr0 = float(rec.get("reflectance_L", np.nan))
                        if not np.isfinite(lr0) or lr0 < 1.0 or lr0 > 95.0:
                            print(
                                f"Warning: pass-1 reflectance L*={lr0} out of range for {sid}; "
                                "skipping FitSkin lightness fit row",
                                file=sys.stderr,
                            )
                        else:
                            pass1_rows.append(
                                {
                                    "subject_id": sid,
                                    "participant": row.get("participant", ""),
                                    "reflectance_L": lr0,
                                    "fitskin_cheek_L": rec.get("fitskin_cheek_L"),
                                }
                            )
                        continue

                    for method in (
                        "reflectance",
                        "noflash",
                        "flash_aligned",
                        "lu_wb",
                        "lu_booth_wb",
                        "scr_awb_wb",
                    ):
                        if f"{method}_L" not in rec:
                            continue
                        rec[f"{method}_cheek_de00"] = _de00_cheek(
                            rec[f"{method}_L"],
                            rec[f"{method}_a"],
                            rec[f"{method}_b"],
                            rec["fitskin_cheek_L"],
                            rec["fitskin_cheek_a"],
                            rec["fitskin_cheek_b"],
                        )

                    if args.write_overlays:
                        psl.write_skin_sampling_overlay_png(
                            overlay_dir / f"{sid}_noflash_skin.png",
                            dbg["noflash_bgr"],
                            dbg["oval_pts"],
                            dbg["kept_tris"],
                            dbg["mask"],
                            dbg["excl_dil"],
                            mesh_xy=dbg["mesh_xy"],
                            max_width=args.overlay_max_width,
                        )

                    sub = args.out_dir / sid
                    sub.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(sub / "noflash_aligned.png"), dbg["noflash_bgr"])
                    cv2.imwrite(str(sub / "flash_aligned.png"), dbg["flash_bgr"])
                    cv2.imwrite(str(sub / "lu_wb.png"), dbg["wb_bgr"])
                    cv2.imwrite(str(sub / "reflectance_preview.png"), dbg["albedo_bgr"])
                    if dbg.get("scr_wb_bgr") is not None:
                        cv2.imwrite(str(sub / "scr_awb_wb.png"), dbg["scr_wb_bgr"])

                    with (sub / "summary.json").open("w", encoding="utf-8") as jf:
                        json.dump(
                            {
                                k: v
                                for k, v in rec.items()
                                if isinstance(v, (int, float, str, bool)) or v is None
                            },
                            jf,
                            indent=2,
                        )

                    rows_out.append(rec)
                    de_str = (
                        f"{rec['reflectance_cheek_de00']:.2f}"
                        if np.isfinite(rec["reflectance_cheek_de00"])
                        else "n/a"
                    )
                    print(
                        f"{sid}: ambient CCT={rec['ambient_cct_k']:.0f}K  "
                        f"flash CCT={rec['flash_cct_k']:.0f}K ({rec['flash_cct_source']})  "
                        f"reflectance L*a*b*=({rec['reflectance_L']:.1f},{rec['reflectance_a']:.1f},{rec['reflectance_b']:.1f})  "
                        f"ΔE00 vs FitSkin={de_str}",
                        flush=True,
                    )
    if not rows_out:
        raise SystemExit("No rows processed.")

    _annotate_best_method_vs_fitskin(rows_out)

    csv_path = args.out_dir / "flash_noflash_skin_lab.csv"
    fieldnames = list(rows_out[0].keys())
    with csv_path.open("w", newline="", encoding="utf-8") as cf:
        w = csv.DictWriter(cf, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows_out)

    summary: Dict[str, Any] = {
        "n_trials": len(rows_out),
        "input_mode": args.input_mode,
        "raw_half_size": args.raw_half_size,
        "raw_camera_wb": bool(args.raw_camera_wb),
        "skin_mask": f"physio_skin_lab_monk.build_skin_mask_from_mesh ({args.skin_triangulation})",
        "skin_trims": {
            "l_star_trim_lo": args.skin_l_star_trim_lo,
            "l_star_trim_hi": args.skin_l_star_trim_hi,
            "a_star_trim_lo": args.skin_a_star_trim_lo,
            "a_star_trim_hi": args.skin_a_star_trim_hi,
            "b_star_trim_lo": args.skin_b_star_trim_lo,
            "b_star_trim_hi": args.skin_b_star_trim_hi,
            "skin_min_chroma_ab": args.skin_min_chroma_ab,
        },
        "align": (
            "linear RAW u01: ECC + exposure (align_flash_to_noflash_linear)"
            if args.input_mode == "dng"
            else "mabl-flash-illumination align_flash_to_noflash (ECC + exposure)"
        ),
        "raw_note": (
            "DNG/CR2: rawpy linear camera RGB, 99.5% percentile → u01; "
            + (
                "Lab via trained camera_rgb_to_xyz when --iphone-calibration set."
                if cal is not None
                else "Lab via default sRGB D65 matrix."
            )
            if args.input_mode == "dng"
            else None
        ),
        "illuminant": (
            "Lu & Drew 2006 log-diff geometric chromaticity (secondary comparison path); "
            "flash CCT auto from pure-flash neutrals unless --flash-cct-k > 0"
        ),
        "known_ambient_cct_k": known_ambient_cct_k,
        "known_ambient_duv": float(args.known_ambient_duv) if known_ambient_cct_k else None,
        "primary_skin_metric": (
            "reflectance: sqrt(A ⊙ B) on aligned linear RGB; "
            "A = no-flash, B = ECC-aligned exposure-matched flash (practical recipe, not Lu 2006)"
        ),
        "flash_cct_k": args.flash_cct_k if args.flash_cct_k > 0 else "auto",
        "lu_spectral_sharpening": "matrix" if lu_sharpening is not None else "none",
        "reflectance": (
            "R = sqrt(A ⊙ B); B = flash after ECC registration and exposure scaling to match A"
        ),
        "lab_reflectance": (
            "linear camera RGB → XYZ (trained matrix if --iphone-calibration) → Lab, D65 white"
        ),
        "lab_comparison_paths": "skimage rgb2lab D65 on 8-bit BGR previews (noflash / flash / lu_wb)",
        "excluded_trials": sorted(exclude),
        "fitskin_scan_csv": str(args.fitskin_scan_csv) if not args.no_fitskin else None,
        "fitskin_mapping_csv": str(args.fitskin_mapping_csv) if not args.no_fitskin else None,
    }
    summary_methods = ("reflectance", "noflash", "flash_aligned", "lu_wb", "lu_booth_wb", "scr_awb_wb")
    if args.exposure_anchor_from_training and exposure_anchors:
        summary["reflectance_exposure_anchor"] = exposure_anchors
    if args.fitskin_lightness_calibration and fitskin_l_gains:
        summary["fitskin_lightness_gain"] = fitskin_l_gains
    if args.cheek_roi:
        summary["reflectance_lab_roi"] = "cheek (MediaPipe hull ∩ mesh)"
    if args.exposure_scale_skin_mask:
        summary["align_exposure"] = (
            "ECC; flash exposure scale from skin mask median luma (not global scene)"
        )
    if args.reflectance_pre_wb != "none":
        summary["reflectance_pre_wb"] = args.reflectance_pre_wb
    if args.reflectance_fusion != "geometric":
        summary["reflectance_fusion"] = args.reflectance_fusion
    if args.reflectance_cat != "none":
        summary["reflectance_cat"] = args.reflectance_cat
    if args.raw_u01_percentile_skin:
        summary["raw_u01_scale"] = "skin-mask 99.5th percentile"
    best_counts: Dict[str, int] = {}
    for r in rows_out:
        m = str(r.get("best_method_vs_fitskin", ""))
        if m:
            best_counts[m] = best_counts.get(m, 0) + 1
    if best_counts:
        summary["best_method_vs_fitskin_counts"] = best_counts
    if args.scr_awb:
        summary["scr_awb"] = (
            "Zhou et al. 2025 SCR-AWB: M @ alpha = median skin RGB; "
            "ISSA cheek priors (P1 caucasian, P2 african by default)"
        )
        summary["skin_reflectance_prior_override"] = args.skin_reflectance_prior
    for method in summary_methods:
        col = f"{method}_cheek_de00"
        de = [float(r[col]) for r in rows_out if col in r and np.isfinite(r[col])]
        if de:
            summary[f"{method}_cheek_de00_mean"] = float(np.mean(de))
            summary[f"{method}_cheek_de00_median"] = float(np.median(de))

    with (args.out_dir / "summary.json").open("w", encoding="utf-8") as sf:
        json.dump(summary, sf, indent=2)

    print(f"\nWrote {csv_path} ({len(rows_out)} trials)")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
