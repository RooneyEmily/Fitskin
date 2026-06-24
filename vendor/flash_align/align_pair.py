"""Register flash image to no-flash (reference) and match exposure."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import cv2
import numpy as np

from .color_linear import bgr_uint8_to_linear_rgb, linear_rgb_to_bgr_uint8


@dataclass
class AlignResult:
    flash_aligned_linear: np.ndarray
    noflash_linear: np.ndarray
    warp_matrix: np.ndarray
    exposure_scale: float
    ecc_cc: float


def _to_gray01(linear_rgb: np.ndarray) -> np.ndarray:
    g = 0.2126 * linear_rgb[..., 0] + 0.7152 * linear_rgb[..., 1] + 0.0722 * linear_rgb[..., 2]
    return np.clip(g, 0.0, 1.0).astype(np.float32)


def estimate_exposure_scale(
    noflash: np.ndarray,
    flash: np.ndarray,
    *,
    max_sat: float = 0.98,
    min_luma: float = 0.05,
) -> float:
    """Scale flash linear RGB so median luma matches no-flash on reliable pixels."""
    g0 = _to_gray01(noflash)
    g1 = _to_gray01(flash)
    mask = (g0 > min_luma) & (g0 < max_sat) & (g1 > min_luma) & (g1 < max_sat)
    if mask.sum() < 100:
        mask = np.ones(g0.shape, dtype=bool)
    r = np.median(g0[mask] / np.maximum(g1[mask], 1e-6))
    return float(np.clip(r, 0.25, 4.0))


def align_flash_to_noflash(
    noflash_bgr: np.ndarray,
    flash_bgr: np.ndarray,
    *,
    motion_ecc: str = "euclidean",
) -> AlignResult:
    """
    ECC alignment: warp flash → no-flash grid.
    Returns both images in linear RGB on the no-flash pixel grid.
    """
    nf_lin = bgr_uint8_to_linear_rgb(noflash_bgr)
    fl_lin = bgr_uint8_to_linear_rgb(flash_bgr)

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

    scale = estimate_exposure_scale(nf_lin, fl_warp)
    fl_scaled = np.clip(fl_warp * scale, 0.0, None)

    return AlignResult(
        flash_aligned_linear=fl_scaled,
        noflash_linear=nf_lin,
        warp_matrix=warp,
        exposure_scale=scale,
        ecc_cc=float(cc),
    )


def align_result_to_bgr_preview(res: AlignResult) -> Tuple[np.ndarray, np.ndarray]:
    return linear_rgb_to_bgr_uint8(res.noflash_linear), linear_rgb_to_bgr_uint8(res.flash_aligned_linear)
