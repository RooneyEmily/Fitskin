"""
ColorChecker **Classic 24** layout as used by OpenCV ``mcc`` (``CCheckerDetector.process(..., 0, 1)``).

**Patch index** ``0 … 23`` is **row-major** from the **top row** of the chart **as detected in the image**
(same ordering as ``getChartsRGB()`` / ``getColorCharts()`` when the chart is found). If the chart is
rotated 180° in frame, “top” in pixel space is the physical bottom of the card—indices are still
0–23 in **image** row-major order, not the printed “patch ID” on the cardboard.

Canonical names follow the usual X-Rite / ColorChecker Classic naming. ``stem_suffix`` matches
lighting export filenames (e.g. ``darkskinfinal``).

**Bottom row (``mcc_patch_index`` 18–23):** Matches the usual **left→right** Classic layout on the
chart in standard orientation: **White**, **Neutral 8**, **Neutral 6.5**, **Neutral 5**,
**Neutral 3.5**, **Black** — aligned with OpenCV ``mcc`` quads and ``lighting_reference_patches.json``.

``WHITE_PATCH_INDEX == 18`` is the bottom-row **white** patch. **Black** is index **23**.

**Anchor patches for skin colorimetry (``physio_skin_lab_raw_pr250``):** MCC indices **0, 1** (dark / light skin)
and **18–23** (white, Neutral 8 … Neutral 3.5, black) share PR-250 spectrometer rows that anchor facial skin
and the gray axis; see ``PR250_SKIN_NEUTRAL_ANCHOR_PATCH_INDICES``.

**Neutral column (gray-axis WB):** mean linear RGB of **19–22** (all neutrals between white and black),
excluding white (18) and black (23). Used by ``physio_skin_lab_raw_pr250`` optional chart gray balance.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np

# MCC "white" patch used for WB in physio_skin_lab_monk.py
WHITE_PATCH_INDEX = 18

# Mean linear RGB of these indices → diagonal gray WB target (R=G=B); excludes white (18) and black (23).
NEUTRAL_COLUMN_WB_PATCH_INDICES: Tuple[int, ...] = (19, 20, 21, 22)

# MCC indices whose PR-250 XYZ rows anchor skin + grayscale in 24-patch RGB→XYZ lstsq (physio_skin_lab_raw_pr250).
PR250_SKIN_NEUTRAL_ANCHOR_PATCH_INDICES: Tuple[int, ...] = (0, 1, 18, 19, 20, 21, 22, 23)

# (display_name, lighting_xyz_filename_stem_suffix, patch_label_slug) — list index == mcc_patch_index
MCC24_PATCHES: List[Tuple[str, str, str]] = [
    ("Dark skin", "darkskinfinal", "dark_skin"),
    ("Light skin", "lightskinfinal", "light_skin"),
    ("Blue sky", "blueskyfinal", "blue_sky"),
    ("Foliage", "foliagefinal", "foliage"),
    ("Blue flower", "blueflowerfinal", "blue_flower"),
    ("Bluish green", "blueishgreenfinal", "bluish_green"),
    ("Orange", "orangefinal", "orange"),
    ("Purplish blue", "purplishbluefinal", "purplish_blue"),
    ("Moderate red", "moderateredfinal", "moderate_red"),
    ("Purple", "purplefinal", "purple"),
    ("Yellow green", "yellowgreenfinal", "yellow_green"),
    ("Orange yellow", "orangeyellowfinal", "orange_yellow"),
    ("Blue", "bluefinal", "blue"),
    ("Green", "greenfinal", "green"),
    ("Red", "redfinal", "red"),
    ("Yellow", "yellowfinal", "yellow"),
    ("Magenta", "magentafinal", "magenta"),
    ("Cyan", "cyanfinal", "cyan"),
    ("White", "whitefinal", "white"),
    ("Neutral 8", "neutral8final", "neutral_8"),
    ("Neutral 6.5", "neutral6.5final", "neutral_6.5"),
    ("Neutral 5", "neutral5final", "neutral_5"),
    ("Neutral 3.5", "neutral3.5final", "neutral_3.5"),
    ("Black", "blackfinal", "black"),
]

# stem_lower -> mcc_index (for lighting file grouping)
STEM_SUFFIX_TO_MCC: dict[str, int] = {stem.lower(): i for i, (_, stem, _) in enumerate(MCC24_PATCHES)}


def patch_display_name(mcc_index: int) -> str:
    return MCC24_PATCHES[mcc_index][0]


def patch_stem_suffix(mcc_index: int) -> str:
    return MCC24_PATCHES[mcc_index][1]


def patch_label_slug(mcc_index: int) -> str:
    return MCC24_PATCHES[mcc_index][2]


def decode_patch_srgb_255_from_charts_matrix(charts_rgb: np.ndarray, patch_index: int) -> Optional[np.ndarray]:
    """
    One patch: sRGB channel means ~0–255 from ``getChartsRGB()`` **average** column (OpenCV: col 1),
    using the usual **72×5** layout (3 rows per patch: R, G, B).
    """
    if charts_rgb is None or charts_rgb.ndim < 2:
        return None
    nrows, ncols = charts_rgb.shape[0], charts_rgb.shape[1]
    if nrows % 3 == 0 and nrows >= 3 * (patch_index + 1) and ncols >= 2:
        base = 3 * patch_index
        rrow, grow, brow = charts_rgb[base], charts_rgb[base + 1], charts_rgb[base + 2]
        out = np.array([float(rrow[1]), float(grow[1]), float(brow[1])], dtype=np.float64)
        if float(np.max(out)) <= 1.0:
            out = out * 255.0
        return out
    if nrows <= patch_index or ncols < 2:
        return None
    row = charts_rgb[patch_index]
    if ncols >= 4:
        cand = np.array(row[1:4], dtype=np.float64)
        if np.all((cand > 10) & (cand < 300)):
            if float(np.max(cand)) <= 1.0:
                cand = cand * 255.0
            return cand
    if ncols >= 3:
        cand = np.array(row[-3:], dtype=np.float64)
        if np.all((cand > 10) & (cand < 300)):
            if float(np.max(cand)) <= 1.0:
                cand = cand * 255.0
            return cand
    return None


def decode_all_patches_srgb_255(charts_rgb: np.ndarray) -> Optional[np.ndarray]:
    """Shape ``(24, 3)`` sRGB means, or ``None`` if layout unsupported."""
    if charts_rgb is None or charts_rgb.ndim < 2:
        return None
    nrows, ncols = charts_rgb.shape[0], charts_rgb.shape[1]
    # OpenCV MCC24 classic: exactly 72 rows (24 patches × 3 channel stats)
    if nrows != 72 or ncols < 2:
        return None
    out = np.zeros((24, 3), dtype=np.float64)
    for i in range(24):
        rgb = decode_patch_srgb_255_from_charts_matrix(charts_rgb, i)
        if rgb is None:
            return None
        out[i] = rgb
    return out


def get_patch_quads_xy(checker) -> Optional[np.ndarray]:
    """
    From ``cv2.mcc`` ``CChecker`` after ``process``: ``(24, 4, 2)`` float patch corners
    in **image pixel coordinates** (``getColorCharts()`` layout; same patch order as ``getChartsRGB``).
    """
    if not hasattr(checker, "getColorCharts"):
        return None
    pts = np.asarray(checker.getColorCharts(), dtype=np.float64)
    if pts.shape != (96, 2):
        return None
    return pts.reshape(24, 4, 2)


def roi_mean_srgb_255_bgr_image(bgr: np.ndarray, quad_xy: np.ndarray) -> np.ndarray:
    """Mean sRGB (R,G,B order) 0–255 inside convex quad; ``quad_xy`` shape (4,2)."""
    h, w = bgr.shape[:2]
    poly = np.round(quad_xy).astype(np.int32)
    poly[:, 0] = np.clip(poly[:, 0], 0, w - 1)
    poly[:, 1] = np.clip(poly[:, 1], 0, h - 1)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillConvexPoly(mask, poly, 255)
    mb = cv2.mean(bgr, mask=mask)[:3]
    return np.array([mb[2], mb[1], mb[0]], dtype=np.float64)


def verify_charts_average_vs_roi_mean(
    bgr: np.ndarray,
) -> Optional[dict]:
    """
    One ``mcc`` pass on ``bgr``: compare OpenCV ``getChartsRGB()`` **average** (col 1, 72×5 layout)
    to an **independent** per-patch mean RGB computed by filling each ``getColorCharts()`` quad
    and calling ``cv2.mean`` (same convention as ``roi_mean_srgb_255_bgr_image``).

    If these agree within ~0.02 per channel, decoding and quad alignment are almost certainly correct.

    Returns dict with arrays shape (24, 3) and scalars, or ``None`` if detection fails.
    """
    if not hasattr(cv2, "mcc"):
        return None
    detector = cv2.mcc.CCheckerDetector.create()
    if not detector.process(bgr, 0, 1):
        return None
    checkers = detector.getListColorChecker()
    if not checkers:
        return None
    checker = checkers[0]
    charts = checker.getChartsRGB()
    if charts is None:
        return None
    charts_srgb = decode_all_patches_srgb_255(charts)
    quads = get_patch_quads_xy(checker)
    if charts_srgb is None or quads is None:
        return None

    roi_srgb = np.zeros((24, 3), dtype=np.float64)
    p_sizes = np.zeros(24, dtype=np.int64)
    for i in range(24):
        roi_srgb[i] = roi_mean_srgb_255_bgr_image(bgr, quads[i])
        p_sizes[i] = int(round(float(charts[3 * i, 0])))

    abs_diff = np.abs(charts_srgb - roi_srgb)
    return {
        "charts_srgb_255": charts_srgb,
        "roi_srgb_255": roi_srgb,
        "abs_diff": abs_diff,
        "p_sizes": p_sizes,
        "max_abs_per_channel": abs_diff.max(axis=0),
        "mean_abs_per_channel": abs_diff.mean(axis=0),
        "max_abs_any": float(abs_diff.max()),
        "mean_abs_any": float(abs_diff.mean()),
    }


def draw_mcc_patch_overlay_from_quads(
    bgr: np.ndarray,
    quads: np.ndarray,
    out_path: Union[str, Path],
    *,
    max_width: int = 1920,
) -> bool:
    """
    Draw **green** quads and red **0–23** labels on ``bgr`` using quads from the **same** ``mcc`` pass
    as ROI sampling (``get_patch_quads_xy``). ``quads`` shape ``(24, 4, 2)`` float ``(x, y)`` pixels.
    """
    quads = np.asarray(quads, dtype=np.float64)
    if quads.shape != (24, 4, 2):
        return False
    outp = Path(out_path)
    vis = bgr.copy()
    h0, w0 = vis.shape[:2]
    thickness = max(1, int(round(min(h0, w0) / 800)))
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.35, min(h0, w0) / 3500.0)
    for i in range(24):
        pts = np.round(quads[i]).astype(np.int32)
        cv2.polylines(vis, [pts.reshape(-1, 1, 2)], True, (0, 220, 0), thickness, cv2.LINE_AA)
        cx, cy = float(pts[:, 0].mean()), float(pts[:, 1].mean())
        label = str(i)
        (tw, th), _ = cv2.getTextSize(label, font, font_scale, 1)
        cv2.putText(
            vis,
            label,
            (int(cx) - tw // 2, int(cy) + th // 2),
            font,
            font_scale,
            (0, 0, 255),
            max(1, thickness),
            cv2.LINE_AA,
        )

    if max_width > 0 and w0 > max_width:
        scale = max_width / float(w0)
        vis = cv2.resize(vis, (int(w0 * scale), int(h0 * scale)), interpolation=cv2.INTER_AREA)

    outp.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(outp), vis)
    return True


def draw_mcc_patch_overlay(
    bgr: np.ndarray,
    out_path: Union[str, Path],
    *,
    max_width: int = 1920,
) -> bool:
    """
    Save a copy of ``bgr`` with green patch quads and mcc indices 0–23 for visual inspection.
    Runs **its own** ``mcc`` detection (for ad-hoc images). For RAW pipeline parity with patch ROIs,
    prefer ``draw_mcc_patch_overlay_from_quads`` with quads from the same pass as ``patch_linear_rgb_24``.
    Returns ``True`` if written.
    """
    if not hasattr(cv2, "mcc"):
        return False
    detector = cv2.mcc.CCheckerDetector.create()
    if not detector.process(bgr, 0, 1):
        return False
    checker = detector.getListColorChecker()[0]
    quads = get_patch_quads_xy(checker)
    if quads is None:
        return False
    return draw_mcc_patch_overlay_from_quads(bgr, quads, out_path, max_width=max_width)
