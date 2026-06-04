"""
ColorChecker **Classic 24** detection using ``colour_checker_detection`` segmentation — same
approach as ``WB_videos/videos.py`` (``detect_colour_checkers_segmentation``).

Returns ``swatch_colours`` as a **(24, 3)** array in **0–1** linear-ish RGB (library convention);
multiply by **255** for comparison with OpenCV / sRGB 8-bit style values (as in ``videos.py``).

White patch for WB is index **18** (same physical swatch as ``mcc24_classic.WHITE_PATCH_INDEX``).

Depends: ``pip install colour_checker_detection`` (Colour-science stack).
"""

from __future__ import annotations

from typing import Any, Optional, Tuple

import cv2
import numpy as np

try:
    from colour_checker_detection import detect_colour_checkers_segmentation
    from colour_checker_detection.detection.common import DataDetectionColourChecker
except ImportError:  # pragma: no cover
    detect_colour_checkers_segmentation = None  # type: ignore[misc, assignment]
    DataDetectionColourChecker = None  # type: ignore[misc, assignment]

WHITE_PATCH_INDEX = 18


def library_available() -> bool:
    return detect_colour_checkers_segmentation is not None


def _unpack_detection(colour_checker_data: Any) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Return (swatch_colours (24,3), quadrilateral (4,2)) or None."""
    if DataDetectionColourChecker is None:
        return None
    if isinstance(colour_checker_data, DataDetectionColourChecker):
        swatch_colours, _swatch_masks, _colour_checker_image, quadrilateral = colour_checker_data.values
    elif isinstance(colour_checker_data, (tuple, list)) and len(colour_checker_data) == 4:
        swatch_colours, _swatch_masks, _colour_checker_image, quadrilateral = colour_checker_data
    else:
        return None
    sw = np.asarray(swatch_colours, dtype=np.float64)
    quad = np.asarray(quadrilateral, dtype=np.float64)
    if sw.shape != (24, 3) or quad.shape != (4, 2):
        return None
    return sw, quad


def chart_area_fraction_from_quad(quad_xy: np.ndarray, height: int, width: int) -> float:
    poly = quad_xy.reshape(-1, 1, 2).astype(np.float32)
    return float(cv2.contourArea(poly)) / float(max(height * width, 1))


def detect_classic24_rgb(rgb: np.ndarray) -> Optional[dict]:
    """
    Run segmentation on an **RGB** image (H,W,3), uint8 or float.

    Returns dict:
      - ``swatch_rgb_01`` (24, 3) clipped 0–1
      - ``swatch_rgb_255`` (24, 3) = swatch * 255 (same scaling as ``videos.py`` WB)
      - ``white_rgb_255`` (3,)
      - ``quadrilateral`` (4, 2) chart corners in pixels
      - ``chart_area_fraction`` float
    or ``None`` if not detected / library missing.
    """
    if not library_available():
        return None
    results = detect_colour_checkers_segmentation(rgb, additional_data=True)
    if not results:
        return None
    for item in results:
        unpacked = _unpack_detection(item)
        if unpacked is None:
            continue
        sw01, quad = unpacked
        sw01 = np.clip(sw01, 0.0, 1.0)
        sw255 = sw01 * 255.0
        h, w = rgb.shape[:2]
        return {
            "swatch_rgb_01": sw01,
            "swatch_rgb_255": sw255,
            "white_rgb_255": sw255[WHITE_PATCH_INDEX].copy(),
            "quadrilateral": quad,
            "chart_area_fraction": chart_area_fraction_from_quad(quad, h, w),
        }
    return None


def detect_classic24_bgr(bgr: np.ndarray) -> Optional[dict]:
    """BGR uint8 image → same return as ``detect_classic24_rgb``."""
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return detect_classic24_rgb(rgb)
