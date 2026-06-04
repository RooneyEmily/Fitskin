"""
Canonical **ColorChecker Classic 24** reference under **D65** (sRGB → XYZ / Lab).

Patch order matches ``mcc24_classic.MCC24_PATCHES`` / OpenCV ``mcc`` row-major index 0–23.
Values are the usual published sRGB (8-bit) set (BabelColor / Calibrite-style Classic 24, post-2014),
converted with ``skimage.color`` (D65, 2° observer) — **not** physio-lab PR-250 spectrometer rows.

Used by the FitSkin in-scene chart pipeline (``mabl-flash-illumination``).
"""

from __future__ import annotations

import numpy as np

from mcc24_classic import MCC24_PATCHES, WHITE_PATCH_INDEX
from srgb_eotf import srgb_255_to_linear as _srgb255_to_linear

# sRGB 0–255, row-major Classic 24 (same layout as OpenCV mcc index 0..23)
MCC24_CANONICAL_SRGB_255: np.ndarray = np.array(
    [
        [115, 82, 68],
        [194, 150, 130],
        [98, 122, 157],
        [87, 108, 67],
        [133, 128, 177],
        [103, 189, 170],
        [214, 126, 44],
        [80, 91, 166],
        [193, 90, 99],
        [94, 60, 108],
        [157, 188, 64],
        [224, 163, 46],
        [56, 61, 150],
        [70, 148, 73],
        [175, 54, 60],
        [231, 199, 31],
        [187, 86, 149],
        [8, 133, 161],
        [243, 243, 242],
        [200, 200, 200],
        [160, 160, 160],
        [122, 122, 121],
        [85, 85, 85],
        [52, 52, 52],
    ],
    dtype=np.float64,
)

try:
    from skimage import color as _skcolor
except ImportError as e:
    raise ImportError("skimage required for mcc24_canonical_d65") from e

_XYZ2SRGB = np.array(
    [
        [3.2404542, -1.5371385, -0.4985314],
        [-0.9692660, 1.8760108, 0.0415560],
        [0.0556434, -0.2040259, 1.0572252],
    ],
    dtype=np.float64,
)
_INV_XYZ2SRGB = np.linalg.inv(_XYZ2SRGB)


def srgb255_linear_to_xyz_d65_100(rgb_255: np.ndarray) -> np.ndarray:
    """(N,3) or (3,) display sRGB 0–255 → XYZ (Y roughly 0–100 scale)."""
    one = np.asarray(rgb_255).ndim == 1
    lin = _srgb255_to_linear(np.atleast_2d(rgb_255))
    xyz = 100.0 * (lin @ _INV_XYZ2SRGB.T)
    return xyz.reshape(3) if one else xyz


def load_canonical_xyz_d65() -> np.ndarray:
    """``(24, 3)`` XYZ D65 for mcc indices 0..23."""
    return srgb255_linear_to_xyz_d65_100(MCC24_CANONICAL_SRGB_255)


def load_canonical_lab_d65() -> np.ndarray:
    """``(24, 3)`` CIE Lab D65 for mcc indices 0..23."""
    rgb = MCC24_CANONICAL_SRGB_255 / 255.0
    return _skcolor.rgb2lab(rgb.reshape(24, 1, 3), illuminant="D65", observer="2").reshape(24, 3)


def patch_display_names() -> list[str]:
    return [MCC24_PATCHES[i][0] for i in range(24)]


CANONICAL_WHITE_XYZ_D65 = srgb255_linear_to_xyz_d65_100(MCC24_CANONICAL_SRGB_255[WHITE_PATCH_INDEX])
