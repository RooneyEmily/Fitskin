"""
sRGB electro-optical transfer (display → linear-light).

IEC 61966-2-1 piecewise curve is the default for chart CCM and canonical MCC reference XYZ.
"""

from __future__ import annotations

import numpy as np


def srgb_u01_to_linear(rgb: np.ndarray) -> np.ndarray:
    """Non-linear sRGB in [0, 1] → linear-light RGB, same shape (…, 3)."""
    x = np.clip(np.asarray(rgb, dtype=np.float64), 0.0, 1.0)
    lo = x <= 0.04045
    out = np.empty_like(x, dtype=np.float64)
    out[lo] = x[lo] / 12.92
    out[~lo] = ((x[~lo] + 0.055) / 1.055) ** 2.4
    return out


def srgb_255_to_linear(rgb_255: np.ndarray) -> np.ndarray:
    """8-bit display sRGB → linear-light RGB."""
    v = np.clip(np.asarray(rgb_255, dtype=np.float64) / 255.0, 0.0, 1.0)
    return srgb_u01_to_linear(v)


def srgb_255_to_linear_gamma22(rgb_255: np.ndarray) -> np.ndarray:
    """Legacy |v/255|^2.2 approximation (pre-2026 chart pipelines)."""
    v = np.clip(np.asarray(rgb_255, dtype=np.float64) / 255.0, 0.0, 1.0)
    return v**2.2


def linear_u01_to_srgb_u01(rgb: np.ndarray) -> np.ndarray:
    """Linear-light RGB in [0, ∞) → non-linear sRGB in [0, 1] (IEC 61966-2-1)."""
    x = np.maximum(np.asarray(rgb, dtype=np.float64), 0.0)
    lo = x <= 0.0031308
    out = np.empty_like(x, dtype=np.float64)
    out[lo] = x[lo] * 12.92
    out[~lo] = (1.0 + 0.055) * np.power(x[~lo], 1.0 / 2.4) - 0.055
    return np.clip(out, 0.0, 1.0)
