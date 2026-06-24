"""sRGB ↔ linear RGB for flash/no-flash pipelines (display-referred inputs)."""
from __future__ import annotations

import numpy as np

_EPS = 1e-8


def srgb_to_linear(rgb: np.ndarray) -> np.ndarray:
    """rgb in [0, 1], shape (..., 3)."""
    x = np.asarray(rgb, dtype=np.float64)
    a = 0.055
    return np.where(x <= 0.04045, x / 12.92, ((x + a) / (1.0 + a)) ** 2.4)


def linear_to_srgb(rgb: np.ndarray) -> np.ndarray:
    x = np.asarray(rgb, dtype=np.float64)
    a = 0.055
    y = np.where(x <= 0.0031308, x * 12.92, (1.0 + a) * np.power(np.maximum(x, 0.0), 1.0 / 2.4) - a)
    return np.clip(y, 0.0, 1.0)


def bgr_uint8_to_linear_rgb(bgr: np.ndarray) -> np.ndarray:
    import cv2

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float64) / 255.0
    return srgb_to_linear(rgb)


def linear_rgb_to_bgr_uint8(rgb: np.ndarray) -> np.ndarray:
    import cv2

    srgb = linear_to_srgb(rgb)
    return cv2.cvtColor((srgb * 255.0 + 0.5).astype(np.uint8), cv2.COLOR_RGB2BGR)
