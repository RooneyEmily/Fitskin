"""MediaPipe face-mesh cheek ROI for FitSkin-aligned skin sampling."""
from __future__ import annotations

from typing import Any, Sequence

import cv2
import numpy as np

# Cheek landmark indices (MediaPipe face mesh), aligned with chart_cc_fitskin_lib.py
CHEEK_LANDMARKS = (
    50, 101, 36, 205, 206, 207, 187, 123, 116, 117, 118, 119, 120, 121, 128, 245, 193, 194, 188,
    174, 196, 197, 177, 137, 147,
)
CHEEK_R_LANDMARKS = (
    280, 330, 266, 425, 426, 427, 411, 352, 345, 346, 347, 348, 349, 350, 357, 465, 416, 415, 404,
    399, 421, 419, 401, 366, 376,
)


def cheek_mask_from_landmarks(
    h: int,
    w: int,
    landmarks: Sequence[Any],
    mesh_mask: np.ndarray,
) -> np.ndarray:
    """Convex hull over cheek landmarks, intersected with tessellation skin mask."""
    ids = list(set(CHEEK_LANDMARKS + CHEEK_R_LANDMARKS))
    pts = np.array([[int(landmarks[i].x * w), int(landmarks[i].y * h)] for i in ids], dtype=np.int32)
    cheek = np.zeros((h, w), dtype=np.uint8)
    if pts.shape[0] >= 3:
        hull = cv2.convexHull(pts)
        cv2.fillConvexPoly(cheek, hull, 255)
    return cv2.bitwise_and(cheek, mesh_mask)
