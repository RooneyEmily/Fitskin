"""helpers/skin_regions.py

Clean interface for cheek and forehead mask creation from pre-computed landmark JSON files.

Workflow
--------
1.  Run ``export_landmark_json.py`` once locally (requires MediaPipe).
    Writes  landmarks/<participant>_<trial>.json  for every trial.

2.  Ship those JSON files with your app / notebook.
    No MediaPipe required at inference time.

3.  Call::

        lm   = load_landmarks('landmarks/participant1_trial2.json')
        mask = make_cheek_mask(h, w, lm)
        vis  = draw_overlay(img_linear, lm, cheek_mask=mask)

JSON schema
-----------
{
    "image":              "IMG_0787_NoFlash.DNG",
    "image_hw":           [2316, 3088],
    "left_eye":           [431, 382],
    "right_eye":          [689, 379],
    "nose":               [560, 501],
    "mouth":              [561, 670],
    "left_cheek_polygon": [[350,450], [360,520], ...],
    "right_cheek_polygon":[[...], ...],
    "forehead_polygon":   [[...], ...]
}

All coordinates are [col, row] in pixels at the resolution stored in ``image_hw``.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Load / save
# ---------------------------------------------------------------------------

def load_landmarks(json_path: str | Path) -> dict:
    """Load a landmark JSON file and return the dictionary."""
    with open(json_path) as f:
        return json.load(f)


def save_landmarks(out_path: str | Path, data: dict) -> None:
    """Write landmark dict to JSON (creates parent dirs if needed)."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Saved → {out_path}")


# ---------------------------------------------------------------------------
# Mask creation
# ---------------------------------------------------------------------------

def _polygon_mask(h: int, w: int, polygon: list[list[int]]) -> np.ndarray:
    """Boolean mask filled inside a polygon given as [[col,row], ...] points."""
    mask = np.zeros((h, w), dtype=np.uint8)
    pts  = np.array(polygon, dtype=np.int32)  # (N, 2)  col, row
    if len(pts) >= 3:
        cv2.fillConvexPoly(mask, cv2.convexHull(pts), 255)
    return mask.astype(bool)


def _scale_polygon(polygon: list, src_hw: tuple, dst_hw: tuple) -> list:
    """Scale polygon coordinates from src resolution to dst resolution."""
    sh, sw = src_hw
    dh, dw = dst_hw
    return [[int(c * dw / sw), int(r * dh / sh)] for c, r in polygon]


def make_cheek_mask(h: int, w: int, landmarks: dict) -> np.ndarray:
    """Boolean mask for both cheeks at resolution (h, w).

    Polygons are automatically rescaled if the stored image_hw differs from (h, w).
    """
    src_hw = tuple(landmarks["image_hw"])
    dst_hw = (h, w)

    left  = _scale_polygon(landmarks["left_cheek_polygon"],  src_hw, dst_hw)
    right = _scale_polygon(landmarks["right_cheek_polygon"], src_hw, dst_hw)

    mask = _polygon_mask(h, w, left) | _polygon_mask(h, w, right)
    return mask


def make_forehead_mask(h: int, w: int, landmarks: dict) -> np.ndarray:
    """Boolean mask for the forehead region at resolution (h, w)."""
    src_hw = tuple(landmarks["image_hw"])
    dst_hw = (h, w)

    poly = _scale_polygon(landmarks["forehead_polygon"], src_hw, dst_hw)
    return _polygon_mask(h, w, poly)


# ---------------------------------------------------------------------------
# Overlay visualisation
# ---------------------------------------------------------------------------

def draw_overlay(
    img_linear: np.ndarray,
    landmarks: dict,
    *,
    cheek_mask:    Optional[np.ndarray] = None,
    forehead_mask: Optional[np.ndarray] = None,
    show_keypoints: bool = True,
    show_polygons:  bool = True,
) -> np.ndarray:
    """Return a gamma-corrected uint8 BGR image with landmarks and masks overlaid.

    Parameters
    ----------
    img_linear   : (H, W, 3) float64 linear camera RGB in [0, 1]
    landmarks    : dict loaded by load_landmarks()
    cheek_mask   : optional boolean mask (H, W) — green tint
    forehead_mask: optional boolean mask (H, W) — blue tint
    show_keypoints: draw eye / nose / mouth circles
    show_polygons  : outline cheek + forehead polygons
    """
    h, w = img_linear.shape[:2]
    src_hw = tuple(landmarks["image_hw"])

    # Gamma for display, convert to BGR uint8
    vis = np.clip(img_linear ** (1/2.2), 0, 1)
    vis = (vis * 255).astype(np.uint8)
    vis = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)

    # --- tint masks --------------------------------------------------------
    if cheek_mask is not None:
        overlay            = vis.copy()
        overlay[cheek_mask] = np.clip(
            overlay[cheek_mask].astype(np.int16) + [0, 60, 0], 0, 255
        ).astype(np.uint8)
        vis = cv2.addWeighted(vis, 0.5, overlay, 0.5, 0)

    if forehead_mask is not None:
        overlay                  = vis.copy()
        overlay[forehead_mask]   = np.clip(
            overlay[forehead_mask].astype(np.int16) + [60, 0, 0], 0, 255
        ).astype(np.uint8)
        vis = cv2.addWeighted(vis, 0.5, overlay, 0.5, 0)

    def scale_pt(c, r):
        """Scale a (col, row) point from src_hw to (h, w)."""
        sh, sw = src_hw
        return (int(c * w / sw), int(r * h / sh))

    # --- keypoints ---------------------------------------------------------
    if show_keypoints:
        kp_colors = {
            "left_eye":  (0, 255, 255),
            "right_eye": (0, 255, 255),
            "nose":      (255, 0, 255),
            "mouth":     (255, 255, 0),
        }
        for key, color in kp_colors.items():
            if key in landmarks:
                c, r = landmarks[key]
                pt = scale_pt(c, r)
                cv2.circle(vis, pt, 8, color, -1)
                cv2.circle(vis, pt, 9, (255, 255, 255), 1)
                cv2.putText(vis, key.replace("_", " "), (pt[0]+10, pt[1]),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1, cv2.LINE_AA)

    # --- polygon outlines --------------------------------------------------
    if show_polygons:
        poly_colors = {
            "left_cheek_polygon":  (0, 200, 0),
            "right_cheek_polygon": (0, 200, 0),
            "forehead_polygon":    (200, 100, 0),
        }
        for key, color in poly_colors.items():
            if key in landmarks:
                pts = np.array([scale_pt(c, r) for c, r in landmarks[key]],
                               dtype=np.int32)
                hull = cv2.convexHull(pts)
                cv2.polylines(vis, [hull], isClosed=True, color=color, thickness=2)

    return vis
