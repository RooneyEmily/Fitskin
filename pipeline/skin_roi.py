"""Apple Vision cheek / forehead ROI masks for chart-free skin Lab."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


def load_apple_landmarks(path: Path) -> dict:
    with Path(path).open(encoding="utf-8") as f:
        return json.load(f)


def _region_map(face: dict) -> dict:
    return {str(r.get("name")): r for r in (face.get("regions") or []) if r.get("name")}


def _pts_px(region, src_w, src_h, dst_w, dst_h, origin_ll=True) -> np.ndarray:
    pts = region.get("imagePoints")
    if pts:
        xy = np.array([[float(p["x"]), float(p["y"])] for p in pts], dtype=np.float64)
    else:
        npts = region.get("normalizedImagePoints") or []
        if not npts:
            return np.zeros((0, 2), dtype=np.int32)
        xy = np.array(
            [[float(p["x"]) * src_w, float(p["y"]) * src_h] for p in npts],
            dtype=np.float64,
        )
    if origin_ll:
        xy[:, 1] = float(src_h) - xy[:, 1]
    if src_w != dst_w or src_h != dst_h:
        xy[:, 0] *= dst_w / float(src_w)
        xy[:, 1] *= dst_h / float(src_h)
    return np.round(xy).astype(np.int32)


def _fill_hull(h, w, pts) -> np.ndarray:
    m = np.zeros((h, w), dtype=np.uint8)
    if pts is None or len(pts) < 3:
        return m
    cv2.fillConvexPoly(m, cv2.convexHull(pts.reshape(-1, 1, 2)), 255)
    return m


def _fill_poly(h, w, pts) -> np.ndarray:
    m = np.zeros((h, w), dtype=np.uint8)
    if pts is None or len(pts) < 3:
        return m
    cv2.fillPoly(m, [pts.reshape(-1, 1, 2)], 255)
    return m


def _apple_face_landmark_ctx(landmarks: dict, dst_h: int, dst_w: int):
    """Shared Apple Vision landmark helpers for cheek / forehead ROI masks."""
    faces = landmarks.get("faces") or []
    if not faces:
        raise ValueError("No faces in landmark JSON")
    regions = _region_map(faces[0])
    isize = landmarks.get("imageSize") or {}
    src_w = int(isize.get("width") or dst_w)
    src_h = int(isize.get("height") or dst_h)
    origin_ll = "lower" in str(landmarks.get("coordinateOrigin") or "lowerLeft").lower()

    def pts(name):
        r = regions.get(name)
        return _pts_px(r, src_w, src_h, dst_w, dst_h, origin_ll) if r else np.zeros((0, 2), dtype=np.int32)

    parts = [p for p in (pts("faceContour"), pts("leftEyebrow"), pts("rightEyebrow")) if len(p)]
    hull_pts = np.vstack(parts) if parts else pts("allPoints")
    face = _fill_hull(dst_h, dst_w, hull_pts)
    for name in ("leftEye", "rightEye", "leftEyebrow", "rightEyebrow", "outerLips", "innerLips"):
        p = pts(name)
        if len(p) >= 3:
            excl = _fill_poly(dst_h, dst_w, p)
            k = max(3, int(0.01 * min(dst_h, dst_w)) | 1)
            excl = cv2.dilate(excl, np.ones((k, k), np.uint8), 1)
            face = cv2.bitwise_and(face, cv2.bitwise_not(excl))
    return face, hull_pts, pts


def apple_face_cheek_masks(landmarks: dict, dst_h: int, dst_w: int):
    face, hull_pts, pts = _apple_face_landmark_ctx(landmarks, dst_h, dst_w)
    eye_y, mouth_y = [], []
    for n in ("leftEye", "rightEye", "leftPupil", "rightPupil"):
        p = pts(n)
        if len(p):
            eye_y.extend(p[:, 1].tolist())
    for n in ("outerLips", "innerLips"):
        p = pts(n)
        if len(p):
            mouth_y.extend(p[:, 1].tolist())
    if eye_y and mouth_y:
        y0, y1 = float(np.median(eye_y)), float(np.median(mouth_y))
        top, bot = int(y0 + 0.15 * (y1 - y0)), int(y0 + 0.95 * (y1 - y0))
    else:
        ys = hull_pts[:, 1]
        top = int(ys.min() + 0.35 * (ys.max() - ys.min()))
        bot = int(ys.min() + 0.78 * (ys.max() - ys.min()))
    band = np.zeros_like(face)
    band[max(0, top) : min(dst_h, bot), :] = 255
    cheek = cv2.bitwise_and(face, band)
    nose = pts("nose") if len(pts("nose")) else pts("medianLine")
    if len(nose):
        cx = int(np.median(nose[:, 0]))
        half = max(4, int(0.04 * dst_w))
        cheek[:, max(0, cx - half) : min(dst_w, cx + half)] = 0
    return face, cheek


def apple_face_forehead_mask(landmarks: dict, dst_h: int, dst_w: int):
    """Boolean forehead mask from Apple Vision landmarks (FitSkin scan site)."""
    face, hull_pts, pts = _apple_face_landmark_ctx(landmarks, dst_h, dst_w)
    ys = hull_pts[:, 1]
    face_h = float(max(1.0, ys.max() - ys.min()))
    face_top = float(ys.min())
    brow_y, eye_y = [], []
    for n in ("leftEyebrow", "rightEyebrow"):
        p = pts(n)
        if len(p):
            brow_y.extend(p[:, 1].tolist())
    for n in ("leftEye", "rightEye", "leftPupil", "rightPupil"):
        p = pts(n)
        if len(p):
            eye_y.extend(p[:, 1].tolist())
    if brow_y:
        bot = int(np.median(brow_y) - 0.02 * face_h)
    elif eye_y:
        bot = int(np.median(eye_y) - 0.08 * face_h)
    else:
        bot = int(face_top + 0.35 * face_h)
    top = int(face_top + 0.28 * max(1.0, bot - face_top))
    band = np.zeros_like(face)
    band[max(0, top) : min(dst_h, bot), :] = 255
    return cv2.bitwise_and(face, band)


def refine_forehead_mask(
    mask: np.ndarray,
    linear_rgb: Optional[np.ndarray] = None,
    *,
    top_row_quantile: float = 0.20,
    min_chroma_rgb: float = 0.012,
) -> np.ndarray:
    """Drop hairline rows and dark, low-chroma pixels (hair) inside a forehead mask."""
    m = (np.asarray(mask) > 0).astype(np.uint8)
    ys, xs = np.where(m > 0)
    if len(ys) < 10:
        return m.astype(bool)

    y_cut = int(np.quantile(ys, float(np.clip(top_row_quantile, 0.0, 0.45))))
    m[: max(0, y_cut), :] = 0

    if linear_rgb is not None and linear_rgb.shape[:2] == m.shape:
        sel = m > 0
        pix = np.maximum(np.asarray(linear_rgb, dtype=np.float64)[sel], 0.0)
        if len(pix) >= 10:
            chroma = pix.max(axis=1) - pix.min(axis=1)
            luma = pix.max(axis=1)
            luma_floor = float(np.quantile(luma, 0.30))
            keep = (chroma >= float(min_chroma_rgb)) | (luma >= luma_floor)
            flat_idx = np.flatnonzero(sel)
            drop = flat_idx[~keep]
            m.reshape(-1)[drop] = 0

    return m.astype(bool)


def apple_face_skin_roi_mask(
    landmarks: dict,
    dst_h: int,
    dst_w: int,
    *,
    roi: str = "cheek",
    linear_rgb: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Return cheek or forehead boolean mask from Apple Vision landmarks."""
    roi_key = str(roi or "cheek").strip().lower()
    if roi_key == "forehead":
        mask = apple_face_forehead_mask(landmarks, dst_h, dst_w)
        return refine_forehead_mask(mask, linear_rgb)
    if roi_key == "cheek":
        return apple_face_cheek_masks(landmarks, dst_h, dst_w)[1]
    raise ValueError(f"unknown roi={roi!r} (expected cheek or forehead)")
