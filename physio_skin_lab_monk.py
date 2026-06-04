#!/usr/bin/env python3
"""
Physio calibration photos: white balance (ColorChecker) → MediaPipe face mesh →
**dense face-mesh tessellation** (same small triangles as standard MediaPipe topology, like
face-mesh RGB extraction) → **main facial skin** mask (excludes lips, eyes, eyebrows, irises with
dilated margin; triangle kept if its centroid lies inside the face-oval convex hull and outside
exclusion). Optional **oval-only Delaunay** via ``--skin-triangulation oval_delaunay`` (coarse legacy).
Mean L*a*b* (CIE Lab, **D65** illuminant via ``skimage.color.rgb2lab(..., illuminant='D65')``). Optional **L* histogram** per photo
and **trim darkest / brightest** masked pixels by quantile (``--skin-l-star-trim-lo`` / ``…-hi``),
optional **min chroma** ``--skin-min-chroma-ab`` to drop near-neutral (gray/black/white) pixels,
similar to luma filtering on face-mesh triangle CSVs.

Optional **skin sampling overlay** (``--write-skin-mask-overlays``): per-photo PNG with full
tessellation wireframe (gray), **mask** triangles (yellow outline), face-oval convex hull (white),
green tint = pixels averaged after morphological open, magenta tint = dilated exclusion (eyes/lips/…).

Optional: correlate mean L* with Monk Skin Tone (MST) ratings per participant (CSV).

Depends on:
  pip install opencv-contrib-python mediapipe scikit-image scipy matplotlib numpy colour_checker_detection

ColorChecker for WB: **segmentation** (``colour_checker_detection``, same as ``WB_videos/videos.py``)
first; falls back to OpenCV **mcc** if segmentation fails or the package is missing.

**Which file type:** use **JPEG/PNG exports** (``*.jpg`` …) with this script. ``cv2.imread`` does **not**
read Canon **.cr2** RAW; using RAW well needs ``rawpy``/dcraw + a demosaic + color pipeline, then export
to sRGB JPEG for the same ColorChecker + face path used here.

**One image per participant:** ``--one-image-per-participant`` with ``--pick-photo largest_chart`` (default)
prefers the frame where the ColorChecker occupies more of the image (usually more reliable WB). Alternatives:
``newest_jpg`` (latest timestamp in filename), ``first_sorted`` (alphabetical).

Override a participant's file after picking: ``--force-photo 14=20250421T152917Z.jpg`` (repeatable; path is
``<data-root>/<PID>/P1/Photos/<FILENAME>``).

Example (writes CSV/plots under ./skin_lab_output if --out-dir omitted)::

  python physio_skin_lab_monk.py --plot

  python physio_skin_lab_monk.py \\
    --data-root /media/mabl-main/Data/Physio-code/Data \\
    --one-image-per-participant --pick-photo largest_chart \\
    --out-dir ./skin_lab_outputs \\
    --plot

Optional Monk CSV (header: participant_id,monk with monk in 1..10)::

  python physio_skin_lab_monk.py --monk-csv monk_ratings.csv --plot
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import json
import os
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import cv2
import numpy as np

try:
    from scipy.spatial import Delaunay
    from scipy import stats as scipy_stats
except ImportError as e:
    raise SystemExit("Install scipy: pip install scipy") from e

try:
    from skimage import color as skcolor
except ImportError as e:
    raise SystemExit("Install scikit-image: pip install scikit-image") from e

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

try:
    import mediapipe as mp
except ImportError as e:
    raise SystemExit("Install mediapipe: pip install mediapipe") from e

from delta_e_2000 import de2000_csv_header, de2000_csv_values, load_mst_lab_matrix_10x3, mst_de2000_row


# ---------------------------------------------------------------------------
# White balance: colour_checker_detection (videos.py) first, then OpenCV mcc
# ---------------------------------------------------------------------------

try:
    from colour_checker_segmentation import detect_classic24_bgr as _detect_chart_seg_bgr
    from colour_checker_segmentation import library_available as _segmentation_lib_available
except ImportError:
    _detect_chart_seg_bgr = None  # type: ignore[misc, assignment]

    def _segmentation_lib_available() -> bool:
        return False

REFERENCE_WHITE_SRGB = np.array([243.0, 243.0, 242.0], dtype=np.float64)
# Classic 24-patch chart: neutral white patch index (see OpenCV mcc ordering).
WHITE_PATCH_INDEX = 18


def _rgb_from_charts_matrix(charts_rgb: np.ndarray, patch_index: int) -> Optional[np.ndarray]:
    """
    Decode getChartsRGB() layout. OpenCV 4.x often returns shape (72, 5): 24 patches × 3 rows
    (R, G, B channel statistics); column 1 holds the per-channel mean used for WB.
    Fallback: single row per patch (e.g. 24×N) with RGB in columns 1:4 or last three columns.
    """
    nrows, ncols = charts_rgb.shape[0], charts_rgb.shape[1]
    if nrows % 3 == 0 and nrows >= 3 * (patch_index + 1) and ncols >= 2:
        # e.g. 72×5: triplets per patch (R, G, B statistics rows)
        base = 3 * patch_index
        rrow, grow, brow = charts_rgb[base], charts_rgb[base + 1], charts_rgb[base + 2]
        if ncols >= 2:
            return np.array([float(rrow[1]), float(grow[1]), float(brow[1])], dtype=np.float64)
        return None
    if nrows <= patch_index:
        return None
    row = charts_rgb[patch_index]
    white_rgb = None
    if ncols >= 4:
        cand = np.array(row[1:4], dtype=np.float64)
        if np.all((cand > 10) & (cand < 300)):
            white_rgb = cand
    if white_rgb is None and ncols >= 3:
        cand = np.array(row[-3:], dtype=np.float64)
        if np.all((cand > 10) & (cand < 300)):
            white_rgb = cand
    return white_rgb

def check_mcc() -> bool:
    try:
        cv2.mcc.CCheckerDetector.create()
        return True
    except Exception:
        return False


def measured_white_from_chart_bgr(frame_bgr: np.ndarray, debug: bool = False) -> Optional[np.ndarray]:
    """Measured white patch RGB (float, ~0–255), or None. Segmentation (videos.py path) first."""
    if _segmentation_lib_available() and _detect_chart_seg_bgr is not None:
        try:
            seg = _detect_chart_seg_bgr(frame_bgr)
            if seg is not None:
                if debug:
                    print("ColorChecker: colour_checker_detection segmentation (videos.py path)", file=sys.stderr)
                return seg["white_rgb_255"]
        except Exception as ex:
            if debug:
                print(f"segmentation chart detection error: {ex}", file=sys.stderr)
    if not hasattr(cv2, "mcc"):
        if debug:
            print("opencv mcc missing — pip install opencv-contrib-python", file=sys.stderr)
        return None
    try:
        detector = cv2.mcc.CCheckerDetector.create()
        # Positional args required by OpenCV Python bindings (e.g. 4.11): chartType, nc
        ok = detector.process(frame_bgr, 0, 1)
        if not ok:
            return None
        checkers = detector.getListColorChecker()
        if not checkers:
            return None
        checker = checkers[0]
        charts_rgb = checker.getChartsRGB()
        if charts_rgb is None or len(charts_rgb.shape) < 1:
            return None
        white_rgb = _rgb_from_charts_matrix(charts_rgb, WHITE_PATCH_INDEX)
        if white_rgb is None:
            return None
        if float(np.max(white_rgb)) <= 1.0:
            white_rgb = white_rgb * 255.0
        return white_rgb
    except Exception as ex:
        if debug:
            print(f"mcc detection error: {ex}", file=sys.stderr)
        return None


def _mcc_chart_area_fraction(frame_bgr: np.ndarray) -> float:
    """Convex hull area of mcc getBox() / image area; 0 if no chart."""
    if not hasattr(cv2, "mcc"):
        return 0.0
    try:
        detector = cv2.mcc.CCheckerDetector.create()
        if not detector.process(frame_bgr, 0, 1):
            return 0.0
        checkers = detector.getListColorChecker()
        if not checkers:
            return 0.0
        checker = checkers[0]
        if not hasattr(checker, "getBox"):
            return 0.0
        box = np.asarray(checker.getBox(), dtype=np.float64)
        if box.size < 8:
            return 0.0
        h, w = frame_bgr.shape[:2]
        poly = box.reshape(-1, 1, 2).astype(np.float32)
        return float(cv2.contourArea(poly)) / float(max(w * h, 1))
    except Exception:
        return 0.0


def chart_area_score_bgr(frame_bgr: np.ndarray) -> float:
    """
    Proxy for “good calibration frame”: ColorChecker area / frame (segmentation quad or mcc box).
    Larger usually means chart is closer / easier WB (same idea as validate_chart scan mode).
    """
    if _segmentation_lib_available() and _detect_chart_seg_bgr is not None:
        try:
            seg = _detect_chart_seg_bgr(frame_bgr)
            if seg is not None:
                return float(seg["chart_area_fraction"])
        except Exception:
            pass
    return _mcc_chart_area_fraction(frame_bgr)


def white_balance_multipliers(measured_white_rgb: np.ndarray) -> np.ndarray:
    m = np.maximum(measured_white_rgb.astype(np.float64), 1e-6)
    return REFERENCE_WHITE_SRGB / m


def apply_wb_bgr(frame_bgr: np.ndarray, mult_rgb: np.ndarray) -> np.ndarray:
    """Apply per-channel gains in **8-bit display (gamma-encoded) space**; OpenCV order is BGR."""
    x = frame_bgr.astype(np.float64)
    x[:, :, 2] *= mult_rgb[0]
    x[:, :, 1] *= mult_rgb[1]
    x[:, :, 0] *= mult_rgb[2]
    return np.clip(x, 0, 255).astype(np.uint8)


def _srgb_array_to_linear(rgb: np.ndarray) -> np.ndarray:
    """IEC 61966-2-1 sRGB non-linear RGB (0–1) → linear-light RGB, same shape (…, 3)."""
    x = np.clip(rgb.astype(np.float64), 0.0, 1.0)
    lo = x <= 0.04045
    out = np.empty_like(x, dtype=np.float64)
    out[lo] = x[lo] / 12.92
    out[~lo] = ((x[~lo] + 0.055) / 1.055) ** 2.4
    return out


def _linear_array_to_srgb(lin: np.ndarray) -> np.ndarray:
    """Linear-light RGB (0–1) → sRGB non-linear (0–1)."""
    x = np.clip(lin.astype(np.float64), 0.0, 1.0)
    lo = x <= 0.0031308
    out = np.empty_like(x, dtype=np.float64)
    out[lo] = 12.92 * x[lo]
    out[~lo] = 1.055 * np.power(x[~lo], 1.0 / 2.4) - 0.055
    return np.clip(out, 0.0, 1.0)


def white_balance_multipliers_linear_srgb(measured_white_rgb: np.ndarray) -> np.ndarray:
    """
    Per-channel gains to apply in **linear** sRGB (matches skimage Lab’s internal linearization).

    ``measured_white_rgb`` is still the chart readout in 0–255 display code values (same as the
    default path); we linearize measured and reference whites before dividing.
    """
    m = np.maximum(measured_white_rgb.astype(np.float64), 1e-6) / 255.0
    m_lin = _srgb_array_to_linear(m.reshape(1, 1, 3)).reshape(3)
    ref = (REFERENCE_WHITE_SRGB.astype(np.float64) / 255.0).reshape(1, 1, 3)
    ref_lin = _srgb_array_to_linear(ref).reshape(3)
    return ref_lin / np.maximum(m_lin, 1e-8)


def apply_wb_bgr_linear_srgb(frame_bgr: np.ndarray, mult_lin: np.ndarray) -> np.ndarray:
    """WB in linear sRGB, then encode to 8-bit for downstream OpenCV / MediaPipe."""
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB).astype(np.float64) / 255.0
    lin = _srgb_array_to_linear(rgb) * mult_lin.reshape(1, 1, 3)
    srgb = _linear_array_to_srgb(lin)
    u8 = np.clip(np.round(srgb * 255.0), 0, 255).astype(np.uint8)
    return cv2.cvtColor(u8, cv2.COLOR_RGB2BGR)


# ---------------------------------------------------------------------------
# MediaPipe face mesh + skin mask (tessellation or oval Delaunay)
# ---------------------------------------------------------------------------

_TESSELLATION_TRIS_CACHE: Optional[np.ndarray] = None


def _landmark_xy(lm, w: int, h: int) -> Tuple[int, int]:
    return int(lm.x * w), int(lm.y * h)


def _enumerate_facemesh_tessellation() -> np.ndarray:
    """
    MediaPipe ``FACEMESH_TESSELATION`` is an edge set; recover unique triangles (854 for 468-pt mesh).
    Cached after first call.
    """
    global _TESSELLATION_TRIS_CACHE
    if _TESSELLATION_TRIS_CACHE is not None:
        return _TESSELLATION_TRIS_CACHE
    edges = mp.solutions.face_mesh_connections.FACEMESH_TESSELATION
    adj: Dict[int, set] = defaultdict(set)
    for a, b in edges:
        adj[int(a)].add(int(b))
        adj[int(b)].add(int(a))
    tris: set[tuple[int, int, int]] = set()
    for a, b in edges:
        a, b = int(a), int(b)
        if a < b:
            for c in adj[a] & adj[b]:
                sa, sb, sc = sorted((a, b, int(c)))
                tris.add((sa, sb, sc))
    _TESSELLATION_TRIS_CACHE = np.array(sorted(tris), dtype=np.int32)
    return _TESSELLATION_TRIS_CACHE


def _collect_indices_from_connections(conn_list: Sequence[Tuple[int, int]]) -> np.ndarray:
    idx = set()
    for a, b in conn_list:
        idx.add(a)
        idx.add(b)
    return np.array(sorted(idx), dtype=np.int32)


def _morphology_radius_from_iod(
    iod_px: float,
    *,
    frac: float,
    min_px: int = 2,
    max_px: int = 20,
) -> int:
    """Kernel radius (pixels) from inter-ocular distance; used for skin-mask open/close."""
    if not np.isfinite(iod_px) or iod_px <= 0.0:
        return min_px
    return int(max(min_px, min(max_px, round(float(iod_px) * float(frac)))))


def build_skin_mask_from_mesh(
    h: int,
    w: int,
    landmarks: Sequence,
    *,
    skin_triangulation: str = "tessellation",
    exclusion_dilate_iod_fraction: float = 0.12,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], np.ndarray, Optional[np.ndarray]]:
    """
    Returns ``(skin_binary_mask uint8 0|255, oval_pts, kept_tris, excl_dil, mesh_xy)``.

    ``mesh_xy`` is ``(V, 2)`` landmark image coordinates for tessellation indices (``None`` if
    ``skin_triangulation == "oval_delaunay"``).

    ``kept_tris`` is ``(K, 3)`` int vertex indices: **landmark indices** (tessellation) or **indices
    into oval_pts** (oval_delaunay). Triangles are filled if centroid is inside the face-oval convex
    hull (when hull exists) and not in ``excl_dil`` (tessellation), or Delaunay-in-oval rules
    (oval_delaunay). ``None`` if no triangles were kept / hull fallback only.
    ``excl_dil`` is the dilated lip/eye/eyebrow/iris exclusion mask (for debug overlays).

    After triangle fill, the mask is opened then closed with ellipse kernels whose radii scale
    with inter-ocular distance (~1.5% / ~2.5% of IOD), then intersected again with
    ``~excl_dil`` so closing cannot encroach on the exclusion zone.
    """
    if skin_triangulation not in ("tessellation", "oval_delaunay"):
        raise ValueError(f"skin_triangulation must be 'tessellation' or 'oval_delaunay', got {skin_triangulation!r}")

    conn = mp.solutions.face_mesh_connections

    oval_idx = _collect_indices_from_connections(conn.FACEMESH_FACE_OVAL)
    lip_idx = _collect_indices_from_connections(conn.FACEMESH_LIPS)
    le_idx = _collect_indices_from_connections(conn.FACEMESH_LEFT_EYE)
    re_idx = _collect_indices_from_connections(conn.FACEMESH_RIGHT_EYE)
    lb_idx = _collect_indices_from_connections(getattr(conn, "FACEMESH_LEFT_EYEBROW", ()))
    rb_idx = _collect_indices_from_connections(getattr(conn, "FACEMESH_RIGHT_EYEBROW", ()))
    li_idx = _collect_indices_from_connections(getattr(conn, "FACEMESH_LEFT_IRIS", ()))
    ri_idx = _collect_indices_from_connections(getattr(conn, "FACEMESH_RIGHT_IRIS", ()))

    def pts(indices: np.ndarray) -> np.ndarray:
        if indices.size == 0:
            return np.zeros((0, 2), dtype=np.float32)
        return np.array([_landmark_xy(landmarks[i], w, h) for i in indices], dtype=np.float32)

    oval_pts = pts(oval_idx)
    lip_pts = pts(lip_idx)
    le_pts = pts(le_idx)
    re_pts = pts(re_idx)
    lb_pts = pts(lb_idx)
    rb_pts = pts(rb_idx)
    li_pts = pts(li_idx)
    ri_pts = pts(ri_idx)

    def _fill_hull(canvas: np.ndarray, arr: np.ndarray) -> None:
        if len(arr) >= 3:
            hull = cv2.convexHull(arr.astype(np.float32))
            cv2.fillConvexPoly(canvas, hull.astype(np.int32), 255)

    excl = np.zeros((h, w), dtype=np.uint8)
    for arr in (lip_pts, le_pts, re_pts, lb_pts, rb_pts, li_pts, ri_pts):
        _fill_hull(excl, arr)

    def _inter_ocular() -> float:
        if len(le_pts) >= 2 and len(re_pts) >= 2:
            lc = le_pts.mean(axis=0)
            rc = re_pts.mean(axis=0)
            return float(np.linalg.norm(lc - rc))
        return float(max(h, w)) * 0.22

    iod = _inter_ocular()
    rad = int(round(exclusion_dilate_iod_fraction * iod))
    rad = max(6, min(rad, 56))
    ksz = 2 * rad + 1
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksz, ksz))
    excl_dil = cv2.dilate(excl, ker, iterations=1)

    mask = np.zeros((h, w), dtype=np.uint8)
    kept_tris: Optional[np.ndarray] = None
    kept_list: List[np.ndarray] = []
    mesh_xy: Optional[np.ndarray] = None

    oval_hull = None
    if len(oval_pts) >= 3:
        oval_hull = cv2.convexHull(oval_pts.astype(np.float32))

    if skin_triangulation == "tessellation":
        T = _enumerate_facemesh_tessellation()
        vmax = int(T.max()) + 1
        mesh_xy = np.array([_landmark_xy(landmarks[i], w, h) for i in range(vmax)], dtype=np.float32)
        for tri in T:
            poly = mesh_xy[tri].astype(np.int32)
            cx = float(poly[:, 0].mean())
            cy = float(poly[:, 1].mean())
            xi, yi = int(round(cx)), int(round(cy))
            if not (0 <= yi < h and 0 <= xi < w):
                continue
            if excl_dil[yi, xi] > 0:
                continue
            if oval_hull is not None and cv2.pointPolygonTest(oval_hull, (cx, cy), False) < 0:
                continue
            cv2.fillConvexPoly(mask, poly, 255)
            kept_list.append(tri.astype(np.int32))
        if kept_list:
            kept_tris = np.stack(kept_list, axis=0)
    else:
        if len(oval_pts) >= 4:
            tri = Delaunay(oval_pts)
            for simplex in tri.simplices:
                poly = oval_pts[simplex].astype(np.int32)
                cx, cy = float(poly[:, 0].mean()), float(poly[:, 1].mean())
                xi, yi = int(round(cx)), int(round(cy))
                if 0 <= yi < h and 0 <= xi < w and excl_dil[yi, xi] > 0:
                    continue
                cv2.fillConvexPoly(mask, poly, 255)
                kept_list.append(simplex.astype(np.int32))
            if kept_list:
                kept_tris = np.stack(kept_list, axis=0)
        elif len(oval_pts) >= 3 and oval_hull is not None:
            cv2.fillConvexPoly(mask, oval_hull.astype(np.int32), 255)

    mask = cv2.bitwise_and(mask, cv2.bitwise_not(excl_dil))

    open_r = _morphology_radius_from_iod(iod, frac=0.015)
    close_r = _morphology_radius_from_iod(iod, frac=0.025)
    open_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * open_r + 1, 2 * open_r + 1))
    close_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * close_r + 1, 2 * close_r + 1))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_k, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_k, iterations=1)
    mask = cv2.bitwise_and(mask, cv2.bitwise_not(excl_dil))

    return mask, oval_pts, kept_tris, excl_dil, mesh_xy


def write_skin_sampling_overlay_png(
    out_path: Path,
    wb_bgr: np.ndarray,
    oval_pts: np.ndarray,
    kept_tris: Optional[np.ndarray],
    mask: np.ndarray,
    excl_dil: np.ndarray,
    *,
    mesh_xy: Optional[np.ndarray] = None,
    max_width: int = 1600,
) -> None:
    """
    Save a debug PNG: WB image + **final skin mask** (green tint) + **dilated exclusion** (magenta tint)
    + wireframe: full tessellation (gray) when ``mesh_xy`` is set, else oval Delaunay (gray);
    **kept** mask triangles (yellow); face-oval convex hull (white).

    If ``max_width`` > 0 and the frame is wider, downscale for file size (landmarks scaled to match).
    """
    img = wb_bgr
    msk = mask
    excl = excl_dil
    oval = oval_pts.astype(np.float32)
    mesh = None if mesh_xy is None else mesh_xy.astype(np.float32)
    h0, w0 = img.shape[:2]
    if max_width > 0 and w0 > max_width:
        scale = max_width / float(w0)
        nw, nh = max_width, int(round(h0 * scale))
        img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
        msk = cv2.resize(msk, (nw, nh), interpolation=cv2.INTER_NEAREST)
        excl = cv2.resize(excl, (nw, nh), interpolation=cv2.INTER_NEAREST)
        oval = (oval * scale).astype(np.float32)
        if mesh is not None:
            mesh = (mesh * scale).astype(np.float32)

    vis = img.astype(np.float64)
    m = msk > 0
    ex_only = (excl > 0) & (~m)
    vis[m] = vis[m] * 0.52 + np.array([0.0, 210.0, 40.0], dtype=np.float64) * 0.48
    vis[ex_only] = vis[ex_only] * 0.55 + np.array([180.0, 40.0, 220.0], dtype=np.float64) * 0.45
    vis = np.clip(vis, 0.0, 255.0).astype(np.uint8)

    if mesh is not None and mesh.shape[0] >= 4:
        T = _enumerate_facemesh_tessellation()
        for tri in T:
            poly = mesh[tri].astype(np.int32)
            cv2.polylines(vis, [poly], True, (55, 55, 55), 1)
        if kept_tris is not None and kept_tris.size > 0:
            for si in range(kept_tris.shape[0]):
                poly = mesh[kept_tris[si]].astype(np.int32)
                cv2.polylines(vis, [poly], True, (0, 255, 255), 1)
    else:
        if oval.shape[0] >= 4:
            tri = Delaunay(oval)
            for sim in tri.simplices:
                poly = oval[sim].astype(np.int32)
                cv2.polylines(vis, [poly], True, (55, 55, 55), 1)
        if kept_tris is not None and kept_tris.size > 0:
            for si in range(kept_tris.shape[0]):
                poly = oval[kept_tris[si]].astype(np.int32)
                cv2.polylines(vis, [poly], True, (0, 255, 255), 2)
    if oval.shape[0] >= 3:
        hull = cv2.convexHull(oval.astype(np.float32))
        cv2.polylines(vis, [hull.astype(np.int32)], True, (255, 255, 255), 1)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), vis)


def masked_lab_flatten(img_bgr_wb: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Masked pixels → 1D L*, a*, b* via ``skimage.color.rgb2lab`` (**D65** illuminant, CIE 1931 2° observer)."""
    m = mask > 0
    if not np.any(m):
        return (
            np.zeros(0, dtype=np.float64),
            np.zeros(0, dtype=np.float64),
            np.zeros(0, dtype=np.float64),
        )
    rgb = cv2.cvtColor(img_bgr_wb, cv2.COLOR_BGR2RGB).astype(np.float64) / 255.0
    pix = rgb[m].reshape(-1, 3)
    # (N,1,3) for skimage < 0.19 compatibility; newer skimage accepts (N,3).
    lab = skcolor.rgb2lab(
        pix.reshape(-1, 1, 3),
        illuminant="D65",
        observer="2",
    ).reshape(-1, 3)
    return (
        lab[:, 0].astype(np.float64),
        lab[:, 1].astype(np.float64),
        lab[:, 2].astype(np.float64),
    )


def _clip_skin_trim_q(q: float) -> float:
    if q <= 0.0:
        return 0.0
    return min(float(q), 0.45)


def _apply_channel_quantile_trim(
    sel: np.ndarray,
    channel: np.ndarray,
    trim_lo: float,
    trim_hi: float,
) -> np.ndarray:
    """Keep pixels inside [Q_lo, Q_hi] quantile bounds on one Lab channel."""
    out = sel.copy()
    tlo = _clip_skin_trim_q(trim_lo)
    thi = _clip_skin_trim_q(trim_hi)
    if tlo > 0.0:
        out &= channel >= float(np.quantile(channel, tlo))
    if thi > 0.0:
        out &= channel <= float(np.quantile(channel, 1.0 - thi))
    return out


def skin_lab_trim_selection(
    L_flat: np.ndarray,
    a_flat: np.ndarray,
    b_flat: np.ndarray,
    *,
    l_star_trim_lo: float = 0.0,
    l_star_trim_hi: float = 0.0,
    a_star_trim_lo: float = 0.0,
    a_star_trim_hi: float = 0.0,
    b_star_trim_lo: float = 0.0,
    b_star_trim_hi: float = 0.0,
    min_chroma_ab: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, bool, bool, bool, bool]:
    """
    Boolean masks over flattened skin Lab samples.

    Returns ``(sel_final, sel_channel_trim, chroma_relaxed, channel_trim_relaxed,
    channel_trim_effective, chroma_trim_effective)``.

    ``sel_channel_trim`` = after L*/a*/b* quantile gates, before ``min_chroma_ab``.
    If too few pixels remain (``< max(2000, n/25)``), chroma trim is dropped first, then all channel trims.
    """
    n_raw = int(L_flat.size)
    sel_ch = np.ones(n_raw, dtype=bool)
    sel_ch = _apply_channel_quantile_trim(sel_ch, L_flat, l_star_trim_lo, l_star_trim_hi)
    sel_ch = _apply_channel_quantile_trim(sel_ch, a_flat, a_star_trim_lo, a_star_trim_hi)
    sel_ch = _apply_channel_quantile_trim(sel_ch, b_flat, b_star_trim_lo, b_star_trim_hi)

    sel = sel_ch.copy()
    if min_chroma_ab > 0.0:
        sel &= np.hypot(a_flat, b_flat) >= float(min_chroma_ab)

    min_keep = max(2000, n_raw // 25)
    chroma_relaxed = False
    channel_trim_relaxed = False
    n_kept = int(np.count_nonzero(sel))
    if n_kept < min_keep and min_chroma_ab > 0.0:
        chroma_relaxed = True
        sel = sel_ch.copy()
        n_kept = int(np.count_nonzero(sel))
    if n_kept < min_keep:
        channel_trim_relaxed = True
        sel = np.ones(n_raw, dtype=bool)
        n_kept = n_raw

    any_channel_trim = any(
        x > 0.0
        for x in (
            l_star_trim_lo,
            l_star_trim_hi,
            a_star_trim_lo,
            a_star_trim_hi,
            b_star_trim_lo,
            b_star_trim_hi,
        )
    )
    channel_trim_effective = any_channel_trim and not channel_trim_relaxed
    chroma_trim_effective = (min_chroma_ab > 0.0) and not chroma_relaxed and not channel_trim_relaxed
    return sel, sel_ch, chroma_relaxed, channel_trim_relaxed, channel_trim_effective, chroma_trim_effective


def mean_lab_masked(
    img_bgr_wb: np.ndarray,
    mask: np.ndarray,
    *,
    l_star_trim_lo: float = 0.0,
    l_star_trim_hi: float = 0.0,
    a_star_trim_lo: float = 0.0,
    a_star_trim_hi: float = 0.0,
    b_star_trim_lo: float = 0.0,
    b_star_trim_hi: float = 0.0,
    min_chroma_ab: float = 0.0,
    histogram_png: Optional[Path] = None,
    histogram_title: str = "",
) -> Tuple[float, float, float, int, int, float, float, bool, bool, bool, bool, float, float, float]:
    """
    sRGB image [0,255] → Lab; mean over mask pixels, optionally after trimming L* tails.

    ``l_star_trim_lo`` / ``l_star_trim_hi``: fractions in ``[0, 0.45)`` of **masked** L* distribution
    to drop from the bottom / top before averaging (reduces shadows, beard, specular), same idea as
    luma quantiles on face-mesh triangle CSVs.

    ``a_star_trim_lo`` / ``a_star_trim_hi`` and ``b_star_trim_lo`` / ``b_star_trim_hi``: same quantile
    trimming on a* and b* (e.g. 0.05 drops the lowest 5% and highest 5% of each channel).

    ``min_chroma_ab``: if > 0, keep only pixels with C*_ab = hypot(a*,b*) ≥ this value (drops
    near-neutral grays toward black/white in chroma, not only L*). This **biases** ``a_mean`` /
    ``b_mean`` toward more saturated pixels vs. the ``*_ltrim_only`` means (mean over the L*-trimmed
    set **before** the chroma gate).

    Returns final ``(L_mean, a_mean, b_mean, …)`` plus ``L_mean_ltrim_only``, ``a_mean_ltrim_only``,
    ``b_mean_ltrim_only`` (mean over channel-trimmed pixels **before** the chroma gate).
    """
    L_flat, a_flat, b_flat = masked_lab_flatten(img_bgr_wb, mask)
    n_raw = int(L_flat.size)
    if n_raw == 0:
        return (
            float("nan"),
            float("nan"),
            float("nan"),
            0,
            0,
            float("nan"),
            float("nan"),
            False,
            False,
            False,
            False,
            float("nan"),
            float("nan"),
            float("nan"),
        )

    lo_thr = float("nan")
    hi_thr = float("nan")
    tlo = float(l_star_trim_lo)
    thi = float(l_star_trim_hi)
    if tlo > 0.0:
        lo_thr = float(np.quantile(L_flat, min(tlo, 0.45)))
    if thi > 0.0:
        hi_thr = float(np.quantile(L_flat, 1.0 - min(thi, 0.45)))

    sel, sel_ch, chroma_relaxed, ltrim_relaxed, l_trim_effective, chroma_trim_effective = skin_lab_trim_selection(
        L_flat,
        a_flat,
        b_flat,
        l_star_trim_lo=l_star_trim_lo,
        l_star_trim_hi=l_star_trim_hi,
        a_star_trim_lo=a_star_trim_lo,
        a_star_trim_hi=a_star_trim_hi,
        b_star_trim_lo=b_star_trim_lo,
        b_star_trim_hi=b_star_trim_hi,
        min_chroma_ab=min_chroma_ab,
    )

    if np.any(sel_ch):
        L_mean_ltrim_only = float(L_flat[sel_ch].mean())
        a_mean_ltrim_only = float(a_flat[sel_ch].mean())
        b_mean_ltrim_only = float(b_flat[sel_ch].mean())
    else:
        L_mean_ltrim_only = float("nan")
        a_mean_ltrim_only = float("nan")
        b_mean_ltrim_only = float("nan")

    n_kept = int(np.count_nonzero(sel))

    if histogram_png is not None and plt is not None:
        write_skin_lab_histogram_panel(
            histogram_png,
            L_flat,
            a_flat,
            b_flat,
            sel,
            lo_thr=lo_thr,
            hi_thr=hi_thr,
            min_chroma_ab=min_chroma_ab,
            l_trim_relaxed=ltrim_relaxed,
            chroma_relaxed=chroma_relaxed,
            title=histogram_title or "Skin Lab (masked)",
            lab_source="skimage D65",
        )

    if n_kept == 0:
        return (
            float("nan"),
            float("nan"),
            float("nan"),
            0,
            n_raw,
            lo_thr,
            hi_thr,
            l_trim_effective,
            chroma_trim_effective,
            chroma_relaxed,
            ltrim_relaxed,
            L_mean_ltrim_only,
            a_mean_ltrim_only,
            b_mean_ltrim_only,
        )
    return (
        float(L_flat[sel].mean()),
        float(a_flat[sel].mean()),
        float(b_flat[sel].mean()),
        n_kept,
        n_raw,
        lo_thr,
        hi_thr,
        l_trim_effective,
        chroma_trim_effective,
        chroma_relaxed,
        ltrim_relaxed,
        L_mean_ltrim_only,
        a_mean_ltrim_only,
        b_mean_ltrim_only,
    )


def write_skin_lab_histogram_panel(
    out_path: Path,
    L_flat: np.ndarray,
    a_flat: np.ndarray,
    b_flat: np.ndarray,
    sel: np.ndarray,
    *,
    lo_thr: float,
    hi_thr: float,
    min_chroma_ab: float,
    l_trim_relaxed: bool,
    chroma_relaxed: bool,
    title: str,
    lab_source: str = "D65 Lab",
) -> None:
    """2×2 L*, C*_ab, a*, b* binning; tan = pixels kept for mean, gray = dropped by trim gates."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    C = np.hypot(a_flat, b_flat)
    dropped = ~sel
    nb = int(np.clip(len(L_flat) // 400, 24, 100))
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)

    def _panel(ax, values: np.ndarray, xlabel: str, vlines: list) -> None:
        if np.any(sel):
            ax.hist(values[sel], bins=nb, color="tan", edgecolor="k", alpha=0.85, label="kept for mean")
        if np.any(dropped):
            ax.hist(values[dropped], bins=nb, color="0.75", edgecolor="none", alpha=0.45, label="dropped")
        for x, color, lab in vlines:
            if np.isfinite(x):
                ax.axvline(x, color=color, ls="--", lw=1.4, label=lab)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("count")
        ax.legend(fontsize=7, loc="upper right")

    l_lines: list = []
    if not l_trim_relaxed:
        if np.isfinite(lo_thr):
            l_lines.append((lo_thr, "tab:blue", f"L* ≥ {lo_thr:.1f}"))
        if np.isfinite(hi_thr):
            l_lines.append((hi_thr, "tab:red", f"L* ≤ {hi_thr:.1f}"))
    _panel(axes[0, 0], L_flat, f"L* ({lab_source})", l_lines)

    c_keep = float(min_chroma_ab) if min_chroma_ab > 0 else float("nan")
    c_lines: list = []
    if not chroma_relaxed and np.isfinite(c_keep):
        c_lines.append((c_keep, "tab:purple", f"C* ≥ {c_keep:.1f}"))
    _panel(axes[0, 1], C, "C*_ab = hypot(a*, b*)", c_lines)
    _panel(axes[1, 0], a_flat, "a*", [])
    _panel(axes[1, 1], b_flat, "b*", [])

    note = f"n={int(np.count_nonzero(sel))}/{L_flat.size} pixels in mean"
    if l_trim_relaxed:
        note += "; L* trim relaxed"
    if chroma_relaxed:
        note += "; chroma trim relaxed"
    fig.suptitle(f"{title}\n{note}", fontsize=10)
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


@dataclass
class PhotoResult:
    participant_id: str
    image_path: str
    L_mean: float
    a_mean: float
    b_mean: float
    n_pixels: int
    wb_ok: bool
    n_pixels_raw_mask: int = 0
    l_star_trim_lo_frac: float = 0.0
    l_star_trim_hi_frac: float = 0.0
    l_star_threshold_lo: float = float("nan")
    l_star_threshold_hi: float = float("nan")
    skin_min_chroma_ab: float = 0.0
    skin_l_trim_effective: bool = True
    skin_chroma_trim_effective: bool = True
    L_mean_ltrim_only: float = float("nan")
    a_mean_ltrim_only: float = float("nan")
    b_mean_ltrim_only: float = float("nan")
    wb_linear_srgb: bool = False


def process_one_image(
    path: Path,
    face_mesh: Any,
    *,
    skip_wb_if_no_chart: bool = False,
    debug: bool = False,
    skin_exclusion_dilate_iod_fraction: float = 0.12,
    skin_triangulation: str = "tessellation",
    l_star_trim_lo: float = 0.0,
    l_star_trim_hi: float = 0.0,
    a_star_trim_lo: float = 0.0,
    a_star_trim_hi: float = 0.0,
    b_star_trim_lo: float = 0.0,
    b_star_trim_hi: float = 0.0,
    skin_min_chroma_ab: float = 0.0,
    skin_l_histogram_dir: Optional[Path] = None,
    skin_overlay_dir: Optional[Path] = None,
    skin_overlay_max_width: int = 1600,
    wb_linear_srgb: bool = False,
) -> Optional[PhotoResult]:
    bgr = cv2.imread(str(path))
    if bgr is None:
        return None
    h, w = bgr.shape[:2]
    pid = path.parent.parent.parent.name  # .../Data/<pid>/P1/Photos/file.jpg

    measured = measured_white_from_chart_bgr(bgr, debug=debug)
    if measured is None:
        if skip_wb_if_no_chart:
            wb_bgr = bgr
            wb_ok = False
        else:
            return None
    else:
        if wb_linear_srgb:
            mult = white_balance_multipliers_linear_srgb(measured)
            wb_bgr = apply_wb_bgr_linear_srgb(bgr, mult)
        else:
            mult = white_balance_multipliers(measured)
            wb_bgr = apply_wb_bgr(bgr, mult)
        wb_ok = True

    rgb = cv2.cvtColor(wb_bgr, cv2.COLOR_BGR2RGB)
    rgb.flags.writeable = False
    results = face_mesh.process(rgb)
    if not results.multi_face_landmarks:
        return None
    lm = results.multi_face_landmarks[0].landmark

    mask, oval_pts, kept_tris, excl_dil, mesh_xy = build_skin_mask_from_mesh(
        h,
        w,
        lm,
        skin_triangulation=skin_triangulation,
        exclusion_dilate_iod_fraction=skin_exclusion_dilate_iod_fraction,
    )
    if skin_overlay_dir is not None:
        skin_overlay_dir.mkdir(parents=True, exist_ok=True)
        safe = path.stem.replace("/", "_")
        ovp = skin_overlay_dir / f"pid{pid}_{safe}_skin_delaunay_overlay.png"
        write_skin_sampling_overlay_png(
            ovp,
            wb_bgr,
            oval_pts,
            kept_tris,
            mask,
            excl_dil,
            mesh_xy=mesh_xy,
            max_width=skin_overlay_max_width,
        )
    hist_png: Optional[Path] = None
    if skin_l_histogram_dir is not None:
        skin_l_histogram_dir.mkdir(parents=True, exist_ok=True)
        safe = path.stem.replace("/", "_")
        hist_png = skin_l_histogram_dir / f"pid{pid}_{safe}_skin_lab_hists.png"
    (
        L,
        a,
        bb,
        npx,
        nraw,
        lo_thr,
        hi_thr,
        l_trim_ok,
        chroma_ok,
        chroma_relaxed,
        ltrim_relaxed,
        L_lt,
        a_lt,
        b_lt,
    ) = mean_lab_masked(
        wb_bgr,
        mask,
        l_star_trim_lo=l_star_trim_lo,
        l_star_trim_hi=l_star_trim_hi,
        a_star_trim_lo=a_star_trim_lo,
        a_star_trim_hi=a_star_trim_hi,
        b_star_trim_lo=b_star_trim_lo,
        b_star_trim_hi=b_star_trim_hi,
        min_chroma_ab=skin_min_chroma_ab,
        histogram_png=hist_png,
        histogram_title=f"{path.name}  (pid {pid})",
    )
    if chroma_relaxed:
        print(
            f"Warning pid={pid} {path.name}: dropped chroma_ab trim (min_keep); "
            f"skin_min_chroma_ab={skin_min_chroma_ab} was not applied to the mean.",
            file=sys.stderr,
        )
    if ltrim_relaxed:
        print(
            f"Warning pid={pid} {path.name}: dropped L*/a*/b* quantile trims (min_keep); "
            f"mean uses all {nraw} masked skin pixels.",
            file=sys.stderr,
        )

    return PhotoResult(
        participant_id=pid,
        image_path=str(path),
        L_mean=L,
        a_mean=a,
        b_mean=bb,
        n_pixels=npx,
        wb_ok=wb_ok,
        n_pixels_raw_mask=nraw,
        l_star_trim_lo_frac=l_star_trim_lo,
        l_star_trim_hi_frac=l_star_trim_hi,
        l_star_threshold_lo=lo_thr,
        l_star_threshold_hi=hi_thr,
        skin_min_chroma_ab=skin_min_chroma_ab,
        skin_l_trim_effective=l_trim_ok,
        skin_chroma_trim_effective=chroma_ok,
        L_mean_ltrim_only=L_lt,
        a_mean_ltrim_only=a_lt,
        b_mean_ltrim_only=b_lt,
        wb_linear_srgb=wb_linear_srgb and wb_ok,
    )


def find_calibration_photos(data_root: Path) -> List[Path]:
    """Photos under Data/<participant>/P1/Photos/*.{jpg,jpeg,png,JPG} (not .cr2 — export JPEG for this pipeline)."""
    out: List[Path] = []
    for photos in sorted(data_root.glob("*/P1/Photos")):
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG"):
            out.extend(sorted(photos.glob(ext)))
    return out


_PHOTO_TS = re.compile(r"(\d{8}T\d{6}Z)")


def _photo_timestamp_key(path: Path) -> str:
    m = _PHOTO_TS.search(path.stem)
    return m.group(1) if m else path.stem


def pick_one_photo_per_participant(
    paths: List[Path],
    strategy: str,
    *,
    debug: bool = False,
) -> List[Path]:
    """
    Reduce to one path per participant_id (parent of P1/Photos).

    * ``largest_chart`` — maximize ColorChecker area / frame (then newest filename, then name).
    * ``newest_jpg`` — maximize timestamp embedded in filename (``YYYYMMDDTHHMMSSZ``).
    * ``first_sorted`` — lexicographically first filename under that participant.
    """
    by_pid: Dict[str, List[Path]] = defaultdict(list)
    for p in paths:
        try:
            pid = p.parent.parent.parent.name
        except Exception:
            continue
        by_pid[pid].append(p)

    n_pid = len(by_pid)
    print(f"Picking one photo per participant ({strategy}) for {n_pid} participant(s)…", flush=True)
    out: List[Path] = []
    for pid in sorted(by_pid.keys(), key=lambda s: (0, int(s)) if str(s).isdigit() else (1, s)):
        lst = sorted(by_pid[pid], key=lambda x: x.name)
        if not lst:
            continue
        if strategy == "first_sorted":
            chosen = lst[0]
        elif strategy == "newest_jpg":
            chosen = max(lst, key=lambda p: _photo_timestamp_key(p))
        elif strategy == "largest_chart":
            ranked: List[Tuple[float, str, str, Path]] = []
            for p in lst:
                bgr = cv2.imread(str(p))
                if bgr is None:
                    if debug:
                        print(f"pick_photo: could not read {p}", file=sys.stderr)
                    continue
                score = chart_area_score_bgr(bgr)
                ranked.append((score, _photo_timestamp_key(p), p.name, p))
            if not ranked:
                chosen = lst[0]
            else:
                ranked.sort(key=lambda t: (t[0], t[1], t[2]))
                chosen = ranked[-1][3]
                if debug:
                    print(
                        f"pick_photo pid={pid} -> {chosen.name} (chart_area={ranked[-1][0]:.5f})",
                        file=sys.stderr,
                    )
        else:
            raise ValueError(f"unknown strategy {strategy!r}")
        out.append(chosen)
    return out


def apply_forced_photos(
    paths: List[Path],
    forced: Dict[str, str],
    data_root: Path,
    *,
    debug: bool = False,
) -> List[Path]:
    """
    For each participant id in ``forced``, replace any listed photo from that participant with
    ``data_root/<pid>/P1/Photos/<filename>`` (exact basename). Other paths for that pid are dropped.
    Pids not in ``forced`` are unchanged.
    """
    if not forced:
        return paths
    resolved: Dict[str, Path] = {}
    for pid, name in forced.items():
        cand = data_root / pid / "P1" / "Photos" / name
        if cand.is_file():
            resolved[pid] = cand
        else:
            print(f"Warning: --force-photo {pid}={name} not found ({cand})", file=sys.stderr)

    if not resolved:
        return paths

    out: List[Path] = []
    emitted_override: set[str] = set()
    for p in paths:
        try:
            pid = p.parent.parent.parent.name
        except Exception:
            out.append(p)
            continue
        if pid not in resolved:
            out.append(p)
            continue
        if pid in emitted_override:
            continue
        emitted_override.add(pid)
        fp = resolved[pid]
        if debug or p.resolve() != fp.resolve():
            print(f"Forced photo pid={pid}: {p.name} -> {fp.name}", flush=True)
        out.append(fp)
    return out


def _parse_pid_photo_override(spec: str) -> Tuple[str, str]:
    s = spec.strip()
    for sep in ("=", ":"):
        if sep in s:
            pid, name = s.split(sep, 1)
            pid, name = pid.strip(), name.strip()
            if pid and name:
                return pid, name
    raise argparse.ArgumentTypeError(
        f"--force-photo expects PID=FILENAME or PID:FILENAME (e.g. 14=20250421T152917Z.jpg), got {spec!r}"
    )


def load_monk_csv(path: Path) -> Dict[str, int]:
    """participant_id -> monk 1..10"""
    d: Dict[str, int] = {}
    with open(path, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            pid = str(row.get("participant_id") or row.get("pid") or row.get("id")).strip()
            mk = row.get("monk") or row.get("MST") or row.get("Monk")
            if not pid or mk is None:
                continue
            d[pid] = int(float(mk))
    return d


@contextlib.contextmanager
def _silence_stderr() -> Iterator[None]:
    """Hide native stderr (EGL/TensorFlow Lite/MediaPipe C++) during Face Mesh use."""
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


def main() -> None:
    ap = argparse.ArgumentParser(description="Skin L*a*b* from Physio calibration photos + optional Monk correlation.")
    ap.add_argument(
        "--data-root",
        type=Path,
        default=Path("/media/mabl-main/Data/Physio-code/Data"),
        help="Physio-code Data root containing <pid>/P1/Photos/",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=Path("skin_lab_output"),
        help="CSV + plots output directory (default: ./skin_lab_output). Must be writable — do not use placeholders like /path/to/output.",
    )
    ap.add_argument("--monk-csv", type=Path, default=None, help="Optional CSV: participant_id, monk (1-10)")
    ap.add_argument(
        "--mst-csv",
        type=Path,
        default=Path(__file__).resolve().parent / "mst_reference_cheng2024_table1.csv",
        help="Cheng Table I Lab (monk,L,a,b,…) for ΔE₀₀ vs each MST step on the per-photo CSV.",
    )
    ap.add_argument("--plot", action="store_true", help="Write PNG plots")
    ap.add_argument("--limit", type=int, default=0, help="Max images to process after other filters (0=all)")
    ap.add_argument(
        "--one-image-per-participant",
        action="store_true",
        help="Keep one calibration photo per participant (see --pick-photo). Recommended for n≈35.",
    )
    ap.add_argument(
        "--pick-photo",
        choices=("largest_chart", "newest_jpg", "first_sorted"),
        default="largest_chart",
        help="With --one-image-per-participant: how to choose the photo (default: largest ColorChecker in frame).",
    )
    ap.add_argument(
        "--force-photo",
        action="append",
        default=[],
        type=_parse_pid_photo_override,
        metavar="PID=FILENAME",
        help=(
            "Use Data/<PID>/P1/Photos/<FILENAME> for that participant instead of the picked path. "
            "Repeatable (e.g. 14=20250421T152917Z.jpg when largest_chart chose a bad pose)."
        ),
    )
    ap.add_argument("--skip-wb-if-no-chart", action="store_true", help="Use raw image if ColorChecker not detected")
    ap.add_argument(
        "--skin-exclusion-dilate-iod",
        type=float,
        default=0.12,
        metavar="FRAC",
        help="Dilate eye/lip/eyebrow/iris exclusion by FRAC × inter-ocular distance (px); larger = more margin, less sclera",
    )
    ap.add_argument(
        "--skin-triangulation",
        choices=("tessellation", "oval_delaunay"),
        default="tessellation",
        help=(
            "Skin mask geometry: tessellation = dense MediaPipe face-mesh triangles (~854, same "
            "topology as face-mesh RGB CSV tools); oval_delaunay = scipy Delaunay on face-oval only (coarse)."
        ),
    )
    ap.add_argument(
        "--skin-l-star-trim-lo",
        type=float,
        default=0.0,
        metavar="Q",
        help="Drop masked pixels with L* below the Q-quantile of the mask (0=off). Try 0.10–0.15 to remove shadow tail like face-mesh CSV luma filters.",
    )
    ap.add_argument(
        "--skin-l-star-trim-hi",
        type=float,
        default=0.0,
        metavar="Q",
        help="Drop masked pixels with L* above the (1−Q)-quantile (0=off). Small e.g. 0.02 trims specular highlights.",
    )
    ap.add_argument(
        "--skin-a-star-trim-lo",
        type=float,
        default=0.0,
        metavar="Q",
        help="Drop masked pixels with a* below the Q-quantile (0=off). Use with --skin-a-star-trim-hi (e.g. 0.05 each).",
    )
    ap.add_argument(
        "--skin-a-star-trim-hi",
        type=float,
        default=0.0,
        metavar="Q",
        help="Drop masked pixels with a* above the (1−Q)-quantile (0=off).",
    )
    ap.add_argument(
        "--skin-b-star-trim-lo",
        type=float,
        default=0.0,
        metavar="Q",
        help="Drop masked pixels with b* below the Q-quantile (0=off).",
    )
    ap.add_argument(
        "--skin-b-star-trim-hi",
        type=float,
        default=0.0,
        metavar="Q",
        help="Drop masked pixels with b* above the (1−Q)-quantile (0=off).",
    )
    ap.add_argument(
        "--skin-min-chroma-ab",
        type=float,
        default=0.0,
        metavar="C",
        help="Drop masked pixels with C*_ab=hypot(a*,b*) < C (0=off). Try 4–10 to suppress near-neutral black/gray/white pixels.",
    )
    ap.add_argument(
        "--write-skin-l-histograms",
        action="store_true",
        help="Write per-photo L*/a*/b*/C* histogram panels under OUT_DIR/skin_L_histograms/ (tan=kept for mean; L* trim + --skin-min-chroma-ab).",
    )
    ap.add_argument(
        "--write-skin-mask-overlays",
        action="store_true",
        help="Write per-photo PNG under OUT_DIR/skin_mask_overlays/ (mesh wireframe + kept triangles + mask tint).",
    )
    ap.add_argument(
        "--clean-skin-mask-overlays",
        action="store_true",
        help="Before writing overlays, delete existing *_skin_delaunay_overlay.png in OUT_DIR/skin_mask_overlays/ (removes stale frames from old picks).",
    )
    ap.add_argument(
        "--skin-overlay-max-width",
        type=int,
        default=1600,
        help="Max width in px for overlay PNGs (0 = full resolution).",
    )
    ap.add_argument(
        "--wb-linear-srgb",
        action="store_true",
        help=(
            "Apply chart WB gains in linear sRGB (then encode to 8-bit). Default applies gains in 8-bit "
            "display space, which is fast but not the same as skimage rgb2lab’s internal sRGB linearization; "
            "linear WB often shifts a* and b* more than L* when corrections are large."
        ),
    )
    ap.add_argument("--debug", action="store_true", help="Verbose logs (includes MediaPipe/EGL stderr)")
    ap.add_argument(
        "--progress-every",
        type=int,
        default=3,
        metavar="N",
        help="Print [i/total] path every N images (1=all). Default 3. Set 0 to disable.",
    )
    args = ap.parse_args()

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

    if not check_mcc():
        print("Warning: OpenCV mcc not available — white balance will fail unless --skip-wb-if-no-chart", file=sys.stderr)

    out_dir = args.out_dir.expanduser()
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        raise SystemExit(
            f"Cannot create --out-dir {args.out_dir}: {e}\n"
            "Use a real writable path (e.g. ./skin_lab_output or ~/results/skin_lab). "
            "The literal /path/to/output from generic tutorials is not a valid location."
        ) from e
    args.out_dir = out_dir

    hist_dir: Optional[Path] = None
    if args.write_skin_l_histograms:
        if plt is None:
            print("Warning: matplotlib not installed; skipping --write-skin-l-histograms", file=sys.stderr)
        else:
            hist_dir = args.out_dir / "skin_L_histograms"

    overlay_dir: Optional[Path] = None
    if args.write_skin_mask_overlays:
        overlay_dir = args.out_dir / "skin_mask_overlays"
        overlay_dir.mkdir(parents=True, exist_ok=True)
        if args.clean_skin_mask_overlays:
            n_gone = 0
            for stale in overlay_dir.glob("*_skin_delaunay_overlay.png"):
                try:
                    stale.unlink()
                    n_gone += 1
                except OSError:
                    pass
            if n_gone:
                print(f"Removed {n_gone} prior skin overlay PNG(s) (--clean-skin-mask-overlays)", flush=True)

    if args.skin_overlay_max_width < 0:
        raise SystemExit("--skin-overlay-max-width must be >= 0")

    photos = find_calibration_photos(args.data_root)
    if args.one_image_per_participant:
        photos = pick_one_photo_per_participant(photos, args.pick_photo, debug=args.debug)
        print(f"One image per participant ({args.pick_photo}): {len(photos)} photos", flush=True)
    if args.force_photo:
        photos = apply_forced_photos(photos, dict(args.force_photo), args.data_root, debug=args.debug)
    if args.limit > 0:
        photos = photos[: args.limit]

    mst_csv_path = args.mst_csv.expanduser().resolve()
    if not mst_csv_path.is_file():
        raise SystemExit(f"--mst-csv not found: {mst_csv_path}")
    mst_lab_10 = load_mst_lab_matrix_10x3(mst_csv_path)

    mp_fm = mp.solutions.face_mesh
    rows: List[PhotoResult] = []
    stderr_cm = _silence_stderr() if not args.debug else contextlib.nullcontext()
    with stderr_cm:
        # static_image_mode=True: tracking is off; min_tracking_confidence is ignored by MediaPipe.
        with mp_fm.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        ) as face_mesh:
            for i, p in enumerate(photos):
                if args.debug:
                    print(f"[{i+1}/{len(photos)}] {p}", flush=True)
                elif args.progress_every > 0:
                    pe = args.progress_every
                    if i == 0 or (i + 1) % pe == 0 or (i + 1) == len(photos):
                        print(f"[{i+1}/{len(photos)}] {p}", flush=True)
                r = process_one_image(
                    p,
                    face_mesh,
                    skip_wb_if_no_chart=args.skip_wb_if_no_chart,
                    debug=args.debug,
                    skin_exclusion_dilate_iod_fraction=args.skin_exclusion_dilate_iod,
                    skin_triangulation=args.skin_triangulation,
                    l_star_trim_lo=args.skin_l_star_trim_lo,
                    l_star_trim_hi=args.skin_l_star_trim_hi,
                    a_star_trim_lo=args.skin_a_star_trim_lo,
                    a_star_trim_hi=args.skin_a_star_trim_hi,
                    b_star_trim_lo=args.skin_b_star_trim_lo,
                    b_star_trim_hi=args.skin_b_star_trim_hi,
                    skin_min_chroma_ab=args.skin_min_chroma_ab,
                    skin_l_histogram_dir=hist_dir,
                    skin_overlay_dir=overlay_dir,
                    skin_overlay_max_width=args.skin_overlay_max_width,
                    wb_linear_srgb=args.wb_linear_srgb,
                )
                if r:
                    rows.append(r)

    csv_path = args.out_dir / "skin_lab_per_photo.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "participant_id",
                "image_path",
                "L_mean",
                "a_mean",
                "b_mean",
                "n_skin_pixels_used",
                "n_skin_pixels_raw_mask",
                "l_star_trim_lo_frac",
                "l_star_trim_hi_frac",
                "l_star_threshold_lo",
                "l_star_threshold_hi",
                "skin_min_chroma_ab",
                "skin_l_trim_effective",
                "skin_chroma_trim_effective",
                "L_mean_ltrim_only",
                "a_mean_ltrim_only",
                "b_mean_ltrim_only",
                "wb_ok",
                "wb_linear_srgb",
            ]
            + de2000_csv_header()
        )
        for r in rows:
            if all(np.isfinite(float(x)) for x in (r.L_mean, r.a_mean, r.b_mean)):
                de10, near_m, dmin = mst_de2000_row(np.array([r.L_mean, r.a_mean, r.b_mean], dtype=np.float64), mst_lab_10)
                de_cols = de2000_csv_values(de10, near_m, dmin)
            else:
                de_cols = [float("nan")] * 10 + [float("nan"), float("nan")]
            w.writerow(
                [
                    r.participant_id,
                    r.image_path,
                    r.L_mean,
                    r.a_mean,
                    r.b_mean,
                    r.n_pixels,
                    r.n_pixels_raw_mask,
                    r.l_star_trim_lo_frac,
                    r.l_star_trim_hi_frac,
                    "" if not np.isfinite(r.l_star_threshold_lo) else r.l_star_threshold_lo,
                    "" if not np.isfinite(r.l_star_threshold_hi) else r.l_star_threshold_hi,
                    r.skin_min_chroma_ab,
                    int(r.skin_l_trim_effective),
                    int(r.skin_chroma_trim_effective),
                    "" if not np.isfinite(r.L_mean_ltrim_only) else r.L_mean_ltrim_only,
                    "" if not np.isfinite(r.a_mean_ltrim_only) else r.a_mean_ltrim_only,
                    "" if not np.isfinite(r.b_mean_ltrim_only) else r.b_mean_ltrim_only,
                    int(r.wb_ok),
                    int(r.wb_linear_srgb),
                ]
                + de_cols
            )

    # Aggregate per participant (mean across photos)
    by_pid: Dict[str, List[PhotoResult]] = {}
    for r in rows:
        by_pid.setdefault(r.participant_id, []).append(r)

    agg_path = args.out_dir / "skin_lab_per_participant.json"
    doc: Dict[str, Any] = {
        "_pipeline": {
            "skin_triangulation": args.skin_triangulation,
            "mst_reference_csv": str(mst_csv_path),
            "delta_e_2000": "Per-photo and per-participant (from mean Lab) vs Table I Lab rows in --mst-csv (CIEDE2000).",
            "wb_in_linear_srgb": bool(args.wb_linear_srgb),
            "skin_l_star_trim_lo": float(args.skin_l_star_trim_lo),
            "skin_l_star_trim_hi": float(args.skin_l_star_trim_hi),
            "skin_min_chroma_ab": float(args.skin_min_chroma_ab),
            "mean_lab_columns": (
                "L_mean,a_mean,b_mean = mean after optional L* quantiles and optional min chroma_ab on masked pixels. "
                "CSV also has L_mean_ltrim_only,a_mean_ltrim_only,b_mean_ltrim_only (L* trim only, before chroma gate) "
                "to diagnose upward a*b* bias from --skin-min-chroma-ab."
            ),
            "lab_illuminant": "D65 (explicit skimage.color.rgb2lab illuminant='D65', observer='2'); same convention as typical sRGB→Lab for MST Table I comparisons.",
            "skin_region": (
                "Mean L*a*b* (D65-referenced skimage Lab) over white-balanced sRGB pixels in the skin mask. "
                "Default: MediaPipe face-mesh tessellation triangles (centroid inside face-oval convex hull, "
                "outside dilated lip/eye/eyebrow/iris). "
                "Alternative --skin-triangulation oval_delaunay: scipy Delaunay on face-oval landmarks only. "
                "Per-photo optional L* quantile trim and min chroma_ab in CSV; "
                "skin_l_trim_effective / skin_chroma_trim_effective record whether trims were applied "
                "(min_keep can relax chroma then L* tails)."
            ),
        }
    }
    for pid, lst in by_pid.items():
        labs = [
            (x.L_mean, x.a_mean, x.b_mean)
            for x in lst
            if np.isfinite(x.L_mean) and np.isfinite(x.a_mean) and np.isfinite(x.b_mean)
        ]
        entry: Dict[str, Any] = {
            "n_photos": len(lst),
            "L_mean": float(np.mean([t[0] for t in labs])) if labs else float("nan"),
            "a_mean": float(np.mean([t[1] for t in labs])) if labs else float("nan"),
            "b_mean": float(np.mean([t[2] for t in labs])) if labs else float("nan"),
        }
        if labs:
            arr = np.array(labs, dtype=np.float64)
            Lm, am, bm = float(arr[:, 0].mean()), float(arr[:, 1].mean()), float(arr[:, 2].mean())
            de10, near_m, dmin = mst_de2000_row(np.array([Lm, am, bm], dtype=np.float64), mst_lab_10)
            for k in range(10):
                entry[f"de2000_mst{k + 1:02d}"] = float(de10[k])
            entry["de2000_nearest_mst"] = int(near_m)
            entry["de2000_min"] = float(dmin)
        doc[pid] = entry
    agg_path.write_text(json.dumps(doc, indent=2), encoding="utf-8")

    print(f"Wrote {csv_path} ({len(rows)} photos)")
    print(f"Wrote {agg_path} ({len(by_pid)} participants)")
    if hist_dir is not None:
        print(f"L* histograms: {hist_dir} ({len(rows)} PNGs)")
    if overlay_dir is not None:
        print(f"Skin mask / mesh overlays: {overlay_dir} ({len(rows)} PNGs)")

    monk_map = load_monk_csv(args.monk_csv) if args.monk_csv and args.monk_csv.is_file() else {}

    if args.plot and plt is not None:
        # L* distribution by participant
        pids = sorted(
            (k for k in doc if not str(k).startswith("_")),
            key=lambda p: doc[p]["L_mean"],
        )
        Ls = [doc[p]["L_mean"] for p in pids]
        plt.figure(figsize=(12, 4))
        plt.bar(range(len(pids)), Ls, color="tan", edgecolor="k", alpha=0.85)
        plt.xticks(range(len(pids)), pids, rotation=45, ha="right")
        plt.ylabel("Mean L* (skin)")
        plt.title("Mean skin L* by participant (WB + skin mask; optional L* quantile trim)")
        plt.tight_layout()
        plt.savefig(args.out_dir / "skin_L_by_participant.png", dpi=160)
        plt.close()

        if monk_map:
            xs: List[int] = []
            ys: List[float] = []
            for pid in doc:
                if str(pid).startswith("_"):
                    continue
                if pid in monk_map:
                    xs.append(monk_map[pid])
                    ys.append(doc[pid]["L_mean"])
            if len(xs) >= 3:
                rho, pval = scipy_stats.spearmanr(xs, ys)
                plt.figure(figsize=(6, 5))
                plt.scatter(xs, ys, c="brown", edgecolors="k", alpha=0.8)
                plt.xlabel("Monk Skin Tone (1–10)")
                plt.ylabel("Mean facial skin L*")
                plt.title(f"Spearman ρ = {rho:.3f}, p = {pval:.4g}")
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(args.out_dir / "skin_L_vs_monk.png", dpi=160)
                plt.close()
                print(f"Spearman(L*, Monk) = {rho:.4f}, p = {pval:.4g}")

    print("Done.")


if __name__ == "__main__":
    main()
