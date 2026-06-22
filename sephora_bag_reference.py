"""
Sephora bag stripe reference — uses the same MediaPipe + flash/no-flash linear stack
as ``flash_no_flash_skin_lab.py``.

Primary bag ROI: **MediaPipe Hands** — full bag body between both hands, then white/black stripe split.
Optional: **SAM 2** full-bag box prompt with hand negative points (``--sam2``).
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import cv2
import numpy as np

_CHIN = 152
_LEFT_EYE_OUTER = 33
_RIGHT_EYE_OUTER = 263


@dataclass
class NixBagReference:
    white_lab: np.ndarray
    black_lab: np.ndarray
    white_xyz: np.ndarray
    black_xyz: np.ndarray
    white_y: float
    black_y: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "white_lab": self.white_lab.tolist(),
            "black_lab": self.black_lab.tolist(),
            "white_xyz": self.white_xyz.tolist(),
            "black_xyz": self.black_xyz.tolist(),
            "white_y": self.white_y,
            "black_y": self.black_y,
            "illuminant": "D65",
            "instrument": "Nix Spectro 2 M2",
        }


@dataclass
class BagSegmentation:
    bag_bbox: Tuple[int, int, int, int]
    white_mask: np.ndarray
    black_mask: np.ndarray
    bag_mask: np.ndarray
    white_rgb_mean: np.ndarray       # three-zone median RGB
    black_rgb_mean: np.ndarray
    white_y: float
    black_y: float
    n_white: int
    n_black: int
    detection_mode: str = "face_mesh"
    white_zone_rgb: Optional[np.ndarray] = None   # (3, 3) — L/C/R zone medians


def load_nix_bag_reference(csv_path: Path) -> NixBagReference:
    rows: list[dict[str, str]] = []
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header_idx = None
        for i, row in enumerate(reader):
            if row and row[0] == "Index":
                header_idx = i
                break
        if header_idx is None:
            raise ValueError(f"No header row in {csv_path}")
        f.seek(0)
        all_rows = list(csv.reader(f))
        col = {h: j for j, h in enumerate(all_rows[header_idx])}
        for row in all_rows[header_idx + 1 :]:
            if len(row) <= max(col.values()) or not row[col["User Color Name"]].strip():
                continue
            rows.append({h: row[col[h]] for h in col})

    def _pick(prefix: str) -> list[dict[str, str]]:
        return [r for r in rows if r["User Color Name"].lower().startswith(prefix.lower())]

    whites, blacks = _pick("sephorawhite"), _pick("sephorablack")
    if not whites or not blacks:
        raise ValueError(f"Expected Sephorawhite* / Sephorablack* in {csv_path}")

    def _mean(subset: list[dict[str, str]]) -> Tuple[np.ndarray, np.ndarray]:
        labs = [[float(r["L"]), float(r["a"]), float(r["b"])] for r in subset]
        xyzs = [[float(r["X"]), float(r["Y"]), float(r["Z"])] for r in subset]
        return np.mean(labs, axis=0), np.mean(xyzs, axis=0)

    w_lab, w_xyz = _mean(whites)
    b_lab, b_xyz = _mean(blacks)
    return NixBagReference(
        white_lab=w_lab,
        black_lab=b_lab,
        white_xyz=w_xyz,
        black_xyz=b_xyz,
        white_y=float(w_xyz[1]),
        black_y=float(b_xyz[1]),
    )


def save_nix_reference_json(ref: NixBagReference, path: Path) -> None:
    path.write_text(json.dumps(ref.to_dict(), indent=2) + "\n")


def _luma_y(rgb: np.ndarray) -> np.ndarray:
    return 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]


def _clamp_bbox(
    x0: int, y0: int, x1: int, y1: int, h: int, w: int, *, pad: int = 4, min_size: int = 20
) -> Tuple[int, int, int, int]:
    x0 = max(pad, min(x0, w - pad - 1))
    x1 = max(x0 + min_size, min(x1, w - pad))
    y0 = max(pad, min(y0, h - pad - 1))
    y1 = max(y0 + min_size, min(y1, h - pad))
    return x0, y0, x1, y1


def _face_metrics(landmarks: Sequence[Any], h: int, w: int) -> Tuple[float, float, float, float]:
    def pt(idx: int) -> np.ndarray:
        lm = landmarks[idx]
        return np.array([lm.x * w, lm.y * h], dtype=np.float64)

    chin = pt(_CHIN)
    le = pt(_LEFT_EYE_OUTER)
    re = pt(_RIGHT_EYE_OUTER)
    iod = float(np.linalg.norm(le - re))
    if iod < 8.0:
        iod = 0.22 * float(min(h, w))
    cx = float((le[0] + re[0]) * 0.5)
    cy = float((le[1] + re[1]) * 0.5)
    return float(chin[1]), iod, cx, cy


def _hand_union_bbox(hand_landmarks: Sequence[Any], h: int, w: int) -> Tuple[int, int, int, int]:
    xs: list[float] = []
    ys: list[float] = []
    for hl in hand_landmarks:
        pts = hl.landmark if hasattr(hl, "landmark") else hl
        for p in pts:
            xs.append(p.x * w)
            ys.append(p.y * h)
    return int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))


def _horizontal_stripe_peak_row(
    rgb_u01: np.ndarray,
    x0: int,
    x1: int,
    y_lo: int,
    y_hi: int,
) -> Optional[int]:
    y_lo = max(0, y_lo)
    y_hi = min(rgb_u01.shape[0], y_hi)
    if y_hi - y_lo < 8:
        return None

    band = rgb_u01[y_lo:y_hi, x0:x1]
    lum = _luma_y(band)
    chroma = np.ptp(band, axis=2)
    best_score = -1.0
    best_row: Optional[int] = None
    for ri in range(lum.shape[0]):
        row_l = lum[ri]
        row_c = chroma[ri]
        valid = row_c < 0.25
        if valid.sum() < max(12, int(0.35 * row_l.size)):
            continue
        score = float(np.std(np.diff(row_l[valid])))
        if score > best_score:
            best_score = score
            best_row = y_lo + ri
    return best_row


def _bag_stripe_vertical_extent(
    rgb_u01: np.ndarray,
    x0: int,
    x1: int,
    y_lo: int,
    y_hi: int,
    iod: float,
) -> Optional[Tuple[int, int]]:
    """Top/bottom of visible horizontal stripes (handles → bottom edge)."""
    y_lo = max(0, y_lo)
    y_hi = min(rgb_u01.shape[0], y_hi)
    if y_hi - y_lo < 12:
        return None

    band = rgb_u01[y_lo:y_hi, x0:x1]
    lum = _luma_y(band)
    chroma = np.ptp(band, axis=2)
    stripe_rows: list[int] = []
    for ri in range(lum.shape[0]):
        valid = chroma[ri] < 0.28
        if valid.sum() < max(10, int(0.35 * lum.shape[1])):
            continue
        row_l = lum[ri][valid]
        if row_l.size < 8:
            continue
        if float(np.std(np.diff(row_l))) >= 0.035:
            stripe_rows.append(y_lo + ri)

    if len(stripe_rows) < 4:
        return None
    pad = int(0.12 * iod)
    return int(min(stripe_rows) - pad), int(max(stripe_rows) + pad)


def _row_alternation_scores(
    rgb_u01: np.ndarray,
    x0: int,
    x1: int,
    y0: int,
    y1: int,
    *,
    chroma_thresh: float = 0.32,
    min_valid_frac: float = 0.12,
) -> np.ndarray:
    """
    Per-row median luma inside the search band, NaN where row is too chromatic.

    The Sephora bag is the only object in the scene with strong row-to-row
    luma alternation (white→black→white...).  Computing std(row_medians) in
    a sliding window picks that pattern even in dim images.
    """
    h = rgb_u01.shape[0]
    x0, x1 = max(0, x0), min(rgb_u01.shape[1], x1)
    y0, y1 = max(0, y0), min(h, y1)
    band = rgb_u01[y0:y1, x0:x1]
    lum = _luma_y(band)
    chroma = np.ptp(band, axis=2)
    min_px = max(6, int(min_valid_frac * band.shape[1]))
    medians = np.full(y1 - y0, np.nan, dtype=np.float64)
    for ri in range(band.shape[0]):
        valid = chroma[ri] < chroma_thresh
        if valid.sum() >= min_px:
            medians[ri] = float(np.median(lum[ri][valid]))
    return medians


def _trim_bbox_by_edges(
    rgb_u01: np.ndarray,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    iod: float,
) -> Tuple[int, int, int, int]:
    """
    Tighten a candidate bag bbox using Sobel edge detection.

    Sobel-x finds the strong vertical edges at the left/right paper boundary.
    Sobel-y finds horizontal edges which confirm the top/bottom stripe extent.
    A column (row) energy profile is smoothed; the outermost columns (rows)
    that exceed a fraction of the peak define the bag boundary.
    """
    patch = rgb_u01[y0:y1, x0:x1]
    if patch.shape[0] < 16 or patch.shape[1] < 16:
        return x0, y0, x1, y1

    lum = _luma_y(patch)

    # Normalise luma to [0,255] locally (handles dark images cleanly)
    lmin, lmax = float(lum.min()), float(lum.max())
    if lmax - lmin < 0.01:
        return x0, y0, x1, y1
    lum_u8 = np.clip((lum - lmin) / (lmax - lmin) * 255, 0, 255).astype(np.uint8)

    # --- Vertical edges → horizontal (left/right) clamping ---
    sobel_x = cv2.Sobel(lum_u8, cv2.CV_64F, 1, 0, ksize=5)
    col_energy = np.abs(sobel_x).sum(axis=0)
    # Smooth with a 3%-width kernel
    k = max(3, int(0.03 * col_energy.size))
    col_sm = np.convolve(col_energy, np.ones(k) / k, mode="same")
    thresh_x = float(np.percentile(col_sm, 70))
    active_x = np.where(col_sm >= thresh_x)[0]
    if active_x.size >= 2:
        pad_x = max(4, int(0.04 * iod))
        new_x0 = x0 + max(0, int(active_x[0]) - pad_x)
        new_x1 = x0 + min(patch.shape[1], int(active_x[-1]) + pad_x)
    else:
        new_x0, new_x1 = x0, x1

    # --- Horizontal edges → vertical (top/bottom) clamping ---
    sobel_y = cv2.Sobel(lum_u8, cv2.CV_64F, 0, 1, ksize=5)
    row_energy = np.abs(sobel_y).sum(axis=1)
    k2 = max(3, int(0.03 * row_energy.size))
    row_sm = np.convolve(row_energy, np.ones(k2) / k2, mode="same")
    thresh_y = float(np.percentile(row_sm, 65))
    active_y = np.where(row_sm >= thresh_y)[0]
    if active_y.size >= 2:
        pad_y = max(4, int(0.04 * iod))
        new_y0 = y0 + max(0, int(active_y[0]) - pad_y)
        new_y1 = y0 + min(patch.shape[0], int(active_y[-1]) + pad_y)
    else:
        new_y0, new_y1 = y0, y1

    # Sanity: never shrink below 0.6 IOD in either dimension
    min_side = int(0.6 * iod)
    if new_x1 - new_x0 < min_side:
        new_x0, new_x1 = x0, x1
    if new_y1 - new_y0 < min_side:
        new_y0, new_y1 = y0, y1

    h, w = rgb_u01.shape[:2]
    return _clamp_bbox(new_x0, new_y0, new_x1, new_y1, h, w)


def bag_bbox_by_stripe_scan(
    rgb_u01: np.ndarray,
    face_landmarks: Sequence[Any],
    h: int,
    w: int,
) -> Optional[Tuple[int, int, int, int]]:
    """
    Locate the Sephora bag purely from stripe energy — no hand landmarks required.

    Strategy:
      1. Below the chin, scan row-median lumas in a central search band.
      2. A sliding window std of those medians peaks over the alternating
         black/white stripes.  The peak window defines the vertical extent.
      3. ``_bag_stripe_horizontal_extent`` trims to the actual paper edges.
    """
    chin_y, iod, face_cx, _ = _face_metrics(face_landmarks, h, w)

    # Search band: below chin, horizontally centred on face
    search_x0 = int(max(0, face_cx - 2.8 * iod))
    search_x1 = int(min(w, face_cx + 2.8 * iod))
    search_y0 = int(chin_y + 0.6 * iod)
    search_y1 = h

    medians = _row_alternation_scores(rgb_u01, search_x0, search_x1, search_y0, search_y1)
    if np.isnan(medians).all():
        return None

    # Sliding window std — find the window with the highest alternation
    win = max(12, int(0.30 * iod))   # ~30% of IOD in rows
    n = len(medians)
    if n < win + 2:
        return None

    best_std, best_center = 0.0, n // 2
    for i in range(n - win):
        chunk = medians[i : i + win]
        valid = chunk[~np.isnan(chunk)]
        if len(valid) < win * 0.55:
            continue
        s = float(np.std(valid))
        if s > best_std:
            best_std, best_center = s, i + win // 2

    # Require meaningful alternation (white/black differ by ≥ 0.06 in luma)
    if best_std < 0.03:
        return None

    # Expand around best_center to include all rows with std > half peak
    half_thresh = best_std * 0.50
    stripe_rows = []
    for i in range(n - win):
        chunk = medians[i : i + win]
        valid = chunk[~np.isnan(chunk)]
        if len(valid) < win * 0.40 or np.std(valid) < half_thresh:
            continue
        stripe_rows.extend(range(i, i + win))

    if len(stripe_rows) < 4:
        return None

    stripe_rows_abs = np.array(sorted(set(stripe_rows))) + search_y0
    pad = int(0.10 * iod)
    y_top = int(stripe_rows_abs.min()) - pad
    y_bot = int(stripe_rows_abs.max()) + pad

    # Horizontal extent from stripe contrast
    x_extent = _bag_stripe_horizontal_extent(
        rgb_u01, search_x0, search_x1, y_top, y_bot,
        center_x=face_cx, iod=iod,
    )
    if x_extent is not None:
        x0_f, x1_f = x_extent
    else:
        # Fallback: use a face-width region centred on face_cx
        hw = int(0.72 * iod)
        x0_f = int(face_cx - hw)
        x1_f = int(face_cx + hw)

    rough = _clamp_bbox(x0_f, y_top, x1_f, y_bot, h, w)

    # Refine with Sobel edge detection — snaps to actual paper boundaries
    rx0, ry0, rx1, ry1 = _trim_bbox_by_edges(rgb_u01, *rough, iod)
    return _clamp_bbox(rx0, ry0, rx1, ry1, h, w)


def _contiguous_segments(indices: np.ndarray) -> list[Tuple[int, int]]:
    if indices.size == 0:
        return []
    splits = np.where(np.diff(indices) > 1)[0] + 1
    groups = np.split(indices, splits)
    return [(int(g[0]), int(g[-1]) + 1) for g in groups if g.size]


def _bag_stripe_horizontal_extent(
    rgb_u01: np.ndarray,
    x0: int,
    x1: int,
    y0: int,
    y1: int,
    *,
    center_x: float,
    iod: float,
) -> Optional[Tuple[int, int]]:
    """
    Left/right paper edges from vertical stripe contrast.

    The Sephora bag paper has strong vertical luma alternation because the stripes
    run horizontally. Hands and shirt do not have the same repeated row pattern,
    so this snaps the hand-guided ROI to the striped rectangle.
    """
    h, w = rgb_u01.shape[:2]
    x0 = max(0, min(x0, w - 2))
    x1 = max(x0 + 2, min(x1, w))
    y0 = max(0, min(y0, h - 2))
    y1 = max(y0 + 2, min(y1, h))

    patch = rgb_u01[y0:y1, x0:x1]
    if patch.shape[0] < 16 or patch.shape[1] < 16:
        return None

    lum = _luma_y(patch)
    chroma = np.ptp(patch, axis=2)
    scores = np.zeros(patch.shape[1], dtype=np.float64)
    for ci in range(patch.shape[1]):
        valid = chroma[:, ci] < 0.26
        if valid.sum() < max(8, int(0.25 * patch.shape[0])):
            continue
        col = lum[:, ci][valid]
        scores[ci] = float(np.std(col))

    if not np.any(scores > 0):
        return None

    k = max(5, int(0.025 * (x1 - x0)))
    kernel = np.ones(k, dtype=np.float64) / float(k)
    smooth = np.convolve(scores, kernel, mode="same")
    threshold = max(0.015, float(np.percentile(smooth[smooth > 0], 35)))
    active = np.where(smooth >= threshold)[0]
    segments = _contiguous_segments(active)
    min_width = max(18, int(0.20 * iod))
    segments = [(a, b) for a, b in segments if (b - a) >= min_width]
    if not segments:
        return None

    local_cx = center_x - x0
    keep: list[Tuple[int, int]] = []
    max_dist = 1.65 * iod
    for a, b in segments:
        seg_center = 0.5 * (a + b)
        if abs(seg_center - local_cx) <= max_dist or a <= local_cx <= b:
            keep.append((a, b))

    if keep:
        a = min(s[0] for s in keep)
        b = max(s[1] for s in keep)
    else:
        # Prefer a segment containing face/body center; otherwise nearest segment.
        def segment_cost(seg: Tuple[int, int]) -> float:
            a0, b0 = seg
            if a0 <= local_cx <= b0:
                center_penalty = 0.0
            else:
                center_penalty = min(abs(local_cx - a0), abs(local_cx - b0))
            width_bonus = 0.05 * (b0 - a0)
            return center_penalty - width_bonus

        a, b = min(segments, key=segment_cost)

    # Constrain implausibly wide unions, but keep enough width for the bag body.
    max_width = int(3.0 * iod)
    if (b - a) > max_width:
        mid = int(local_cx)
        half = max_width // 2
        a = max(a, mid - half)
        b = min(b, mid + half)

    if (b - a) < int(1.15 * iod):
        return None
    pad = int(0.03 * iod)
    return x0 + a - pad, x0 + b + pad


def bag_bbox_from_hands(
    face_landmarks: Sequence[Any],
    hand_landmarks: Sequence[Any],
    rgb_u01: np.ndarray,
    h: int,
    w: int,
) -> Optional[Tuple[int, int, int, int]]:
    """Full bag body between both hands (all visible stripes)."""
    if len(hand_landmarks) < 2:
        return None

    chin_y, iod, face_cx, _ = _face_metrics(face_landmarks, h, w)
    hx0, hy0, hx1, hy1 = _hand_union_bbox(hand_landmarks, h, w)

    x_pad = 0.04 * iod
    x0 = int(hx0 - x_pad)
    x1 = int(hx1 + x_pad)

    y_search_lo = int(chin_y + 1.15 * iod)
    y_search_hi = int(hy1 + 0.14 * iod)
    y0_hand = int(hy0 - 1.08 * iod)
    y1_hand = int(hy1 + 0.12 * iod)

    extent = _bag_stripe_vertical_extent(rgb_u01, x0, x1, y_search_lo, y_search_hi, iod)
    if extent is not None:
        y0 = min(y0_hand, extent[0])
        y1 = max(y1_hand, extent[1])
    else:
        y0, y1 = y0_hand, y1_hand

    y0 = max(y0, int(chin_y + 1.22 * iod))
    x_extent = _bag_stripe_horizontal_extent(
        rgb_u01,
        x0,
        x1,
        y0,
        y1,
        center_x=face_cx,
        iod=iod,
    )
    if x_extent is not None:
        x0, x1 = x_extent
    return _clamp_bbox(x0, y0, x1, y1, h, w)


def bag_bbox_from_landmarks(
    landmarks: Sequence[Any],
    h: int,
    w: int,
    *,
    below_chin_iod: float = 2.45,
    bag_height_iod: float = 0.55,
    bag_half_width_iod: float = 0.55,
) -> Tuple[int, int, int, int]:
    chin_y, iod, cx, _ = _face_metrics(landmarks, h, w)
    y0 = int(chin_y + below_chin_iod * iod)
    y1 = int(chin_y + (below_chin_iod + bag_height_iod) * iod)
    x0 = int(cx - bag_half_width_iod * iod)
    x1 = int(cx + bag_half_width_iod * iod)
    return _clamp_bbox(x0, y0, x1, y1, h, w)


def _hand_exclusion_mask(
    hand_landmarks: Sequence[Any],
    h: int,
    w: int,
    *,
    dilate_px: int = 10,
) -> np.ndarray:
    mask = np.zeros((h, w), dtype=np.uint8)
    for hl in hand_landmarks:
        pts = hl.landmark if hasattr(hl, "landmark") else hl
        poly = np.array([[int(p.x * w), int(p.y * h)] for p in pts], dtype=np.int32)
        cv2.fillConvexPoly(mask, poly, 1)
    if dilate_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_px, dilate_px))
        mask = cv2.dilate(mask, k)
    return mask.astype(bool)


def _three_zone_median_rgb(
    patch: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    """
    Robust RGB estimate from a stripe mask using three horizontal zones.

    The white stripe is split into left / centre / right thirds.  Each zone
    contributes a per-channel median; the final value is the channel-wise
    median of those three zone medians.  This makes the estimate insensitive
    to hand occlusion, specular highlights, or shadow falling on any single
    side of the bag.

    Falls back to a global median when a zone has fewer than 20 valid pixels.
    """
    w = patch.shape[1]
    thirds = [
        (0,          w // 3),
        (w // 3,     2 * w // 3),
        (2 * w // 3, w),
    ]
    zone_medians: list[np.ndarray] = []
    for xa, xb in thirds:
        zone_mask = mask.copy()
        zone_mask[:, :xa] = False
        zone_mask[:, xb:] = False
        n = int(zone_mask.sum())
        if n < 20:
            continue
        zone_medians.append(np.median(patch[zone_mask], axis=0))

    if not zone_medians:
        # All zones empty — fall back to global median
        n_all = int(mask.sum())
        if n_all < 1:
            return np.zeros(3, dtype=np.float64)
        return np.median(patch[mask], axis=0).astype(np.float64)

    # Channel-wise median of the zone medians (robust to one bad zone)
    return np.median(np.stack(zone_medians, axis=0), axis=0).astype(np.float64)


def segment_bag_stripes_in_bbox(
    rgb_u01: np.ndarray,
    bag_bbox: Tuple[int, int, int, int],
    *,
    hand_landmarks: Optional[Sequence[Any]] = None,
    paper_mask: Optional[np.ndarray] = None,
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, int, float, float]]:
    h, w = rgb_u01.shape[:2]
    x0, y0, x1, y1 = bag_bbox
    patch = rgb_u01[y0:y1, x0:x1]
    if patch.size == 0:
        return None

    y = _luma_y(patch)
    chroma = np.ptp(patch, axis=2)
    valid = chroma < 0.15
    if paper_mask is not None:
        valid &= paper_mask[y0:y1, x0:x1]
    if hand_landmarks:
        hand_local = _hand_exclusion_mask(hand_landmarks, h, w, dilate_px=18)[y0:y1, x0:x1]
        valid &= ~hand_local
        side = max(6, int(0.14 * (x1 - x0)))
        valid[:, :side] = False
        valid[:, -side:] = False
    if valid.sum() < 40:
        valid = chroma < 0.22
        if paper_mask is not None:
            valid &= paper_mask[y0:y1, x0:x1]
        if hand_landmarks:
            hand_local = _hand_exclusion_mask(hand_landmarks, h, w, dilate_px=18)[y0:y1, x0:x1]
            valid &= ~hand_local
            side = max(6, int(0.14 * (x1 - x0)))
            valid[:, :side] = False
            valid[:, -side:] = False
    if valid.sum() < 40:
        valid = np.ones(patch.shape[:2], dtype=bool)

    y_valid = y[valid]
    if y_valid.size < 40:
        return None

    # Row-wise stripe labels fill each horizontal band across the bag width.
    white_local = np.zeros(patch.shape[:2], dtype=bool)
    black_local = np.zeros(patch.shape[:2], dtype=bool)
    row_medians: list[Tuple[int, float]] = []
    for ri in range(patch.shape[0]):
        row_valid = valid[ri]
        if row_valid.sum() < max(10, int(0.22 * patch.shape[1])):
            continue
        row_medians.append((ri, float(np.median(y[ri][row_valid]))))

    if len(row_medians) >= 4:
        thr = float(np.median([rv for _, rv in row_medians]))
        for ri, rv in row_medians:
            if rv >= thr:
                white_local[ri, valid[ri]] = True
            else:
                black_local[ri, valid[ri]] = True
    else:
        q_lo, q_hi = np.quantile(y_valid, [0.38, 0.62])
        white_local = valid & (y >= q_hi)
        black_local = valid & (y <= q_lo)

    rgb_max = np.max(patch, axis=2)
    if white_local.sum() > 30:
        spec = np.quantile(y[white_local], 0.94)
        white_local &= ~((rgb_max > 0.96) & (y >= spec))

    bw = max(9, (x1 - x0) // 10)
    k_h = cv2.getStructuringElement(cv2.MORPH_RECT, (bw, 3))
    k_s = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    white_local = cv2.morphologyEx(white_local.astype(np.uint8), cv2.MORPH_CLOSE, k_h)
    black_local = cv2.morphologyEx(black_local.astype(np.uint8), cv2.MORPH_CLOSE, k_h)
    white_local = cv2.morphologyEx(white_local, cv2.MORPH_OPEN, k_s).astype(bool)
    black_local = cv2.morphologyEx(black_local, cv2.MORPH_OPEN, k_s).astype(bool)

    if white_local.sum() < 15 or black_local.sum() < 15:
        return None

    white_full = np.zeros((h, w), dtype=bool)
    black_full = np.zeros((h, w), dtype=bool)
    white_full[y0:y1, x0:x1] = white_local
    black_full[y0:y1, x0:x1] = black_local
    bag_full = white_full | black_full

    # Use clean stripe cores for the NIX comparison while keeping full-band masks
    # for visual QC. This avoids shaded edges/finger-adjacent pixels dominating
    # the reported white and black references.
    white_measure = white_local.copy()
    black_measure = black_local.copy()
    if white_measure.sum() > 50:
        white_floor = np.quantile(y[white_measure], 0.55)
        white_measure &= y >= white_floor
    if black_measure.sum() > 50:
        black_ceiling = np.quantile(y[black_measure], 0.45)
        black_measure &= y <= black_ceiling
    if white_measure.sum() < 15:
        white_measure = white_local
    if black_measure.sum() < 15:
        black_measure = black_local

    rw = _three_zone_median_rgb(patch, white_measure)
    rb = _three_zone_median_rgb(patch, black_measure)
    yw = float(_luma_y(rw))
    yb = float(_luma_y(rb))

    # Collect per-zone white medians for diagnostics / overlay
    w_patch = patch.shape[1]
    thirds = [(0, w_patch // 3), (w_patch // 3, 2 * w_patch // 3), (2 * w_patch // 3, w_patch)]
    zone_rgb_list = []
    for xa, xb in thirds:
        zm = white_measure.copy()
        zm[:, :xa] = False
        zm[:, xb:] = False
        if zm.sum() >= 20:
            zone_rgb_list.append(np.median(patch[zm], axis=0).astype(np.float64))
        else:
            zone_rgb_list.append(np.full(3, np.nan))
    white_zone_rgb = np.stack(zone_rgb_list, axis=0)   # (3, 3) — L/C/R

    return (
        white_full,
        black_full,
        bag_full,
        rw,
        rb,
        int(white_measure.sum()),
        int(black_measure.sum()),
        yw,
        yb,
        white_zone_rgb,
    )


def _stripe_mask_quality(yw: float, yb: float, nw: int, nb: int) -> bool:
    if nw < 400 or nb < 400:
        return False
    if yw < 0.12 or yb > 0.20:
        return False
    ratio = yw / max(yb, 1e-6)
    return 3.0 <= ratio <= 30.0


def segment_sephora_bag(
    rgb_u01: np.ndarray,
    face_landmarks: Sequence[Any],
    hand_landmarks: Optional[Sequence[Sequence[Any]]] = None,
    *,
    sam2_segmenter: Any = None,
    sam2_rgb_uint8: Optional[np.ndarray] = None,
) -> Optional[BagSegmentation]:
    """Segment full bag stripes; prefers SAM2+Hands, then Hands-only, then face mesh."""
    h, w = rgb_u01.shape[:2]
    mode = "face_mesh"
    bbox: Optional[Tuple[int, int, int, int]] = None
    paper_mask: Optional[np.ndarray] = None
    sam2_paper_mask: Optional[np.ndarray] = None

    # Primary: pure stripe-energy scan — no hands required
    bbox = bag_bbox_by_stripe_scan(rgb_u01, face_landmarks, h, w)
    if bbox is not None:
        mode = "stripe_scan"

    # Refine with hand bbox when two hands are available (tighter lateral bounds)
    if hand_landmarks is not None and len(hand_landmarks) >= 2:
        hand_bbox = bag_bbox_from_hands(face_landmarks, hand_landmarks, rgb_u01, h, w)
        if hand_bbox is not None:
            if bbox is not None:
                # Intersect the two estimates for tighter bounds
                hx0, hy0, hx1, hy1 = hand_bbox
                bx0, by0, bx1, by1 = bbox
                x0 = max(hx0, bx0)
                y0 = max(hy0, by0)
                x1 = min(hx1, bx1)
                y1 = min(hy1, by1)
                if x1 - x0 > 20 and y1 - y0 > 20:
                    bbox = _clamp_bbox(x0, y0, x1, y1, h, w)
                    mode = "stripe_scan+hands"
            else:
                _, _, _, iod_tmp = _face_metrics(face_landmarks, h, w)
                hand_bbox = _trim_bbox_by_edges(rgb_u01, *hand_bbox, iod_tmp)
                bbox = hand_bbox
                mode = "hands_full"

    # Last resort: face-mesh geometry estimate
    if bbox is None:
        bbox = bag_bbox_from_landmarks(face_landmarks, h, w)
        mode = "face_mesh"

    if (
        sam2_segmenter is not None
        and sam2_rgb_uint8 is not None
        and bbox is not None
    ):
        from sephora_bag_sam2 import (
            bag_sam2_box_from_hands,
            mask_coverage_in_box,
            sam2_stripe_point_prompts,
        )

        use_hand_prompts = hand_landmarks is not None and len(hand_landmarks) >= 2
        sam2_box = (
            bag_sam2_box_from_hands(face_landmarks, hand_landmarks, h, w)
            if use_hand_prompts
            else bbox
        )
        if sam2_box is not None:
            if use_hand_prompts:
                pts, labels = sam2_stripe_point_prompts(
                    rgb_u01, face_landmarks, hand_landmarks, sam2_box, h, w
                )
            else:
                # No hands: positive on bag centre, negative below bag
                from sephora_bag_sam2 import _black_stripe_row_centers
                bx0, by0, bx1, by1 = sam2_box
                pos = _black_stripe_row_centers(rgb_u01, sam2_box, max_points=4)
                if not pos:
                    pos = [((bx0 + bx1) * 0.5, (by0 + by1) * 0.5)]
                chin_y, iod, face_cx, _ = _face_metrics(face_landmarks, h, w)
                neg = [[face_cx, min(h - 4, by1 + 0.15 * iod)]]
                pts = np.vstack([np.asarray(pos, np.float32), np.asarray(neg, np.float32)])
                labels = np.array([1] * len(pos) + [0] * len(neg), dtype=np.int32)
            sam2_paper_mask = sam2_segmenter.predict_mask(
                sam2_rgb_uint8,
                sam2_box,
                point_coords=pts,
                point_labels=labels,
                rgb_u01=rgb_u01,
            )
            if mask_coverage_in_box(sam2_paper_mask, sam2_box):
                mode = mode.replace("stripe_scan", "stripe_scan+sam").replace("hands_full", "sam2_stripe")
                if "sam" not in mode:
                    mode += "+sam"

    if bbox is None:
        bbox = bag_bbox_from_landmarks(face_landmarks, h, w)
        mode = "face_mesh"

    use_hand_excl = (
        hand_landmarks is not None
        and len(hand_landmarks) >= 2
        and "hands" in mode or "sam2" in mode
    )
    seg = segment_bag_stripes_in_bbox(
        rgb_u01,
        bbox,
        hand_landmarks=hand_landmarks if use_hand_excl else None,
        paper_mask=None,
    )
    if seg is None:
        return None
    white_full, black_full, bag_full, rw, rb, nw, nb, yw, yb, wzone = seg

    # If SAM mask produced bad stripe stats, retry without it.
    if "sam" in mode and not _stripe_mask_quality(yw, yb, nw, nb):
        seg2 = segment_bag_stripes_in_bbox(
            rgb_u01, bbox, hand_landmarks=hand_landmarks, paper_mask=None
        )
        if seg2 is not None:
            white_full, black_full, bag_full, rw, rb, nw, nb, yw, yb, wzone = seg2
            mode = mode.replace("+sam", "").replace("sam2_stripe", "hands_full")
    return BagSegmentation(
        bag_bbox=bbox,
        white_mask=white_full,
        black_mask=black_full,
        bag_mask=bag_full,
        white_rgb_mean=rw,
        black_rgb_mean=rb,
        white_y=yw,
        black_y=yb,
        n_white=nw,
        n_black=nb,
        detection_mode=mode,
        white_zone_rgb=wzone,
    )


def segment_sephora_bag_from_landmarks(
    rgb_u01: np.ndarray,
    landmarks: Sequence[Any],
) -> Optional[BagSegmentation]:
    return segment_sephora_bag(rgb_u01, landmarks, hand_landmarks=None)


def exposure_scale_from_bag_white(seg: BagSegmentation, nix_white_y: float) -> float:
    return float(nix_white_y / max(seg.white_y, 1e-6))


def render_segmentation_overlay(
    bgr_preview: np.ndarray,
    seg: BagSegmentation,
    *,
    title: str = "",
    landmarks: Optional[Sequence[Any]] = None,
    hand_landmarks: Optional[Sequence[Sequence[Any]]] = None,
) -> np.ndarray:
    out = bgr_preview.copy()
    x0, y0, x1, y1 = seg.bag_bbox
    cv2.rectangle(out, (x0, y0), (x1, y1), (0, 255, 255), 2)
    h, w = bgr_preview.shape[:2]
    if landmarks is not None:
        chin = (int(landmarks[_CHIN].x * w), int(landmarks[_CHIN].y * h))
        cv2.circle(out, chin, 5, (255, 0, 255), -1)
    if hand_landmarks is not None:
        for hl in hand_landmarks:
            pts = hl.landmark if hasattr(hl, "landmark") else hl
            wrist = (int(pts[0].x * w), int(pts[0].y * h))
            cv2.circle(out, wrist, 4, (255, 255, 0), -1)
    green = np.array([0, 255, 0], dtype=np.uint8)
    orange = np.array([0, 128, 255], dtype=np.uint8)
    out[seg.white_mask] = (0.5 * out[seg.white_mask] + 0.5 * green).astype(np.uint8)
    out[seg.black_mask] = (0.5 * out[seg.black_mask] + 0.5 * orange).astype(np.uint8)
    if title:
        cv2.putText(out, title, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(out, title, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 1, cv2.LINE_AA)
    return out
