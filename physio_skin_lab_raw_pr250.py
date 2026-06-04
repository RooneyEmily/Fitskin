#!/usr/bin/env python3
"""
Physio calibration **Canon RAW** (.cr2 / .cr3): linear camera RGB (rawpy) → **24-patch ROI means**
in camera space → **least-squares 3×3 RGB → XYZ** fit to **PR-250** ``lighting_reference_patches``
(same MCC 0–23 order as ``mcc24_classic`` / ``validate_chart_rgb_vs_lighting_xyz.py``) → per-pixel XYZ
→ **Lab** (white point = **measured white patch XYZ**, index 18, same convention as chart validation).

By default rawpy uses **unity white balance** (no as-shot correction); the 3×3 fit partly absorbs
illuminant + camera RGB. For explicit balance use ``--raw-camera-wb`` (embedded CR2 multipliers) or
``--raw-chart-gray-wb`` (diagonal gains in linear camera RGB before the fit: patch **18** white by default,
or ``--raw-chart-gray-wb-from neutral_column_mean`` for mean of gray patches **19–22**). Do not pass both.

The PR-250 JSON rows **must** use ``mcc_patch_index`` **0…23** in **OpenCV mcc** / ``mcc24_classic`` order
(patch **0** = dark skin, **1** = light skin, **18** = white; **19–23** = Neutral 8 → … → Black on the bottom row).
On load, labels (and ``stem_canonical`` when present) are checked against that table; use
``--strict-pr250-patch-order`` to exit on mismatch.

Optional ``--patch-lstsq-upweight-skin F`` (``F`` > 1) multiplies row weights on MCC **0–1** only (extra
stress on skin rows on top of the default anchor weight).

By default ``--patch-lstsq-anchor-weight`` (> 1) multiplies row weights on **PR-250 skin + grayscale**
patches **0, 1, 18–23** (dark skin, light skin, white, neutrals, black) so the 3×3/affine map tracks
spectrometer anchors that matter for facial skin and the neutral axis; chromatic patches still
participate at weight 1. Set to **1** for uniform row weights (before Huber IRLS).

By default ``--patch-lstsq-robust huber`` runs **iteratively reweighted** least squares: Huber weights from
per-patch **ΔE_ab** (same metric as ``patch_de_ab_mean``) so spectrally poorly fit primaries (often **Blue** /
**Green** under a linear 3×3) do not dominate ``M``. Use ``--patch-lstsq-robust none`` for plain OLS.

**Colorimetric grounding (vs Lightroom / DNG profile):** A **DNG camera profile** is typically a full
3D/LUT-style transform learned from many patches under one workflow. Here we **do not** embed Adobe’s
profile: we **measure** each patch’s linear camera RGB in the .cr2, **pair row ``i``** with **PR-250
spectrometer XYZ** for the same ``mcc_patch_index``, and solve a **linear 3×3** (or 24×4 affine) so
**all 24 patches** jointly pin camera space to instrument XYZ. Facial skin is never a patch on the
chart; its accuracy comes from that global mapping plus Lab white: **Bradford CAT to D65 is the default**
(same white point as ``physio_skin_lab_monk`` / Cheng MST Table I). Use ``--no-lab-cat-to-d65`` only for
scene-referred Lab (XYZn = PR-250 white patch 18).

**``--raw-chart-gray-wb`` (“gray white balance”):** Only **three scalar gains** (one per camera R, G, B)
applied to the **linear** image so a **chosen chart reference** (white patch 18, or mean of neutral
grays **19–22**) has **R=G=B** before the matrix fit. That mimics the *idea* of clicking a gray swatch
with the WB eyedropper (remove channel imbalance) but **does not** replace the 24-patch **color**
fit—it runs **before** lstsq so the matrix still sees neutralized raw-ish RGB. Default is **unity**
WB (no gains); many runs leave it off and let the 3×3 absorb illuminant + camera coupling.

**Visual index check:** ``--write-chart-mcc-overlays`` writes ``*_mcc24_indices.png`` using the **same**
``mcc`` quads as ROI sampling (not a second detection). Confirm **0** = dark skin chip, **1** = light skin,
**18** = white, **19–22** = Neutral 8 … Neutral 3.5, **23** = black.

Then: **MediaPipe face mesh** + **same skin mask** as ``physio_skin_lab_monk.py`` (tessellation default),
optional L* / chroma trims, CSV + JSON + optional overlays / L* histograms.

**Chart geometry** comes from **OpenCV mcc** on an **8-bit preview** built from the linear RAW (robust
per-channel percentile stretch). **Patch colors** for the fit are **means inside mcc quads** on the
**linear float** image so the matrix maps true camera responses to spectrometer XYZ.

Depends:
  pip install rawpy opencv-contrib-python mediapipe scikit-image scipy matplotlib numpy
  (optional ``colour_checker_detection`` not required here — mcc only for patch quads)

RAW discovery: ``--data-root`` / ``**/P1/Photos`` / ``*.cr2`` or ``*.cr3`` (nested ``…/<pid>/P1/Photos`` is OK).
Default ``--data-root`` is ``$PHYSIO_DATA_ROOT`` if set, else ``/media/mabl-main/Data/Physio-code/Data``.

**Validate patch fit:** ``validate_pr250_raw_patch_fit.py`` recomputes per-patch ΔE_ab from each ``image_path``
in the photo CSV and writes bar charts + ``aggregate_patch_de_ab.png`` (see that script's ``--help``).

Example::
    --data-root /media/mabl-main/Data/Physio-code/Data \\
    --pr250-json ./lighting_output/lighting_reference_patches.json \\
    --out-dir ./skin_lab_raw_pr250_output \\
    --one-image-per-participant --pick-photo largest_chart \\
    --raw-half-size 1 \\
    --raw-camera-wb \\
    --write-skin-mask-overlays --skin-overlay-max-width 1600 \\
    --write-chart-mcc-overlays
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import cv2
import numpy as np

try:
    import rawpy
except ImportError:
    rawpy = None  # type: ignore[misc, assignment]  # required only for .cr2/.cr3 paths

try:
    from scipy import stats as scipy_stats
except ImportError as e:
    raise SystemExit("pip install scipy") from e

plt = None  # set below if matplotlib loads; always defined for main()
try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except BaseException:
    plt = None  # type: ignore[misc, assignment]

import mediapipe as mp

from mcc24_classic import (
    NEUTRAL_COLUMN_WB_PATCH_INDICES,
    PR250_SKIN_NEUTRAL_ANCHOR_PATCH_INDICES,
    WHITE_PATCH_INDEX,
    draw_mcc_patch_overlay_from_quads,
    get_patch_quads_xy,
    patch_label_slug,
    patch_stem_suffix,
)

import physio_skin_lab_monk as psl

from delta_e_2000 import de2000_csv_header, de2000_csv_values, load_mst_lab_matrix_10x3, mst_de2000_row


# Override without editing the command line: ``export PHYSIO_DATA_ROOT=/path/to/Physio-code/Data``
_DEFAULT_PHYSIO_DATA_ROOT = Path(os.environ.get("PHYSIO_DATA_ROOT", "/media/mabl-main/Data/Physio-code/Data"))


# CIE D65 reference white (Y = 1). Used for skin Lab after Bradford CAT (default ``--lab-cat-to-d65``).
# Scene XYZ from the lstsq fit shares the PR-250 absolute scale; before CAT we divide by the
# measured white-patch Y so pixel XYZ and D65 white are both on Y=1-relative units for Lab.
D65_XYZ_Y1 = np.array([0.95047, 1.0, 1.08883], dtype=np.float64)


def bradford_cat_matrix(xyz_w_src: np.ndarray, xyz_w_dst: np.ndarray) -> np.ndarray:
    """3×3 Bradford–von Kries chromatic adaptation between absolute (or consistent) XYZ white points."""
    M_lms = np.array(
        [
            [0.8951, 0.2664, -0.1614],
            [-0.7502, 1.7135, 0.0367],
            [0.0389, -0.0685, 1.0296],
        ],
        dtype=np.float64,
    )
    s = np.asarray(xyz_w_src, dtype=np.float64).reshape(3)
    d = np.asarray(xyz_w_dst, dtype=np.float64).reshape(3)
    lms_s = M_lms @ s
    lms_d = M_lms @ d
    rho = lms_d / np.maximum(lms_s, 1e-12)
    return np.linalg.inv(M_lms) @ np.diag(rho) @ M_lms


def apply_bradford_cat_hwc(xyz_hwc: np.ndarray, xyz_w_src: np.ndarray, xyz_w_dst: np.ndarray) -> np.ndarray:
    """Apply Bradford CAT to each pixel; ``xyz_hwc`` shape (H,W,3)."""
    cat = bradford_cat_matrix(xyz_w_src, xyz_w_dst)
    flat = xyz_hwc.reshape(-1, 3)
    return (flat @ cat.T).reshape(xyz_hwc.shape)


# --- Lab from XYZ (same as validate_chart_rgb_vs_lighting_xyz) -----------------


def _f_xyz_to_lab_ratio(t: np.ndarray) -> np.ndarray:
    t = np.asarray(t, dtype=np.float64)
    out = np.empty_like(t, dtype=np.float64)
    m = t <= 216.0 / 24389.0
    out[m] = ((24389.0 / 27.0) * t[m] + 16.0) / 116.0
    out[~m] = np.cbrt(t[~m])
    return out


def xyz_to_lab(xyz: np.ndarray, xyzn: np.ndarray) -> np.ndarray:
    xyz = np.asarray(xyz, dtype=np.float64).reshape(3)
    w = np.asarray(xyzn, dtype=np.float64).reshape(3)
    r = xyz / np.maximum(w, 1e-12)
    fx, fy, fz = _f_xyz_to_lab_ratio(r[0]), _f_xyz_to_lab_ratio(r[1]), _f_xyz_to_lab_ratio(r[2])
    return np.array([116.0 * fy - 16.0, 500.0 * (fx - fy), 200.0 * (fy - fz)], dtype=np.float64)


def xyz_to_lab_batch(xyz: np.ndarray, xyzn: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """xyz (N,3) → L*, a*, b* with same white point for all rows."""
    w = np.asarray(xyzn, dtype=np.float64).reshape(1, 3)
    r = np.asarray(xyz, dtype=np.float64) / np.maximum(w, 1e-12)
    fx = _f_xyz_to_lab_ratio(r[:, 0])
    fy = _f_xyz_to_lab_ratio(r[:, 1])
    fz = _f_xyz_to_lab_ratio(r[:, 2])
    L = 116.0 * fy - 16.0
    a = 500.0 * (fx - fy)
    b = 200.0 * (fy - fz)
    return L, a, b


def delta_e_ab(lab1: np.ndarray, lab2: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(lab1) - np.asarray(lab2)))


def load_pr250_xyz(path: Path) -> Tuple[np.ndarray, List[str], List[str]]:
    """
    Reference patch XYZ under your illuminant, rows mcc index 0..23.
    Accepts ``lighting_reference_patches.json`` (list of dicts) or ``.csv`` (same columns as validate script).

    Returns ``(xyz (24,3), patch_labels, stem_canonical_per_row)``. Stems may be empty strings if absent.
    """
    path = path.expanduser().resolve()
    if not path.is_file():
        raise SystemExit(f"PR-250 reference not found: {path}")
    if path.suffix.lower() == ".json":
        rows = json.loads(path.read_text(encoding="utf-8"))
        by_i: Dict[int, Tuple[float, float, float, str, str]] = {}
        for row in rows:
            if not isinstance(row, dict):
                continue
            k = int(row["mcc_patch_index"])
            by_i[k] = (
                float(row["X_mean"]),
                float(row["Y_mean"]),
                float(row["Z_mean"]),
                str(row.get("patch_label") or ""),
                str(row.get("stem_canonical") or row.get("stem_group") or ""),
            )
        if set(by_i.keys()) != set(range(24)):
            raise SystemExit(f"JSON {path}: expected mcc_patch_index 0..23, got {sorted(by_i.keys())}")
        labels = [by_i[i][3] for i in range(24)]
        stems = [by_i[i][4] for i in range(24)]
        xyz = np.array([[by_i[i][0], by_i[i][1], by_i[i][2]] for i in range(24)], dtype=np.float64)
        return xyz, labels, stems
    # CSV
    by_i = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            k = int(row["mcc_patch_index"])
            by_i[k] = (
                float(row["X_mean"]),
                float(row["Y_mean"]),
                float(row["Z_mean"]),
                str(row.get("patch_label") or ""),
                str(row.get("stem_canonical") or row.get("stem_group") or ""),
            )
    if set(by_i.keys()) != set(range(24)):
        raise SystemExit(f"CSV {path}: expected mcc_patch_index 0..23, got {sorted(by_i.keys())}")
    labels = [by_i[i][3] for i in range(24)]
    stems = [by_i[i][4] for i in range(24)]
    xyz = np.array([[by_i[i][0], by_i[i][1], by_i[i][2]] for i in range(24)], dtype=np.float64)
    return xyz, labels, stems


def _norm_pr250_label_slug(s: str) -> str:
    t = (s or "").strip().lower().replace("-", "_")
    out: List[str] = []
    for ch in t:
        if ch.isalnum() or ch == "_":
            out.append(ch)
        elif ch.isspace():
            out.append("_")
    return "".join(out)


def validate_pr250_reference_alignment(
    path: Path,
    labels: List[str],
    stems: List[str],
    *,
    strict: bool,
) -> None:
    """
    Ensure ``lighting_reference_patches`` rows align with OpenCV mcc / ``mcc24_classic`` patch order.

    Compares non-empty ``patch_label`` to ``patch_label_slug(i)`` and non-empty ``stem_canonical`` to
    ``patch_stem_suffix(i)`` (case-insensitive).
    """
    issues: List[str] = []
    for i in range(24):
        exp_l_raw = patch_label_slug(i)
        exp_l = _norm_pr250_label_slug(exp_l_raw)
        got_raw = labels[i] if i < len(labels) else ""
        got_l = _norm_pr250_label_slug(got_raw)
        if got_l and got_l != exp_l:
            issues.append(
                f"  mcc_patch_index {i}: patch_label {got_raw!r} — expected slug {exp_l_raw!r} "
                f"(normalized {got_l!r} vs {exp_l!r})"
            )
        exp_s = patch_stem_suffix(i).lower()
        got_s = (stems[i] or "").strip().lower() if i < len(stems) else ""
        if got_s and got_s != exp_s:
            issues.append(
                f"  mcc_patch_index {i}: stem_canonical/group {stems[i]!r} — expected {patch_stem_suffix(i)!r}"
            )
    if not issues:
        print(
            f"PR-250 reference patch order OK ({path.name}): rows 0–23 match mcc24_classic / OpenCV mcc.",
            flush=True,
        )
        return
    msg = f"PR-250 vs mcc24_classic alignment issues for {path}:\n" + "\n".join(issues)
    if strict:
        raise SystemExit(msg + "\nFix the JSON/CSV or drop --strict-pr250-patch-order (warnings only).")
    print(f"Warning:\n{msg}", file=sys.stderr)


def _require_rawpy():
    if rawpy is None:
        raise SystemExit("Install rawpy for Canon RAW (.cr2/.cr3): pip install rawpy")


def read_raw_linear_rgb(path: Path, *, half_size: int = 1, use_camera_wb: bool = False) -> np.ndarray:
    """
    Demosaiced **linear** camera RGB, float64, shape (H,W,3). ``half_size`` is rawpy's speed/resolution tradeoff
    (0 = full res; 1 = half linear dimensions each axis, etc.).

    Default ``use_camera_wb=False`` applies **unity** WB (no as-shot correction). Set
    ``use_camera_wb=True`` to use the camera / file white-balance multipliers from the RAW metadata.

    The returned array is **linear camera RGB in raw count space** (``output_bps=16`` → typically
    0–65535 as ``float64``), **not** normalized ``[0,1]`` sRGB. The 3×3 lstsq and ``@ M`` pipeline stays
    self-consistent; do **not** feed this array to ``skimage.color.rgb2lab`` without a proper transform.
    """
    _require_rawpy()
    kw = dict(
        output_color=rawpy.ColorSpace.raw,
        output_bps=16,
        no_auto_bright=True,
        gamma=(1, 1),
        half_size=int(half_size),
    )
    with rawpy.imread(str(path)) as raw:
        if use_camera_wb:
            rgb = raw.postprocess(
                use_camera_wb=True,
                use_auto_wb=False,
                **kw,
            )
        else:
            rgb = raw.postprocess(
                use_camera_wb=False,
                use_auto_wb=False,
                user_wb=(1.0, 1.0, 1.0, 1.0),
                **kw,
            )
    return np.asarray(rgb, dtype=np.float64)


def apply_diagonal_gray_wb_from_chart_patches(
    rgb_lin: np.ndarray,
    patch_rgb: np.ndarray,
    *,
    wb_from: str = "white",
    white_patch_index: int = WHITE_PATCH_INDEX,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Per-channel gains so a **reference** linear RGB vector is neutral (R=G=B), then applied to the
    full image and ``patch_rgb`` before the PR-250 lstsq fit.

    ``wb_from``:
      - ``white``: use MCC white patch (default index 18).
      - ``neutral_column_mean``: mean linear RGB of ``NEUTRAL_COLUMN_WB_PATCH_INDICES`` (Neutral 8 …
        Neutral 3.5; excludes white 18 and black 23).
    """
    if wb_from == "white":
        w = patch_rgb[int(white_patch_index)].astype(np.float64)
    elif wb_from == "neutral_column_mean":
        idx = np.asarray(NEUTRAL_COLUMN_WB_PATCH_INDICES, dtype=int)
        w = patch_rgb[idx].astype(np.float64).mean(axis=0)
    else:
        raise ValueError(f"unknown wb_from {wb_from!r}")
    m = float(np.mean(w))
    gain = m / np.maximum(w, 1e-12)
    rgb2 = rgb_lin * gain.reshape(1, 1, 3)
    patch2 = patch_rgb * gain.reshape(1, 3)
    return rgb2, patch2


def linear_rgb_to_preview_bgr(rgb_lin: np.ndarray) -> np.ndarray:
    """Robust 8-bit BGR preview for OpenCV mcc / MediaPipe (detection only)."""
    out = np.zeros_like(rgb_lin, dtype=np.float64)
    for c in range(3):
        ch = rgb_lin[:, :, c].astype(np.float64).ravel()
        lo, hi = np.percentile(ch, [0.5, 99.5])
        if hi <= lo + 1e-12:
            hi = lo + 1e-6
        out[:, :, c] = np.clip((rgb_lin[:, :, c] - lo) / (hi - lo), 0.0, 1.0)
    rgb8 = (out * 255.0).astype(np.uint8)
    return cv2.cvtColor(rgb8, cv2.COLOR_RGB2BGR)


def roi_mean_linear_rgb(
    rgb_lin: np.ndarray,
    quad_xy: np.ndarray,
    *,
    use_median: bool = True,
    center_fraction: float = 0.7,
) -> np.ndarray:
    """
    Aggregate linear RGB inside the convex ColorChecker quad (same geometry as mcc getColorCharts).

    ``use_median=True`` (default) is more robust to specular hotspots and border bleed than the mean.
    ``center_fraction`` in (0, 1) erodes the filled quad so only the central fraction of the area is
    sampled; set to ``0`` or ``>= 1`` to disable erosion.
    """
    h, w = rgb_lin.shape[:2]
    poly = np.round(quad_xy).astype(np.int32)
    poly[:, 0] = np.clip(poly[:, 0], 0, w - 1)
    poly[:, 1] = np.clip(poly[:, 1], 0, h - 1)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillConvexPoly(mask, poly, 255)

    if 0.0 < center_fraction < 1.0:
        area = float(np.count_nonzero(mask))
        if area > 0.0:
            erode_px = max(1, int(round((1.0 - center_fraction) * np.sqrt(area / np.pi))))
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (2 * erode_px + 1, 2 * erode_px + 1)
            )
            mask = cv2.erode(mask, kernel, iterations=1)

    m = mask > 0
    if not np.any(m):
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillConvexPoly(mask, poly, 255)
        m = mask > 0
    if not np.any(m):
        return np.zeros(3, dtype=np.float64)
    pixels = rgb_lin[m]
    if use_median:
        return np.median(pixels, axis=0).astype(np.float64)
    return pixels.mean(axis=0).astype(np.float64)


def mcc_quads_from_bgr(bgr_u8: np.ndarray) -> Optional[Tuple[Any, np.ndarray]]:
    """Returns (checker, quads (24,4,2)) or None."""
    if not hasattr(cv2, "mcc"):
        return None
    detector = cv2.mcc.CCheckerDetector.create()
    if not detector.process(bgr_u8, 0, 1):
        return None
    checkers = detector.getListColorChecker()
    if not checkers:
        return None
    checker = checkers[0]
    quads = get_patch_quads_xy(checker)
    if quads is None:
        return None
    return checker, quads


def patch_linear_rgb_24(
    rgb_lin: np.ndarray,
    preview_bgr: np.ndarray,
    *,
    use_median: bool = True,
    center_fraction: float = 0.7,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    ``(24,3)`` linear camera RGB per patch (median or mean over ROI) and ``(24,4,2)`` quad corners
    from the **same** ``mcc`` detection, or ``None`` if mcc fails.

    Quads come from ``mcc24_classic.get_patch_quads_xy``: **pixel** ``(x,y)`` corners in the same
    resolution as ``preview_bgr`` / ``rgb_lin`` (``rawpy`` ``half_size`` must match for both arrays).
    """
    got = mcc_quads_from_bgr(preview_bgr)
    if got is None:
        return None
    _checker, quads = got
    out = np.zeros((24, 3), dtype=np.float64)
    for i in range(24):
        out[i] = roi_mean_linear_rgb(
            rgb_lin,
            quads[i],
            use_median=use_median,
            center_fraction=center_fraction,
        )
    return out, quads


def build_patch_lstsq_row_weights(
    *,
    anchor_weight: float,
    skin_weight: float,
) -> Optional[np.ndarray]:
    """
    Row weights for 24-patch RGB→XYZ lstsq: **PR-250 skin + grayscale** anchors (indices
    ``PR250_SKIN_NEUTRAL_ANCHOR_PATCH_INDICES``) × ``anchor_weight``, then patches **0–1** × ``skin_weight``
    again. Returns ``None`` if both weights are 1 (ordinary unweighted lstsq before Huber).
    """
    aw = float(anchor_weight)
    sw = float(skin_weight)
    if aw == 1.0 and sw == 1.0:
        return None
    if aw <= 0.0 or sw <= 0.0:
        raise ValueError("anchor_weight and skin_weight must be > 0")
    w = np.ones(24, dtype=np.float64)
    for i in PR250_SKIN_NEUTRAL_ANCHOR_PATCH_INDICES:
        w[i] *= aw
    w[0] *= sw
    w[1] *= sw
    return w


def fit_rgb_to_xyz_lstsq(
    patch_rgb: np.ndarray,
    ref_xyz: np.ndarray,
    *,
    with_intercept: bool = False,
    row_weights: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    ``patch_rgb`` ``(24,3)``, ``ref_xyz`` ``(24,3)`` → ``M`` with ``ref ≈ A @ M``.

    If ``with_intercept``, ``A`` is ``(24,4)`` with a column of ones (affine camera RGB → XYZ).

    Optional ``row_weights`` shape ``(24,)`` (nonnegative): weighted least squares (higher weight =
    more influence on ``M``). Unity weights recover ordinary lstsq.
    """
    n = int(patch_rgb.shape[0])
    if row_weights is None:
        sw = np.ones(n, dtype=np.float64)
    else:
        sw = np.sqrt(np.maximum(np.asarray(row_weights, dtype=np.float64).reshape(n), 0.0))
    if with_intercept:
        A = np.column_stack([patch_rgb, np.ones((n, 1), dtype=np.float64)])
    else:
        A = patch_rgb
    Aw = A * sw[:, np.newaxis]
    try:
        cond = float(np.linalg.cond(Aw))
    except Exception:
        cond = float("nan")
    if np.isfinite(cond) and cond > 1000.0:
        print(
            f"Warning: patch RGB design matrix cond≈{cond:.0f} ({A.shape[0]}×{A.shape[1]}) — lstsq may be unstable",
            file=sys.stderr,
        )
    m_cols: List[np.ndarray] = []
    for k in range(3):
        bw = ref_xyz[:, k] * sw
        x, *_ = np.linalg.lstsq(Aw, bw, rcond=None)
        m_cols.append(np.asarray(x, dtype=np.float64))
    return np.column_stack(m_cols)


def per_patch_delta_e_ab(
    patch_rgb: np.ndarray,
    ref_xyz: np.ndarray,
    M: np.ndarray,
    with_intercept: bool,
) -> np.ndarray:
    """
    Per-patch **ΔE_ab** (Euclidean in Lab, ``XYZn`` = PR-250 white patch 18) between reference and
    fitted patch XYZ. Shape ``(24,)``, index = ``mcc_patch_index``.
    """
    if with_intercept:
        aug = np.column_stack([patch_rgb, np.ones((patch_rgb.shape[0], 1), dtype=np.float64)])
        pred = aug @ M
    else:
        pred = patch_rgb @ M
    n = int(patch_rgb.shape[0])
    white_i = int(WHITE_PATCH_INDEX) if n >= 24 else 0
    xyzn = ref_xyz[white_i].copy()
    des = np.zeros(n, dtype=np.float64)
    for i in range(n):
        des[i] = delta_e_ab(xyz_to_lab(ref_xyz[i], xyzn), xyz_to_lab(pred[i], xyzn))
    return des


def mean_patch_de_ab(
    patch_rgb: np.ndarray,
    ref_xyz: np.ndarray,
    M: np.ndarray,
    with_intercept: bool,
) -> float:
    """Mean **ΔE_ab** (Euclidean Δ in Lab) between reference and predicted patch XYZ — not ΔE₀₀.

    ``with_intercept`` must match how ``M`` was fitted (3×3 vs 4×3).
    """
    return float(np.mean(per_patch_delta_e_ab(patch_rgb, ref_xyz, M, with_intercept)))


def fit_rgb_to_xyz_lstsq_huber_irls(
    patch_rgb: np.ndarray,
    ref_xyz: np.ndarray,
    *,
    with_intercept: bool = False,
    row_weights: Optional[np.ndarray] = None,
    max_iter: int = 8,
    tuning_constant: float = 1.345,
    weight_rel_tol: float = 1e-3,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Iteratively reweighted least squares: **Huber** weights from per-patch **ΔE_ab** (``XYZn`` = white
    patch 18 — same as ``per_patch_delta_e_ab``). Downweights rows whose camera RGB is poorly explained
    by a linear (or affine) map to PR-250 XYZ (common on saturated primaries such as Blue / Green).

    ``row_weights`` are base weights (e.g. skin-row upweight), multiplied by Huber factors each iteration.
    """
    n = int(patch_rgb.shape[0])
    base = np.ones(n, dtype=np.float64)
    if row_weights is not None:
        rw = np.asarray(row_weights, dtype=np.float64).reshape(n)
        base = np.maximum(rw, 1e-12)
    w = base.copy()
    M: Optional[np.ndarray] = None
    last_it = 0
    for it in range(max_iter):
        M = fit_rgb_to_xyz_lstsq(
            patch_rgb, ref_xyz, with_intercept=with_intercept, row_weights=w
        )
        de = per_patch_delta_e_ab(patch_rgb, ref_xyz, M, with_intercept)
        med_de = float(np.median(de))
        mad = float(np.median(np.abs(de - med_de)))
        # Scale σ from MAD but keep a floor tied to typical patch error so mild runs are not over-penalized.
        sigma = max(1.4826 * mad, 0.25 * med_de + 1e-9, 0.75)
        thresh = float(tuning_constant) * sigma
        huber = np.minimum(1.0, thresh / np.maximum(de, 1e-12))
        w_new = base * huber.astype(np.float64)
        scale_w = float(np.max(w)) if float(np.max(w)) > 0.0 else 1.0
        Dw = float(np.max(np.abs(w_new - w)))
        w = w_new
        last_it = it + 1
        if Dw <= weight_rel_tol * max(scale_w, 1.0):
            break
    assert M is not None
    return M, w, last_it


def raw_preview_bgr_for_chart_score(
    path: Path,
    *,
    half_size: int,
    use_camera_wb: bool = False,
) -> Optional[np.ndarray]:
    try:
        rgb = read_raw_linear_rgb(path, half_size=half_size, use_camera_wb=use_camera_wb)
        return linear_rgb_to_preview_bgr(rgb)
    except SystemExit:
        raise
    except Exception as ex:
        if rawpy is not None:
            unsupported = getattr(rawpy, "LibRawFileUnsupportedError", None)
            if unsupported is not None and isinstance(ex, unsupported):
                return None
        print(f"Warning: could not decode RAW {path}: {ex}", file=sys.stderr)
        return None


def pick_one_raw_per_participant(
    paths: List[Path],
    strategy: str,
    *,
    raw_half_size: int,
    use_camera_wb: bool = False,
    debug: bool = False,
) -> List[Path]:
    from collections import defaultdict

    by_pid: Dict[str, List[Path]] = defaultdict(list)
    for p in paths:
        try:
            pid = p.parent.parent.parent.name
        except Exception:
            continue
        by_pid[pid].append(p)

    print(f"Picking one RAW per participant ({strategy}) for {len(by_pid)} participant(s)…", flush=True)
    out: List[Path] = []
    for pid in sorted(by_pid.keys(), key=lambda s: (0, int(s)) if str(s).isdigit() else (1, s)):
        lst = sorted(by_pid[pid], key=lambda x: x.name)
        if not lst:
            continue
        if strategy == "first_sorted":
            chosen = lst[0]
        elif strategy == "newest_jpg":
            chosen = max(lst, key=lambda p: psl._photo_timestamp_key(p))
        elif strategy == "largest_chart":
            ranked: List[Tuple[float, str, str, Path]] = []
            for p in lst:
                bgr = raw_preview_bgr_for_chart_score(p, half_size=raw_half_size, use_camera_wb=use_camera_wb)
                if bgr is None:
                    if debug:
                        print(f"pick_photo: could not read {p}", file=sys.stderr)
                    continue
                score = psl.chart_area_score_bgr(bgr)
                ranked.append((score, psl._photo_timestamp_key(p), p.name, p))
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


def find_calibration_raw_paths(data_root: Path) -> List[Path]:
    """
    Canon RAW paths under ``data_root/**/P1/Photos`` (any depth: supports ``Data/<pid>/P1/Photos`` and
    ``Data/<cohort>/<pid>/P1/Photos``).
    """
    root = Path(data_root).expanduser().resolve()
    out: List[Path] = []
    hits = sorted({p for p in root.glob("**/P1/Photos") if p.is_dir()})
    for photos in hits:
        for ext in ("*.cr2", "*.CR2", "*.cr3", "*.CR3"):
            out.extend(photos.glob(ext))
    return sorted({p.resolve() for p in out})


@contextlib.contextmanager
def _silence_stderr() -> Iterator[None]:
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


def _pixel_sem(x: np.ndarray) -> float:
    """Standard error of the mean over skin-mask Lab samples (ddof=1)."""
    n = int(np.asarray(x).size)
    if n < 2:
        return float("nan")
    return float(np.std(x, ddof=1) / np.sqrt(n))


def mean_lab_masked_xyz_scene(
    xyz_img: np.ndarray,
    mask: np.ndarray,
    xyzn: np.ndarray,
    *,
    l_star_trim_lo: float,
    l_star_trim_hi: float,
    skin_min_chroma_ab: float,
    histogram_png: Optional[Path],
    histogram_title: str,
    a_star_trim_lo: float = 0.0,
    a_star_trim_hi: float = 0.0,
    b_star_trim_lo: float = 0.0,
    b_star_trim_hi: float = 0.0,
) -> Tuple[float, float, float, float, float, float, float, int, int, float, float, bool, bool, bool, bool, int]:
    """
    Same trim logic as physio_skin_lab_monk.mean_lab_masked, but Lab from scene-referred XYZ.

    Skin pixels with any negative XYZ (infeasible extrapolation of the 3×3/affine fit) are excluded
    from the mean rather than clipped; see returned ``n_skin_pixels_xyz_invalid``.

    Returns ``n_raw`` = count of skin pixels with valid (nonnegative) XYZ **before** L* / chroma
    trimming (not the full mesh-mask count).
    """
    ms = mask > 0
    xyz_valid = np.all(xyz_img >= 0.0, axis=2)
    n_skin_pixels_xyz_invalid = int(np.count_nonzero(ms & ~xyz_valid))
    m = ms & xyz_valid
    if not np.any(m):
        return (
            float("nan"),
            float("nan"),
            float("nan"),
            float("nan"),
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
            n_skin_pixels_xyz_invalid,
        )
    xyz_flat = xyz_img[m].reshape(-1, 3)
    L_flat, a_flat, b_flat = xyz_to_lab_batch(xyz_flat, xyzn)
    n_raw = int(L_flat.size)
    lo_thr = float("nan")
    hi_thr = float("nan")
    tlo = float(l_star_trim_lo)
    thi = float(l_star_trim_hi)
    if tlo > 0.0:
        lo_thr = float(np.quantile(L_flat, min(tlo, 0.45)))
    if thi > 0.0:
        hi_thr = float(np.quantile(L_flat, 1.0 - min(thi, 0.45)))

    sel, _sel_ch, chroma_relaxed, ltrim_relaxed, l_trim_effective, chroma_trim_effective = psl.skin_lab_trim_selection(
        L_flat,
        a_flat,
        b_flat,
        l_star_trim_lo=l_star_trim_lo,
        l_star_trim_hi=l_star_trim_hi,
        a_star_trim_lo=a_star_trim_lo,
        a_star_trim_hi=a_star_trim_hi,
        b_star_trim_lo=b_star_trim_lo,
        b_star_trim_hi=b_star_trim_hi,
        min_chroma_ab=skin_min_chroma_ab,
    )
    n_kept = int(np.count_nonzero(sel))

    if histogram_png is not None and plt is not None:
        psl.write_skin_lab_histogram_panel(
            histogram_png,
            L_flat.astype(np.float64),
            a_flat.astype(np.float64),
            b_flat.astype(np.float64),
            sel,
            lo_thr=lo_thr,
            hi_thr=hi_thr,
            min_chroma_ab=skin_min_chroma_ab,
            l_trim_relaxed=ltrim_relaxed,
            chroma_relaxed=chroma_relaxed,
            title=histogram_title or "Skin Lab (masked, scene PR-250 matrix)",
            lab_source="scene Lab, PR-250 ref",
        )

    if n_kept == 0:
        return (
            float("nan"),
            float("nan"),
            float("nan"),
            float("nan"),
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
            n_skin_pixels_xyz_invalid,
        )
    Ls = L_flat[sel]
    asel = a_flat[sel]
    bsel = b_flat[sel]
    Csel = np.hypot(asel, bsel)
    return (
        float(Ls.mean()),
        float(asel.mean()),
        float(bsel.mean()),
        _pixel_sem(Ls),
        _pixel_sem(asel),
        _pixel_sem(bsel),
        _pixel_sem(Csel),
        n_kept,
        n_raw,
        lo_thr,
        hi_thr,
        l_trim_effective,
        chroma_trim_effective,
        chroma_relaxed,
        ltrim_relaxed,
        n_skin_pixels_xyz_invalid,
    )


@dataclass
class RawPhotoResult:
    participant_id: str
    image_path: str
    L_mean: float
    a_mean: float
    b_mean: float
    L_sem: float
    a_sem: float
    b_sem: float
    C_sem: float
    n_pixels: int
    chart_ok: bool
    n_skin_pixels_mesh_mask: int
    n_skin_pixels_valid_xyz: int
    patch_de_ab_mean: float
    l_star_trim_lo_frac: float
    l_star_trim_hi_frac: float
    l_star_threshold_lo: float
    l_star_threshold_hi: float
    skin_min_chroma_ab: float
    skin_l_trim_effective: bool = True
    skin_chroma_trim_effective: bool = True
    raw_wb_mode: str = "unity"
    lstsq_intercept: bool = False
    lab_cat_to_d65: bool = True
    n_skin_pixels_xyz_invalid: int = 0
    patch_lstsq_robust: str = "none"


def process_one_raw(
    path: Path,
    face_mesh: Any,
    ref_xyz: np.ndarray,
    *,
    raw_half_size: int,
    skin_exclusion_dilate_iod_fraction: float,
    skin_triangulation: str,
    l_star_trim_lo: float,
    l_star_trim_hi: float,
    skin_min_chroma_ab: float,
    a_star_trim_lo: float = 0.0,
    a_star_trim_hi: float = 0.0,
    b_star_trim_lo: float = 0.0,
    b_star_trim_hi: float = 0.0,
    skin_l_histogram_dir: Optional[Path],
    skin_overlay_dir: Optional[Path],
    skin_overlay_max_width: int,
    chart_mcc_overlay_dir: Optional[Path],
    raw_camera_wb: bool = False,
    raw_chart_gray_wb: bool = False,
    chart_gray_wb_from: str = "white",
    raw_wb_mode: str = "unity",
    lstsq_intercept: bool = False,
    lab_cat_to_d65: bool = True,
    patch_rgb_use_median: bool = True,
    patch_center_fraction: float = 0.7,
    patch_row_weights: Optional[np.ndarray] = None,
    patch_lstsq_robust: str = "none",
    debug: bool = False,
) -> Optional[RawPhotoResult]:
    pid = path.parent.parent.parent.name
    try:
        rgb_lin = read_raw_linear_rgb(path, half_size=raw_half_size, use_camera_wb=raw_camera_wb)
    except Exception as ex:
        if debug:
            print(f"raw read failed {path}: {ex}", file=sys.stderr)
        return None
    preview_bgr = linear_rgb_to_preview_bgr(rgb_lin)
    got_patches = patch_linear_rgb_24(
        rgb_lin,
        preview_bgr,
        use_median=patch_rgb_use_median,
        center_fraction=patch_center_fraction,
    )
    if got_patches is None:
        if debug:
            print(f"mcc chart failed {path}", file=sys.stderr)
        return None
    patch_rgb, mcc_quads_xy = got_patches
    if raw_chart_gray_wb:
        rgb_lin, patch_rgb = apply_diagonal_gray_wb_from_chart_patches(
            rgb_lin, patch_rgb, wb_from=chart_gray_wb_from
        )
        preview_bgr = linear_rgb_to_preview_bgr(rgb_lin)
    h, w = rgb_lin.shape[:2]
    if patch_lstsq_robust == "huber":
        M, _w_huber, _n_irls = fit_rgb_to_xyz_lstsq_huber_irls(
            patch_rgb,
            ref_xyz,
            with_intercept=lstsq_intercept,
            row_weights=patch_row_weights,
        )
    elif patch_lstsq_robust == "none":
        M = fit_rgb_to_xyz_lstsq(
            patch_rgb,
            ref_xyz,
            with_intercept=lstsq_intercept,
            row_weights=patch_row_weights,
        )
    else:
        raise ValueError(f"unknown patch_lstsq_robust {patch_lstsq_robust!r}")
    de_mean = mean_patch_de_ab(patch_rgb, ref_xyz, M, lstsq_intercept)
    if chart_mcc_overlay_dir is not None:
        chart_mcc_overlay_dir.mkdir(parents=True, exist_ok=True)
        safe = path.stem.replace("/", "_")
        ochart = chart_mcc_overlay_dir / f"pid{pid}_{safe}_mcc24_indices.png"
        if not draw_mcc_patch_overlay_from_quads(
            preview_bgr, mcc_quads_xy, ochart, max_width=skin_overlay_max_width
        ):
            if debug:
                print(f"chart mcc overlay write failed {ochart}", file=sys.stderr)
    if lstsq_intercept:
        rgb_aug = np.concatenate(
            [rgb_lin, np.ones((h, w, 1), dtype=np.float64)],
            axis=2,
        )
        xyz_img = np.einsum("hwc,cd->hwd", rgb_aug, M)
    else:
        xyz_img = np.einsum("hwc,cd->hwd", rgb_lin, M)
    xyzn_scene = ref_xyz[WHITE_PATCH_INDEX].copy()
    xyzn = xyzn_scene
    if lab_cat_to_d65:
        y_white = max(float(xyzn_scene[1]), 1e-12)
        ws_norm = xyzn_scene / y_white
        xyz_img = apply_bradford_cat_hwc(xyz_img / y_white, ws_norm, D65_XYZ_Y1)
        xyzn = D65_XYZ_Y1.copy()

    rgb_u8 = cv2.cvtColor(preview_bgr, cv2.COLOR_BGR2RGB)
    rgb_u8.flags.writeable = False
    res = face_mesh.process(rgb_u8)
    if not res.multi_face_landmarks:
        return None
    lm = res.multi_face_landmarks[0].landmark
    mask, oval_pts, kept_tris, excl_dil, mesh_xy = psl.build_skin_mask_from_mesh(
        h,
        w,
        lm,
        skin_triangulation=skin_triangulation,
        exclusion_dilate_iod_fraction=skin_exclusion_dilate_iod_fraction,
    )
    if skin_overlay_dir is not None:
        skin_overlay_dir.mkdir(parents=True, exist_ok=True)
        safe = path.stem.replace("/", "_")
        ovp = skin_overlay_dir / f"pid{pid}_{safe}_skin_mesh_overlay_rawpr250.png"
        psl.write_skin_sampling_overlay_png(
            ovp,
            preview_bgr,
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
        hist_png = skin_l_histogram_dir / f"pid{pid}_{path.stem.replace('/', '_')}_skin_lab_hists_rawpr250.png"

    hist_title_bits = [f"{path.name}  (pid {pid}, PR250 matrix)"]
    if lab_cat_to_d65:
        hist_title_bits.append("Bradford CAT→D65 Lab")
    if lstsq_intercept:
        hist_title_bits.append("affine RGB→XYZ (intercept)")
    n_mesh = int(np.count_nonzero(mask > 0))
    L, a, bb, L_sem, a_sem, b_sem, C_sem, npx, nraw, lo_thr, hi_thr, l_trim_ok, chroma_ok, chroma_relaxed, ltrim_relaxed, n_xyz_inv = (
        mean_lab_masked_xyz_scene(
            xyz_img,
            mask,
            xyzn,
            l_star_trim_lo=l_star_trim_lo,
            l_star_trim_hi=l_star_trim_hi,
            skin_min_chroma_ab=skin_min_chroma_ab,
            a_star_trim_lo=a_star_trim_lo,
            a_star_trim_hi=a_star_trim_hi,
            b_star_trim_lo=b_star_trim_lo,
            b_star_trim_hi=b_star_trim_hi,
            histogram_png=hist_png,
            histogram_title=" — ".join(hist_title_bits),
        )
    )
    if chroma_relaxed:
        print(
            f"Warning pid={pid} {path.name}: dropped chroma_ab trim (min_keep); "
            f"skin_min_chroma_ab={skin_min_chroma_ab} was not applied to the mean.",
            file=sys.stderr,
        )
    if ltrim_relaxed:
        print(
            f"Warning pid={pid} {path.name}: dropped L* quantile trim (min_keep); "
            f"mean uses all {nraw} skin pixels with valid nonnegative XYZ (before any L* trim relaxation).",
            file=sys.stderr,
        )
    if n_mesh > 0 and n_xyz_inv > 0:
        pct = 100.0 * float(n_xyz_inv) / float(n_mesh)
        if pct > 15.0:
            print(
                f"Warning pid={pid} {path.name}: {n_xyz_inv}/{n_mesh} skin-mask pixels ({pct:.1f}%) "
                f"had negative fitted XYZ (excluded from Lab mean); check lighting vs PR-250 or matrix conditioning.",
                file=sys.stderr,
            )
    return RawPhotoResult(
        participant_id=pid,
        image_path=str(path),
        L_mean=L,
        a_mean=a,
        b_mean=bb,
        L_sem=L_sem,
        a_sem=a_sem,
        b_sem=b_sem,
        C_sem=C_sem,
        n_pixels=npx,
        chart_ok=True,
        n_skin_pixels_mesh_mask=n_mesh,
        n_skin_pixels_valid_xyz=nraw,
        patch_de_ab_mean=de_mean,
        l_star_trim_lo_frac=l_star_trim_lo,
        l_star_trim_hi_frac=l_star_trim_hi,
        l_star_threshold_lo=lo_thr,
        l_star_threshold_hi=hi_thr,
        skin_min_chroma_ab=skin_min_chroma_ab,
        skin_l_trim_effective=l_trim_ok,
        skin_chroma_trim_effective=chroma_ok,
        raw_wb_mode=raw_wb_mode,
        lstsq_intercept=lstsq_intercept,
        lab_cat_to_d65=lab_cat_to_d65,
        n_skin_pixels_xyz_invalid=n_xyz_inv,
        patch_lstsq_robust=patch_lstsq_robust,
    )


def main() -> None:
    root = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(
        description="Skin Lab from CR2/CR3 via PR-250 24-patch RGB→XYZ fit + face mesh (see module docstring)."
    )
    ap.add_argument(
        "--data-root",
        type=Path,
        default=_DEFAULT_PHYSIO_DATA_ROOT,
        help=(
            "Root containing **/P1/Photos/*.cr2|*.cr3. Default: this script’s built-in default "
            f"(see code: PHYSIO_DATA_ROOT env or {_DEFAULT_PHYSIO_DATA_ROOT}). "
            "If you pass --data-root \"$PHYSIO_DATA_ROOT\" and the variable is unset, the path is empty—omit "
            "--data-root or export PHYSIO_DATA_ROOT."
        ),
    )
    ap.add_argument("--out-dir", type=Path, default=root / "skin_lab_raw_pr250_output")
    ap.add_argument(
        "--pr250-json",
        type=Path,
        default=root / "lighting_output" / "lighting_reference_patches.json",
        help="PR-250 patch XYZ (JSON list or CSV with mcc_patch_index, X_mean, Y_mean, Z_mean).",
    )
    ap.add_argument("--one-image-per-participant", action="store_true")
    ap.add_argument(
        "--pick-photo",
        choices=("largest_chart", "newest_jpg", "first_sorted"),
        default="largest_chart",
    )
    ap.add_argument("--force-photo", action="append", default=[], type=psl._parse_pid_photo_override, metavar="PID=FILENAME")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument(
        "--raw-half-size",
        type=int,
        default=1,
        help="rawpy postprocess half_size (0=full resolution; 1=half; larger=faster, smaller images).",
    )
    ap.add_argument(
        "--raw-camera-wb",
        action="store_true",
        help="Use white-balance multipliers embedded in the CR2/CR3 (rawpy use_camera_wb). Default is unity WB.",
    )
    ap.add_argument(
        "--raw-chart-gray-wb",
        action="store_true",
        help=(
            "After patch extraction, apply per-channel gains so a chart reference is neutral in linear "
            "camera RGB, then refit M. Default reference is white patch (MCC index 18); see "
            "--raw-chart-gray-wb-from. Mutually exclusive with --raw-camera-wb."
        ),
    )
    ap.add_argument(
        "--raw-chart-gray-wb-from",
        choices=("white", "neutral_column_mean"),
        default="white",
        help=(
            "With --raw-chart-gray-wb: neutralize using patch 18 only, or mean RGB of gray patches "
            f"{tuple(NEUTRAL_COLUMN_WB_PATCH_INDICES)} (Neutral 8 … Neutral 3.5; excludes white 18 and black 23)."
        ),
    )
    ap.add_argument(
        "--patch-lstsq-anchor-weight",
        type=float,
        default=2.5,
        metavar="F",
        help=(
            "Multiply lstsq row weights on PR-250 skin + grayscale patches "
            f"{tuple(PR250_SKIN_NEUTRAL_ANCHOR_PATCH_INDICES)} (MCC skin rows, white, neutrals, black). "
            "Default stresses spectrometer anchors for facial colorimetry; use 1 for uniform row weights before Huber."
        ),
    )
    ap.add_argument(
        "--patch-lstsq-upweight-skin",
        type=float,
        default=1.0,
        metavar="F",
        help=(
            "Extra multiplier on MCC patches 0 and 1 only (applied after --patch-lstsq-anchor-weight on those rows). "
            "Default 1 = no extra skin emphasis beyond anchor weight."
        ),
    )
    ap.add_argument(
        "--patch-lstsq-robust",
        choices=("none", "huber"),
        default="huber",
        help=(
            "none: ordinary (weighted) least squares on all 24 patches. huber (default): iteratively reweighted "
            "Huber loss on per-patch ΔE_ab vs PR-250 — reduces leverage of poorly fit primaries (often Blue/Green) "
            "on the linear 3×3/affine map."
        ),
    )
    ap.add_argument(
        "--strict-pr250-patch-order",
        action="store_true",
        help=(
            "Exit if lighting_reference_patches patch_label or stem_canonical disagrees with mcc24_classic "
            "(default: print warnings only)."
        ),
    )
    ap.add_argument(
        "--lstsq-intercept",
        action="store_true",
        help="24×4 lstsq (affine linear RGB→XYZ with intercept column). Can lower patch_de_ab_mean if bias matters.",
    )
    ap.add_argument(
        "--lab-cat-to-d65",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Bradford CAT from the PR-250 white patch to CIE D65, then Lab with D65 white (Y=1) — same Lab "
            "illuminant as physio_skin_lab_monk / Cheng MST Table I (default: on). Full-image XYZ and the "
            "source white are divided by the measured white-patch Y first for consistent L* scaling. "
            "Use --no-lab-cat-to-d65 for scene-referred Lab (XYZn = PR-250 white patch 18 only)."
        ),
    )
    ap.add_argument(
        "--patch-rgb-agg",
        choices=("median", "mean"),
        default="median",
        help="Per-patch linear RGB statistic inside each mcc quad (median resists hotspots/border bleed).",
    )
    ap.add_argument(
        "--patch-center-fraction",
        type=float,
        default=0.7,
        metavar="F",
        help="Sample only central F of each patch quad (erode mask); 0 or >=1 disables erosion.",
    )
    ap.add_argument("--skin-exclusion-dilate-iod", type=float, default=0.12)
    ap.add_argument(
        "--skin-triangulation",
        choices=("tessellation", "oval_delaunay"),
        default="tessellation",
    )
    ap.add_argument("--skin-l-star-trim-lo", type=float, default=0.0)
    ap.add_argument("--skin-l-star-trim-hi", type=float, default=0.0)
    ap.add_argument("--skin-a-star-trim-lo", type=float, default=0.0)
    ap.add_argument("--skin-a-star-trim-hi", type=float, default=0.0)
    ap.add_argument("--skin-b-star-trim-lo", type=float, default=0.0)
    ap.add_argument("--skin-b-star-trim-hi", type=float, default=0.0)
    ap.add_argument("--skin-min-chroma-ab", type=float, default=0.0)
    ap.add_argument("--write-skin-l-histograms", action="store_true")
    ap.add_argument("--write-skin-mask-overlays", action="store_true")
    ap.add_argument(
        "--write-chart-mcc-overlays",
        action="store_true",
        help=(
            "Write out-dir/chart_mcc_overlays_raw_pr250/*_mcc24_indices.png: green quads + red 0–23 labels "
            "on the same 8-bit preview and mcc geometry used for patch ROI means (verify swatch–index alignment)."
        ),
    )
    ap.add_argument("--skin-overlay-max-width", type=int, default=1600)
    ap.add_argument("--monk-csv", type=Path, default=None)
    ap.add_argument(
        "--mst-csv",
        type=Path,
        default=root / "mst_reference_cheng2024_table1.csv",
        help="Cheng Table I Lab for ΔE₀₀ columns on the per-photo CSV.",
    )
    ap.add_argument("--plot", action="store_true", help="Bar chart L* by participant + optional Monk Spearman")
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--progress-every", type=int, default=3)
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
    if args.patch_center_fraction < 0.0:
        raise SystemExit("--patch-center-fraction must be >= 0")
    if args.raw_half_size < 0:
        raise SystemExit("--raw-half-size must be >= 0")
    if args.raw_chart_gray_wb_from != "white" and not args.raw_chart_gray_wb:
        raise SystemExit("--raw-chart-gray-wb-from requires --raw-chart-gray-wb")
    if args.patch_lstsq_upweight_skin <= 0.0:
        raise SystemExit("--patch-lstsq-upweight-skin must be > 0")
    if args.patch_lstsq_anchor_weight <= 0.0:
        raise SystemExit("--patch-lstsq-anchor-weight must be > 0")
    if args.raw_camera_wb and args.raw_chart_gray_wb:
        raise SystemExit("Use at most one of --raw-camera-wb and --raw-chart-gray-wb")
    raw_wb_mode = (
        f"chart_gray:{args.raw_chart_gray_wb_from}"
        if args.raw_chart_gray_wb
        else ("camera" if args.raw_camera_wb else "unity")
    )

    out_dir = args.out_dir.expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    # ``--data-root "$PHYSIO_DATA_ROOT"`` with an unset variable passes ``""``; Path("") resolves to cwd.
    if not str(args.data_root).strip():
        data_root = _DEFAULT_PHYSIO_DATA_ROOT.expanduser().resolve()
        print(
            f"Note: --data-root was empty (e.g. PHYSIO_DATA_ROOT unset); using {data_root}",
            file=sys.stderr,
            flush=True,
        )
    else:
        data_root = args.data_root.expanduser().resolve()
    if not data_root.is_dir():
        raise SystemExit(
            f"--data-root is not a directory: {data_root}\n"
            "Mount the drive or set the tree explicitly, e.g.\n"
            "  export PHYSIO_DATA_ROOT=/path/to/Physio-code/Data\n"
            "  python3 physio_skin_lab_raw_pr250.py …\n"
            "or:  python3 physio_skin_lab_raw_pr250.py --data-root /path/to/Physio-code/Data …"
        )

    mst_csv_path = args.mst_csv.expanduser().resolve()
    if not mst_csv_path.is_file():
        raise SystemExit(f"--mst-csv not found: {mst_csv_path}")
    mst_lab_10 = load_mst_lab_matrix_10x3(mst_csv_path)

    ref_xyz, labels, stems = load_pr250_xyz(args.pr250_json)
    validate_pr250_reference_alignment(
        args.pr250_json,
        labels,
        stems,
        strict=args.strict_pr250_patch_order,
    )

    patch_row_weights = build_patch_lstsq_row_weights(
        anchor_weight=float(args.patch_lstsq_anchor_weight),
        skin_weight=float(args.patch_lstsq_upweight_skin),
    )

    hist_dir: Optional[Path] = None
    if args.write_skin_l_histograms:
        if plt is None:
            print("Warning: matplotlib missing; skipping histograms", file=sys.stderr)
        else:
            hist_dir = out_dir / "skin_L_histograms_raw_pr250"
    overlay_dir: Optional[Path] = None
    if args.write_skin_mask_overlays:
        overlay_dir = out_dir / "skin_mask_overlays_raw_pr250"
    chart_mcc_dir: Optional[Path] = None
    if args.write_chart_mcc_overlays:
        chart_mcc_dir = out_dir / "chart_mcc_overlays_raw_pr250"

    photos = find_calibration_raw_paths(data_root)
    if not photos:
        hits = sorted({p for p in data_root.glob("**/P1/Photos") if p.is_dir()})
        if not hits:
            raise SystemExit(
                f"No **/P1/Photos directory under:\n  {data_root}\n"
                "Expected …/<participant_id>/P1/Photos/*.cr2 (any depth under --data-root).\n"
                "Fix: mount the dataset drive and pass the real tree, e.g.\n"
                "  export PHYSIO_DATA_ROOT=/media/…/Physio-code/Data\n"
                "  python3 physio_skin_lab_raw_pr250.py …\n"
                "or omit --data-root so the default above is used (see stderr if you passed an empty "
                "\"$PHYSIO_DATA_ROOT\")."
            )
        raise SystemExit(
            f"No .cr2/.cr3 under {data_root} (found {len(hits)} P1/Photos dir(s), e.g. {hits[0]!s}). "
            "Add Canon RAW files there or fix --data-root."
        )
    if args.one_image_per_participant:
        photos = pick_one_raw_per_participant(
            photos,
            args.pick_photo,
            raw_half_size=args.raw_half_size,
            use_camera_wb=args.raw_camera_wb,
            debug=args.debug,
        )
        print(f"One RAW per participant ({args.pick_photo}): {len(photos)} files", flush=True)
    if args.force_photo:
        photos = psl.apply_forced_photos(photos, dict(args.force_photo), data_root, debug=args.debug)
    if args.limit > 0:
        photos = photos[: args.limit]

    mp_fm = mp.solutions.face_mesh
    rows: List[RawPhotoResult] = []
    stderr_cm = _silence_stderr() if not args.debug else contextlib.nullcontext()
    with stderr_cm:
        # static_image_mode=True: min_tracking_confidence is ignored by MediaPipe.
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
                r = process_one_raw(
                    p,
                    face_mesh,
                    ref_xyz,
                    raw_half_size=args.raw_half_size,
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
                    chart_mcc_overlay_dir=chart_mcc_dir,
                    raw_camera_wb=args.raw_camera_wb,
                    raw_chart_gray_wb=args.raw_chart_gray_wb,
                    chart_gray_wb_from=args.raw_chart_gray_wb_from,
                    raw_wb_mode=raw_wb_mode,
                    lstsq_intercept=args.lstsq_intercept,
                    lab_cat_to_d65=args.lab_cat_to_d65,
                    patch_rgb_use_median=(args.patch_rgb_agg == "median"),
                    patch_center_fraction=args.patch_center_fraction,
                    patch_row_weights=patch_row_weights,
                    patch_lstsq_robust=args.patch_lstsq_robust,
                    debug=args.debug,
                )
                if r:
                    rows.append(r)

    csv_path = out_dir / "skin_lab_per_photo_raw_pr250.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "participant_id",
                "image_path",
                "L_mean",
                "a_mean",
                "b_mean",
                "L_sem",
                "a_sem",
                "b_sem",
                "C_sem",
                "n_skin_pixels_used",
                "n_skin_pixels_mesh_mask",
                "n_skin_pixels_valid_xyz",
                "n_skin_pixels_xyz_invalid",
                "patch_de_ab_mean",
                "l_star_trim_lo_frac",
                "l_star_trim_hi_frac",
                "l_star_threshold_lo",
                "l_star_threshold_hi",
                "skin_min_chroma_ab",
                "skin_l_trim_effective",
                "skin_chroma_trim_effective",
                "chart_ok",
                "raw_wb_mode",
                "lstsq_intercept",
                "lab_cat_to_d65",
                "patch_lstsq_robust",
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
                    r.L_sem,
                    r.a_sem,
                    r.b_sem,
                    r.C_sem,
                    r.n_pixels,
                    r.n_skin_pixels_mesh_mask,
                    r.n_skin_pixels_valid_xyz,
                    r.n_skin_pixels_xyz_invalid,
                    r.patch_de_ab_mean,
                    r.l_star_trim_lo_frac,
                    r.l_star_trim_hi_frac,
                    "" if not np.isfinite(r.l_star_threshold_lo) else r.l_star_threshold_lo,
                    "" if not np.isfinite(r.l_star_threshold_hi) else r.l_star_threshold_hi,
                    r.skin_min_chroma_ab,
                    int(r.skin_l_trim_effective),
                    int(r.skin_chroma_trim_effective),
                    int(r.chart_ok),
                    r.raw_wb_mode,
                    int(r.lstsq_intercept),
                    int(r.lab_cat_to_d65),
                    r.patch_lstsq_robust,
                ]
                + de_cols
            )

    by_pid: Dict[str, List[RawPhotoResult]] = {}
    for r in rows:
        by_pid.setdefault(r.participant_id, []).append(r)

    agg_path = out_dir / "skin_lab_per_participant_raw_pr250.json"
    doc: Dict[str, Any] = {
        "_pipeline": {
            "source": "physio_skin_lab_raw_pr250.py",
            "pr250_reference": str(args.pr250_json.resolve()),
            "mst_reference_csv": str(mst_csv_path),
            "delta_e_2000": "Per-photo and per-participant (from mean Lab) vs Table I in --mst-csv (CIEDE2000).",
            "raw_half_size": args.raw_half_size,
            "raw_wb_mode": raw_wb_mode,
            "raw_chart_gray_wb_from": args.raw_chart_gray_wb_from if args.raw_chart_gray_wb else None,
            "patch_lstsq_upweight_skin": float(args.patch_lstsq_upweight_skin),
            "patch_lstsq_anchor_weight": float(args.patch_lstsq_anchor_weight),
            "patch_lstsq_robust": str(args.patch_lstsq_robust),
            "strict_pr250_patch_order": bool(args.strict_pr250_patch_order),
            "lstsq_intercept": bool(args.lstsq_intercept),
            "lab_cat_to_d65": bool(args.lab_cat_to_d65),
            "patch_fit_metric": (
                "patch_de_ab_mean is mean ΔE_ab (Euclidean in Lab, XYZn = PR-250 white patch 18) between "
                "predicted and reference patch XYZ — not ΔE₀₀. Always scene-referred (CAT is not applied to "
                "patch residuals; metric matches the lstsq fit to PR-250 XYZ). "
                "With patch_lstsq_robust huber, the matrix is fit by IRLS Huber weights on those residuals; "
                "patch_de_ab_mean remains the simple mean of all 24 per-patch ΔE_ab (diagnostic)."
            ),
            "demosaic_note": (
                "rawpy output_bps=16 yields linear camera RGB in raw count space (typ. 0–65535 as float64), "
                "not sRGB [0,1]; do not pass to skimage.color.rgb2lab without a proper transform."
            ),
            "mst_comparison_note": (
                "physio_skin_lab_monk skin Lab uses D65 (skimage.color.rgb2lab); Cheng Table I is D65-style. "
                "By default this script uses Bradford CAT→D65 for skin Lab (same white point). "
                "Use --no-lab-cat-to-d65 only if you want scene-referred Lab (XYZn = PR-250 white patch 18)."
            ),
            "patch_rgb_agg": args.patch_rgb_agg,
            "patch_center_fraction": args.patch_center_fraction,
            "skin_xyz_exclusion": (
                "Skin mean Lab uses only pixels with fitted XYZ ≥ 0 (no global XYZ clip). "
                "n_skin_pixels_mesh_mask = face mesh skin mask count; n_skin_pixels_valid_xyz = mesh ∩ nonnegative XYZ "
                "(before L* / chroma trim); n_skin_pixels_xyz_invalid = mesh pixels dropped (negative XYZ). "
                "With Bradford CAT→D65 (default), negatives may come from matrix extrapolation or from the CAT; "
                "both are excluded and counted in n_skin_pixels_xyz_invalid."
            ),
            "skin_mask_morphology": (
                "physio_skin_lab_monk.build_skin_mask_from_mesh: IOD-scaled morphological open then close, "
                "then re-intersect ~excl_dil so closing cannot enter lip/eye exclusion."
            ),
            "characterization": (
                "24-patch linear camera RGB (mcc quad ROI on rawpy linear demosaic, ColorSpace.raw) "
                f"with raw_wb_mode={raw_wb_mode!r} → "
                + (
                    (
                        "Huber IRLS (per-patch ΔE_ab) on weighted lstsq to PR-250 XYZ"
                        if patch_row_weights is not None
                        else "Huber IRLS (per-patch ΔE_ab) on lstsq to PR-250 XYZ"
                    )
                    if args.patch_lstsq_robust == "huber"
                    else (
                        "weighted lstsq to PR-250 XYZ"
                        if patch_row_weights is not None
                        else "lstsq to PR-250 XYZ"
                    )
                )
            )
            + (
                f"({' (24×4 affine RGB→XYZ)' if args.lstsq_intercept else ' (24×3 linear RGB→XYZ)'}); "
                "full image XYZ = "
                + ("[R,G,B,1] @ M" if args.lstsq_intercept else "RGB @ M")
                + "; skin Lab from nonnegative-XYZ pixels only (invalid extrapolation excluded)."
                + (
                    " Bradford CAT to D65: divide full-image XYZ by PR-250 white-patch Y, CAT, Lab with D65 (Y=1)."
                    if args.lab_cat_to_d65
                    else " Skin Lab: scene-referred (XYZn = PR-250 white patch 18; no Bradford CAT)."
                )
                + (
                    f" Row weights: patches {tuple(PR250_SKIN_NEUTRAL_ANCHOR_PATCH_INDICES)} ×{float(args.patch_lstsq_anchor_weight):g}; "
                    f"patches 0–1 additionally ×{float(args.patch_lstsq_upweight_skin):g}."
                    if patch_row_weights is not None
                    else ""
                )
            ),
            "lab_white_point": (
                "Default: Lab XYZn = D65 (Y=1) after Y-normalized Bradford CAT from PR-250 white. "
                "With --no-lab-cat-to-d65: XYZn = PR-250 white patch 18 (scene-referred)."
            ),
            "skin_lab_sem": (
                "L_sem, a_sem, b_sem, C_sem are standard errors of the mean over skin-mask pixels "
                "(same mask/trim as L_mean); C* = hypot(a*, b*) per pixel. For n_photos>1, JSON "
                "participant sem is between-photo SE of per-photo means when multiple photos exist."
            ),
            "skin_mask": f"Same as physio_skin_lab_monk ({args.skin_triangulation}).",
            "face_preview": "MediaPipe runs on 8-bit preview from linear RAW (geometry only).",
            "chart_mcc_overlay": (
                "Optional --write-chart-mcc-overlays: same mcc quads as patch_linear_rgb_24 ROI sampling; "
                "indices 0–17 color rows; 18=white, 19–22 neutrals, 23=black (X-Rite bottom row per mcc24_classic)."
                if args.write_chart_mcc_overlays
                else None
            ),
        }
    }
    for pid, lst in by_pid.items():
        labs = [
            (x.L_mean, x.a_mean, x.b_mean)
            for x in lst
            if np.isfinite(x.L_mean) and np.isfinite(x.a_mean) and np.isfinite(x.b_mean)
        ]

        def _participant_sem(field_mean: str, field_sem: str) -> float:
            means = [
                float(getattr(x, field_mean))
                for x in lst
                if np.isfinite(getattr(x, field_mean))
            ]
            sems = [
                float(getattr(x, field_sem))
                for x in lst
                if np.isfinite(getattr(x, field_sem))
            ]
            if not means:
                return float("nan")
            if len(means) == 1:
                return sems[0] if sems else float("nan")
            return float(np.std(means, ddof=1) / np.sqrt(len(means)))

        def _participant_c_sem() -> float:
            c_means = [
                float(np.hypot(x.a_mean, x.b_mean))
                for x in lst
                if np.isfinite(x.a_mean) and np.isfinite(x.b_mean)
            ]
            c_sems = [float(x.C_sem) for x in lst if np.isfinite(x.C_sem)]
            if not c_means:
                return float("nan")
            if len(c_means) == 1:
                return c_sems[0] if c_sems else float("nan")
            return float(np.std(c_means, ddof=1) / np.sqrt(len(c_means)))

        entry: Dict[str, Any] = {
            "n_photos": len(lst),
            "L_mean": float(np.mean([t[0] for t in labs])) if labs else float("nan"),
            "a_mean": float(np.mean([t[1] for t in labs])) if labs else float("nan"),
            "b_mean": float(np.mean([t[2] for t in labs])) if labs else float("nan"),
            "L_sem": _participant_sem("L_mean", "L_sem"),
            "a_sem": _participant_sem("a_mean", "a_sem"),
            "b_sem": _participant_sem("b_mean", "b_sem"),
            "C_sem": _participant_c_sem(),
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
    if hist_dir:
        print(f"L* histograms: {hist_dir} ({len(rows)} PNGs)")
    if overlay_dir:
        print(f"Skin overlays: {overlay_dir} ({len(rows)} PNGs)")
    if chart_mcc_dir and chart_mcc_dir.is_dir():
        n_chart_png = len(list(chart_mcc_dir.glob("*.png")))
        print(f"Chart MCC index overlays: {chart_mcc_dir} ({n_chart_png} PNGs)")

    monk_map = psl.load_monk_csv(args.monk_csv) if args.monk_csv and args.monk_csv.is_file() else {}
    if args.plot and plt is not None:
        pids = sorted((k for k in doc if not str(k).startswith("_")), key=lambda p: doc[p]["L_mean"])
        Ls = [doc[p]["L_mean"] for p in pids]
        plt.figure(figsize=(12, 4))
        plt.bar(range(len(pids)), Ls, color="steelblue", edgecolor="k", alpha=0.85)
        plt.xticks(range(len(pids)), pids, rotation=45, ha="right")
        plt.ylabel("Mean L* (PR250 matrix, scene Lab)")
        plt.title("Mean skin L* by participant (RAW + 24-patch XYZ fit)")
        plt.tight_layout()
        plt.savefig(out_dir / "skin_L_by_participant_raw_pr250.png", dpi=160)
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
                plt.scatter(xs, ys, c="navy", edgecolors="k", alpha=0.8)
                plt.xlabel("Monk Skin Tone (1–10)")
                plt.ylabel("Mean facial skin L* (RAW PR250)")
                plt.title(f"Spearman ρ = {rho:.3f}, p = {pval:.4g}")
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(out_dir / "skin_L_vs_monk_raw_pr250.png", dpi=160)
                plt.close()
                print(f"Spearman(L*, Monk) = {rho:.4f}, p = {pval:.4g}")

    print("Done.")


if __name__ == "__main__":
    main()
