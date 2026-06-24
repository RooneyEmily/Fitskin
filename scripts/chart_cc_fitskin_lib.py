"""
In-scene ColorChecker → facial Lab for FitSkin validation (Wakholi-style CC, MABL wiring).

Reference: canonical Classic 24 D65 (``mcc24_canonical_d65``), **not** physio PR-250 spectrometer.
Stages per image: chart detect → white-patch WB → linear RGB → 3×3 (optional Huber) → XYZ → D65 Lab
→ MediaPipe skin mask (full mesh or cheek ROI).
"""
from __future__ import annotations

import contextlib
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import mediapipe as mp
import physio_skin_lab_monk as psl
import physio_skin_lab_raw_pr250 as pr250
import validate_chart_rgb_vs_lighting_xyz as vchart
from mcc24_canonical_d65 import (
    CANONICAL_WHITE_XYZ_D65,
    load_canonical_lab_d65,
    load_canonical_xyz_d65,
)
from mcc24_classic import WHITE_PATCH_INDEX

# Cheek landmark indices (MediaPipe face mesh) for FitSkin-aligned sampling
CHEEK_LANDMARKS = (
    50, 101, 36, 205, 206, 207, 187, 123, 116, 117, 118, 119, 120, 121, 128, 245, 193, 194, 188,
    174, 196, 197, 177, 137, 147,
)
CHEEK_R_LANDMARKS = (
    280, 330, 266, 425, 426, 427, 411, 352, 345, 346, 347, 348, 349, 350, 357, 465, 416, 415, 404,
    399, 421, 419, 401, 366, 376,
)
D65_XYZN = pr250.D65_XYZ_Y1.copy()


@contextlib.contextmanager
def silence_stderr():
    stderr_fd = sys.stderr.fileno()
    saved = os.dup(stderr_fd)
    devnull = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull, stderr_fd)
        yield
    finally:
        os.dup2(saved, stderr_fd)
        os.close(saved)
        os.close(devnull)


def cheek_mask_from_landmarks(h: int, w: int, lm: Any, mesh_mask: np.ndarray) -> np.ndarray:
    ids = list(set(CHEEK_LANDMARKS + CHEEK_R_LANDMARKS))
    pts = np.array([[int(lm[i].x * w), int(lm[i].y * h)] for i in ids], dtype=np.int32)
    m = np.zeros((h, w), dtype=np.uint8)
    if pts.shape[0] >= 3:
        hull = cv2.convexHull(pts)
        cv2.fillConvexPoly(m, hull, 255)
    return cv2.bitwise_and(m, mesh_mask)


def chart_detect_and_wb(bgr: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], float, str]:
    """Returns ``(wb_bgr, patch_rgb_255, chart_area_fraction, status)``."""
    ev = vchart.evaluate_bgr_vs_ref(bgr, load_canonical_xyz_d65(), [""] * 24)
    if ev is None:
        return None, None, 0.0, "no_chart"
    return (
        ev["wb_bgr"],
        np.asarray(ev["patch_rgb"], dtype=np.float64),
        float(ev.get("chart_area_fraction", 0.0)),
        "ok",
    )


def fit_cc_matrix(
    patch_rgb_255: np.ndarray,
    *,
    huber: bool = False,
    upweight_skin_neutral: bool = True,
    affine: bool = False,
) -> Tuple[np.ndarray, float, np.ndarray]:
    """
    3×3 (or 3×4 affine) linear RGB → canonical D65 XYZ (Y~0–100).
    Returns ``(M, mean_patch_deltaE_ab, pred_xyz)``.

    Default is plain weighted least squares (Huber off) — best median ΔE₀₀ on the
    bundled JPEG cohort vs Huber IRWS.
    """
    ref_xyz = load_canonical_xyz_d65()
    lin = vchart.srgb_255_to_linear(patch_rgb_255)
    row_w = None
    if upweight_skin_neutral:
        row_w = pr250.build_patch_lstsq_row_weights(anchor_weight=2.5, skin_weight=1.0)
    if huber:
        M, _, _ = pr250.fit_rgb_to_xyz_lstsq_huber_irls(
            lin, ref_xyz, with_intercept=affine, row_weights=row_w
        )
    else:
        M = pr250.fit_rgb_to_xyz_lstsq(
            lin, ref_xyz, with_intercept=affine, row_weights=row_w
        )
    if affine:
        aug = np.column_stack([lin, np.ones((lin.shape[0], 1), dtype=np.float64)])
        pred = aug @ M
    else:
        pred = lin @ M
    de = pr250.per_patch_delta_e_ab(lin, ref_xyz, M, affine)
    return M, float(np.mean(de)), pred


def apply_cc_to_bgr(wb_bgr: np.ndarray, M: np.ndarray) -> np.ndarray:
    """WB BGR uint8 → D65 XYZ image (H,W,3) via chart 3×3 / 3×4 matrix + legacy Bradford normalize."""
    rgb255 = cv2.cvtColor(wb_bgr, cv2.COLOR_BGR2RGB).astype(np.float64)
    lin = vchart.srgb_255_to_linear(rgb255)
    if M.shape[0] == 4:
        flat = np.column_stack([lin.reshape(-1, 3), np.ones((lin.shape[0] * lin.shape[1], 1))])
        xyz = (flat @ M).reshape(lin.shape[0], lin.shape[1], 3)
    else:
        xyz = lin @ M
    y_white = max(float(CANONICAL_WHITE_XYZ_D65[1]), 1e-12)
    ws = CANONICAL_WHITE_XYZ_D65 / y_white
    return pr250.apply_bradford_cat_hwc(xyz / y_white, ws, D65_XYZN)


def mean_skin_lab_xyz(
    xyz_img: np.ndarray,
    mask: np.ndarray,
    *,
    l_trim: float,
    min_chroma: float,
) -> Dict[str, float]:
    out = pr250.mean_lab_masked_xyz_scene(
        xyz_img,
        mask,
        D65_XYZN,
        l_star_trim_lo=l_trim,
        l_star_trim_hi=l_trim,
        skin_min_chroma_ab=min_chroma,
        histogram_png=None,
        histogram_title="",
    )
    L, a, b = float(out[0]), float(out[1]), float(out[2])
    npx = int(out[7])
    return {
        "L": L,
        "a": a,
        "b": b,
        "C": float(np.hypot(a, b)) if npx else float("nan"),
        "n_pixels": npx,
    }


def naive_wb_lab(
    wb_bgr: np.ndarray,
    face_mesh: Any,
    *,
    l_trim: float,
    min_chroma: float,
    mask: np.ndarray,
) -> Dict[str, float]:
    L, a, b, npx, *_ = psl.mean_lab_masked(
        wb_bgr,
        mask,
        l_star_trim_lo=l_trim,
        l_star_trim_hi=l_trim,
        min_chroma_ab=min_chroma,
    )
    return {"L": L, "a": a, "b": b, "C": float(np.hypot(a, b)) if npx else float("nan"), "n_pixels": int(npx)}


def process_one_image(
    bgr: np.ndarray,
    face_mesh: Any,
    *,
    l_trim: float = 0.05,
    min_chroma: float = 2.0,
    roi: str = "mesh",
    huber: bool = False,
    affine: bool = False,
    write_overlay: Optional[Path] = None,
    write_lab_histogram: Optional[Path] = None,
    write_ab_histogram_stem: Optional[Path] = None,
    histogram_frame_label: str = "",
) -> Dict[str, Any]:
    """
    Full per-image pipeline. ``roi``: ``mesh`` | ``cheek`` | ``both``.
    """
    wb, patches, frac, status = chart_detect_and_wb(bgr)
    if wb is None:
        return {"chart_ok": False, "status": status}

    M, patch_de_mean, pred_xyz = fit_cc_matrix(patches, huber=huber, affine=affine)
    xyz_img = apply_cc_to_bgr(wb, M)

    rgb = cv2.cvtColor(wb, cv2.COLOR_BGR2RGB)
    rgb.flags.writeable = False
    res = face_mesh.process(rgb)
    if not res.multi_face_landmarks:
        return {"chart_ok": True, "status": "no_face", "chart_area_fraction": frac, "patch_de_ab_mean": patch_de_mean}

    h, w = wb.shape[:2]
    lm = res.multi_face_landmarks[0].landmark
    mesh_mask, oval, kept, excl, mesh_xy = psl.build_skin_mask_from_mesh(
        h, w, lm, skin_triangulation="tessellation", exclusion_dilate_iod_fraction=0.12
    )
    cheek = cheek_mask_from_landmarks(h, w, lm, mesh_mask)

    ref_xyz = load_canonical_xyz_d65()
    ref_lab = load_canonical_lab_d65()
    lin_p = vchart.srgb_255_to_linear(patches)
    patch_lab_fit = np.array(
        [pr250.xyz_to_lab(pred_xyz[i], pred_xyz[WHITE_PATCH_INDEX]) for i in range(24)]
    )

    rois: Dict[str, np.ndarray] = {"mesh": mesh_mask}
    if roi in ("cheek", "both"):
        rois["cheek"] = cheek

    labs: Dict[str, Dict[str, float]] = {}
    for name, m in rois.items():
        labs[name] = mean_skin_lab_xyz(xyz_img, m, l_trim=l_trim, min_chroma=min_chroma)
        labs[f"{name}_wb_only"] = naive_wb_lab(wb, face_mesh, l_trim=l_trim, min_chroma=min_chroma, mask=m)

    primary = labs["cheek"] if roi in ("cheek", "both") else labs["mesh"]

    if write_overlay is not None:
        write_overlay.parent.mkdir(parents=True, exist_ok=True)
        preview = wb.copy()
        psl.write_skin_sampling_overlay_png(
            write_overlay,
            preview,
            oval,
            kept,
            mesh_mask,
            excl,
            mesh_xy=mesh_xy,
            max_width=1600,
        )

    hist_mask = cheek if roi in ("cheek", "both") else mesh_mask
    pix = None
    if hist_mask is not None and (write_lab_histogram is not None or write_ab_histogram_stem is not None):
        pix = skin_lab_pixels_for_mask(xyz_img, hist_mask, l_trim=l_trim, min_chroma=min_chroma)
    if pix is not None and write_lab_histogram is not None:
        stem = write_lab_histogram.stem.replace("_skin_lab_hists", "")
        label = f"{histogram_frame_label} — {stem}" if histogram_frame_label else stem
        write_skin_lab_histogram_panel(
            write_lab_histogram,
            pix["L"],
            pix["a"],
            pix["b"],
            pix["sel"],
            lo_thr=pix["lo_thr"],
            hi_thr=pix["hi_thr"],
            min_chroma_ab=min_chroma,
            l_trim_relaxed=pix["l_trim_relaxed"],
            chroma_relaxed=pix["chroma_relaxed"],
            title=label,
        )
    if pix is not None and write_ab_histogram_stem is not None:
        stem = write_ab_histogram_stem.name
        label = f"{histogram_frame_label} — {stem}" if histogram_frame_label else stem
        write_skin_ab_marginal_histograms(
            write_ab_histogram_stem,
            pix["L"],
            pix["a"],
            pix["b"],
            pix["sel"],
            lo_thr=pix["lo_thr"],
            hi_thr=pix["hi_thr"],
            min_chroma_ab=min_chroma,
            l_trim_relaxed=pix["l_trim_relaxed"],
            chroma_relaxed=pix["chroma_relaxed"],
            title=f"a* / b* binning — {label}",
        )

    return {
        "chart_ok": True,
        "status": "ok",
        "chart_area_fraction": frac,
        "patch_de_ab_mean": patch_de_mean,
        "patch_de_ab_skin01": float(
            (
                pr250.delta_e_ab(patch_lab_fit[0], ref_lab[0])
                + pr250.delta_e_ab(patch_lab_fit[1], ref_lab[1])
            )
            / 2.0
        ),
        "pipeline_L": primary["L"],
        "pipeline_a": primary["a"],
        "pipeline_b": primary["b"],
        "pipeline_C": primary["C"],
        "pipeline_n_pixels": primary["n_pixels"],
        "mesh_L": labs["mesh"]["L"],
        "mesh_a": labs["mesh"]["a"],
        "mesh_b": labs["mesh"]["b"],
        "cheek_L": labs.get("cheek", labs["mesh"])["L"],
        "cheek_a": labs.get("cheek", labs["mesh"])["a"],
        "cheek_b": labs.get("cheek", labs["mesh"])["b"],
        "wb_only_mesh_a": labs["mesh_wb_only"]["a"],
        "wb_only_mesh_L": labs["mesh_wb_only"]["L"],
        "wb_only_cheek_a": labs.get("cheek_wb_only", labs["mesh_wb_only"])["a"],
        "wb_only_cheek_L": labs.get("cheek_wb_only", labs["mesh_wb_only"])["L"],
    }


def plt_available() -> bool:
    try:
        import matplotlib.pyplot as _plt  # noqa: F401

        return True
    except ImportError:
        return False


def skin_lab_trim_selection(
    L: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    *,
    l_star_trim_lo: float,
    l_star_trim_hi: float,
    skin_min_chroma_ab: float,
) -> Tuple[np.ndarray, float, float, bool, bool]:
    """
    Same gates as ``mean_lab_masked`` / ``mean_lab_masked_xyz_scene`` (L* quantiles, then C*_ab).

    Returns ``(sel, lo_thr, hi_thr, l_trim_relaxed, chroma_relaxed)``.
    """
    n_raw = int(L.size)
    lo_thr = float("nan")
    hi_thr = float("nan")
    sel_L = np.ones(n_raw, dtype=bool)
    tlo = float(l_star_trim_lo)
    thi = float(l_star_trim_hi)
    if tlo > 0.0:
        tlo = min(tlo, 0.45)
        lo_thr = float(np.quantile(L, tlo))
        sel_L &= L >= lo_thr
    if thi > 0.0:
        thi = min(thi, 0.45)
        hi_thr = float(np.quantile(L, 1.0 - thi))
        sel_L &= L <= hi_thr
    sel = sel_L.copy()
    if skin_min_chroma_ab > 0.0:
        sel &= np.hypot(a, b) >= float(skin_min_chroma_ab)
    n_kept = int(np.count_nonzero(sel))
    min_keep = max(2000, n_raw // 25)
    chroma_relaxed = False
    ltrim_relaxed = False
    if n_kept < min_keep and skin_min_chroma_ab > 0.0:
        chroma_relaxed = True
        sel = sel_L.copy()
        n_kept = int(np.count_nonzero(sel))
    if n_kept < min_keep:
        ltrim_relaxed = True
        sel = np.ones(n_raw, dtype=bool)
        lo_thr = float("nan")
        hi_thr = float("nan")
    return sel, lo_thr, hi_thr, ltrim_relaxed, chroma_relaxed


def _hist_bins(n: int) -> int:
    return int(np.clip(n // 400, 24, 100))


def write_skin_lab_histogram_panel(
    path: Path,
    L: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    sel: np.ndarray,
    *,
    lo_thr: float,
    hi_thr: float,
    min_chroma_ab: float,
    l_trim_relaxed: bool,
    chroma_relaxed: bool,
    title: str = "",
    panel_title: str = "Skin Pixel Gating",
) -> None:
    """L*, C*ab, a*, b* histograms; tan = kept for mean, gray = dropped by L* / chroma gates."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    path.parent.mkdir(parents=True, exist_ok=True)
    C = np.hypot(a, b)
    C_keep = float(min_chroma_ab) if min_chroma_ab > 0 else float("nan")
    nb = _hist_bins(int(L.size))
    fig, axes = plt.subplots(2, 2, figsize=(8.6, 6.4))
    fig.subplots_adjust(left=0.17, right=0.97, top=0.78, bottom=0.13, hspace=0.36, wspace=0.30)
    dropped = ~sel
    legend_handles: List[Line2D] = []

    def _hist_pair(ax, values: np.ndarray) -> None:
        nonlocal legend_handles
        if np.any(sel):
            ax.hist(
                values[sel],
                bins=nb,
                color="#c4a574",
                edgecolor="0.35",
                linewidth=0.35,
                alpha=0.9,
            )
        if np.any(dropped):
            ax.hist(
                values[dropped],
                bins=nb,
                color="0.82",
                edgecolor="none",
                alpha=0.5,
            )
        if not legend_handles and (np.any(sel) or np.any(dropped)):
            legend_handles = [
                Line2D([0], [0], color="#c4a574", lw=8, label="Retained for mean"),
                Line2D([0], [0], color="0.82", lw=8, label="Excluded"),
            ]

    def _panel(
        ax,
        values: np.ndarray,
        xlabel: str,
        vlines: List[Tuple[float, str]],
    ) -> None:
        _hist_pair(ax, values)
        for x, color in vlines:
            if np.isfinite(x):
                ax.axvline(x, color=color, ls="--", lw=1.15, alpha=0.85)
        ax.set_xlabel(xlabel, fontsize=10)
        ax.tick_params(labelsize=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    l_lines: List[Tuple[float, str]] = []
    if not l_trim_relaxed:
        if np.isfinite(lo_thr):
            l_lines.append((lo_thr, "tab:blue"))
        if np.isfinite(hi_thr):
            l_lines.append((hi_thr, "tab:red"))
    _panel(axes[0, 0], L, r"$L^*$", l_lines)

    c_lines: List[Tuple[float, str]] = []
    if not chroma_relaxed and np.isfinite(C_keep):
        c_lines.append((C_keep, "tab:purple"))
    _panel(axes[0, 1], C, r"$C^*_{ab}$", c_lines)
    _panel(axes[1, 0], a, r"$a^*$", [])
    _panel(axes[1, 1], b, r"$b^*$", [])

    for ax in axes[:, 0]:
        ax.set_ylabel("")
    fig.text(
        0.062,
        0.455,
        "Pixel count",
        rotation=90,
        va="center",
        ha="center",
        fontsize=9,
    )

    n_kept = int(np.count_nonzero(sel))
    prefix = f"{title} — " if title else ""
    foot = f"{prefix}{n_kept:,} / {L.size:,} mask pixels retained for mean"
    if l_trim_relaxed or chroma_relaxed:
        parts = []
        if l_trim_relaxed:
            parts.append("L* trim relaxed")
        if chroma_relaxed:
            parts.append("chroma gate relaxed")
        foot += f" ({'; '.join(parts)})"

    if panel_title:
        fig.suptitle(panel_title, fontsize=11, fontweight="medium", y=0.92)
    fig.text(0.5, 0.05, foot, ha="center", va="bottom", fontsize=8, color="0.45")
    if legend_handles:
        fig.legend(
            handles=legend_handles,
            loc="center",
            bbox_to_anchor=(0.5, 0.798),
            ncol=2,
            frameon=False,
            fontsize=8,
        )
    fig.savefig(path, dpi=120)
    plt.close(fig)


def skin_lab_pixels_for_mask(
    xyz_img: np.ndarray,
    mask: np.ndarray,
    *,
    l_trim: float,
    min_chroma: float,
) -> Optional[Dict[str, Any]]:
    """Masked cheek/mesh pixels in D65 Lab plus trim selection (for marginal histograms)."""
    m = mask > 0
    xyz_valid = np.all(xyz_img >= 0.0, axis=2)
    m &= xyz_valid
    if not np.any(m):
        return None
    L, a, b = pr250.xyz_to_lab_batch(xyz_img[m].reshape(-1, 3), D65_XYZN)
    sel, lo, hi, l_rel, c_rel = skin_lab_trim_selection(
        L, a, b, l_star_trim_lo=l_trim, l_star_trim_hi=l_trim, skin_min_chroma_ab=min_chroma
    )
    return {
        "L": L,
        "a": a,
        "b": b,
        "sel": sel,
        "lo_thr": lo,
        "hi_thr": hi,
        "l_trim_relaxed": l_rel,
        "chroma_relaxed": c_rel,
    }


def write_skin_ab_marginal_histograms(
    path_stem: Path,
    L: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    sel: np.ndarray,
    *,
    lo_thr: float,
    hi_thr: float,
    min_chroma_ab: float,
    l_trim_relaxed: bool,
    chroma_relaxed: bool,
    title: str,
) -> Tuple[Path, Path]:
    """Dedicated a* and b* binning panels (1×2), same kept/dropped styling as the 2×2 panel."""
    import matplotlib.pyplot as plt

    path_stem = Path(path_stem)
    path_stem.parent.mkdir(parents=True, exist_ok=True)
    path_a = path_stem.parent / f"{path_stem.name}_a_hist.png"
    path_b = path_stem.parent / f"{path_stem.name}_b_hist.png"
    dropped = ~sel
    nb = _hist_bins(int(L.size))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)

    def _panel(ax, values: np.ndarray, xlabel: str) -> None:
        if np.any(sel):
            ax.hist(values[sel], bins=nb, color="tan", edgecolor="k", alpha=0.85, label="kept for mean")
        if np.any(dropped):
            ax.hist(
                values[dropped],
                bins=nb,
                color="0.75",
                edgecolor="none",
                alpha=0.45,
                label="dropped (L* trim / low C*)",
            )
        ax.set_xlabel(xlabel)
        ax.set_ylabel("count")
        ax.legend(fontsize=8, loc="upper right")

    _panel(axes[0], a, "a* (D65, chart CC skin mask)")
    _panel(axes[1], b, "b* (D65, chart CC skin mask)")
    note = f"n={int(np.count_nonzero(sel))}/{L.size} pixels in mean"
    if l_trim_relaxed:
        note += "; L* trim relaxed"
    if chroma_relaxed:
        note += "; chroma trim relaxed"
    if not l_trim_relaxed and np.isfinite(lo_thr):
        note += f"; L* ∈ [{lo_thr:.1f}, {hi_thr:.1f}]" if np.isfinite(hi_thr) else f"; L* ≥ {lo_thr:.1f}"
    if not chroma_relaxed and min_chroma_ab > 0:
        note += f"; C* ≥ {min_chroma_ab:.1f}"
    path_combined = path_stem.parent / f"{path_stem.name}_ab_hists.png"
    fig.suptitle(f"{title}\n{note}", fontsize=10)
    fig.savefig(path_combined, dpi=120)
    plt.close(fig)
    for path, vals, xlab in ((path_a, a, "a*"), (path_b, b, "b*")):
        fig1, ax1 = plt.subplots(figsize=(6.5, 4))
        if np.any(sel):
            ax1.hist(vals[sel], bins=nb, color="tan", edgecolor="k", alpha=0.85, label="kept")
        if np.any(dropped):
            ax1.hist(vals[dropped], bins=nb, color="0.75", edgecolor="none", alpha=0.45, label="dropped")
        ax1.set_xlabel(f"{xlab} (D65, chart CC)")
        ax1.set_ylabel("count")
        ax1.set_title(f"{title} — {xlab}\n{note}", fontsize=9)
        ax1.legend(fontsize=8)
        fig1.tight_layout()
        fig1.savefig(path, dpi=120)
        plt.close(fig1)
    return path_a, path_b


def _maybe_write_skin_lab_histograms(
    path: Path,
    xyz_img: np.ndarray,
    mask: np.ndarray,
    *,
    l_trim: float,
    min_chroma: float,
    title_stem: str,
) -> None:
    if not plt_available():
        return
    m = mask > 0
    xyz_valid = np.all(xyz_img >= 0.0, axis=2)
    m &= xyz_valid
    if not np.any(m):
        return
    L, a, b = pr250.xyz_to_lab_batch(xyz_img[m].reshape(-1, 3), D65_XYZN)
    sel, lo, hi, l_rel, c_rel = skin_lab_trim_selection(
        L, a, b, l_star_trim_lo=l_trim, l_star_trim_hi=l_trim, skin_min_chroma_ab=min_chroma
    )
    write_skin_lab_histogram_panel(
        path,
        L,
        a,
        b,
        sel,
        lo_thr=lo,
        hi_thr=hi,
        min_chroma_ab=min_chroma,
        l_trim_relaxed=l_rel,
        chroma_relaxed=c_rel,
        title=title_stem,
    )


def try_color_correction_package(
    bgr: np.ndarray,
    *,
    do_ffc: bool = False,
) -> Optional[np.ndarray]:
    """
    Optional: run ``ColorCorrectionPipeline`` if installed (FFC off by default for iPhone).
    Returns float RGB 0–1 or None.
    """
    try:
        from ColorCorrectionPipeline import ColorCorrection, Config
        from ColorCorrectionPipeline.core.utils import to_float64
    except ImportError:
        return None

    rgb = to_float64(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    cc = ColorCorrection()
    config = Config(
        do_ffc=do_ffc,
        do_gc=True,
        do_wb=True,
        do_cc=True,
        save=False,
        check_saturation=True,
        CC_kwargs={
            "cc_method": "ours",
            "mtd": "nn",
            "degree": 2,
            "hidden_layers": [64],
            "get_deltaE": False,
            "show": False,
        },
    )
    metrics, corrected, _errors = cc.run(Image=rgb, White_Image=None, name_="fitskin_frame", config=config)
    if not corrected:
        return None
    # first corrected stage image is usually final after CC
    key = max(corrected.keys()) if corrected else None
    if key is None:
        return None
    return np.clip(corrected[key], 0.0, 1.0)
