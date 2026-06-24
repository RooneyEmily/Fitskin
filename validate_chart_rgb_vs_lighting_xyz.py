#!/usr/bin/env python3
"""
Compare ColorChecker patch RGB from a calibration photo to spectrometer XYZ
(`lighting_reference_patches.csv` from aggregate_lighting_chart_xyz.py).

**Video pipeline note:** ``batch_extract_face_mesh_rgb.py`` (WB_videos) does **not** sample the
ColorChecker — it reads **already white-balanced** MP4s and averages **face-mesh tessellation**
triangles (BGR→RGB same as ``pixels[:,::-1]``). Video WB that **does** use the chart is
``WB_videos/videos.py`` (``colour_checker_detection`` segmentation, ``swatch_colours[18]``) and/or
``WB_videos/import cv2.py`` (OpenCV mcc). Spectrometer vs camera validation: this script.

Default chart path matches **videos.py**: ``colour_checker_detection`` segmentation first; use
``--chart-backend mcc`` to force OpenCV mcc only.

Pipeline:
  1. Detect chart / white on **original** BGR (segmentation or mcc), apply same WB as ``physio_skin_lab_monk``.
  2. Read **24** patch sRGB means on **WB** image (same backend order; fallback to mcc if needed).
  3. sRGB gamma decode (IEC 61966-2-1 piecewise EOTF) → linear RGB rows.
  4. **Linear least-squares** 3×3: ref_XYZ ≈ linear_RGB @ M (absorbs camera matrix + absolute scale vs instrument).
  5. ΔE*ab between Lab(ref XYZ) and Lab(fitted XYZ), both with **XYZn = measured white patch** (MCC index 18)
     so L*a*b* is expressed relative to your scene white (not D65).

  6. **Scan mode** (`--scan-data-root`): glob ``<root>/*/P1/Photos/*.jpg``, rank by chart area / frame
     (segmentation quadrilateral or mcc ``getBox()``), then lowest mean fitted ΔE; optional summary CSV.

  7. **Sampling check** (`--verify-sampling`): for the WB image, print a table comparing OpenCV
     ``getChartsRGB()`` patch means to **independent** means from ``getColorCharts()`` polygon ROIs
     (``mcc24_classic.verify_charts_average_vs_roi_mean``). If primary patch RGB came from segmentation,
     this table is an **mcc-only** consistency audit on the same frame. Optional ``--verify-overlay`` PNG.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import cv2
except ImportError as e:
    raise SystemExit("pip install opencv-python") from e

from mcc24_classic import (
    WHITE_PATCH_INDEX,
    decode_all_patches_srgb_255,
    decode_patch_srgb_255_from_charts_matrix,
    draw_mcc_patch_overlay,
    patch_display_name,
    patch_label_slug,
    verify_charts_average_vs_roi_mean,
)

from colour_checker_segmentation import detect_classic24_bgr, library_available as segmentation_available
from srgb_eotf import srgb_255_to_linear

# --- Same WB / chart decoding as physio_skin_lab_monk.py ---------------------------------
REFERENCE_WHITE_SRGB = np.array([243.0, 243.0, 242.0], dtype=np.float64)

XYZ2SRGB = np.array(
    [
        [3.2404542, -1.5371385, -0.4985314],
        [-0.9692660, 1.8760108, 0.0415560],
        [0.0556434, -0.2040259, 1.0572252],
    ],
    dtype=np.float64,
)
_INV_XYZ2SRGB = np.linalg.inv(XYZ2SRGB)


def _rgb_from_charts_matrix(charts_rgb: np.ndarray, patch_index: int) -> Optional[np.ndarray]:
    return decode_patch_srgb_255_from_charts_matrix(charts_rgb, patch_index)


def mcc_white_and_box(
    frame_bgr: np.ndarray, debug: bool = False
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    One mcc pass on frame_bgr. Returns (white_rgb 0..255, chart_box 4x2 float32 corners)
    or (None, None) if detection fails.
    """
    if not hasattr(cv2, "mcc"):
        if debug:
            print("opencv mcc missing — pip install opencv-contrib-python", file=sys.stderr)
        return None, None
    try:
        detector = cv2.mcc.CCheckerDetector.create()
        ok = detector.process(frame_bgr, 0, 1)
        if not ok:
            return None, None
        checkers = detector.getListColorChecker()
        if not checkers:
            return None, None
        checker = checkers[0]
        charts_rgb = checker.getChartsRGB()
        if charts_rgb is None or len(charts_rgb.shape) < 1:
            return None, None
        white_rgb = _rgb_from_charts_matrix(charts_rgb, WHITE_PATCH_INDEX)
        if white_rgb is None:
            return None, None
        if float(np.max(white_rgb)) <= 1.0:
            white_rgb = white_rgb * 255.0
        box = None
        if hasattr(checker, "getBox"):
            box = np.asarray(checker.getBox(), dtype=np.float64)
        return white_rgb, box
    except Exception as ex:
        if debug:
            print(f"mcc detection error: {ex}", file=sys.stderr)
        return None, None


def chart_area_fraction(frame_bgr: np.ndarray, box: np.ndarray) -> float:
    """Convex hull area of mcc getBox() / image area (how much of the frame the chart covers)."""
    h, w = frame_bgr.shape[:2]
    if box is None or box.size < 8:
        return 0.0
    poly = box.reshape(-1, 1, 2).astype(np.float32)
    area = float(cv2.contourArea(poly))
    return area / float(max(w * h, 1))


def white_balance_multipliers(measured_white_rgb: np.ndarray) -> np.ndarray:
    m = np.maximum(measured_white_rgb.astype(np.float64), 1e-6)
    return REFERENCE_WHITE_SRGB / m


def apply_wb_bgr(frame_bgr: np.ndarray, mult_rgb: np.ndarray) -> np.ndarray:
    x = frame_bgr.astype(np.float64)
    x[:, :, 2] *= mult_rgb[0]
    x[:, :, 1] *= mult_rgb[1]
    x[:, :, 0] *= mult_rgb[2]
    return np.clip(x, 0, 255).astype(np.uint8)


def charts_patch_rgb_255(wb_bgr: np.ndarray, debug: bool = False) -> Optional[np.ndarray]:
    """Return (24,3) sRGB means 0–255 in mcc patch order, or None."""
    if not hasattr(cv2, "mcc"):
        return None
    try:
        detector = cv2.mcc.CCheckerDetector.create()
        ok = detector.process(wb_bgr, 0, 1)
        if not ok:
            return None
        checkers = detector.getListColorChecker()
        if not checkers:
            return None
        charts_rgb = checkers[0].getChartsRGB()
        if charts_rgb is None:
            return None
        out = decode_all_patches_srgb_255(charts_rgb)
        return out
    except Exception as ex:
        if debug:
            print(f"mcc charts error: {ex}", file=sys.stderr)
        return None


def white_and_chart_area_backend(
    bgr: np.ndarray, chart_backend: str, debug: bool = False
) -> Tuple[Optional[np.ndarray], float, str]:
    """
    White patch RGB (~0–255) and chart area / frame. ``chart_backend``: ``auto`` | ``segmentation`` | ``mcc``.
    Returns ``(white_rgb | None, area_fraction, method)``.
    """
    if chart_backend in ("auto", "segmentation"):
        if segmentation_available():
            try:
                d = detect_classic24_bgr(bgr)
                if d is not None:
                    return d["white_rgb_255"], float(d["chart_area_fraction"]), "segmentation"
            except Exception as ex:
                if debug:
                    print(f"segmentation detector error: {ex}", file=sys.stderr)
            if chart_backend == "segmentation":
                return None, 0.0, "segmentation_failed"
    if chart_backend in ("auto", "mcc"):
        w, box = mcc_white_and_box(bgr, debug=debug)
        frac = chart_area_fraction(bgr, box) if box is not None else 0.0
        if w is not None:
            return w, frac, "mcc"
    return None, 0.0, "failed"


def patches_24_srgb_255_wb(
    wb_bgr: np.ndarray, chart_backend: str, debug: bool = False
) -> Tuple[Optional[np.ndarray], str]:
    """24×3 patch sRGB means on WB frame (``videos.py`` segmentation path or mcc fallback)."""
    if chart_backend in ("auto", "segmentation"):
        if segmentation_available():
            try:
                d = detect_classic24_bgr(wb_bgr)
                if d is not None:
                    return d["swatch_rgb_255"].copy(), "segmentation"
            except Exception as ex:
                if debug:
                    print(f"segmentation on WB image error: {ex}", file=sys.stderr)
            if chart_backend == "segmentation":
                return None, "segmentation_failed"
    if chart_backend in ("auto", "mcc"):
        m = charts_patch_rgb_255(wb_bgr, debug=debug)
        if m is not None:
            return m, "mcc"
    return None, "failed"


# --- MATLAB-style Lab (same as plot_face_mesh_mst_combined) -----------------------------
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


def srgb_linear_to_xyz_d65_100(rgb_lin: np.ndarray) -> np.ndarray:
    """(N,3) or (3,) linear sRGB 0–1 → XYZ same convention as face-mesh scripts."""
    one = rgb_lin.ndim == 1
    v = np.atleast_2d(np.asarray(rgb_lin, dtype=np.float64))
    xyz = 100.0 * (v @ _INV_XYZ2SRGB.T)
    return xyz.reshape(3) if one else xyz


def delta_e_ab(lab1: np.ndarray, lab2: np.ndarray) -> float:
    return float(np.linalg.norm(lab1 - lab2))


def load_lighting_xyz(path: Path) -> Tuple[np.ndarray, List[str]]:
    """Returns ref_xyz (24,3) row i = mcc index i, and patch labels."""
    rows: Dict[int, Tuple[float, float, float, str]] = {}
    with open(path, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            k = int(row["mcc_patch_index"])
            rows[k] = (
                float(row["X_mean"]),
                float(row["Y_mean"]),
                float(row["Z_mean"]),
                str(row.get("patch_label") or ""),
            )
    if set(rows.keys()) != set(range(24)):
        raise SystemExit(f"Expected mcc_patch_index 0..23 in {path}, got {sorted(rows.keys())}")
    labels = [rows[i][3] for i in range(24)]
    xyz = np.array([[rows[i][0], rows[i][1], rows[i][2]] for i in range(24)], dtype=np.float64)
    return xyz, labels


def evaluate_bgr_vs_ref(
    bgr: np.ndarray,
    ref_xyz: np.ndarray,
    labels: List[str],
    *,
    skip_wb_if_no_chart: bool = False,
    debug: bool = False,
    chart_backend: str = "auto",
) -> Optional[dict]:
    """
    Returns dict with metrics + patch_rgb, pred_xyz, de_fit, de_naive, wb_ok, chart_area_fraction,
    chart_detector (segmentation | mcc), or None if chart unusable.
    """
    white_rgb, frac, det_method = white_and_chart_area_backend(bgr, chart_backend, debug=debug)
    if white_rgb is None:
        if not skip_wb_if_no_chart:
            return None
        wb_bgr = bgr
        wb_ok = False
        det_method = "none"
    else:
        wb_bgr = apply_wb_bgr(bgr, white_balance_multipliers(white_rgb))
        wb_ok = True

    patch_rgb, patch_method = patches_24_srgb_255_wb(wb_bgr, chart_backend, debug=debug)
    if patch_rgb is None:
        return None

    lin = srgb_255_to_linear(patch_rgb)
    m_fit, *_ = np.linalg.lstsq(lin, ref_xyz, rcond=None)
    pred_xyz = lin @ m_fit
    xyzn = ref_xyz[WHITE_PATCH_INDEX].copy()
    de_fit: List[float] = []
    de_naive: List[float] = []
    for i in range(24):
        lab_r = xyz_to_lab(ref_xyz[i], xyzn)
        lab_f = xyz_to_lab(pred_xyz[i], xyzn)
        de_fit.append(delta_e_ab(lab_r, lab_f))
        lab_n = xyz_to_lab(srgb_linear_to_xyz_d65_100(lin[i]), xyzn)
        de_naive.append(delta_e_ab(lab_r, lab_n))
    de_fit_arr = np.array(de_fit)
    de_naive_arr = np.array(de_naive)
    return {
        "wb_bgr": wb_bgr,
        "chart_area_fraction": frac,
        "wb_ok": wb_ok,
        "chart_detector": det_method,
        "patch_rgb_source": patch_method,
        "patch_rgb": patch_rgb,
        "pred_xyz": pred_xyz,
        "de_fit": de_fit,
        "de_naive": de_naive,
        "de_fit_mean": float(de_fit_arr.mean()),
        "de_fit_median": float(np.median(de_fit_arr)),
        "de_fit_p95": float(np.percentile(de_fit_arr, 95)),
        "de_fit_max": float(de_fit_arr.max()),
        "de_fit_argmax": int(de_fit_arr.argmax()),
        "de_naive_mean": float(de_naive_arr.mean()),
        "de_naive_median": float(np.median(de_naive_arr)),
        "de_naive_max": float(de_naive_arr.max()),
    }


def write_per_patch_csv(
    path: Path,
    labels: List[str],
    patch_rgb: np.ndarray,
    ref_xyz: np.ndarray,
    pred_xyz: np.ndarray,
    de_fit: List[float],
    de_naive: List[float],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "mcc_patch_index",
                "patch_label",
                "R_mean",
                "G_mean",
                "B_mean",
                "X_ref",
                "Y_ref",
                "Z_ref",
                "X_fit",
                "Y_fit",
                "Z_fit",
                "dE_ab_fitted",
                "dE_ab_naive_srgb_d65",
            ]
        )
        for i in range(24):
            w.writerow(
                [
                    i,
                    labels[i],
                    patch_rgb[i, 0],
                    patch_rgb[i, 1],
                    patch_rgb[i, 2],
                    ref_xyz[i, 0],
                    ref_xyz[i, 1],
                    ref_xyz[i, 2],
                    pred_xyz[i, 0],
                    pred_xyz[i, 1],
                    pred_xyz[i, 2],
                    de_fit[i],
                    de_naive[i],
                ]
            )


def main() -> None:
    ap = argparse.ArgumentParser(
        description="ΔE*ab: photo ColorChecker vs spectrometer XYZ reference.",
        epilog=(
            "Example (use a real JPG path on your machine — not a tutorial placeholder):\n"
            "  %(prog)s --image /media/mabl-main/Data/Physio-code/Data/24/P1/Photos/20250428T151132Z.jpg \\\n"
            "    --lighting-ref ./lighting_output/lighting_reference_patches.csv \\\n"
            "    --out-csv ./lighting_output/chart_validate_per_patch.csv \\\n"
            "    --verify-sampling --verify-overlay ./lighting_output/mcc_patch_overlay.png\n"
            "\n"
            "Default chart detection follows WB_videos/videos.py (colour_checker_detection). "
            "Install with: pip install colour_checker_detection\n"
            "Force OpenCV mcc only: --chart-backend mcc (requires opencv-contrib-python)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--image", type=Path, default=None, help="Single calibration photo (omit if --scan-data-root)")
    ap.add_argument(
        "--scan-data-root",
        type=Path,
        default=None,
        help="Scan <root>/*/P1/Photos/*.jpg — rank by chart bbox area / frame, then mean ΔE (fitted)",
    )
    ap.add_argument("--scan-top", type=int, default=15, help="How many rows to print in scan mode")
    ap.add_argument(
        "--write-best-per-patch",
        type=Path,
        default=None,
        help="In scan mode: write per-patch CSV for the best-ranked image (largest chart area, then lowest mean ΔE)",
    )
    ap.add_argument(
        "--scan-summary-csv",
        type=Path,
        default=None,
        help="In scan mode: write all scanned images with chart_area_fraction and ΔE stats",
    )
    ap.add_argument(
        "--lighting-ref",
        type=Path,
        default=Path("lighting_output/lighting_reference_patches.csv"),
        help="CSV from aggregate_lighting_chart_xyz.py",
    )
    ap.add_argument("--out-csv", type=Path, default=None, help="Per-patch table for --image mode")
    ap.add_argument("--skip-wb-if-no-chart", action="store_true", help="Use raw image if chart not found on first pass")
    ap.add_argument("--debug", action="store_true")
    ap.add_argument(
        "--chart-backend",
        choices=("auto", "segmentation", "mcc"),
        default="auto",
        help="Chart CV: auto = colour_checker_detection (videos.py) then mcc; segmentation/mcc = force one",
    )
    ap.add_argument(
        "--verify-sampling",
        action="store_true",
        help="Print table: getChartsRGB average vs independent ROI mean per patch (WB image)",
    )
    ap.add_argument(
        "--verify-sampling-csv",
        type=Path,
        default=None,
        help="With --verify-sampling: write per-patch charts vs ROI comparison CSV",
    )
    ap.add_argument(
        "--verify-overlay",
        type=Path,
        default=None,
        help="Write PNG with green patch quads and mcc indices (WB image; use to see what is sampled)",
    )
    ap.add_argument(
        "--verify-overlay-max-width",
        type=int,
        default=1920,
        help="Max width in pixels for --verify-overlay (0 = full resolution)",
    )
    args = ap.parse_args()

    if args.chart_backend == "segmentation" and not segmentation_available():
        raise SystemExit(
            "colour_checker_detection not installed (required for --chart-backend segmentation). "
            "pip install colour_checker_detection"
        )
    if args.chart_backend == "mcc" and not hasattr(cv2, "mcc"):
        raise SystemExit("opencv-contrib-python required (cv2.mcc) for --chart-backend mcc")
    if args.chart_backend == "auto" and not segmentation_available() and not hasattr(cv2, "mcc"):
        raise SystemExit(
            "For --chart-backend auto, install colour_checker_detection and/or opencv-contrib-python (cv2.mcc)"
        )

    ref_path = args.lighting_ref.expanduser().resolve()
    if not ref_path.is_file():
        raise SystemExit(f"Lighting reference CSV not found: {ref_path}")
    ref_xyz, labels = load_lighting_xyz(ref_path)

    if args.scan_data_root is not None:
        if args.image is not None:
            raise SystemExit("Use either --image or --scan-data-root, not both.")
        root = args.scan_data_root.expanduser().resolve()
        if not root.is_dir():
            raise SystemExit(f"--scan-data-root not a directory: {root}")
        paths = sorted(root.glob("*/P1/Photos/*.jpg"))
        rows: List[Tuple[float, float, str, dict]] = []
        skipped = 0
        for p in paths:
            bgr = cv2.imread(str(p))
            if bgr is None:
                skipped += 1
                continue
            ev = evaluate_bgr_vs_ref(
                bgr,
                ref_xyz,
                labels,
                skip_wb_if_no_chart=args.skip_wb_if_no_chart,
                debug=args.debug,
                chart_backend=args.chart_backend,
            )
            if ev is None:
                skipped += 1
                continue
            # Sort: prefer larger chart in frame, then lower mean ΔE
            rows.append((-ev["chart_area_fraction"], ev["de_fit_mean"], str(p), ev))

        rows.sort(key=lambda t: (t[0], t[1]))
        print(f"scanned {len(paths)} paths under {root}/<id>/P1/Photos/*.jpg")
        print(f"usable: {len(rows)}  skipped (no chart / no WB): {skipped}")
        print(f"reference: {ref_path}")
        print(
            "rank  chart_area_frac  mean_dE_fit  median_dE  max_dE  wb  det  path\n"
            + "-" * 120
        )
        for rank, (_, _, pstr, ev) in enumerate(rows[: max(1, args.scan_top)], 1):
            det = str(ev.get("chart_detector", "?"))[:4]
            print(
                f"{rank:4d}  {ev['chart_area_fraction']:.5f}          {ev['de_fit_mean']:.3f}       "
                f"{ev['de_fit_median']:.3f}    {ev['de_fit_max']:.3f}   {int(ev['wb_ok'])}  {det:4s}  {pstr}"
            )

        if rows:
            best_path, best_ev = rows[0][2], rows[0][3]
            print("\nBest-ranked (largest chart area in frame, then lowest mean fitted ΔE):")
            print(f"  {best_path}")
            print(
                f"  chart_area_fraction={best_ev['chart_area_fraction']:.5f}  "
                f"mean ΔE (fitted)={best_ev['de_fit_mean']:.3f}  naive mean={best_ev['de_naive_mean']:.3f}  "
                f"chart_detector={best_ev.get('chart_detector', '?')}  patch_rgb={best_ev.get('patch_rgb_source', '?')}"
            )
            if args.write_best_per_patch is not None:
                outp = args.write_best_per_patch.expanduser().resolve()
                write_per_patch_csv(
                    outp,
                    labels,
                    best_ev["patch_rgb"],
                    ref_xyz,
                    best_ev["pred_xyz"],
                    best_ev["de_fit"],
                    best_ev["de_naive"],
                )
                print(f"Wrote per-patch CSV: {outp}")
            if args.verify_sampling or args.verify_overlay is not None:
                wb_best = best_ev["wb_bgr"]
                if args.verify_sampling:
                    print_and_save_sampling_verification(
                        wb_best,
                        csv_path=args.verify_sampling_csv,
                    )
                if args.verify_overlay is not None:
                    mw = args.verify_overlay_max_width
                    ok = draw_mcc_patch_overlay(
                        wb_best,
                        args.verify_overlay.expanduser().resolve(),
                        max_width=mw if mw > 0 else 0,
                    )
                    if ok:
                        print(f"Wrote verify overlay: {args.verify_overlay.expanduser().resolve()}")
                    else:
                        print("VERIFY OVERLAY: mcc failed on best WB image", file=sys.stderr)

        if args.scan_summary_csv is not None and rows:
            outp = args.scan_summary_csv.expanduser().resolve()
            outp.parent.mkdir(parents=True, exist_ok=True)
            with open(outp, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(
                    [
                        "rank",
                        "image_path",
                        "chart_area_fraction",
                        "wb_ok",
                        "chart_detector",
                        "patch_rgb_source",
                        "de_fit_mean",
                        "de_fit_median",
                        "de_fit_p95",
                        "de_fit_max",
                        "de_naive_mean",
                        "worst_patch_index",
                        "worst_patch_label",
                    ]
                )
                for rank, (_, _, pstr, ev) in enumerate(rows, 1):
                    wi = ev["de_fit_argmax"]
                    w.writerow(
                        [
                            rank,
                            pstr,
                            f"{ev['chart_area_fraction']:.8f}",
                            int(ev["wb_ok"]),
                            str(ev.get("chart_detector", "")),
                            str(ev.get("patch_rgb_source", "")),
                            f"{ev['de_fit_mean']:.6f}",
                            f"{ev['de_fit_median']:.6f}",
                            f"{ev['de_fit_p95']:.6f}",
                            f"{ev['de_fit_max']:.6f}",
                            f"{ev['de_naive_mean']:.6f}",
                            wi,
                            labels[wi],
                        ]
                    )
            print(f"Wrote scan summary: {outp}")
        return

    if args.image is None:
        raise SystemExit("Provide --image or --scan-data-root")

    img_path = args.image.expanduser().resolve()
    if not img_path.is_file():
        msg = f"Image not found: {img_path}"
        if "/path/to" in str(img_path).lower() or str(img_path).endswith("calibration.jpg"):
            msg += (
                "\n  → Replace --image with a real file (e.g. under "
                "…/Physio-code/Data/<id>/P1/Photos/*.jpg). Run: python3 validate_chart_rgb_vs_lighting_xyz.py --help"
            )
        raise SystemExit(msg)

    bgr = cv2.imread(str(img_path))
    if bgr is None:
        raise SystemExit(f"OpenCV could not read image: {img_path}")

    ev = evaluate_bgr_vs_ref(
        bgr,
        ref_xyz,
        labels,
        skip_wb_if_no_chart=args.skip_wb_if_no_chart,
        debug=args.debug,
        chart_backend=args.chart_backend,
    )
    if ev is None:
        raise SystemExit("ColorChecker not detected or patch read failed (see --skip-wb-if-no-chart)")

    print(f"image: {img_path}")
    print(f"reference: {ref_path}")
    print(f"chart_backend: {args.chart_backend}")
    print(f"chart_detector: {ev.get('chart_detector', '?')}  patch_rgb_source: {ev.get('patch_rgb_source', '?')}")
    print(f"chart_area_fraction (bbox / frame): {ev['chart_area_fraction']:.5f}")
    print(f"wb_applied: {ev['wb_ok']}")
    print(f"XYZn for Lab: measured white patch (Classic 24 index {WHITE_PATCH_INDEX}) from reference CSV")
    print(
        f"ΔE*ab (fitted 3×3 RGB_lin→XYZ): mean={ev['de_fit_mean']:.3f}  median={ev['de_fit_median']:.3f}  "
        f"p95={ev['de_fit_p95']:.3f}  max={ev['de_fit_max']:.3f}  "
        f"(patch {ev['de_fit_argmax']}, {labels[ev['de_fit_argmax']]})"
    )
    print(
        f"ΔE*ab (naive sRGB/D65 matrix, same XYZn): mean={ev['de_naive_mean']:.3f}  median={ev['de_naive_median']:.3f}  "
        f"max={ev['de_naive_max']:.3f}"
    )

    if args.out_csv is not None:
        write_per_patch_csv(
            args.out_csv.expanduser().resolve(),
            labels,
            ev["patch_rgb"],
            ref_xyz,
            ev["pred_xyz"],
            ev["de_fit"],
            ev["de_naive"],
        )
        print(f"Wrote {args.out_csv.expanduser().resolve()}")

    if args.verify_sampling:
        if ev.get("patch_rgb_source") != "mcc":
            print(
                "\nNote: fitted ΔE used swatch RGB from colour_checker_detection (videos.py path); "
                "the table below is an OpenCV mcc consistency check on the same WB frame.\n"
            )
        print_and_save_sampling_verification(
            ev["wb_bgr"],
            csv_path=args.verify_sampling_csv,
        )
    if args.verify_overlay is not None:
        mw = args.verify_overlay_max_width
        ok = draw_mcc_patch_overlay(
            ev["wb_bgr"],
            args.verify_overlay.expanduser().resolve(),
            max_width=mw if mw > 0 else 0,
        )
        if ok:
            print(f"Wrote verify overlay: {args.verify_overlay.expanduser().resolve()}")
        else:
            print("VERIFY OVERLAY: mcc failed on WB image", file=sys.stderr)


def print_and_save_sampling_verification(
    wb_bgr: np.ndarray,
    *,
    csv_path: Optional[Path] = None,
) -> None:
    """Compare OpenCV ``getChartsRGB`` averages to independent polygon ROI means (see mcc24_classic)."""
    v = verify_charts_average_vs_roi_mean(wb_bgr)
    if v is None:
        print("VERIFY SAMPLING: mcc failed on WB image", file=sys.stderr)
        return
    charts = v["charts_srgb_255"]
    roi = v["roi_srgb_255"]
    diff = v["abs_diff"]
    pz = v["p_sizes"]
    print("\n--- Patch sampling check: OpenCV getChartsRGB (average col) vs polygon ROI mean ---")
    print(
        "mcc  patch_name            n_pix    R_ch   R_roi  ΔR     G_ch   G_roi  ΔG     B_ch   B_roi  ΔB"
    )
    print("-" * 102)
    for i in range(24):
        name = patch_display_name(i)[:20].ljust(20)
        print(
            f"{i:2d}  {name}  {int(pz[i]):5d}  "
            f"{charts[i, 0]:6.2f} {roi[i, 0]:6.2f} {diff[i, 0]:6.3f}  "
            f"{charts[i, 1]:6.2f} {roi[i, 1]:6.2f} {diff[i, 1]:6.3f}  "
            f"{charts[i, 2]:6.2f} {roi[i, 2]:6.2f} {diff[i, 2]:6.3f}"
        )
    print(
        f"\nSummary: max |Δ| (any patch, any channel) = {v['max_abs_any']:.4f}  "
        f"mean |Δ| = {v['mean_abs_any']:.4f}"
    )
    print(
        "Per-channel max |Δ|: R={:.4f} G={:.4f} B={:.4f}   mean |Δ|: R={:.4f} G={:.4f} B={:.4f}".format(
            float(v["max_abs_per_channel"][0]),
            float(v["max_abs_per_channel"][1]),
            float(v["max_abs_per_channel"][2]),
            float(v["mean_abs_per_channel"][0]),
            float(v["mean_abs_per_channel"][1]),
            float(v["mean_abs_per_channel"][2]),
        )
    )
    if float(v["max_abs_any"]) < 0.05:
        print(
            "OK: OpenCV chart table matches independent ROI sampling (~sub‑0.05 RGB); "
            "decode order and quads are aligned."
        )
    else:
        print(
            "WARNING: larger mismatch — unusual getChartsRGB layout, partial occlusion, "
            "or non‑convex quad; inspect --verify-overlay image."
        )

    if csv_path is not None:
        outp = csv_path.expanduser().resolve()
        outp.parent.mkdir(parents=True, exist_ok=True)
        with open(outp, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(
                [
                    "mcc_patch_index",
                    "patch_display_name",
                    "patch_label_slug",
                    "p_size_opencv",
                    "R_getChartsRGB",
                    "G_getChartsRGB",
                    "B_getChartsRGB",
                    "R_roi_mean",
                    "G_roi_mean",
                    "B_roi_mean",
                    "abs_dR",
                    "abs_dG",
                    "abs_dB",
                ]
            )
            for i in range(24):
                w.writerow(
                    [
                        i,
                        patch_display_name(i),
                        patch_label_slug(i),
                        int(pz[i]),
                        charts[i, 0],
                        charts[i, 1],
                        charts[i, 2],
                        roi[i, 0],
                        roi[i, 1],
                        roi[i, 2],
                        diff[i, 0],
                        diff[i, 1],
                        diff[i, 2],
                    ]
                )
        print(f"Wrote sampling verification CSV: {outp}")


if __name__ == "__main__":
    main()
