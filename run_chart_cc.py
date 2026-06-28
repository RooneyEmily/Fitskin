#!/usr/bin/env python3
"""
In-scene ColorChecker pipeline — cheek Lab vs FitSkin (no flash/no-flash math).

Detects Classic 24 in each JPEG, white-balances from chart, fits 3×3 to canonical MCC D65,
applies to face cheek ROI, compares to FitSkin scanner Lab.

Usage::

    python3 run_chart_cc.py

Writes comparison CSV, summary, scatter plot, and **skin mask overlays** (mesh + cheek
segmentation tinted on the image) under ``chart_cc_output/``.

Bundled JPEG cohort: ``data/chart_cc_jpeg/`` + ``data/manifest_chart_cc_fitskin.csv``.

Pansor iPhone DNG (ColorChecker in scene, not in git)::

    export PANSOR_DATA_ROOT="/path/to/Pansor Images"
    python3 scripts/build_pansor_manifest.py --data-root "$PANSOR_DATA_ROOT"
    python3 run_chart_cc.py \\
        --input-mode dng \\
        --manifest data/pansor/manifest_pansor_fitskin.csv \\
        --cc-only \\
        --out-dir chart_cc_output/pansor_dng
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from statistics import median
from typing import Any, Dict, List

import cv2
import matplotlib.pyplot as plt
import mediapipe as mp
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from chart_cc_fitskin_lib import process_one_dng, process_one_image, silence_stderr  # noqa: E402
from delta_e_2000 import delta_e_2000  # noqa: E402

DEFAULT_MANIFEST = ROOT / "data" / "manifest_chart_cc_fitskin.csv"
DEFAULT_OUT = ROOT / "chart_cc_output"


def _subject_key(row: dict) -> str:
    sid = str(row.get("subject_id", "")).strip()
    if sid:
        return sid
    part = row.get("participant", "unknown")
    m = re.search(r"(\d+)", part)
    pid = m.group(1) if m else part.replace(" ", "_")
    return f"P{pid}_T{row.get('trial', 0)}"


def _resolve_path(p: str) -> Path:
    path = Path(p)
    if not path.is_absolute():
        path = ROOT / path
    return path.expanduser().resolve()


def _de00(Lp, ap, bp, Lf, af, bf) -> float:
    if not all(np.isfinite([Lp, ap, bp, Lf, af, bf])):
        return float("nan")
    return float(
        delta_e_2000(
            np.array([[[Lp, ap, bp]]], dtype=np.float64),
            np.array([[[Lf, af, bf]]], dtype=np.float64),
        )[0, 0]
    )


def _print_summary(rows: List[Dict[str, Any]], out_dir: Path) -> None:
    de = [float(r["deltaE00_cheek"]) for r in rows if np.isfinite(r["deltaE00_cheek"])]
    if not de:
        return
    print("\n" + "=" * 60)
    print("RESULTS  (chart CC cheek Lab vs FitSkin)")
    print("=" * 60)
    for r in rows:
        sid = r["subject_id"]
        de_val = float(r["deltaE00_cheek"])
        print(f"\n{sid}")
        print(
            f"  Pipeline:  L*={r['pipeline_L']:.2f}  a*={r['pipeline_a']:.2f}  b*={r['pipeline_b']:.2f}"
        )
        print(
            f"  FitSkin:   L*={r['fitskin_cheek_L']}  a*={r['fitskin_cheek_a']}  b*={r['fitskin_cheek_b']}"
        )
        print(f"  ΔE₀₀ = {de_val:.2f}")
    print("\n" + "-" * 60)
    print(
        f"Trials: {len(de)}   mean ΔE₀₀ = {sum(de)/len(de):.2f}   median ΔE₀₀ = {median(de):.2f}"
    )
    print("=" * 60)
    print(f"\nOverlays:  {out_dir / 'skin_mask_overlays'}")
    print(f"Full CSV:  {out_dir / 'comparison.csv'}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="In-scene ColorChecker → cheek Lab vs FitSkin.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help=f"Trial manifest (default: {DEFAULT_MANIFEST.relative_to(ROOT)})",
    )
    ap.add_argument(
        "--input-mode",
        choices=("jpeg", "dng"),
        default="jpeg",
        help="jpeg: bundled chart_cc JPEGs; dng: iPhone RAW (Pansor cohort)",
    )
    ap.add_argument(
        "--cc-only",
        action="store_true",
        help="Pansor manifest: keep ColorChecker trials only (skip Sephora Bag).",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT,
        help=f"Output directory (default: {DEFAULT_OUT.name}/)",
    )
    ap.add_argument("--skin-l-trim", type=float, default=0.05)
    ap.add_argument("--skin-min-chroma", type=float, default=2.0)
    ap.add_argument("--roi", choices=("mesh", "cheek", "both"), default="cheek")
    ap.add_argument(
        "--skin-tone",
        choices=("auto", "light", "dark"),
        default=None,
        help="Adaptive chart CC: auto=probe preview cheek L* → dark→mesh+affine, light→cheek+3×3. "
        "Overrides --roi and --affine when set.",
    )
    ap.add_argument("--huber", action="store_true", help="Huber IRWS fit (default: plain weighted lstsq)")
    ap.add_argument("--affine", action="store_true", help="3×4 affine RGB→XYZ fit")
    ap.add_argument(
        "--chart-gray-wb-from",
        choices=("white", "neutral_column_mean"),
        default="white",
        help="DNG gray WB reference patch(es) before matrix fit (default: white).",
    )
    ap.add_argument(
        "--no-overlays",
        action="store_true",
        help="Skip skin mesh / cheek segmentation overlay PNGs.",
    )
    ap.add_argument("--no-histograms", action="store_true")
    ap.add_argument(
        "--include-flash",
        action="store_true",
        default=True,
        help="Also process flash frames (default: on).",
    )
    ap.add_argument(
        "--no-include-flash",
        action="store_false",
        dest="include_flash",
    )
    args = ap.parse_args()

    manifest = args.manifest.expanduser().resolve()
    if not manifest.is_file():
        raise SystemExit(f"Missing manifest: {manifest}")

    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    overlay_noflash = out_dir / "skin_mask_overlays" / "noflash"
    overlay_flash = out_dir / "skin_mask_overlays" / "flash"
    hist_noflash = out_dir / "skin_lab_histograms" / "noflash"
    hist_flash = out_dir / "skin_lab_histograms" / "flash"

    rows_out: List[Dict[str, Any]] = []
    mp_fm = mp.solutions.face_mesh

    with silence_stderr():
        with mp_fm.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
        ) as face_mesh:
            with manifest.open(newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    if args.cc_only and row.get("condition_code", "CC") != "CC":
                        continue
                    if row.get("include_in_eval", "yes") == "no":
                        continue
                    sid = _subject_key(row)
                    frames = [
                        ("noflash", _resolve_path(row["path_noflash"]), overlay_noflash, hist_noflash)
                    ]
                    if args.include_flash and row.get("path_flash"):
                        frames.append(
                            ("flash", _resolve_path(row["path_flash"]), overlay_flash, hist_flash)
                        )

                    pipe: Dict[str, Any] = {"chart_ok": False}
                    for frame_kind, img_path, odir, hdir in frames:
                        if not img_path.is_file():
                            print(f"skip {sid} {frame_kind}: missing {img_path}", file=sys.stderr)
                            continue
                        stem = img_path.stem
                        ovp = None if args.no_overlays else odir / f"{sid}_{stem}_skin_overlay.png"
                        hist = (
                            None
                            if args.no_histograms
                            else hdir / f"{sid}_{stem}_{frame_kind}_skin_lab_hists.png"
                        )
                        ab_stem = (
                            None
                            if args.no_histograms
                            else hdir / f"{sid}_{stem}_{frame_kind}"
                        )
                        if args.input_mode == "dng":
                            frame_pipe = process_one_dng(
                                img_path,
                                face_mesh,
                                l_trim=args.skin_l_trim,
                                min_chroma=args.skin_min_chroma,
                                roi=args.roi,
                                huber=args.huber,
                                affine=args.affine,
                                chart_gray_wb_from=args.chart_gray_wb_from,
                                skin_tone=args.skin_tone,
                                write_overlay=ovp,
                                write_lab_histogram=hist,
                                write_ab_histogram_stem=ab_stem,
                                histogram_frame_label=frame_kind,
                            )
                        else:
                            bgr = cv2.imread(str(img_path))
                            if bgr is None:
                                print(
                                    f"skip {sid} {frame_kind}: imread failed {img_path}",
                                    file=sys.stderr,
                                )
                                continue
                            frame_pipe = process_one_image(
                                bgr,
                                face_mesh,
                                l_trim=args.skin_l_trim,
                                min_chroma=args.skin_min_chroma,
                                roi=args.roi,
                                huber=args.huber,
                                affine=args.affine,
                                write_overlay=ovp,
                                write_lab_histogram=hist,
                                write_ab_histogram_stem=ab_stem,
                                histogram_frame_label=frame_kind,
                            )
                        if frame_pipe.get("chart_ok") and frame_kind == "noflash":
                            pipe = frame_pipe
                        if frame_pipe.get("chart_ok"):
                            print(
                                f"{sid} {frame_kind}: cheek L*={frame_pipe.get('cheek_L', float('nan')):.1f}  "
                                f"a*={frame_pipe.get('cheek_a', float('nan')):.1f}  "
                                f"ΔE₀₀ vs FitSkin pending"
                            )
                        else:
                            print(
                                f"{sid} {frame_kind}: skip ({frame_pipe.get('status', 'chart_fail')})",
                                file=sys.stderr,
                            )

                    if not pipe.get("chart_ok"):
                        continue

                    fcL = float(row["fitskin_cheek_L"])
                    fca = float(row["fitskin_cheek_a"])
                    fcb = float(row["fitskin_cheek_b"])
                    ffL = float(row["fitskin_forehead_L"])
                    ffa = float(row["fitskin_forehead_a"])
                    ffb = float(row["fitskin_forehead_b"])

                    rec: Dict[str, Any] = {
                        "subject_id": sid,
                        "participant": row["participant"],
                        "trial": row["trial"],
                        "scan_session_id": row.get("scan_session_id", ""),
                        "reference": "mcc24_canonical_d65",
                        "roi_primary": args.roi,
                        "patch_de_ab_mean": pipe.get("patch_de_ab_mean"),
                        "patch_de_ab_skin01": pipe.get("patch_de_ab_skin01"),
                        "chart_area_fraction": pipe.get("chart_area_fraction"),
                        "skin_tone_tier": pipe.get("skin_tone_tier"),
                        "skin_tone_probe_L": pipe.get("skin_tone_probe_L"),
                        "effective_roi": pipe.get("effective_roi"),
                        "effective_affine": pipe.get("effective_affine"),
                        "pipeline_L": pipe["pipeline_L"],
                        "pipeline_a": pipe["pipeline_a"],
                        "pipeline_b": pipe["pipeline_b"],
                        "pipeline_C": pipe["pipeline_C"],
                        "pipeline_n_pixels": pipe["pipeline_n_pixels"],
                        "cheek_L": pipe.get("cheek_L"),
                        "cheek_a": pipe.get("cheek_a"),
                        "cheek_b": pipe.get("cheek_b"),
                        "fitskin_cheek_L": fcL,
                        "fitskin_cheek_a": fca,
                        "fitskin_cheek_b": fcb,
                        "fitskin_forehead_L": ffL,
                        "fitskin_forehead_a": ffa,
                        "fitskin_forehead_b": ffb,
                        "deltaL_cheek": pipe["pipeline_L"] - fcL,
                        "delta_a_cheek": pipe["pipeline_a"] - fca,
                        "delta_b_cheek": pipe["pipeline_b"] - fcb,
                        "deltaE00_cheek": _de00(
                            pipe["pipeline_L"], pipe["pipeline_a"], pipe["pipeline_b"], fcL, fca, fcb
                        ),
                        "deltaE00_forehead": _de00(
                            pipe["pipeline_L"], pipe["pipeline_a"], pipe["pipeline_b"], ffL, ffa, ffb
                        ),
                    }
                    rows_out.append(rec)

    if not rows_out:
        raise SystemExit("No rows processed.")

    out_csv = out_dir / "comparison.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
        w.writeheader()
        w.writerows(rows_out)

    de = [r["deltaE00_cheek"] for r in rows_out]
    da = [r["delta_a_cheek"] for r in rows_out]
    summary = {
        "n": len(rows_out),
        "input_mode": args.input_mode,
        "reference": "mcc24_canonical_d65",
        "roi_primary": args.roi,
        "deltaE00_cheek_mean": float(np.mean(de)),
        "deltaE00_cheek_median": float(np.median(de)),
        "delta_a_cheek_mean": float(np.mean(da)),
        "delta_a_cheek_median": float(np.median(da)),
    }
    with (out_dir / "summary.json").open("w", encoding="utf-8") as jf:
        json.dump(summary, jf, indent=2)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    for ax, (xk, yk, lab) in zip(
        axes,
        [
            ("fitskin_cheek_L", "pipeline_L", "L*"),
            ("fitskin_cheek_a", "pipeline_a", "a*"),
            ("fitskin_cheek_b", "pipeline_b", "b*"),
        ],
    ):
        for r in rows_out:
            ax.scatter(r[xk], r[yk], s=80, edgecolors="k", linewidths=0.5)
            ax.annotate(r["subject_id"], (r[xk], r[yk]), fontsize=7, xytext=(3, 3))
        lo = min(min(r[xk] for r in rows_out), min(r[yk] for r in rows_out)) - 2
        hi = max(max(r[xk] for r in rows_out), max(r[yk] for r in rows_out)) + 2
        ax.plot([lo, hi], [lo, hi], "k--", lw=0.8, alpha=0.5)
        ax.set_xlabel(f"FitSkin cheek {lab}")
        ax.set_ylabel(f"Chart CC pipeline {lab}")
    fig.suptitle(f"Canonical MCC D65 CC vs FitSkin (n={len(rows_out)})", fontsize=11)
    fig.savefig(out_dir / "Lab_chart_cc_vs_fitskin_cheek.png", dpi=160)
    plt.close(fig)

    _print_summary(rows_out, out_dir)


if __name__ == "__main__":
    main()
