#!/usr/bin/env python3
"""
Regenerate cheek vs mesh segmentation overlays from a chart CC manifest.

Green tint  = cheek ROI (convex hull of MediaPipe cheek landmarks ∩ skin mesh).
Yellow tint = rest of face mesh.
White outline = cheek landmark hull.

Usage::

    python3 scripts/visualize_cheek_segmentation.py
    python3 scripts/visualize_cheek_segmentation.py --manifest data/manifest_chart_cc_fitskin.csv
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2
import mediapipe as mp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from chart_cc_fitskin_lib import (  # noqa: E402
    chart_detect_and_wb,
    cheek_mask_from_landmarks,
    silence_stderr,
    write_cheek_vs_mesh_overlay_png,
)
import physio_skin_lab_monk as psl  # noqa: E402


def _resolve_path(p: str) -> Path:
    path = Path(p)
    if not path.is_absolute():
        path = ROOT / path
    return path.expanduser().resolve()


def main() -> None:
    ap = argparse.ArgumentParser(description="Cheek vs mesh ROI overlays for chart CC cohort.")
    ap.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "data" / "manifest_chart_cc_fitskin.csv",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "chart_cc_output" / "cheek_segmentation",
    )
    args = ap.parse_args()

    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = list(csv.DictReader(args.manifest.open(newline="", encoding="utf-8")))
    with silence_stderr():
        with mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
        ) as face_mesh:
            for row in rows:
                sid = row.get("subject_id") or f"P{row['participant'].split()[-1]}_T{row['trial']}"
                img_path = _resolve_path(row["path_noflash"])
                bgr = cv2.imread(str(img_path))
                if bgr is None:
                    print(f"skip {sid}: cannot read {img_path}", file=sys.stderr)
                    continue
                wb, _, _, st = chart_detect_and_wb(bgr)
                if wb is None:
                    wb = bgr
                h, w = wb.shape[:2]
                res = face_mesh.process(cv2.cvtColor(wb, cv2.COLOR_BGR2RGB))
                if not res.multi_face_landmarks:
                    print(f"skip {sid}: no face", file=sys.stderr)
                    continue
                lm = res.multi_face_landmarks[0].landmark
                mesh_mask, _, _, _, _ = psl.build_skin_mask_from_mesh(
                    h, w, lm, skin_triangulation="tessellation", exclusion_dilate_iod_fraction=0.12
                )
                cheek = cheek_mask_from_landmarks(h, w, lm, mesh_mask)
                out = out_dir / f"{sid}_cheek_vs_mesh.png"
                write_cheek_vs_mesh_overlay_png(out, wb, mesh_mask, cheek, lm)
                n_c = int((cheek > 0).sum())
                n_m = int((mesh_mask > 0).sum())
                print(f"{sid}: cheek {n_c:,} px / mesh {n_m:,} px -> {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
