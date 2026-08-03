#!/usr/bin/env python3
"""Train FitSkin-free cheek tone/chroma → ethnicity classifier.

Uses frozen preAWB+5500K cheek features and demographics ethnicity labels only
(never FitSkin Lab). Write JSON for ``--l-sampling tone_chroma``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from flash_noflash_spectral import planck_xyz_y1  # noqa: E402
from models.roi_tone_classifier import (  # noqa: E402
    DEFAULT_TONES_PATH,
    cheek_tone_features,
    train_roi_tone_classifier,
)
from scripts.evaluate_pansor20_chartfree_d65 import (  # noqa: E402
    D65,
    apple_face_cheek_masks,
    discover_indoor_trials,
    extract_zip,
    load_affine,
    load_apple_landmarks,
    load_demographics,
    load_dng_linear,
    match_flash_exposure,
    rgb_to_xyz_affine,
    xyz_to_lab,
)
import physio_skin_lab_raw_pr250 as pr250  # noqa: E402


def cheek_labs(R, cheek, M, cat):
    pix = R[cheek > 0]
    xyz = np.maximum(rgb_to_xyz_affine(pix, M), 0.0) @ cat.T
    lab = xyz_to_lab(xyz)
    C = np.hypot(lab[:, 1], lab[:, 2])
    if (C >= 2).sum() >= 10:
        lab = lab[C >= 2]
    return lab


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--data-root",
        type=Path,
        default=Path("/home/mabl-main/Documents/Pansor Dataset"),
    )
    ap.add_argument(
        "--cal-dir",
        type=Path,
        default=ROOT / "calibration" / "tier3_affine",
    )
    ap.add_argument("--fixed-cat-k", type=float, default=5500.0)
    ap.add_argument(
        "--work-dir",
        type=Path,
        default=ROOT / "results" / "pansor20_preawb_5500cat" / "_extract",
    )
    ap.add_argument("--out", type=Path, default=DEFAULT_TONES_PATH)
    args = ap.parse_args()

    demo = load_demographics(args.data_root / "Pansor Dataset Demographics.xlsx")
    trials = discover_indoor_trials(args.data_root)
    M = load_affine(args.cal_dir)
    cat = pr250.bradford_cat_matrix(planck_xyz_y1(float(args.fixed_cat_k), 0.0), D65)

    feats = []
    eths = []
    names = []
    for t in trials:
        pid = int(t["participant_id"])
        meta = demo[pid]
        wdir = args.work_dir / t["subject_id"]
        nf, fl, lm = extract_zip(Path(t["zip_path"]), wdir)
        A0 = load_dng_linear(nf, half_size=True, use_camera_wb=False)
        B0 = load_dng_linear(fl, half_size=True, use_camera_wb=False)
        if B0.shape != A0.shape:
            B0 = cv2.resize(B0, (A0.shape[1], A0.shape[0]), interpolation=cv2.INTER_AREA)
        _, cheek = apple_face_cheek_masks(
            load_apple_landmarks(lm), A0.shape[0], A0.shape[1]
        )
        B0m, _ = match_flash_exposure(A0, B0, cheek)
        R0 = np.sqrt(np.maximum(A0, 0) * np.maximum(B0m, 0) + 1e-8)
        lab = cheek_labs(R0, cheek, M, cat)
        feats.append(cheek_tone_features(lab))
        eths.append(meta["ethnicity"])
        names.append(meta["name"])

    clf = train_roi_tone_classifier(feats, eths)
    pred = [clf.predict(f) for f in feats]
    acc = float(np.mean([p == e for p, e in zip(pred, eths)]))
    clf.save(args.out)
    summary = {
        "n_trials": len(feats),
        "n_people": len(set(names)),
        "train_accuracy_in_sample": acc,
        "classes": clf.classes,
        "out": str(args.out),
        "fitskin_used": False,
        "label_source": "demographics_ethnicity",
        "color_path": f"preawb_cat@{args.fixed_cat_k:.0f}K",
    }
    (args.out.parent / "train_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
