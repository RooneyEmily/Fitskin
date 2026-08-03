#!/usr/bin/env python3
"""Leave-one-subject-out validation of ROI sampling on the frozen color path.

Frozen color path: pre-AWB reflectance + tier3_affine + 5500 K Bradford CAT→D65.

Compares:
  - off: trimmed-mean cheek Lab (no ROI heuristic)
  - specular_tone: demographics ethnicity oracle (heuristic upper bound)
  - tone_chroma: ethnicity predicted from FitSkin-free tone features
    (classifier retrained each LOSO fold on other subjects only)

FitSkin Lab is used ONLY as the evaluation target (ΔE00), never as a training label.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, median

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from delta_e_2000 import delta_e_2000  # noqa: E402
from flash_noflash_spectral import planck_xyz_y1  # noqa: E402
from models.roi_tone_classifier import (  # noqa: E402
    cheek_tone_features,
    train_roi_tone_classifier,
)
from scripts.evaluate_pansor20_chartfree_d65 import (  # noqa: E402
    D65,
    _trimmed_mean_lab,
    apple_face_cheek_masks,
    apply_specular_tone_sampling,
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
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "results" / "pansor20_roi_loso",
    )
    args = ap.parse_args()

    demo = load_demographics(args.data_root / "Pansor Dataset Demographics.xlsx")
    trials = discover_indoor_trials(args.data_root)
    M = load_affine(args.cal_dir)
    cat = pr250.bradford_cat_matrix(planck_xyz_y1(float(args.fixed_cat_k), 0.0), D65)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
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
        labs = cheek_labs(R0, cheek, M, cat)
        base = _trimmed_mean_lab(labs)
        eth_lab, _ = apply_specular_tone_sampling(labs, meta["ethnicity"])
        fit = np.array([meta["fitskin_L"], meta["fitskin_a"], meta["fitskin_b"]])
        feat = cheek_tone_features(labs)
        rows.append(
            {
                "name": meta["name"],
                "eth": meta["ethnicity"],
                "pid": pid,
                "trial": int(t["trial"]),
                "labs": labs,
                "base": base,
                "eth_lab": eth_lab,
                "fit": fit,
                "feat": feat,
                "de_off": float(delta_e_2000(base, fit)),
                "de_oracle": float(delta_e_2000(eth_lab, fit)),
            }
        )

    people = sorted({r["name"] for r in rows})
    loso_rows = []
    for hold in people:
        train = [r for r in rows if r["name"] != hold]
        test = [r for r in rows if r["name"] == hold]
        clf = train_roi_tone_classifier(
            [r["feat"] for r in train], [r["eth"] for r in train]
        )
        for r in test:
            pred_eth = clf.predict(r["feat"])
            pred_lab, _ = apply_specular_tone_sampling(r["labs"], pred_eth)
            de_pred = float(delta_e_2000(pred_lab, r["fit"]))
            loso_rows.append(
                {
                    "name": r["name"],
                    "ethnicity": r["eth"],
                    "participant_id": r["pid"],
                    "trial": r["trial"],
                    "predicted_ethnicity": pred_eth,
                    "eth_match": pred_eth == r["eth"],
                    "de_off": r["de_off"],
                    "de_specular_tone_oracle": r["de_oracle"],
                    "de_tone_chroma_loso": de_pred,
                    "pipeline_L": round(float(pred_lab[0]), 4),
                    "pipeline_a": round(float(pred_lab[1]), 4),
                    "pipeline_b": round(float(pred_lab[2]), 4),
                }
            )

    def summarize(key: str):
        vals = [r[key] for r in loso_rows]
        by_eth: dict = defaultdict(list)
        for r in loso_rows:
            by_eth[r["ethnicity"]].append(r[key])
        return {
            "mean": float(mean(vals)),
            "median": float(median(vals)),
            "by_ethnicity": {
                eth: {"n": len(v), "mean": float(mean(v)), "median": float(median(v))}
                for eth, v in sorted(by_eth.items())
            },
        }

    eth_acc = float(np.mean([r["eth_match"] for r in loso_rows]))
    summary = {
        "n_trials": len(loso_rows),
        "n_people": len(people),
        "color_path": f"preawb_cat@{args.fixed_cat_k:.0f}K + tier3_affine (FROZEN)",
        "fitskin_role": "evaluation_target_only",
        "classifier_labels": "demographics_ethnicity_not_fitskin",
        "loso_ethnicity_accuracy": eth_acc,
        "off_trimmed_mean": summarize("de_off"),
        "specular_tone_ethnicity_oracle": summarize("de_specular_tone_oracle"),
        "tone_chroma_loso": summarize("de_tone_chroma_loso"),
        "interpretation": {
            "off": "Claimable frozen colorimetric path (no ROI heuristic).",
            "specular_tone_oracle": (
                "Heuristic upper bound using true demographics ethnicity; "
                "cohort-tuned thresholds — not for generalization claims."
            ),
            "tone_chroma_loso": (
                "Same ROI rules with ethnicity predicted from FitSkin-free "
                "tone/chroma features; classifier never trained on held-out subject."
            ),
        },
    }

    (args.out_dir / "loso_trials.json").write_text(
        json.dumps(loso_rows, indent=2) + "\n", encoding="utf-8"
    )
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )

    print("=== ROI sampling LOSO (frozen preAWB+5500 color path) ===")
    print(f"n={summary['n_trials']} people={summary['n_people']}")
    print(f"ethnicity LOSO acc={eth_acc:.3f}")
    for label, key in [
        ("off (frozen color)", "off_trimmed_mean"),
        ("specular_tone oracle", "specular_tone_ethnicity_oracle"),
        ("tone_chroma LOSO", "tone_chroma_loso"),
    ]:
        st = summary[key]
        print(f"{label:28s} mean ΔE00={st['mean']:.3f}  median={st['median']:.3f}")
        for eth, e in st["by_ethnicity"].items():
            print(f"  {eth:10s} {e['mean']:6.2f}")
    print(f"\nWrote {args.out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
