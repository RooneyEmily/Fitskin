#!/usr/bin/env python3
"""Train illuminant-aware Lab→Lab correctors on Pansor-20 + ring-light hybrid_deploy Labs.

Fits ``Lab_corr = [L,a,b,1] @ W`` with:
  - W_global — all trials
  - W_d65 — indoor Pansor-20 + ring D65
  - W_f12 — ring F12 only
  - W_routed — apply W_d65 or W_f12 by illuminant label at inference

Reports participant leave-one-out ΔE₀₀ on each cohort and combined.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from delta_e_2000 import delta_e_2000  # noqa: E402


def fit_W(rows: List[Dict[str, Any]]) -> np.ndarray:
    X = np.array(
        [
            [
                float(r["pipeline_L"]),
                float(r["pipeline_a"]),
                float(r["pipeline_b"]),
                1.0,
            ]
            for r in rows
        ],
        dtype=np.float64,
    )
    Y = np.array(
        [
            [float(r["fitskin_L"]), float(r["fitskin_a"]), float(r["fitskin_b"])]
            for r in rows
        ],
        dtype=np.float64,
    )
    W, *_ = np.linalg.lstsq(X, Y, rcond=None)
    return W


def apply_W(lab: np.ndarray, W: np.ndarray) -> np.ndarray:
    lab = np.asarray(lab, dtype=np.float64).reshape(3)
    return np.array([lab[0], lab[1], lab[2], 1.0], dtype=np.float64) @ W


def de00(pred: np.ndarray, fit: np.ndarray) -> float:
    return float(delta_e_2000(pred, fit))


def load_pansor_rows(csv_path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with csv_path.open(newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append(
                {
                    "cohort": "pansor20",
                    "person_key": f"pansor_{r['participant_id']}",
                    "person": str(r["name"]),
                    "illuminant": "D65",
                    "pipeline_L": float(r["pipeline_L"]),
                    "pipeline_a": float(r["pipeline_a"]),
                    "pipeline_b": float(r["pipeline_b"]),
                    "fitskin_L": float(r["fitskin_L"]),
                    "fitskin_a": float(r["fitskin_a"]),
                    "fitskin_b": float(r["fitskin_b"]),
                    "subject_id": str(r["subject_id"]),
                }
            )
    return rows


def load_ring_rows(csv_path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with csv_path.open(newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append(
                {
                    "cohort": "ring",
                    "person_key": f"ring_{r['person']}",
                    "person": str(r["person"]),
                    "illuminant": str(r["illuminant"]).upper(),
                    "pipeline_L": float(r["pred_L_hybrid_deploy"]),
                    "pipeline_a": float(r["pred_a_hybrid_deploy"]),
                    "pipeline_b": float(r["pred_b_hybrid_deploy"]),
                    "fitskin_L": float(r["fitskin_L"]),
                    "fitskin_a": float(r["fitskin_a"]),
                    "fitskin_b": float(r["fitskin_b"]),
                    "subject_id": str(r["subject_id"]),
                }
            )
    return rows


def loo_person_weights(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, List[List[float]]]]:
    """Per-person LOO W_d65 / W_f12 for honest ring evaluation."""
    people = sorted({r["person_key"] for r in rows})
    out: Dict[str, Dict[str, List[List[float]]]] = {}
    for pk in people:
        train = [r for r in rows if r["person_key"] != pk]
        W_d65 = fit_W([r for r in train if r["illuminant"] == "D65"])
        f12_train = [r for r in train if r["illuminant"] == "F12"]
        W_f12 = fit_W(f12_train) if f12_train else W_d65
        out[pk] = {"W_d65": W_d65.tolist(), "W_f12": W_f12.tolist()}
    return out


def loo_stats(
    rows: List[Dict[str, Any]],
    *,
    matrices: Dict[str, np.ndarray],
    routed: bool = False,
) -> Dict[str, Any]:
    people = sorted({r["person_key"] for r in rows})
    base: List[float] = []
    corr: List[float] = []
    for pk in people:
        train = [r for r in rows if r["person_key"] != pk]
        test = [r for r in rows if r["person_key"] == pk]
        if routed:
            W_d65 = fit_W([r for r in train if r["illuminant"] == "D65"])
            W_f12 = fit_W([r for r in train if r["illuminant"] == "F12"])
            if not any(r["illuminant"] == "F12" for r in train):
                W_f12 = W_d65
        else:
            W = matrices["W_global"]
        for r in test:
            fit = np.array(
                [r["fitskin_L"], r["fitskin_a"], r["fitskin_b"]], dtype=np.float64
            )
            base_lab = np.array(
                [r["pipeline_L"], r["pipeline_a"], r["pipeline_b"]], dtype=np.float64
            )
            base.append(de00(base_lab, fit))
            if routed:
                W = W_f12 if r["illuminant"] == "F12" else W_d65
            corr.append(de00(apply_W(base_lab, W), fit))
    return {
        "n": len(corr),
        "baseline_mean_de00": float(np.mean(base)) if base else None,
        "loo_mean_de00": float(np.mean(corr)) if corr else None,
        "loo_median_de00": float(np.median(corr)) if corr else None,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--pansor-csv",
        type=Path,
        default=ROOT / "figures" / "pansor20_fairface7" / "cohort_de00.csv",
    )
    ap.add_argument(
        "--ring-csv",
        type=Path,
        default=ROOT / "results" / "torch_illuminant_ringlight" / "torch_illuminant_ringlight.csv",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "calibration" / "multi_illuminant_lab_affine",
    )
    args = ap.parse_args()

    rows: List[Dict[str, Any]] = []
    if args.pansor_csv.is_file():
        rows.extend(load_pansor_rows(args.pansor_csv))
    if args.ring_csv.is_file():
        rows.extend(load_ring_rows(args.ring_csv))
    if not rows:
        raise SystemExit("No training rows — provide --pansor-csv and/or --ring-csv")

    d65_rows = [r for r in rows if r["illuminant"] == "D65"]
    f12_rows = [r for r in rows if r["illuminant"] == "F12"]

    matrices = {
        "W_global": fit_W(rows),
        "W_d65": fit_W(d65_rows) if d65_rows else fit_W(rows),
        "W_f12": fit_W(f12_rows) if f12_rows else fit_W(d65_rows),
    }

    meta: Dict[str, Any] = {
        "n_total": len(rows),
        "n_d65": len(d65_rows),
        "n_f12": len(f12_rows),
        "n_pansor": sum(1 for r in rows if r["cohort"] == "pansor20"),
        "n_ring": sum(1 for r in rows if r["cohort"] == "ring"),
        "pansor_csv": str(args.pansor_csv) if args.pansor_csv.is_file() else None,
        "ring_csv": str(args.ring_csv) if args.ring_csv.is_file() else None,
        "matrices": {k: v.tolist() for k, v in matrices.items()},
        "loo_weights_by_person": loo_person_weights(rows),
        "loo": {
            "combined_global": loo_stats(rows, matrices=matrices, routed=False),
            "combined_routed": loo_stats(rows, matrices=matrices, routed=True),
            "ring_routed": loo_stats(
                [r for r in rows if r["cohort"] == "ring"], matrices=matrices, routed=True
            ),
            "pansor_global": loo_stats(
                [r for r in rows if r["cohort"] == "pansor20"], matrices=matrices, routed=False
            ),
        },
        "note": (
            "Trained on hybrid_deploy cheek Lab (ring) + frozen-5500 cheek Lab (Pansor-20). "
            "Use mode=routed at inference."
        ),
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_json = args.out_dir / "multi_illuminant_lab_affine.json"
    out_json.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    for key, W in matrices.items():
        np.save(args.out_dir / f"{key}.npy", W)

    print(json.dumps(meta["loo"], indent=2))
    print(f"Wrote {out_json}")


if __name__ == "__main__":
    main()
