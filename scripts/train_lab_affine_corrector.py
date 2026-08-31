#!/usr/bin/env python3
"""
Train a Lab→Lab affine corrector for chart-free Pansor-20.

Fits ``Lab_corr = [L,a,b,1] @ W`` (W is 4×3) from pipeline cheek Lab to
FitSkin Inside Lab. Reports participant leave-one-out ΔE00 (honest gate) and
writes a frozen all-data artifact for inference.
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


def _load_rows(csv_path: Path) -> List[Dict[str, Any]]:
    with csv_path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _mats(rows: List[Dict[str, Any]]) -> Tuple[np.ndarray, np.ndarray]:
    X = np.array(
        [[float(r["pipeline_L"]), float(r["pipeline_a"]), float(r["pipeline_b"]), 1.0] for r in rows],
        dtype=np.float64,
    )
    Y = np.array(
        [[float(r["fitskin_L"]), float(r["fitskin_a"]), float(r["fitskin_b"])] for r in rows],
        dtype=np.float64,
    )
    return X, Y


def fit_W(rows: List[Dict[str, Any]]) -> np.ndarray:
    X, Y = _mats(rows)
    W, *_ = np.linalg.lstsq(X, Y, rcond=None)
    return W  # (4, 3)


def apply_W(L: float, a: float, b: float, W: np.ndarray) -> np.ndarray:
    return np.array([L, a, b, 1.0], dtype=np.float64) @ W


def de00(lab: np.ndarray, fit: np.ndarray) -> float:
    return float(delta_e_2000(lab, fit))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--baseline-csv",
        type=Path,
        default=ROOT / "results" / "pansor20_chartfree_d65_affine" / "pansor20_chartfree_d65.csv",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "calibration" / "lab_affine_corrector_pansor",
    )
    args = ap.parse_args()

    rows = _load_rows(args.baseline_csv)
    if not rows:
        raise SystemExit(f"No rows in {args.baseline_csv}")

    pids = sorted({int(r["participant_id"]) for r in rows})
    loo_de: List[float] = []
    base_de: List[float] = []
    loo_by_eth: Dict[str, List[float]] = {}
    loo_rows: List[Dict[str, Any]] = []

    for pid in pids:
        train = [r for r in rows if int(r["participant_id"]) != pid]
        test = [r for r in rows if int(r["participant_id"]) == pid]
        W = fit_W(train)
        for r in test:
            fit = np.array(
                [float(r["fitskin_L"]), float(r["fitskin_a"]), float(r["fitskin_b"])],
                dtype=np.float64,
            )
            base = np.array(
                [float(r["pipeline_L"]), float(r["pipeline_a"]), float(r["pipeline_b"])],
                dtype=np.float64,
            )
            corr = apply_W(base[0], base[1], base[2], W)
            d0 = de00(base, fit)
            d1 = de00(corr, fit)
            base_de.append(d0)
            loo_de.append(d1)
            eth = str(r["ethnicity"])
            loo_by_eth.setdefault(eth, []).append(d1)
            loo_rows.append(
                {
                    "subject_id": r["subject_id"],
                    "participant_id": pid,
                    "name": r.get("name"),
                    "ethnicity": eth,
                    "trial": r["trial"],
                    "de00_affine": d0,
                    "de00_loo_lab_affine": d1,
                    "corr_L": float(corr[0]),
                    "corr_a": float(corr[1]),
                    "corr_b": float(corr[2]),
                }
            )

    W_all = fit_W(rows)
    frozen_de = []
    for r in rows:
        fit = np.array(
            [float(r["fitskin_L"]), float(r["fitskin_a"]), float(r["fitskin_b"])],
            dtype=np.float64,
        )
        corr = apply_W(float(r["pipeline_L"]), float(r["pipeline_a"]), float(r["pipeline_b"]), W_all)
        frozen_de.append(de00(corr, fit))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    np.save(args.out_dir / "lab_affine_4x3.npy", W_all)
    meta = {
        "baseline_csv": str(args.baseline_csv),
        "n_trials": len(rows),
        "n_participants": len(pids),
        "affine_mean_de00": float(np.mean(base_de)),
        "loo_lab_affine_mean_de00": float(np.mean(loo_de)),
        "loo_lab_affine_median_de00": float(np.median(loo_de)),
        "frozen_all_fit_mean_de00": float(np.mean(frozen_de)),
        "frozen_all_fit_median_de00": float(np.median(frozen_de)),
        "by_ethnicity_loo": {
            eth: {
                "n": len(v),
                "mean_de00": float(np.mean(v)),
                "median_de00": float(np.median(v)),
            }
            for eth, v in sorted(loo_by_eth.items())
        },
        "lab_affine_4x3": W_all.tolist(),
        "note": (
            "LOO mean is the honest generalization estimate. "
            "Frozen W is fit on all Pansor-20 trials for deployment; "
            "frozen_all_fit_* is optimistic (not held-out)."
        ),
    }
    (args.out_dir / "lab_affine_corrector.json").write_text(
        json.dumps(meta, indent=2) + "\n", encoding="utf-8"
    )
    loo_csv = args.out_dir / "loo_results.csv"
    with loo_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(loo_rows[0].keys()))
        w.writeheader()
        w.writerows(loo_rows)

    print(json.dumps({k: meta[k] for k in meta if k != "lab_affine_4x3"}, indent=2))
    print(f"Wrote {args.out_dir / 'lab_affine_4x3.npy'}")
    print(f"Wrote {args.out_dir / 'lab_affine_corrector.json'}")
    print(f"Wrote {loo_csv}")


if __name__ == "__main__":
    main()
