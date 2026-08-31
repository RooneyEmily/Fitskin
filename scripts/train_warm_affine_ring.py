#!/usr/bin/env python3
"""Chart-free warm-light RGB→XYZ affine from ring F12 torch captures.

No ColorChecker under the F12 ring exists on disk. This fits a 4×3 affine
``[R,G,B,1] → XYZ_scene`` from cheek flash/no-flash reflectance on the
Variable Lighting ring cohort, with targets derived from FitSkin forehead Lab
(D65-referred) and MK350 in-situ F12 white for the forward CAT model used
during training.

Also fits a D65-ring variant (``tier3_affine_d65_ring``) for illuminant-routed
deployment alongside the frozen indoor ``tier3_affine``.

Example::

  python3 scripts/train_warm_affine_ring.py \\
    --ring-csv results/torch_illuminant_ringlight/torch_illuminant_ringlight.csv \\
    --out-dir calibration/tier3_affine_warm
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from delta_e_2000 import delta_e_2000  # noqa: E402
from flash_noflash_spectral import planck_xyz_y1  # noqa: E402
from pipeline.illuminant_estimation import xy_white_y1  # noqa: E402
from scripts.evaluate_pansor20_chartfree_d65 import (  # noqa: E402
    D65,
    apple_face_cheek_masks,
    bradford_cat_matrix,
    extract_zip,
    load_apple_landmarks,
    load_dng_linear,
    match_flash_exposure,
    xyz_to_lab,
)
from scripts.evaluate_ringlight_torch_illuminant import (  # noqa: E402
    default_data_root,
    discover_trials,
    load_booth_fitskin_labs,
    load_ring_illuminant_xy,
)


def lab_to_xyz(lab: np.ndarray, xyzn: np.ndarray = D65) -> np.ndarray:
    lab = np.asarray(lab, dtype=np.float64).reshape(3)
    L, a, b = float(lab[0]), float(lab[1]), float(lab[2])
    fy = (L + 16.0) / 116.0
    fx = fy + a / 500.0
    fz = fy - b / 200.0
    d = 6.0 / 29.0

    def f_inv(t: np.ndarray) -> np.ndarray:
        return np.where(t > d**3, t**3, 3.0 * d * d * (t - 4.0 / 29.0))

    xr, yr, zr = f_inv(fx), f_inv(fy), f_inv(fz)
    return np.array([xr, yr, zr], dtype=np.float64) * np.asarray(xyzn, dtype=np.float64)


def fit_affine_rgb_to_xyz_d65(
    rgb: np.ndarray,
    xyz_d65_target: np.ndarray,
    cat_t: np.ndarray,
    *,
    ridge: float = 0.0,
    M0: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Solve ``(X @ M) @ cat_t = Y`` for 4×3 ``M`` (``cat_t`` = Bradford CAT 3×3)."""
    X = np.asarray(rgb, dtype=np.float64)
    Y = np.asarray(xyz_d65_target, dtype=np.float64)
    cat_t = np.asarray(cat_t, dtype=np.float64)
    if X.ndim == 1:
        X = X.reshape(1, 4)
    if Y.ndim == 1:
        Y = Y.reshape(1, 3)
    # Mc = M @ cat_t  →  Y = X @ Mc
    n = X.shape[0]
    A = np.zeros((n * 3, 12), dtype=np.float64)
    b = Y.reshape(-1)
    for i in range(n):
        x = X[i]
        for ch in range(3):
            row = i * 3 + ch
            for j in range(4):
                for k in range(3):
                    A[row, j * 3 + k] = x[j] * cat_t[k, ch]
    if ridge > 0 and M0 is not None:
        Mc0 = np.asarray(M0, dtype=np.float64) @ cat_t
        A_aug = np.vstack([A, np.sqrt(ridge) * np.eye(12)])
        b_aug = np.concatenate([b, np.sqrt(ridge) * Mc0.reshape(-1)])
        Mc, *_ = np.linalg.lstsq(A_aug, b_aug, rcond=None)
    else:
        Mc, *_ = np.linalg.lstsq(A, b, rcond=None)
    Mc = Mc.reshape(4, 3)
    M = Mc @ np.linalg.inv(cat_t)
    return M


def cheek_rgb_rows(
    zip_path: Path,
    *,
    half_size: bool,
    max_pixels: int,
    rng: np.random.Generator,
) -> np.ndarray:
    tmp = Path(tempfile.mkdtemp(prefix="warm_affine_"))
    try:
        nf, fl, lm_path = extract_zip(zip_path, tmp)
        A0 = load_dng_linear(nf, half_size=half_size, use_camera_wb=False)
        B0 = load_dng_linear(fl, half_size=half_size, use_camera_wb=False)
        if B0.shape != A0.shape:
            B0 = cv2.resize(B0, (A0.shape[1], A0.shape[0]), interpolation=cv2.INTER_AREA)
        lm = load_apple_landmarks(lm_path)
        _, cheek = apple_face_cheek_masks(lm, A0.shape[0], A0.shape[1])
        if int(np.count_nonzero(cheek)) < 50:
            raise RuntimeError("empty cheek mask")
        B0m, _ = match_flash_exposure(A0, B0, cheek)
        R0 = np.sqrt(np.maximum(A0, 0) * np.maximum(B0m, 0) + 1e-8)
        pix = R0[cheek > 0]
        pix = pix[np.all(np.isfinite(pix), axis=1)]
        pix = pix[np.all(pix > 1e-8, axis=1)]
        if len(pix) == 0:
            raise RuntimeError("no valid cheek pixels")
        if len(pix) > max_pixels:
            idx = rng.choice(len(pix), size=max_pixels, replace=False)
            pix = pix[idx]
        ones = np.ones((len(pix), 1), dtype=np.float64)
        return np.hstack([pix.astype(np.float64), ones])
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def collect_rows(
    trials: List[Dict[str, Any]],
    fitskin_map: Dict[str, Dict[str, np.ndarray]],
    *,
    illuminant: str,
    half_size: bool,
    max_pixels: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    rng = np.random.default_rng(seed)
    rgb_rows: List[np.ndarray] = []
    xyz_targets: List[np.ndarray] = []
    person_keys: List[str] = []
    for trial in trials:
        if trial["illuminant"] != illuminant:
            continue
        person = trial["person"]
        ill = trial["illuminant"]
        if person not in fitskin_map or ill not in fitskin_map[person]:
            continue
        fit = fitskin_map[person][ill]
        X = cheek_rgb_rows(
            Path(trial["zip_path"]),
            half_size=half_size,
            max_pixels=max_pixels,
            rng=rng,
        )
        y = lab_to_xyz(fit)
        rgb_rows.append(X)
        xyz_targets.append(np.tile(y.reshape(1, 3), (X.shape[0], 1)))
        person_keys.extend([person] * X.shape[0])
    if not rgb_rows:
        raise ValueError(f"No training rows for illuminant {illuminant}")
    return np.vstack(rgb_rows), np.vstack(xyz_targets), person_keys


def loo_de00(
    trials: List[Dict[str, Any]],
    fitskin_map: Dict[str, Dict[str, np.ndarray]],
    M: np.ndarray,
    xyz_white: np.ndarray,
    *,
    half_size: bool,
    max_pixels: int,
    seed: int,
) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    cat_t = bradford_cat_matrix(xyz_white, D65).T
    people = sorted({t["person"] for t in trials})
    des: List[float] = []
    for person in people:
        sub = [t for t in trials if t["person"] == person]
        ill = sub[0]["illuminant"]
        fit = fitskin_map[person][ill]
        labs: List[np.ndarray] = []
        for trial in sub:
            X = cheek_rgb_rows(
                Path(trial["zip_path"]),
                half_size=half_size,
                max_pixels=max_pixels,
                rng=rng,
            )
            xyz_d65 = (X @ M) @ cat_t
            lab = xyz_to_lab(xyz_d65).mean(axis=0)
            labs.append(lab)
        pred = np.mean(np.stack(labs, axis=0), axis=0)
        des.append(float(delta_e_2000(pred, fit)))
    return {
        "n": len(des),
        "mean_de00": float(np.mean(des)),
        "median_de00": float(np.median(des)),
    }


def train_for_illuminant(
    trials: List[Dict[str, Any]],
    fitskin_map: Dict[str, Dict[str, np.ndarray]],
    ring_xy: Dict[str, Tuple[float, float]],
    *,
    illuminant: str,
    half_size: bool,
    max_pixels: int,
    ridge: float,
    M0: np.ndarray,
    seed: int,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    sub = [t for t in trials if t["illuminant"] == illuminant]
    X, Y, person_keys = collect_rows(
        sub,
        fitskin_map,
        illuminant=illuminant,
        half_size=half_size,
        max_pixels=max_pixels,
        seed=seed,
    )
    xy = ring_xy.get(illuminant, ring_xy["D65"])
    xyz_white = xy_white_y1(xy[0], xy[1])
    cat_t = bradford_cat_matrix(xyz_white, D65).T
    M = fit_affine_rgb_to_xyz_d65(X, Y, cat_t, ridge=ridge, M0=M0)

    people = sorted({t["person"] for t in sub})
    loo_des: List[float] = []
    for holdout in people:
        train_trials = [t for t in sub if t["person"] != holdout]
        Xt, Yt, _ = collect_rows(
            train_trials,
            fitskin_map,
            illuminant=illuminant,
            half_size=half_size,
            max_pixels=max_pixels,
            seed=seed + hash(holdout) % 10000,
        )
        M_loo = fit_affine_rgb_to_xyz_d65(Xt, Yt, cat_t, ridge=ridge, M0=M0)
        test_trials = [t for t in sub if t["person"] == holdout]
        ill = illuminant
        fit = fitskin_map[holdout][ill]
        labs = []
        for trial in test_trials:
            Xp = cheek_rgb_rows(
                Path(trial["zip_path"]),
                half_size=half_size,
                max_pixels=max_pixels,
                rng=np.random.default_rng(seed),
            )
            xyz_d65 = (Xp @ M_loo) @ cat_t
            labs.append(xyz_to_lab(xyz_d65).mean(axis=0))
        pred = np.mean(np.stack(labs, axis=0), axis=0)
        loo_des.append(float(delta_e_2000(pred, fit)))

    meta = {
        "illuminant": illuminant,
        "n_trials": len(sub),
        "n_pixels": int(X.shape[0]),
        "n_people": len(people),
        "mk350_xy": list(xy),
        "ridge": float(ridge),
        "loo_mean_de00": float(np.mean(loo_des)) if loo_des else None,
        "loo_median_de00": float(np.median(loo_des)) if loo_des else None,
        "all_fit_eval": loo_de00(
            sub, fitskin_map, M, xyz_white,
            half_size=half_size, max_pixels=max_pixels, seed=seed,
        ),
    }
    return M, meta


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", type=Path, default=None)
    ap.add_argument("--booth-xlsx", type=Path, default=Path.home() / "Downloads" / "Booth Lighting.xlsx")
    ap.add_argument("--ring-csv", type=Path, default=None, help="Optional; if set, skip re-extracting zips.")
    ap.add_argument("--base-cal-dir", type=Path, default=ROOT / "calibration" / "tier3_affine")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "calibration" / "tier3_affine_warm")
    ap.add_argument("--out-dir-d65", type=Path, default=ROOT / "calibration" / "tier3_affine_d65_ring")
    ap.add_argument("--half-size", action="store_true", default=True)
    ap.add_argument("--max-pixels-per-trial", type=int, default=400)
    ap.add_argument("--ridge", type=float, default=0.01, help="L2 pull toward base tier3 affine.")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from scripts.evaluate_pansor20_chartfree_d65 import load_affine

    M0 = load_affine(args.base_cal_dir)
    data_root = Path(args.data_root or default_data_root()).expanduser().resolve()
    booth_xlsx = Path(args.booth_xlsx).expanduser().resolve()
    fitskin_map = load_booth_fitskin_labs(booth_xlsx)
    ring_xy = load_ring_illuminant_xy(booth_xlsx=booth_xlsx)
    trials = discover_trials(data_root)

    M_warm, warm_meta = train_for_illuminant(
        trials,
        fitskin_map,
        ring_xy,
        illuminant="F12",
        half_size=bool(args.half_size),
        max_pixels=int(args.max_pixels_per_trial),
        ridge=float(args.ridge),
        M0=M0,
        seed=int(args.seed),
    )
    M_d65, d65_meta = train_for_illuminant(
        trials,
        fitskin_map,
        ring_xy,
        illuminant="D65",
        half_size=bool(args.half_size),
        max_pixels=int(args.max_pixels_per_trial),
        ridge=float(args.ridge),
        M0=M0,
        seed=int(args.seed) + 1,
    )

    bundle: Dict[str, Any] = {
        "method": "chart_free_cheek_reflectance_vs_fitskin",
        "base_cal_dir": str(args.base_cal_dir),
        "ridge": float(args.ridge),
        "max_pixels_per_trial": int(args.max_pixels_per_trial),
        "mk350_ring_xy": {k: list(v) for k, v in ring_xy.items()},
        "warm_f12": warm_meta,
        "d65_ring": d65_meta,
        "note": (
            "Warm/d65 ring affines fit cheek R0 → XYZ with MK350 scene white + FitSkin Lab targets. "
            "Not CC-supervised; validate with LOO. Route: F12→M_warm, D65 ring→M_d65, else tier3_affine."
        ),
    }

    for out_dir, M, tag in (
        (args.out_dir, M_warm, "warm_f12"),
        (args.out_dir_d65, M_d65, "d65_ring"),
    ):
        out_dir = Path(out_dir).expanduser().resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        np.save(out_dir / "camera_rgb_to_xyz_affine.npy", M)
        jpath = out_dir / "iphone_calibration_bundle.json"
        payload: Dict[str, Any] = {
            "device_label": f"chart-free ring {tag} affine",
            "matrix_fit": "affine lstsq cheek R0 with MK350 CAT to D65 FitSkin Lab",
            "matrix_affine": True,
            "camera_rgb_to_xyz_affine": M.tolist(),
            "training_meta": bundle,
        }
        jpath.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote {out_dir / 'camera_rgb_to_xyz_affine.npy'}  LOO ΔE≈{bundle[tag]['loo_mean_de00']:.2f}")

    bundle_path = Path(args.out_dir).parent / "tier3_affine_illuminant_routed.json"
    bundle_path.write_text(
        json.dumps(
            {
                **bundle,
                "paths": {
                    "default_cool": str(args.base_cal_dir),
                    "warm_f12": str(args.out_dir),
                    "d65_ring": str(args.out_dir_d65),
                },
                "matrices": {
                    "M_default": M0.tolist(),
                    "M_warm_f12": M_warm.tolist(),
                    "M_d65_ring": M_d65.tolist(),
                },
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"warm_f12": warm_meta, "d65_ring": d65_meta}, indent=2))
    print(f"Wrote {bundle_path}")


if __name__ == "__main__":
    main()
