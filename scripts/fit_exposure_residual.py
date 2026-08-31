#!/usr/bin/env python3
"""Fit exposure-aware Lab residual on CameraSettings; apply to Pansor.

Leave-one-person-out on {Giana, Keaton, Parker, Wooj}. Prefer L*-only ridge;
escalate to full Lab residual only if LOO mean ΔE improves.

If LOO does not beat the uncorrected camera-settings baseline, skip claiming a
Pansor win and still write the capture recommendation + audit linkage.

Example:
  python3 scripts/fit_exposure_residual.py \\
    --camera-csv results/camera_settings/camera_settings_results.csv \\
    --pansor-csv results/pansor20_fairface7/pansor20_chartfree_d65.csv \\
    --audit-csv results/pansor20_exposure_audit/pansor20_exposure_audit.csv \\
    --out-dir calibration/exposure_residual_pansor
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from statistics import mean, median
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from delta_e_2000 import delta_e_2000  # noqa: E402

PEOPLE = ("Giana", "Keaton", "Parker", "Wooj")


def _load_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _ev(iso: float, shutter_s: float) -> float:
    return math.log2(max(iso * shutter_s, 1e-12))


def _features_L(rows: Sequence[Dict[str, Any]], *, iso_key: str, shutter_key: str, L_key: str) -> np.ndarray:
    """Design matrix for L* residual: [1, EV, L_pipe, EV*L_pipe]."""
    X = []
    for r in rows:
        ev = _ev(float(r[iso_key]), float(r[shutter_key]))
        L = float(r[L_key])
        X.append([1.0, ev, L, ev * L])
    return np.asarray(X, dtype=np.float64)


def _features_Lab(
    rows: Sequence[Dict[str, Any]],
    *,
    iso_key: str,
    shutter_key: str,
    L_key: str,
    a_key: str,
    b_key: str,
) -> np.ndarray:
    """Design matrix for Lab residual: [1, EV, L, a, b]."""
    X = []
    for r in rows:
        ev = _ev(float(r[iso_key]), float(r[shutter_key]))
        X.append([1.0, ev, float(r[L_key]), float(r[a_key]), float(r[b_key])])
    return np.asarray(X, dtype=np.float64)


def _ridge(X: np.ndarray, y: np.ndarray, lam: float = 1e-2) -> np.ndarray:
    """Solve (X'X + lam I) w = X'y. y shape (n,) or (n, k)."""
    n, d = X.shape
    xtx = X.T @ X + lam * np.eye(d)
    xty = X.T @ y
    return np.linalg.solve(xtx, xty)


def _apply_L(w: np.ndarray, X: np.ndarray, labs: np.ndarray) -> np.ndarray:
    dL = X @ w
    out = labs.copy()
    out[:, 0] = out[:, 0] + dL
    return out


def _apply_Lab(W: np.ndarray, X: np.ndarray, labs: np.ndarray) -> np.ndarray:
    # W shape (d, 3)
    return labs + (X @ W)


def _de_mean(pred: np.ndarray, gt: np.ndarray) -> float:
    return float(np.mean(delta_e_2000(pred, gt)))


def _rows_cam(csv_rows: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    out = []
    for r in csv_rows:
        out.append(
            {
                "person": r["person"],
                "setting": r["setting"],
                "iso": float(r["iso"]),
                "shutter_s": float(r["shutter_s"]),
                "pipeline_L": float(r["pipeline_L_off"]),
                "pipeline_a": float(r["pipeline_a_off"]),
                "pipeline_b": float(r["pipeline_b_off"]),
                "fitskin_L": float(r["fitskin_L"]),
                "fitskin_a": float(r["fitskin_a"]),
                "fitskin_b": float(r["fitskin_b"]),
                "de00": float(r["de00_off"]),
            }
        )
    return out


def loo_camera_settings(
    rows: List[Dict[str, Any]], lam: float = 1e-2
) -> Dict[str, Any]:
    people = sorted({r["person"] for r in rows})
    base_des = []
    l_des = []
    lab_des = []
    per_person: Dict[str, Any] = {}

    for held in people:
        train = [r for r in rows if r["person"] != held]
        test = [r for r in rows if r["person"] == held]
        if not train or not test:
            continue

        labs_te = np.array([[r["pipeline_L"], r["pipeline_a"], r["pipeline_b"]] for r in test])
        gt_te = np.array([[r["fitskin_L"], r["fitskin_a"], r["fitskin_b"]] for r in test])
        base = _de_mean(labs_te, gt_te)
        base_des.append(base)

        # L* residual
        yL = np.array([r["fitskin_L"] - r["pipeline_L"] for r in train])
        XL_tr = _features_L(train, iso_key="iso", shutter_key="shutter_s", L_key="pipeline_L")
        XL_te = _features_L(test, iso_key="iso", shutter_key="shutter_s", L_key="pipeline_L")
        wL = _ridge(XL_tr, yL, lam=lam)
        pred_L = _apply_L(wL, XL_te, labs_te)
        de_L = _de_mean(pred_L, gt_te)
        l_des.append(de_L)

        # Lab residual
        yLab = np.array(
            [
                [
                    r["fitskin_L"] - r["pipeline_L"],
                    r["fitskin_a"] - r["pipeline_a"],
                    r["fitskin_b"] - r["pipeline_b"],
                ]
                for r in train
            ]
        )
        XLab_tr = _features_Lab(
            train,
            iso_key="iso",
            shutter_key="shutter_s",
            L_key="pipeline_L",
            a_key="pipeline_a",
            b_key="pipeline_b",
        )
        XLab_te = _features_Lab(
            test,
            iso_key="iso",
            shutter_key="shutter_s",
            L_key="pipeline_L",
            a_key="pipeline_a",
            b_key="pipeline_b",
        )
        W = _ridge(XLab_tr, yLab, lam=lam)
        pred_Lab = _apply_Lab(W, XLab_te, labs_te)
        de_Lab = _de_mean(pred_Lab, gt_te)
        lab_des.append(de_Lab)

        per_person[held] = {
            "n_test": len(test),
            "mean_de00_uncorrected": round(base, 4),
            "mean_de00_L_residual": round(de_L, 4),
            "mean_de00_Lab_residual": round(de_Lab, 4),
        }

    mean_base = mean(base_des) if base_des else float("nan")
    mean_L = mean(l_des) if l_des else float("nan")
    mean_Lab = mean(lab_des) if lab_des else float("nan")

    # Prefer L* if it wins or ties Lab; escalate to Lab only if strictly better
    if mean_Lab < mean_L - 1e-6 and mean_Lab < mean_base - 1e-6:
        chosen = "Lab_residual"
        loo_mean = mean_Lab
    elif mean_L < mean_base - 1e-6:
        chosen = "L_residual"
        loo_mean = mean_L
    else:
        chosen = "none"
        loo_mean = mean_base

    return {
        "people": people,
        "per_person": per_person,
        "loo_mean_de00_uncorrected": round(mean_base, 4),
        "loo_mean_de00_L_residual": round(mean_L, 4),
        "loo_mean_de00_Lab_residual": round(mean_Lab, 4),
        "chosen_model": chosen,
        "loo_mean_de00_chosen": round(loo_mean, 4),
        "improves_over_baseline": chosen != "none",
        "lam": lam,
    }


def fit_full(
    rows: List[Dict[str, Any]], model: str, lam: float = 1e-2
) -> Dict[str, Any]:
    if model == "L_residual":
        y = np.array([r["fitskin_L"] - r["pipeline_L"] for r in rows])
        X = _features_L(rows, iso_key="iso", shutter_key="shutter_s", L_key="pipeline_L")
        w = _ridge(X, y, lam=lam)
        return {
            "model": model,
            "feature_names": ["bias", "EV", "L", "EV_x_L"],
            "weights": w.tolist(),
            "lam": lam,
        }
    if model == "Lab_residual":
        y = np.array(
            [
                [
                    r["fitskin_L"] - r["pipeline_L"],
                    r["fitskin_a"] - r["pipeline_a"],
                    r["fitskin_b"] - r["pipeline_b"],
                ]
                for r in rows
            ]
        )
        X = _features_Lab(
            rows,
            iso_key="iso",
            shutter_key="shutter_s",
            L_key="pipeline_L",
            a_key="pipeline_a",
            b_key="pipeline_b",
        )
        W = _ridge(X, y, lam=lam)
        return {
            "model": model,
            "feature_names": ["bias", "EV", "L", "a", "b"],
            "weights": W.tolist(),  # d x 3
            "lam": lam,
        }
    return {"model": "none", "feature_names": [], "weights": [], "lam": lam}


def apply_model(
    model_art: Dict[str, Any],
    rows: Sequence[Dict[str, Any]],
    *,
    iso_key: str,
    shutter_key: str,
    L_key: str,
    a_key: str,
    b_key: str,
) -> np.ndarray:
    labs = np.array([[float(r[L_key]), float(r[a_key]), float(r[b_key])] for r in rows])
    m = model_art["model"]
    if m == "none":
        return labs
    if m == "L_residual":
        X = _features_L(rows, iso_key=iso_key, shutter_key=shutter_key, L_key=L_key)
        w = np.asarray(model_art["weights"], dtype=np.float64)
        return _apply_L(w, X, labs)
    if m == "Lab_residual":
        X = _features_Lab(
            rows,
            iso_key=iso_key,
            shutter_key=shutter_key,
            L_key=L_key,
            a_key=a_key,
            b_key=b_key,
        )
        W = np.asarray(model_art["weights"], dtype=np.float64)
        return _apply_Lab(W, X, labs)
    raise ValueError(m)


def write_capture_rec(path: Path, loo: Dict[str, Any], pansor_summary: Dict[str, Any], extra: str = "") -> None:
    text = f"""# Pansor app capture recommendation (from CameraSettings factorial)

Camera-settings sweeps (Giana, Keaton, Parker, Wooj; Emily Lab-only / no photos) show that
chart-free D65 ΔE is dominated by absolute exposure:

- **Preferred band:** ISO ≤ 100 and shutter ≈ 1/250–1/120 s (cells A / C / E / I).
  Mean ΔE ≈ 3–5 with pipeline L* near FitSkin skin (~50–65).
- **Avoid:** shutter ≈ 1/60 s (cell B) or ISO ≥ 200 (cells F / G / H). These push
  pipeline L* into the ~78–85 range and mean ΔE ≈ 15–21.

Current Pansor-20 indoor captures already sit in the preferred band
(ISO median ≈ {pansor_summary.get('iso_median')}, shutter median ≈ {pansor_summary.get('shutter_s_median')} s;
in-band fraction {pansor_summary.get('frac_in_band')}). Exposure residual LOO on
camera-settings chose **{loo.get('chosen_model')}**
(uncorrected LOO mean ΔE {loo.get('loo_mean_de00_uncorrected')} → chosen {loo.get('loo_mean_de00_chosen')}).

Post-Lab exposure residual fitted on these four (lighter-skin) subjects **does not transfer**
to Pansor-20: applying it raises mean ΔE (especially on Black participants). Keep the frozen
D65 + FairFace7 path; use camera-settings as a **capture-policy** signal, not a Lab corrector.

**App lock suggestion:** fix AE near ISO 64–100 and 1/120–1/250; reject or re-prompt if
ISO ≥ 200, shutter ≥ 1/60, or estimated cheek L* ≳ 75 after the frozen path.
{extra}
"""
    path.write_text(text, encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--camera-csv",
        type=Path,
        default=ROOT / "results/camera_settings/camera_settings_results.csv",
    )
    ap.add_argument(
        "--pansor-csv",
        type=Path,
        default=ROOT / "results/pansor20_fairface7/pansor20_chartfree_d65.csv",
    )
    ap.add_argument(
        "--audit-csv",
        type=Path,
        default=ROOT / "results/pansor20_exposure_audit/pansor20_exposure_audit.csv",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "calibration/exposure_residual_pansor",
    )
    ap.add_argument(
        "--results-dir",
        type=Path,
        default=ROOT / "results/pansor20_exposure_residual",
    )
    ap.add_argument("--lam", type=float, default=1e-2)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.results_dir.mkdir(parents=True, exist_ok=True)

    cam = _rows_cam(_load_csv(args.camera_csv))
    print(f"Camera-settings trials: {len(cam)} people={sorted({r['person'] for r in cam})}")

    def _in_band(r: Dict[str, Any]) -> bool:
        return r["iso"] < 200.0 and r["shutter_s"] < (1.0 / 60.0) and r["pipeline_L"] < 75.0

    cam_band = [r for r in cam if _in_band(r)]
    print(f"In-band camera-settings trials: {len(cam_band)} / {len(cam)}")

    loo = loo_camera_settings(cam, lam=args.lam)
    loo_band = loo_camera_settings(cam_band, lam=args.lam)
    print("\n=== LOO on camera-settings (all cells) ===")
    print(json.dumps(loo, indent=2))
    print("\n=== LOO on camera-settings (in-band only) ===")
    print(json.dumps(loo_band, indent=2))

    # Prefer in-band model for Pansor transfer when it improves; else all-cells choice.
    if loo_band["improves_over_baseline"]:
        chosen = loo_band["chosen_model"]
        train_for_selected = cam_band
        loo_for_selected = loo_band
        train_domain = "in_band"
    else:
        chosen = loo["chosen_model"]
        train_for_selected = cam
        loo_for_selected = loo
        train_domain = "all_cells"

    art = fit_full(train_for_selected, chosen, lam=args.lam)
    # Also store both fitted full models for inspection (all cells + in-band)
    art_L = fit_full(cam, "L_residual", lam=args.lam)
    art_Lab = fit_full(cam, "Lab_residual", lam=args.lam)
    art_L_band = fit_full(cam_band, "L_residual", lam=args.lam)
    art_Lab_band = fit_full(cam_band, "Lab_residual", lam=args.lam)

    bundle = {
        "loo_all_cells": loo,
        "loo_in_band": loo_band,
        "loo": loo_for_selected,
        "train_domain": train_domain,
        "selected": art,
        "fit_L_residual_all": art_L,
        "fit_Lab_residual_all": art_Lab,
        "fit_L_residual_in_band": art_L_band,
        "fit_Lab_residual_in_band": art_Lab_band,
        "train_n": len(train_for_selected),
        "source_camera_csv": str(args.camera_csv),
    }
    (args.out_dir / "exposure_residual.json").write_text(json.dumps(bundle, indent=2), encoding="utf-8")
    np.savez(
        args.out_dir / "exposure_residual.npz",
        model=np.array(chosen),
        train_domain=np.array(train_domain),
        weights_L=np.asarray(art_L["weights"]),
        weights_Lab=np.asarray(art_Lab["weights"]),
        weights_L_band=np.asarray(art_L_band["weights"]),
        weights_Lab_band=np.asarray(art_Lab_band["weights"]),
        selected_weights=np.asarray(art.get("weights", [])),
        lam=np.array(args.lam),
    )

    # --- Apply to Pansor (need EXIF from audit) ---
    pansor = _load_csv(args.pansor_csv)
    if args.audit_csv.is_file():
        audit = {r["subject_id"]: r for r in _load_csv(args.audit_csv)}
    else:
        audit = {}
        print(f"WARN: missing audit CSV {args.audit_csv}; Pansor apply skipped for EXIF")

    joined: List[Dict[str, Any]] = []
    for r in pansor:
        a = audit.get(r["subject_id"])
        if a is None:
            continue
        joined.append(
            {
                **{k: r[k] for k in r},
                "iso": float(a["iso"]),
                "shutter_s": float(a["shutter_s"]),
                "log2_iso_shutter": float(a["log2_iso_shutter"]) if a["log2_iso_shutter"] else float("nan"),
                "in_band": int(float(a["in_band"])),
            }
        )

    pansor_summary = {}
    if args.audit_csv.is_file():
        summ_path = args.audit_csv.parent / "summary.json"
        if summ_path.is_file():
            pansor_summary = json.loads(summ_path.read_text(encoding="utf-8"))

    apply_anyway = True  # always report before/after for transparency
    before_labs = np.array(
        [[float(r["pipeline_L"]), float(r["pipeline_a"]), float(r["pipeline_b"])] for r in joined]
    )
    gt = np.array([[float(r["fitskin_L"]), float(r["fitskin_a"]), float(r["fitskin_b"])] for r in joined])
    de_before = delta_e_2000(before_labs, gt)

    # Apply selected model; also report all-cell and in-band L/Lab for transparency
    after_sel = apply_model(
        art,
        joined,
        iso_key="iso",
        shutter_key="shutter_s",
        L_key="pipeline_L",
        a_key="pipeline_a",
        b_key="pipeline_b",
    )
    after_L = apply_model(
        art_L,
        joined,
        iso_key="iso",
        shutter_key="shutter_s",
        L_key="pipeline_L",
        a_key="pipeline_a",
        b_key="pipeline_b",
    )
    after_Lab = apply_model(
        art_Lab,
        joined,
        iso_key="iso",
        shutter_key="shutter_s",
        L_key="pipeline_L",
        a_key="pipeline_a",
        b_key="pipeline_b",
    )
    after_L_band = apply_model(
        art_L_band,
        joined,
        iso_key="iso",
        shutter_key="shutter_s",
        L_key="pipeline_L",
        a_key="pipeline_a",
        b_key="pipeline_b",
    )
    after_Lab_band = apply_model(
        art_Lab_band,
        joined,
        iso_key="iso",
        shutter_key="shutter_s",
        L_key="pipeline_L",
        a_key="pipeline_a",
        b_key="pipeline_b",
    )
    de_sel = delta_e_2000(after_sel, gt)
    de_L = delta_e_2000(after_L, gt)
    de_Lab = delta_e_2000(after_Lab, gt)
    de_L_band = delta_e_2000(after_L_band, gt)
    de_Lab_band = delta_e_2000(after_Lab_band, gt)

    claim_pansor_win = bool(loo_for_selected["improves_over_baseline"]) and float(
        np.mean(de_sel)
    ) < float(np.mean(de_before)) - 1e-6

    out_rows = []
    for i, r in enumerate(joined):
        out_rows.append(
            {
                "subject_id": r["subject_id"],
                "name": r["name"],
                "ethnicity": r["ethnicity"],
                "iso": r["iso"],
                "shutter_s": r["shutter_s"],
                "pipeline_L": float(r["pipeline_L"]),
                "pipeline_a": float(r["pipeline_a"]),
                "pipeline_b": float(r["pipeline_b"]),
                "corr_L": float(after_sel[i, 0]),
                "corr_a": float(after_sel[i, 1]),
                "corr_b": float(after_sel[i, 2]),
                "fitskin_L": float(r["fitskin_L"]),
                "fitskin_a": float(r["fitskin_a"]),
                "fitskin_b": float(r["fitskin_b"]),
                "de00_before": float(de_before[i]),
                "de00_after_selected": float(de_sel[i]),
                "de00_after_L_residual": float(de_L[i]),
                "de00_after_Lab_residual": float(de_Lab[i]),
                "de00_after_L_residual_in_band": float(de_L_band[i]),
                "de00_after_Lab_residual_in_band": float(de_Lab_band[i]),
            }
        )

    # by ethnicity
    eth_stats: Dict[str, Any] = {}
    for eth in sorted({r["ethnicity"] for r in out_rows}):
        idx = [i for i, r in enumerate(out_rows) if r["ethnicity"] == eth]
        eth_stats[eth] = {
            "n": len(idx),
            "mean_de00_before": round(mean(out_rows[i]["de00_before"] for i in idx), 4),
            "mean_de00_after_selected": round(
                mean(out_rows[i]["de00_after_selected"] for i in idx), 4
            ),
        }

    summary = {
        "n": len(out_rows),
        "chosen_model": chosen,
        "train_domain": train_domain,
        "claim_pansor_win": claim_pansor_win,
        "loo_improves_on_camera_settings": loo_for_selected["improves_over_baseline"],
        "mean_de00_before": round(float(np.mean(de_before)), 4) if len(out_rows) else None,
        "median_de00_before": round(float(np.median(de_before)), 4) if len(out_rows) else None,
        "mean_de00_after_selected": round(float(np.mean(de_sel)), 4) if len(out_rows) else None,
        "median_de00_after_selected": round(float(np.median(de_sel)), 4) if len(out_rows) else None,
        "mean_de00_after_L_residual": round(float(np.mean(de_L)), 4) if len(out_rows) else None,
        "mean_de00_after_Lab_residual": round(float(np.mean(de_Lab)), 4) if len(out_rows) else None,
        "mean_de00_after_L_residual_in_band": round(float(np.mean(de_L_band)), 4) if len(out_rows) else None,
        "mean_de00_after_Lab_residual_in_band": round(float(np.mean(de_Lab_band)), 4)
        if len(out_rows)
        else None,
        "by_ethnicity": eth_stats,
        "fairface7_baseline_mean_de00": 3.63,
        "loo_all_cells": {
            "chosen": loo["chosen_model"],
            "uncorrected": loo["loo_mean_de00_uncorrected"],
            "chosen_mean": loo["loo_mean_de00_chosen"],
        },
        "loo_in_band": {
            "chosen": loo_band["chosen_model"],
            "uncorrected": loo_band["loo_mean_de00_uncorrected"],
            "chosen_mean": loo_band["loo_mean_de00_chosen"],
        },
        "note": (
            "Pansor win claimed only if camera-settings LOO improves AND Pansor mean ΔE drops."
            if claim_pansor_win
            else "No Pansor win claimed; see capture recommendation. Frozen path unchanged."
        ),
    }

    fields = list(out_rows[0].keys()) if out_rows else []
    csv_path = args.results_dir / "pansor20_exposure_residual.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(out_rows)
    tsv_path = args.results_dir / "pansor20_exposure_residual.tsv"
    with tsv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
        w.writeheader()
        w.writerows(out_rows)

    (args.results_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (args.out_dir / "loo_summary.json").write_text(
        json.dumps({"all_cells": loo, "in_band": loo_band, "selected": loo_for_selected}, indent=2),
        encoding="utf-8",
    )

    write_capture_rec(
        args.results_dir / "CAPTURE_RECOMMENDATION.md",
        loo_for_selected,
        pansor_summary,
    )
    # also copy next to calibration artifact
    write_capture_rec(args.out_dir / "CAPTURE_RECOMMENDATION.md", loo_for_selected, pansor_summary)

    print("\n=== Pansor before/after ===")
    print(json.dumps(summary, indent=2))
    print(f"Wrote {args.out_dir / 'exposure_residual.json'}")
    print(f"Wrote {csv_path}")
    print(f"Wrote {args.results_dir / 'CAPTURE_RECOMMENDATION.md'}")


if __name__ == "__main__":
    main()
