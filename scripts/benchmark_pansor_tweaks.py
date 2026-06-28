#!/usr/bin/env python3
"""Benchmark Pansor CC-trial tweaks; writes results/pansor_tweaks/summary.csv + summary.json."""
from __future__ import annotations

import csv
import json
import statistics as stats
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import mediapipe as mp
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from chart_cc_fitskin_lib import process_one_dng, silence_stderr  # noqa: E402
from delta_e_2000 import delta_e_2000  # noqa: E402

MANIFEST = ROOT / "data" / "pansor" / "manifest_pansor_fitskin.csv"
OUT_DIR = ROOT / "results" / "pansor_tweaks"


def _de00(lab: Tuple[float, float, float], fit: Tuple[float, float, float]) -> float:
    a = np.array([[[*lab]]], dtype=np.float64)
    b = np.array([[[*fit]]], dtype=np.float64)
    return float(delta_e_2000(a, b)[0, 0])


def _load_cc_rows() -> List[dict]:
    with MANIFEST.open(newline="", encoding="utf-8") as f:
        return [
            r
            for r in csv.DictReader(f)
            if r.get("condition_code") == "CC" and r.get("include_in_eval", "yes") == "yes"
        ]


def _run_chart_cc_variants(rows: List[dict]) -> Dict[str, Dict[str, float]]:
    configs = [
        ("cc_cheek_white", dict(roi="cheek", chart_gray_wb_from="white")),
        ("cc_mesh_white", dict(roi="mesh", chart_gray_wb_from="white")),
        ("cc_cheek_neutral_wb", dict(roi="cheek", chart_gray_wb_from="neutral_column_mean")),
        ("cc_mesh_neutral_wb", dict(roi="mesh", chart_gray_wb_from="neutral_column_mean")),
        ("cc_cheek_affine", dict(roi="cheek", chart_gray_wb_from="white", affine=True)),
        ("cc_mesh_affine", dict(roi="mesh", chart_gray_wb_from="white", affine=True)),
    ]
    out: Dict[str, Dict[str, float]] = {name: {} for name, _ in configs}
    mp_fm = mp.solutions.face_mesh
    with silence_stderr():
        with mp_fm.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
        ) as fm:
            for row in rows:
                sid = row["subject_id"]
                fit = (
                    float(row["fitskin_cheek_L"]),
                    float(row["fitskin_cheek_a"]),
                    float(row["fitskin_cheek_b"]),
                )
                path = Path(row["path_noflash"])
                for name, kw in configs:
                    r = process_one_dng(path, fm, **kw)
                    if not r.get("chart_ok") or r.get("status") != "ok":
                        out[name][sid] = float("nan")
                    else:
                        lab = (r["pipeline_L"], r["pipeline_a"], r["pipeline_b"])
                        out[name][sid] = _de00(lab, fit)
    return out


def _median(vals: List[float]) -> float:
    v = [x for x in vals if np.isfinite(x)]
    return float(stats.median(v)) if v else float("nan")


def _mean(vals: List[float]) -> float:
    v = [x for x in vals if np.isfinite(x)]
    return float(stats.mean(v)) if v else float("nan")


def main() -> None:
    rows = _load_cc_rows()
    sids = [r["subject_id"] for r in rows]
    chart = _run_chart_cc_variants(rows)

    # Load pre-run FNF CSVs if present
    fnf_runs = {
        "fnf_mesh": ROOT / "results" / "pansor_fnf_mesh" / "flash_noflash_skin_lab.csv",
        "fnf_booth_cat": ROOT / "results" / "pansor_fnf_booth_cat" / "flash_noflash_skin_lab.csv",
        "fnf_scr_awb": ROOT / "results" / "pansor_fnf_scr_awb" / "flash_noflash_skin_lab.csv",
    }
    fnf: Dict[str, Dict[str, float]] = {}
    for name, path in fnf_runs.items():
        if not path.is_file():
            continue
        col = "scr_awb_wb_cheek_de00" if "scr" in name else "reflectance_cheek_de00"
        with path.open(newline="", encoding="utf-8") as f:
            fnf[name] = {
                r["subject_id"]: float(r[col])
                for r in csv.DictReader(f)
                if col in r and r[col]
            }

    # Oracle: P1 -> best of fnf_mesh / cc_cheek; P2 -> cc_mesh_white
    oracle: Dict[str, float] = {}
    for sid in sids:
        p1 = sid.startswith("P1")
        opts = []
        if "fnf_mesh" in fnf:
            opts.append(fnf["fnf_mesh"].get(sid, float("nan")))
        opts.append(chart["cc_cheek_white"].get(sid, float("nan")))
        if not p1:
            opts.append(chart["cc_mesh_white"].get(sid, float("nan")))
        opts = [x for x in opts if np.isfinite(x)]
        oracle[sid] = min(opts) if opts else float("nan")

    # Exclude P1_CC_T1 (dark outlier)
    keep = [s for s in sids if s != "P1_CC_T1"]

    all_methods: Dict[str, Dict[str, float]] = {**chart, **fnf, "oracle_mixed_roi": oracle}

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUT_DIR / "summary.csv"
    fieldnames = ["subject_id", *sorted(all_methods.keys())]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for sid in sids:
            rec = {"subject_id": sid}
            for m in all_methods:
                rec[m] = all_methods[m].get(sid, "")
            w.writerow(rec)

    summary: Dict[str, Any] = {
        "cohort": "Pansor ColorChecker DNG (n=6)",
        "fitskin_reference": "May 20 median cheek (no same-session June 16 export found)",
        "methods": {},
    }
    for name, per_sid in all_methods.items():
        vals = [per_sid[s] for s in sids]
        vals_excl = [per_sid[s] for s in keep]
        summary["methods"][name] = {
            "median_de00": _median(vals),
            "mean_de00": _mean(vals),
            "median_excl_P1_T1": _median(vals_excl),
            "per_trial": per_sid,
        }

    best = min(summary["methods"].items(), key=lambda kv: kv[1]["median_de00"])
    summary["best_median"] = {"method": best[0], **{k: best[1][k] for k in ("median_de00", "mean_de00")}}

    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(f"Wrote {csv_path}")
    print(f"Wrote {OUT_DIR / 'summary.json'}")
    print("\nMethod                          median  mean   excl P1_T1")
    for name in sorted(all_methods.keys()):
        m = summary["methods"][name]
        print(
            f"  {name:30s}  {m['median_de00']:5.2f}  {m['mean_de00']:5.2f}  {m['median_excl_P1_T1']:5.2f}"
        )
    print(f"\nBest median: {summary['best_median']['method']} ({summary['best_median']['median_de00']:.2f})")


if __name__ == "__main__":
    main()
