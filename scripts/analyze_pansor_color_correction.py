#!/usr/bin/env python3
"""
Post-hoc analysis of Pansor bag vs ColorChecker color correction (from pansor_ablation.csv).

Reports:
  - L* vs chroma error decomposition (|ΔL*|, |Δa*|, |Δb*|)
  - Within-subject repeatability (T1–T3 SD / CV of ΔE₀₀)
  - Correction hit-rate: fraction of trials where corrected ΔE < baseline none
  - P1 vs P2 stratification
  - Quality covariates vs correction gain (patch_de_ab, chart area, bag n_white)

Writes JSON summary + CSV tables under results/pansor_ablation/analysis/
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean, median, pstdev
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CSV = ROOT / "results" / "pansor_ablation" / "pansor_ablation.csv"
DEFAULT_OUT = ROOT / "results" / "pansor_ablation" / "analysis"

# Production mode from evaluate_pansor_ablation.py (best bag chromatic correction).
PRODUCTION_BAG_MODE = "cat02_bag"

BAG_CHROMATIC = (PRODUCTION_BAG_MODE,)
BAG_EXPOSURE: Tuple[str, ...] = ()
CC_CHROMATIC: Tuple[str, ...] = ()
CC_EXPOSURE: Tuple[str, ...] = ()

COLOR_CORRECTION_PAIRS = [
    ("none", "none", "baseline"),
    (PRODUCTION_BAG_MODE, "none", "production_bag_vs_cc_baseline"),
]

PARTICIPANT_ORDER = ("P1", "P2")


def _f(v: Any) -> float:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return float("nan")
    return x if math.isfinite(x) else float("nan")


def _participant_key(subject_id: str) -> str:
    s = str(subject_id or "")
    if s.startswith("P1"):
        return "P1"
    if s.startswith("P2"):
        return "P2"
    return s.split("_")[0] if "_" in s else s


def _load_rows(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            de = _f(row.get("reflectance_cheek_de00"))
            if not math.isfinite(de):
                continue
            row["_de"] = de
            row["_dL"] = _f(row.get("delta_L"))
            row["_da"] = _f(row.get("delta_a"))
            row["_db"] = _f(row.get("delta_b"))
            row["_pk"] = _participant_key(row.get("subject_id", ""))
            row["_trial"] = str(row.get("trial", ""))
            rows.append(row)
    return rows


def _filter(
    rows: Sequence[Dict[str, Any]],
    *,
    condition: Optional[str] = None,
    mode: Optional[str] = None,
    pk: Optional[str] = None,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for r in rows:
        if condition is not None and r.get("condition") != condition:
            continue
        if mode is not None and r.get("anchor_mode") != mode:
            continue
        if pk is not None and r.get("_pk") != pk:
            continue
        out.append(r)
    return out


def _lab_delta_stats(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    if not rows:
        return {}
    de = [r["_de"] for r in rows]
    return {
        "n": len(rows),
        "mean_de00": float(mean(de)),
        "median_de00": float(median(de)),
        "std_de00": float(pstdev(de)) if len(de) > 1 else 0.0,
        "mean_abs_delta_L": float(mean(abs(r["_dL"]) for r in rows)),
        "mean_abs_delta_a": float(mean(abs(r["_da"]) for r in rows)),
        "mean_abs_delta_b": float(mean(abs(r["_db"]) for r in rows)),
        "mean_delta_L": float(mean(r["_dL"] for r in rows)),
        "mean_delta_a": float(mean(r["_da"] for r in rows)),
        "mean_delta_b": float(mean(r["_db"] for r in rows)),
    }


def _repeatability(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Per (participant, condition, mode): SD and CV of ΔE across trials 1–3."""
    buckets: Dict[Tuple[str, str, str], List[float]] = {}
    for r in rows:
        key = (str(r["_pk"]), str(r.get("condition", "")), str(r.get("anchor_mode", "")))
        buckets.setdefault(key, []).append(r["_de"])

    out: List[Dict[str, Any]] = []
    for (pk, cond, mode), vals in sorted(buckets.items()):
        if len(vals) < 2:
            continue
        m = float(mean(vals))
        sd = float(pstdev(vals))
        out.append(
            {
                "participant": pk,
                "condition": cond,
                "anchor_mode": mode,
                "n_trials": len(vals),
                "mean_de00": m,
                "sd_de00": sd,
                "cv_de00": float(sd / m) if m > 1e-6 else float("nan"),
                "trial_de00": vals,
            }
        )
    return out


def _none_lookup(rows: Sequence[Dict[str, Any]], condition: str) -> Dict[str, float]:
    return {
        str(r["trial_id"]): r["_de"]
        for r in rows
        if r.get("condition") == condition and r.get("anchor_mode") == "none"
    }


def _correction_gains(
    rows: Sequence[Dict[str, Any]],
    *,
    condition: str,
    corrected_modes: Sequence[str],
) -> List[Dict[str, Any]]:
    """Per trial: gain = ΔE_none − ΔE_corrected (positive = correction helped)."""
    none = _none_lookup(rows, condition)
    out: List[Dict[str, Any]] = []
    for mode in corrected_modes:
        for r in _filter(rows, condition=condition, mode=mode):
            tid = str(r["trial_id"])
            base = none.get(tid)
            if base is None or not math.isfinite(base):
                continue
            gain = float(base - r["_de"])
            out.append(
                {
                    "trial_id": tid,
                    "participant": r["_pk"],
                    "condition": condition,
                    "anchor_mode": mode,
                    "de_none": base,
                    "de_corrected": r["_de"],
                    "gain_de00": gain,
                    "helped": gain > 0,
                    "helped_margin_0p5": gain > 0.5,
                    "patch_de_ab_mean": _f(r.get("patch_de_ab_mean")),
                    "chart_area_fraction": _f(r.get("chart_area_fraction")),
                    "bag_n_white": _f(r.get("bag_white_flash_aligned_n_white")),
                    "bag_wb_ratio": _f(r.get("bag_white_flash_aligned_wb_ratio")),
                    "bag_cct_k": _f(r.get("bag_white_flash_aligned_cct_estimate")),
                }
            )
    return out


def _summarize_gains(gains: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    if not gains:
        return {}
    g = [float(x["gain_de00"]) for x in gains]
    helped = sum(1 for x in gains if x["helped"])
    helped_05 = sum(1 for x in gains if x["helped_margin_0p5"])
    return {
        "n": len(gains),
        "mean_gain_de00": float(mean(g)),
        "median_gain_de00": float(median(g)),
        "hit_rate": float(helped / len(gains)),
        "hit_rate_margin_0p5": float(helped_05 / len(gains)),
        "n_helped": helped,
        "n_hurt": len(gains) - helped,
    }


def _gains_by_participant(gains: Sequence[Dict[str, Any]], mode: str) -> Dict[str, Any]:
    sub = [g for g in gains if g["anchor_mode"] == mode]
    out: Dict[str, Any] = {"all": _summarize_gains(sub)}
    for pk in PARTICIPANT_ORDER:
        out[pk] = _summarize_gains([g for g in sub if g["participant"] == pk])
    return out


def _linreg(x: Sequence[float], y: Sequence[float]) -> Dict[str, Any]:
    xv = np.asarray(x, dtype=np.float64)
    yv = np.asarray(y, dtype=np.float64)
    m = np.isfinite(xv) & np.isfinite(yv)
    if int(m.sum()) < 3:
        return {"n": int(m.sum()), "slope": float("nan"), "intercept": float("nan"), "r": float("nan")}
    xv, yv = xv[m], yv[m]
    if float(np.std(xv)) < 1e-12:
        return {"n": len(xv), "slope": float("nan"), "intercept": float("nan"), "r": float("nan")}
    slope, intercept = np.polyfit(xv, yv, 1)
    r = float(np.corrcoef(xv, yv)[0, 1])
    return {
        "n": len(xv),
        "slope": float(slope),
        "intercept": float(intercept),
        "r": r,
    }


def _quality_regressions(gains: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    cc = [g for g in gains if g["condition"] == "Color Checker" and g["anchor_mode"] == "cc_twopoint"]
    if len(cc) >= 3:
        out["cc_twopoint_gain_vs_patch_de"] = _linreg(
            [g["patch_de_ab_mean"] for g in cc],
            [g["gain_de00"] for g in cc],
        )
        out["cc_twopoint_gain_vs_chart_area"] = _linreg(
            [g["chart_area_fraction"] for g in cc],
            [g["gain_de00"] for g in cc],
        )
    bag = [g for g in gains if g["condition"] == "Sephora Bag" and g["anchor_mode"] == "cat02_bag"]
    if bag:
        out["cat02_bag_gain_vs_n_white"] = _linreg(
            [g["bag_n_white"] for g in bag],
            [g["gain_de00"] for g in bag],
        )
        out["cat02_bag_gain_vs_wb_ratio"] = _linreg(
            [g["bag_wb_ratio"] for g in bag],
            [g["gain_de00"] for g in bag],
        )
    return out


def _write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    if not rows:
        return
    keys = sorted({k for r in rows for k in r.keys()})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            row = dict(r)
            if isinstance(row.get("trial_de00"), list):
                row["trial_de00"] = json.dumps(row["trial_de00"])
            w.writerow(row)


def _try_plots(out_dir: Path, gains: Sequence[Dict[str, Any]]) -> List[str]:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return []

    written: List[str] = []
    out_dir.mkdir(parents=True, exist_ok=True)

    cc = [g for g in gains if g["condition"] == "Color Checker" and g["anchor_mode"] == "cc_twopoint"]
    if len(cc) >= 3:
        fig, ax = plt.subplots(figsize=(6, 4.5))
        colors = {"P1": "#2166ac", "P2": "#b2182b"}
        for g in cc:
            ax.scatter(
                g["patch_de_ab_mean"],
                g["gain_de00"],
                c=colors.get(g["participant"], "#666666"),
                s=80,
                edgecolors="k",
                linewidths=0.5,
            )
            ax.annotate(g["trial_id"].replace("_", "\n"), (g["patch_de_ab_mean"], g["gain_de00"]), fontsize=6)
        ax.axhline(0, color="k", lw=0.8, ls="--")
        ax.set_xlabel("Chart patch mean ΔE_ab (fit quality)")
        ax.set_ylabel("Gain: ΔE_none − ΔE_cc_twopoint")
        ax.set_title("CC correction gain vs chart fit (Pansor)")
        from matplotlib.lines import Line2D

        ax.legend(
            handles=[Line2D([0], [0], marker="o", color="w", markerfacecolor=colors[p], label=p) for p in colors],
            title="Participant",
        )
        p = out_dir / "cc_gain_vs_patch_de.png"
        fig.tight_layout()
        fig.savefig(p, dpi=150)
        plt.close(fig)
        written.append(str(p))

    bag = [g for g in gains if g["condition"] == "Sephora Bag" and g["anchor_mode"] == "cat02_bag"]
    if len(bag) >= 3:
        fig, ax = plt.subplots(figsize=(6, 4.5))
        colors = {"P1": "#2166ac", "P2": "#b2182b"}
        for g in bag:
            ax.scatter(
                g["bag_n_white"],
                g["gain_de00"],
                c=colors.get(g["participant"], "#666666"),
                s=80,
                edgecolors="k",
                linewidths=0.5,
            )
            ax.annotate(g["trial_id"].replace("_", "\n"), (g["bag_n_white"], g["gain_de00"]), fontsize=6)
        ax.axhline(0, color="k", lw=0.8, ls="--")
        ax.set_xlabel("Bag white stripe pixel count")
        ax.set_ylabel("Gain: ΔE_none − ΔE_cat02_bag")
        ax.set_title("Bag CAT02 gain vs stripe sampling (Pansor)")
        p = out_dir / "bag_gain_vs_n_white.png"
        fig.tight_layout()
        fig.savefig(p, dpi=150)
        plt.close(fig)
        written.append(str(p))

    # Hit-rate bar chart: production bag mode
    modes_plot = [
        ("Sephora Bag", PRODUCTION_BAG_MODE),
    ]
    labels, rates, ns = [], [], []
    for cond, mode in modes_plot:
        s = _summarize_gains([g for g in gains if g["condition"] == cond and g["anchor_mode"] == mode])
        if s:
            labels.append(f"{mode}")
            rates.append(100.0 * s["hit_rate"])
            ns.append(s["n"])
    if labels:
        fig, ax = plt.subplots(figsize=(7, 4))
        bars = ax.bar(labels, rates, color=["#4dac26"])
        ax.axhline(50, color="k", ls="--", lw=0.8)
        ax.set_ylabel("% trials where correction improved ΔE₀₀")
        ax.set_ylim(0, 105)
        ax.set_title("Correction hit-rate (n=6 per mode)")
        for bar, n, rate in zip(bars, ns, rates):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2, f"n={n}", ha="center", fontsize=8)
        plt.xticks(rotation=20, ha="right")
        p = out_dir / "correction_hit_rate.png"
        fig.tight_layout()
        fig.savefig(p, dpi=150)
        plt.close(fig)
        written.append(str(p))

    return written


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument(
        "--full-ablation",
        action="store_true",
        help="Analyze all bag/CC correction modes (CSV must be from eval --full-ablation)",
    )
    args = ap.parse_args()

    if args.full_ablation:
        global BAG_CHROMATIC, BAG_EXPOSURE, CC_CHROMATIC, CC_EXPOSURE, COLOR_CORRECTION_PAIRS
        BAG_CHROMATIC = ("twopoint_bag", "vonkries_bag", "cat02_bag")
        BAG_EXPOSURE = ("bag_white_flash_aligned_xyz_y", "bag_white_flash_aligned_xyz_lstsq")
        CC_CHROMATIC = ("cc_twopoint", "cc_vonkries", "cc_cat02")
        CC_EXPOSURE = ("cc_white_y_scale",)
        COLOR_CORRECTION_PAIRS = [
            ("none", "none", "baseline"),
            ("twopoint_bag", "cc_twopoint", "chromatic_twopoint"),
            ("vonkries_bag", "cc_vonkries", "chromatic_vonkries"),
            ("cat02_bag", "cc_cat02", "chromatic_cat02"),
            ("bag_white_flash_aligned_xyz_y", "cc_white_y_scale", "exposure_y_match"),
        ]

    rows = _load_rows(args.csv)
    if not rows:
        raise SystemExit(f"No rows in {args.csv}")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # --- L* vs chroma by mode ---
    lab_by_mode: Dict[str, Any] = {"all": {}, "by_participant": {pk: {} for pk in PARTICIPANT_ORDER}}
    key_modes = (
        ["none"]
        + list(BAG_CHROMATIC)
        + list(BAG_EXPOSURE)
        + list(CC_CHROMATIC)
        + list(CC_EXPOSURE)
    )
    for mode in key_modes:
        sub = _filter(rows, mode=mode)
        if not sub:
            continue
        lab_by_mode["all"][mode] = _lab_delta_stats(sub)
        for pk in PARTICIPANT_ORDER:
            stats = _lab_delta_stats(_filter(sub, pk=pk))
            if stats:
                lab_by_mode["by_participant"][pk][mode] = stats

    # --- Repeatability ---
    rep_rows = _repeatability(rows)
    rep_summary: Dict[str, Any] = {}
    for cond in ("Sephora Bag", "Color Checker"):
        for mode in key_modes:
            cells = [r for r in rep_rows if r["condition"] == cond and r["anchor_mode"] == mode]
            if not cells:
                continue
            rep_summary.setdefault(cond, {})[mode] = {
                "mean_sd_de00": float(mean(r["sd_de00"] for r in cells)),
                "mean_cv_de00": float(mean(r["cv_de00"] for r in cells if math.isfinite(r["cv_de00"]))),
                "cells": len(cells),
            }

    # --- Correction gains / hit-rate ---
    bag_gains = _correction_gains(rows, condition="Sephora Bag", corrected_modes=BAG_CHROMATIC + BAG_EXPOSURE)
    cc_gains = _correction_gains(rows, condition="Color Checker", corrected_modes=CC_CHROMATIC + CC_EXPOSURE)
    all_gains = bag_gains + cc_gains

    hit_rate: Dict[str, Any] = {"bag": {}, "cc": {}}
    for mode in BAG_CHROMATIC + BAG_EXPOSURE:
        hit_rate["bag"][mode] = _gains_by_participant(bag_gains, mode)
    for mode in CC_CHROMATIC + CC_EXPOSURE:
        hit_rate["cc"][mode] = _gains_by_participant(cc_gains, mode)

    quality_reg = _quality_regressions(all_gains)

    summary: Dict[str, Any] = {
        "source_csv": str(args.csv.resolve()),
        "n_rows": len(rows),
        "lab_error_decomposition": lab_by_mode,
        "repeatability": {
            "per_cell": rep_rows,
            "aggregated_mean_sd": rep_summary,
        },
        "correction_hit_rate": hit_rate,
        "quality_regressions": quality_reg,
        "interpretation_notes": [
            "gain_de00 = ΔE_none − ΔE_corrected; positive means correction helped.",
            f"Production bag correction: {PRODUCTION_BAG_MODE} (best chromatic on Pansor).",
            "CC trials use none only — in-scene CC correction hurt on this cohort.",
            "Cross-session May 20 FitSkin reference; n=3 trials per participant×condition.",
            "P1=Emily (lighter), P2=Liki (darker).",
        ],
    }

    summary_path = args.out_dir / "color_correction_analysis.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    _write_csv(args.out_dir / "repeatability_by_cell.csv", rep_rows)
    _write_csv(args.out_dir / "correction_gains_per_trial.csv", all_gains)

    plot_paths = _try_plots(args.out_dir, all_gains)
    if plot_paths:
        summary["plots"] = plot_paths
        summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    # --- Console report ---
    print(f"Wrote {summary_path}")
    print(f"Wrote {args.out_dir / 'correction_gains_per_trial.csv'}")
    print(f"Wrote {args.out_dir / 'repeatability_by_cell.csv'}")
    for p in plot_paths:
        print(f"Wrote {p}")

    print("\n=== L* vs chroma (mean |Δ|), all trials ===")
    for mode in ["none", PRODUCTION_BAG_MODE]:
        s = lab_by_mode["all"].get(mode)
        if s:
            print(
                f"  {mode:28s}  ΔE={s['mean_de00']:.2f}  "
                f"|ΔL*|={s['mean_abs_delta_L']:.2f}  |Δa*|={s['mean_abs_delta_a']:.2f}  |Δb*|={s['mean_abs_delta_b']:.2f}"
            )

    print("\n=== Repeatability (mean SD of ΔE across T1–T3) ===")
    for cond in ("Sephora Bag", "Color Checker"):
        for mode in ("none", PRODUCTION_BAG_MODE):
            agg = rep_summary.get(cond, {}).get(mode)
            if agg:
                print(f"  {cond[:3]:3s} {mode:28s}  mean_sd={agg['mean_sd_de00']:.2f}  (n cells={agg['cells']})")

    print("\n=== Correction hit-rate (% trials helped vs none) ===")
    for block, modes in (("BAG", BAG_CHROMATIC + BAG_EXPOSURE), ("CC", CC_CHROMATIC + CC_EXPOSURE)):
        for mode in modes:
            bucket = hit_rate["bag" if block == "BAG" else "cc"].get(mode, {})
            all_s = bucket.get("all", {})
            if not all_s:
                continue
            p1 = bucket.get("P1", {})
            p2 = bucket.get("P2", {})
            print(
                f"  {mode:35s}  all {100*all_s.get('hit_rate',0):.0f}% ({all_s.get('n_helped','?')}/{all_s.get('n','?')})  "
                f"P1 {100*p1.get('hit_rate',0):.0f}%  P2 {100*p2.get('hit_rate',0):.0f}%  "
                f"mean_gain={all_s.get('mean_gain_de00', float('nan')):+.2f}"
            )

    if quality_reg:
        print("\n=== Quality vs correction gain (linear r) ===")
        for k, v in quality_reg.items():
            if v.get("n", 0) >= 3:
                print(f"  {k:35s}  n={v['n']}  r={v['r']:.3f}  slope={v['slope']:.4f}")


if __name__ == "__main__":
    main()
