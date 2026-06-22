#!/usr/bin/env python3
"""
Pansor iPhone cohort (2026-06-16): flash/no-flash cheek Lab vs May 20 FitSkin (cross-session).

Default (**production**): best correction from ablation only —
  - **Sephora Bag trials:** ``none`` + ``cat02_bag`` (in-scene bag white → CAT02 chromatic fix)
  - **Color Checker trials:** ``none`` only (in-scene CC chromatic correction hurt ΔE)

Use ``--full-ablation`` to re-run all bag/CC anchor modes for research comparisons.

App-exported ProRAW DNGs need ``--use-camera-wb`` (default on).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from statistics import mean, median, pstdev
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

EMONET = Path(__file__).resolve().parent
SCRIPTS = EMONET / "scripts"
sys.path.insert(0, str(EMONET))
sys.path.insert(0, str(SCRIPTS))

import evaluate_sephora_bag_white_reference as bag_eval  # noqa: E402
import flash_no_flash_skin_lab as fnf  # noqa: E402
import mediapipe as mp  # noqa: E402
import physio_skin_lab_monk as psl  # noqa: E402
import physio_skin_lab_raw_pr250 as pr250  # noqa: E402
from chart_cc_fitskin_lib import process_one_image, silence_stderr  # noqa: E402
from exposure_anchor import load_exposure_anchors, participant_key  # noqa: E402
from flash_no_flash_skin_lab import (  # noqa: E402
    load_lu_sharpening_matrix,
    xyz_to_lab_batch,
)
from mcc24_canonical_d65 import WHITE_PATCH_INDEX, load_canonical_xyz_d65  # noqa: E402
from sephora_bag_reference import NixBagReference  # noqa: E402
from train_flash_noflash_checker_calibration import _chart_patches_camera_linear  # noqa: E402

BLACK_PATCH_INDEX = 23
D65_XYZN = pr250.D65_XYZ_Y1.copy()

DEFAULT_MANIFEST = EMONET / "data" / "pansor" / "manifest_pansor_fitskin.csv"
DEFAULT_CALIBRATION = EMONET / "calibration" / "tier3_affine"
DEFAULT_NIX_JSON = EMONET / "calibration" / "sephora_bag_nix_reference.json"
DEFAULT_OUT = EMONET / "results" / "pansor_ablation"

# Best in-scene chromatic correction on bag trials (Pansor ablation, Jun 2026).
PRODUCTION_BAG_MODE = "cat02_bag"

KEY_MODES = [
    "none",
    PRODUCTION_BAG_MODE,
    "training_anchor",
    "twopoint_bag",
    "twopoint_bag_x_training",
    "vonkries_bag_x_training",
    "lstar_training_x_bag_white_flash_aligned_xyz_y_rel",
    "bag_white_flash_aligned_xyz_lstsq",
    "bag_white_noflash",
]

# Paired modes for bag vs in-scene ColorChecker color-correction usability.
COLOR_CORRECTION_PAIRS = [
    ("none", "none", "baseline"),
    ("twopoint_bag", "cc_twopoint", "chromatic_twopoint"),
    ("vonkries_bag", "cc_vonkries", "chromatic_vonkries"),
    ("cat02_bag", "cc_cat02", "chromatic_cat02"),
    ("bag_white_flash_aligned_xyz_y", "cc_white_y_scale", "exposure_y_match"),
]

# Same trim gates as ``evaluate_sephora_bag_white_reference`` / Phase 4 RAW runner.
_SKIN_TRIM = dict(
    l_star_trim_lo=0.05,
    l_star_trim_hi=0.05,
    a_star_trim_lo=0.05,
    a_star_trim_hi=0.05,
    b_star_trim_lo=0.05,
    b_star_trim_hi=0.05,
    skin_min_chroma_ab=2.0,
)


def _load_nix_json(path: Path) -> NixBagReference:
    d = json.loads(path.read_text(encoding="utf-8"))
    return NixBagReference(
        white_lab=np.asarray(d["white_lab"], dtype=np.float64),
        black_lab=np.asarray(d["black_lab"], dtype=np.float64),
        white_xyz=np.asarray(d["white_xyz"], dtype=np.float64),
        black_xyz=np.asarray(d["black_xyz"], dtype=np.float64),
        white_y=float(d["white_y"]),
        black_y=float(d["black_y"]),
    )


def _load_manifest(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("include_in_eval") != "yes":
                continue
            if not all(row.get(k) for k in ("path_noflash", "path_flash", "fitskin_cheek_L")):
                continue
            rows.append(
                {
                    "trial_id": row["subject_id"],
                    "subject_id": row["subject_id"],
                    "participant": row["participant"],
                    "trial": str(row["trial"]),
                    "condition": row["condition"],
                    "condition_code": row.get("condition_code", ""),
                    "noflash": Path(row["path_noflash"]),
                    "flash": Path(row["path_flash"]),
                    "fitskin_csv_linked": row.get("fitskin_csv_linked", ""),
                    "fit_lab": (
                        float(row["fitskin_cheek_L"]),
                        float(row["fitskin_cheek_a"]),
                        float(row["fitskin_cheek_b"]),
                    ),
                }
            )
    if not rows:
        raise SystemExit(f"No eval rows in manifest: {path}")
    return rows


def _summarize(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for mode in sorted({str(r["anchor_mode"]) for r in rows}):
        vals = [
            float(r["reflectance_cheek_de00"])
            for r in rows
            if r["anchor_mode"] == mode and np.isfinite(float(r["reflectance_cheek_de00"]))
        ]
        if not vals:
            continue
        out[mode] = {
            "n": len(vals),
            "mean_de00": float(np.mean(vals)),
            "median_de00": float(median(vals)),
            "std_de00": float(pstdev(vals)) if len(vals) > 1 else 0.0,
        }
    return out


def _summarize_lab_deltas(rows: List[Dict[str, Any]], mode: str) -> Dict[str, Any]:
    sub = [r for r in rows if r["anchor_mode"] == mode and np.isfinite(float(r["reflectance_cheek_de00"]))]
    if not sub:
        return {}
    de = [float(r["reflectance_cheek_de00"]) for r in sub]
    dL = [abs(float(r["delta_L"])) for r in sub]
    da = [abs(float(r["delta_a"])) for r in sub]
    db = [abs(float(r["delta_b"])) for r in sub]
    return {
        "n": len(sub),
        "mean_de00": float(np.mean(de)),
        "median_de00": float(median(de)),
        "mean_abs_delta_L": float(np.mean(dL)),
        "mean_abs_delta_a": float(np.mean(da)),
        "mean_abs_delta_b": float(np.mean(db)),
    }


def _summarize_color_correction_usability(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compare paired bag vs in-scene CC correction modes (different trial subsets, same pipeline)."""
    bag_rows = [r for r in rows if r.get("condition") == "Sephora Bag"]
    cc_rows = [r for r in rows if r.get("condition") == "Color Checker"]
    pairs_out: List[Dict[str, Any]] = []
    for bag_mode, cc_mode, label in COLOR_CORRECTION_PAIRS:
        bag_stats = _summarize_lab_deltas(bag_rows, bag_mode)
        cc_stats = _summarize_lab_deltas(cc_rows, cc_mode)
        if not bag_stats and not cc_stats:
            continue
        entry: Dict[str, Any] = {"comparison": label, "bag_mode": bag_mode, "cc_mode": cc_mode}
        if bag_stats:
            entry["bag_trials"] = bag_stats
        if cc_stats:
            entry["cc_trials"] = cc_stats
        if bag_stats and cc_stats:
            entry["delta_mean_de00_bag_minus_cc"] = float(
                bag_stats["mean_de00"] - cc_stats["mean_de00"]
            )
        pairs_out.append(entry)

    cc_only = _summarize_lab_deltas(cc_rows, "cc_matrix_24")
    return {
        "description": (
            "Bag trials (n=6) vs Color Checker trials (n=6); same flash/no-flash reflectance "
            "pipeline; cross-session May 20 FitSkin reference."
        ),
        "paired_comparisons": pairs_out,
        "cc_full_matrix_24": cc_only,
        "best_bag_chromatic": _best_of_modes(
            bag_rows, ("twopoint_bag", "vonkries_bag", "cat02_bag")
        ),
        "best_cc_chromatic": _best_of_modes(
            cc_rows, ("cc_twopoint", "cc_vonkries", "cc_cat02")
        ),
        "cc_matrix_24_note": (
            "Per-trial 24-patch RGB→XYZ applied directly on albedo (experimental; "
            "scale mismatch vs tier3 — not comparable to bag twopoint/vonkries)."
        ),
    }


def _best_of_modes(rows: List[Dict[str, Any]], modes: Tuple[str, ...]) -> Dict[str, Any]:
    stats = [_summarize_lab_deltas(rows, m) for m in modes]
    stats = [s for s in stats if s]
    if not stats:
        return {}
    return min(stats, key=lambda s: s["mean_de00"])


def _lab_from_albedo_rgb_to_xyz(
    cache: bag_eval._TrialCache,
    M_rgb_to_xyz: np.ndarray,
    *,
    scale: Optional[float] = None,
) -> Tuple[float, float, float]:
    """Per-trial 24-patch RGB→XYZ matrix on flash/no-flash albedo (bypasses tier3_affine)."""
    albedo = cache.albedo_base if scale is None else np.clip(cache.albedo_base * scale, 0.0, None)
    m = cache.lab_mask > 0
    if not np.any(m):
        return float("nan"), float("nan"), float("nan")
    xyz = np.asarray(albedo[m], dtype=np.float64) @ np.asarray(M_rgb_to_xyz, dtype=np.float64).T
    L, a, b = xyz_to_lab_batch(xyz, D65_XYZN)
    keep, *_rest = psl.skin_lab_trim_selection(
        L,
        a,
        b,
        l_star_trim_lo=_SKIN_TRIM["l_star_trim_lo"],
        l_star_trim_hi=_SKIN_TRIM["l_star_trim_hi"],
        a_star_trim_lo=_SKIN_TRIM["a_star_trim_lo"],
        a_star_trim_hi=_SKIN_TRIM["a_star_trim_hi"],
        b_star_trim_lo=_SKIN_TRIM["b_star_trim_lo"],
        b_star_trim_hi=_SKIN_TRIM["b_star_trim_hi"],
        min_chroma_ab=_SKIN_TRIM["skin_min_chroma_ab"],
    )
    if not np.any(keep):
        return float("nan"), float("nan"), float("nan")
    return float(np.mean(L[keep])), float(np.mean(a[keep])), float(np.mean(b[keep]))


def _cc_inscene_corrections(patches: np.ndarray) -> Dict[str, Any]:
    """
    In-scene ColorChecker corrections on aligned camera-linear patches.

    Targets are canonical D65 MCC white/black (same scale as tier3 training: Y≈0–1).
    """
    xyz_ref = load_canonical_xyz_d65() / 100.0
    white_rgb = np.asarray(patches[WHITE_PATCH_INDEX], dtype=np.float64)
    black_rgb = np.asarray(patches[BLACK_PATCH_INDEX], dtype=np.float64)
    white_xyz = xyz_ref[WHITE_PATCH_INDEX]
    black_xyz = xyz_ref[BLACK_PATCH_INDEX]

    tp = bag_eval._twopoint_affine_rgb(white_rgb, black_rgb, white_xyz, black_xyz)
    vk = bag_eval._vonkries_rgb_scales(white_rgb, white_xyz)
    white_xyz_cam = fnf.linear_rgb_to_xyz_d65(white_rgb.reshape(1, 3))[0]
    cat = bag_eval._cat02_rgb_correction_matrix(white_xyz_cam, white_xyz)

    row_w = pr250.build_patch_lstsq_row_weights(anchor_weight=2.5, skin_weight=1.0)
    M, _, _ = pr250.fit_rgb_to_xyz_lstsq_huber_irls(
        patches, xyz_ref, with_intercept=False, row_weights=row_w
    )
    pred = patches @ M.T
    ref_lab = np.array([pr250.xyz_to_lab(xyz_ref[i], D65_XYZN) for i in range(24)])
    fit_lab = np.array([pr250.xyz_to_lab(pred[i], D65_XYZN) for i in range(24)])
    patch_de = float(np.mean([pr250.delta_e_ab(fit_lab[i], ref_lab[i]) for i in range(24)]))

    y_cam = float(0.2126 * white_rgb[0] + 0.7152 * white_rgb[1] + 0.0722 * white_rgb[2])
    white_y_scale = float(white_xyz[1] / max(y_cam, 1e-12))

    return {
        "twopoint_a": tp[0] if tp is not None else None,
        "twopoint_b": tp[1] if tp is not None else None,
        "vonkries_scales": vk,
        "cat02_P": cat,
        "matrix_rgb_to_xyz": M,
        "white_y_scale": white_y_scale,
        "patch_de_ab_mean": patch_de,
    }


def _append_lab_row(
    rows: List[Dict[str, Any]],
    *,
    rec: Dict[str, Any],
    mode_name: str,
    lab: Tuple[float, float, float],
    fit_lab: Tuple[float, float, float],
    scale: Optional[float],
    lstar_correction: Optional[float] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    if lstar_correction is not None and lstar_correction > 0:
        lab = (float(np.clip(lab[0] * lstar_correction, 0.0, 100.0)), lab[1], lab[2])
    row: Dict[str, Any] = {
        "trial_id": rec["pair"]["trial_id"],
        "subject_id": rec["subject_id"],
        "participant": rec["participant"],
        "trial": rec["trial"],
        "condition": rec["condition"],
        "anchor_mode": mode_name,
        "reflectance_exposure_scale": scale,
        "lstar_correction": lstar_correction,
        "fitskin_cheek_L": fit_lab[0],
        "fitskin_cheek_a": fit_lab[1],
        "fitskin_cheek_b": fit_lab[2],
        "reflectance_L": lab[0],
        "reflectance_a": lab[1],
        "reflectance_b": lab[2],
        "delta_L": lab[0] - fit_lab[0],
        "delta_a": lab[1] - fit_lab[1],
        "delta_b": lab[2] - fit_lab[2],
        "reflectance_cheek_de00": bag_eval._de00(lab, fit_lab),
        "fitskin_reference": rec.get("fitskin_csv_linked", ""),
    }
    if extra:
        row.update(extra)
    rows.append(row)


def _run_cc_trial(
    rec: Dict[str, Any],
    *,
    training_scale: Optional[float],
    face_mesh: Any,
    max_align_width: int,
    rows: List[Dict[str, Any]],
    streamlined: bool,
) -> None:
    pair = rec["pair"]
    fit_lab = rec["fit_lab"]
    nf_lin = rec["nf_lin"]
    fl_lin = rec["nf_lin"]

    try:
        tcache = bag_eval._build_trial_cache(
            nf_lin,
            fl_lin,
            face_mesh,
            max_align_width=max_align_width,
            skin_exclusion_dilate_iod_fraction=0.12,
        )
    except Exception as exc:
        print(f"WARN CC cache failed {pair['trial_id']}: {exc}", file=sys.stderr)
        return

    try:
        lab_none = bag_eval._lab_from_scale(tcache, None)
        _append_lab_row(rows, rec=rec, mode_name="none", lab=lab_none, fit_lab=fit_lab, scale=None)
    except Exception as exc:
        print(f"WARN none failed {pair['trial_id']}: {exc}", file=sys.stderr)
        return

    if streamlined:
        print(f"  {pair['trial_id']}: none (no in-scene CC correction)")
        return

    if training_scale is not None:
        try:
            lab = bag_eval._lab_from_scale(tcache, training_scale)
            _append_lab_row(
                rows, rec=rec, mode_name="training_anchor", lab=lab, fit_lab=fit_lab, scale=training_scale
            )
        except Exception as exc:
            print(f"WARN training_anchor failed {pair['trial_id']}: {exc}", file=sys.stderr)

    # Single-frame chart CC (legacy comparison; not flash/no-flash fused).
    try:
        nf_bgr = pr250.linear_rgb_to_preview_bgr(nf_lin)
        with silence_stderr():
            cc = process_one_image(nf_bgr, face_mesh, roi="cheek")
        if cc.get("chart_ok") and cc.get("status") == "ok":
            lab = (float(cc["cheek_L"]), float(cc["cheek_a"]), float(cc["cheek_b"]))
            _append_lab_row(
                rows,
                rec=rec,
                mode_name="chart_cc_noflash_cheek",
                lab=lab,
                fit_lab=fit_lab,
                scale=None,
                extra={
                    "patch_de_ab_mean": cc.get("patch_de_ab_mean"),
                    "chart_area_fraction": cc.get("chart_area_fraction"),
                },
            )
    except Exception as exc:
        print(f"WARN chart_cc failed {pair['trial_id']}: {exc}", file=sys.stderr)

    # In-scene CC chromatic correction on flash/no-flash reflectance (parallel to bag modes).
    patches = _chart_patches_camera_linear(tcache.align.noflash_linear)
    if patches is None:
        print(f"WARN no ColorChecker patches {pair['trial_id']}", file=sys.stderr)
        return

    cc_corr = _cc_inscene_corrections(patches)
    cc_extra = {"patch_de_ab_mean": cc_corr["patch_de_ab_mean"], "correction_family": "cc_inscene"}

    def _cc_record(mode_name: str, lab: Tuple[float, float, float], scale: Optional[float]) -> None:
        _append_lab_row(
            rows,
            rec=rec,
            mode_name=mode_name,
            lab=lab,
            fit_lab=fit_lab,
            scale=scale,
            extra={**cc_extra, "correction_family": "cc_inscene"},
        )

    if cc_corr["white_y_scale"] > 0:
        try:
            lab = bag_eval._lab_from_scale(tcache, cc_corr["white_y_scale"])
            _cc_record("cc_white_y_scale", lab, cc_corr["white_y_scale"])
        except Exception as exc:
            print(f"WARN cc_white_y_scale failed: {exc}", file=sys.stderr)

    if cc_corr["twopoint_a"] is not None and cc_corr["twopoint_b"] is not None:
        tp_a = np.array(cc_corr["twopoint_a"], dtype=np.float64)
        tp_b = np.array(cc_corr["twopoint_b"], dtype=np.float64)
        try:
            lab = bag_eval._lab_from_corrected(tcache, affine_a=tp_a, affine_b=tp_b)
            _cc_record("cc_twopoint", lab, None)
        except Exception as exc:
            print(f"WARN cc_twopoint failed: {exc}", file=sys.stderr)

    if cc_corr["vonkries_scales"] is not None:
        vk = np.array(cc_corr["vonkries_scales"], dtype=np.float64)
        try:
            lab = bag_eval._lab_from_corrected(tcache, diag=vk)
            _cc_record("cc_vonkries", lab, None)
        except Exception as exc:
            print(f"WARN cc_vonkries failed: {exc}", file=sys.stderr)

    if cc_corr["cat02_P"] is not None:
        cat_P = np.array(cc_corr["cat02_P"], dtype=np.float64)
        try:
            lab = bag_eval._lab_from_corrected(tcache, matrix=cat_P)
            _cc_record("cc_cat02", lab, None)
        except Exception as exc:
            print(f"WARN cc_cat02 failed: {exc}", file=sys.stderr)

    try:
        M = np.asarray(cc_corr["matrix_rgb_to_xyz"], dtype=np.float64)
        lab = _lab_from_albedo_rgb_to_xyz(tcache, M)
        _cc_record("cc_matrix_24", lab, None)
    except Exception as exc:
        print(f"WARN cc_matrix_24 failed: {exc}", file=sys.stderr)

    print(f"  {pair['trial_id']}: CC color correction modes recorded")


def _run_bag_trial(
    rec: Dict[str, Any],
    *,
    training_scale: Optional[float],
    participant_medians: Dict[str, Dict[str, float]],
    face_mesh: Any,
    hands_detector: Any,
    nix_ref: NixBagReference,
    max_align_width: int,
    sam2_segmenter: Any,
    rows: List[Dict[str, Any]],
    streamlined: bool,
) -> None:
    pair = rec["pair"]
    subject_id = rec["subject_id"]
    participant = rec["participant"]
    trial = rec["trial"]
    pk = rec["participant_key"]
    fit_lab = rec["fit_lab"]
    bag_scales = rec["bag_scales"]
    bag_info = rec["bag_info"]
    nf_lin = rec["nf_lin"]
    fl_lin = rec["fl_lin"]

    trial_modes: Dict[str, Tuple[Optional[float], Optional[float]]] = {
        "none": (None, None),
        "training_anchor": (training_scale, None),
    }
    for name, scale in bag_scales.items():
        trial_modes[name] = (scale, None)

    for key in (
        "bag_white_reflectance_xyz_y",
        "bag_white_flash_aligned_xyz_y",
        "bag_white_noflash_xyz_y",
        "bag_white_reflectance_xyz_lstsq",
        "bag_white_flash_aligned_xyz_lstsq",
        "bag_white_noflash_xyz_lstsq",
    ):
        med = participant_medians.get(pk, {}).get(key)
        if training_scale is None or med is None or key not in bag_scales or med <= 0:
            continue
        rel = float(bag_scales[key]) / float(med)
        trial_modes[f"hybrid_training_x_{key}_rel"] = (float(training_scale) * rel, None)
        trial_modes[f"lstar_training_x_{key}_rel"] = (training_scale, rel)

    _src = "bag_white_flash_aligned"
    _vk = bag_info.get(f"{_src}_vonkries_scales")
    _tp_a = bag_info.get(f"{_src}_twopoint_a")
    _tp_b = bag_info.get(f"{_src}_twopoint_b")
    _cat = bag_info.get(f"{_src}_cat02_P")
    _cct = bag_info.get(f"{_src}_cct_estimate")

    try:
        tcache = bag_eval._build_trial_cache(
            nf_lin,
            fl_lin,
            face_mesh,
            max_align_width=max_align_width,
            skin_exclusion_dilate_iod_fraction=0.12,
        )
    except Exception as exc:
        print(f"WARN bag cache failed {pair['trial_id']}: {exc}", file=sys.stderr)
        return

    bag_info_scalar = {
        k: v
        for k, v in bag_info.items()
        if not isinstance(v, (list, np.ndarray)) or k == "bag_bbox"
    }

    def _record(
        mode_name: str,
        lab: Tuple[float, float, float],
        scale: Optional[float],
        lstar_corr: Optional[float],
    ) -> None:
        _append_lab_row(
            rows,
            rec=rec,
            mode_name=mode_name,
            lab=lab,
            fit_lab=fit_lab,
            scale=scale,
            lstar_correction=lstar_corr,
            extra=bag_info_scalar,
        )

    if streamlined:
        try:
            _record("none", bag_eval._lab_from_scale(tcache, None), None, None)
        except Exception as exc:
            print(f"WARN none failed: {exc}", file=sys.stderr)
            return
        if _cat is not None:
            try:
                cat_P = np.array(_cat, dtype=np.float64)
                lab = bag_eval._lab_from_corrected(tcache, matrix=cat_P)
                _record(PRODUCTION_BAG_MODE, lab, None, None)
            except Exception as exc:
                print(f"WARN {PRODUCTION_BAG_MODE} failed: {exc}", file=sys.stderr)
        print(f"  {pair['trial_id']}: none + {PRODUCTION_BAG_MODE}")
        return

    for mode, (scale, lstar_correction) in trial_modes.items():
        try:
            lab = bag_eval._lab_from_scale(tcache, scale)
        except Exception as exc:
            print(f"WARN {mode} failed: {exc}", file=sys.stderr)
            continue
        _record(mode, lab, scale, lstar_correction)

    if _cct is not None and training_scale is not None:
        cct_xyz_raw = [
            bag_info.get("bag_white_flash_aligned_white_xyz_x"),
            bag_info.get("bag_white_flash_aligned_white_xyz_y"),
            bag_info.get("bag_white_flash_aligned_white_xyz_z"),
        ]
        if all(v is not None for v in cct_xyz_raw):
            bag_xyz_w = np.array(cct_xyz_raw, dtype=np.float64)
            bag_xyz_w_norm = bag_xyz_w / max(float(bag_xyz_w[1]), 1e-8)
            try:
                lab = bag_eval._lab_from_scale(tcache, training_scale, xyz_scene_white=bag_xyz_w_norm)
                _record("cct_from_bag", lab, training_scale, None)
            except Exception as exc:
                print(f"WARN cct_from_bag failed: {exc}", file=sys.stderr)

    if _vk is not None:
        vk = np.array(_vk, dtype=np.float64)
        for mode_name, extra_scale in [("vonkries_bag", None), ("vonkries_bag_x_training", training_scale)]:
            if mode_name.endswith("_x_training") and training_scale is None:
                continue
            try:
                lab = bag_eval._lab_from_corrected(tcache, diag=vk, scale=extra_scale)
                _record(mode_name, lab, extra_scale, None)
            except Exception as exc:
                print(f"WARN {mode_name} failed: {exc}", file=sys.stderr)

    if _tp_a is not None and _tp_b is not None:
        tp_a = np.array(_tp_a, dtype=np.float64)
        tp_b = np.array(_tp_b, dtype=np.float64)
        for mode_name, extra_scale in [("twopoint_bag", None), ("twopoint_bag_x_training", training_scale)]:
            if mode_name.endswith("_x_training") and training_scale is None:
                continue
            try:
                lab = bag_eval._lab_from_corrected(tcache, affine_a=tp_a, affine_b=tp_b, scale=extra_scale)
                _record(mode_name, lab, extra_scale, None)
            except Exception as exc:
                print(f"WARN {mode_name} failed: {exc}", file=sys.stderr)

    if _cat is not None:
        cat_P = np.array(_cat, dtype=np.float64)
        for mode_name, extra_scale in [("cat02_bag", None), ("cat02_bag_x_training", training_scale)]:
            if mode_name.endswith("_x_training") and training_scale is None:
                continue
            try:
                lab = bag_eval._lab_from_corrected(tcache, matrix=cat_P, scale=extra_scale)
                _record(mode_name, lab, extra_scale, None)
            except Exception as exc:
                print(f"WARN {mode_name} failed: {exc}", file=sys.stderr)

    print(f"  {pair['trial_id']}: {len([r for r in rows if r['trial_id'] == pair['trial_id']])} modes")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    ap.add_argument("--iphone-calibration", type=Path, default=DEFAULT_CALIBRATION)
    ap.add_argument("--nix-json", type=Path, default=DEFAULT_NIX_JSON)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--max-align-width", type=int, default=1600)
    ap.add_argument("--raw-half-size", type=int, default=1)
    ap.add_argument("--known-ambient-cct-k", type=float, default=6546.0)
    ap.add_argument("--known-ambient-duv", type=float, default=0.0017)
    ap.add_argument(
        "--use-camera-wb",
        action="store_true",
        default=True,
        help="Apply DNG as-shot white balance (required for app-export ProRAW; default on)",
    )
    ap.add_argument(
        "--no-use-camera-wb",
        action="store_false",
        dest="use_camera_wb",
        help="Unity WB (Phase 4 booth DNG style)",
    )
    ap.add_argument("--mobile-sam", action="store_true", default=True)
    ap.add_argument("--no-mobile-sam", action="store_false", dest="mobile_sam")
    ap.add_argument("--mobile-sam-ckpt", type=Path, default=EMONET / "mobile_sam.pt")
    ap.add_argument(
        "--full-ablation",
        action="store_true",
        help=f"Run all anchor modes (default: production only — none + {PRODUCTION_BAG_MODE} on bag trials)",
    )
    args = ap.parse_args()
    streamlined = not args.full_ablation

    manifest_rows = _load_manifest(args.manifest)
    nix_ref = _load_nix_json(args.nix_json)
    bag_eval._load_iphone_calibration(args.iphone_calibration)
    load_lu_sharpening_matrix(args.iphone_calibration / "lu_sharpening_M.npy")
    training_anchors = load_exposure_anchors(args.iphone_calibration)

    sam2_segmenter = None
    if args.mobile_sam:
        from sephora_bag_mobile_sam import MobileSamBagSegmenter, mobile_sam_available

        if not mobile_sam_available():
            raise SystemExit(f"MobileSAM unavailable (ckpt: {args.mobile_sam_ckpt})")
        print(f"Loading MobileSAM from {args.mobile_sam_ckpt}...")
        sam2_segmenter = MobileSamBagSegmenter(checkpoint=args.mobile_sam_ckpt)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, Any]] = []
    trial_records: List[Dict[str, Any]] = []

    mp_fm = mp.solutions.face_mesh
    mp_hands = mp.solutions.hands
    with mp_fm.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
    ) as face_mesh, mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=2,
        min_detection_confidence=0.4,
    ) as hands_detector:
        for mrow in manifest_rows:
            print(f"Loading {mrow['trial_id']} ({mrow['condition']})...")
            nf_lin = bag_eval._read_dng_u01(
                mrow["noflash"], args.raw_half_size, use_camera_wb=args.use_camera_wb
            )
            fl_lin = bag_eval._read_dng_u01(
                mrow["flash"], args.raw_half_size, use_camera_wb=args.use_camera_wb
            )
            pk = participant_key(mrow["subject_id"], mrow["participant"])
            rec: Dict[str, Any] = {
                "pair": mrow,
                "subject_id": mrow["subject_id"],
                "participant": mrow["participant"],
                "trial": mrow["trial"],
                "condition": mrow["condition"],
                "participant_key": pk,
                "fit_lab": mrow["fit_lab"],
                "fitskin_csv_linked": mrow.get("fitskin_csv_linked", ""),
                "nf_lin": nf_lin,
                "fl_lin": fl_lin,
            }
            if mrow["condition"] == "Sephora Bag":
                try:
                    bag_scales, bag_info = bag_eval._bag_white_scales(
                        nf_lin,
                        fl_lin,
                        face_mesh=face_mesh,
                        hands_detector=hands_detector,
                        nix_white_y=float(nix_ref.white_y),
                        nix_white_xyz=nix_ref.white_xyz,
                        nix_ref_black_xyz=nix_ref.black_xyz,
                        max_align_width=args.max_align_width,
                        sam2_segmenter=sam2_segmenter,
                    )
                    rec["bag_scales"] = bag_scales
                    rec["bag_info"] = bag_info
                    trial_records.append(rec)
                except Exception as exc:
                    print(f"WARN bag segmentation failed {mrow['trial_id']}: {exc}", file=sys.stderr)
            else:
                trial_records.append(rec)

        participant_medians: Dict[str, Dict[str, float]] = {}
        if not streamlined:
            for pk in sorted({str(r["participant_key"]) for r in trial_records if "bag_scales" in r}):
                participant_medians[pk] = {}
                subset = [r for r in trial_records if r.get("participant_key") == pk and "bag_scales" in r]
                for key in (
                    "bag_white_reflectance_xyz_y",
                    "bag_white_flash_aligned_xyz_y",
                    "bag_white_noflash_xyz_y",
                    "bag_white_reflectance_xyz_lstsq",
                    "bag_white_flash_aligned_xyz_lstsq",
                    "bag_white_noflash_xyz_lstsq",
                ):
                    vals = [
                        float(r["bag_scales"][key])
                        for r in subset
                        if key in r["bag_scales"] and np.isfinite(float(r["bag_scales"][key]))
                    ]
                    if vals:
                        participant_medians[pk][key] = float(np.median(vals))

        for rec in trial_records:
            training_scale = training_anchors.get(rec["participant_key"])
            if rec["condition"] == "Sephora Bag":
                _run_bag_trial(
                    rec,
                    training_scale=training_scale,
                    participant_medians=participant_medians,
                    face_mesh=face_mesh,
                    hands_detector=hands_detector,
                    nix_ref=nix_ref,
                    max_align_width=args.max_align_width,
                    sam2_segmenter=sam2_segmenter,
                    rows=rows,
                    streamlined=streamlined,
                )
            else:
                _run_cc_trial(
                    rec,
                    training_scale=training_scale,
                    face_mesh=face_mesh,
                    max_align_width=args.max_align_width,
                    rows=rows,
                    streamlined=streamlined,
                )

    csv_path = args.out_dir / "pansor_ablation.csv"
    fieldnames = sorted({k for r in rows for k in r.keys()})
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            out = dict(row)
            if isinstance(out.get("bag_bbox"), (list, tuple)):
                out["bag_bbox"] = json.dumps(out["bag_bbox"])
            writer.writerow(out)

    summary_all = _summarize(rows)
    bag_rows = [r for r in rows if r.get("condition") == "Sephora Bag"]
    cc_rows = [r for r in rows if r.get("condition") == "Color Checker"]
    cc_usability = _summarize_color_correction_usability(rows) if not streamlined else None
    summary = {
        "manifest": str(args.manifest.resolve()),
        "fitskin_reference": "May 20 median cheek (cross-session)",
        "streamlined": streamlined,
        "production_bag_mode": PRODUCTION_BAG_MODE if streamlined else None,
        "use_camera_wb": bool(args.use_camera_wb),
        "known_ambient_cct_k": float(args.known_ambient_cct_k),
        "n_trials": len(manifest_rows),
        "n_bag_trials": sum(1 for r in manifest_rows if r["condition"] == "Sephora Bag"),
        "n_cc_trials": sum(1 for r in manifest_rows if r["condition"] == "Color Checker"),
        "color_correction_usability": cc_usability,
        "summary_all_modes": summary_all,
        "summary_bag_trials_only": _summarize(bag_rows),
        "summary_cc_trials_only": _summarize(cc_rows),
        "key_modes_bag": {},
    }
    for m in KEY_MODES:
        vals = [
            float(r["reflectance_cheek_de00"])
            for r in bag_rows
            if r["anchor_mode"] == m and np.isfinite(float(r["reflectance_cheek_de00"]))
        ]
        if vals:
            summary["key_modes_bag"][m] = {
                "n": len(vals),
                "mean_de00": float(mean(vals)),
                "median_de00": float(median(vals)),
                "std_de00": float(pstdev(vals)) if len(vals) > 1 else 0.0,
            }

    summary_path = args.out_dir / "pansor_ablation_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(f"\nWrote {csv_path} ({len(rows)} rows)")
    print(f"Wrote {summary_path}")
    print(f"\n(use_camera_wb={args.use_camera_wb}, streamlined={streamlined})")
    if streamlined:
        print(f"\n--- Production modes (mean ΔE₀₀) ---")
        for cond, modes in (
            ("Sephora Bag", ("none", PRODUCTION_BAG_MODE)),
            ("Color Checker", ("none",)),
        ):
            for mode in modes:
                stats = summary.get(
                    "summary_bag_trials_only" if cond == "Sephora Bag" else "summary_cc_trials_only",
                    {},
                ).get(mode)
                if stats:
                    print(f"  {cond:16s} {mode:20s}  mean={stats['mean_de00']:.2f}  n={stats['n']}")
    else:
        print("\n--- Bag trials: key modes (mean ΔE₀₀) ---")
        for mode in KEY_MODES:
            stats = summary.get("key_modes_bag", {}).get(mode)
            if stats:
                print(f"  {mode:45s}  mean={stats['mean_de00']:.2f}  n={stats['n']}")
        print("\n--- CC trials (in-scene chart correction on flash/no-flash) ---")
        for mode in (
            "none",
            "cc_twopoint",
            "cc_vonkries",
            "cc_cat02",
            "cc_matrix_24",
            "cc_white_y_scale",
            "chart_cc_noflash_cheek",
        ):
            stats = summary.get("summary_cc_trials_only", {}).get(mode)
            if stats:
                print(f"  {mode:45s}  mean={stats['mean_de00']:.2f}  n={stats['n']}")

        if cc_usability:
            print("\n--- Bag vs ColorChecker color correction (paired methods, mean ΔE₀₀) ---")
            for pair in cc_usability.get("paired_comparisons", []):
                bag_s = pair.get("bag_trials", {})
                cc_s = pair.get("cc_trials", {})
                if not bag_s or not cc_s:
                    continue
                print(
                    f"  {pair['comparison']:22s}  bag={bag_s['mean_de00']:.2f}  "
                    f"cc={cc_s['mean_de00']:.2f}  "
                    f"(Δ bag-cc={pair.get('delta_mean_de00_bag_minus_cc', float('nan')):+.2f})"
                )
            best_b = cc_usability.get("best_bag_chromatic", {})
            best_c = cc_usability.get("best_cc_chromatic", {})
            if best_b and best_c:
                print(
                    f"\n  Best chromatic bag: mean ΔE₀₀={best_b['mean_de00']:.2f}  |  "
                    f"Best chromatic CC: mean ΔE₀₀={best_c['mean_de00']:.2f}"
                )


if __name__ == "__main__":
    main()
