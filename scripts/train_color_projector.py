#!/usr/bin/env python3
"""
Train residual color projector for chart-free flash/no-flash skin Lab.

Primary goal: improve Pansor chart-free ΔE00 vs FitSkin while keeping affine
``tier3_affine`` as the frozen baseline.

Training supervision:
  - ColorChecker reflectance patches from Pansor CC DNGs (camera WB)
  - optional booth / tone-manifest MCC pairs
  - ISSA cheek-median spectral synthetic rows (full bank or subset)

Never trains on FitSkin Lab labels.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from delta_e_2000 import delta_e_2000  # noqa: E402
from flash_no_flash_skin_lab import (  # noqa: E402
    _resize_linear_max_width,
    align_flash_to_noflash_linear,
    estimate_exposure_scale_masked,
    estimate_reflectance_linear,
)
from flash_noflash_spectral import issa_skin_calibration_rows  # noqa: E402
from mcc24_canonical_d65 import load_canonical_xyz_d65  # noqa: E402
from models.color_projector import (  # noqa: E402
    ColorProjectorArtifact,
    ColorProjectorSettings,
    ResidualColorProjector,
    affine_rgb_to_xyz,
    fourier_encode_numpy,
    save_color_projector_artifact,
)
from train_flash_noflash_checker_calibration import (  # noqa: E402
    _chart_patches_camera_linear,
    _read_linear_rgb,
)

# Prefer full ethnic median names; skip short aliases to avoid duplicates.
FULL_ISSA_BANK = [
    "issa_median_african",
    "issa_median_caucasian",
    "issa_median_chinese",
    "issa_median_east_asian",
    "issa_median_european",
    "issa_median_japanese",
    "issa_median_middle_eastern",
    "issa_median_south_asian",
    "issa_median_thai",
]
TONE_ISSA = ["issa_median_caucasian", "issa_median_south_asian"]

# MCC skin-like patches get higher weight (dark skin, light skin).
SKIN_PATCH_IDX = {0, 1}


def _xyz_to_lab(xyz: np.ndarray, xyzn: np.ndarray) -> np.ndarray:
    xyz = np.asarray(xyz, dtype=np.float64)
    w = np.asarray(xyzn, dtype=np.float64).reshape(1, 3)
    t = xyz / np.maximum(w, 1e-12)
    d = 6.0 / 29.0

    def f(u: np.ndarray) -> np.ndarray:
        return np.where(u > d**3, np.cbrt(u), u / (3.0 * d**2) + 4.0 / 29.0)

    fx, fy, fz = f(t[:, 0]), f(t[:, 1]), f(t[:, 2])
    return np.stack([116.0 * fy - 16.0, 500.0 * (fx - fy), 200.0 * (fy - fz)], axis=1)


D65_XYZN = np.array([0.95047, 1.0, 1.08883], dtype=np.float64)


def _load_cc_rows(manifest: Path) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with manifest.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if str(row.get("include_in_eval", "yes")).strip().lower() not in ("yes", "1", "true"):
                continue
            code = row.get("condition_code", row.get("condition", "CC"))
            if code not in ("CC", "Color Checker") and "ColorChecker" not in str(
                row.get("condition", "")
            ):
                if row.get("condition_code") != "CC":
                    continue
            nf = str(row.get("path_noflash", "")).strip()
            fl = str(row.get("path_flash", "")).strip()
            if not nf or not fl:
                continue
            if not Path(nf).is_file() or not Path(fl).is_file():
                print(f"skip missing files: {row.get('subject_id')}", file=sys.stderr)
                continue
            rows.append(row)
    return rows


def _extract_reflectance_patches(
    path_noflash: Path,
    path_flash: Path,
    *,
    camera_wb: bool,
    half_size: int,
    max_align_width: int,
) -> Optional[np.ndarray]:
    nf = _read_linear_rgb(path_noflash, half_size=half_size, camera_wb=camera_wb)
    fl = _read_linear_rgb(path_flash, half_size=half_size, camera_wb=camera_wb)
    nf_w = _resize_linear_max_width(nf, max_align_width)
    fl_w = _resize_linear_max_width(fl, max_align_width)
    align = align_flash_to_noflash_linear(nf_w, fl_w, motion_ecc="euclidean")
    # Exposure-match flash on whole frame for chart (no face mask at train time).
    h, w = align.noflash_linear.shape[:2]
    mask = np.ones((h, w), dtype=np.uint8) * 255
    scale = estimate_exposure_scale_masked(
        align.noflash_linear, align.flash_aligned_linear, mask
    )
    fl_scaled = np.clip(align.flash_aligned_linear * scale, 0.0, None)
    R = estimate_reflectance_linear(align.noflash_linear, fl_scaled, fusion="geometric")
    patches = _chart_patches_camera_linear(R)
    return patches


def build_samples(
    *,
    pansor_manifest: Path,
    extra_manifests: Sequence[Path],
    cal_dir: Path,
    issa_mode: str,
    issa_weight: float,
    camera_wb: bool,
    half_size: int,
    max_align_width: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str], List[str]]:
    """Returns rgb, xyz, weights, sources, subject_ids."""
    xyz_ref = load_canonical_xyz_d65() / 100.0
    rgbs: List[np.ndarray] = []
    xyzs: List[np.ndarray] = []
    weights: List[float] = []
    sources: List[str] = []
    sids: List[str] = []

    manifests = [pansor_manifest, *extra_manifests]
    for man in manifests:
        if not man.is_file():
            continue
        for row in _load_cc_rows(man):
            sid = str(row.get("subject_id") or row.get("participant") or "unknown")
            try:
                patches = _extract_reflectance_patches(
                    Path(row["path_noflash"]),
                    Path(row["path_flash"]),
                    camera_wb=camera_wb
                    if "pansor" in str(man).lower() or "Pansor" in str(row.get("path_noflash", ""))
                    else False,
                    half_size=half_size,
                    max_align_width=max_align_width,
                )
            except Exception as exc:
                print(f"skip {sid}: {exc}", file=sys.stderr)
                continue
            if patches is None:
                print(f"skip {sid}: no ColorChecker", file=sys.stderr)
                continue
            for i in range(24):
                rgbs.append(patches[i])
                xyzs.append(xyz_ref[i])
                weights.append(2.5 if i in SKIN_PATCH_IDX else 1.0)
                sources.append("mcc_reflectance")
                sids.append(sid)
            print(f"MCC reflectance OK: {sid}", file=sys.stderr)

    if issa_mode != "none":
        names = TONE_ISSA if issa_mode == "tone" else FULL_ISSA_BANK
        bundle = cal_dir / "iphone_calibration_bundle.json"
        with bundle.open(encoding="utf-8") as f:
            cal = json.load(f)
        if "spectral_sensitivity_rgb" not in cal:
            raise SystemExit(f"Missing spectral_sensitivity_rgb in {bundle}")
        s_arr = np.asarray(cal["spectral_sensitivity_rgb"], dtype=np.float64)
        wl = np.asarray(cal["wavelengths_nm"], dtype=np.float64)
        extra_rgb, extra_xyz = issa_skin_calibration_rows(s_arr, wl, names)
        for i in range(len(extra_rgb)):
            rgbs.append(extra_rgb[i])
            xyzs.append(extra_xyz[i])
            weights.append(float(issa_weight))
            sources.append(f"issa:{names[i]}")
            sids.append(f"ISSA_{names[i]}")
        print(f"ISSA rows: {len(extra_rgb)} ({issa_mode})", file=sys.stderr)

    if not rgbs:
        raise SystemExit("No training samples collected")
    return (
        np.stack(rgbs, axis=0),
        np.stack(xyzs, axis=0),
        np.asarray(weights, dtype=np.float64),
        sources,
        sids,
    )


def _encode_torch(
    rgb: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
    settings: ColorProjectorSettings,
) -> torch.Tensor:
    # rgb (N,3) → features with green context = G
    gctx = rgb[:, 1:2]
    feat = torch.cat([rgb, gctx], dim=1)
    feat = (feat - mean) / torch.clamp(std, min=1e-6)
    if not settings.use_fourier:
        return feat
    # mirror numpy fourier
    freqs = (2.0 ** torch.arange(settings.num_frequencies, device=feat.device, dtype=feat.dtype)) * np.pi
    angles = feat.unsqueeze(-1) * freqs.view(1, 1, -1)
    enc = torch.cat([feat.unsqueeze(-1), torch.sin(angles), torch.cos(angles)], dim=-1)
    return enc.reshape(feat.shape[0], -1)


def train_projector(
    rgb: np.ndarray,
    xyz: np.ndarray,
    weights: np.ndarray,
    affine: np.ndarray,
    settings: ColorProjectorSettings,
    *,
    epochs: int,
    batch_size: int,
    lr: float,
    seed: int,
    device: str,
    l_weight: float = 0.25,
    ab_weight: float = 1.0,
) -> ColorProjectorArtifact:
    torch.manual_seed(seed)
    np.random.seed(seed)
    device_t = torch.device(device if (device == "cpu" or torch.cuda.is_available()) else "cpu")

    feat4 = np.concatenate([rgb, rgb[:, 1:2]], axis=1)
    mean4 = feat4.mean(axis=0)
    std4 = np.maximum(feat4.std(axis=0), 1e-3)

    base = affine_rgb_to_xyz(rgb, affine)
    target_resid = xyz - base
    lab_tgt = _xyz_to_lab(xyz, D65_XYZN)

    model = ResidualColorProjector(settings).to(device_t)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)

    x_t = torch.tensor(rgb, dtype=torch.float32)
    y_t = torch.tensor(target_resid, dtype=torch.float32)
    lab_t = torch.tensor(lab_tgt, dtype=torch.float32)
    base_t = torch.tensor(base, dtype=torch.float32)
    w_t = torch.tensor(weights, dtype=torch.float32)
    mean_t = torch.tensor(mean4, dtype=torch.float32, device=device_t)
    std_t = torch.tensor(std4, dtype=torch.float32, device=device_t)
    xyzn_t = torch.tensor(D65_XYZN, dtype=torch.float32, device=device_t)
    ds = TensorDataset(x_t, y_t, lab_t, base_t, w_t)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

    def xyz_to_lab_t(xyz_b: torch.Tensor) -> torch.Tensor:
        t = xyz_b / torch.clamp(xyzn_t.view(1, 3), min=1e-12)
        d = 6.0 / 29.0
        f = torch.where(t > d**3, torch.pow(torch.clamp(t, min=0.0), 1.0 / 3.0), t / (3.0 * d**2) + 4.0 / 29.0)
        L = 116.0 * f[:, 1] - 16.0
        a = 500.0 * (f[:, 0] - f[:, 1])
        b = 200.0 * (f[:, 1] - f[:, 2])
        return torch.stack([L, a, b], dim=1)

    model.train()
    for epoch in range(epochs):
        total = 0.0
        n = 0
        for xb, yb, lab_b, base_b, wb in loader:
            xb = xb.to(device_t)
            yb = yb.to(device_t)
            lab_b = lab_b.to(device_t)
            base_b = base_b.to(device_t)
            wb = wb.to(device_t)
            feat = _encode_torch(xb, mean_t, std_t, settings)
            pred = model(feat)
            xyz_hat = base_b + pred
            lab_hat = xyz_to_lab_t(xyz_hat)
            dlab = lab_hat - lab_b
            err = (
                l_weight * dlab[:, 0].abs()
                + ab_weight * dlab[:, 1].abs()
                + ab_weight * dlab[:, 2].abs()
            )
            # Keep residual small vs affine.
            err = err + 0.05 * pred.abs().mean(dim=1)
            loss = (err * wb).sum() / wb.sum().clamp_min(1e-6)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.item()) * len(xb)
            n += len(xb)
        if (epoch + 1) % 50 == 0 or epoch == 0:
            print(f"epoch {epoch+1:4d}/{epochs}  loss={total/max(n,1):.6f}", flush=True)

    model.eval()
    weights_np, biases_np = model.export_numpy_layers()
    return ColorProjectorArtifact(
        settings=settings,
        affine_4x3=np.asarray(affine, dtype=np.float64),
        weights=weights_np,
        biases=biases_np,
        rgb_mean=mean4.astype(np.float64),
        rgb_std=std4.astype(np.float64),
        meta={},
    )


def eval_patch_metrics(
    artifact: ColorProjectorArtifact, rgb: np.ndarray, xyz: np.ndarray
) -> Dict[str, float]:
    xyz_aff = affine_rgb_to_xyz(rgb, artifact.affine_4x3)
    xyz_p = artifact(rgb)
    lab_t = _xyz_to_lab(xyz, D65_XYZN)
    lab_a = _xyz_to_lab(xyz_aff, D65_XYZN)
    lab_p = _xyz_to_lab(xyz_p, D65_XYZN)
    de_a = delta_e_2000(lab_a, lab_t)
    de_p = delta_e_2000(lab_p, lab_t)
    return {
        "affine_mean_de00": float(np.mean(de_a)),
        "affine_median_de00": float(np.median(de_a)),
        "projector_mean_de00": float(np.mean(de_p)),
        "projector_median_de00": float(np.median(de_p)),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--pansor-manifest",
        type=Path,
        default=ROOT / "data" / "pansor" / "manifest_pansor_fitskin.csv",
    )
    ap.add_argument(
        "--extra-manifest",
        type=Path,
        action="append",
        default=[],
        help="Optional additional CC manifests (tone training, etc.).",
    )
    ap.add_argument(
        "--cal-dir",
        type=Path,
        default=ROOT / "calibration" / "tier3_affine",
    )
    ap.add_argument(
        "--issa-mode",
        choices=("none", "tone", "full"),
        default="full",
        help="ISSA synthetic rows: none / tone(2) / full cheek-median bank.",
    )
    ap.add_argument("--issa-weight", type=float, default=1.5)
    ap.add_argument("--hidden-dim", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=500)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--residual-scale", type=float, default=0.04)
    ap.add_argument("--skin-patches-only", action="store_true",
                    help="Keep only MCC skin patches 0/1 (+ ISSA rows).")
    ap.add_argument("--raw-half-size", type=int, default=1)
    ap.add_argument("--max-align-width", type=int, default=1600)
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "calibration" / "color_projector_pansor",
    )
    args = ap.parse_args()

    affine_path = args.cal_dir / "camera_rgb_to_xyz_affine.npy"
    if not affine_path.is_file():
        raise SystemExit(f"Missing affine matrix: {affine_path}")
    affine = np.load(affine_path)

    rgb, xyz, weights, sources, sids = build_samples(
        pansor_manifest=args.pansor_manifest,
        extra_manifests=args.extra_manifest,
        cal_dir=args.cal_dir,
        issa_mode=args.issa_mode,
        issa_weight=args.issa_weight,
        camera_wb=True,
        half_size=args.raw_half_size,
        max_align_width=args.max_align_width,
    )
    if args.skin_patches_only:
        keep = []
        # rebuild by filtering: MCC rows come in groups of 24; keep idx 0,1 of each group + ISSA
        mcc_count = sum(1 for s in sources if s == "mcc_reflectance")
        n_trials = mcc_count // 24
        mask = np.zeros(len(sources), dtype=bool)
        for t in range(n_trials):
            base = t * 24
            mask[base] = True
            mask[base + 1] = True
        for i, s in enumerate(sources):
            if s.startswith("issa"):
                mask[i] = True
        rgb, xyz, weights = rgb[mask], xyz[mask], weights[mask]
        sources = [sources[i] for i in range(len(sources)) if mask[i]]
        sids = [sids[i] for i in range(len(sids)) if mask[i]]

    print(f"Samples: {len(rgb)}  MCC={sum(1 for s in sources if s=='mcc_reflectance')}  "
          f"ISSA={sum(1 for s in sources if s.startswith('issa'))}")

    settings = ColorProjectorSettings(
        input_dim=4,
        num_frequencies=4,
        hidden_dim=int(args.hidden_dim),
        hidden_layers=2,
        use_fourier=True,
        residual_scale=float(args.residual_scale),
    )
    artifact = train_projector(
        rgb,
        xyz,
        weights,
        affine,
        settings,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
        device=args.device,
    )
    metrics = eval_patch_metrics(artifact, rgb, xyz)
    artifact.meta = {
        "issa_mode": args.issa_mode,
        "issa_weight": args.issa_weight,
        "n_samples": int(len(rgb)),
        "n_mcc": int(sum(1 for s in sources if s == "mcc_reflectance")),
        "n_issa": int(sum(1 for s in sources if s.startswith("issa"))),
        "subjects": sorted(set(sids)),
        "train_patch_metrics": metrics,
        "cal_dir": str(args.cal_dir),
        "goal": "pansor_show",
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / "color_projector.npz"
    save_color_projector_artifact(out_path, artifact)
    (args.out_dir / "train_summary.json").write_text(
        json.dumps(artifact.meta, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2))
    print(f"Wrote {out_path}")
    print(f"Wrote {args.out_dir / 'train_summary.json'}")


if __name__ == "__main__":
    main()
