#!/usr/bin/env python3
"""
iPhone camera calibration from CameraColorProject + Lab 3 (monochromator) workflow.

Lab 3 (``Lab 3.m`` / Rooney monochromator report):
  - 400--700 nm, 10 nm steps (31 bands)
  - Mean RAW R,G,B per wavelength (CSV / RawDigger ROI)
  - PR-655 radiance ``spdAvg`` per wavelength
  - Spectral sensitivity: ``rgb ./ sum(spd)``, normalized (Eq. 1 in lab report)

CameraColorProject mapping:
  - Monochromator SPD: ``CR250_camera sensitivity/Result_cr20/SPD/{wl}nm.mat`` (``SPDavg``, ``XYZavg``)
  - iPhone RAW: ``iphone17pro_raw/*.DNG`` — 93 frames = 31 wavelengths × 3 repeats (sorted by IMG number)
  - Flash SPD: ``iphoneFlash_mea/ESPD_F*.xls`` (MK350 tab-separated export, 380--780 nm)

Outputs a **calibration bundle** (JSON + ``.npy``) for ``flash_no_flash_skin_lab.py``:
  - Measured flash CCT / Duv / linear RGB (for Lu 2006 flash locus)
  - ``camera_rgb_to_xyz`` 3×3 (least-squares from narrowband sensitivities + CIE 1931 CMFs)
  - Optional diagnostic plots
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parent
DEFAULT_CAMERA_COLOR_ROOT = Path(
    "/media/mabl-main/Data-Karthik/CameraColorProject/CameraColorProject"
)

WAVELENGTHS_NM = np.arange(400, 701, 10, dtype=np.int32)
_EPS = 1e-12


@dataclass
class CalibrationBundle:
    """Serialized iPhone calibration for the flash/no-flash pipeline."""

    source_root: str
    device_label: str
    wavelengths_nm: List[int]
    spectral_sensitivity_rgb: List[List[float]]  # (31, 3) R,G,B normalized
    monochromator_spd_scalar: List[float]  # integrated scalar per band (Lab 3 style)
    flash_cct_k: float
    flash_duv: float
    flash_xyz: List[float]
    flash_rgb_linear: List[float]  # relative, norm=1
    flash_spd_wl_nm: List[float]
    flash_spd_power: List[float]
    camera_rgb_to_xyz: List[List[float]]  # 3×3
    lab3_method: str
    notes: str

    def save(self, out_dir: Path) -> Path:
        out_dir.mkdir(parents=True, exist_ok=True)
        bundle_path = out_dir / "iphone_calibration_bundle.json"
        payload = asdict(self)
        with bundle_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        M = np.asarray(self.camera_rgb_to_xyz, dtype=np.float64)
        np.save(out_dir / "camera_rgb_to_xyz.npy", M)
        np.save(out_dir / "spectral_sensitivity_rgb.npy", np.asarray(self.spectral_sensitivity_rgb))
        return bundle_path


def load_calibration_bundle(path: Path) -> CalibrationBundle:
    from dataclasses import fields

    p = Path(path)
    if p.is_dir():
        p = p / "iphone_calibration_bundle.json"
    with p.open(encoding="utf-8") as f:
        d = json.load(f)
    allowed = {f.name for f in fields(CalibrationBundle)}
    return CalibrationBundle(**{k: v for k, v in d.items() if k in allowed})


def parse_mk350_espd(path: Path) -> Dict[str, Any]:
    """Parse MK350 ``ESPD_F*.xls`` (tab-separated text) → metadata + SPD."""
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    meta: Dict[str, float] = {}
    wl: List[float] = []
    spd: List[float] = []
    for line in lines:
        if "\t" not in line:
            continue
        key, val = line.split("\t", 1)
        key = key.strip()
        val = val.strip()
        m = re.match(r"^(\d+)nm\(mW/m\^2\)$", key)
        if m:
            wl.append(float(m.group(1)))
            spd.append(float(val))
            continue
        try:
            meta[key] = float(val)
        except ValueError:
            meta[key] = val
    return {
        "file": path.name,
        "cct_k": float(meta.get("CCT(K)", float("nan"))),
        "duv": float(meta.get("Duv", 0.0)),
        "X": float(meta.get("X", float("nan"))),
        "Y": float(meta.get("Y", float("nan"))),
        "Z": float(meta.get("Z", float("nan"))),
        "wl_nm": np.asarray(wl, dtype=np.float64),
        "spd": np.asarray(spd, dtype=np.float64),
    }


def load_monochromator_spd_table(spd_dir: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Load CR250 ``{wl}nm.mat`` files → (31,) integrated radiance, (31,3) XYZ."""
    import scipy.io as sio

    scalars = []
    xyz_rows = []
    for wl in WAVELENGTHS_NM:
        p = spd_dir / f"{wl}nm.mat"
        if not p.is_file():
            raise FileNotFoundError(f"Missing monochromator SPD: {p}")
        d = sio.loadmat(str(p))
        spd = np.asarray(d["SPDavg"], dtype=np.float64).reshape(-1)
        xyz = np.asarray(d["XYZavg"], dtype=np.float64).reshape(3)
        scalars.append(float(np.sum(spd)))  # Lab 3.m: sum(spdAvg(:,1))
        xyz_rows.append(xyz)
    return np.asarray(scalars, dtype=np.float64), np.asarray(xyz_rows, dtype=np.float64)


def read_dng_mean_linear_rgb(
    path: Path,
    *,
    roi_fraction: float = 0.15,
    half_size: int = 1,
    use_camera_wb: bool = False,
) -> np.ndarray:
    """Mean linear camera RGB in central ROI (rawpy), Lab 3 / RawDigger analogue."""
    import rawpy

    with rawpy.imread(str(path)) as raw:
        rgb = raw.postprocess(
            use_camera_wb=use_camera_wb,
            half_size=half_size,
            no_auto_bright=True,
            output_color=rawpy.ColorSpace.raw,
            gamma=(1, 1),
            user_flip=0,
        )
    rgb = np.asarray(rgb, dtype=np.float64)
    h, w = rgb.shape[:2]
    fh = max(8, int(h * roi_fraction))
    fw = max(8, int(w * roi_fraction))
    cy, cx = h // 2, w // 2
    roi = rgb[cy - fh : cy + fh, cx - fw : cx + fw]
    return np.mean(roi.reshape(-1, 3), axis=0)


def load_iphone_mono_rgb_series(
    dng_dir: Path,
    *,
    repeats_per_wl: int = 3,
    dng_start_index: int = 0,
    half_size: int = 1,
    use_camera_wb: bool = False,
) -> np.ndarray:
    """
    Sorted DNG list → (31, 3) mean linear RGB.

    Default: first 93 files, 3 consecutive frames per wavelength (CameraColorProject capture).
    """
    files = sorted(dng_dir.glob("*.DNG")) + sorted(dng_dir.glob("*.dng"))
    need = len(WAVELENGTHS_NM) * repeats_per_wl
    block = files[dng_start_index : dng_start_index + need]
    if len(block) < need:
        raise ValueError(
            f"Need {need} DNGs from index {dng_start_index}; found {len(block)} in {dng_dir}"
        )
    rgb = np.zeros((len(WAVELENGTHS_NM), 3), dtype=np.float64)
    for i, wl in enumerate(WAVELENGTHS_NM):
        chunk = block[i * repeats_per_wl : (i + 1) * repeats_per_wl]
        means = [
            read_dng_mean_linear_rgb(p, half_size=half_size, use_camera_wb=use_camera_wb)
            for p in chunk
        ]
        rgb[i] = np.mean(means, axis=0)
    return rgb


def spectral_sensitivity_lab3(rgb: np.ndarray, spd_scalar: np.ndarray) -> np.ndarray:
    """``spectralsens = rgb ./ spd;`` normalized by max (Lab 3.m)."""
    denom = np.maximum(spd_scalar.reshape(-1, 1), _EPS)
    sens = rgb / denom
    mx = float(np.max(sens))
    if mx > _EPS:
        sens = sens / mx
    return sens


def _cie_cmfs_1nm() -> Tuple[np.ndarray, np.ndarray]:
    """CIE 1931 2° CMFs sampled at 10 nm (400--700 nm)."""
    wl = WAVELENGTHS_NM.astype(np.float64)
    # CIE 1931 2° at 10 nm (approximate, sufficient for narrowband LS fit)
    x_bar = np.array(
        [0.0143, 0.0435, 0.1344, 0.2839, 0.3483, 0.3362, 0.2908, 0.1954, 0.0956, 0.0320,
         0.0049, 0.0093, 0.0633, 0.1655, 0.2904, 0.4334, 0.5945, 0.7621, 0.9163, 1.0263,
         1.0622, 1.0026, 0.8544, 0.6424, 0.4479, 0.2835, 0.1649, 0.0874, 0.0468, 0.0227, 0.0114],
        dtype=np.float64,
    )
    y_bar = np.array(
        [0.0004, 0.0012, 0.0040, 0.0116, 0.0230, 0.0380, 0.0600, 0.0910, 0.1390, 0.2080,
         0.3230, 0.5030, 0.7100, 0.8620, 0.9540, 0.9950, 0.9950, 0.9520, 0.8700, 0.7570,
         0.6310, 0.5030, 0.3810, 0.2650, 0.1750, 0.1070, 0.0610, 0.0320, 0.0170, 0.0082, 0.0041],
        dtype=np.float64,
    )
    z_bar = np.array(
        [0.0679, 0.2074, 0.6456, 1.3856, 1.7471, 1.7721, 1.6692, 1.2876, 0.8130, 0.4652,
         0.2720, 0.1582, 0.0782, 0.0422, 0.0203, 0.0087, 0.0039, 0.0021, 0.0017, 0.0011,
         0.0008, 0.0003, 0.0002, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
        dtype=np.float64,
    )
    return wl, np.stack([x_bar, y_bar, z_bar], axis=1)


def fit_camera_rgb_to_xyz(
    rgb: np.ndarray,
    spd_scalar: np.ndarray,
) -> np.ndarray:
    """
    Narrowband model: at each λ, XYZ ∝ CMF(λ) * spd(λ), camera RGB ∝ sensitivity * spd.

    Solve ``XYZ ≈ (rgb / spd) @ M.T`` with least squares across 31 bands.
    """
    _, cmf = _cie_cmfs_1nm()
    weights = np.maximum(spd_scalar.reshape(-1, 1), _EPS)
    # Target XYZ per band: Y=1 scale from CMFs * scalar radiance
    xyz_tgt = cmf * weights
    xyz_tgt = xyz_tgt / np.maximum(xyz_tgt[:, 1:2], _EPS)
    # Design: normalized camera response per band
    rgb_n = rgb / weights
    M, _, _, _ = np.linalg.lstsq(rgb_n, xyz_tgt, rcond=None)
    return M.T  # rgb @ M.T -> xyz


def spd_to_xyz_and_rgb(
    wl_nm: np.ndarray,
    spd: np.ndarray,
) -> Tuple[np.ndarray, float, float, np.ndarray]:
    """Integrate SPD → XYZ, CCT/Duv; relative linear sRGB for Lu flash vector."""
    wl = np.asarray(wl_nm, dtype=np.float64).reshape(-1)
    p = np.maximum(np.asarray(spd, dtype=np.float64).reshape(-1), 0.0)
    try:
        import colour
        from colour import XYZ_to_xy, xy_to_CCT
        from colour.models import UCS_to_uv, XYZ_to_UCS

        sd = colour.SpectralDistribution(dict(zip(wl, p)))
        cmfs = colour.MSDS_CMFS["CIE 1931 2 Degree Standard Observer"]
        xyz_abs = np.asarray(colour.sd_to_XYZ(sd, cmfs), dtype=np.float64).reshape(3)
        xyz = xyz_abs / max(float(xyz_abs[1]), _EPS)
        xy = XYZ_to_xy(xyz.reshape(1, 3)).reshape(2)
        cct = float(xy_to_CCT(xy.reshape(1, 2))[0])
        duv = 0.0
        rgb = colour.XYZ_to_RGB(
            xyz,
            colour.RGB_COLOURSPACES["sRGB"],
            illuminant=colour.RGB_COLOURSPACES["sRGB"].whitepoint,
            matrix_XYZ_to_RGB=colour.RGB_COLOURSPACES["sRGB"].matrix_XYZ_to_RGB,
        )
        rgb = np.maximum(rgb, _EPS)
        rgb = rgb / max(float(np.linalg.norm(rgb)), _EPS)
        return xyz, cct, duv, rgb
    except Exception:
        xyz = np.array([1.0, 1.0, 1.0], dtype=np.float64)
        return xyz, 4900.0, 0.0, np.array([1.0, 1.0, 1.0]) / np.sqrt(3.0)


def build_calibration_bundle(
    camera_color_root: Path,
    *,
    spd_dir: Optional[Path] = None,
    dng_dir: Optional[Path] = None,
    flash_dir: Optional[Path] = None,
    device_label: str = "iPhone 17 Pro (monochromator + MK350 flash)",
    dng_start_index: int = 0,
    repeats_per_wl: int = 3,
    raw_half_size: int = 1,
    raw_camera_wb: bool = False,
) -> CalibrationBundle:
    root = Path(camera_color_root)
    spd_dir = spd_dir or root / "CR250_camera sensitivity" / "Result_cr20" / "SPD"
    dng_dir = dng_dir or root / "CR250_camera sensitivity" / "iphone17pro_raw"
    flash_dir = flash_dir or root / "iphoneFlash_mea"

    spd_scalar, _xyz_mono = load_monochromator_spd_table(spd_dir)
    rgb = load_iphone_mono_rgb_series(
        dng_dir,
        repeats_per_wl=repeats_per_wl,
        dng_start_index=dng_start_index,
        half_size=raw_half_size,
        use_camera_wb=raw_camera_wb,
    )
    sens = spectral_sensitivity_lab3(rgb, spd_scalar)
    M = fit_camera_rgb_to_xyz(rgb, spd_scalar)

    flash_files = sorted(flash_dir.glob("ESPD_F*.xls"))
    if not flash_files:
        raise FileNotFoundError(f"No ESPD_F*.xls in {flash_dir}")
    flash_meas = [parse_mk350_espd(p) for p in flash_files]
    wl_f = flash_meas[0]["wl_nm"]
    spd_mean = np.mean([m["spd"] for m in flash_meas], axis=0)
    cct_mean = float(np.mean([m["cct_k"] for m in flash_meas]))
    duv_mean = float(np.mean([m["duv"] for m in flash_meas]))
    xyz_f, cct_i, duv_i, rgb_f = spd_to_xyz_and_rgb(wl_f, spd_mean)

    return CalibrationBundle(
        source_root=str(root),
        device_label=device_label,
        wavelengths_nm=WAVELENGTHS_NM.tolist(),
        spectral_sensitivity_rgb=sens.tolist(),
        monochromator_spd_scalar=spd_scalar.tolist(),
        flash_cct_k=cct_mean,
        flash_duv=duv_mean,
        flash_xyz=xyz_f.tolist(),
        flash_rgb_linear=rgb_f.tolist(),
        flash_spd_wl_nm=wl_f.tolist(),
        flash_spd_power=spd_mean.tolist(),
        camera_rgb_to_xyz=M.tolist(),
        lab3_method="rgb/spd_scalar per 10nm band; same as Lab 3.m spectralsens = rgb./spd",
        notes=(
            f"DNG block start index {dng_start_index}, {repeats_per_wl} frames/band; "
            f"flash averaged over {len(flash_files)} MK350 captures."
        ),
    )


def plot_lab3_style_figures(bundle: CalibrationBundle, out_dir: Path, *, dpi: int = 160) -> None:
    """Reproduce Lab 3 Figures 1--3 (RGB response, SPD, spectral sensitivity)."""
    import matplotlib.pyplot as plt

    wl = np.asarray(bundle.wavelengths_nm, dtype=np.float64)
    rgb = bundle.spectral_sensitivity_rgb
    # Recover unnorm rgb proxy: sens * spd
    spd = np.asarray(bundle.monochromator_spd_scalar, dtype=np.float64)
    sens = np.asarray(rgb, dtype=np.float64)
    rgb_resp = sens * spd.reshape(-1, 1)
    rgb_resp = rgb_resp / max(float(np.max(rgb_resp)), _EPS)

    out_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(wl, rgb_resp[:, 0], "r-", label="R")
    ax.plot(wl, rgb_resp[:, 1], "g-", label="G")
    ax.plot(wl, rgb_resp[:, 2], "b-", label="B")
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("RAW RGB code response (relative)")
    ax.set_title("Figure 1 style: monochromator RGB response")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "lab3_rgb_response.png", dpi=dpi)
    plt.close(fig)

    spd_n = spd / max(float(np.max(spd)), _EPS)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(wl, spd_n, "k-")
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Normalized radiance (CR250)")
    ax.set_title("Figure 2 style: monochromator SPD")
    fig.tight_layout()
    fig.savefig(out_dir / "lab3_monochromator_spd.png", dpi=dpi)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(wl, sens[:, 0], "r-", label="R")
    ax.plot(wl, sens[:, 1], "g-", label="G")
    ax.plot(wl, sens[:, 2], "b-", label="B")
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Spectral sensitivity (normalized)")
    ax.set_title("Figure 3 style: camera spectral sensitivity")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "lab3_spectral_sensitivity.png", dpi=dpi)
    plt.close(fig)

    # Flash SPD
    wl_f = np.asarray(bundle.flash_spd_wl_nm)
    p_f = np.asarray(bundle.flash_spd_power)
    p_f = p_f / max(float(np.max(p_f)), _EPS)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(wl_f, p_f, "k-")
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Normalized flash SPD (MK350 mean)")
    ax.set_title(f"iPhone flash SPD (CCT≈{bundle.flash_cct_k:.0f} K)")
    fig.tight_layout()
    fig.savefig(out_dir / "iphone_flash_spd.png", dpi=dpi)
    plt.close(fig)
