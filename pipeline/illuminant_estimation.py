"""Flash/no-flash illuminant estimation for variable-lighting CAT.

Uses MK350 torch SPD prior + Lu & Drew (2006) ambient CCT from pure-flash residual.
Optional ECC alignment and Lu/no-flash chroma fusion.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import cv2
import numpy as np

from flash_noflash_spectral import planck_xyz_y1
from vendor.flash_align.align_pair import AlignResult, estimate_exposure_scale
from vendor.flash_align.lu2006_ambient import (
    Lu2006Result,
    _planck_rgb_linear,
    estimate_ambient_lu2006,
)

_EPS = 1e-8
DEFAULT_RING_XY: Dict[str, Tuple[float, float]] = {
    "D65": (0.309932, 0.324021),
    "F12": (0.4297, 0.3914),
}
ORACLE_CCT: Dict[str, float] = {"D65": 6500.0, "F12": 3000.0, "D12": 3000.0}


def infer_illuminant_label(path: Union[str, Path]) -> Optional[str]:
    """Infer D65 / F12 from zip path, parent folder, or filename stem."""
    p = Path(path).expanduser().resolve()
    for parent in p.parents:
        name = parent.name.upper()
        if name == "F12":
            return "F12"
        if name in ("D65", "D65 "):
            return "D65"
    stem = p.stem.upper().replace(" ", "")
    if "D65" in stem:
        return "D65"
    if "F12" in stem or "D12" in stem:
        return "F12"
    if re.search(r"(^|[-_])F12([-_]|$)|D12", stem):
        return "F12"
    if re.search(r"(^|[-_])D65([-_]|$)", stem):
        return "D65"
    return None


@dataclass
class TorchPrior:
    torch_cct_k: float
    flash_rgb: np.ndarray  # unit-norm linear RGB
    wavelengths_nm: np.ndarray
    mean_spd: np.ndarray
    n_files: int
    files: Tuple[str, ...]


def parse_spectrum_file(path: Path) -> Tuple[Dict[str, Any], np.ndarray, np.ndarray]:
    metadata: Dict[str, Any] = {}
    wavelengths: list[float] = []
    values: list[float] = []
    spectral_line = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*nm\s*[\t, ]+\s*(-?\d+(?:\.\d+)?)\s*$")
    with Path(path).open("r", errors="replace") as f:
        for line in f:
            text = line.strip()
            if not text:
                continue
            match = spectral_line.match(text)
            if match:
                wavelengths.append(float(match.group(1)))
                values.append(float(match.group(2)))
                continue
            parts = re.split(r"\t+", text)
            if len(parts) >= 2:
                key = parts[0].strip()
                val_text = parts[1].strip()
                try:
                    metadata[key] = float(val_text)
                except ValueError:
                    metadata[key] = val_text
    return metadata, np.array(wavelengths, dtype=float), np.array(values, dtype=float)


def spd_to_unit_linear_rgb(wavelengths_nm: np.ndarray, spd: np.ndarray) -> np.ndarray:
    """Integrate SPD → relative linear sRGB (unit norm)."""
    try:
        import colour

        wl = np.asarray(wavelengths_nm, dtype=float)
        sp = np.maximum(np.asarray(spd, dtype=float), 0.0)
        if wl.shape[0] != sp.shape[0]:
            sp = np.interp(wl, wl, sp)
        sd_dict = {int(round(w)): float(v) for w, v in zip(wl, sp)}
        sd = colour.SpectralDistribution(sd_dict)
        cmfs = colour.MSDS_CMFS["CIE 1931 2 Degree Standard Observer"]
        xyz = colour.sd_to_XYZ(sd, cmfs)
        xyz = xyz / max(float(xyz[1]), _EPS)
        cs = colour.RGB_COLOURSPACES["sRGB"]
        rgb = colour.XYZ_to_RGB(
            xyz,
            cs,
            illuminant=cs.whitepoint,
            matrix_XYZ_to_RGB=cs.matrix_XYZ_to_RGB,
        )
        rgb = np.maximum(np.asarray(rgb, dtype=np.float64), _EPS)
        return rgb / max(float(np.linalg.norm(rgb)), _EPS)
    except Exception:
        cct = 4917.0
        return _planck_rgb_linear(cct)


def load_torch_prior(torch_dir: Path) -> TorchPrior:
    torch_dir = Path(torch_dir)
    files = sorted(torch_dir.glob("ESPD_T0*.xls")) or sorted(torch_dir.glob("ESPD_T*.xls"))[:3]
    if not files:
        raise FileNotFoundError(f"No torch ESPD files in {torch_dir}")
    spectra = []
    header_ccts = []
    wl_ref: Optional[np.ndarray] = None
    for path in files:
        meta, wl, spd = parse_spectrum_file(path)
        spectra.append(spd)
        wl_ref = wl
        if "CCT" in meta:
            header_ccts.append(float(meta["CCT"]))
    assert wl_ref is not None
    mean_spd = np.mean(np.stack(spectra, axis=0), axis=0)
    flash_rgb = spd_to_unit_linear_rgb(wl_ref, mean_spd)
    cct_k = float(np.nanmean(header_ccts)) if header_ccts else 4917.0
    try:
        import colour

        sd_dict = {int(round(w)): float(v) for w, v in zip(wl_ref, mean_spd)}
        sd = colour.SpectralDistribution(sd_dict)
        xyz = colour.sd_to_XYZ(sd, colour.MSDS_CMFS["CIE 1931 2 Degree Standard Observer"])
        cct_k = float(colour.xy_to_CCT(colour.XYZ_to_xy(xyz), method="McCamy 1992"))
    except Exception:
        pass
    return TorchPrior(
        torch_cct_k=cct_k,
        flash_rgb=flash_rgb,
        wavelengths_nm=wl_ref,
        mean_spd=mean_spd,
        n_files=len(files),
        files=tuple(p.name for p in files),
    )


def load_torch_prior_from_cal_bundle(cal_dir: Path) -> TorchPrior:
    """Fallback torch SPD prior from ``tier3_affine/iphone_calibration_bundle.json``."""
    import json

    bundle_path = Path(cal_dir).expanduser().resolve() / "iphone_calibration_bundle.json"
    if not bundle_path.is_file():
        raise FileNotFoundError(f"No iphone_calibration_bundle.json under {cal_dir}")
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    flash_rgb = np.asarray(bundle["flash_rgb_linear"], dtype=np.float64)
    flash_rgb = flash_rgb / max(float(np.linalg.norm(flash_rgb)), _EPS)
    cct_k = float(bundle.get("flash_cct_k", 4917.0))
    wl = np.asarray(bundle.get("flash_spd_wl_nm") or [], dtype=float)
    spd = np.asarray(bundle.get("flash_spd_power") or [], dtype=float)
    if wl.size == 0 or spd.size == 0 or wl.shape != spd.shape:
        wl = np.linspace(380.0, 780.0, 401)
        spd = np.ones_like(wl)
    return TorchPrior(
        torch_cct_k=cct_k,
        flash_rgb=flash_rgb,
        wavelengths_nm=wl,
        mean_spd=spd,
        n_files=0,
        files=(str(bundle_path.name),),
    )


def _to_gray01(linear_rgb: np.ndarray) -> np.ndarray:
    g = 0.2126 * linear_rgb[..., 0] + 0.7152 * linear_rgb[..., 1] + 0.0722 * linear_rgb[..., 2]
    return np.clip(g, 0.0, 1.0).astype(np.float32)


def align_flash_linear(
    noflash: np.ndarray,
    flash: np.ndarray,
    *,
    cheek_mask: Optional[np.ndarray] = None,
    use_ecc: bool = True,
) -> AlignResult:
    """ECC warp flash→no-flash, then exposure match (cheek mask if given)."""
    nf = np.asarray(noflash, dtype=np.float64)
    fl = np.asarray(flash, dtype=np.float64)
    if use_ecc:
        g0 = _to_gray01(nf)
        g1 = _to_gray01(fl)
        warp = np.eye(2, 3, dtype=np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 500, 1e-6)
        try:
            cc, warp = cv2.findTransformECC(g0, g1, warp, cv2.MOTION_EUCLIDEAN, criteria, None, 5)
        except cv2.error:
            cc = 0.0
        h, w = g0.shape
        fl = cv2.warpAffine(
            fl.astype(np.float32),
            warp,
            (w, h),
            flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_REPLICATE,
        ).astype(np.float64)
    else:
        cc = 0.0
        warp = np.eye(2, 3, dtype=np.float32)

    if cheek_mask is not None and int(np.count_nonzero(cheek_mask)) >= 100:
        m = cheek_mask > 0
        g0 = _to_gray01(nf)
        g1 = _to_gray01(fl)
        scale = float(np.median(g0[m]) / max(float(np.median(g1[m])), _EPS))
        scale = float(np.clip(scale, 0.25, 4.0))
    else:
        scale = estimate_exposure_scale(nf, fl)
    fl_scaled = np.clip(fl * scale, 0.0, None)
    return AlignResult(
        noflash_linear=nf,
        flash_aligned_linear=fl_scaled,
        warp_matrix=warp,
        exposure_scale=float(scale),
        ecc_cc=float(cc),
    )


def chroma_to_planck_cct(rgb: np.ndarray) -> float:
    rgb = np.maximum(np.asarray(rgb, dtype=np.float64).reshape(3), _EPS)
    rgb = rgb / rgb.sum()
    temps = np.arange(2300, 10001, 250, dtype=np.float64)
    best_cct = float(temps[0])
    best_cos = -2.0
    for t in temps:
        ref = _planck_rgb_linear(float(t))
        ref = ref / max(float(np.linalg.norm(ref)), _EPS)
        cos = float(np.dot(rgb, ref) / (np.linalg.norm(rgb) + _EPS))
        if cos > best_cos:
            best_cos = cos
            best_cct = float(t)
    return best_cct


def cct_to_xy(cct_k: float) -> Tuple[float, float]:
    xyz = planck_xyz_y1(float(cct_k), 0.0)
    s = float(np.sum(xyz))
    return float(xyz[0] / s), float(xyz[1] / s)


def xy_to_cct_mccamy(x: float, y: float) -> float:
    try:
        import colour

        return float(colour.xy_to_CCT((float(x), float(y)), method="McCamy 1992"))
    except Exception:
        return chroma_to_planck_cct(np.array([x, y, max(1.0 - x - y, _EPS)]))


def blend_xy(
    xy_a: Tuple[float, float],
    xy_b: Tuple[float, float],
    weight_a: float = 0.8,
) -> Tuple[float, float]:
    w = float(np.clip(weight_a, 0.0, 1.0))
    return (
        w * xy_a[0] + (1.0 - w) * xy_b[0],
        w * xy_a[1] + (1.0 - w) * xy_b[1],
    )


def xy_white_y1(x: float, y: float) -> np.ndarray:
    x, y = float(x), float(y)
    if y <= _EPS:
        return planck_xyz_y1(5500.0)
    return np.array([x / y, 1.0, (1.0 - x - y) / y], dtype=np.float64)


def estimate_lu(
    align: AlignResult,
    torch_prior: TorchPrior,
    *,
    use_measured_spd_rgb: bool = False,
) -> Lu2006Result:
    kwargs: Dict[str, Any] = {"measured_flash_cct_k": torch_prior.torch_cct_k}
    if use_measured_spd_rgb:
        kwargs["flash_rgb_measured"] = torch_prior.flash_rgb
    return estimate_ambient_lu2006(align, **kwargs)


def fused_lu_nf_cct(
    lu_cct: float,
    nf_rgb_median: np.ndarray,
    *,
    lu_weight: float = 0.8,
) -> float:
    lu_xy = cct_to_xy(lu_cct)
    nf_xy = cct_to_xy(chroma_to_planck_cct(nf_rgb_median))
    fused = blend_xy(lu_xy, nf_xy, weight_a=lu_weight)
    return xy_to_cct_mccamy(fused[0], fused[1])


def hybrid_deploy_cct(
    illuminant_label: str,
    lu_cct: float,
    *,
    warm_cct_threshold: float = 4500.0,
) -> float:
    """Use Lu estimate on warm / F12; frozen 5500 on D65-like captures."""
    ill = str(illuminant_label).upper()
    if ill in ("F12", "D12") or float(lu_cct) < warm_cct_threshold:
        return float(lu_cct)
    return 5500.0


def illuminant_vs_mk350(
    estimated_cct: float,
    mk350_xy: Tuple[float, float],
) -> Dict[str, float]:
    """Errors vs in-situ MK350 ring measurement at the face."""
    mk_cct = xy_to_cct_mccamy(mk350_xy[0], mk350_xy[1])
    est_xy = cct_to_xy(estimated_cct)
    du = float(estimated_cct) - mk_cct
    dv = float(est_xy[1] - mk350_xy[1])
    return {
        "mk350_cct_k": mk_cct,
        "delta_cct_k": du,
        "abs_delta_cct_k": abs(du),
        "delta_y": dv,
    }


def scene_white_for_cat(
    mode: str,
    *,
    illuminant_label: str,
    lu_cct: float,
    mk350_xy: Optional[Tuple[float, float]] = None,
    fused_cct: Optional[float] = None,
    hybrid_cct: Optional[float] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Resolve Bradford W_src for a named CAT mode."""
    meta: Dict[str, Any] = {"cat_mode": mode}
    if mode == "frozen_5500":
        cct = 5500.0
        xyz = planck_xyz_y1(cct)
        meta["cat_cct"] = cct
    elif mode == "mk350_ring_xy" and mk350_xy is not None:
        xyz = xy_white_y1(mk350_xy[0], mk350_xy[1])
        meta["cat_xy"] = mk350_xy
        meta["cat_cct"] = xy_to_cct_mccamy(mk350_xy[0], mk350_xy[1])
    elif mode == "hybrid_deploy":
        cct = float(hybrid_cct if hybrid_cct is not None else hybrid_deploy_cct(illuminant_label, lu_cct))
        xyz = planck_xyz_y1(cct)
        meta["cat_cct"] = cct
    elif mode in ("lu_fused",) and fused_cct is not None:
        cct = float(fused_cct)
        xyz = planck_xyz_y1(cct)
        meta["cat_cct"] = cct
    else:
        cct = float(lu_cct)
        xyz = planck_xyz_y1(cct)
        meta["cat_cct"] = cct
    return xyz, meta
