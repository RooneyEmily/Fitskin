"""D65-FairFace7-ROI inference pipeline (deployable entry point).

Frozen color path:
  pre-AWB flash/no-flash reflectance → tier3 affine RGB→XYZ →
  Bradford CAT 5500K→D65 → skin Lab on Apple Vision ROI.

ROI:
  ``roi='forehead'`` (FitSkin scan site) or ``roi='cheek'`` (legacy Pansor path).
  FairFace-7 routes ``specular_tone`` L* sampling on cheek and forehead (same rules
  as Pansor). On uniform forehead Lab (narrow L* spread), cheek pixels are pooled
  for specular_tone when illuminant is F12 or FairFace ethnicity is Indian.

No FitSkin, demographics, SCR-AWB, Lab correctors, or person-specific gates.
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Union

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline.illuminant_estimation import (  # noqa: E402
    TorchPrior,
    align_flash_linear,
    estimate_lu,
    hybrid_deploy_cct,
    infer_illuminant_label,
    load_torch_prior,
    load_torch_prior_from_cal_bundle,
    scene_white_for_cat,
)
from pipeline.post_corrections import (  # noqa: E402
    apply_lab_affine,
    apply_multi_illuminant_lab_affine,
)
from models.fairface_race import FairFacePredictor, face_rgb_crop_from_landmarks  # noqa: E402
from pipeline.skin_roi import (  # noqa: E402
    apple_face_forehead_lab_pool_mask,
    apple_face_skin_roi_mask,
)
from scripts.evaluate_pansor20_chartfree_d65 import (  # noqa: E402
    FOREHEAD_L_UNIFORM_STD,
    FOREHEAD_SKIN_LAB_TRIM,
    extract_zip,
    linear_rgb_to_preview_bgr,
    load_affine,
    load_apple_landmarks,
    load_dng_linear,
    match_flash_exposure,
    mean_lab_on_mask,
    probe_forehead_lab_l_std,
)

DEFAULT_CAL_DIR = ROOT / "calibration" / "tier3_affine"
DEFAULT_FAIRFACE_DIR = ROOT / "calibration" / "fairface"
DEFAULT_TORCH_DIR = Path.home() / "Downloads" / "Torch_meas-20260829T192551Z-1-001 (2)" / "Torch_meas"

# Soft capture gate (camera-settings note)
EXPOSURE_ISO_REJECT = 200.0
EXPOSURE_SHUTTER_REJECT_S = 1.0 / 60.0
EXPOSURE_L_REJECT = 75.0


def exposure_flags(
    *,
    iso: Optional[float] = None,
    shutter_s: Optional[float] = None,
    pipeline_L: Optional[float] = None,
) -> Dict[str, Any]:
    """Soft reject/re-prompt flags from EXIF + pipeline L*."""
    flags = {
        "iso_ge_200": False,
        "shutter_ge_1_60": False,
        "L_ge_75": False,
        "out_of_band": False,
    }
    if iso is not None and np.isfinite(iso):
        flags["iso_ge_200"] = float(iso) >= EXPOSURE_ISO_REJECT
    if shutter_s is not None and np.isfinite(shutter_s):
        flags["shutter_ge_1_60"] = float(shutter_s) >= EXPOSURE_SHUTTER_REJECT_S
    if pipeline_L is not None and np.isfinite(pipeline_L):
        flags["L_ge_75"] = float(pipeline_L) >= EXPOSURE_L_REJECT
    flags["out_of_band"] = bool(
        flags["iso_ge_200"] or flags["shutter_ge_1_60"] or flags["L_ge_75"]
    )
    return flags


@dataclass
class D65FairFace7ROIPipeline:
    """Load-once affine + FairFace-7 runner for chart-free skin Lab."""

    M: np.ndarray
    fairface: Optional[FairFacePredictor]
    fixed_cat_k: float = 5500.0
    cat_mode: str = "frozen_5500"  # frozen_5500 | lu_torch | hybrid_deploy
    torch_prior: Optional[TorchPrior] = None
    lab_affine_W: Optional[np.ndarray] = None
    multi_lab_bundle: Optional[Dict[str, Any]] = None
    color_projector: Optional[Any] = None
    half_size: bool = True
    cat_degree: float = 1.0
    sampling: str = "fairface7"  # fairface7 | off
    roi: str = "forehead"  # forehead | cheek
    cal_dir: Optional[Path] = None
    fairface_dir: Optional[Path] = None

    @classmethod
    def from_defaults(
        cls,
        *,
        cal_dir: Optional[Path] = None,
        fairface_dir: Optional[Path] = None,
        fixed_cat_k: float = 5500.0,
        cat_mode: str = "frozen_5500",
        torch_dir: Optional[Path] = None,
        lab_affine: Optional[Path] = None,
        multi_lab_affine: Optional[Path] = None,
        color_projector: Optional[Path] = None,
        half_size: bool = True,
        sampling: str = "fairface7",
        roi: str = "forehead",
        cat_degree: float = 1.0,
    ) -> "D65FairFace7ROIPipeline":
        cal = Path(cal_dir or DEFAULT_CAL_DIR).expanduser().resolve()
        ff_dir = Path(fairface_dir or DEFAULT_FAIRFACE_DIR).expanduser().resolve()
        M = load_affine(cal)
        fairface: Optional[FairFacePredictor] = None
        if sampling in ("fairface", "fairface7", "fairface4"):
            mode = "4" if sampling == "fairface4" else "7"
            fairface = FairFacePredictor.load(mode=mode, weights_dir=ff_dir)
        torch_prior: Optional[TorchPrior] = None
        if cat_mode in ("lu_torch", "hybrid_deploy"):
            td = Path(torch_dir or DEFAULT_TORCH_DIR).expanduser().resolve()
            try:
                torch_prior = load_torch_prior(td)
            except FileNotFoundError:
                torch_prior = load_torch_prior_from_cal_bundle(cal)
                print(
                    f"Torch_meas not found at {td}; using flash SPD from {cal}/iphone_calibration_bundle.json",
                    file=sys.stderr,
                )
        lab_W: Optional[np.ndarray] = None
        if lab_affine is not None:
            from pipeline.post_corrections import load_lab_affine

            lab_W = load_lab_affine(lab_affine)
        multi_bundle: Optional[Dict[str, Any]] = None
        if multi_lab_affine is not None:
            from pipeline.post_corrections import load_multi_illuminant_lab_affine

            multi_bundle = load_multi_illuminant_lab_affine(multi_lab_affine)
        projector = None
        if color_projector is not None:
            from pipeline.post_corrections import load_color_projector

            projector = load_color_projector(color_projector)
        return cls(
            M=M,
            fairface=fairface,
            fixed_cat_k=float(fixed_cat_k),
            cat_mode=str(cat_mode),
            torch_prior=torch_prior,
            lab_affine_W=lab_W,
            multi_lab_bundle=multi_bundle,
            color_projector=projector,
            half_size=bool(half_size),
            cat_degree=float(cat_degree),
            sampling=str(sampling),
            roi=str(roi),
            cal_dir=cal,
            fairface_dir=ff_dir,
        )

    def run_zip(
        self,
        zip_path: Union[str, Path],
        *,
        work_dir: Optional[Path] = None,
        keep_extract: bool = False,
    ) -> Dict[str, Any]:
        """Run on a Pansor-style flash/no-flash zip with Apple landmarks."""
        zip_path = Path(zip_path).expanduser().resolve()
        if work_dir is None:
            tmp = Path(tempfile.mkdtemp(prefix="d65_ff7_"))
            cleanup = not keep_extract
            extract_dir = tmp / zip_path.stem
        else:
            extract_dir = Path(work_dir).expanduser().resolve() / zip_path.stem
            cleanup = False
        try:
            nf, fl, lm = extract_zip(zip_path, extract_dir)
            ill = infer_illuminant_label(zip_path)
            out = self.run_files(nf, fl, lm, illuminant_label=ill)
            out["zip_path"] = str(zip_path)
            out["zip_stem"] = zip_path.stem
            if ill:
                out["illuminant_label"] = ill
            # Prefer EXIF from zip member (same bytes as extracted no-flash).
            try:
                from scripts.dng_exif import read_noflash_exposure_from_zip

                exif = read_noflash_exposure_from_zip(zip_path)
                out["exposure"] = {
                    "iso": exif.get("iso"),
                    "shutter_s": exif.get("shutter_s"),
                    "shutter_raw": exif.get("shutter_raw"),
                }
                out["exposure_flags"] = exposure_flags(
                    iso=exif.get("iso"),
                    shutter_s=exif.get("shutter_s"),
                    pipeline_L=out.get("L"),
                )
            except Exception as exc:
                out["exposure_error"] = str(exc)
                out["exposure_flags"] = exposure_flags(pipeline_L=out.get("L"))
            return out
        finally:
            if cleanup and work_dir is None:
                shutil.rmtree(tmp, ignore_errors=True)

    def run_files(
        self,
        noflash_dng: Union[str, Path],
        flash_dng: Union[str, Path],
        landmarks_json: Union[str, Path],
        *,
        illuminant_label: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Run on explicit no-flash DNG, flash DNG, and Apple landmark JSON."""
        nf = Path(noflash_dng).expanduser().resolve()
        fl = Path(flash_dng).expanduser().resolve()
        lm_path = Path(landmarks_json).expanduser().resolve()

        A0 = load_dng_linear(nf, half_size=self.half_size, use_camera_wb=False)
        B0 = load_dng_linear(fl, half_size=self.half_size, use_camera_wb=False)
        if B0.shape != A0.shape:
            B0 = cv2.resize(B0, (A0.shape[1], A0.shape[0]), interpolation=cv2.INTER_AREA)

        lm = load_apple_landmarks(lm_path)
        roi_key = str(self.roi or "forehead").strip().lower()
        skin_mask = apple_face_skin_roi_mask(
            lm, A0.shape[0], A0.shape[1], roi=roi_key, linear_rgb=A0 if roi_key == "forehead" else None
        )
        n_roi = int(np.count_nonzero(skin_mask))
        if n_roi < 50:
            raise RuntimeError(f"empty {roi_key} mask ({n_roi} px)")

        fairface_meta: Dict[str, Any] = {}
        sampling_mode = "off"
        ethnicity_for_sampling: Optional[str] = None
        use_specular_tone = self.sampling in ("fairface", "fairface7", "fairface4")
        if self.sampling in ("fairface", "fairface7", "fairface4"):
            if self.fairface is None:
                raise RuntimeError("FairFace sampling requested but predictor not loaded")
            preview = linear_rgb_to_preview_bgr(A0)
            face_rgb = face_rgb_crop_from_landmarks(preview, lm, padding=0.35)
            ff = self.fairface.predict_rgb(face_rgb)
            ethnicity_for_sampling = ff["predicted_ethnicity"]
            if use_specular_tone:
                sampling_mode = "specular_tone"
            fairface_meta = {
                "predicted_ethnicity": ff["predicted_ethnicity"],
                "fairface_label": ff["fairface_label"],
                "fairface_confidence": float(ff["confidence"]),
                "fairface_mode": ff["mode"],
                "fairface_probs": {k: float(v) for k, v in (ff.get("race_probs") or {}).items()},
            }

        B0m, flash_scale0 = match_flash_exposure(A0, B0, skin_mask)
        R0 = np.sqrt(np.maximum(A0, 0) * np.maximum(B0m, 0) + 1e-8)

        cat_meta: Dict[str, Any] = {"cat_mode": self.cat_mode}
        if self.cat_mode == "frozen_5500":
            xyz_white, cat_meta = scene_white_for_cat(
                "frozen_5500",
                illuminant_label=illuminant_label or "D65",
                lu_cct=float(self.fixed_cat_k),
            )
        elif self.cat_mode in ("lu_torch", "hybrid_deploy"):
            if self.torch_prior is None:
                raise RuntimeError(f"cat_mode={self.cat_mode} requires torch_prior")
            align = align_flash_linear(A0, B0, cheek_mask=skin_mask, use_ecc=True)
            lu_res = estimate_lu(align, self.torch_prior, use_measured_spd_rgb=True)
            lu_cct = float(lu_res.ambient_cct_estimated_k)
            cat_meta["lu_cct_k"] = lu_cct
            cat_meta["ecc_cc"] = float(align.ecc_cc)
            if self.cat_mode == "hybrid_deploy":
                hybrid_cct = hybrid_deploy_cct(illuminant_label or "D65", lu_cct)
                xyz_white, cat_meta = scene_white_for_cat(
                    "hybrid_deploy",
                    illuminant_label=illuminant_label or "D65",
                    lu_cct=lu_cct,
                    hybrid_cct=hybrid_cct,
                )
            else:
                xyz_white, cat_meta = scene_white_for_cat(
                    "lu_torch",
                    illuminant_label=illuminant_label or "D65",
                    lu_cct=lu_cct,
                )
        else:
            from flash_noflash_spectral import planck_xyz_y1

            xyz_white = planck_xyz_y1(float(self.fixed_cat_k), 0.0)
            cat_meta["cat_cct"] = float(self.fixed_cat_k)

        lab_mask = skin_mask
        lab_pool_meta: Dict[str, Any] = {}
        if (
            roi_key == "forehead"
            and sampling_mode == "specular_tone"
        ):
            l_std = probe_forehead_lab_l_std(
                R0,
                skin_mask,
                self.M,
                projector=self.color_projector,
                xyz_scene_white=xyz_white,
                cat_degree=float(self.cat_degree),
                skin_lab_trim=FOREHEAD_SKIN_LAB_TRIM,
            )
            ill_u = str(illuminant_label or "D65").upper()
            eth_l = str(ethnicity_for_sampling or "").strip().lower()
            expand_pool = l_std < FOREHEAD_L_UNIFORM_STD and (
                eth_l == "indian" or ill_u in ("F12", "D12")
            )
            lab_pool_meta["forehead_L_std"] = l_std
            if expand_pool:
                lab_mask = apple_face_forehead_lab_pool_mask(
                    lm, A0.shape[0], A0.shape[1], linear_rgb=A0
                )
                lab_pool_meta["lab_pool_expanded"] = True
            else:
                lab_pool_meta["lab_pool_expanded"] = False
                sampling_mode = "off"

        Lab, sample_meta = mean_lab_on_mask(
            R0,
            lab_mask,
            self.M,
            projector=self.color_projector,
            xyz_scene_white=xyz_white,
            cat_degree=float(self.cat_degree),
            l_sampling=sampling_mode,
            ethnicity=ethnicity_for_sampling,
            skin_lab_trim=FOREHEAD_SKIN_LAB_TRIM if roi_key == "forehead" else None,
        )
        sample_meta.update(lab_pool_meta)

        lab_corr_key: Optional[str] = None
        if self.multi_lab_bundle is not None:
            Lab, lab_corr_key = apply_multi_illuminant_lab_affine(
                Lab,
                illuminant_label or "D65",
                self.multi_lab_bundle,
                mode="routed",
            )
        elif self.lab_affine_W is not None:
            Lab = apply_lab_affine(Lab, self.lab_affine_W)
            lab_corr_key = "lab_affine"

        out: Dict[str, Any] = {
            "L": float(Lab[0]),
            "a": float(Lab[1]),
            "b": float(Lab[2]),
            "roi": roi_key,
            "n_roi": n_roi,
            "n_cheek": n_roi if roi_key == "cheek" else None,
            "n_forehead": n_roi if roi_key == "forehead" else None,
            "flash_scale": float(flash_scale0),
            "shape": [int(x) for x in A0.shape],
            "scr_mode": "preawb_cat",
            "fixed_cat_k": float(self.fixed_cat_k),
            "cat_mode": self.cat_mode,
            "cat_cct": cat_meta.get("cat_cct"),
            "lu_cct_k": cat_meta.get("lu_cct_k"),
            "ecc_cc": cat_meta.get("ecc_cc"),
            "illuminant_label": illuminant_label,
            "lab_corrector": lab_corr_key,
            "color_projector": bool(self.color_projector),
            "cat_degree": float(self.cat_degree),
            "half_size": bool(self.half_size),
            "l_sampling": sampling_mode,
            "roi_sampling_mode": sampling_mode,
            "skin_binning": sample_meta.get("skin_binning"),
            "skin_binning_kept_frac": sample_meta.get("skin_binning_kept_frac"),
            "forehead_L_std": sample_meta.get("forehead_L_std"),
            "lab_pool_expanded": sample_meta.get("lab_pool_expanded"),
            "cal_dir": str(self.cal_dir) if self.cal_dir else None,
            "l_percentile": sample_meta.get("l_percentile"),
            "indian_branch": sample_meta.get("indian_branch"),
            "asian_branch": sample_meta.get("asian_branch"),
            "iranian_branch": sample_meta.get("iranian_branch"),
            "white_branch": sample_meta.get("white_branch"),
            "noflash_dng": str(nf),
            "flash_dng": str(fl),
            "landmarks_json": str(lm_path),
        }
        out.update(fairface_meta)
        if sample_meta.get("predicted_ethnicity"):
            out["predicted_ethnicity"] = sample_meta["predicted_ethnicity"]

        try:
            from scripts.dng_exif import read_dng_exposure

            exif = read_dng_exposure(nf)
            out["exposure"] = {
                "iso": exif.get("iso"),
                "shutter_s": exif.get("shutter_s"),
                "shutter_raw": exif.get("shutter_raw"),
            }
            out["exposure_flags"] = exposure_flags(
                iso=exif.get("iso"),
                shutter_s=exif.get("shutter_s"),
                pipeline_L=out["L"],
            )
        except Exception as exc:
            out["exposure_error"] = str(exc)
            out["exposure_flags"] = exposure_flags(pipeline_L=out["L"])

        return out


def result_to_jsonable(result: Dict[str, Any]) -> Dict[str, Any]:
    """Ensure JSON-serializable copy (numpy scalars → Python)."""

    def _conv(v: Any) -> Any:
        if isinstance(v, dict):
            return {str(k): _conv(x) for k, x in v.items()}
        if isinstance(v, (list, tuple)):
            return [_conv(x) for x in v]
        if isinstance(v, (np.floating, np.integer)):
            return v.item()
        if isinstance(v, np.ndarray):
            return v.tolist()
        if isinstance(v, Path):
            return str(v)
        return v

    return _conv(result)


def write_result_json(result: Dict[str, Any], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result_to_jsonable(result), indent=2) + "\n", encoding="utf-8")
