"""
Load skin reflectance priors r(lambda) for SCR-AWB-style portrait AWB.

Priors live under ``calibration/skin_reflectance_priors/`` as JSON:
  { "name", "source", "wavelengths_nm": [...], "reflectance": [...] }

Resample to monochromator grid (400--700 nm, 10 nm) via linear interpolation.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np

ROOT = Path(__file__).resolve().parent
DEFAULT_PRIORS_DIR = ROOT / "calibration" / "skin_reflectance_priors"
MONO_WL = np.arange(400, 701, 10, dtype=np.float64)


@dataclass
class SkinReflectancePrior:
    name: str
    source: str
    wavelengths_nm: np.ndarray  # (N,)
    reflectance: np.ndarray  # (N,) normalized 0-1

    def resample_to(self, target_wl: np.ndarray) -> np.ndarray:
        wl = np.asarray(self.wavelengths_nm, dtype=np.float64)
        r = np.asarray(self.reflectance, dtype=np.float64)
        r = r / max(float(np.max(r)), 1e-12)
        return np.interp(target_wl, wl, r, left=r[0], right=r[-1])

    def on_monochromator_grid(self) -> np.ndarray:
        return self.resample_to(MONO_WL)

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "name": self.name,
            "source": self.source,
            "wavelengths_nm": self.wavelengths_nm.astype(int).tolist(),
            "reflectance": self.reflectance.tolist(),
        }
        with path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    @classmethod
    def load(cls, path: Path) -> "SkinReflectancePrior":
        with Path(path).open(encoding="utf-8") as f:
            d = json.load(f)
        return cls(
            name=str(d["name"]),
            source=str(d.get("source", "")),
            wavelengths_nm=np.asarray(d["wavelengths_nm"], dtype=np.float64),
            reflectance=np.asarray(d["reflectance"], dtype=np.float64),
        )


def list_priors(priors_dir: Optional[Path] = None) -> List[Path]:
    d = Path(priors_dir or DEFAULT_PRIORS_DIR)
    if not d.is_dir():
        return []
    return sorted(d.glob("*.json"))


def load_prior(name_or_path: Union[str, Path], priors_dir: Optional[Path] = None) -> SkinReflectancePrior:
    p = Path(name_or_path)
    if p.suffix == ".json" and p.is_file():
        return SkinReflectancePrior.load(p)
    d = Path(priors_dir or DEFAULT_PRIORS_DIR)
    candidate = d / f"{name_or_path}.json"
    if candidate.is_file():
        return SkinReflectancePrior.load(candidate)
    raise FileNotFoundError(f"No prior {name_or_path!r} under {d}")

