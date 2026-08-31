"""Read ISO / shutter / WB from Apple ProRAW DNG IFDs via tifffile."""

from __future__ import annotations

import io
import zipfile
from pathlib import Path
from typing import Any, Dict, Optional, Union


def _as_float(v: Any) -> float:
    if isinstance(v, tuple) and len(v) == 2 and v[1]:
        return float(v[0]) / float(v[1])
    if isinstance(v, (list, tuple)) and len(v) == 1:
        return _as_float(v[0])
    return float(v)


def read_dng_exposure(source: Union[Path, str, bytes]) -> Dict[str, Any]:
    """Return no-flash-style exposure fields from a DNG.

    Keys: iso, shutter_s, shutter_raw, white_balance, exposure_mode,
    model, white_level, baseline_exposure (optional).
    """
    import tifffile

    if isinstance(source, (str, Path)):
        handle: Any = str(source)
        closer = None
    else:
        handle = io.BytesIO(source)
        closer = handle

    try:
        with tifffile.TiffFile(handle) as tif:
            page = tif.pages[0]
            tags = {tag.name: tag.value for tag in page.tags.values()}
            exif = tags.get("ExifTag") or {}
            if not isinstance(exif, dict):
                exif = {}

            iso_raw = exif.get("ISOSpeedRatings")
            if iso_raw is None:
                iso_raw = tags.get("ISOSpeedRatings")
            if isinstance(iso_raw, (list, tuple)):
                iso = float(iso_raw[0])
            elif iso_raw is not None:
                iso = float(iso_raw)
            else:
                iso = float("nan")

            et = exif.get("ExposureTime")
            if et is None:
                shutter_s = float("nan")
                shutter_raw = ""
            else:
                shutter_s = _as_float(et)
                if isinstance(et, tuple) and len(et) == 2 and et[1]:
                    shutter_raw = f"1/{int(round(et[1] / et[0]))}s" if et[0] == 1 else f"{et[0]}/{et[1]}s"
                else:
                    shutter_raw = f"{shutter_s:.6g}s"

            return {
                "iso": iso,
                "shutter_s": shutter_s,
                "shutter_raw": shutter_raw,
                "white_balance": int(exif.get("WhiteBalance", -1)),
                "exposure_mode": int(exif.get("ExposureMode", -1)),
                "model": str(tags.get("Model", "")),
                "white_level": float(tags["WhiteLevel"]) if "WhiteLevel" in tags else float("nan"),
                "baseline_exposure": (
                    _as_float(tags["BaselineExposure"]) if "BaselineExposure" in tags else float("nan")
                ),
                "flash": int(exif.get("Flash", -1)),
            }
    finally:
        if closer is not None:
            closer.close()


def read_noflash_exposure_from_zip(zip_path: Path) -> Dict[str, Any]:
    """Extract EXIF from ``1_raw_Photo.dng`` (or first non-flash DNG) inside a zip."""
    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
        nf: Optional[str] = None
        for n in names:
            if Path(n).name == "1_raw_Photo.dng":
                nf = n
                break
        if nf is None:
            for n in names:
                low = Path(n).name.lower()
                if low.endswith(".dng") and "flash" not in low:
                    nf = n
                    break
        if nf is None:
            raise FileNotFoundError(f"No no-flash DNG in {zip_path}")
        return read_dng_exposure(zf.read(nf))
