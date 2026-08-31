#!/usr/bin/env python3
"""Copy eight demo *Torch.zip files (4 participants × D65/F12) into data/ring_light/demo_zips/."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "ring_light" / "demo_zips"
MANIFEST = ROOT / "data" / "ring_light" / "demo_manifest.json"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="Variable Lighting Ring Light root (auto-detected from ~/Downloads if omitted)",
    )
    args = ap.parse_args()

    if args.data_root is None:
        from scripts.evaluate_ringlight_torch_illuminant import default_data_root

        data_root = default_data_root()
    else:
        data_root = Path(args.data_root).expanduser().resolve()

    picks = {
        "AnjanaF12B1Torch.zip": list(data_root.rglob("AnjanaF12B1Torch.zip")),
        "Anjana-D65-C3Torch.zip": list(data_root.rglob("Anjana-D65-C3Torch.zip")),
        "Lihn-F12-B1Torch.zip": list(data_root.rglob("Lihn-F12-B1Torch.zip")),
        "LihnD65-C1Torch.zip": list(data_root.rglob("LihnD65-C1Torch.zip")),
        "Parker-F12-E1Torch.zip": list(data_root.rglob("Parker-F12-E1Torch.zip")),
        "Parker-D65-E1Torch.zip": list(data_root.rglob("Parker-D65-E1Torch.zip")),
        "Woojae-F12-E3Torch.zip": list(data_root.rglob("Woojae-F12-E3Torch.zip")),
        "Woojae-D65-A1Torch.zip": list(data_root.rglob("Woojae-D65-A1Torch.zip")),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    for name, hits in picks.items():
        if not hits:
            raise SystemExit(f"Missing {name} under {data_root}")
        src = hits[0]
        dst = OUT / name
        print(f"{src} -> {dst}")
        shutil.copy2(src, dst)

    if MANIFEST.is_file():
        print("Manifest:", MANIFEST)
    print(f"Done — {len(picks)} zips in {OUT}")


if __name__ == "__main__":
    main()
