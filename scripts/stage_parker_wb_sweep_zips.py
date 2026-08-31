#!/usr/bin/env python3
"""Copy Parker WB-sweep *Torch.zip files into data/ring_light/demo_zips/."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "ring_light" / "demo_zips"
MANIFEST = ROOT / "data" / "ring_light" / "wb_sweep_parker.json"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--d65-dir",
        type=Path,
        default=Path.home() / "Downloads" / "drive-download-20260831T103650Z-1-001",
        help="Flat folder with Parker D65 torch zips",
    )
    ap.add_argument(
        "--f12-dir",
        type=Path,
        default=Path.home() / "Downloads" / "drive-download-20260831T103757Z-1-001",
        help="Flat folder with Parker F12 torch zips",
    )
    args = ap.parse_args()

    man = json.loads(MANIFEST.read_text(encoding="utf-8"))
    OUT.mkdir(parents=True, exist_ok=True)
    search_dirs = [Path(args.d65_dir), Path(args.f12_dir)]

    for d in man.get("demos", []):
        fname = d["file"]
        dst = OUT / fname
        if dst.is_file():
            print("skip (exists):", dst.name)
            continue
        src = None
        for folder in search_dirs:
            if not folder.is_dir():
                continue
            for cand in folder.glob("*.zip"):
                if cand.name.replace(" ", "") == fname.replace(" ", ""):
                    src = cand
                    break
            if src:
                break
        if src is None:
            raise SystemExit(f"Missing {fname} — not found under {search_dirs}")
        print(f"{src} -> {dst}")
        shutil.copy2(src, dst)

    n = len(list(OUT.glob("Parker*Torch.zip"))) + len(list(OUT.glob("Parker*torch.zip")))
    print(f"Done — {n} Parker zips in {OUT}")


if __name__ == "__main__":
    main()
