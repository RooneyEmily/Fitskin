#!/usr/bin/env python3
"""Train separate offline camera matrices for light vs dark skin tone tiers.

Stacks all available ColorChecker sessions per person:
  - Pansor iPhone ProRAW DNGs (camera WB)
  - Phase-4 booth RAW DNGs (unity WB)
  - Bundled chart_cc JPEG pairs

Example::

    python3 scripts/train_skin_tone_bundles.py \\
        --booth-raw-root "/path/to/RAW Dataset" \\
        --monochromator-bundle calibration/tier3_affine \\
        --out-root calibration/tier3_by_tone
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "train_flash_noflash_checker_calibration.py"
BUILD = ROOT / "scripts" / "build_tone_training_manifest.py"
MONO_DEFAULT = ROOT / "calibration" / "tier3_affine"
TRAINING_DIR = ROOT / "data" / "training"


def _run_build(
    *,
    pansor_manifest: Path,
    booth_raw_root: Path,
    jpeg_root: Path,
    out_dir: Path,
) -> tuple[Path, Path]:
    cmd = [
        sys.executable,
        str(BUILD),
        "--pansor-manifest",
        str(pansor_manifest),
        "--booth-raw-root",
        str(booth_raw_root),
        "--jpeg-root",
        str(jpeg_root),
        "--out-dir",
        str(out_dir),
    ]
    print(">>>", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(ROOT), check=True)
    return out_dir / "manifest_tone_light.csv", out_dir / "manifest_tone_dark.csv"


def _run_train(*, out_dir: Path, manifest: Path, issa_rows: str, monochromator: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(TRAIN),
        "--out-dir",
        str(out_dir),
        "--manifest",
        str(manifest),
        "--monochromator-bundle",
        str(monochromator),
        "--matrix-affine",
        "--issa-skin-rows",
        issa_rows,
        "--raw-half-size",
        "1",
    ]
    print("\n>>>", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(ROOT), check=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--pansor-manifest",
        type=Path,
        default=ROOT / "data" / "pansor" / "manifest_pansor_fitskin.csv",
    )
    ap.add_argument(
        "--booth-raw-root",
        type=Path,
        default=Path(
            "/home/mabl-main/Documents/RAW Dataset-20260531T233644Z-3-001/RAW Dataset"
        ),
    )
    ap.add_argument(
        "--jpeg-root",
        type=Path,
        default=ROOT / "data" / "chart_cc_jpeg",
    )
    ap.add_argument("--monochromator-bundle", type=Path, default=MONO_DEFAULT)
    ap.add_argument("--out-root", type=Path, default=ROOT / "calibration" / "tier3_by_tone")
    ap.add_argument(
        "--training-manifest-dir",
        type=Path,
        default=TRAINING_DIR,
        help="Where build_tone_training_manifest.py writes CSVs.",
    )
    args = ap.parse_args()

    if not TRAIN.is_file() or not BUILD.is_file():
        raise SystemExit(f"Missing trainer or manifest builder under {ROOT}")

    args.out_root.mkdir(parents=True, exist_ok=True)
    light_manifest, dark_manifest = _run_build(
        pansor_manifest=args.pansor_manifest,
        booth_raw_root=args.booth_raw_root,
        jpeg_root=args.jpeg_root,
        out_dir=args.training_manifest_dir,
    )

    _run_train(
        out_dir=args.out_root / "dark",
        manifest=dark_manifest,
        issa_rows="issa_median_south_asian",
        monochromator=args.monochromator_bundle,
    )
    _run_train(
        out_dir=args.out_root / "light",
        manifest=light_manifest,
        issa_rows="issa_median_caucasian",
        monochromator=args.monochromator_bundle,
    )

    readme = args.out_root / "README.md"
    readme.write_text(
        f"""# Skin-tone offline calibration bundles

Chart-free inference (`flash_no_flash_skin_lab.py`). When a ColorChecker is in frame, use chart CC with `--skin-tone auto`.

| Subdir | Training manifest | ISSA prior |
|--------|-------------------|------------|
| `dark/` | `{dark_manifest.relative_to(ROOT)}` (Pansor ProRAW CC; booth/JPEG excluded — they bias L* high on ProRAW) | issa_median_south_asian (Liki, Indian) |
| `light/` | `{light_manifest.relative_to(ROOT)}` (Pansor + booth RAW + chart_cc JPEG) | issa_median_caucasian |

```bash
python3 scripts/train_skin_tone_bundles.py

python3 flash_no_flash_skin_lab.py \\
  --iphone-calibration-tone-root calibration/tier3_by_tone \\
  --skin-tone auto \\
  ...
```
""",
        encoding="utf-8",
    )
    print(f"\nWrote bundles under {args.out_root}/{{dark,light}}")
    print(f"Wrote {readme}")


if __name__ == "__main__":
    main()
