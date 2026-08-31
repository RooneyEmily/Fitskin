#!/usr/bin/env python3
"""CLI for D65-FairFace7-ROI chart-free cheek Lab inference.

Examples
--------
Best stack (ring variable lighting)::

  python scripts/run_d65_fairface7_roi.py \\
    --zip "$HOME/Downloads/.../AnjanaF12B1Torch.zip" \\
    --cat-mode hybrid_deploy \\
    --multi-lab-corrector calibration/multi_illuminant_lab_affine/multi_illuminant_lab_affine.json

Indoor / frozen CAT baseline::

  python scripts/run_d65_fairface7_roi.py \\
    --zip path/to/capture.zip \\
    --out results/one_trial.json

Variable lighting (hybrid CAT + auto illuminant from path)::

  python scripts/run_d65_fairface7_roi.py \\
    --zip "$HOME/Downloads/Variable Lighting Ring Light-.../Anjana P3/F12/AnjanaF12B1Torch.zip" \\
    --cat-mode hybrid_deploy \\
    --multi-lab-corrector calibration/multi_illuminant_lab_affine/multi_illuminant_lab_affine.json

Use the real zip path on your machine (not ``path/to/...``). Illuminant ``F12`` is
auto-detected from the folder name or filename when ``--illuminant`` is omitted.

Batch a directory of zips::

  python scripts/run_d65_fairface7_roi.py \\
    --zip-dir "/path/to/Participant 1" \\
    --out-dir results/pipeline_runs

Frozen trimmed-mean only (no FairFace ROI)::

  python scripts/run_d65_fairface7_roi.py --zip capture.zip --sampling off
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline.d65_fairface7_roi import (  # noqa: E402
    D65FairFace7ROIPipeline,
    result_to_jsonable,
    write_result_json,
)
from pipeline.illuminant_estimation import infer_illuminant_label  # noqa: E402


def _collect_zips(zip_dir: Path) -> List[Path]:
    return sorted(p for p in Path(zip_dir).expanduser().resolve().rglob("*.zip") if p.is_file())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--zip", type=Path, help="Single Pansor-style flash/no-flash zip.")
    src.add_argument("--zip-dir", type=Path, help="Directory of zips (recursive).")
    src.add_argument(
        "--files",
        nargs=3,
        metavar=("NOFLASH_DNG", "FLASH_DNG", "LANDMARKS_JSON"),
        help="Explicit no-flash DNG, flash DNG, and Apple landmarks JSON.",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="JSON output path for a single run (--zip or --files).",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "results" / "pipeline_runs",
        help="Directory for batch JSON outputs (--zip-dir).",
    )
    ap.add_argument(
        "--cal-dir",
        type=Path,
        default=ROOT / "calibration" / "tier3_affine",
    )
    ap.add_argument(
        "--fairface-dir",
        type=Path,
        default=ROOT / "calibration" / "fairface",
    )
    ap.add_argument("--fixed-cat-k", type=float, default=5500.0)
    ap.add_argument(
        "--cat-mode",
        choices=("frozen_5500", "lu_torch", "hybrid_deploy"),
        default="frozen_5500",
        help="Bradford W_src: frozen 5500 K, Lu+torch SPD, or hybrid deploy rule.",
    )
    ap.add_argument(
        "--torch-dir",
        type=Path,
        default=None,
        help="MK350 Torch_meas folder (required for lu_torch / hybrid_deploy).",
    )
    ap.add_argument(
        "--illuminant",
        choices=("D65", "F12"),
        default=None,
        help="Ring illuminant label for hybrid_deploy. Auto-detected from zip path if omitted.",
    )
    ap.add_argument(
        "--lab-corrector",
        type=Path,
        default=None,
        help="Optional global Lab affine 4x3 (.npy or .json).",
    )
    ap.add_argument(
        "--multi-lab-corrector",
        type=Path,
        default=None,
        help="Illuminant-routed Lab corrector bundle from train_multi_illuminant_lab_affine.py.",
    )
    ap.add_argument(
        "--color-projector",
        type=Path,
        default=None,
        help="Optional residual RGB→XYZ projector (.npz).",
    )
    ap.add_argument(
        "--sampling",
        choices=("fairface7", "off"),
        default="fairface7",
        help="fairface7 = FairFace-7 ROI (default); off = trimmed-mean only.",
    )
    ap.add_argument("--full-res", action="store_true", help="Disable half-size demosaic.")
    ap.add_argument(
        "--work-dir",
        type=Path,
        default=None,
        help="Keep extracted zip contents under this directory.",
    )
    args = ap.parse_args()

    pipe = D65FairFace7ROIPipeline.from_defaults(
        cal_dir=args.cal_dir,
        fairface_dir=args.fairface_dir,
        fixed_cat_k=float(args.fixed_cat_k),
        cat_mode=str(args.cat_mode),
        torch_dir=args.torch_dir,
        lab_affine=args.lab_corrector,
        multi_lab_affine=args.multi_lab_corrector,
        color_projector=args.color_projector,
        half_size=not bool(args.full_res),
        sampling=str(args.sampling),
    )
    print(
        f"Loaded pipeline  cal={args.cal_dir}  sampling={args.sampling}  "
        f"cat_mode={args.cat_mode}  half_size={not args.full_res}"
    )

    if args.files is not None:
        nf, fl, lm = [Path(p) for p in args.files]
        ill = args.illuminant or infer_illuminant_label(nf) or infer_illuminant_label(lm)
        result = pipe.run_files(nf, fl, lm, illuminant_label=ill)
        _print_one(result)
        out = args.out or (args.out_dir / f"{nf.stem}_lab.json")
        write_result_json(result, out)
        print(f"Wrote {out}")
        return

    if args.zip is not None:
        zp = Path(args.zip).expanduser().resolve()
        ill = args.illuminant or infer_illuminant_label(zp)
        result = pipe.run_zip(zp, work_dir=args.work_dir, keep_extract=bool(args.work_dir))
        if ill and not result.get("illuminant_label"):
            result["illuminant_label"] = ill
        _print_one(result)
        out = args.out or (args.out_dir / f"{zp.stem}.json")
        write_result_json(result, out)
        print(f"Wrote {out}")
        return

    # batch
    zips = _collect_zips(args.zip_dir)
    if not zips:
        raise SystemExit(f"No zips under {args.zip_dir}")
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = []
    for i, zp in enumerate(zips, 1):
        try:
            result = pipe.run_zip(zp, work_dir=args.work_dir, keep_extract=bool(args.work_dir))
        except Exception as exc:
            print(f"[{i:02d}/{len(zips)}] FAIL {zp.name}: {exc}", flush=True)
            summary.append({"zip": str(zp), "error": str(exc)})
            continue
        out_path = out_dir / f"{zp.stem}.json"
        write_result_json(result, out_path)
        flag = ""
        ef = result.get("exposure_flags") or {}
        if ef.get("out_of_band"):
            flag = "  OUT_OF_BAND"
        print(
            f"[{i:02d}/{len(zips)}] {zp.name:40s}  "
            f"Lab=({result['L']:.1f},{result['a']:.1f},{result['b']:.1f})  "
            f"cat={result.get('cat_mode')}  ill={result.get('illuminant_label')}  "
            f"FF={result.get('fairface_label')}→{result.get('predicted_ethnicity')}"
            f"{flag}",
            flush=True,
        )
        summary.append(
            {
                "zip": str(zp),
                "out": str(out_path),
                "L": result["L"],
                "a": result["a"],
                "b": result["b"],
                "cat_mode": result.get("cat_mode"),
                "illuminant_label": result.get("illuminant_label"),
                "fairface_label": result.get("fairface_label"),
                "predicted_ethnicity": result.get("predicted_ethnicity"),
                "exposure_flags": result.get("exposure_flags"),
            }
        )
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(result_to_jsonable({"n": len(summary), "trials": summary}), indent=2) + "\n")
    print(f"Wrote {summary_path}  n={len(summary)}")


def _print_one(result: dict) -> None:
    ff = result.get("fairface_label")
    eth = result.get("predicted_ethnicity")
    print(
        f"Lab=({result['L']:.2f}, {result['a']:.2f}, {result['b']:.2f})  "
        f"n_cheek={result.get('n_cheek')}  "
        f"cat={result.get('cat_mode')}  ill={result.get('illuminant_label')}  "
        f"FF={ff}→{eth}  "
        f"sampling={result.get('l_sampling')}"
    )
    ef = result.get("exposure_flags") or {}
    if ef.get("out_of_band"):
        print(f"exposure_flags: {ef}")


if __name__ == "__main__":
    main()
