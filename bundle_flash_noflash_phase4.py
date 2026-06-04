#!/usr/bin/env python3
"""Pin Phase-4 flash/no-flash results for GitHub (tables, CSVs, summaries, manifest)."""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parent
OUT = REPO / "figures" / "flash_noflash_phase4"
NOCAL = REPO / "flash_noflash_tier3_affine_nocal"
FITSKIN = REPO / "flash_noflash_tier3_affine_fitskin"

SLIM_COLUMNS = [
    "subject_id",
    "participant",
    "trial",
    "fitskin_cheek_L",
    "fitskin_cheek_a",
    "fitskin_cheek_b",
    "reflectance_L",
    "reflectance_a",
    "reflectance_b",
    "reflectance_fitskin_lightness_gain",
    "delta_L",
    "delta_a",
    "delta_b",
    "reflectance_cheek_de00",
]

TABLES = {
    "phase4_chartfree.tex": {
        "caption": (
            "Phase 4 chart-free evaluation: per-trial CIELAB coordinates, "
            "per-channel offsets (pipeline $-$ FitSkin), and $\\Delta E_{00}$ "
            "($N = 5$ trials, P2~T1 excluded). Flash/no-flash geometric reflectance "
            "with Tier-3 affine camera calibration (trained on separate checker "
            "sessions), cheek ROI, and skin-mask exposure scaling; booth ambient "
            "CCT/Duv fixed. No in-scene color checker and no FitSkin lightness "
            "calibration on evaluation trials."
        ),
        "label": "tab:phase4_chartfree",
        "csv": NOCAL,
    },
    "phase4_fitskin.tex": {
        "caption": (
            "Phase 4 scanner lightness calibration: per-trial CIELAB coordinates, "
            "per-channel offsets (pipeline $-$ FitSkin), and $\\Delta E_{00}$ "
            "($N = 5$ trials, P2~T1 excluded). Same affine stack as "
            "Table~\\ref{tab:phase4_chartfree} with per-participant FitSkin lightness "
            "gain $s_{\\mathrm{FS}}$ (P1 $= 1.053$, P2 $= 1.063$) estimated on these "
            "same trials. Not independent validation."
        ),
        "label": "tab:phase4_fitskin",
        "csv": FITSKIN,
    },
}


def _r1(x: float) -> str:
    return f"{x:.1f}"


def _r2(x: float) -> str:
    return f"{x:.2f}"


def _load_rows(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open(newline="") as f:
        return list(csv.DictReader(f))


def _slim_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for r in rows:
        Lp = float(r["reflectance_L"])
        La = float(r["reflectance_a"])
        Lb = float(r["reflectance_b"])
        Lf = float(r["fitskin_cheek_L"])
        af = float(r["fitskin_cheek_a"])
        bf = float(r["fitskin_cheek_b"])
        gain = r.get("reflectance_fitskin_lightness_gain", "") or ""
        out.append(
            {
                "subject_id": r["subject_id"],
                "participant": r["participant"],
                "trial": r["trial"],
                "fitskin_cheek_L": _r1(Lf),
                "fitskin_cheek_a": _r1(af),
                "fitskin_cheek_b": _r1(bf),
                "reflectance_L": _r1(Lp),
                "reflectance_a": _r1(La),
                "reflectance_b": _r1(Lb),
                "reflectance_fitskin_lightness_gain": gain,
                "delta_L": _r1(Lp - Lf),
                "delta_a": _r1(La - af),
                "delta_b": _r1(Lb - bf),
                "reflectance_cheek_de00": _r2(float(r["reflectance_cheek_de00"])),
            }
        )
    return out


def _stats(rows: list[dict[str, str]]) -> dict[str, float]:
    import numpy as np

    def f(k: str) -> np.ndarray:
        return np.array([float(r[k]) for r in rows], dtype=float)

    Lp, La, Lb = f("reflectance_L"), f("reflectance_a"), f("reflectance_b")
    Lf, af, bf = f("fitskin_cheek_L"), f("fitskin_cheek_a"), f("fitskin_cheek_b")
    de = f("reflectance_cheek_de00")
    dL, da, db = Lp - Lf, La - af, Lb - bf
    return {
        "mean_pipeline_L": float(Lp.mean()),
        "mean_pipeline_a": float(La.mean()),
        "mean_pipeline_b": float(Lb.mean()),
        "mean_fitskin_L": float(Lf.mean()),
        "mean_fitskin_a": float(af.mean()),
        "mean_fitskin_b": float(bf.mean()),
        "mean_delta_L": float(dL.mean()),
        "mean_delta_a": float(da.mean()),
        "mean_delta_b": float(db.mean()),
        "mean_de00": float(de.mean()),
        "median_pipeline_L": float(np.median(Lp)),
        "median_pipeline_a": float(np.median(La)),
        "median_pipeline_b": float(np.median(Lb)),
        "median_fitskin_L": float(np.median(Lf)),
        "median_fitskin_a": float(np.median(af)),
        "median_fitskin_b": float(np.median(bf)),
        "median_delta_L": float(np.median(dL)),
        "median_delta_a": float(np.median(da)),
        "median_delta_b": float(np.median(db)),
        "median_de00": float(np.median(de)),
    }


def _trial_label(subject_id: str) -> str:
    # P1_T1 -> P1 T1
    p, t = subject_id.split("_", 1)
    return f"{p} {t[1:] if t.startswith('T') else t}"


def _latex_table(rows: list[dict[str, str]], caption: str, label: str) -> str:
    s = _stats(rows)
    lines = [
        r"\begin{table}[h]",
        r"\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        r"\begin{tabular}{lcccccccccc}",
        r"\toprule",
        r"& \multicolumn{3}{c}{Pipeline} & \multicolumn{3}{c}{FitSkin} &",
        r"\multicolumn{3}{c}{Offset} & \\",
        r"\cmidrule(lr){2-4} \cmidrule(lr){5-7} \cmidrule(lr){8-10}",
        r"Trial & $L^*$ & $a^*$ & $b^*$ & $L^*$ & $a^*$ & $b^*$ &",
        r"$\Delta L^*$ & $\Delta a^*$ & $\Delta b^*$ & $\Delta E_{00}$ \\",
        r"\midrule",
    ]
    for r in rows:
        Lp = float(r["reflectance_L"])
        La = float(r["reflectance_a"])
        Lb = float(r["reflectance_b"])
        Lf = float(r["fitskin_cheek_L"])
        af = float(r["fitskin_cheek_a"])
        bf = float(r["fitskin_cheek_b"])
        lines.append(
            f"{_trial_label(r['subject_id'])} & "
            f"{_r1(Lp)} & {_r1(La)} & {_r1(Lb)} & "
            f"{_r1(Lf)} & {_r1(af)} & {_r1(bf)} & "
            f"${_r1(Lp - Lf)}$ & ${_r1(La - af)}$ & "
            f"${_r1(Lb - bf)}$ & {_r2(float(r['reflectance_cheek_de00']))} \\\\"
        )
    lines += [
        r"\midrule",
        "Mean   & "
        f"{_r1(s['mean_pipeline_L'])} & {_r1(s['mean_pipeline_a'])} & {_r1(s['mean_pipeline_b'])} & "
        f"{_r1(s['mean_fitskin_L'])} & {_r1(s['mean_fitskin_a'])} & {_r1(s['mean_fitskin_b'])} & "
        f"${_r1(s['mean_delta_L'])}$ & ${_r1(s['mean_delta_a'])}$ & "
        f"${_r1(s['mean_delta_b'])}$ & {_r2(s['mean_de00'])} \\\\",
        "Median & "
        f"{_r1(s['median_pipeline_L'])} & {_r1(s['median_pipeline_a'])} & {_r1(s['median_pipeline_b'])} & "
        f"{_r1(s['median_fitskin_L'])} & {_r1(s['median_fitskin_a'])} & {_r1(s['median_fitskin_b'])} & "
        f"${_r1(s['median_delta_L'])}$ & ${_r1(s['median_delta_a'])}$ & "
        f"${_r1(s['median_delta_b'])}$ & {_r2(s['median_de00'])} \\\\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    return "\n".join(lines) + "\n"


def _write_manifest(out_dir: Path) -> None:
    lines: list[str] = []
    for p in sorted(out_dir.rglob("*")):
        if not p.is_file() or p.name == "MANIFEST-sha256.txt":
            continue
        rel = p.relative_to(out_dir).as_posix()
        h = hashlib.sha256(p.read_bytes()).hexdigest()
        lines.append(f"{h}  {rel}")
    (out_dir / "MANIFEST-sha256.txt").write_text("\n".join(lines) + "\n")


def main() -> None:
    if not NOCAL.is_dir() or not FITSKIN.is_dir():
        raise SystemExit(
            "Missing source runs. Re-run:\n"
            "  OUT_DIR=./flash_noflash_tier3_affine_nocal ./run_flash_noflash_skin_lab_raw.sh\n"
            "  OUT_DIR=./flash_noflash_tier3_affine_fitskin ./run_flash_noflash_skin_lab_raw.sh "
            "--fitskin-lightness-calibration"
        )

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "tables").mkdir(exist_ok=True)

    for name, src in [
        ("flash_noflash_dual_reporting_stack.json", REPO / "flash_noflash_dual_reporting_stack.json"),
        ("flash_noflash_ablation_decomposition.json", REPO / "flash_noflash_ablation_decomposition.json"),
        ("flash_noflash_tier3_ablation.json", REPO / "flash_noflash_tier3_ablation.json"),
    ]:
        if src.is_file():
            shutil.copy2(src, OUT / name)

    for stem, src_dir in [
        ("chartfree", NOCAL),
        ("fitskin_reporting", FITSKIN),
    ]:
        raw = _load_rows(src_dir / "flash_noflash_skin_lab.csv")
        slim = _slim_rows(raw)
        csv_out = OUT / f"flash_noflash_{stem}_per_trial.csv"
        with csv_out.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=SLIM_COLUMNS)
            w.writeheader()
            w.writerows(slim)

        summary_src = src_dir / "summary.json"
        if summary_src.is_file():
            summary = json.loads(summary_src.read_text())
            for k in ("fitskin_scan_csv", "fitskin_mapping_csv"):
                summary.pop(k, None)
            (OUT / f"flash_noflash_{stem}_summary.json").write_text(
                json.dumps(summary, indent=2) + "\n"
            )

    for tex_name, meta in TABLES.items():
        raw_rows = _load_rows(meta["csv"] / "flash_noflash_skin_lab.csv")
        (OUT / "tables" / tex_name).write_text(
            _latex_table(raw_rows, meta["caption"], meta["label"])
        )

    _write_manifest(OUT)
    print(f"Wrote {OUT} ({len(list(OUT.rglob('*')))} paths)")


if __name__ == "__main__":
    main()
