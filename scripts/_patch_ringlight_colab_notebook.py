#!/usr/bin/env python3
"""Refresh ringlight_best_stack_inference_colab.ipynb (Lihn WB sweep, Colab fixes)."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NB = ROOT / "ringlight_best_stack_inference_colab.ipynb"
LIHN_MANIFEST = ROOT / "data" / "ring_light" / "wb_sweep_lihn.json"
PARKER_MANIFEST = ROOT / "data" / "ring_light" / "wb_sweep_parker.json"
EVAL_N84 = ROOT / "data" / "ring_light" / "eval_n84_by_wb_cell.json"


def lines(text: str) -> list[str]:
    if not text.endswith("\n"):
        text += "\n"
    return text.splitlines(keepends=True)


def main() -> None:
    nb = json.loads(NB.read_text(encoding="utf-8"))
    lihn_json = json.dumps(json.loads(LIHN_MANIFEST.read_text()), indent=2)
    parker_json = json.dumps(json.loads(PARKER_MANIFEST.read_text()), indent=2)
    eval_n84_json = json.dumps(json.loads(EVAL_N84.read_text()), indent=2)

    # ── Cell 0 ───────────────────────────────────────────────────────────────
    nb["cells"][0]["source"] = lines(
        r"""# Ring-Light Best Stack — Forehead Lab Inference (Colab)

Production-style **chart-free** forehead Lab from Variable Lighting **torch zips**. ROI matches **FitSkin forehead**.

## Stack (best deployment)

| Step | What |
|---|---|
| 1 | Pre-AWB demosaic → reflectance \(R_0=\sqrt{A_0\odot B_0'}\) |
| 2 | Apple Vision **forehead** mask |
| 3 | **`tier3_affine`** indoor RGB→XYZ |
| 4 | **`hybrid_deploy` CAT** — Lu+torch SPD on F12/warm; frozen 5500 K on D65 |
| 5 | **Illuminant-routed multi-Lab corrector** |
| 6 | FairFace-7 → **specular_tone** (+ cheek pool when forehead L* is uniform) → Lab |

## Run order

1. **Runtime → GPU** optional
2. **Cell 1** — clone repo, extract assets, mount Drive
3. **Cell 2b** — Parker WB sweep from **`data/ring_light/demo_zips/`** in git (no Drive) — or Drive fallback
4. **Cell 3** — run pipeline + compare FitSkin ΔE by `wb_cell` (illumination grid + segmentation viz)
5. **Cell 4** — optional frozen vs best-stack on one zip
6. **Cell 6** — pinned n=84 cohort tables (optional; no re-run)

> Drive needs `Variable Lighting Ring Light/…/Parker P2/{D65,F12}/*.zip`.
"""
    )

    # ── Cell 1 ───────────────────────────────────────────────────────────────
    cell1 = (ROOT / "scripts" / "_colab_cell1_setup.py").read_text()
    cell1 = cell1.replace(
        "# Mount Drive (needed for Woojae zips in Cell 2b)",
        "# Mount Drive (needed for Parker WB-sweep zips in Cell 2b)",
    ).replace(
        "Woojae zips",
        "Parker WB-sweep zips",
    )
    nb["cells"][2]["source"] = lines(cell1)

    # ── Cell 2a ──────────────────────────────────────────────────────────────
    nb["cells"][4]["source"] = lines(
        r"""# ══════════════════════════════════════════════════════════════════════════════
# CELL 2a — OPTIONAL upload (skip — Cell 2b loads Parker from Drive)
# ══════════════════════════════════════════════════════════════════════════════
DO_UPLOAD = False
if DO_UPLOAD and IN_COLAB:
    from google.colab import files
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    uploaded = files.upload()
    for name in uploaded:
        (UPLOAD_DIR / name).write_bytes(uploaded[name])
        print("Saved", UPLOAD_DIR / name)
else:
    print("Upload skipped. Cell 2b finds Parker zips on Drive automatically.")
"""
    )

    # ── Cell 2b ──────────────────────────────────────────────────────────────
    cell2b = (ROOT / "scripts" / "_colab_cell2b_parker.py").read_text()
    nb["cells"][5]["source"] = lines(cell2b)

    # ── Cell 3 markdown ──────────────────────────────────────────────────────
    nb["cells"][6]["source"] = lines(
        r"""## 3 — Parker WB sweep inference

Runs every zip from Cell 2b, prints FitSkin ΔE **by capture wb_cell**, then:

- **`SHOW_ILLUM_GRID=True`** — face crops: D65 row vs F12 row
- **`SHOW_VIZ_INDEX=0`** — segmentation (forehead=green, cheek=blue, Lab pool=yellow)
- **`SHOW_ALL_SEGMENTATIONS=True`** — overlay montage for every capture
"""
    )

    # ── Cell 3 code ──────────────────────────────────────────────────────────
    nb["cells"][7]["source"] = lines((ROOT / "scripts" / "_colab_cell3_lihn.py").read_text())

    # ── Cell 4 markdown ──────────────────────────────────────────────────────
    nb["cells"][8]["source"] = lines(
        r"""## 4 — Optional: frozen vs best stack

Uses the zip selected by Cell 3 `SHOW_VIZ_INDEX` (or first loaded zip if viz skipped).
"""
    )

    # ── Cell 5 batch ─────────────────────────────────────────────────────────
    nb["cells"][10]["source"] = lines(
        r"""## 5 — Optional extras

Cell 3 already runs the full Parker WB sweep. Set `RUN_BATCH=True` in the next cell only if Cell 2b found extra zips beyond the sweep.
"""
    )

    nb["cells"][11]["source"] = lines(
        r"""# ══════════════════════════════════════════════════════════════════════════════
# CELL 5 — OPTIONAL batch extras
# ══════════════════════════════════════════════════════════════════════════════
RUN_BATCH = False

if RUN_BATCH and "candidates" in dir() and "demo_names_ordered" in dir():
    batch_zips = [p for p in candidates if p.name not in set(demo_names_ordered)]
    if not batch_zips:
        print("No extra zips beyond the Lihn WB sweep.")
    else:
        for i, zp in enumerate(batch_zips, 1):
            try:
                r = pipe.run_zip(zp)
                write_result_json(r, OUT_DIR / f"{zp.stem}.json")
                print(f"[{i}] {zp.name}  Lab=({r['L']:.1f},{r['a']:.1f},{r['b']:.1f})")
            except Exception as exc:
                print(f"[{i}] FAIL {zp.name}: {exc}")
else:
    print("Batch skipped (RUN_BATCH=False).")
"""
    )

    # ── Cell 6 reference ─────────────────────────────────────────────────────
    nb["cells"][12]["source"] = lines(
        r"""## 6 — Full cohort reference (n=84): ΔE by wb_cell

Pre-computed locally — **does not re-run 84 trials**. Compare with Cell 3 (Parker, specular_tone deploy).

Loaded from repo, asset zip (Cell 1), or inline fallback if GitHub tip lags.
"""
    )

    nb["cells"][13]["source"] = lines(
        f"""# ══════════════════════════════════════════════════════════════════════════════
# CELL 6 — Pinned n=84 ΔE₀₀ by wb_cell (optional)
# ══════════════════════════════════════════════════════════════════════════════
import json
import matplotlib.pyplot as plt

EVAL_JSON = REPO / "data" / "ring_light" / "eval_n84_by_wb_cell.json"
if EVAL_JSON.is_file():
    payload = json.loads(EVAL_JSON.read_text(encoding="utf-8"))
    print(f"Loaded {{EVAL_JSON.relative_to(REPO)}}")
else:
    payload = json.loads(r\"\"\"{eval_n84_json}\"\"\")
    EVAL_JSON.parent.mkdir(parents=True, exist_ok=True)
    EVAL_JSON.write_text(json.dumps(payload, indent=2) + "\\n", encoding="utf-8")
    print("Using inline n=84 eval table (file not in git clone yet); wrote", EVAL_JSON)

ARMS = payload.get("arms") or ["frozen_5500", "hybrid_deploy", "hybrid_multi_lab"]
WB_K = payload.get("capture_wb_k") or {{}}
cells = ["A", "B", "C", "D", "E"]

print(payload.get("source_note", ""))
print(f"n={{payload.get('n_trials')}} trials\\n")

ov = payload.get("overall", {{}}).get("hybrid_multi_lab", {{}})
if ov.get("mean") is not None:
    print(f"Overall hybrid+multi-lab mean ΔE₀₀: {{ov['mean']:.2f}}  (median {{ov.get('median', float('nan')):.2f}})\\n")

for ill in ("D65", "F12"):
    grp = payload.get("by_illuminant_wb_cell", {{}}).get(ill, {{}})
    if not grp:
        continue
    print(f"=== {{ill}} — hybrid+multi-lab mean ΔE by wb_cell ===")
    for cell in cells:
        if cell not in grp:
            continue
        st = grp[cell].get("hybrid_multi_lab", {{}})
        if st.get("mean") is None:
            continue
        wbk = WB_K.get(ill, {{}}).get(cell, "?")
        print(f"  {{cell}} ({{wbk}} K): {{st['mean']:.2f}}  n={{st['n']}}")
    print()

best = "hybrid_multi_lab"
fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
for ax, ill in zip(axes, ("D65", "F12")):
    grp = payload.get("by_illuminant_wb_cell", {{}}).get(ill, {{}})
    xs, ys = [], []
    for cell in cells:
        st = grp.get(cell, {{}}).get(best)
        if st and st.get("mean") is not None:
            xs.append(cell)
            ys.append(st["mean"])
    if xs:
        ax.bar(xs, ys, color="#2563eb" if ill == "D65" else "#dc2626", alpha=0.85)
        for i, y in enumerate(ys):
            ax.text(i, y + 0.15, f"{{y:.1f}}", ha="center", fontsize=8)
    ax.set_title(f"{{ill}} cohort (n=84, trimmed-mean eval)")
    ax.set_xlabel("wb_cell")
    ax.set_ylabel("mean ΔE₀₀")
plt.suptitle("Pinned reference — trimmed mean; Cell 3 uses specular_tone on Parker", fontsize=10)
plt.tight_layout()
plt.show()
"""
    )

    # Clear stale outputs
    for i in [2, 4, 5, 7, 9, 11, 13]:
        nb["cells"][i]["outputs"] = []
        nb["cells"][i]["execution_count"] = None

    NB.write_text(json.dumps(nb, indent=1) + "\n", encoding="utf-8")

    # Syntax-check (IPython magics → indented pass)
    for idx in [2, 5, 7, 11, 13]:
        src = "".join(nb["cells"][idx]["source"])
        check_lines = []
        for ln in src.splitlines():
            if ln.lstrip().startswith("%"):
                check_lines.append(" " * (len(ln) - len(ln.lstrip())) + "pass  # magic")
            else:
                check_lines.append(ln)
        compile("\n".join(check_lines) + "\n", f"cell{idx}", "exec")

    print("Updated", NB)


if __name__ == "__main__":
    main()
