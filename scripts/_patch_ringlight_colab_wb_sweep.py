#!/usr/bin/env python3
"""Patch ringlight_best_stack_inference_colab.ipynb: fix skin_roi import + Woojae WB sweep."""

from __future__ import annotations

import json
from pathlib import Path

NB = Path(__file__).resolve().parents[1] / "ringlight_best_stack_inference_colab.ipynb"


def lines(s: str) -> list[str]:
    if not s.endswith("\n"):
        s += "\n"
    return s.splitlines(keepends=True)


def main() -> None:
    nb = json.loads(NB.read_text(encoding="utf-8"))

    nb["cells"][0]["source"] = lines(
        r"""# Ring-Light Best Stack — Forehead Lab Inference (Colab)

Production-style **chart-free** forehead Lab from Variable Lighting **torch zips**. ROI matches **FitSkin forehead**.

## Frozen color stack

| Step | What |
|---|---|
| 1 | Pre-AWB demosaic → reflectance \(R_0=\sqrt{A_0\odot B_0'}\) |
| 2 | Apple Vision **forehead** mask |
| 3 | **`tier3_affine`** indoor RGB→XYZ |
| 4 | **`hybrid_deploy` CAT** — Lu+torch SPD on F12/warm; frozen 5500 K on D65 |
| 5 | **Illuminant-routed multi-Lab corrector** (`W_d65` / `W_f12`) |
| 6 | FairFace-7 → **specular_tone** (pools cheek when forehead L* is uniform) → Lab |

### How to run
1. **Runtime → GPU** optional; mount Drive when prompted (Setup)
2. **Cell 1 (Setup)** → **Cell 2b** (auto-loads **Woojae** × all WB cells from Drive — no upload)
3. **Cell 3** — runs those zips and compares FitSkin ΔE by capture `wb_cell`
4. **Cell 6** — pinned n=84 tables (optional)
5. **Cell 2a** — only if you have a zip outside Drive

> Requires Drive folder `Variable Lighting Ring Light/…` (same tree as the local Downloads dump). Torch SPD: `Torch_meas/` if present.
"""
    )

    nb["cells"][2]["source"] = lines(
        r'''# ══════════════════════════════════════════════════════════════════════════════
# CELL 1 — SETUP
# ══════════════════════════════════════════════════════════════════════════════
!pip install -q rawpy opencv-python-headless numpy matplotlib gdown

try:
    import torch, torchvision  # noqa: F401
except ImportError:
    !pip install -q torch torchvision

import json, sys, zipfile, textwrap
from pathlib import Path

IN_COLAB = False
try:
    from google.colab import drive
    IN_COLAB = True
except ImportError:
    pass

REPO_URL = "https://github.com/RooneyEmily/Fitskin.git"
if Path("Fitskin").is_dir():
    !cd Fitskin && git pull --ff-only 2>/dev/null || true
    REPO = Path("Fitskin").resolve()
elif (Path.cwd() / "pipeline" / "d65_fairface7_roi.py").is_file():
    REPO = Path.cwd().resolve()
else:
    !git clone -q {REPO_URL}
    REPO = Path("Fitskin").resolve()

if IN_COLAB and not Path("/content/drive/MyDrive").is_dir():
    drive.mount("/content/drive")

sys.path = [str(REPO)] + [p for p in sys.path if Path(p).resolve() != REPO]

MULTI_LAB = REPO / "calibration" / "multi_illuminant_lab_affine" / "multi_illuminant_lab_affine.json"
PIPE = REPO / "pipeline" / "d65_fairface7_roi.py"
SKIN_ROI = REPO / "pipeline" / "skin_roi.py"
ASSET_ZIP = REPO / "colab_assets" / "ringlight_best_stack.zip"

# Refresh pipeline/calibration from the asset zip whenever it exists (git tip may lag).
if ASSET_ZIP.is_file():
    print("Extracting colab_assets/ringlight_best_stack.zip …")
    with zipfile.ZipFile(ASSET_ZIP) as zf:
        zf.extractall(REPO)
elif (not MULTI_LAB.is_file()) or (not PIPE.is_file()):
    print("WARN: missing assets — push colab_assets/ringlight_best_stack.zip to repo")

# Self-heal forehead∪cheek pool helper if still missing after extract/pull.
src = SKIN_ROI.read_text(encoding="utf-8") if SKIN_ROI.is_file() else ""
if "def apple_face_forehead_lab_pool_mask" not in src:
    print("Patching pipeline/skin_roi.py with apple_face_forehead_lab_pool_mask …")
    if not SKIN_ROI.is_file():
        raise FileNotFoundError(f"Missing {SKIN_ROI}")
    SKIN_ROI.write_text(
        src.rstrip()
        + "\n\n"
        + textwrap.dedent(
            """
            def apple_face_forehead_lab_pool_mask(
                landmarks: dict,
                dst_h: int,
                dst_w: int,
                *,
                linear_rgb=None,
            ):
                \"\"\"Forehead ∪ cheek mask for specular_tone Lab pooling on forehead ROI.\"\"\"
                forehead = apple_face_skin_roi_mask(
                    landmarks, dst_h, dst_w, roi="forehead", linear_rgb=linear_rgb
                )
                _, cheek = apple_face_cheek_masks(landmarks, dst_h, dst_w)
                return (forehead | (cheek > 0)).astype(bool)
            """
        ),
        encoding="utf-8",
    )

CAL_DIR = REPO / "calibration" / "tier3_affine"
FAIRFACE_DIR = REPO / "calibration" / "fairface"
FAIRFACE_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR = Path("/content/ringlight_best_stack_runs")
OUT_DIR.mkdir(parents=True, exist_ok=True)
UPLOAD_DIR = Path("/content/ringlight_uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

TORCH_DIR = Path("/content/drive/MyDrive/Torch_meas")  # optional

FF7 = FAIRFACE_DIR / "res34_fair_align_multi_7_20190809.pt"
if not FF7.is_file():
    print("Downloading FairFace-7 weights (~82 MB)…")
    !gdown 11y0Wi3YQf21a_VcspUV4FwqzhMcfaVAB -O "{FF7}"
assert FF7.is_file(), "FairFace weights missing"

# Drop stale modules so patched skin_roi / pipeline are loaded
for mod in list(sys.modules):
    if mod == "pipeline" or mod.startswith("pipeline."):
        del sys.modules[mod]

import torch
from pipeline.d65_fairface7_roi import D65FairFace7ROIPipeline, write_result_json
from pipeline.skin_roi import apple_face_forehead_lab_pool_mask  # verify

print("torch:", torch.__version__, "| cuda:", torch.cuda.is_available())
print("REPO:", REPO)
print("tier3 affine:", (CAL_DIR / "camera_rgb_to_xyz_affine.npy").is_file())
print("multi-lab corrector:", MULTI_LAB.is_file())
print("forehead pool mask: OK")
print("Setup OK.")
'''
    )

    nb["cells"][3]["source"] = lines(
        r"""## Cells 2a / 2b — Get torch zips

**Cell 2b (default):** loads **Woojae** across every available capture WB cell (D65 A–E, F12 B–E) from Drive — **no upload**.

- **Cell 2a** — optional; skip unless you need a zip that is not on Drive
"""
    )

    # Keep cell 4 (2a) as-is if it already has DO_UPLOAD=False; rewrite safely
    nb["cells"][4]["source"] = lines(
        r'''# ══════════════════════════════════════════════════════════════════════════════
# CELL 2a — OPTIONAL: upload a custom torch zip (skip by default)
# ══════════════════════════════════════════════════════════════════════════════
DO_UPLOAD = False  # leave False — Cell 2b pulls Woojae WB sweep from Drive
if DO_UPLOAD and IN_COLAB:
    from google.colab import files
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    uploaded = files.upload()
    for name in uploaded:
        (UPLOAD_DIR / name).write_bytes(uploaded[name])
        print("Saved", UPLOAD_DIR / name)
else:
    print("Upload skipped (default). Cell 2b uses Drive / repo zips.")
'''
    )

    nb["cells"][5]["source"] = lines(
        r'''# ══════════════════════════════════════════════════════════════════════════════
# CELL 2b — Woojae × all capture WB cells (auto from Drive / local Downloads)
# ══════════════════════════════════════════════════════════════════════════════
import re
from pipeline.illuminant_estimation import infer_illuminant_label

# Fixed WB-factorial subject (full D65 A–E + F12 B–E; no F12-A in cohort).
WB_SWEEP_PERSON = "Woojae"
MANIFEST_PATH = REPO / "data" / "ring_light" / "wb_sweep_woojae.json"
DEMO_DIR = REPO / "data" / "ring_light" / "demo_zips"

RING_ROOT_CANDIDATES = [
    Path("/content/drive/MyDrive/Variable Lighting Ring Light/Variable Lighting Ring Light"),
    Path("/content/drive/MyDrive/Variable Lighting Ring Light"),
    Path.home() / "Downloads" / "Variable Lighting Ring Light-20260829T185351Z-1-001" / "Variable Lighting Ring Light",
]
RING_ROOT = next((p for p in RING_ROOT_CANDIDATES if p.is_dir()), RING_ROOT_CANDIDATES[0])

def _parse_wb_cell(stem: str) -> str:
    s = stem.replace(" ", "")
    m = (
        re.search(r"(?:D65|F12|D12)[\-_]?([A-E])\d", s, re.I)
        or re.search(r"(?:D65|F12)([A-E])\d", s, re.I)
    )
    return m.group(1).upper() if m else "?"

assert MANIFEST_PATH.is_file(), f"Missing {MANIFEST_PATH} — git pull / push wb_sweep_woojae.json"
manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
demo_rows = []
candidates = []
seen = set()
MODE = "wb_sweep"  # Cell 3 comparison tables

def _add_zip(path: Path, meta=None) -> None:
    path = Path(path)
    key = path.name
    if key in seen or not path.is_file():
        return
    seen.add(key)
    candidates.append(path)
    row = dict(meta or {"file": key, "person": "?", "illuminant": infer_illuminant_label(path) or "?"})
    row.setdefault("wb_cell", _parse_wb_cell(path.stem))
    demo_rows.append(row)

missing = []
for d in manifest.get("demos", []):
    fname = d["file"]
    hits = []
    if (DEMO_DIR / fname).is_file():
        hits.append(DEMO_DIR / fname)
    if RING_ROOT.is_dir():
        hits.extend(RING_ROOT.rglob(fname))
    if hits:
        _add_zip(hits[0], d)
    else:
        missing.append(fname)
        print("WARN: missing zip", fname)

if UPLOAD_DIR.is_dir():
    for p in sorted(UPLOAD_DIR.glob("*.zip")):
        _add_zip(p)

demo_names_ordered = [d["file"] for d in manifest.get("demos", [])]
demo_zips = [p for fname in demo_names_ordered for p in candidates if p.name == fname]

print(f"WB sweep person: {WB_SWEEP_PERSON}")
print(f"RING_ROOT: {RING_ROOT}  ({'OK' if RING_ROOT.is_dir() else 'MISSING — mount Drive'})")
if missing:
    print(f"Missing {len(missing)}/{len(demo_names_ordered)} zips under RING_ROOT / demo_zips.")
assert demo_zips, (
    "No Woojae WB-sweep zips found. Mount Drive with Variable Lighting Ring Light "
    "(or copy zips into data/ring_light/demo_zips). No Cell 2a upload needed if Drive is mounted."
)

print(f"\nLoaded {len(demo_zips)} zip(s)\n")
print(f"{'idx':>3}  {'person':7}  {'ill':3}  {'wb':2}  {'WB K':>5}  file")
print("-" * 64)
for i, p in enumerate(demo_zips):
    row = next((r for r in demo_rows if r.get("file") == p.name), {})
    ill = row.get("illuminant") or "?"
    wb = row.get("wb_cell") or _parse_wb_cell(p.stem)
    wbk = row.get("capture_wb_k", "?")
    print(f"{i:3d}  {row.get('person','?'):7}  {ill:3}  {wb:2}  {wbk!s:>5}  {p.name}")

print(f"\nCell 3 will run all {len(demo_zips)} zips and compare ΔE by wb_cell.")
'''
    )

    nb["cells"][6]["source"] = lines(
        r"""## 2 — Load pipeline + run Woojae WB sweep

Runs every zip from Cell 2b, then prints FitSkin ΔE **by capture wb_cell** (D65 and F12). Set `SHOW_VIZ_INDEX` (0–8) to preview one capture; `None` skips plots.
"""
    )

    cell3 = r'''# ══════════════════════════════════════════════════════════════════════════════
# CELL 3 — Best stack: Woojae WB sweep + ΔE comparison
# ══════════════════════════════════════════════════════════════════════════════
import cv2
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from delta_e_2000 import delta_e_2000
from scripts.evaluate_pansor20_chartfree_d65 import (
    extract_zip,
    linear_rgb_to_preview_bgr,
    load_apple_landmarks,
    load_dng_linear,
)
from pipeline.skin_roi import apple_face_cheek_masks, apple_face_skin_roi_mask

try:
    from pipeline.skin_roi import apple_face_forehead_lab_pool_mask
except ImportError:

    def apple_face_forehead_lab_pool_mask(landmarks, dst_h, dst_w, *, linear_rgb=None):
        forehead = apple_face_skin_roi_mask(
            landmarks, dst_h, dst_w, roi="forehead", linear_rgb=linear_rgb
        )
        _, cheek = apple_face_cheek_masks(landmarks, dst_h, dst_w)
        return (forehead | (cheek > 0)).astype(bool)

from models.fairface_race import face_rgb_crop_from_landmarks

CAT_MODE = "hybrid_deploy"
SAMPLING = "fairface7"
ROI = "forehead"
SHOW_VIZ_INDEX = 0  # 0–N preview one zip; None = table only

pipe = D65FairFace7ROIPipeline.from_defaults(
    cal_dir=CAL_DIR,
    fairface_dir=FAIRFACE_DIR,
    cat_mode=CAT_MODE,
    torch_dir=TORCH_DIR if TORCH_DIR.is_dir() else None,
    multi_lab_affine=MULTI_LAB,
    half_size=True,
    sampling=SAMPLING,
    roi=ROI,
)
print(f"Pipeline: roi={ROI}  cat_mode={CAT_MODE}  multi_lab={MULTI_LAB.name}  sampling={SAMPLING}\n")

cohort_results = {}
summary_rows = []

for i, zp in enumerate(demo_zips):
    meta = next((r for r in demo_rows if r.get("file") == zp.name), {})
    person = meta.get("person", "?")
    ill = meta.get("illuminant") or "?"
    wb = meta.get("wb_cell") or "?"
    wbk = meta.get("capture_wb_k")
    ref = meta.get("fitskin_forehead")
    try:
        r = pipe.run_zip(zp)
    except Exception as exc:
        print(f"[{i:02d}] FAIL {zp.name}: {exc}")
        summary_rows.append(
            {"idx": i, "file": zp.name, "person": person, "ill": ill, "wb": wb, "wbk": wbk, "error": str(exc)}
        )
        continue
    out_json = OUT_DIR / f"{zp.stem}.json"
    write_result_json(r, out_json)
    cohort_results[zp.name] = r
    pred = np.array([r["L"], r["a"], r["b"]], dtype=np.float64)
    de = float("nan")
    if ref:
        ref_lab = np.array([ref["L"], ref["a"], ref["b"]], dtype=np.float64)
        de = float(delta_e_2000(pred, ref_lab))
    ef = r.get("exposure_flags") or {}
    summary_rows.append(
        {
            "idx": i,
            "file": zp.name,
            "person": person,
            "ill": ill,
            "wb": wb,
            "wbk": wbk,
            "L": r["L"],
            "a": r["a"],
            "b": r["b"],
            "de": de,
            "cat_cct": r.get("cat_cct"),
            "l_sampling": r.get("l_sampling"),
            "pool": r.get("lab_pool_expanded"),
            "L_std": r.get("forehead_L_std"),
            "n_roi": r.get("n_roi"),
            "warn": "⚠" if ef.get("out_of_band") else "",
        }
    )

print(f"{'idx':>3}  {'ill':3}  {'wb':2}  {'ΔE':>5}  {'L*':>5}  {'a*':>5}  {'b*':>5}  pool  file")
print("-" * 78)
for row in summary_rows:
    if row.get("error"):
        print(f"{row['idx']:3d}  FAIL  {row['file']}: {row['error'][:50]}")
        continue
    pool_s = "Y" if row.get("pool") else "."
    print(
        f"{row['idx']:3d}  {row['ill']:3}  {row['wb']:2}  {row['de']:5.2f}  "
        f"{row['L']:5.1f}  {row['a']:5.1f}  {row['b']:5.1f}  {pool_s:4}  {row['file']}{row.get('warn','')}"
    )

ok_de = [row["de"] for row in summary_rows if row.get("de") == row.get("de")]
if ok_de:
    print(f"\nMean ΔE₀₀ (n={len(ok_de)}): {sum(ok_de) / len(ok_de):.2f}")

cohort_summary = OUT_DIR / "cohort_summary.json"
cohort_summary.write_text(json.dumps({"n": len(summary_rows), "rows": summary_rows}, indent=2) + "\n")
print("Wrote", cohort_summary)

# ── Compare ΔE across capture WB cells ───────────────────────────────────────
by_ill = defaultdict(list)
for row in summary_rows:
    if row.get("error") or row.get("de") != row.get("de"):
        continue
    by_ill[row["ill"]].append(row)

print("\n" + "=" * 72)
print(f"WB SWEEP — {WB_SWEEP_PERSON}  (pipeline ignores capture WB; pre-AWB + CAT)")
print("=" * 72)
for ill in ("D65", "F12"):
    rows_i = sorted(by_ill.get(ill, []), key=lambda r: r.get("wb") or "")
    if not rows_i:
        continue
    print(f"\n{ill} ring — FitSkin forehead ΔE₀₀ by capture wb_cell")
    print(f"{'wb':>3}  {'WB K':>5}  {'ΔE':>6}  {'L*':>6}  {'a*':>6}  {'b*':>6}  file")
    print("-" * 72)
    for r in rows_i:
        print(
            f"{r['wb']:>3}  {r.get('wbk','?'):!s>5}  {r['de']:6.2f}  "
            f"{r['L']:6.1f}  {r['a']:6.1f}  {r['b']:6.1f}  {r['file']}"
        )
    des = [r["de"] for r in rows_i]
    best = min(rows_i, key=lambda r: r["de"])
    worst = max(rows_i, key=lambda r: r["de"])
    print(
        f"  mean={sum(des)/len(des):.2f}  "
        f"best={best['wb']}({best['de']:.2f})  worst={worst['wb']}({worst['de']:.2f})  "
        f"range={max(des)-min(des):.2f}"
    )

fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
for ax, ill in zip(axes, ("D65", "F12")):
    rows_i = sorted(by_ill.get(ill, []), key=lambda r: r.get("wb") or "")
    if not rows_i:
        ax.set_visible(False)
        continue
    xs = [r["wb"] for r in rows_i]
    ys = [r["de"] for r in rows_i]
    ax.bar(xs, ys, color="#2563eb" if ill == "D65" else "#dc2626", alpha=0.85)
    for i, y in enumerate(ys):
        ax.text(i, y + 0.1, f"{y:.1f}", ha="center", fontsize=8)
    ax.set_title(f"{ill} — {WB_SWEEP_PERSON}")
    ax.set_xlabel("capture wb_cell")
    ax.set_ylabel("ΔE₀₀ vs FitSkin forehead")
plt.suptitle("Capture WB factorial (pipeline does not apply phone WB)", fontsize=11)
plt.tight_layout()
plt.show()

# Optional single-zip viz
if SHOW_VIZ_INDEX is not None and summary_rows:
    vi = int(SHOW_VIZ_INDEX) % len(demo_zips)
    ZIP_PATH = demo_zips[vi]
    result = cohort_results.get(ZIP_PATH.name)
    if result is None:
        print(f"\nNo result for viz index {vi} ({ZIP_PATH.name})")
    else:
        print(f"\nViz zip [{vi}] {ZIP_PATH.name}")
        work = OUT_DIR / "_viz"
        nf, fl, lm_path = extract_zip(ZIP_PATH, work / ZIP_PATH.stem)
        A0 = load_dng_linear(nf, half_size=True, use_camera_wb=False)
        lm = load_apple_landmarks(lm_path)
        roi_mask = apple_face_skin_roi_mask(lm, A0.shape[0], A0.shape[1], roi=ROI, linear_rgb=A0)
        if result.get("lab_pool_expanded"):
            roi_mask = apple_face_forehead_lab_pool_mask(lm, A0.shape[0], A0.shape[1], linear_rgb=A0)
        preview = linear_rgb_to_preview_bgr(A0)
        overlay = preview.copy()
        overlay[roi_mask > 0] = (0.55 * overlay[roi_mask > 0] + 0.45 * np.array([0, 220, 80])).astype(np.uint8)
        face_rgb = face_rgb_crop_from_landmarks(preview, lm, padding=0.35)

        def lab_to_srgb_u8(L, a, b):
            fy = (L + 16.0) / 116.0
            fx, fz = fy + a / 500.0, fy - b / 200.0
            eps, kappa = 216 / 24389, 24389 / 27

            def finv(t):
                return t**3 if t**3 > eps else (116 * t - 16) / kappa

            X, Y, Z = 0.95047 * finv(fx), finv(fy), 1.08883 * finv(fz)
            M = np.array(
                [[3.2406, -1.5372, -0.4986], [-0.9689, 1.8758, 0.0415], [0.0557, -0.2040, 1.0570]]
            )
            rgb = M @ np.array([X, Y, Z])
            lin2s = lambda u: 12.92 * u if u <= 0.0031308 else 1.055 * (max(u, 0) ** (1 / 2.4)) - 0.055
            return (np.clip([lin2s(float(c)) for c in rgb], 0, 1) * 255).astype(np.uint8)

        swatch = np.full((180, 180, 3), lab_to_srgb_u8(result["L"], result["a"], result["b"]), dtype=np.uint8)
        roi_title = "Forehead Lab pool" if result.get("lab_pool_expanded") else "Forehead ROI"
        fig, ax = plt.subplots(1, 4, figsize=(14, 3.6))
        ax[0].imshow(cv2.cvtColor(preview, cv2.COLOR_BGR2RGB))
        ax[0].set_title("No-flash")
        ax[0].axis("off")
        ax[1].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
        ax[1].set_title(f"{roi_title} (n={result.get('n_roi')})")
        ax[1].axis("off")
        ax[2].imshow(face_rgb)
        ax[2].set_title(f"FairFace-7\n{result.get('fairface_label')} → {result.get('predicted_ethnicity')}")
        ax[2].axis("off")
        ax[3].imshow(swatch)
        ax[3].set_title(f"Forehead Lab\n({result['L']:.1f}, {result['a']:.1f}, {result['b']:.1f})")
        ax[3].axis("off")
        plt.suptitle(f"{ZIP_PATH.name}  ·  {result.get('illuminant_label')}", fontsize=11)
        plt.tight_layout()
        plt.show()
else:
    ZIP_PATH = demo_zips[0]
    result = cohort_results.get(ZIP_PATH.name)
'''

    # Fix the format string typo I almost introduced
    cell3 = cell3.replace("{r.get('wbk','?'):!s>5}", "{str(r.get('wbk', '?')):>5}")

    nb["cells"][7]["source"] = lines(cell3)

    # Cell 4 compare still uses ZIP_PATH / result / ROI / SAMPLING — OK
    NB.write_text(json.dumps(nb, indent=1) + "\n", encoding="utf-8")
    print("Updated", NB)

    # sanity
    s3 = "".join(nb["cells"][7]["source"])
    assert "except ImportError" in s3
    assert "WB SWEEP" in s3
    s1 = "".join(nb["cells"][2]["source"])
    assert "apple_face_forehead_lab_pool_mask" in s1
    s2b = "".join(nb["cells"][5]["source"])
    assert "wb_sweep_woojae.json" in s2b
    assert "MODE" in s2b and "selectable" not in s2b.lower()
    print("sanity OK")


if __name__ == "__main__":
    main()
