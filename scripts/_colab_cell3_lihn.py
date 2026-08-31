# ══════════════════════════════════════════════════════════════════════════════
# CELL 3 — Lihn WB sweep + ΔE comparison + illumination / segmentation viz
# ══════════════════════════════════════════════════════════════════════════════
import cv2
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from pathlib import Path
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
SHOW_VIZ_INDEX = 0  # deep-dive one zip (segmentation + Lab swatch)
SHOW_ILLUM_GRID = True  # thumbnail grid: D65 row + F12 row
SHOW_ALL_SEGMENTATIONS = False  # overlay every loaded zip (can be large)

if "from physio_skin_lab_monk import" in (REPO / "scripts" / "evaluate_pansor20_chartfree_d65.py").read_text(
    encoding="utf-8"
):
    raise RuntimeError("Re-run Cell 1 — eval script still imports mediapipe via physio_skin_lab_monk")

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
print(f"Pipeline: roi={ROI}  cat_mode={CAT_MODE}  multi_lab={MULTI_LAB.name}  sampling={SAMPLING}")
if pipe.torch_prior is not None:
    tp = pipe.torch_prior
    _ts = "MK350 measured" if tp.n_files else "calibration bundle (Cell 1 plot)"
    print(f"Torch SPD in use: {_ts}  CCT≈{tp.torch_cct_k:.0f} K  files={tp.files}\n")
else:
    print()

cohort_results = {}
summary_rows = []
preview_cache = {}  # zip stem -> (preview_bgr, lm, result meta for viz)


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


def segmentation_overlay(preview_bgr, lm, linear_rgb, result):
    """Forehead=green, cheek=blue, Lab pool=yellow (forehead∪cheek when expanded)."""
    h, w = preview_bgr.shape[:2]
    forehead = apple_face_skin_roi_mask(lm, h, w, roi="forehead", linear_rgb=linear_rgb)
    _, cheek = apple_face_cheek_masks(lm, h, w)
    pool = apple_face_forehead_lab_pool_mask(lm, h, w, linear_rgb=linear_rgb)
    lab_mask = pool if result.get("lab_pool_expanded") else forehead

    ov = preview_bgr.copy().astype(np.float32)
    # cheek only (not forehead)
    cheek_only = (cheek > 0) & ~forehead
    ov[cheek_only] = 0.55 * ov[cheek_only] + 0.45 * np.array([60, 140, 255])  # blue
    ov[forehead] = 0.50 * ov[forehead] + 0.50 * np.array([40, 220, 80])  # green
    if result.get("lab_pool_expanded"):
        ov[lab_mask] = 0.45 * ov[lab_mask] + 0.55 * np.array([40, 255, 255])  # yellow pool
    return ov.astype(np.uint8), forehead, cheek, lab_mask


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
    dL = da = db = float("nan")
    if ref:
        ref_lab = np.array([ref["L"], ref["a"], ref["b"]], dtype=np.float64)
        de = float(delta_e_2000(pred, ref_lab))
        dL, da, db = pred[0] - ref_lab[0], pred[1] - ref_lab[1], pred[2] - ref_lab[2]
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
            "dL": dL,
            "da": da,
            "db": db,
            "cat_cct": r.get("cat_cct"),
            "l_sampling": r.get("l_sampling"),
            "pool": r.get("lab_pool_expanded"),
            "L_std": r.get("forehead_L_std"),
            "n_roi": r.get("n_roi"),
            "warn": "⚠" if ef.get("out_of_band") else "",
            "L_high": ef.get("L_ge_75"),
            "ref_L": ref["L"] if ref else None,
        }
    )

    # cache preview for illumination grid / segmentation
    try:
        work = OUT_DIR / "_viz_cache"
        nf, fl, lm_path = extract_zip(zp, work / zp.stem)
        A0 = load_dng_linear(nf, half_size=True, use_camera_wb=False)
        lm = load_apple_landmarks(lm_path)
        preview = linear_rgb_to_preview_bgr(A0)
        preview_cache[zp.stem] = (preview, lm, A0, r)
    except Exception as exc:
        print(f"  (viz cache skip {zp.name}: {exc})")

print(f"{'idx':>3}  {'ill':3}  {'wb':2}  {'ΔE':>5}  {'L*':>5}  {'ΔL*':>5}  {'pool':4}  file")
print("-" * 78)
for row in summary_rows:
    if row.get("error"):
        print(f"{row['idx']:3d}  FAIL  {row['file']}: {row['error'][:50]}")
        continue
    pool_s = "pool" if row.get("pool") else "fore"
    print(
        f"{row['idx']:3d}  {row['ill']:3}  {row['wb']:2}  {row['de']:5.2f}  "
        f"{row['L']:5.1f}  {row['dL']:+5.1f}  {pool_s:4}  {row['file']}{row.get('warn','')}"
    )

ok_de = [row["de"] for row in summary_rows if row.get("de") == row.get("de")]
if ok_de:
    print(f"\nMean ΔE₀₀ (n={len(ok_de)}): {sum(ok_de) / len(ok_de):.2f}")

# ── Why is D65 ΔE high? (diagnostic) ─────────────────────────────────────────
d65_rows = [r for r in summary_rows if r.get("ill") == "D65" and r.get("de") == r.get("de")]
f12_rows = [r for r in summary_rows if r.get("ill") == "F12" and r.get("de") == r.get("de")]
if d65_rows:
    mean_d65 = sum(r["de"] for r in d65_rows) / len(d65_rows)
    mean_L_d65 = sum(r["L"] for r in d65_rows) / len(d65_rows)
    ref_L = d65_rows[0].get("ref_L") or 55.24
    print("\n" + "=" * 72)
    print(f"D65 ΔE diagnostic ({WB_SWEEP_PERSON})")
    print("=" * 72)
    print(f"  FitSkin forehead ref L* ≈ {ref_L:.1f}  |  {WB_SWEEP_PERSON} D65 mean L* ≈ {mean_L_d65:.1f}  (ΔL* ≈ {mean_L_d65 - ref_L:+.1f})")
    print(f"  Mean ΔE₀₀ on D65: {mean_d65:.1f}  (cells range {min(r['de'] for r in d65_rows):.1f}–{max(r['de'] for r in d65_rows):.1f})")
    if f12_rows:
        mean_f12 = sum(r["de"] for r in f12_rows) / len(f12_rows)
        print(f"  Compare F12 mean ΔE₀₀: {mean_f12:.1f}  — warm ring + cheek pool usually tracks FitSkin")
    print(
        "  Likely cause: D65 captures run bright (L* 65–75). Pipeline CAT is OK (~5500 K frozen on D65);"
        "\n  error is mostly ΔL* (exposure), not wrong illuminant. Re-check ring SPD (~6500 K) vs capture exposure."
    )
    n_pool = sum(1 for r in d65_rows if r.get("pool"))
    print(f"  Cheek pool on D65: {n_pool}/{len(d65_rows)} zips (needs uniform forehead L* + Indian/F12 rule).")

cohort_summary = OUT_DIR / "cohort_summary.json"
cohort_summary.write_text(json.dumps({"n": len(summary_rows), "rows": summary_rows}, indent=2) + "\n")
print("\nWrote", cohort_summary)

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
    print(f"{'wb':>3}  {'WB K':>5}  {'ΔE':>6}  {'L*':>6}  {'ΔL*':>6}  file")
    print("-" * 72)
    for r in rows_i:
        print(
            f"{r['wb']:>3}  {str(r.get('wbk', '?')):>5}  {r['de']:6.2f}  "
            f"{r['L']:6.1f}  {r['dL']:+6.1f}  {r['file']}"
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
    for i, (r, y) in enumerate(zip(rows_i, ys)):
        ax.text(i, y + 0.15, f"{y:.1f}", ha="center", fontsize=8)
        if r.get("L_high") or r.get("warn"):
            ax.text(i, y * 0.5, "L*↑", ha="center", fontsize=7, color="white", fontweight="bold")
    ax.set_title(f"{ill} — {WB_SWEEP_PERSON}")
    ax.set_xlabel("capture wb_cell")
    ax.set_ylabel("ΔE₀₀ vs FitSkin forehead")
    ax.axhline(5, color="gray", ls=":", lw=0.8, alpha=0.6)
plt.suptitle("Capture WB factorial (phone WB not applied in pipeline)", fontsize=11)
plt.tight_layout()
plt.show()

# ── Illumination thumbnail grid (D65 vs F12 appearance) ────────────────────
if SHOW_ILLUM_GRID and preview_cache:
    cells_d65 = sorted(by_ill.get("D65", []), key=lambda r: r["wb"])
    cells_f12 = sorted(by_ill.get("F12", []), key=lambda r: r["wb"])
    ncols = max(len(cells_d65), len(cells_f12), 1)
    fig, axes = plt.subplots(2, ncols, figsize=(2.2 * ncols, 5.2))
    if ncols == 1:
        axes = np.array([[axes[0]], [axes[1]]])

    for j in range(ncols):
        ax = axes[0, j]
        if j < len(cells_d65):
            row = cells_d65[j]
            stem = Path(row["file"]).stem
            cached = preview_cache.get(stem)
            if cached:
                preview, lm, A0, res = cached
                face_rgb = face_rgb_crop_from_landmarks(preview, lm, padding=0.35)
                ax.imshow(face_rgb)
                ax.set_title(
                    f"D65 cell {row['wb']}  ΔE={row['de']:.1f}\nL*={row['L']:.0f} (ref {row.get('ref_L', 55):.0f})",
                    fontsize=8,
                )
            else:
                ax.text(0.5, 0.5, row["file"], ha="center", va="center", transform=ax.transAxes, fontsize=7)
        else:
            ax.axis("off")
        ax.set_ylabel("D65 ring" if j == 0 else "", fontsize=9)

    for j in range(ncols):
        ax = axes[1, j]
        if j < len(cells_f12):
            row = cells_f12[j]
            stem = Path(row["file"]).stem
            cached = preview_cache.get(stem)
            if cached:
                preview, lm, A0, res = cached
                face_rgb = face_rgb_crop_from_landmarks(preview, lm, padding=0.35)
                ax.imshow(face_rgb)
                ax.set_title(
                    f"F12 cell {row['wb']}  ΔE={row['de']:.1f}\nL*={row['L']:.0f} (ref {row.get('ref_L', 57):.0f})",
                    fontsize=8,
                )
            else:
                ax.text(0.5, 0.5, row["file"], ha="center", va="center", transform=ax.transAxes, fontsize=7)
        else:
            ax.axis("off")

    plt.suptitle(
        f"{WB_SWEEP_PERSON}: face crops under D65 (~6500 K ring) vs F12 (~3000 K ring)\n"
        "D65 often looks brighter / washed — high L* drives ΔE, not CAT failure",
        fontsize=10,
    )
    plt.tight_layout()
    plt.show()

# ── Deep-dive: segmentation for one zip ──────────────────────────────────────
if SHOW_VIZ_INDEX is not None and summary_rows:
    vi = int(SHOW_VIZ_INDEX) % len(demo_zips)
    ZIP_PATH = demo_zips[vi]
    result = cohort_results.get(ZIP_PATH.name)
    meta = next((r for r in demo_rows if r.get("file") == ZIP_PATH.name), {})
    ref = meta.get("fitskin_forehead")
    if result is None:
        print(f"\nNo result for viz index {vi} ({ZIP_PATH.name})")
    else:
        print(f"\nSegmentation viz [{vi}] {ZIP_PATH.name}")
        cached = preview_cache.get(ZIP_PATH.stem)
        if cached is None:
            work = OUT_DIR / "_viz"
            nf, fl, lm_path = extract_zip(ZIP_PATH, work / ZIP_PATH.stem)
            A0 = load_dng_linear(nf, half_size=True, use_camera_wb=False)
            lm = load_apple_landmarks(lm_path)
            preview = linear_rgb_to_preview_bgr(A0)
        else:
            preview, lm, A0, _ = cached

        seg_ov, forehead, cheek, lab_mask = segmentation_overlay(preview, lm, A0, result)
        face_rgb = face_rgb_crop_from_landmarks(preview, lm, padding=0.35)
        pred_swatch = np.full((160, 160, 3), lab_to_srgb_u8(result["L"], result["a"], result["b"]), dtype=np.uint8)
        ref_swatch = None
        if ref:
            ref_swatch = np.full((160, 160, 3), lab_to_srgb_u8(ref["L"], ref["a"], ref["b"]), dtype=np.uint8)

        fig, ax = plt.subplots(2, 3, figsize=(12, 7.5))
        ax[0, 0].imshow(cv2.cvtColor(preview, cv2.COLOR_BGR2RGB))
        ax[0, 0].set_title("No-flash (pre-AWB preview)")
        ax[0, 0].axis("off")

        ax[0, 1].imshow(cv2.cvtColor(seg_ov, cv2.COLOR_BGR2RGB))
        pool_note = "yellow=Lab pool (forehead∪cheek)" if result.get("lab_pool_expanded") else "green=forehead Lab ROI"
        ax[0, 1].set_title(f"Segmentation\n{pool_note}\nblue=cheek band")
        ax[0, 1].axis("off")

        ax[0, 2].imshow(face_rgb)
        ax[0, 2].set_title(
            f"FairFace-7: {result.get('fairface_label')} → {result.get('predicted_ethnicity')}\n"
            f"sampling={result.get('l_sampling')}  n_roi={result.get('n_roi')}"
        )
        ax[0, 2].axis("off")

        ax[1, 0].imshow(pred_swatch)
        ax[1, 0].set_title(f"Predicted Lab\n({result['L']:.1f}, {result['a']:.1f}, {result['b']:.1f})")
        ax[1, 0].axis("off")

        if ref_swatch is not None:
            ax[1, 1].imshow(ref_swatch)
            de = float(delta_e_2000(
                np.array([result["L"], result["a"], result["b"]]),
                np.array([ref["L"], ref["a"], ref["b"]]),
            ))
            ax[1, 1].set_title(
                f"FitSkin ref ({meta.get('illuminant')})\n"
                f"({ref['L']:.1f}, {ref['a']:.1f}, {ref['b']:.1f})\nΔE₀₀={de:.2f}"
            )
        else:
            ax[1, 1].axis("off")

        ef = result.get("exposure_flags") or {}
        diag = (
            f"illuminant (zip): {result.get('illuminant_label')}\n"
            f"CAT CCT: {result.get('cat_cct')}  Lu: {result.get('lu_cct_k')}\n"
            f"forehead L* std: {result.get('forehead_L_std', 0):.2f}\n"
            f"cheek pool: {result.get('lab_pool_expanded')}\n"
            f"capture wb_cell: {meta.get('wb_cell')} ({meta.get('capture_wb_k')} K setting)\n"
            f"exposure: {'OUT OF BAND' if ef.get('out_of_band') else 'OK'}"
            + ("  L*≥75" if ef.get("L_ge_75") else "")
        )
        ax[1, 2].text(0.05, 0.95, diag, va="top", fontsize=10, family="monospace", transform=ax[1, 2].transAxes)
        ax[1, 2].axis("off")
        ax[1, 2].set_title("Diagnostics")

        plt.suptitle(f"{ZIP_PATH.name}  ·  {result.get('illuminant_label')} ring", fontsize=11)
        plt.tight_layout()
        plt.show()

# Optional: segmentation montage for every zip
if SHOW_ALL_SEGMENTATIONS and preview_cache:
    keys = [Path(r["file"]).stem for r in summary_rows if not r.get("error")]
    n = len(keys)
    ncols = min(5, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.5 * ncols, 2.8 * nrows))
    axes = np.atleast_2d(axes)
    for k, stem in enumerate(keys):
        i, j = divmod(k, ncols)
        ax = axes[i, j]
        row = next(r for r in summary_rows if Path(r["file"]).stem == stem)
        res = cohort_results.get(row["file"])
        preview, lm, A0, _ = preview_cache[stem]
        seg_ov, _, _, _ = segmentation_overlay(preview, lm, A0, res)
        ax.imshow(cv2.cvtColor(seg_ov, cv2.COLOR_BGR2RGB))
        ax.set_title(f"{row['ill']} {row['wb']} ΔE={row['de']:.1f}", fontsize=8)
        ax.axis("off")
    for k in range(n, nrows * ncols):
        i, j = divmod(k, ncols)
        axes[i, j].axis("off")
    plt.suptitle("All captures — green forehead, blue cheek, yellow=Lab pool", fontsize=10)
    plt.tight_layout()
    plt.show()
else:
    ZIP_PATH = demo_zips[0]
    result = cohort_results.get(ZIP_PATH.name)
