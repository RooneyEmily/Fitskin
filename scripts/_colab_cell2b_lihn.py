# ══════════════════════════════════════════════════════════════════════════════
# CELL 2b — Lihn × all capture WB cells (auto-find on Drive; no upload)
# ══════════════════════════════════════════════════════════════════════════════
import re
from collections import defaultdict

WB_SWEEP_PERSON = "Lihn"
MODE = "wb_sweep"
MANIFEST_PATH = REPO / "data" / "ring_light" / "wb_sweep_lihn.json"

# Zip stems often say "Linh" — treat both as the same person.
_NAME_RE = re.compile(r"lihn|linh", re.I)

CAPTURE_WB_K = {
    "D65": {"A": 5500, "B": 6000, "C": 6500, "D": 7000, "E": 7500},
    "F12": {"A": 2500, "B": 2500, "C": 3000, "D": 3500, "E": 4000},
}
FITSKIN = {
    "D65": {"L": 55.24, "a": 10.98, "b": 20.58},
    "F12": {"L": 57.44, "a": 10.46, "b": 21.16},
}
NEEDED = [("D65", c) for c in "ABCDE"] + [("F12", c) for c in "BCDE"]

if IN_COLAB and not Path("/content/drive/MyDrive").is_dir():
    print("Mounting Google Drive …")
    from google.colab import drive
    drive.mount("/content/drive")
if IN_COLAB and not Path("/content/drive/MyDrive").is_dir():
    raise RuntimeError("Drive not mounted — re-run Cell 1 and accept the permission prompt.")


def _parse_lihn_zip(stem: str):
    s = stem.replace(" ", "")
    if not _NAME_RE.search(s):
        return None
    m = re.search(r"(D65|F12|D12)[\-_]?([A-E])(\d+)", s, re.I)
    if not m:
        return None
    ill = m.group(1).upper()
    if ill == "D12":
        ill = "F12"
    return ill, m.group(2).upper(), int(m.group(3))


def _search_roots():
    """Prefer known folder names — avoid rglob of all MyDrive (very slow)."""
    roots = []
    md = Path("/content/drive/MyDrive")
    if md.is_dir():
        explicit = [
            md / "Variable Lighting Ring Light" / "Variable Lighting Ring Light",
            md / "Variable Lighting Ring Light",
            md / "Variable Lighting Ring Light-20260829T185351Z-1-001" / "Variable Lighting Ring Light",
        ]
        for p in explicit:
            if p.is_dir():
                roots.append(p)
        for p in md.glob("Variable Lighting*"):
            if p.is_dir():
                roots.append(p)
                nested = p / "Variable Lighting Ring Light"
                if nested.is_dir():
                    roots.append(nested)
        for base in list(roots):
            for person in list(base.glob("Lihn*")) + list(base.glob("Linh*")):
                if person.is_dir():
                    roots.append(person)
                    for ill in ("D65", "F12"):
                        sub = person / ill
                        if sub.is_dir():
                            roots.append(sub)
    for p in [
        REPO / "data" / "ring_light" / "demo_zips",
        Path.home()
        / "Downloads"
        / "Variable Lighting Ring Light-20260829T185351Z-1-001"
        / "Variable Lighting Ring Light",
        UPLOAD_DIR,
    ]:
        if p.is_dir():
            roots.append(p)

    out, seen = [], set()
    for r in roots:
        key = str(r.resolve())
        if key not in seen:
            seen.add(key)
            out.append(r)
    return out


def _glob_person_zips(root: Path):
    """Match Lihn/Linh *Torch.zip with any capitalization."""
    hits = []
    for pat in ("*Lihn*Torch.zip", "*Lihn*torch.zip", "*Linh*Torch.zip", "*Linh*torch.zip"):
        hits.extend(root.glob(pat))
    if not hits:
        for pat in ("*Lihn*Torch.zip", "*Lihn*torch.zip", "*Linh*Torch.zip", "*Linh*torch.zip"):
            hits.extend(root.rglob(pat))
    # de-dupe
    seen, out = set(), []
    for p in hits:
        k = str(p.resolve())
        if k not in seen:
            seen.add(k)
            out.append(p)
    return out


print("Searching for Lihn/Linh *Torch.zip (limited folder list) …")
roots = _search_roots()
print(f"Search roots ({len(roots)}):")
for r in roots:
    print(" ", r)
if not roots:
    raise RuntimeError(
        "No search roots found. Put the Variable Lighting Ring Light folder in My Drive "
        "(or Lihn P1/D65 and Lihn P1/F12), then re-run Cell 1 + Cell 2b."
    )

by_cell = defaultdict(list)
scanned = 0
for root in roots:
    for zp in _glob_person_zips(root):
        scanned += 1
        parsed = _parse_lihn_zip(zp.stem)
        if not parsed:
            continue
        ill, cell, trial = parsed
        by_cell[(ill, cell)].append((trial, zp))

print(
    f"Parsed Lihn/Linh torch zips covering {len(by_cell)} illuminant×cell buckets "
    f"(scanned {scanned} paths)"
)

preferred_name = {}
if MANIFEST_PATH.is_file():
    man = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    for d in man.get("demos", []):
        preferred_name[(d["illuminant"], d["wb_cell"])] = d["file"]

demo_rows, demo_zips, missing = [], [], []
for ill, cell in NEEDED:
    cands = by_cell.get((ill, cell), [])
    if not cands:
        missing.append(f"{ill}-{cell}")
        print(f"WARN: no zip for {ill} cell {cell}")
        continue
    pref = preferred_name.get((ill, cell))
    chosen = None
    if pref:
        for trial, zp in cands:
            if zp.name == pref or zp.name.lower() == pref.lower():
                chosen = zp
                break
    if chosen is None:
        chosen = sorted(cands, key=lambda t: (t[0], t[1].name))[0][1]
    meta = {
        "file": chosen.name,
        "person": WB_SWEEP_PERSON,
        "illuminant": ill,
        "wb_cell": cell,
        "capture_wb_k": CAPTURE_WB_K[ill][cell],
        "fitskin_forehead": FITSKIN[ill],
    }
    demo_rows.append(meta)
    demo_zips.append(chosen)

if not demo_zips:
    raise RuntimeError(
        "No Lihn WB-sweep zips found under the search roots above.\n"
        "Expected files like Lihn-F12-B1Torch.zip / LinhD65-A1Torch.zip under …/Lihn P1/.\n"
        "Confirm the folder is in My Drive, re-run Cell 1 (accept Drive), then Cell 2b."
    )

candidates = list(demo_zips)
demo_names_ordered = [r["file"] for r in demo_rows]

print(f"\nWB sweep person: {WB_SWEEP_PERSON}")
print(
    f"Loaded {len(demo_zips)}/{len(NEEDED)} cells"
    + (f"  (missing: {', '.join(missing)})" if missing else "")
)
print(f"\n{'idx':>3}  {'ill':3}  {'wb':2}  {'WB K':>5}  file")
print("-" * 64)
for i, (meta, p) in enumerate(zip(demo_rows, demo_zips)):
    print(f"{i:3d}  {meta['illuminant']:3}  {meta['wb_cell']:2}  {meta['capture_wb_k']!s:>5}  {p.name}")
    print(f"      ↳ {p}")

print(f"\nCell 3 will run all {len(demo_zips)} zips and compare ΔE by wb_cell.")
