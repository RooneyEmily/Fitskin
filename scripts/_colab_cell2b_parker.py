# ══════════════════════════════════════════════════════════════════════════════
# CELL 2b — Parker × all capture WB cells (auto-find zips; no upload)
# ══════════════════════════════════════════════════════════════════════════════
#
# Filename cheat sheet (Parker-D65-A1Torch.zip):
#   • D65 / F12  = ring light program (booth illuminant)
#   • A–E        = capture wb_cell (phone white-balance setting; NOT applied in pipeline)
#   • 1,2,3…     = trial / replicate number (A1 = cell A, first take)
#
import re
from collections import defaultdict

WB_SWEEP_PERSON = "Parker"
MODE = "wb_sweep"
MANIFEST_PATH = REPO / "data" / "ring_light" / "wb_sweep_parker.json"
DEMO_DIR = REPO / "data" / "ring_light" / "demo_zips"
USE_REPO_ZIPS = True  # True = use zips shipped in git (no Drive / Shared-with-me)

# ── Optional: flat Drive download folders if repo zips missing ────────────────
# Colab example — Shared-with-me data must be shortcutted to My Drive, OR use flat folders:
# EXTRA_ZIP_DIRS = [
#     Path("/content/drive/MyDrive/drive-download-20260831T103650Z-1-001"),  # D65
#     Path("/content/drive/MyDrive/drive-download-20260831T103757Z-1-001"),  # F12
# ]
EXTRA_ZIP_DIRS = []  # e.g. local: Path.home() / "Downloads" / "drive-download-..."

_NAME_RE = re.compile(r"parker", re.I)

CAPTURE_WB_K = {
    "D65": {"A": 5500, "B": 6000, "C": 6500, "D": 7000, "E": 7500},
    "F12": {"A": 2500, "B": 2500, "C": 3000, "D": 3500, "E": 4000},
}
FITSKIN = {
    "D65": {"L": 55.99, "a": 11.89, "b": 20.53},
    "F12": {"L": 56.54, "a": 11.27, "b": 20.21},
}
NEEDED = [("D65", c) for c in "ABCDE"] + [("F12", c) for c in "BCDE"]

print(
    "Labeling: Parker-{D65|F12}-{A-E}{trial}Torch.zip  →  "
    "ring × capture wb_cell × replicate (pipeline ignores phone WB)"
)

if not MANIFEST_PATH.is_file():
    raise RuntimeError(f"Missing {MANIFEST_PATH} — re-run Cell 1 (asset zip includes wb_sweep_parker.json)")
man = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def _parse_parker_zip(stem: str):
    """Parse Parker-D65-A1Torch / Parker-F12C2Torch / Parker-D12-D1Torch ."""
    s = stem.replace(" ", "").strip()
    if not _NAME_RE.search(s):
        return None
    m = re.search(r"(D65|F12|D12|F1)[\-_]?([A-E])(\d+)", s, re.I)
    if not m:
        return None
    ill = m.group(1).upper()
    if ill in ("D12", "F1"):
        ill = "F12"
    return ill, m.group(2).upper(), int(m.group(3))


def _search_roots():
    """Booth tree, Shared drives, flat drive-download-* folders, user extras."""
    roots = []
    md = Path("/content/drive/MyDrive")
    shareddrives = Path("/content/drive/Shareddrives")

    def _add_variable_lighting_tree(base: Path) -> None:
        if not base.is_dir():
            return
        for p in [
            base / "Variable Lighting Ring Light" / "Variable Lighting Ring Light",
            base / "Variable Lighting Ring Light",
        ]:
            if p.is_dir():
                roots.append(p)
        for p in base.glob("Variable Lighting*"):
            if p.is_dir():
                roots.append(p)
                nested = p / "Variable Lighting Ring Light"
                if nested.is_dir():
                    roots.append(nested)
        for p in base.glob("drive-download-*"):
            if p.is_dir():
                roots.append(p)

    if md.is_dir():
        _add_variable_lighting_tree(md)
        # Shortcuts from Shared with me often sit one level deep in My Drive
        for p in md.glob("*Variable Lighting*"):
            if p.is_dir() and p not in roots:
                roots.append(p)
                _add_variable_lighting_tree(p)
        for base in list(roots):
            for person in base.glob("Parker*"):
                if person.is_dir():
                    roots.append(person)
                    for ill in ("D65", "F12"):
                        sub = person / ill
                        if sub.is_dir():
                            roots.append(sub)

    if shareddrives.is_dir():
        for team in shareddrives.iterdir():
            if team.is_dir():
                _add_variable_lighting_tree(team)
                for person in team.glob("**/Parker*"):
                    if person.is_dir():
                        roots.append(person)
                        for ill in ("D65", "F12"):
                            sub = person / ill
                            if sub.is_dir():
                                roots.append(sub)

    for base in [Path.home() / "Downloads", UPLOAD_DIR]:
        if base.is_dir():
            roots.append(base)
            for p in base.glob("drive-download-*"):
                if p.is_dir():
                    roots.append(p)

    roots.append(
        Path.home()
        / "Downloads"
        / "Variable Lighting Ring Light-20260829T185351Z-1-001"
        / "Variable Lighting Ring Light"
    )
    roots.append(REPO / "data" / "ring_light" / "demo_zips")

    for p in EXTRA_ZIP_DIRS:
        p = Path(p)
        if p.is_dir():
            roots.append(p)
        else:
            print("WARN: EXTRA_ZIP_DIRS path not found:", p)

    out, seen = [], set()
    for r in roots:
        r = Path(r)
        if not r.is_dir():
            continue
        key = str(r.resolve())
        if key not in seen:
            seen.add(key)
            out.append(r)
    return out


def _glob_person_zips(root: Path):
    """Find Parker torch zips in root (flat folder) or one level down."""
    hits = []
    for pat in ("*Parker*Torch.zip", "*Parker*torch.zip", "*Parker*Torch .zip"):
        hits.extend(root.glob(pat))
    if not hits:
        for pat in ("*Parker*Torch.zip", "*Parker*torch.zip", "*Parker*Torch .zip"):
            hits.extend(root.rglob(pat))
    seen, out = set(), []
    for p in hits:
        k = str(p.resolve())
        if k not in seen:
            seen.add(k)
            out.append(p)
    return out


def _load_from_repo(manifest: dict):
    """Return (rows, zips) if all manifest files exist in DEMO_DIR."""
    if not USE_REPO_ZIPS or not DEMO_DIR.is_dir():
        return None
    rows, zips = [], []
    for d in manifest.get("demos", []):
        fname = d["file"]
        zp = DEMO_DIR / fname
        if not zp.is_file():
            for cand in DEMO_DIR.glob("*.zip"):
                if cand.name.replace(" ", "").lower() == fname.replace(" ", "").lower():
                    zp = cand
                    break
        if not zp.is_file():
            return None
        parsed = _parse_parker_zip(zp.stem)
        trial = parsed[2] if parsed else "?"
        rows.append(
            {
                "file": zp.name,
                "person": WB_SWEEP_PERSON,
                "illuminant": d["illuminant"],
                "wb_cell": d["wb_cell"],
                "trial": trial,
                "capture_wb_k": d.get("capture_wb_k") or CAPTURE_WB_K[d["illuminant"]][d["wb_cell"]],
                "fitskin_forehead": d.get("fitskin_forehead") or FITSKIN[d["illuminant"]],
            }
        )
        zips.append(zp)
    return rows, zips


_repo = _load_from_repo(man)
if _repo is not None:
    demo_rows, demo_zips = _repo
    candidates = list(demo_zips)
    demo_names_ordered = [r["file"] for r in demo_rows]
    print(f"\n✓ Loaded {len(demo_zips)} Parker zips from git ({DEMO_DIR.relative_to(REPO)}) — no Drive needed\n")
    print(f"{'idx':>3}  {'ring':3}  {'wb':2}  {'trial':5}  {'phone WB':>8}  file")
    print("-" * 72)
    for i, (meta, p) in enumerate(zip(demo_rows, demo_zips)):
        print(
            f"{i:3d}  {meta['illuminant']:3}  {meta['wb_cell']:2}  "
            f"{meta.get('trial', '?'):5}  {meta['capture_wb_k']!s:>7} K  {p.name}"
        )
    print(f"\nCell 3 will run all {len(demo_zips)} zips and compare ΔE by wb_cell.")
else:
    n_repo = len(list(DEMO_DIR.glob("Parker*.zip"))) if DEMO_DIR.is_dir() else 0
    print(f"\nRepo has {n_repo}/{len(man.get('demos', []))} Parker WB zips — searching Drive …")
    if IN_COLAB and not Path("/content/drive/MyDrive").is_dir():
        print("Mounting Google Drive …")
        from google.colab import drive
        drive.mount("/content/drive")
    print("\nSearching for Parker *Torch.zip …")
    roots = _search_roots()
    print(f"Search roots ({len(roots)}):")
    for r in roots:
        n = len(_glob_person_zips(r))
        tag = f"  ({n} Parker zips)" if n else ""
        print(f"  {r}{tag}")

    if not roots:
        raise RuntimeError(
            "No search roots. Mount Drive, set EXTRA_ZIP_DIRS, or push Parker zips to demo_zips on GitHub."
        )

    by_cell = defaultdict(list)
    scanned = 0
    unparsed = []
    for root in roots:
        for zp in _glob_person_zips(root):
            scanned += 1
            parsed = _parse_parker_zip(zp.stem)
            if not parsed:
                if _NAME_RE.search(zp.stem):
                    unparsed.append(zp.name)
                continue
            ill, cell, trial = parsed
            by_cell[(ill, cell)].append((trial, zp))

    print(
        f"\nParsed {sum(len(v) for v in by_cell.values())} Parker zips "
        f"→ {len(by_cell)} ring×wb_cell buckets (scanned {scanned} paths)"
    )
    if unparsed:
        print("Could not parse (check filename):", ", ".join(unparsed[:6]))

    preferred_name = {}
    for d in man.get("demos", []):
        preferred_name[(d["illuminant"], d["wb_cell"])] = d["file"]

    demo_rows, demo_zips, missing = [], [], []
    for ill, cell in NEEDED:
        cands = by_cell.get((ill, cell), [])
        if not cands:
            missing.append(f"{ill}-{cell}")
            wbk = CAPTURE_WB_K[ill][cell]
            print(
                f"WARN: no zip for ring {ill} wb_cell {cell} "
                f"(phone WB {wbk} K — look for Parker-{ill}-{cell}*Torch.zip)"
            )
            continue
        pref = preferred_name.get((ill, cell))
        chosen = None
        if pref:
            for trial, zp in cands:
                if zp.name == pref or zp.name.lower().replace(" ", "") == pref.lower():
                    chosen = zp
                    break
        if chosen is None:
            chosen = sorted(cands, key=lambda t: (t[0], t[1].name))[0][1]
        trial_n = next(t for t, zp in cands if zp == chosen)
        demo_rows.append(
            {
                "file": chosen.name,
                "person": WB_SWEEP_PERSON,
                "illuminant": ill,
                "wb_cell": cell,
                "trial": trial_n,
                "capture_wb_k": CAPTURE_WB_K[ill][cell],
                "fitskin_forehead": FITSKIN[ill],
            }
        )
        demo_zips.append(chosen)

    if not demo_zips:
        raise RuntimeError(
            "No Parker WB-sweep zips found.\n"
            "Push zips to GitHub demo_zips, set EXTRA_ZIP_DIRS, or add Shared-with-me shortcut to My Drive."
        )

    candidates = list(demo_zips)
    demo_names_ordered = [r["file"] for r in demo_rows]

    print(f"\nWB sweep person: {WB_SWEEP_PERSON}")
    print(
        f"Loaded {len(demo_zips)}/{len(NEEDED)} cells"
        + (f"  (missing: {', '.join(missing)})" if missing else "")
    )
    if missing:
        print(
            "\nTip: add flat download paths to EXTRA_ZIP_DIRS, or git push Parker zips to demo_zips/."
        )

    print(f"\n{'idx':>3}  {'ring':3}  {'wb':2}  {'trial':5}  {'phone WB':>8}  file")
    print("-" * 72)
    for i, (meta, p) in enumerate(zip(demo_rows, demo_zips)):
        print(
            f"{i:3d}  {meta['illuminant']:3}  {meta['wb_cell']:2}  "
            f"{meta.get('trial', '?'):5}  {meta['capture_wb_k']!s:>7} K  {p.name}"
        )
        print(f"      ↳ {p}")

    print(f"\nCell 3 will run all {len(demo_zips)} zips and compare ΔE by wb_cell.")
