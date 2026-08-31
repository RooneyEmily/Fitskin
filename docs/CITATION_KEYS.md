# Citation key map (Overleaf `?` fix)

The `?` marks appear when `\cite{key}` does not match any entry in your `.bib` file.

## Step 1 — Rename your main bib file

In `skin_tone_calibration_overleaf.tex`, line near the end:

```latex
\bibliography{references,references_supplement}
```

Change `references` to **your actual `.bib` filename** (without extension), e.g.:

```latex
\bibliography{main,references_supplement}
```

## Step 2 — Upload `references_supplement.bib`

Upload `docs/references_supplement.bib` to Overleaf. It adds **4 keys missing** from your library:

| Key | What it is |
|-----|------------|
| `wu1999` | Classic camera RGB→XYZ matrix calibration |
| `issa2025` | ISSA median skin reflectance priors |
| `zhou2025scr` | SCR-AWB (Technologies 2025) |
| `fitskin2026pansor` | Internal Pansor bag CAT02 note |

Alternatively, paste those four entries into your main `.bib` and use `\bibliography{main}` only.

## Step 3 — Compile order

```
pdflatex skin_tone_calibration_overleaf
bibtex skin_tone_calibration_overleaf
pdflatex skin_tone_calibration_overleaf
pdflatex skin_tone_calibration_overleaf
```

## Keys we changed in the `.tex` (old → your bib)

| Old (broken) | Your `.bib` key |
|--------------|-----------------|
| `lu2006` | `lu2006practical` |
| `maralan2023flash` | `maralan2023computational` |
| `upadhyay2025` | `upadhyay2025low` |
| `harville2005` | `harville2005consistent` |
| `hayanchoi2017` | `choi2017performance` |
| `cook2025` | `cook2025colorimetric` |
| `petschnigg2004` | `petschnigg2004digital` |
| `hui2016` | `hui2016white` |
| `hui2018` | `hui2018illuminant` |
| `xie2023` | `xie2023camera` |
| `kang2024` | `kang2026graph` |
| `liu2025cra` | `liu2026illuminant` |
| `mcc24` | `BabelColorMCC` |
| `pansor2026bag` | `fitskin2026pansor` (supplement) |

## Keys already correct in your bib

`romero2006spectral`, `gijsenij2011`, `dicarlo2001illuminating`, `flashambient`, `hubel1994comparison`, `fitzpatrick1988`, `cheng2024monastic`, `sharma2005ciede2000`, `BabelColorMCC`, `cie2004cat`, `Bradford1984`, `evangelidis2008parametric`, `wakholi2026systematic`, `wang2021optimized`, `wang2025perform`

## Duplicate in your bib (harmless but tidy)

`liu2026illuminant` and `maralan2023computational` and `kang2026graph` each appear **twice** — delete one copy in Overleaf to avoid BibTeX warnings.

## Huber entry

Your `@article{ef4f3c67-619e-365c-8056-8c20d2b002d5,...}` is valid but we do not cite it in this draft. To use it, add `\cite{ef4f3c67-619e-365c-8056-8c20d2b002d5}` or rename the key to `huber1964` for readability.
