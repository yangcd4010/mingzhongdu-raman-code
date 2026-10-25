# Ming Zhongdu Raman processing and screening code

Version 2.0.2 processes and screens Raman spectra from historic pink-to-red lime finishes at Wumen Gate, Ming Zhongdu. It provides the point-based phase evidence for comparison of coloured surfaces with the fresh white fracture of S2. The code computes per-spectrum peak matches, group screening counts, threshold sensitivity checks, the Raman panels of Figure 5, and Supplementary Figure 1.

## Run

Python 3.12 was used for validation. Install the pinned dependencies in a virtual environment:

```sh
python -m venv .venv
# Activate the environment using the command appropriate for your shell.
python -m pip install -r requirements.txt
python scripts/raman_pipeline.py --check-reference
```

The default input and output paths are relative to the repository, so the script can be called from another working directory. For numerical verification without plots:

```sh
python scripts/raman_pipeline.py --check-reference --skip-figures
```

`--input-dir`, `--output-dir` and `--reference-dir` accept alternative directories. A failed comparison exits with an error. Results and verification reports are written under `outputs/`.

## Input and reference files

- `data/raw/`: 39 numerical two-column spectra (Raman shift in cm⁻¹ and original detector counts), retaining every acquisition point at round-trip floating-point precision. Instrument headers and workstation metadata are excluded. Each file contains 976 points, spanning 102.984–2498.39 cm⁻¹.
- `data/raw_manifest.json`: spectrum identifiers, source-group paths, point counts, ranges and file checksums.
- `data/rruff/`: the four processed RRUFF reference spectra used in Figure 5c, with their original source metadata retained.
- `reference_outputs/`: numerical tables exported from Supplementary Data 1, with the fourth group labelled S2-FW. These are comparison targets, not computed replacements. The `_v03` filenames identify the numerical tables; the software version is 2.0.2.

Source folders 1, 2 and 3 correspond to coloured points from S1, S2 and S4. Source folder 4 (4-YDC through 4-YDC_5) contains the six spectra from the fresh white fracture of S2 (S2-FW), as identified in the original manuscript and confirmed by the author. These six spectra provide a within-sample comparison with the nine coloured-surface spectra from S2. Instrument filenames are retained unchanged; Raman and SEM-EDS measurements are not assumed to be fully co-located.

## Processing and screening

1. Apply asymmetric least-squares baseline correction to each full recorded spectrum: lambda = 1,000,000; p = 0.01; 12 iterations.
2. Clip negative baseline-subtracted values to zero; apply a 5-point, third-order Savitzky–Golay filter (`mode="interp"`); clip negative values again.
3. Divide by the maximum processed intensity over 100–1800 cm⁻¹.
4. Detect local peaks over 100–1800 cm⁻¹ with height ≥0.10 and prominence ≥0.03. The minimum peak distance is `max(1, round(8 / median_grid_step))` acquisition points. Within each ±8 cm⁻¹ target window, select the qualifying local peak nearest the target.
5. Call a spectrum screen-positive only if the 225, 412 and 613 cm⁻¹ targets all match. The 294 cm⁻¹ band is supporting evidence. The 1006 cm⁻¹ match is reported separately.

Sensitivity checks use half-widths of 6 and 8 cm⁻¹ and height thresholds of 0.08, 0.10 and 0.12, with prominence fixed at 0.03. This produces 24 group-level rows (six parameter combinations × four groups).

Figure 5a uses unmodified recorded counts. Supplementary Figure 1 uses only a per-spectrum minimum shift and constant vertical offsets. RRUFF curves are minimum-shifted and independently maximum-normalised over their available ranges. They are not processed as new experimental spectra.

## Verified results

| Group | Screen-positive / total |
|---|---:|
| S1 | 7 / 7 |
| S2 | 9 / 9 |
| S4 | 17 / 17 |
| S2-FW | 0 / 6 |

The 1006 cm⁻¹ match occurs in 26 of the 33 coloured-point spectra. The representative spectra are `1-YDC_16`, `2-YDC_9`, `3-YDC_9` and `4-YDC`. RRUFF reference window matches are hematite 3/3, maghemite 2/3, magnetite 0/3 and goethite 0/3.

`--check-reference` compares all fields in the 39-row screening table, the 24-row sensitivity table and the 38,064-row processed table against the manuscript workbook export. Numerical tolerances respect the stored decimal precision. It also checks counts, representative identities and reference-spectrum matches. See `outputs/verification.json` and `outputs/summary.json` after running.

The figures are regenerated from the same numerical records. Font metrics and rendering can vary across operating systems; pixel-identical figure files are not the numerical verification criterion.

## Archive and citation

Version 2.0.2 is available as the [GitHub release](https://github.com/yangcd4010/mingzhongdu-raman-code/releases/tag/v2.0.2) and archived on [Zenodo](https://doi.org/10.5281/zenodo.22802683). Cite the specific software version used:

Yang, C., X. Zhou, and Z. Wang. 2026. *Ming Zhongdu Raman Processing and Screening Code*. V. 2.0.2. Zenodo. https://doi.org/10.5281/zenodo.22802683.

The [concept DOI](https://doi.org/10.5281/zenodo.21224832) provides access to all archived versions. The version DOI above fixes the numerical code and data used for this study.

## RRUFF sources

The reference spectra are third-party data from RRUFF: hematite R060190 (785 nm), maghemite R140712 (780 nm), magnetite R080025 (780 nm), and goethite R050142 (780 nm). Their sample pages are linked in the generated `RRUFF_reference_Raman_manifest.csv`. Original source: https://www.rruff.net/zipped_data_files/raman/excellent_unoriented.zip. Preserve the supplied source metadata and attribution when reusing these files.

Lafuente, B., Downs, R. T., Yang, H. & Stone, N. The power of databases: the RRUFF project. In *Highlights in Mineralogical Crystallography* (eds Armbruster, T. & Danisi, R. M.) 1–30 (De Gruyter, 2015).
