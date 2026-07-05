# Ming Zhongdu V35 Raman Processing Code

This repository contains the custom scripts and tabular input needed to reproduce the Raman preprocessing, diagnostic-band extraction, threshold-sensitivity checks and summary figure generation used for the Ming Zhongdu Wumen Gate pink lime plaster manuscript.

## Contents

- `scripts/raman_pipeline.py` - reruns Raman baseline correction, Savitzky-Golay smoothing, percentile normalisation, diagnostic-band extraction, operational hematite screening and sensitivity checks.
- `scripts/fe_map_display.py` - generic display-only helper for SEM-EDS Fe map visualisation. Raw elemental-map image data are not included.
- `data/raman_processed_long.csv` - long-format Raman spectral table containing raw intensities and the archived processed columns used in the manuscript workflow.
- `reference_outputs/` - reference CSV outputs generated from the original manuscript workflow.
- `outputs/` - default destination for regenerated tables and figures.

## Reproduce Raman Outputs

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Run the pipeline:

```bash
python scripts/raman_pipeline.py --check-reference
```

The script writes regenerated outputs to `outputs/`:

- `raman_processed_long.csv`
- `raman_group_mean.csv`
- `raman_diagnostic_bands.csv`
- `raman_diagnostic_band_group_scores.csv`
- `raman_single_spectrum_hematite_screen.csv`
- `raman_hematite_positive_summary.csv`
- `raman_operational_criterion_sensitivity.csv`
- `raman_summary_figure.png`
- `raman_summary_figure.svg`

## Operational Screening Definition

A spectrum is classified as operationally hematite-screen-positive only when the three core bands near 225, 412 and 613 cm-1 all show local maxima within +/-8 cm-1 of the target positions and each has normalised peak height >=0.10 in the same smoothed, baseline-corrected and percentile-normalised spectrum. The 294 cm-1 band is retained as supporting evidence only and is not used for the positive call.

## Data Scope

This repository is intended for code availability and reproducibility of the Raman screening workflow. Heritage-sensitive site coordinates, sampling-location details, field photographs and conservation records are not included. Raw SEM-EDS image/map data are also not included because they are controlled by the relevant data custodian.

## Citation

If using this code, cite the associated manuscript and this repository URL.
