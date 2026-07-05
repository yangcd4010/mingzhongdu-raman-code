from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.signal import savgol_filter
from scipy.sparse.linalg import spsolve


GROUP_SAMPLE_IDS = {
    "1": "S1",
    "2": "S2",
    "3": "S4",
    "4": "S2-FW",
}

GROUP_LABELS = {
    "1": "S1, red surface",
    "2": "S2, pink-red surface",
    "3": "S4, coloured surface points",
    "4": "S2-FW, fresh white fracture",
}

GROUP_SHORT_LABELS = {
    "1": "S1",
    "2": "S2",
    "3": "S4",
    "4": "S2-FW",
}

COLORS = {
    "1": "#9F3D35",
    "2": "#C86943",
    "3": "#7E4C8A",
    "4": "#4E8C8A",
    "grey": "#6E6E6E",
}

DIAGNOSTIC_BANDS = [
    ("Hem_225", 225.0, "hematite"),
    ("Hem_294", 294.0, "hematite_support"),
    ("Hem_412", 412.0, "hematite"),
    ("Hem_613", 613.0, "hematite"),
    ("Gyp_1006", 1006.0, "gypsum"),
    ("Gyp_1134", 1134.0, "gypsum"),
    ("Cal_279", 279.0, "calcite"),
    ("Cal_711", 711.0, "calcite"),
    ("Cal_1085", 1085.0, "calcite"),
    ("C_1350", 1350.0, "carbon_D"),
    ("C_1580", 1580.0, "carbon_G"),
]

HEMATITE_CORE_BANDS = [("Hem_225", 225.0), ("Hem_412", 412.0), ("Hem_613", 613.0)]
HEMATITE_SUPPORT_BANDS = [("Hem_294", 294.0)]
PHASE_SCORE_BANDS = {
    "Hematite": ["Hem_225", "Hem_412", "Hem_613"],
    "Gypsum": ["Gyp_1006"],
    "Calcite": ["Cal_711", "Cal_1085"],
}
DEFAULT_HALF_WIDTH = 8.0
DEFAULT_HEIGHT_THRESHOLD = 0.10


def group_sort_key(value: str) -> tuple[int, str]:
    return (0, f"{int(value):04d}") if str(value).isdigit() else (1, str(value))


def baseline_asls(y: np.ndarray, lam: float = 1e6, p: float = 0.01, niter: int = 12) -> np.ndarray:
    """Asymmetric least-squares baseline correction."""
    y = np.asarray(y, dtype=float)
    length = len(y)
    diff = sparse.diags([1, -2, 1], [0, 1, 2], shape=(length - 2, length), format="csc", dtype=float)
    weights = np.ones(length)
    for _ in range(niter):
        w_mat = sparse.spdiags(weights, 0, length, length)
        z_mat = w_mat + lam * (diff.T @ diff)
        baseline = spsolve(z_mat, weights * y)
        weights = p * (y > baseline) + (1 - p) * (y < baseline)
    return baseline


def preprocess_spectrum(x: np.ndarray, raw: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    baseline = baseline_asls(raw)
    corrected = np.clip(np.asarray(raw, dtype=float) - baseline, 0, None)
    window = 15 if len(corrected) >= 15 else max(5, (len(corrected) // 2) * 2 + 1)
    if window % 2 == 0:
        window += 1
    if window < len(corrected):
        corrected = savgol_filter(corrected, window, 3)
    corrected = np.clip(corrected, 0, None)
    norm_mask = (x >= 120) & (x <= 1800)
    scale = float(np.percentile(corrected[norm_mask], 99.5))
    if scale <= 0:
        scale = float(corrected[norm_mask].max()) or 1.0
    return baseline, corrected, corrected / scale


def load_and_reprocess(input_csv: Path) -> pd.DataFrame:
    input_df = pd.read_csv(input_csv)
    required = {"group", "spectrum", "file", "raman_shift_cm-1", "raw_intensity"}
    missing = required.difference(input_df.columns)
    if missing:
        raise ValueError(f"Input CSV is missing required columns: {sorted(missing)}")

    rows: list[pd.DataFrame] = []
    for (_, _, _), spectrum_df in input_df.groupby(["group", "spectrum", "file"], sort=False):
        spectrum_df = spectrum_df.sort_values("raman_shift_cm-1").copy()
        x = spectrum_df["raman_shift_cm-1"].to_numpy(dtype=float)
        raw = spectrum_df["raw_intensity"].to_numpy(dtype=float)
        baseline, corrected, normalized = preprocess_spectrum(x, raw)
        spectrum_df["baseline"] = baseline
        spectrum_df["baseline_corrected"] = corrected
        spectrum_df["normalized_intensity"] = normalized
        rows.append(spectrum_df)
    return pd.concat(rows, ignore_index=True)


def band_height(x: np.ndarray, y: np.ndarray, center: float, half_width: float = DEFAULT_HALF_WIDTH) -> tuple[float, float]:
    mask = (x >= center - half_width) & (x <= center + half_width)
    if not mask.any():
        return math.nan, math.nan
    local_x = x[mask]
    local_y = y[mask]
    idx = int(np.argmax(local_y))
    return float(local_x[idx]), float(local_y[idx])


def make_group_mean(processed: pd.DataFrame) -> pd.DataFrame:
    x_values = sorted(processed["raman_shift_cm-1"].unique())
    groups = sorted(processed["group"].astype(str).unique(), key=group_sort_key)
    out = pd.DataFrame({"raman_shift_cm-1": x_values})
    for group in groups:
        pivot = (
            processed[processed["group"].astype(str) == group]
            .pivot_table(index="raman_shift_cm-1", columns="spectrum", values="normalized_intensity")
            .reindex(x_values)
        )
        out[f"group_{group}_mean"] = pivot.mean(axis=1).to_numpy()
        out[f"group_{group}_sd"] = pivot.std(axis=1, ddof=1).fillna(0).to_numpy()
        out[f"group_{group}_n"] = pivot.count(axis=1).to_numpy()
    return out


def make_diagnostic_bands(group_mean: pd.DataFrame) -> pd.DataFrame:
    rows = []
    groups = sorted(
        {col[len("group_") : -len("_mean")] for col in group_mean.columns if col.startswith("group_") and col.endswith("_mean")},
        key=group_sort_key,
    )
    x = group_mean["raman_shift_cm-1"].to_numpy(dtype=float)
    for group in groups:
        y = group_mean[f"group_{group}_mean"].to_numpy(dtype=float)
        n = int(group_mean[f"group_{group}_n"].max())
        for band_name, center, phase in DIAGNOSTIC_BANDS:
            observed, height = band_height(x, y, center)
            rows.append(
                {
                    "group": group,
                    "group_label": GROUP_LABELS.get(group, group),
                    "band": band_name,
                    "phase": phase,
                    "target_cm-1": center,
                    "observed_cm-1": observed,
                    "normalized_height": height,
                    "n_spectra": n,
                }
            )
    return pd.DataFrame(rows)


def make_group_scores(diagnostic: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for group, group_df in diagnostic.groupby("group", sort=False):
        group = str(group)
        for phase, bands in PHASE_SCORE_BANDS.items():
            selected = group_df[group_df["band"].isin(bands)]
            rows.append(
                {
                    "group": group,
                    "group_label": GROUP_LABELS.get(group, group),
                    "phase": phase,
                    "bands_used": "+".join(bands),
                    "score_mean_normalized_height": float(selected["normalized_height"].mean()),
                }
            )
    return pd.DataFrame(rows)


def make_single_spectrum_screen(processed: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (group, spectrum, file_name), spectrum_df in processed.groupby(["group", "spectrum", "file"], sort=False):
        x = spectrum_df["raman_shift_cm-1"].to_numpy(dtype=float)
        y = spectrum_df["normalized_intensity"].to_numpy(dtype=float)
        group = str(group)
        row = {
            "group": group,
            "sample_id": GROUP_SAMPLE_IDS.get(group, group),
            "group_label": GROUP_LABELS.get(group, group),
            "spectrum": spectrum,
            "file": file_name,
            "criterion": "bands near 225, 412 and 613 cm-1 all >=0.10 within +/-8 cm-1",
        }
        present = 0
        core_names = {name for name, _ in HEMATITE_CORE_BANDS}
        for band_name, center in HEMATITE_CORE_BANDS + HEMATITE_SUPPORT_BANDS:
            observed, height = band_height(x, y, center)
            positive = bool(height >= DEFAULT_HEIGHT_THRESHOLD)
            row[f"{band_name}_observed_cm-1"] = observed
            row[f"{band_name}_normalized_height"] = height
            row[f"{band_name}_above_threshold"] = positive
            if band_name in core_names and positive:
                present += 1
        row["hematite_screening_bands_present"] = present
        row["hematite_screening_positive"] = present == len(HEMATITE_CORE_BANDS)
        rows.append(row)
    return pd.DataFrame(rows)


def make_positive_summary(single: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for group, group_df in single.groupby("group", sort=False):
        total = int(len(group_df))
        positive = int(group_df["hematite_screening_positive"].sum())
        rows.append(
            {
                "group": str(group),
                "sample_id": GROUP_SAMPLE_IDS.get(str(group), str(group)),
                "group_label": GROUP_LABELS.get(str(group), str(group)),
                "hematite_screening_positive_spectra": positive,
                "total_spectra": total,
                "positive_fraction": positive / total if total else math.nan,
                "criterion": "225, 412 and 613 cm-1 co-occur; +/-8 cm-1; normalized height >=0.10",
                "note": "294 cm-1 retained as supporting evidence only.",
            }
        )
    return pd.DataFrame(rows).sort_values("group", key=lambda s: s.map(group_sort_key))


def make_sensitivity(processed: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for half_width in [6.0, 8.0, 10.0]:
        for threshold in [0.08, 0.10, 0.12]:
            for group, group_df in processed.groupby("group", sort=False):
                positives = 0
                total = 0
                for _, spectrum_df in group_df.groupby("spectrum", sort=False):
                    x = spectrum_df["raman_shift_cm-1"].to_numpy(dtype=float)
                    y = spectrum_df["normalized_intensity"].to_numpy(dtype=float)
                    present = 0
                    for _, center in HEMATITE_CORE_BANDS:
                        _, height = band_height(x, y, center, half_width=half_width)
                        if height >= threshold:
                            present += 1
                    positives += int(present == len(HEMATITE_CORE_BANDS))
                    total += 1
                group = str(group)
                rows.append(
                    {
                        "half_width_cm-1": half_width,
                        "threshold": threshold,
                        "group": group,
                        "sample_id": GROUP_SAMPLE_IDS.get(group, group),
                        "group_label": GROUP_LABELS.get(group, group),
                        "positive_spectra": positives,
                        "total_spectra": total,
                        "positive_fraction": positives / total if total else math.nan,
                    }
                )
    return pd.DataFrame(rows)


def representative_spectrum(processed: pd.DataFrame, group: str, group_mean: pd.DataFrame) -> pd.DataFrame:
    selected = processed[processed["group"].astype(str) == group]
    x_mean = group_mean["raman_shift_cm-1"].to_numpy(dtype=float)
    y_mean = group_mean[f"group_{group}_mean"].to_numpy(dtype=float)
    mask = (x_mean >= 120) & (x_mean <= 1800)
    best_name = None
    best_score = math.inf
    for spectrum, spectrum_df in selected.groupby("spectrum", sort=False):
        y = spectrum_df.sort_values("raman_shift_cm-1")["normalized_intensity"].to_numpy(dtype=float)
        score = float(np.sqrt(np.mean((y[mask] - y_mean[mask]) ** 2)))
        if score < best_score:
            best_name = spectrum
            best_score = score
    return selected[selected["spectrum"] == best_name].sort_values("raman_shift_cm-1")


def make_summary_figure(processed: pd.DataFrame, group_mean: pd.DataFrame, summary: pd.DataFrame, out_dir: Path) -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Arial", "Liberation Sans"],
            "font.size": 8,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )
    groups = sorted(processed["group"].astype(str).unique(), key=group_sort_key)
    fig, axes = plt.subplots(2, 1, figsize=(7.0, 5.2), constrained_layout=True)

    ax = axes[0]
    offsets = np.arange(len(groups))[::-1] * 1.2
    for offset, group in zip(offsets, groups):
        spec = representative_spectrum(processed, group, group_mean)
        x = spec["raman_shift_cm-1"].to_numpy(dtype=float)
        y = spec["normalized_intensity"].to_numpy(dtype=float)
        mask = (x >= 120) & (x <= 750)
        color = COLORS.get(group, COLORS["grey"])
        ax.plot(x[mask], y[mask] + offset, color=color, lw=1.2)
        ax.text(760, offset + np.nanmedian(y[mask]), GROUP_SHORT_LABELS.get(group, group), color=color, va="center")
    for center, label, color in [(225, "225", "#9F3D35"), (294, "294", "#9F3D35"), (412, "412", "#9F3D35"), (613, "613", "#9F3D35"), (711, "Cal", "#2F7F7B")]:
        ax.axvline(center, color=color, lw=0.7, ls="--", alpha=0.75)
        ax.text(center, ax.get_ylim()[1] * 0.95, label, rotation=90, ha="center", va="top", color=color, fontsize=7)
    ax.set_xlim(120, 790)
    ax.set_yticks([])
    ax.set_xlabel("Raman shift (cm-1)")
    ax.set_ylabel("Representative intensity (offset)")
    ax.set_title("Representative low-wavenumber spectra")

    ax = axes[1]
    summary = summary.sort_values("group", key=lambda s: s.map(group_sort_key))
    x = np.arange(len(summary))
    values = summary["positive_fraction"].to_numpy(dtype=float)
    colors = [COLORS.get(str(g), COLORS["grey"]) for g in summary["group"]]
    ax.bar(x, values, color=colors, width=0.62)
    ax.set_xticks(x)
    ax.set_xticklabels(summary["sample_id"])
    ax.set_ylim(0, 1.12)
    ax.set_yticks([0, 0.5, 1.0])
    ax.set_yticklabels(["0", "50%", "100%"])
    ax.set_ylabel("Screen-positive spectra")
    ax.set_title("Operational hematite screening")
    for i, row in enumerate(summary.to_dict("records")):
        ax.text(i, min(values[i] + 0.05, 1.06), f"{int(row['hematite_screening_positive_spectra'])}/{int(row['total_spectra'])}", ha="center", va="bottom")
    ax.grid(axis="y", color="#EEEEEE", lw=0.5)

    fig.savefig(out_dir / "raman_summary_figure.png", dpi=300)
    fig.savefig(out_dir / "raman_summary_figure.svg")
    plt.close(fig)


def compare_reference(output_dir: Path, reference_dir: Path) -> None:
    expected_summary = pd.read_csv(reference_dir / "raman_hematite_positive_summary.csv")
    observed_summary = pd.read_csv(output_dir / "raman_hematite_positive_summary.csv")
    expected_counts = dict(zip(expected_summary["sample_id"], zip(expected_summary["hematite_screening_positive_spectra"], expected_summary["total_spectra"])))
    observed_counts = dict(zip(observed_summary["sample_id"], zip(observed_summary["hematite_screening_positive_spectra"], observed_summary["total_spectra"])))
    if expected_counts != observed_counts:
        raise AssertionError(f"Screening counts differ from reference: expected={expected_counts}, observed={observed_counts}")

    expected_scores = pd.read_csv(reference_dir / "raman_diagnostic_band_group_scores.csv")
    observed_scores = pd.read_csv(output_dir / "raman_diagnostic_band_group_scores.csv")
    pd.testing.assert_frame_equal(expected_scores, observed_scores, check_exact=False, rtol=1e-6, atol=1e-9)


def main() -> None:
    parser = argparse.ArgumentParser(description="Reproduce Ming Zhongdu Raman preprocessing and screening outputs.")
    parser.add_argument("--input", type=Path, default=Path("data/raman_processed_long.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--reference-dir", type=Path, default=Path("reference_outputs"))
    parser.add_argument("--check-reference", action="store_true")
    parser.add_argument("--skip-figures", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    processed = load_and_reprocess(args.input)
    group_mean = make_group_mean(processed)
    diagnostic = make_diagnostic_bands(group_mean)
    group_scores = make_group_scores(diagnostic)
    single = make_single_spectrum_screen(processed)
    summary = make_positive_summary(single)
    sensitivity = make_sensitivity(processed)

    processed.to_csv(args.output_dir / "raman_processed_long.csv", index=False)
    group_mean.to_csv(args.output_dir / "raman_group_mean.csv", index=False)
    diagnostic.to_csv(args.output_dir / "raman_diagnostic_bands.csv", index=False)
    group_scores.to_csv(args.output_dir / "raman_diagnostic_band_group_scores.csv", index=False)
    single.to_csv(args.output_dir / "raman_single_spectrum_hematite_screen.csv", index=False)
    summary.to_csv(args.output_dir / "raman_hematite_positive_summary.csv", index=False)
    sensitivity.to_csv(args.output_dir / "raman_operational_criterion_sensitivity.csv", index=False)

    if not args.skip_figures:
        make_summary_figure(processed, group_mean, summary, args.output_dir)
    if args.check_reference:
        compare_reference(args.output_dir, args.reference_dir)

    counts = ", ".join(
        f"{row.sample_id}={int(row.hematite_screening_positive_spectra)}/{int(row.total_spectra)}"
        for row in summary.itertuples(index=False)
    )
    print(f"Raman screening summary: {counts}")
    print(f"Outputs written to: {args.output_dir}")


if __name__ == "__main__":
    main()
