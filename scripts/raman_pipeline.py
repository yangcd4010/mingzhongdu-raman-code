"""Ming Zhongdu Raman workflow with the S2 fresh white fracture comparison (v2.0.2).
Run from any directory: python scripts/raman_pipeline.py --check-reference
"""
from __future__ import annotations
import argparse, csv, hashlib, json, re
from pathlib import Path
import numpy as np
from scipy import sparse
from scipy.signal import find_peaks, savgol_filter
from scipy.sparse.linalg import spsolve
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
SOURCE_DATA = OUT / "source_data"
QA = OUT / "qa"
FIGURE_WIDTH_IN = 183.0 / 25.4
plt.rcParams.update({"font.family":"DejaVu Sans", "font.size":7, "axes.labelsize":7,
    "axes.titlesize":7.5, "xtick.labelsize":6.2, "ytick.labelsize":6.2,
    "axes.linewidth":0.7, "pdf.fonttype":42, "svg.fonttype":"none", "savefig.facecolor":"white"})

COLORS = {
    "S1": "#A33F35",
    "S2": "#D16D43",
    "S4": "#7F4C8D",
    "S2-FW": "#438C8A",
    "calcite": "#387C53",
    "quartz": "#3E6F9E",
    "gypsum": "#7A5A91",
    "hematite": "#A33F35",
    "maghemite": "#D28C3C",
    "magnetite": "#4A4A4A",
    "goethite": "#B69A38",
}
RAMAN_GROUPS = [("1", "S1"), ("2", "S2"), ("3", "S4"), ("4", "S2-FW")]
RAMAN_TARGETS = (225.0, 294.0, 412.0, 613.0, 1006.0)
RAMAN_SCREEN_TARGETS = (225.0, 412.0, 613.0)
RAMAN_HALF_WIDTH = 8.0
RAMAN_MIN_HEIGHT = 0.10
RAMAN_MIN_PROMINENCE = 0.03
RAMAN_ASLS_LAMBDA = 1.0e6
RAMAN_ASLS_P = 0.01
RAMAN_ASLS_ITERATIONS = 12
RAMAN_SG_WINDOW = 5
RAMAN_SG_ORDER = 3
RRUFF_REFERENCE_FILES = {
    'Hematite': (ROOT / "data/rruff" / 'Hematite__R060190__Raman__785______Raman_Data_Processed__50f59fe2564ec1919574cf0848ad.txt', 'R060190', 785, COLORS['hematite']),
    'Maghemite': (ROOT / "data/rruff" / 'Maghemite__R140712__Raman__780______Raman_Data_Processed__57de8043f21e0418fc8f694e0293.txt', 'R140712', 780, COLORS['maghemite']),
    'Magnetite': (ROOT / "data/rruff" / 'Magnetite__R080025__Raman__780______Raman_Data_Processed__2ae63b611cce3c211d64394df2ad.txt', 'R080025', 780, COLORS['magnetite']),
    'Goethite': (ROOT / "data/rruff" / 'Goethite__R050142__Raman__780______Raman_Data_Processed__0060da5ae80240fbe177af594b62.txt', 'R050142', 780, COLORS['goethite']),
}

def panel_label(ax, label, x=-0.08, y=1.04):
    ax.text(x, y, label.rstrip(")")+")", transform=ax.transAxes,
            ha="left", va="bottom", fontsize=9, fontweight="bold", color="#111111")

def export_figure(fig, stem):
    paths=[]
    for ext in ("png", "pdf", "svg"):
        path=OUT / (stem+"."+ext)
        fig.savefig(path, dpi=300, facecolor="white")
        paths.append(path)
    plt.close(fig)
    return paths


def enforce_minimum_font_size(fig: plt.Figure, minimum_pt: float) -> None:
    """Raise only undersized rendered text in a completed figure."""
    for text_object in fig.findobj(match=matplotlib.text.Text):
        if text_object.get_fontsize() < minimum_pt:
            text_object.set_fontsize(minimum_pt)

def read_raman(path: Path) -> tuple[np.ndarray, np.ndarray]:
    xs: list[float] = []
    ys: list[float] = []
    for raw in path.read_text(errors="ignore").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.replace(",", ".").split()
        if len(parts) < 2:
            continue
        try:
            xs.append(float(parts[0]))
            ys.append(float(parts[1]))
        except ValueError:
            continue
    if not xs:
        raise ValueError(f"No Raman data in {path}")
    x = np.asarray(xs)
    y = np.asarray(ys)
    order = np.argsort(x)
    return x[order], y[order]

def natural_key(value: str) -> list[int | str]:
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", value)]

def list_raman_files(raman_root: Path, group: str) -> list[Path]:
    # The archived spectra are distributed among nested acquisition folders.
    # Sort by the spectrum file ID, not by those folders, so display order is
    # genuinely numerical (for example, 3-YDC_9 precedes 3-YDC_10).
    return sorted((raman_root / group).rglob("*.txt"), key=lambda path: natural_key(path.stem))

def baseline_asls(
    intensity: np.ndarray,
    lam: float = RAMAN_ASLS_LAMBDA,
    p: float = RAMAN_ASLS_P,
    iterations: int = RAMAN_ASLS_ITERATIONS,
) -> np.ndarray:
    n = len(intensity)
    difference = sparse.diags(
        [np.ones(n), -2.0 * np.ones(n), np.ones(n)],
        [0, 1, 2], shape=(n - 2, n), format="csc",
    )
    weights = np.ones(n)
    penalty = lam * (difference.T @ difference)
    for _ in range(iterations):
        weight_matrix = sparse.spdiags(weights, 0, n, n)
        baseline = spsolve(weight_matrix + penalty, weights * intensity)
        weights = np.where(intensity > baseline, p, 1.0 - p)
    return np.asarray(baseline, dtype=float)

def process_raman(x: np.ndarray, raw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    baseline = baseline_asls(raw)
    corrected = np.clip(raw - baseline, 0.0, None)
    smoothed = savgol_filter(corrected, RAMAN_SG_WINDOW, RAMAN_SG_ORDER, mode="interp")
    smoothed = np.clip(smoothed, 0.0, None)
    display_window = (x >= 100.0) & (x <= 1800.0)
    scale = float(np.nanmax(smoothed[display_window]))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Raman spectrum has no positive signal in 100–1800 cm⁻¹")
    return baseline, smoothed / scale

def detect_target_peaks(
    x: np.ndarray,
    intensity: np.ndarray,
    half_width: float = RAMAN_HALF_WIDTH,
    min_height: float = RAMAN_MIN_HEIGHT,
    min_prominence: float = RAMAN_MIN_PROMINENCE,
    targets: tuple[float, ...] = RAMAN_TARGETS,
) -> tuple[dict[float, dict[str, float] | None], np.ndarray, dict[str, np.ndarray]]:
    window = (x >= 100.0) & (x <= 1800.0)
    xw = x[window]
    yw = intensity[window]
    step = float(np.median(np.diff(xw)))
    distance_points = max(1, int(round(8.0 / step)))
    peak_indices, properties = find_peaks(
        yw, height=min_height, prominence=min_prominence, distance=distance_points,
    )
    peak_positions = xw[peak_indices]
    matches: dict[float, dict[str, float] | None] = {}
    for target in targets:
        candidate_indices = np.flatnonzero(np.abs(peak_positions - target) <= half_width)
        if not len(candidate_indices):
            matches[target] = None
            continue
        nearest = int(candidate_indices[np.argmin(np.abs(peak_positions[candidate_indices] - target))])
        matches[target] = {
            "position": float(peak_positions[nearest]),
            "height": float(properties["peak_heights"][nearest]),
            "prominence": float(properties["prominences"][nearest]),
        }
    return matches, peak_positions, properties

def analyse_raman_dataset(raman_root: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for group, label in RAMAN_GROUPS:
        files = list_raman_files(raman_root, group)
        for order, path in enumerate(files, start=1):
            x, raw = read_raman(path)
            baseline, processed = process_raman(x, raw)
            matches, peak_positions, properties = detect_target_peaks(x, processed)
            positive = all(matches[target] is not None for target in RAMAN_SCREEN_TARGETS)
            records.append(
                {
                    "group": group,
                    "label": label,
                    "order": order,
                    "path": path,
                    "relative_path": path.relative_to(raman_root).as_posix(),
                    "spectrum_id": path.stem,
                    "x": x,
                    "raw": raw,
                    "baseline": baseline,
                    "processed": processed,
                    "matches": matches,
                    "all_peak_positions": peak_positions,
                    "all_peak_prominences": properties["prominences"],
                    "positive": positive,
                }
            )
    if len(records) != 39:
        raise ValueError(f"Expected 39 Raman spectra, found {len(records)}")
    return records

def format_peak_value(match: dict[str, float] | None, field: str) -> str:
    return "" if match is None else f"{match[field]:.4f}"

def screening_counts(records: list[dict[str, object]]) -> dict[str, tuple[int, int]]:
    result: dict[str, tuple[int, int]] = {}
    for _, label in RAMAN_GROUPS:
        subset = [record for record in records if record["label"] == label]
        result[label] = (sum(bool(record["positive"]) for record in subset), len(subset))
    return result

def write_raman_analysis_files(records: list[dict[str, object]]) -> None:
    screening_path = SOURCE_DATA / "Raman_screening_results_v03.csv"
    fields = ["group", "display_group", "spectrum_order", "spectrum_id", "raw_file", "screen_positive"]
    for target in RAMAN_TARGETS:
        key = f"peak_{int(target)}"
        fields.extend([f"{key}_position_cm-1", f"{key}_height", f"{key}_prominence"])
    fields.extend(["asls_lambda", "asls_p", "asls_iterations", "sg_window", "sg_order", "normalisation"])
    with screening_path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for record in records:
            row: dict[str, object] = {
                "group": record["group"], "display_group": record["label"],
                "spectrum_order": record["order"], "spectrum_id": record["spectrum_id"],
                "raw_file": record["relative_path"], "screen_positive": int(bool(record["positive"])),
                "asls_lambda": RAMAN_ASLS_LAMBDA, "asls_p": RAMAN_ASLS_P,
                "asls_iterations": RAMAN_ASLS_ITERATIONS, "sg_window": RAMAN_SG_WINDOW,
                "sg_order": RAMAN_SG_ORDER, "normalisation": "maximum in 100–1800 cm-1",
            }
            matches = record["matches"]
            assert isinstance(matches, dict)
            for target in RAMAN_TARGETS:
                key = f"peak_{int(target)}"
                match = matches[target]
                row[f"{key}_position_cm-1"] = format_peak_value(match, "position")
                row[f"{key}_height"] = format_peak_value(match, "height")
                row[f"{key}_prominence"] = format_peak_value(match, "prominence")
            writer.writerow(row)

    processed_path = SOURCE_DATA / "Raman_processed_long_v03.csv"
    with processed_path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.writer(stream)
        writer.writerow(["group", "display_group", "spectrum_id", "raw_file", "raman_shift_cm-1", "raw_counts", "asls_baseline", "processed_normalised_intensity"])
        for record in records:
            x = record["x"]
            raw = record["raw"]
            baseline = record["baseline"]
            processed = record["processed"]
            assert isinstance(x, np.ndarray) and isinstance(raw, np.ndarray)
            assert isinstance(baseline, np.ndarray) and isinstance(processed, np.ndarray)
            mask = (x >= 100.0) & (x <= 2500.0)
            for xv, rv, bv, pv in zip(x[mask], raw[mask], baseline[mask], processed[mask]):
                writer.writerow([record["group"], record["label"], record["spectrum_id"], record["relative_path"], f"{xv:.4f}", f"{rv:.6f}", f"{bv:.6f}", f"{pv:.8f}"])

    sensitivity_path = SOURCE_DATA / "Raman_screening_sensitivity_v03.csv"
    with sensitivity_path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.writer(stream)
        writer.writerow(["half_width_cm-1", "min_height", "min_prominence", "display_group", "positive", "total"])
        for half_width in (6.0, 8.0):
            for min_height in (0.08, 0.10, 0.12):
                for _, label in RAMAN_GROUPS:
                    subset = [record for record in records if record["label"] == label]
                    positives = 0
                    for record in subset:
                        matches, _, _ = detect_target_peaks(
                            record["x"], record["processed"], half_width=half_width,
                            min_height=min_height, min_prominence=RAMAN_MIN_PROMINENCE,
                        )
                        positives += int(all(matches[target] is not None for target in RAMAN_SCREEN_TARGETS))
                    writer.writerow([half_width, min_height, RAMAN_MIN_PROMINENCE, label, positives, len(subset)])

    parameter_path = SOURCE_DATA / "Raman_processing_parameters_v03.csv"
    with parameter_path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.writer(stream)
        writer.writerow(["Parameter", "Value", "Role"])
        writer.writerows(
            [
                ["AsLS lambda", f"{RAMAN_ASLS_LAMBDA:.0f}", "baseline smoothness"],
                ["AsLS p", RAMAN_ASLS_P, "asymmetry"],
                ["AsLS iterations", RAMAN_ASLS_ITERATIONS, "baseline iterations"],
                ["Negative-value clipping", "set to zero after AsLS subtraction and again after SG smoothing", "processed copies only; raw spectra unchanged"],
                ["Savitzky-Golay window", RAMAN_SG_WINDOW, "points"],
                ["Savitzky-Golay order", RAMAN_SG_ORDER, "polynomial order"],
                ["Normalisation", "maximum in 100–1800 cm-1", "per spectrum"],
                ["Peak-window half-width", RAMAN_HALF_WIDTH, "cm-1"],
                ["Minimum normalised height", RAMAN_MIN_HEIGHT, "local-peak threshold"],
                ["Minimum prominence", RAMAN_MIN_PROMINENCE, "local-peak threshold"],
                ["Positive rule", "local maxima near 225, 412 and 613 cm-1", "all three required in one processed spectrum"],
            ]
        )

    counts = screening_counts(records)
    expected = {"S1": (7, 7), "S2": (9, 9), "S4": (17, 17), "S2-FW": (0, 6)}
    if counts != expected:
        raise ValueError(f"Unexpected Raman screening counts: {counts}")
    sulfate = sum(
        1 for record in records
        if record["label"] in {"S1", "S2", "S4"} and record["matches"][1006.0] is not None
    )
    if sulfate != 26:
        raise ValueError(f"Expected 26/33 coloured-point spectra with a 1006 cm-1 local maximum, found {sulfate}/33")

def choose_representatives(records: list[dict[str, object]], raman_root: Path) -> dict[str, dict[str, object]]:
    representatives: dict[str, dict[str, object]] = {}
    for _, label in RAMAN_GROUPS:
        subset = [record for record in records if record["label"] == label]
        candidates = [record for record in subset if bool(record["positive"])] if label != "S2-FW" else subset
        matrix: list[list[float]] = []
        for record in candidates:
            x = record["x"]
            processed = record["processed"]
            assert isinstance(x, np.ndarray) and isinstance(processed, np.ndarray)
            matrix.append([
                float(np.nanmax(processed[np.abs(x - target) <= RAMAN_HALF_WIDTH]))
                for target in RAMAN_SCREEN_TARGETS
            ])
        values = np.asarray(matrix, dtype=float)
        target_vector = np.median(values, axis=0)
        selected_index = int(np.argmin(np.sum((values - target_vector) ** 2, axis=1)))
        representatives[label] = candidates[selected_index]

    manifest = SOURCE_DATA / "Raman_representative_raw_selection.csv"
    with manifest.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.writer(stream)
        writer.writerow(["Display group", "Raw spectrum file", "Spectrum ID", "Raw display treatment", "Processed display treatment"])
        for _, label in RAMAN_GROUPS:
            record = representatives[label]
            writer.writerow(
                [
                    label, record["relative_path"], record["spectrum_id"],
                    "as recorded; no smoothing, baseline correction or normalisation",
                    "AsLS lambda 1e6, p 0.01, 12 iterations; zero clipping; Savitzky-Golay 5/3; zero clipping; maximum normalisation over 100–1800 cm-1",
                ]
            )
    return representatives

def rruff_metadata(path: Path) -> dict[str, str]:
    metadata: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8-sig", errors="ignore").splitlines():
        if not raw.startswith("##") or "=" not in raw:
            continue
        key, value = raw[2:].split("=", 1)
        metadata[key.strip()] = value.strip()
    return metadata

def read_rruff_raman(path: Path) -> tuple[np.ndarray, np.ndarray]:
    xs: list[float] = []
    ys: list[float] = []
    for raw in path.read_text(encoding="utf-8-sig", errors="ignore").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "," not in line:
            continue
        left, right = line.split(",", 1)
        try:
            xs.append(float(left.strip()))
            ys.append(float(right.strip()))
        except ValueError:
            continue
    if not xs:
        raise ValueError(f"No RRUFF Raman data in {path}")
    x = np.asarray(xs, dtype=float)
    y = np.asarray(ys, dtype=float)
    order = np.argsort(x)
    return x[order], y[order]

def write_rruff_reference_manifest(references: dict[str, dict[str, object]]) -> None:
    manifest = SOURCE_DATA / "RRUFF_reference_Raman_manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "Phase",
                "RRUFF ID",
                "Excitation (nm)",
                "RRUFF data type",
                "Local file",
                "Sample URL",
                "Download archive",
                "Display treatment in Fig. 5c",
                "225 cm-1 local-peak match",
                "412 cm-1 local-peak match",
                "613 cm-1 local-peak match",
                "Three-window match count",
                "Passes three-window combination",
                "SHA-256",
            ]
        )
        for phase, (path, expected_id, expected_wavelength, _) in RRUFF_REFERENCE_FILES.items():
            if not path.exists():
                raise FileNotFoundError(path)
            metadata = rruff_metadata(path)
            rruff_id = metadata.get("RRUFFID", "")
            wavelength = int(metadata.get("RAMAN WAVELENGTH", "0"))
            if rruff_id != expected_id or wavelength != expected_wavelength:
                raise ValueError(f"RRUFF metadata mismatch for {path.name}")
            matches = references[phase]["matches"]
            assert isinstance(matches, dict)
            matched = [matches[target] is not None for target in RAMAN_SCREEN_TARGETS]
            writer.writerow(
                [
                    phase,
                    rruff_id,
                    wavelength,
                    metadata.get("FILETYPE", ""),
                    path.name,
                    metadata.get("URL", ""),
                    "https://www.rruff.net/zipped_data_files/raman/excellent_unoriented.zip",
                    "RRUFF processed intensity; shifted to zero; independently max-normalised; constant vertical offset only",
                    *[int(value) for value in matched],
                    sum(matched),
                    int(all(matched)),
                    hashlib.sha256(path.read_bytes()).hexdigest(),
                ]
            )

def reference_records() -> dict[str, dict[str, object]]:
    result: dict[str, dict[str, object]] = {}
    for phase, (path, rruff_id, wavelength, colour) in RRUFF_REFERENCE_FILES.items():
        x, intensity = read_rruff_raman(path)
        mask = (x >= 100.0) & (x <= 1800.0)
        x = x[mask]
        intensity = intensity[mask]
        intensity = intensity - np.nanmin(intensity)
        scale = float(np.nanmax(intensity)) or 1.0
        intensity = intensity / scale
        matches, _, _ = detect_target_peaks(x, intensity)
        result[phase] = {
            "x": x, "intensity": intensity, "matches": matches,
            "rruff_id": rruff_id, "wavelength": wavelength, "colour": colour,
        }
    passed = [phase for phase, record in result.items() if all(record["matches"][target] is not None for target in RAMAN_SCREEN_TARGETS)]
    if passed != ["Hematite"]:
        raise ValueError(f"Reference specificity check failed: {passed}")
    return result

def format_counts_tick(value: float) -> str:
    absolute = abs(value)
    if absolute >= 10000:
        return f"{value / 1000:.0f}k"
    if absolute >= 1000:
        return f"{value / 1000:.1f}k"
    return f"{value:.0f}"

def plot_raw_small_multiples(
    fig: plt.Figure,
    spec,
    representatives: dict[str, dict[str, object]],
) -> None:
    raw_grid = gridspec.GridSpecFromSubplotSpec(4, 1, subplot_spec=spec, hspace=0.06)
    order = ["S2", "S4", "S1", "S2-FW"]
    axes: list[plt.Axes] = []
    for index, label in enumerate(order):
        record = representatives[label]
        x = record["x"]
        raw = record["raw"]
        assert isinstance(x, np.ndarray) and isinstance(raw, np.ndarray)
        mask = (x >= 100.0) & (x <= 1800.0)
        xv = x[mask]
        yv = raw[mask]
        ax = fig.add_subplot(raw_grid[index, 0], sharex=axes[0] if axes else None)
        axes.append(ax)
        # Display-only repair: all four recorded spectra share one wavenumber
        # transform. The input arrays and each raw-count y range stay unchanged.
        ax.set_xlim(100, 1800)
        ax.set_xticks([100, 400, 700, 1000, 1300, 1600, 1800])
        ax.margins(x=0)
        ax.plot(xv, yv, color=COLORS[label], lw=0.72)
        for target in RAMAN_SCREEN_TARGETS:
            ax.axvline(target, color=COLORS["hematite"], lw=0.38, alpha=0.34, ls="--")
        low = float(np.nanmin(yv))
        high = float(np.nanmax(yv))
        pad = max(1.0, 0.08 * (high - low))
        ax.set_ylim(low - pad, high + pad)
        ax.set_yticks([low, high], [format_counts_tick(low), format_counts_tick(high)])
        ax.text(0.995, 0.92, f"{label} | {record['spectrum_id']}", transform=ax.transAxes,
                ha="right", va="top", fontsize=5.6, color=COLORS[label], fontweight="bold")
        ax.spines[["top", "right"]].set_visible(False)
        if index < 3:
            ax.tick_params(labelbottom=False)
        else:
            ax.set_xlabel("Raman shift (cm⁻¹)")
        if index == 0:
            panel_label(ax, "a", x=-0.08, y=0.96)
            ax.set_title("Representative as-recorded spectra; raw detector counts", loc="left", pad=4)
        if index == 1:
            ax.set_ylabel("Raw detector counts", labelpad=10)
    fig.canvas.draw()
    import json
    reference_positions = []
    for ax in axes:
        reference_positions.append({
            "xlim": list(ax.get_xlim()),
            "xticks": ax.get_xticks().tolist(),
            "axes_bounds": list(ax.get_position().bounds),
            "reference_x_pixels": {str(v): float(ax.transData.transform((v, 0))[0])
                                   for v in (225, 412, 613, 1200, 1400)},
        })
    for v in (225, 412, 613, 1200, 1400):
        values = [row["reference_x_pixels"][str(v)] for row in reference_positions]
        assert max(values) - min(values) < 1e-8
    (QA / "Figure_5a_shared_wavenumber_transform.json").write_text(
        json.dumps(reference_positions, indent=2), encoding="utf-8")

def plot_processed_representatives(ax: plt.Axes, representatives: dict[str, dict[str, object]]) -> None:
    order = ["S2-FW", "S1", "S4", "S2"]
    top = 0.0
    for index, label in enumerate(order):
        record = representatives[label]
        x = record["x"]
        processed = record["processed"]
        matches = record["matches"]
        assert isinstance(x, np.ndarray) and isinstance(processed, np.ndarray) and isinstance(matches, dict)
        mask = (x >= 100.0) & (x <= 1800.0)
        offset = 1.25 * index
        ax.plot(x[mask], processed[mask] + offset, color=COLORS[label], lw=0.72)
        for target in RAMAN_SCREEN_TARGETS:
            match = matches[target]
            if match is not None:
                ax.plot(match["position"], match["height"] + offset, marker="o", ms=2.8,
                        mfc="white", mec=COLORS[label], mew=0.7, zorder=5)
        ax.text(1788, offset + 0.48, label, color=COLORS[label],
                ha="right", va="bottom", fontsize=5.4, fontweight="bold")
        ax.text(1788, offset + 0.19, str(record["spectrum_id"]), color=COLORS[label],
                ha="right", va="bottom", fontsize=5.4, fontweight="bold")
        top = max(top, offset + float(np.nanmax(processed[mask])))
    for target in RAMAN_SCREEN_TARGETS:
        ax.axvline(target, color=COLORS["hematite"], lw=0.42, alpha=0.45, ls="--")
        ax.text(target, top + 0.08, f"{target:.0f}", color=COLORS["hematite"], fontsize=5.1,
                rotation=90, rotation_mode="anchor", ha="center", va="bottom")
    ax.axvline(1006, color=COLORS["gypsum"], lw=0.42, alpha=0.45, ls=":")
    ax.text(1006, top + 0.08, "1006", color=COLORS["gypsum"], fontsize=5.1,
            rotation=90, rotation_mode="anchor", ha="center", va="bottom")
    ax.set_xlim(100, 1800)
    ax.set_ylim(0, top + 0.45)
    ax.set_yticks([])
    ax.set_xlabel("Raman shift (cm⁻¹)")
    ax.set_ylabel("Processed intensity\n(max-normalised; offsets)")
    ax.set_title("Processed spectra\nAsLS, zero clipping, SG5; max-normalised", loc="left", pad=3)
    ax.spines[["top", "right"]].set_visible(False)

def plot_reference_spectra(ax: plt.Axes, references: dict[str, dict[str, object]]) -> None:
    order = ["Goethite", "Magnetite", "Maghemite", "Hematite"]
    top = 0.0
    for index, phase in enumerate(order):
        record = references[phase]
        x = record["x"]
        intensity = record["intensity"]
        matches = record["matches"]
        assert isinstance(x, np.ndarray) and isinstance(intensity, np.ndarray) and isinstance(matches, dict)
        mask = (x >= 100.0) & (x <= 1800.0)
        offset = 1.30 * index
        ax.plot(x[mask], intensity[mask] + offset, color=record["colour"], lw=0.72)
        matched_count = 0
        for target in RAMAN_SCREEN_TARGETS:
            match = matches[target]
            if match is not None:
                matched_count += 1
                ax.plot(match["position"], match["height"] + offset, marker="o", ms=2.8,
                        mfc="white", mec=record["colour"], mew=0.7, zorder=5)
        ax.text(1290, offset + 0.60, phase,
                color=record["colour"], ha="right", va="bottom", fontsize=5.1,
                fontweight="bold")
        ax.text(1290, offset + 0.34, str(record["rruff_id"]),
                color=record["colour"], ha="right", va="bottom", fontsize=5.1,
                fontweight="bold")
        ax.text(1290, offset + 0.08, f"match {matched_count}/3",
                color=record["colour"], ha="right", va="bottom", fontsize=5.1,
                fontweight="bold")
        top = max(top, offset + float(np.nanmax(intensity[mask])))
    for target in RAMAN_SCREEN_TARGETS:
        ax.axvline(target, color=COLORS["hematite"], lw=0.42, alpha=0.42, ls="--")
        ax.text(target, top + 0.08, f"{target:.0f}", color=COLORS["hematite"], fontsize=5.1,
                rotation=90, rotation_mode="anchor", ha="center", va="bottom")
    ax.set_xlim(100, 1300)
    ax.set_xticks([100, 400, 700, 1000, 1300])
    ax.set_ylim(0, top + 0.45)
    ax.set_yticks([])
    ax.set_xlabel("Raman shift (cm⁻¹)")
    ax.set_title("RRUFF references (780–785 nm)\navailable ranges only; max-normalised", loc="left", pad=3)
    ax.spines[["top", "right", "left"]].set_visible(False)

def make_figure5(records: list[dict[str, object]], raman_root: Path) -> list[Path]:
    representatives = choose_representatives(records, raman_root)
    references = reference_records()
    write_rruff_reference_manifest(references)
    fig = plt.figure(figsize=(FIGURE_WIDTH_IN, 8.75))
    gs = gridspec.GridSpec(
        3, 2, figure=fig, height_ratios=[1.58, 1.30, 0.72], width_ratios=[1.04, 0.96],
        hspace=0.38, wspace=0.28,
    )

    plot_raw_small_multiples(fig, gs[0, :], representatives)

    ax_processed = fig.add_subplot(gs[1, 0])
    plot_processed_representatives(ax_processed, representatives)
    panel_label(ax_processed, "b", x=-0.13)

    ax_refs = fig.add_subplot(gs[1, 1])
    plot_reference_spectra(ax_refs, references)
    panel_label(ax_refs, "c", x=-0.13)

    counts = screening_counts(records)
    labels = [label for _, label in RAMAN_GROUPS]
    positive = [counts[label][0] for label in labels]
    total = [counts[label][1] for label in labels]
    values = [p / t for p, t in zip(positive, total)]
    ax = fig.add_subplot(gs[2, :])
    xpos = np.arange(len(labels))
    ax.bar(xpos, values, color=[COLORS[label] for label in labels], width=0.58)
    ax.set_xticks(xpos, labels)
    ax.set_ylim(0, 1.18)
    ax.set_yticks([0, 0.5, 1.0], ["0", "50%", "100%"])
    ax.set_ylabel("Screen-positive spectra")
    ax.set_title("Operational local-peak screening results", loc="left", pad=3)
    for index, (p, t, value) in enumerate(zip(positive, total, values)):
        ax.text(index, value + 0.05, f"{p}/{t}", ha="center", va="bottom", fontsize=6.5)
    ax.text(
        0.0, -0.36,
        "Positive = local maxima within ±8 cm⁻¹ of 225, 412 and 613 cm⁻¹ after AsLS + SG5 processing;\n"
        "normalised height ≥0.10 and prominence ≥0.03. S2-FW denotes the fresh white fracture of S2.",
        transform=ax.transAxes, ha="left", va="top", fontsize=5.5, wrap=False,
    )
    ax.grid(axis="y", color="#E8E8E8", lw=0.5)
    ax.spines[["top", "right"]].set_visible(False)
    # Match the physical left offset of panel b despite panel d spanning both columns.
    panel_label(ax, "d", x=-0.059)

    # The manuscript embeds this tall figure below its 183 mm source width.
    # A 6.8 pt source floor remains above 5 pt in the narrower Chinese DOCX placement.
    enforce_minimum_font_size(fig, 6.8)
    fig.subplots_adjust(left=0.09, right=0.970, top=0.982, bottom=0.090)
    return export_figure(fig, "Figure_5_Raman_evidence")

def nice_count_scale(value: float) -> float:
    if value <= 0:
        return 1.0
    exponent = np.floor(np.log10(value))
    fraction = value / (10.0 ** exponent)
    nice_fraction = 1.0 if fraction < 1.5 else 2.0 if fraction < 3.5 else 5.0
    return float(nice_fraction * (10.0 ** exponent))

def make_supplementary_figure_s1(records: list[dict[str, object]]) -> list[Path]:
    fig = plt.figure(figsize=(FIGURE_WIDTH_IN, 11.25))
    gs = gridspec.GridSpec(4, 1, figure=fig, height_ratios=[1.25, 1.50, 2.55, 1.20], hspace=0.23)
    axes: list[plt.Axes] = []
    manifest_rows: list[list[str]] = []
    for panel_index, ((_, label), spec) in enumerate(zip(RAMAN_GROUPS, gs)):
        ax = fig.add_subplot(spec)
        axes.append(ax)
        subset = [record for record in records if record["label"] == label]
        spans: list[float] = []
        for record in subset:
            x = record["x"]
            raw = record["raw"]
            assert isinstance(x, np.ndarray) and isinstance(raw, np.ndarray)
            mask = (x >= 100.0) & (x <= 2500.0)
            spans.append(float(np.nanmax(raw[mask]) - np.nanmin(raw[mask])))
        spacing = max(spans) * 1.13
        ax.axvspan(1200, 1400, color="#F4F4F4", zorder=0)
        for trace_index, record in enumerate(subset):
            x = record["x"]
            raw = record["raw"]
            assert isinstance(x, np.ndarray) and isinstance(raw, np.ndarray)
            mask = (x >= 100.0) & (x <= 2500.0)
            xv = x[mask]
            centred = raw[mask] - np.nanmin(raw[mask])
            offset = trace_index * spacing
            ax.plot(xv, centred + offset, color=COLORS[label], lw=0.46, alpha=0.82)
            ax.text(1.004, offset + 0.10 * spacing, str(record["spectrum_id"]),
                    transform=ax.get_yaxis_transform(), ha="left", va="center", fontsize=5.0,
                    color=COLORS[label], clip_on=False)
            manifest_rows.append([str(record["group"]), label, str(record["spectrum_id"]), str(record["relative_path"]), str(record["order"]), "raw counts; per-spectrum minimum shifted and constant vertical offset only; no baseline correction, smoothing or normalisation"])
        scale_length = nice_count_scale(float(np.median(spans)) * 0.30)
        scale_y = 0.13 * spacing
        ax.plot([2440, 2440], [scale_y, scale_y + scale_length], color="#222222", lw=1.0)
        ax.text(2418, scale_y + scale_length / 2.0, f"{format_counts_tick(scale_length)} counts",
                ha="right", va="center", fontsize=5.0, color="#222222")
        ax.set_xlim(100, 2500)
        # Reserve a clear title band above the upper trace so that a tall peak
        # cannot appear as a clipped mark through the group label.
        ax.set_ylim(-0.08 * spacing, (len(subset) - 1) * spacing + max(spans) * 2.00)
        ax.set_yticks([])
        ax.text(0.006, 0.98, f"{chr(97+panel_index)}) {label} (n={len(subset)})", transform=ax.transAxes,
                ha="left", va="top", fontsize=7.0, fontweight="bold", color=COLORS[label],
                bbox={"fc": "white", "ec": "none", "alpha": 0.80, "pad": 0.8})
        ax.set_ylabel("Minimum-shifted\nraw counts + offset")
        ax.grid(axis="x", color="#EEEEEE", lw=0.4)
        ax.spines[["top", "right"]].set_visible(False)
        if panel_index < 3:
            ax.tick_params(labelbottom=False)
    axes[-1].set_xlabel("Raman shift (cm⁻¹)")
    fig.suptitle("Supplementary Figure 1 | All 39 raw Raman spectra (100–2500 cm⁻¹)",
                 fontsize=8.5, fontweight="bold", y=0.994)
    fig.text(0.50, 0.972, "a) S1, red. b) S2, orange. c) S4, purple. d) S2-FW, teal. k = 1000 counts.\nMinimum-shifted and vertically offset only; no baseline correction, smoothing or normalisation. Grey bands mark 1200–1400 cm⁻¹.",
             ha="center", va="top", fontsize=6.0)
    fig.subplots_adjust(left=0.10, right=0.84, top=0.952, bottom=0.052)

    manifest = SOURCE_DATA / "Raman_all_raw_spectra_manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.writer(stream)
        writer.writerow(["Source group", "Display group", "Spectrum ID", "Raw spectrum file", "Display order", "Display treatment"])
        writer.writerows(manifest_rows)
    return export_figure(fig, "Supplementary_Figure_S1_all_raw_Raman")

def compare_reference(reference_dir):
    """Compare computed outputs with the unchanged submission workbook export."""
    report = {"comparisons": {}, "mismatches": []}
    files = ["Raman_screening_results_v03.csv", "Raman_screening_sensitivity_v03.csv", "Raman_processed_long_v03.csv"]
    for name in files:
        with (SOURCE_DATA/name).open(encoding="utf-8-sig", newline="") as stream:
            actual = list(csv.DictReader(stream))
        with (reference_dir/name).open(encoding="utf-8-sig", newline="") as stream:
            expected = list(csv.DictReader(stream))
        if len(actual) != len(expected):
            raise AssertionError(f"{name}: row counts differ")
        errors=[]
        max_error={}
        for idx,(a,b) in enumerate(zip(actual,expected),start=2):
            for key,observed in a.items():
                wanted=b.get(key)
                if (observed or "")== (wanted or ""):
                    continue
                try:
                    delta=abs(float(observed)-float(wanted))
                    max_error[key]=max(max_error.get(key,0),delta)
                    tolerance=5.1e-9 if key=="processed_normalised_intensity" else 5.1e-7 if key in ("raw_counts","asls_baseline") else 5.1e-5 if "position" in key or key.startswith("peak_") or key=="raman_shift_cm-1" else 1e-9
                    if delta <= tolerance:
                        continue
                except (ValueError,TypeError):
                    pass
                errors.append({"row":idx,"field":key,"expected":wanted,"actual":observed})
        report["comparisons"][name]={"rows":len(actual),"pass":not errors,"max_absolute_errors":max_error}
        report["mismatches"].extend(errors[:20])
    (OUT/"verification.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    if report["mismatches"]:
        raise AssertionError(json.dumps(report["mismatches"],indent=2))
    return report

def main():
    global OUT,SOURCE_DATA,QA
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir",type=Path,default=ROOT/"data/raw")
    parser.add_argument("--output-dir",type=Path,default=ROOT/"outputs")
    parser.add_argument("--reference-dir",type=Path,default=ROOT/"reference_outputs")
    parser.add_argument("--check-reference",action="store_true")
    parser.add_argument("--skip-figures",action="store_true")
    args=parser.parse_args()
    OUT=args.output_dir.resolve(); SOURCE_DATA=OUT/"source_data"
    QA=OUT/"qa"
    QA.mkdir(parents=True,exist_ok=True)
    SOURCE_DATA.mkdir(parents=True,exist_ok=True)
    records=analyse_raman_dataset(args.input_dir)
    write_raman_analysis_files(records)
    counts=screening_counts(records)
    if counts!={"S1":(7,7),"S2":(9,9),"S4":(17,17),"S2-FW":(0,6)}:
        raise AssertionError(f"Unexpected screening counts: {counts}")
    sulfate=sum(r["label"]!="S2-FW" and r["matches"][1006.] is not None for r in records)
    assert sulfate==26, sulfate
    reps=choose_representatives(records,args.input_dir)
    assert {k:r["spectrum_id"] for k,r in reps.items()}=={"S1":"1-YDC_16","S2":"2-YDC_9","S4":"3-YDC_9","S2-FW":"4-YDC"}
    refs=reference_records(); write_rruff_reference_manifest(refs)
    reference_matches={k:sum(r["matches"][t] is not None for t in RAMAN_SCREEN_TARGETS) for k,r in refs.items()}
    assert reference_matches=={"Hematite":3,"Maghemite":2,"Magnetite":0,"Goethite":0}
    if args.check_reference:
        compare_reference(args.reference_dir)
    if not args.skip_figures:
        make_figure5(records,args.input_dir)
        make_supplementary_figure_s1(records)
    summary={"counts":counts,"sulfate_positive_coloured_spectra":sulfate,"rruff_window_matches":reference_matches,"representatives":{k:r["spectrum_id"] for k,r in reps.items()}}
    (OUT/"summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    print(json.dumps(summary,indent=2))
    print("Reference verification:","passed" if args.check_reference else "not requested")

if __name__=="__main__":
    main()
