"""Class-level spectral signatures from a reproducible sample of EuroSAT.

For every class, 50 images are drawn with a fixed seed. Each image is reduced
to one mean value per band, and those means are averaged per class. Three
figures are saved to ``results/figures/eda/``.

Usage:
    python -m src.analysis.spectral_signatures
"""

import matplotlib.pyplot as plt
import numpy as np
import rasterio

from src.analysis.plot_style import (
    BLUE,
    CONTEXT_GRAY,
    DIVERGING_BLUE_RED,
    PAGE_WIDTH,
    apply_style,
    save_figure,
)
from src.config import BAND_NAMES, DATA_DIR, FIGURES_DIR, RANDOM_STATE

EDA_FIGURES_DIR = FIGURES_DIR / "eda"
SAMPLES_PER_CLASS = 50

# Central wavelength of each Sentinel-2 band in nanometres. The TIFF files
# store B8A last, so the bands are re-ordered by wavelength before plotting.
CENTRAL_WAVELENGTH_NM = {
    "B01": 443, "B02": 490, "B03": 560, "B04": 665, "B05": 705,
    "B06": 740, "B07": 783, "B08": 842, "B8A": 865, "B09": 945,
    "B10": 1375, "B11": 1610, "B12": 2190,
}
WAVELENGTH_ORDER = sorted(
    range(len(BAND_NAMES)),
    key=lambda index: CENTRAL_WAVELENGTH_NM[BAND_NAMES[index]],
)
BANDS_BY_WAVELENGTH = [BAND_NAMES[index] for index in WAVELENGTH_ORDER]

# Titles that would not fit above a small panel.
SHORT_TITLES = {"HerbaceousVegetation": "HerbaceousVeg."}


def collect_class_statistics(samples_per_class: int = SAMPLES_PER_CLASS):
    """Return per-class band means and per-image B08 means (file band order)."""
    if not DATA_DIR.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {DATA_DIR}")
    rng = np.random.default_rng(RANDOM_STATE)
    class_band_means = {}
    class_b08_means = {}

    for class_dir in sorted(path for path in DATA_DIR.iterdir() if path.is_dir()):
        image_paths = sorted(class_dir.glob("*.tif"))
        if not image_paths:
            continue
        sample_count = min(samples_per_class, len(image_paths))
        chosen = rng.choice(len(image_paths), size=sample_count, replace=False)
        image_means = []
        for index in chosen:
            with rasterio.open(image_paths[int(index)]) as source:
                image = source.read(masked=True).astype(np.float64)
            per_band = np.ma.mean(image, axis=(1, 2)).filled(np.nan)
            image_means.append(per_band)
        means = np.asarray(image_means)
        class_band_means[class_dir.name] = np.nanmean(means, axis=0)
        class_b08_means[class_dir.name] = means[:, BAND_NAMES.index("B08")]

    if not class_band_means:
        raise ValueError(f"No class TIFF images found under {DATA_DIR}")
    return class_band_means, class_b08_means


def save_spectral_signatures(class_band_means) -> None:
    """One small panel per class: its signature in blue over all the others."""
    class_names = list(class_band_means)
    values = np.asarray([class_band_means[name] for name in class_names])
    values = values[:, WAVELENGTH_ORDER]
    positions = np.arange(len(BANDS_BY_WAVELENGTH))

    fig, axes = plt.subplots(2, 5, figsize=(PAGE_WIDTH, 3.6), sharex=True, sharey=True)
    for ax, class_name, class_values in zip(axes.flat, class_names, values):
        for other_values in values:
            ax.plot(positions, other_values, color=CONTEXT_GRAY, linewidth=0.7)
        ax.plot(positions, class_values, color=BLUE, linewidth=1.8)
        ax.set_title(SHORT_TITLES.get(class_name, class_name), fontsize=8)
        ax.set_xticks(positions[::2], BANDS_BY_WAVELENGTH[::2], rotation=90, fontsize=6.5)
    for ax in axes[:, 0]:
        ax.set_ylabel("Valoare medie pixel")
    fig.tight_layout()
    save_figure(fig, EDA_FIGURES_DIR / "spectral_signatures.png")


def save_spectral_heatmap(class_band_means) -> None:
    """Per band: how far each class mean sits from the mean of all classes."""
    class_names = list(class_band_means)
    values = np.asarray([class_band_means[name] for name in class_names])
    values = values[:, WAVELENGTH_ORDER]
    standardized = (values - values.mean(axis=0)) / np.maximum(values.std(axis=0), 1e-8)

    fig, ax = plt.subplots(figsize=(PAGE_WIDTH, 3.3))
    picture = ax.imshow(standardized, cmap=DIVERGING_BLUE_RED, aspect="auto",
                        vmin=-2.5, vmax=2.5)
    ax.set_xticks(range(len(BANDS_BY_WAVELENGTH)), BANDS_BY_WAVELENGTH)
    ax.set_yticks(range(len(class_names)), class_names)
    ax.set_xlabel("Bandă Sentinel-2 (ordonate după lungimea de undă)")
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.colorbar(picture, ax=ax, label="Scor z față de media claselor", pad=0.02)
    fig.tight_layout()
    save_figure(fig, EDA_FIGURES_DIR / "spectral_heatmap.png")


def save_b08_distributions(class_b08_means) -> None:
    """Box plot of the per-image B08 mean, one box per class."""
    class_names = list(class_b08_means)
    fig, ax = plt.subplots(figsize=(PAGE_WIDTH, 3.0))
    ax.boxplot(
        [class_b08_means[name] for name in class_names],
        vert=False, widths=0.55, patch_artist=True,
        boxprops={"facecolor": "#cde2fb", "edgecolor": BLUE},
        medianprops={"color": BLUE, "linewidth": 1.8},
        whiskerprops={"color": BLUE}, capprops={"color": BLUE},
        flierprops={"marker": "o", "markersize": 3, "markerfacecolor": BLUE,
                    "markeredgecolor": "none"},
    )
    ax.set_yticks(range(1, len(class_names) + 1), class_names)
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Media benzii B08 pe imagine (infraroșu apropiat)")
    fig.tight_layout()
    save_figure(fig, EDA_FIGURES_DIR / "b08_boxplot.png")


def main() -> None:
    """Calculate the sampled spectral summaries and save the figures."""
    apply_style()
    band_means, b08_means = collect_class_statistics()

    for class_name, band_values in band_means.items():
        print(f"\n{class_name}:")
        for index in WAVELENGTH_ORDER:
            print(f"  {BAND_NAMES[index]}: {band_values[index]:.2f}")

    save_spectral_signatures(band_means)
    save_spectral_heatmap(band_means)
    save_b08_distributions(b08_means)


if __name__ == "__main__":
    main()
