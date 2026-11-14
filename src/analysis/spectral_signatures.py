"""Plot class-level spectral signatures from reproducible EuroSAT samples."""

import matplotlib.pyplot as plt
import numpy as np
import rasterio

from src.config import BAND_NAMES, DATA_DIR, RANDOM_STATE

SAMPLES_PER_CLASS = 50


def collect_class_statistics(samples_per_class: int = SAMPLES_PER_CLASS):
    """Return per-class band means and per-image B08 means."""
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


def plot_spectral_signatures(class_band_means) -> None:
    """Plot class mean reflectance profiles and a standardized heatmap."""
    class_names = list(class_band_means)
    values = np.asarray([class_band_means[name] for name in class_names])
    for class_name, band_values in class_band_means.items():
        print(f"\n{class_name}:")
        for band, value in zip(BAND_NAMES, band_values):
            print(f"  {band}: {value:.2f}")

    fig, ax = plt.subplots(figsize=(13, 6))
    for class_name, band_values in class_band_means.items():
        ax.plot(BAND_NAMES, band_values, marker="o", label=class_name)
    ax.set(title="Mean EuroSAT spectral signatures", xlabel="Sentinel-2 band",
           ylabel="Mean pixel value")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    ax.grid(True)
    fig.tight_layout()
    plt.show()

    band_means = values.mean(axis=0)
    band_stds = values.std(axis=0)
    standardized = (values - band_means) / np.maximum(band_stds, 1e-8)
    fig, ax = plt.subplots(figsize=(13, 7))
    image = ax.imshow(standardized, cmap="RdYlBu_r", aspect="auto", vmin=-2, vmax=2)
    ax.set_xticks(range(len(BAND_NAMES)), BAND_NAMES)
    ax.set_yticks(range(len(class_names)), class_names)
    ax.set(title="Standardized spectral signatures", xlabel="Sentinel-2 band", ylabel="Class")
    fig.colorbar(image, ax=ax, label="Z-score")
    fig.tight_layout()
    plt.show()


def plot_b08_distributions(class_b08_means) -> None:
    """Compare per-image B08 mean distributions by class."""
    class_names = list(class_b08_means)
    fig, ax = plt.subplots(figsize=(13, 7))
    ax.boxplot([class_b08_means[name] for name in class_names])
    ax.set_xticks(range(1, len(class_names) + 1), class_names)
    ax.set(title="Per-image B08 mean by EuroSAT class", xlabel="Class",
           ylabel="Mean B08 pixel value")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(axis="y")
    fig.tight_layout()
    plt.show()


def main() -> None:
    """Calculate and display sampled spectral summaries."""
    band_means, b08_means = collect_class_statistics()
    plot_spectral_signatures(band_means)
    plot_b08_distributions(b08_means)


if __name__ == "__main__":
    main()
