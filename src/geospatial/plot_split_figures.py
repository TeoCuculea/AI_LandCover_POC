"""Figures that explain the random and the spatial split. No recomputation.

Reads the tables already saved by the split and leakage steps and saves
three figures to ``results/figures/splits/``:
- image pairs that overlap on the ground but sit in different random subsets
- where the train / validation / test images of the spatial split are
- how far each spatial-split test image is from its nearest training image

Usage:
    python -m src.geospatial.plot_split_figures
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio

from src.analysis.exploration import to_display_rgb
from src.analysis.plot_style import (
    AQUA,
    BLUE,
    INK_SECONDARY,
    MUTED,
    ORANGE,
    PAGE_WIDTH,
    apply_style,
    save_figure,
)
from src.config import (
    COORDINATES_FILE,
    DATA_DIR,
    FIGURES_DIR,
    RANDOM_STATE,
    SPATIAL_SPLIT_FILE,
    TABLES_DIR,
)

SPLIT_FIGURES_DIR = FIGURES_DIR / "splits"
SPLIT_LABELS = {"train": "antrenare", "validation": "validare", "test": "test"}
SPLIT_COLORS = {"train": BLUE, "validation": ORANGE, "test": AQUA}


def _read_rgb(class_name: str, image_name: str) -> np.ndarray:
    with rasterio.open(DATA_DIR / class_name / image_name) as source:
        return to_display_rgb(source.read())


def save_overlap_examples(pair_count: int = 3) -> None:
    """Show the most strongly overlapping pairs that the random split separated."""
    pairs = pd.read_csv(TABLES_DIR / "spatial_leakage_pairs.csv")
    pairs = pairs.sort_values("intersection_over_union", ascending=False).head(pair_count)

    fig, axes = plt.subplots(2, pair_count, figsize=(PAGE_WIDTH, 4.3))
    for column, pair in enumerate(pairs.itertuples()):
        images = (
            (pair.class_1, pair.image_1, pair.split_1),
            (pair.class_2, pair.image_2, pair.split_2),
        )
        for row, (class_name, image_name, split) in enumerate(images):
            ax = axes[row, column]
            ax.imshow(_read_rgb(class_name, image_name))
            ax.set_xlabel(f"{image_name[:-4]} · {SPLIT_LABELS[split]}",
                          fontsize=8, color=INK_SECONDARY)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)
            for spine in ax.spines.values():
                spine.set_visible(False)
        iou = f"{pair.intersection_over_union:.2f}".replace(".", ",")
        axes[0, column].set_title(f"Suprapunere (IoU) {iou}", loc="center")
    fig.tight_layout()
    save_figure(fig, SPLIT_FIGURES_DIR / "overlap_examples.png")


def save_spatial_split_map() -> None:
    """Scatter of image centres (longitude, latitude) coloured by spatial subset."""
    splits = pd.read_csv(SPATIAL_SPLIT_FILE, usecols=["image", "spatial_split"])
    coordinates = pd.read_csv(COORDINATES_FILE, usecols=["image", "longitude", "latitude"])
    data = splits.merge(coordinates, on="image", validate="one_to_one")

    # Points are drawn in random order so that no subset hides the others.
    data = data.sample(frac=1, random_state=RANDOM_STATE)

    fig, ax = plt.subplots(figsize=(PAGE_WIDTH, 4.4))
    ax.scatter(data["longitude"], data["latitude"], s=2.5, linewidths=0,
               color=data["spatial_split"].map(SPLIT_COLORS))
    for split in ("train", "validation", "test"):
        count = int((data["spatial_split"] == split).sum())
        ax.scatter([], [], s=12, color=SPLIT_COLORS[split],
                   label=f"{SPLIT_LABELS[split]} ({count:,})".replace(",", "."))
    ax.set_xlabel("Longitudine (°)")
    ax.set_ylabel("Latitudine (°)")
    ax.set_aspect(1.5)  # rough correction for the latitude of Europe
    ax.legend(loc="upper left")
    fig.tight_layout()
    save_figure(fig, SPLIT_FIGURES_DIR / "spatial_split_map.png")


def save_test_distance_histogram() -> None:
    """Distance from every spatial-split test image to its nearest training image."""
    distances = pd.read_csv(TABLES_DIR / "spatial_distance_analysis.csv")
    test = distances.loc[
        (distances["source_split"] == "test") & (distances["target_split"] == "train"),
        "nearest_distance_m",
    ].to_numpy() / 1000

    fig, ax = plt.subplots(figsize=(PAGE_WIDTH, 2.9))
    bins = np.logspace(np.log10(0.5), np.log10(test.max() * 1.05), 45)
    ax.hist(test, bins=bins, color=BLUE, rwidth=0.9)
    ax.set_xscale("log")
    ax.set_xticks([0.5, 1, 2, 5, 10, 20, 50, 100], ["0,5", "1", "2", "5", "10", "20", "50", "100"])
    ax.set_xlabel("Distanța până la cea mai apropiată imagine de antrenare (km, scară logaritmică)")
    ax.set_ylabel("Imagini de test")

    median = float(np.median(test))
    ax.axvline(median, color=MUTED, linewidth=0.9)
    ax.annotate(f"mediana: {median:.2f} km".replace(".", ","), xy=(median, 1),
                xycoords=("data", "axes fraction"), xytext=(5, -4),
                textcoords="offset points", va="top", fontsize=8, color=INK_SECONDARY)
    fig.tight_layout()
    save_figure(fig, SPLIT_FIGURES_DIR / "test_to_train_distance.png")


def main() -> None:
    apply_style()
    save_overlap_examples()
    save_spatial_split_map()
    save_test_distance_histogram()


if __name__ == "__main__":
    main()
