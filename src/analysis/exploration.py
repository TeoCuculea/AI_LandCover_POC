"""First look at the EuroSAT multispectral dataset.

Prints the metadata of one sample image and saves three figures to
``results/figures/eda/``: the class distribution, one RGB sample per class,
and the NDVI of the sample image.

Usage:
    python -m src.analysis.exploration
"""

from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import rasterio

from src.analysis.plot_style import BLUE, PAGE_WIDTH, apply_style, save_figure
from src.config import BAND_NAMES, DATA_DIR, FIGURES_DIR, RANDOM_STATE, RGB_BANDS

EDA_FIGURES_DIR = FIGURES_DIR / "eda"

# Fixed display range for RGB pictures (raw Sentinel-2 L1C pixel values).
DISPLAY_MIN = 300
DISPLAY_MAX = 2600
DISPLAY_GAMMA = 0.75


def discover_images() -> list:
    """Return all TIFFs under the configured dataset directory."""
    if not DATA_DIR.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {DATA_DIR}")
    images = sorted(DATA_DIR.rglob("*.tif"))
    if not images:
        raise FileNotFoundError(f"No TIFF images found under {DATA_DIR}")
    return images


def _read_image(image_path):
    """Read and validate a 13-band image in channel-first order."""
    with rasterio.open(image_path) as source:
        image = source.read()
        if source.count != len(BAND_NAMES):
            raise ValueError(
                f"Expected {len(BAND_NAMES)} bands in {image_path}, "
                f"found {source.count}"
            )
        metadata = {
            "crs": source.crs,
            "resolution": source.res,
            "dtype": source.dtypes,
            "scales": source.scales,
            "offsets": source.offsets,
            "nodata": source.nodata,
            "descriptions": source.descriptions,
        }
    return image, metadata


def to_display_rgb(image) -> np.ndarray:
    """Turn a 13-band image into an RGB picture for viewing.

    All three channels share one fixed scale (pixel value 0..DISPLAY_MAX), so
    colours stay natural and uniform images such as open water are not
    amplified into noise. This is only for viewing: the model never sees it,
    training uses mean/std normalization instead.
    """
    rgb = np.moveaxis(image[RGB_BANDS], 0, -1).astype(np.float32)
    rgb = np.clip((rgb - DISPLAY_MIN) / (DISPLAY_MAX - DISPLAY_MIN), 0, 1)
    return rgb ** DISPLAY_GAMMA


def save_class_distribution(image_paths) -> None:
    """Bar chart of the number of images in every class."""
    counts = Counter(path.parent.name for path in image_paths)
    print("\nImage counts by class:")
    for class_name, count in sorted(counts.items()):
        print(f"{class_name}: {count}")

    names = sorted(counts)
    fig, ax = plt.subplots(figsize=(PAGE_WIDTH, 2.9))
    bars = ax.barh(names, [counts[name] for name in names], height=0.62, color=BLUE)
    ax.bar_label(bars, padding=3, fontsize=8)
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Număr de imagini")
    ax.set_xlim(0, 3300)
    fig.tight_layout()
    save_figure(fig, EDA_FIGURES_DIR / "class_distribution.png")


def save_sample_grid(image_paths, rng) -> None:
    """One RGB sample image for each of the ten classes."""
    by_class = {}
    for path in image_paths:
        by_class.setdefault(path.parent.name, []).append(path)

    fig, axes = plt.subplots(2, 5, figsize=(PAGE_WIDTH, 2.9))
    for ax, class_name in zip(axes.flat, sorted(by_class)):
        paths = by_class[class_name]
        image, _ = _read_image(paths[int(rng.integers(len(paths)))])
        ax.imshow(to_display_rgb(image))
        ax.set_title(class_name, fontsize=8, loc="center")
        ax.axis("off")
    fig.tight_layout()
    save_figure(fig, EDA_FIGURES_DIR / "sample_grid.png")


def save_ndvi_example(image, image_name: str) -> None:
    """RGB picture next to NDVI = (B08 - B04) / (B08 + B04) for one image."""
    red = image[BAND_NAMES.index("B04")].astype(np.float32)
    nir = image[BAND_NAMES.index("B08")].astype(np.float32)
    ndvi = (nir - red) / (nir + red + 1e-8)
    print(f"NDVI min/mean/max: {ndvi.min():.3f} / {ndvi.mean():.3f} / {ndvi.max():.3f}")

    fig, (rgb_ax, ndvi_ax) = plt.subplots(1, 2, figsize=(PAGE_WIDTH * 0.75, 2.4))
    rgb_ax.imshow(to_display_rgb(image))
    rgb_ax.set_title(f"RGB · {image_name}", fontsize=8)
    rgb_ax.axis("off")
    picture = ndvi_ax.imshow(ndvi, cmap="RdYlGn", vmin=-1, vmax=1)
    ndvi_ax.set_title("NDVI", fontsize=8)
    ndvi_ax.axis("off")
    fig.colorbar(picture, ax=ndvi_ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    save_figure(fig, EDA_FIGURES_DIR / "ndvi_example.png")


def main() -> None:
    """Inspect a reproducible random image and the dataset distribution."""
    apply_style()
    image_paths = discover_images()
    rng = np.random.default_rng(RANDOM_STATE)
    sample_path = image_paths[int(rng.integers(len(image_paths)))]
    image, metadata = _read_image(sample_path)

    print(f"Images: {len(image_paths)}")
    print(f"Sample: {sample_path}")
    for name, value in metadata.items():
        print(f"{name}: {value}")
    print(f"Shape: {image.shape}; dtype: {image.dtype}; range: {image.min()}..{image.max()}")

    print("\nCenter-pixel values:")
    row, column = image.shape[1] // 2, image.shape[2] // 2
    for band, value in zip(BAND_NAMES, image[:, row, column]):
        print(f"{band}: {value}")

    save_class_distribution(image_paths)
    save_sample_grid(image_paths, rng)
    save_ndvi_example(image, sample_path.stem)


if __name__ == "__main__":
    main()
