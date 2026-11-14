"""Interactive inspection of a EuroSAT multispectral sample image."""

from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import rasterio

from src.config import BAND_NAMES, DATA_DIR, RANDOM_STATE, RGB_BANDS


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


def _display_rgb(image) -> None:
    """Display a percentile-stretched RGB composite."""
    rgb = np.moveaxis(image[RGB_BANDS], 0, -1).astype(np.float32)
    low, high = np.percentile(rgb, (2, 98), axis=(0, 1))
    rgb = np.clip((rgb - low) / np.maximum(high - low, 1e-8), 0, 1)
    plt.figure(figsize=(6, 6))
    plt.imshow(rgb)
    plt.title("EuroSAT MS — RGB composite (B04, B03, B02)")
    plt.axis("off")
    plt.show()


def _display_spectrum(image) -> None:
    """Plot the spectrum for the center pixel."""
    row, column = image.shape[1] // 2, image.shape[2] // 2
    values = image[:, row, column]
    for band, value in zip(BAND_NAMES, values):
        print(f"{band}: {value}")
    plt.figure(figsize=(12, 5))
    plt.plot(BAND_NAMES, values, marker="o")
    plt.title("Center-pixel Sentinel-2 spectrum")
    plt.xlabel("Band")
    plt.ylabel("Pixel value")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def _display_ndvi(image) -> None:
    """Plot NDVI from Sentinel-2 B08 and B04."""
    red = image[BAND_NAMES.index("B04")].astype(np.float32)
    nir = image[BAND_NAMES.index("B08")].astype(np.float32)
    ndvi = (nir - red) / (nir + red + 1e-8)
    print(f"NDVI min/mean/max: {ndvi.min():.3f} / {ndvi.mean():.3f} / {ndvi.max():.3f}")
    plt.figure(figsize=(7, 6))
    plt.imshow(ndvi, cmap="RdYlGn", vmin=-1, vmax=1)
    plt.colorbar(label="NDVI")
    plt.title("NDVI — EuroSAT")
    plt.axis("off")
    plt.show()


def _display_class_distribution(image_paths) -> None:
    """Show image counts for every dataset class."""
    counts = Counter(path.parent.name for path in image_paths)
    print("\nImage counts by class:")
    for class_name, count in sorted(counts.items()):
        print(f"{class_name}: {count}")
    plt.figure(figsize=(12, 5))
    plt.bar(counts.keys(), counts.values())
    plt.title("EuroSAT MS image distribution")
    plt.xlabel("Class")
    plt.ylabel("Images")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()


def main() -> None:
    """Inspect a reproducible random image and dataset distribution."""
    image_paths = discover_images()
    rng = np.random.default_rng(RANDOM_STATE)
    sample_path = image_paths[int(rng.integers(len(image_paths)))]
    image, metadata = _read_image(sample_path)

    print(f"Images: {len(image_paths)}")
    print(f"Sample: {sample_path}")
    for name, value in metadata.items():
        print(f"{name}: {value}")
    print(f"Shape: {image.shape}; dtype: {image.dtype}; range: {image.min()}..{image.max()}")

    _display_rgb(image)
    _display_spectrum(image)
    _display_ndvi(image)
    _display_class_distribution(image_paths)


if __name__ == "__main__":
    main()
