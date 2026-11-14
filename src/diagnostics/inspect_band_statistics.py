
"""
Inspect the EuroSAT multispectral dataset.

The script samples GeoTIFF images and reports:
- Band count and image dimensions
- Data type, CRS, NoData values
- Per-band minimum, maximum, mean and standard deviation
- Per-band percentile distributions
- Available dataset split files
"""

import numpy as np
import pandas as pd
import rasterio


from src.config import BAND_NAMES, DATA_DIR, RANDOM_STATE, TABLES_DIR

RESULTS_DIR = TABLES_DIR
RANDOM_SEED = RANDOM_STATE
SAMPLES_PER_CLASS = 10


def discover_sample_images():
    """Select a reproducible sample of images from every class."""

    if not DATA_DIR.exists():
        raise FileNotFoundError(
            f"Dataset directory not found: {DATA_DIR}"
        )

    rng = np.random.default_rng(RANDOM_SEED)
    selected_paths = []

    class_directories = sorted(
        path for path in DATA_DIR.iterdir()
        if path.is_dir()
    )

    for class_directory in class_directories:
        image_paths = sorted(class_directory.glob("*.tif"))

        if not image_paths:
            continue

        sample_count = min(
            SAMPLES_PER_CLASS,
            len(image_paths)
        )

        selected_indices = rng.choice(
            len(image_paths),
            size=sample_count,
            replace=False
        )

        selected_paths.extend(
            image_paths[index]
            for index in selected_indices
        )

    return selected_paths


def inspect_image_metadata(image_path):
    """Print metadata for one representative GeoTIFF image."""

    with rasterio.open(image_path) as src:
        print("\nREPRESENTATIVE IMAGE METADATA")
        print("=" * 65)
        print(f"File: {image_path.name}")
        print(f"Dimensions: {src.width} x {src.height}")
        print(f"Number of bands: {src.count}")
        print(f"Data type: {src.dtypes}")
        print(f"CRS: {src.crs}")
        print(f"NoData value: {src.nodata}")
        print(f"Scale values: {src.scales}")
        print(f"Offset values: {src.offsets}")
        print(f"Band descriptions: {src.descriptions}")
        print(f"Resolution: {src.res}")


def calculate_band_statistics(image_paths):
    """Calculate statistics using valid pixels from sampled images."""

    band_values = [[] for _ in BAND_NAMES]

    for image_path in image_paths:
        with rasterio.open(image_path) as src:

            if src.count != len(BAND_NAMES):
                raise ValueError(
                    f"Unexpected band count in {image_path}: "
                    f"{src.count}"
                )

            image = src.read(masked=True)

            for band_index in range(src.count):
                values = image[band_index].compressed()
                values = values[np.isfinite(values)]

                if values.size > 0:
                    band_values[band_index].append(
                        values.astype(np.float64)
                    )

    statistics = []

    for band_index, band_name in enumerate(BAND_NAMES):

        if not band_values[band_index]:
            continue

        values = np.concatenate(band_values[band_index])

        statistics.append({
            "band": band_name,
            "minimum": np.min(values),
            "percentile_1": np.percentile(values, 1),
            "percentile_25": np.percentile(values, 25),
            "median": np.percentile(values, 50),
            "percentile_75": np.percentile(values, 75),
            "percentile_99": np.percentile(values, 99),
            "maximum": np.max(values),
            "mean": np.mean(values),
            "standard_deviation": np.std(values),
            "valid_pixels": len(values)
        })

    return pd.DataFrame(statistics)


def inspect_split_files():
    """Inspect available dataset split files."""

    split_files = [
        RESULTS_DIR / "dataset_splits.csv",
        RESULTS_DIR / "spatial_dataset_splits.csv"
    ]

    print("\nDATASET SPLIT FILES")
    print("=" * 65)

    for split_file in split_files:

        if not split_file.exists():
            print(f"\nNot found: {split_file.name}")
            continue

        dataframe = pd.read_csv(split_file)

        print(f"\nFile: {split_file.name}")
        print(f"Rows: {len(dataframe)}")
        print(f"Columns: {list(dataframe.columns)}")

        if "split" in dataframe.columns:
            print("\nRandom split distribution:")
            print(dataframe["split"].value_counts())

        if "spatial_split" in dataframe.columns:
            print("\nSpatial split distribution:")
            print(dataframe["spatial_split"].value_counts())


def main():
    """Run the multispectral data inspection."""

    print("EUROSAT MULTISPECTRAL DATA INSPECTION")
    print("=" * 65)

    image_paths = discover_sample_images()

    print(f"Dataset directory: {DATA_DIR}")
    print(f"Number of sampled images: {len(image_paths)}")

    if not image_paths:
        raise RuntimeError("No GeoTIFF images were discovered.")

    inspect_image_metadata(image_paths[0])

    statistics = calculate_band_statistics(image_paths)

    print("\nPER-BAND STATISTICS")
    print("=" * 100)
    print(statistics.to_string(index=False, float_format="%.3f"))

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    output_file = RESULTS_DIR / "multispectral_band_statistics.csv"
    statistics.to_csv(output_file, index=False)

    print(f"\nBand statistics saved to: {output_file}")

    inspect_split_files()


if __name__ == "__main__":
    main()
