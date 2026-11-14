
"""
Compute training-set normalization statistics for EuroSAT.

Statistics are calculated independently for:
- Random train split
- Spatial train split

For each split, statistics are saved for:
- RGB channels: B04, B03, B02
- All 13 Sentinel-2 bands

Only training images are used to calculate normalization parameters.
"""

import json

import numpy as np
import pandas as pd
import rasterio
from src.config import (
    BAND_NAMES,
    DATA_DIR,
    RGB_BANDS,
    RANDOM_SPLIT_FILE,
    SPATIAL_SPLIT_FILE,
    TABLES_DIR,
)

OUTPUT_JSON = TABLES_DIR / "normalization_statistics.json"
OUTPUT_CSV = TABLES_DIR / "normalization_statistics.csv"
RGB_INDICES = RGB_BANDS
RGB_NAMES = [BAND_NAMES[index] for index in RGB_INDICES]


class ChannelStatistics:
    """Accumulate per-channel pixel statistics."""

    def __init__(self, number_of_channels):
        """Initialize accumulators for each channel."""

        self.sums = np.zeros(number_of_channels, dtype=np.float64)
        self.squared_sums = np.zeros(
            number_of_channels,
            dtype=np.float64
        )
        self.counts = np.zeros(
            number_of_channels,
            dtype=np.int64
        )

    def update(self, image):
        """Update statistics using one multispectral image."""

        for channel_index in range(image.shape[0]):

            values = image[channel_index].compressed()
            values = values[np.isfinite(values)]

            if values.size == 0:
                continue

            values = values.astype(np.float64, copy=False)

            self.sums[channel_index] += np.sum(values)
            self.squared_sums[channel_index] += np.sum(
                values * values
            )
            self.counts[channel_index] += values.size

    def calculate(self):
        """Calculate channel means and standard deviations."""

        if np.any(self.counts == 0):
            empty_channels = np.where(self.counts == 0)[0]

            raise ValueError(
                f"Channels without valid pixels: {empty_channels}"
            )

        means = self.sums / self.counts

        variances = (
            self.squared_sums / self.counts
            - means ** 2
        )

        standard_deviations = np.sqrt(
            np.maximum(variances, 0.0)
        )

        if np.any(standard_deviations == 0):
            raise ValueError(
                "At least one channel has zero standard deviation."
            )

        return {
            "mean": means.tolist(),
            "std": standard_deviations.tolist(),
            "valid_pixels": self.counts.tolist()
        }




def load_training_keys(split_file, split_column="split"):
    """Load training image identifiers using the specified split column."""

    if not split_file.exists():
        raise FileNotFoundError(
            f"Split file not found: {split_file}"
        )

    dataframe = pd.read_csv(split_file)

    required_columns = {"image", "label", split_column}

    if not required_columns.issubset(dataframe.columns):
        raise ValueError(
            f"Missing columns in {split_file.name}: "
            f"{required_columns - set(dataframe.columns)}"
        )

    training_data = dataframe[
        dataframe[split_column] == "train"
    ]

    keys = set(
        zip(
            training_data["label"].astype(str),
            training_data["image"].astype(str)
        )
    )

    print(
        f"{split_file.name} [{split_column}]: "
        f"{len(keys)} training images"
    )

    return keys

def initialize_accumulators():
    """Create independent accumulators for both split strategies."""

    return {
        "random": ChannelStatistics(len(BAND_NAMES)),
        "spatial": ChannelStatistics(len(BAND_NAMES))
    }



def calculate_statistics():
    """Calculate statistics using training images from both splits."""

    random_training_keys = load_training_keys(
        RANDOM_SPLIT_FILE,
        split_column="split"
    )

    spatial_training_keys = load_training_keys(
        SPATIAL_SPLIT_FILE,
        split_column="spatial_split"
    )

    print("\nTRAINING SET COMPARISON")

    print(f"Random training keys: {len(random_training_keys)}")
    print(f"Spatial training keys: {len(spatial_training_keys)}")

    print(
        "Common training images:",
        len(random_training_keys & spatial_training_keys)
    )

    print(
        "Random-only training images:",
        len(random_training_keys - spatial_training_keys)
    )

    print(
        "Spatial-only training images:",
        len(spatial_training_keys - random_training_keys)
    )

    accumulators = initialize_accumulators()

    random_images = 0
    spatial_images = 0
    processed_images = 0

    class_directories = sorted(
        path for path in DATA_DIR.iterdir()
        if path.is_dir()
    )

    for class_directory in class_directories:

        image_paths = sorted(class_directory.glob("*.tif"))

        for image_path in image_paths:

            key = (
                class_directory.name,
                image_path.name
            )

            is_random_train = key in random_training_keys
            is_spatial_train = key in spatial_training_keys

            if not is_random_train and not is_spatial_train:
                continue

            with rasterio.open(image_path) as src:

                if src.count != len(BAND_NAMES):
                    raise ValueError(
                        f"Unexpected band count in {image_path}: "
                        f"{src.count}"
                    )

                image = src.read(masked=True)

            if is_random_train:
                accumulators["random"].update(image)
                random_images += 1

            if is_spatial_train:
                accumulators["spatial"].update(image)
                spatial_images += 1

            processed_images += 1

            if processed_images % 1000 == 0:
                print(
                    f"Processed {processed_images} "
                    f"unique training images..."
                )

    if random_images != len(random_training_keys):
        raise ValueError(
            "Some random training images were not found "
            "in the dataset directory."
        )

    if spatial_images != len(spatial_training_keys):
        raise ValueError(
            "Some spatial training images were not found "
            "in the dataset directory."
        )

    print("\nProcessed image counts:")
    print(f"Random training images: {random_images}")
    print(f"Spatial training images: {spatial_images}")

    return accumulators



def build_results(accumulators):
    """Build normalization parameters for RGB and multispectral data."""

    results = {}
    csv_rows = []

    for split_name, accumulator in accumulators.items():

        multispectral = accumulator.calculate()

        rgb = {
            "mean": [
                multispectral["mean"][index]
                for index in RGB_INDICES
            ],
            "std": [
                multispectral["std"][index]
                for index in RGB_INDICES
            ],
            "valid_pixels": [
                multispectral["valid_pixels"][index]
                for index in RGB_INDICES
            ]
        }

        results[split_name] = {
            "rgb": {
                "band_names": RGB_NAMES,
                **rgb
            },
            "multispectral": {
                "band_names": BAND_NAMES,
                **multispectral
            }
        }

        representations = [
            ("RGB", RGB_NAMES, rgb),
            ("Multispectral", BAND_NAMES, multispectral)
        ]

        for representation, band_names, statistics in representations:

            for index, band_name in enumerate(band_names):

                csv_rows.append({
                    "split": split_name,
                    "representation": representation,
                    "band": band_name,
                    "mean": statistics["mean"][index],
                    "std": statistics["std"][index],
                    "valid_pixels": statistics["valid_pixels"][index]
                })

    return results, pd.DataFrame(csv_rows)



def main():
    """Calculate and save normalization parameters."""

    print("EUROSAT NORMALIZATION STATISTICS")
    print("=" * 65)

    if not DATA_DIR.exists():
        raise FileNotFoundError(
            f"Dataset directory not found: {DATA_DIR}"
        )

    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    accumulators = calculate_statistics()

    results, statistics_dataframe = build_results(
        accumulators
    )

    with open(OUTPUT_JSON, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=4)

    statistics_dataframe.to_csv(
        OUTPUT_CSV,
        index=False
    )

    print("\nNORMALIZATION STATISTICS")
    print("=" * 100)

    print(
        statistics_dataframe.to_string(
            index=False,
            float_format="%.4f"
        )
    )

    print("\nResults saved:")
    print(OUTPUT_JSON)
    print(OUTPUT_CSV)


if __name__ == "__main__":
    main()
