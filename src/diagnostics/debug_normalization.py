
"""Debug per-band pixel accumulation for EuroSAT."""

from src.data.normalization import (
    DATA_DIR,
    BAND_NAMES,
    load_training_keys,
    initialize_accumulators,
    RANDOM_SPLIT_FILE,
    SPATIAL_SPLIT_FILE,
)

import rasterio
import numpy as np


def main():
    """Inspect valid pixel counts for each training split."""

    random_keys = load_training_keys(RANDOM_SPLIT_FILE)
    spatial_keys = load_training_keys(SPATIAL_SPLIT_FILE)

    accumulators = initialize_accumulators()

    for class_directory in sorted(DATA_DIR.iterdir()):

        if not class_directory.is_dir():
            continue

        for image_path in sorted(class_directory.glob("*.tif")):

            key = (class_directory.name, image_path.name)

            is_random_train = key in random_keys
            is_spatial_train = key in spatial_keys

            if not is_random_train and not is_spatial_train:
                continue

            with rasterio.open(image_path) as src:
                image = src.read(masked=True)

            if is_random_train:
                accumulators["random"].update(image)

            if is_spatial_train:
                accumulators["spatial"].update(image)

    for split_name, accumulator in accumulators.items():

        print(f"\n{split_name.upper()} SPLIT")
        print("=" * 65)

        for band_name, count in zip(
            BAND_NAMES,
            accumulator.counts
        ):
            print(f"{band_name}: {count:,} valid pixels")


if __name__ == "__main__":
    main()
