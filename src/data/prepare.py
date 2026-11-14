
from collections import Counter, defaultdict
import hashlib
from pathlib import Path

import rasterio
import pandas as pd

from sklearn.model_selection import train_test_split
from src.config import (
    DATA_DIR,
    TABLES_DIR,
    TRAIN_RATIO,
    VALIDATION_RATIO,
    TEST_RATIO,
    RANDOM_STATE,
)

RANDOM_SEED = RANDOM_STATE


class EuroSATDataManager:
    """
    Manage EuroSAT multispectral dataset preparation.

    Responsibilities:
    - Discover TIFF images and extract class labels.
    - Encode class labels.
    - Create stratified train, validation and test splits.
    - Verify dataset split integrity.
    - Detect exact duplicate images.
    - Detect geographic footprint overlaps between splits.

    Coordinates, nearest neighbours and clustering live in src/geospatial.
    - Save dataset splits and analysis results.
    """

    def __init__(self, data_dir):
        """Initialize dataset paths and internal data structures."""

        self.data_dir = Path(data_dir)

        self.image_paths = []
        self.labels = []
        self.encoded_labels = []

        self.class_names = []
        self.class_to_idx = {}
        self.idx_to_class = {}

        self.train_paths = []
        self.train_labels = []

        self.val_paths = []
        self.val_labels = []

        self.test_paths = []
        self.test_labels = []

    # --------------------------------------------------
    # 1. IMAGE DISCOVERY AND LABEL PREPARATION
    # --------------------------------------------------

    def discover_images(self):
        """Find all TIFF images in the dataset directory."""

        if not self.data_dir.exists():
            raise FileNotFoundError(
                f"Dataset directory not found: {self.data_dir}"
            )

        self.image_paths = sorted(
            self.data_dir.rglob("*.tif")
        )

        if not self.image_paths:
            raise FileNotFoundError(
                f"No TIFF images found in {self.data_dir}"
            )

        print(f"Images discovered: {len(self.image_paths)}")

    def extract_labels(self):
        """Extract class labels from parent directory names."""

        self.labels = [
            path.parent.name
            for path in self.image_paths
        ]

    def encode_labels(self):
        """Create mappings between class names and numeric IDs."""

        self.class_names = sorted(set(self.labels))

        self.class_to_idx = {
            class_name: idx
            for idx, class_name in enumerate(self.class_names)
        }

        self.idx_to_class = {
            idx: class_name
            for class_name, idx in self.class_to_idx.items()
        }

        self.encoded_labels = [
            self.class_to_idx[label]
            for label in self.labels
        ]

    def show_class_distribution(self):
        """Display the number of images in each class."""

        class_counts = Counter(self.labels)

        for class_name in sorted(class_counts):
            print(f"{class_name}: {class_counts[class_name]}")

    # --------------------------------------------------
    # 2. DATASET SPLITTING
    # --------------------------------------------------

    def split_dataset(self, random_seed=RANDOM_SEED):
        """
        Split the dataset into stratified train, validation
        and test subsets using a 70/15/15 ratio.
        """

        (
            self.train_paths,
            temp_paths,
            self.train_labels,
            temp_labels
        ) = train_test_split(
            self.image_paths,
            self.encoded_labels,
            test_size=1 - TRAIN_RATIO,
            random_state=random_seed,
            stratify=self.encoded_labels
        )

        (
            self.val_paths,
            self.test_paths,
            self.val_labels,
            self.test_labels
        ) = train_test_split(
            temp_paths,
            temp_labels,
            test_size=TEST_RATIO / (VALIDATION_RATIO + TEST_RATIO),
            random_state=random_seed,
            stratify=temp_labels
        )

        print("\nDataset split:")
        print(f"Train:      {len(self.train_paths)}")
        print(f"Validation: {len(self.val_paths)}")
        print(f"Test:       {len(self.test_paths)}")

    def verify_split_integrity(self):
        """Verify the consistency and uniqueness of dataset splits."""

        total_paths = (
            len(self.train_paths)
            + len(self.val_paths)
            + len(self.test_paths)
        )

        total_labels = (
            len(self.train_labels)
            + len(self.val_labels)
            + len(self.test_labels)
        )

        if total_paths != len(self.image_paths):
            raise ValueError("Mismatch in image paths.")

        if total_labels != len(self.encoded_labels):
            raise ValueError("Mismatch in encoded labels.")

        if len(self.train_paths) != len(self.train_labels):
            raise ValueError("Mismatch in training data.")

        if len(self.val_paths) != len(self.val_labels):
            raise ValueError("Mismatch in validation data.")

        if len(self.test_paths) != len(self.test_labels):
            raise ValueError("Mismatch in test data.")

        all_split_paths = (
            self.train_paths
            + self.val_paths
            + self.test_paths
        )

        normalized_paths = [
            str(Path(path).resolve())
            for path in all_split_paths
        ]

        if len(set(normalized_paths)) != len(normalized_paths):
            raise ValueError(
                "Duplicate image paths detected between splits."
            )

        if len(set(normalized_paths)) != len(self.image_paths):
            raise ValueError(
                "Some dataset images are missing from the splits."
            )

        print("\nDataset split integrity: OK")

    def show_split_distribution(self, labels, split_name):
        """Display class distribution and percentages for a dataset split."""

        class_counts = Counter(labels)
        total_images = len(labels)

        print(f"\n{split_name} distribution:")

        for class_id, class_name in self.idx_to_class.items():

            count = class_counts[class_id]

            percentage = (
                count / total_images * 100
                if total_images > 0 else 0
            )

            print(
                f"{class_name:<25} "
                f"{count:>6} "
                f"({percentage:>5.2f}%)"
            )

    # --------------------------------------------------
    # 3. DATASET SPLIT EXPORT
    # --------------------------------------------------

    def save_splits(self):
        """Save train, validation and test assignments to CSV files."""

        records = []

        split_data = [
            ("train", self.train_paths, self.train_labels),
            ("validation", self.val_paths, self.val_labels),
            ("test", self.test_paths, self.test_labels)
        ]

        for split_name, paths, labels in split_data:

            for path, label_id in zip(paths, labels):

                records.append({
                    "image_path": str(Path(path).resolve()),
                    "image": Path(path).name,
                    "label": self.idx_to_class[label_id],
                    "label_id": label_id,
                    "split": split_name
                })

        dataframe = pd.DataFrame(records)

        TABLES_DIR.mkdir(parents=True, exist_ok=True)

        dataframe.to_csv(
            TABLES_DIR / "dataset_splits.csv",
            index=False
        )

        for split_name in ["train", "validation", "test"]:

            split_df = dataframe[
                dataframe["split"] == split_name
            ]

            split_df.to_csv(
                TABLES_DIR / f"{split_name}.csv",
                index=False
            )

        print("\nDataset splits saved:")
        print(f"Total records: {len(dataframe)}")
        print(f"Output directory: {TABLES_DIR}")

        return dataframe

    # --------------------------------------------------
    # 4. DATA LEAKAGE AND DUPLICATE CHECKS
    # --------------------------------------------------

    def check_data_leakage(self):
        """Check whether image paths overlap between dataset splits."""

        train_set = {
            str(Path(path).resolve())
            for path in self.train_paths
        }

        val_set = {
            str(Path(path).resolve())
            for path in self.val_paths
        }

        test_set = {
            str(Path(path).resolve())
            for path in self.test_paths
        }

        train_val_overlap = train_set.intersection(val_set)
        train_test_overlap = train_set.intersection(test_set)
        val_test_overlap = val_set.intersection(test_set)

        if train_val_overlap or train_test_overlap or val_test_overlap:
            raise ValueError(
                "Overlapping image paths detected between splits."
            )

        print("\nData leakage path check: OK")

    def check_exact_duplicates(self):
        """Find images with identical pixel arrays."""

        image_hashes = defaultdict(list)

        for index, path in enumerate(self.image_paths):

            with rasterio.open(path) as src:
                image = src.read()

            image_hash = hashlib.sha256(
                image.tobytes()
            ).hexdigest()

            image_hashes[image_hash].append(path)

            if (index + 1) % 1000 == 0:
                print(
                    f"Duplicate check progress: "
                    f"{index + 1}/{len(self.image_paths)}"
                )

        duplicate_groups = [
            paths
            for paths in image_hashes.values()
            if len(paths) > 1
        ]

        if not duplicate_groups:
            print("\nExact duplicate check: OK")
            print("No identical pixel arrays found.")
            return

        print(
            f"\nDuplicate groups found: {len(duplicate_groups)}"
        )

        for paths in duplicate_groups:

            print("\nDuplicate images:")

            for path in paths:
                print(path)

    # --------------------------------------------------
    # 5. GEOGRAPHIC METADATA
    # --------------------------------------------------

    def summarize_geographic_metadata(self):
        """Summarize coordinate reference systems used in the dataset."""

        crs_counts = Counter()

        for path in self.image_paths:

            with rasterio.open(path) as src:
                crs_counts[str(src.crs)] += 1

        print("\nCoordinate Reference Systems:")

        for crs, count in sorted(crs_counts.items()):

            percentage = (
                count / len(self.image_paths) * 100
            )

            print(
                f"{crs}: {count} ({percentage:.2f}%)"
            )

        print(
            f"\nTotal images inspected: {sum(crs_counts.values())}"
        )

    def inspect_geographic_metadata(self, sample_size=5):
        """Inspect geographic metadata of a sample of images."""

        print("\nGeographic metadata inspection:")

        sample_paths = self.image_paths[:sample_size]

        for path in sample_paths:

            with rasterio.open(path) as src:

                print(f"\nImage: {path.name}")
                print(f"CRS: {src.crs}")
                print(f"Bounds: {src.bounds}")
                print(f"Resolution: {src.res}")
                print(f"Transform: {src.transform}")

    # --------------------------------------------------
    # 6. SPATIAL LEAKAGE ANALYSIS
    # --------------------------------------------------

    def analyze_spatial_leakage(self, splits_df):
        """Delegate split footprint analysis to the geospatial package."""
        from src.geospatial.leakage import analyze_split_overlaps

        return analyze_split_overlaps(splits_df)

    # --------------------------------------------------
    # 7. MAIN DATASET PREPARATION
    # --------------------------------------------------

    def prepare_dataset(self, random_seed=RANDOM_SEED):
        """Prepare dataset, labels and stratified splits."""

        self.discover_images()
        self.extract_labels()
        self.encode_labels()

        if (
            len(self.image_paths) != len(self.labels)
            or len(self.labels) != len(self.encoded_labels)
        ):
            raise ValueError(
                "Mismatch between images and labels."
            )

        print(f"\nTotal images: {len(self.image_paths)}")
        print(f"Total labels: {len(self.labels)}")
        print(f"Number of classes: {len(self.class_names)}")

        print("\nClass distribution:")
        self.show_class_distribution()

        print("\nClass mapping:")

        for class_name, idx in self.class_to_idx.items():
            print(f"{class_name}: {idx}")

        self.split_dataset(random_seed=random_seed)

        self.verify_split_integrity()

        self.show_split_distribution(
            self.train_labels,
            "Training"
        )

        self.show_split_distribution(
            self.val_labels,
            "Validation"
        )

        self.show_split_distribution(
            self.test_labels,
            "Test"
        )

        self.check_data_leakage()


# --------------------------------------------------
# 8. SCRIPT EXECUTION
# --------------------------------------------------

def main() -> None:
    """Prepare splits and run the associated spatial leakage checks."""
    manager = EuroSATDataManager(DATA_DIR)

    # Prepare the dataset and create stratified splits.
    manager.prepare_dataset()

    # Save split assignments.
    splits_df = manager.save_splits()

    # Analyze geographic overlaps between different splits.
    manager.analyze_spatial_leakage(splits_df)


if __name__ == "__main__":
    main()
