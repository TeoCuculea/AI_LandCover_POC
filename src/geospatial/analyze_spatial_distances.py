
import numpy as np
import pandas as pd

from sklearn.neighbors import NearestNeighbors
from src.config import RESULTS_DIR, SPATIAL_SPLIT_FILE, TABLES_DIR

SPLIT_FILE = SPATIAL_SPLIT_FILE
COORDINATES_FILE = RESULTS_DIR / "eurosat_geographic_coordinates.csv"
OUTPUT_DIR = TABLES_DIR

OUTPUT_FILE = OUTPUT_DIR / "spatial_distance_analysis.csv"
SUMMARY_FILE = OUTPUT_DIR / "spatial_distance_summary.csv"

DISTANCE_THRESHOLDS = [100, 500, 1000, 5000]


# ============================================================
# SPATIAL DISTANCE ANALYZER
# ============================================================

class SpatialDistanceAnalyzer:
    """
    Analyze geographic distances between dataset subsets.

    Images are matched using their unique filenames.
    Geographic distances are calculated in EPSG:3035.
    """

    def __init__(self):
        self.data = None
        self.results = None

    def load_data(self) -> None:
        """Load spatial splits and geographic coordinates."""

        if not SPLIT_FILE.exists():
            raise FileNotFoundError(
                f"Split file not found: {SPLIT_FILE}"
            )

        if not COORDINATES_FILE.exists():
            raise FileNotFoundError(
                f"Coordinates file not found: {COORDINATES_FILE}"
            )

        splits = pd.read_csv(SPLIT_FILE)
        coordinates = pd.read_csv(COORDINATES_FILE)

        required_split_columns = {
            "image",
            "class_name",
            "spatial_split",
        }

        required_coordinate_columns = {
            "image",
            "class_name",
            "center_x",
            "center_y",
        }

        missing_split_columns = (
            required_split_columns - set(splits.columns)
        )

        missing_coordinate_columns = (
            required_coordinate_columns - set(coordinates.columns)
        )

        if missing_split_columns:
            raise ValueError(
                f"Missing split columns: {missing_split_columns}"
            )

        if missing_coordinate_columns:
            raise ValueError(
                f"Missing coordinate columns: {missing_coordinate_columns}"
            )

        # Normalize image filenames before merging.
        splits["image"] = (
            splits["image"]
            .astype(str)
            .str.strip()
            .str.replace("\\", "/", regex=False)
            .str.split("/")
            .str[-1]
        )

        coordinates["image"] = (
            coordinates["image"]
            .astype(str)
            .str.strip()
            .str.replace("\\", "/", regex=False)
            .str.split("/")
            .str[-1]
        )

        # Verify filename uniqueness.
        if splits["image"].duplicated().any():
            raise ValueError(
                "Duplicate image filenames found in spatial split."
            )

        if coordinates["image"].duplicated().any():
            raise ValueError(
                "Duplicate image filenames found in coordinates file."
            )

        # Keep only the columns needed for the analysis.
        coordinate_columns = [
            "image",
            "class_name",
            "center_x",
            "center_y",
        ]

        # Merge by unique image filename.
        self.data = splits.merge(
            coordinates[coordinate_columns],
            on="image",
            how="left",
            suffixes=("_split", "_coordinates"),
            validate="one_to_one",
            indicator=True,
        )

        # Verify that all images were matched.
        unmatched = self.data[
            self.data["_merge"] != "both"
        ]

        if not unmatched.empty:
            raise ValueError(
                f"{len(unmatched)} images could not be matched "
                "with geographic coordinates. "
                f"Examples: {unmatched['image'].head(10).tolist()}"
            )

        self.data.drop(columns=["_merge"], inplace=True)

        # Verify class consistency.
        inconsistent_classes = self.data[
            self.data["class_name_split"]
            != self.data["class_name_coordinates"]
        ]

        if not inconsistent_classes.empty:
            raise ValueError(
                f"Class mismatch detected for "
                f"{len(inconsistent_classes)} images."
            )

        self.data.rename(
            columns={
                "class_name_split": "class_name"
            },
            inplace=True,
        )

        self.data.drop(
            columns=["class_name_coordinates"],
            inplace=True,
        )

        # Validate coordinates.
        if self.data[
            ["center_x", "center_y"]
        ].isna().any().any():

            raise ValueError(
                "Missing geographic coordinates detected."
            )

        if not np.isfinite(
            self.data[["center_x", "center_y"]].to_numpy()
        ).all():

            raise ValueError(
                "Invalid or non-finite geographic coordinates detected."
            )

        print(f"Loaded images: {len(self.data)}")
        print("All images successfully matched with coordinates.")

        print("\nSubset distribution:")
        print(
            self.data["spatial_split"]
            .value_counts()
            .to_string()
        )

    def analyze_source_split(
        self,
        source_split: str,
        target_split: str,
    ) -> pd.DataFrame:
        """
        Find the nearest target image for every source image.

        The target subset is used as the nearest-neighbor reference.
        """

        source = self.data[
            self.data["spatial_split"] == source_split
        ].copy()

        target = self.data[
            self.data["spatial_split"] == target_split
        ].copy()

        if source.empty or target.empty:
            raise ValueError(
                f"Empty subset: {source_split} or {target_split}"
            )

        source_coordinates = source[
            ["center_x", "center_y"]
        ].to_numpy()

        target_coordinates = target[
            ["center_x", "center_y"]
        ].to_numpy()

        model = NearestNeighbors(
            n_neighbors=1,
            metric="euclidean",
        )

        model.fit(target_coordinates)

        distances, indices = model.kneighbors(
            source_coordinates
        )

        nearest_target = target.iloc[
            indices[:, 0]
        ].reset_index(drop=True)

        return pd.DataFrame({
            "source_image": source["image"].to_numpy(),
            "source_path": source["image_path"].to_numpy(),
            "source_class": source["class_name"].to_numpy(),
            "source_split": source_split,
            "target_image": nearest_target["image"].to_numpy(),
            "target_path": nearest_target["image_path"].to_numpy(),
            "target_class": nearest_target["class_name"].to_numpy(),
            "target_split": target_split,
            "nearest_distance_m": distances[:, 0],
        })

    def calculate_distance_counts(
        self,
        source_split: str,
        target_split: str,
    ) -> pd.DataFrame:
        """
        Count target images within the specified distance thresholds.
        """

        source = self.data[
            self.data["spatial_split"] == source_split
        ].copy()

        target = self.data[
            self.data["spatial_split"] == target_split
        ].copy()

        model = NearestNeighbors(
            radius=max(DISTANCE_THRESHOLDS),
            metric="euclidean",
        )

        model.fit(
            target[["center_x", "center_y"]].to_numpy()
        )

        distances, _ = model.radius_neighbors(
            source[["center_x", "center_y"]].to_numpy(),
            return_distance=True,
        )

        counts = {}

        for threshold in DISTANCE_THRESHOLDS:

            counts[f"images_within_{threshold}m"] = [
                int(np.sum(neighbor_distances <= threshold))
                for neighbor_distances in distances
            ]

        result = pd.DataFrame(counts)

        result.insert(
            0,
            "source_image",
            source["image"].to_numpy(),
        )

        return result

    def create_summary(
        self,
        results: pd.DataFrame,
    ) -> pd.DataFrame:
        """Calculate nearest-neighbor distance statistics."""

        summary = (
            results.groupby(
                ["source_split", "target_split"]
            )["nearest_distance_m"]
            .agg(
                count="count",
                minimum="min",
                mean="mean",
                median="median",
                maximum="max",
                standard_deviation="std",
            )
            .reset_index()
        )

        return summary

    def print_test_analysis(
        self,
        test_results: pd.DataFrame,
    ) -> None:
        """Print distance percentages and nearby-image statistics."""

        print("\nTEST IMAGES: DISTANCE TO TRAIN")
        print("=" * 65)

        for threshold in DISTANCE_THRESHOLDS:

            percentage = (
                test_results["nearest_distance_m"]
                .le(threshold)
                .mean()
                * 100
            )

            print(
                f"Nearest train image within {threshold:>5} m: "
                f"{percentage:.2f}%"
            )

        print("\nTEST IMAGES: NUMBER OF NEARBY TRAIN IMAGES")
        print("=" * 65)

        for threshold in DISTANCE_THRESHOLDS:

            column = f"images_within_{threshold}m"

            print(
                f"Mean number within {threshold:>5} m: "
                f"{test_results[column].mean():.2f}"
            )

            print(
                f"Median number within {threshold:>5} m: "
                f"{test_results[column].median():.0f}"
            )

    def run(self) -> None:
        """Execute the complete spatial distance analysis."""

        self.load_data()

        # Calculate nearest train image for test and validation images.
        test_to_train = self.analyze_source_split(
            "test",
            "train",
        )

        validation_to_train = self.analyze_source_split(
            "validation",
            "train",
        )

        # Calculate nearby training-image counts for test images.
        nearby_counts = self.calculate_distance_counts(
            "test",
            "train",
        )

        test_to_train = test_to_train.merge(
            nearby_counts,
            on="source_image",
            how="left",
            validate="one_to_one",
        )

        self.results = pd.concat(
            [
                test_to_train,
                validation_to_train,
            ],
            ignore_index=True,
        )

        OUTPUT_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.results.to_csv(
            OUTPUT_FILE,
            index=False,
        )

        summary = self.create_summary(self.results)

        summary.to_csv(
            SUMMARY_FILE,
            index=False,
        )

        print("\nDISTANCE SUMMARY")
        print("=" * 65)
        print(summary.to_string(index=False))

        self.print_test_analysis(test_to_train)

        print("\nResults saved:")
        print(OUTPUT_FILE)
        print(SUMMARY_FILE)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    analyzer = SpatialDistanceAnalyzer()
    analyzer.run()
