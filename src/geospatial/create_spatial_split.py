
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio

from rasterio.warp import transform_geom
from shapely.geometry import box, shape
from shapely.strtree import STRtree
from src.config import (
    PROJECT_ROOT,
    RANDOM_SPLIT_FILE,
    TABLES_DIR,
    TARGET_CRS,
    TRAIN_RATIO,
    VALIDATION_RATIO,
    TEST_RATIO,
    RANDOM_STATE,
)

INPUT_FILE = RANDOM_SPLIT_FILE
OUTPUT_DIR = TABLES_DIR

OUTPUT_SPLIT_FILE = OUTPUT_DIR / "spatial_dataset_splits.csv"
OUTPUT_GROUPS_FILE = OUTPUT_DIR / "spatial_groups.csv"
OUTPUT_SUMMARY_FILE = OUTPUT_DIR / "spatial_split_summary.csv"

SPLIT_RATIOS = {
    "train": TRAIN_RATIO,
    "validation": VALIDATION_RATIO,
    "test": TEST_RATIO,
}
RANDOM_SEED = RANDOM_STATE


# ============================================================
# UNION-FIND
# ============================================================

class UnionFind:
    """
    Identify connected components in an undirected graph.

    Each image represents a node. Images with overlapping
    geographic footprints are connected by an edge.
    """

    def __init__(self, size: int):
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, item: int) -> int:
        """Find the representative of a connected component."""

        if self.parent[item] != item:
            self.parent[item] = self.find(self.parent[item])

        return self.parent[item]

    def union(self, first: int, second: int) -> None:
        """Merge the components containing two nodes."""

        root_first = self.find(first)
        root_second = self.find(second)

        if root_first == root_second:
            return

        if self.rank[root_first] < self.rank[root_second]:
            root_first, root_second = root_second, root_first

        self.parent[root_second] = root_first

        if self.rank[root_first] == self.rank[root_second]:
            self.rank[root_first] += 1


# ============================================================
# SPATIAL SPLIT MANAGER
# ============================================================

class SpatialSplitManager:
    """
    Create a geographically grouped train/validation/test split.

    Images whose geographic footprints overlap are assigned
    to the same connected component. Entire components are
    subsequently assigned to one dataset subset.
    """

    def __init__(self):
        self.data = None
        self.geometries = []
        self.group_ids = None
        self.overlap_edges = []

    def load_dataset(self) -> pd.DataFrame:
        """
        Load the existing random split CSV.

        The original split assignment is preserved in the output.
        """

        if not INPUT_FILE.exists():
            raise FileNotFoundError(
                f"Input dataset split file not found: {INPUT_FILE}"
            )

        data = pd.read_csv(INPUT_FILE)

        if data.empty:
            raise ValueError("The dataset split file is empty.")

        path_column = self._find_column(
            data,
            ["image_path", "path", "filepath", "file_path"],
        )

        label_column = self._find_column(
            data,
            ["label", "class_name", "class", "category"],
        )

        if path_column is None:
            raise ValueError(
                "Could not identify the image path column. "
                f"Available columns: {data.columns.tolist()}"
            )

        if label_column is None:
            raise ValueError(
                "Could not identify the class label column. "
                f"Available columns: {data.columns.tolist()}"
            )

        data = data.copy()

        data["image_path"] = data[path_column].apply(
            self._resolve_image_path
        )

        data["class_name"] = data[label_column].astype(str)

        missing_files = [
            path for path in data["image_path"]
            if not path.exists()
        ]

        if missing_files:
            raise FileNotFoundError(
                f"{len(missing_files)} image files could not be found. "
                f"First missing file: {missing_files[0]}"
            )

        if data["image_path"].duplicated().any():
            raise ValueError(
                "Duplicate image paths were found in dataset_splits.csv."
            )

        self.data = data.reset_index(drop=True)

        print(f"Loaded images: {len(self.data)}")
        print(
            "Classes:",
            self.data["class_name"].nunique(),
        )

        return self.data

    @staticmethod
    def _find_column(
        data: pd.DataFrame,
        candidates: list[str],
    ):
        """Find a column using a list of accepted names."""

        columns = {
            column.lower(): column
            for column in data.columns
        }

        for candidate in candidates:
            if candidate.lower() in columns:
                return columns[candidate.lower()]

        return None

    @staticmethod
    def _resolve_image_path(raw_path: str) -> Path:
        """Resolve an image path relative to the project if necessary."""

        path = Path(str(raw_path))

        if path.is_absolute():
            return path

        return PROJECT_ROOT / path

    def extract_footprints(self) -> None:
        """
        Extract image footprints and transform them to EPSG:3035.

        The resulting geometries are stored in a common projected CRS
        whose coordinates are measured in metres.
        """

        print("\nExtracting geographic footprints...")

        self.geometries = []

        for index, image_path in enumerate(self.data["image_path"]):

            try:
                with rasterio.open(image_path) as src:

                    if src.crs is None:
                        raise ValueError(
                            f"Missing CRS: {image_path}"
                        )

                    # Create the footprint from raster bounds.
                    footprint = box(*src.bounds)

                    # Transform the footprint to the common CRS.
                    transformed = transform_geom(
                        src.crs,
                        TARGET_CRS,
                        footprint.__geo_interface__,
                    )

                    geometry = shape(transformed)

                    if geometry.is_empty or not geometry.is_valid:
                        raise ValueError(
                            f"Invalid footprint: {image_path}"
                        )

                    self.geometries.append(geometry)

            except Exception as error:
                raise RuntimeError(
                    f"Could not process image {image_path}: {error}"
                ) from error

            if (index + 1) % 1000 == 0:
                print(
                    f"Processed {index + 1}/{len(self.data)} images"
                )

        print(
            f"Successfully extracted {len(self.geometries)} footprints."
        )

    def identify_spatial_groups(self) -> None:
        """
        Build connected components from intersecting footprints.

        Only intersections with positive area are considered.
        Boundary-only contact does not connect two images.
        """

        print("\nIdentifying spatially connected images...")

        number_of_images = len(self.geometries)

        union_find = UnionFind(number_of_images)

        tree = STRtree(self.geometries)

        self.overlap_edges = []

        for index, geometry in enumerate(self.geometries):

            candidate_indices = tree.query(
                geometry,
                predicate="intersects",
            )

            for candidate in candidate_indices:

                candidate = int(candidate)

                # Process every pair only once.
                if candidate <= index:
                    continue

                other_geometry = self.geometries[candidate]

                intersection_area = geometry.intersection(
                    other_geometry
                ).area

                if intersection_area > 0:

                    union_find.union(index, candidate)

                    self.overlap_edges.append(
                        (index, candidate)
                    )

            if (index + 1) % 1000 == 0:
                print(
                    f"Checked {index + 1}/{number_of_images} footprints"
                )

        # Assign a unique group ID to each connected component.
        component_mapping = {}
        group_ids = []

        for index in range(number_of_images):

            root = union_find.find(index)

            if root not in component_mapping:
                component_mapping[root] = len(component_mapping)

            group_ids.append(component_mapping[root])

        self.group_ids = np.asarray(group_ids)

        self.data["spatial_group"] = self.group_ids

        number_of_groups = len(component_mapping)

        group_sizes = (
            self.data.groupby("spatial_group")
            .size()
            .sort_values(ascending=False)
        )

        print("\nSpatial grouping completed.")
        print(f"Total spatial groups: {number_of_groups}")
        print(f"Overlapping image pairs: {len(self.overlap_edges)}")
        print(f"Largest group: {group_sizes.iloc[0]} images")

        print("\nLargest 10 spatial groups:")
        print(group_sizes.head(10).to_string())

    def assign_groups_to_splits(self) -> None:
        """
        Assign complete spatial groups to train, validation and test.

        A greedy balancing strategy attempts to preserve both
        the requested split proportions and class distributions.

        Exact stratification is not guaranteed because entire
        spatial groups must remain together.
        """

        print("\nAssigning spatial groups to dataset splits...")

        classes = sorted(self.data["class_name"].unique())
        splits = list(SPLIT_RATIOS.keys())

        class_to_index = {
            class_name: index
            for index, class_name in enumerate(classes)
        }

        number_of_classes = len(classes)
        number_of_splits = len(splits)

        # Build a class-count vector for each spatial group.
        group_vectors = {}

        for group_id, group_data in self.data.groupby("spatial_group"):

            vector = np.zeros(number_of_classes, dtype=float)

            for class_name, count in group_data[
                "class_name"
            ].value_counts().items():

                vector[class_to_index[class_name]] = count

            group_vectors[group_id] = vector

        total_class_counts = np.zeros(
            number_of_classes,
            dtype=float,
        )

        for vector in group_vectors.values():
            total_class_counts += vector

        total_images = len(self.data)

        target_totals = np.array(
            [
                total_images * SPLIT_RATIOS[split]
                for split in splits
            ],
            dtype=float,
        )

        target_class_counts = np.array(
            [
                total_class_counts * SPLIT_RATIOS[split]
                for split in splits
            ]
        )

        # Largest groups are assigned first.
        rng = np.random.default_rng(RANDOM_SEED)

        group_ids = list(group_vectors.keys())
        rng.shuffle(group_ids)

        group_ids.sort(
            key=lambda group_id: group_vectors[group_id].sum(),
            reverse=True,
        )

        assigned_totals = np.zeros(number_of_splits, dtype=float)

        assigned_class_counts = np.zeros(
            (number_of_splits, number_of_classes),
            dtype=float,
        )

        group_assignments = {}

        for group_id in group_ids:

            vector = group_vectors[group_id]
            group_size = vector.sum()

            best_split_index = None
            best_cost = float("inf")

            for split_index in range(number_of_splits):

                # Calculate the total-size balancing cost.
                old_total_error = (
                    assigned_totals[split_index]
                    - target_totals[split_index]
                ) ** 2 / max(target_totals[split_index], 1)

                new_total_error = (
                    assigned_totals[split_index]
                    + group_size
                    - target_totals[split_index]
                ) ** 2 / max(target_totals[split_index], 1)

                total_cost = new_total_error - old_total_error

                # Calculate the class-distribution balancing cost.
                old_class_error = (
                    (
                        assigned_class_counts[split_index]
                        - target_class_counts[split_index]
                    ) ** 2
                    / np.maximum(
                        target_class_counts[split_index],
                        1,
                    )
                ).mean()

                new_class_error = (
                    (
                        assigned_class_counts[split_index]
                        + vector
                        - target_class_counts[split_index]
                    ) ** 2
                    / np.maximum(
                        target_class_counts[split_index],
                        1,
                    )
                ).mean()

                class_cost = new_class_error - old_class_error

                cost = total_cost + class_cost

                if cost < best_cost:
                    best_cost = cost
                    best_split_index = split_index

            # Assign the entire group to the selected split.
            selected_split = splits[best_split_index]

            group_assignments[group_id] = selected_split

            assigned_totals[best_split_index] += group_size

            assigned_class_counts[best_split_index] += vector

        self.data["spatial_split"] = self.data[
            "spatial_group"
        ].map(group_assignments)

        print("\nGroup assignment completed.")

    def validate_spatial_split(self) -> None:
        """
        Verify that no overlapping pair belongs to different splits.
        """

        print("\nValidating spatial split...")

        violations = []

        for first, second in self.overlap_edges:

            split_first = self.data.iloc[first]["spatial_split"]
            split_second = self.data.iloc[second]["spatial_split"]

            if split_first != split_second:
                violations.append((first, second))

        if violations:
            raise RuntimeError(
                f"Spatial leakage detected: {len(violations)} pairs."
            )

        print(
            "Validation passed: no positive-area overlapping "
            "image pairs cross dataset splits."
        )

    def create_summary(self) -> pd.DataFrame:
        """
        Create a summary of image counts and class distributions.
        """

        summary = (
            self.data.groupby(
                ["spatial_split", "class_name"]
            )
            .size()
            .reset_index(name="image_count")
        )

        total_per_split = (
            self.data.groupby("spatial_split")
            .size()
            .to_dict()
        )

        summary["split_total"] = summary[
            "spatial_split"
        ].map(total_per_split)

        summary["class_percentage"] = (
            summary["image_count"]
            / summary["split_total"]
            * 100
        )

        return summary

    def save_results(self) -> None:
        """
        Save the spatial split, group assignments and summary reports.
        """

        OUTPUT_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        # Preserve original columns and add spatial assignments.
        self.data.to_csv(
            OUTPUT_SPLIT_FILE,
            index=False,
        )

        group_summary = (
            self.data.groupby("spatial_group")
            .agg(
                image_count=("image_path", "count"),
                class_count=("class_name", "nunique"),
                spatial_split=("spatial_split", "first"),
            )
            .reset_index()
        )

        group_summary.to_csv(
            OUTPUT_GROUPS_FILE,
            index=False,
        )

        summary = self.create_summary()

        summary.to_csv(
            OUTPUT_SUMMARY_FILE,
            index=False,
        )

        print("\nResults saved:")
        print(OUTPUT_SPLIT_FILE)
        print(OUTPUT_GROUPS_FILE)
        print(OUTPUT_SUMMARY_FILE)

    def print_final_report(self) -> None:
        """Print final split sizes and class distributions."""

        print("\n" + "=" * 65)
        print("SPATIAL SPLIT SUMMARY")
        print("=" * 65)

        total_images = len(self.data)

        split_summary = (
            self.data.groupby("spatial_split")
            .agg(
                images=("image_path", "count"),
                groups=("spatial_group", "nunique"),
            )
        )

        split_summary["percentage"] = (
            split_summary["images"] / total_images * 100
        )

        print(split_summary.to_string())

        print("\nCLASS DISTRIBUTION")
        print("=" * 65)

        class_distribution = pd.crosstab(
            self.data["class_name"],
            self.data["spatial_split"],
        )

        print(class_distribution.to_string())

        print("\nOriginal random split versus spatial split:")
        if "split" in self.data.columns:
            comparison = pd.crosstab(
                self.data["split"],
                self.data["spatial_split"],
            )

            print(comparison.to_string())

    def run(self) -> None:
        """Execute the complete spatial split workflow."""

        self.load_dataset()
        self.extract_footprints()
        self.identify_spatial_groups()
        self.assign_groups_to_splits()
        self.validate_spatial_split()
        self.save_results()
        self.print_final_report()


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    manager = SpatialSplitManager()
    manager.run()
