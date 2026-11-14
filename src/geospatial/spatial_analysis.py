
import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors


class SpatialAnalyzer:
    """
    Analyze geographic relationships between satellite image patches
    using projected spatial coordinates.
    """

    def __init__(self, neighbors=2, random_state=42):
        """
        Initialize spatial analysis parameters.

        Args:
            neighbors (int): Number of nearest neighbors per image.
            random_state (int): Random seed used by K-Means.
        """

        if not isinstance(neighbors, int) or neighbors < 1:
            raise ValueError(
                "neighbors must be a positive integer."
            )

        self.neighbors = neighbors
        self.random_state = random_state

    # --------------------------------------------------
    # Coordinate validation
    # --------------------------------------------------

    def _get_coordinates(self, df):
        """
        Extract and validate projected image center coordinates.

        Returns:
            numpy.ndarray: Coordinates with shape (n_samples, 2).
        """

        required_columns = ["center_x", "center_y"]

        missing = [
            column
            for column in required_columns
            if column not in df.columns
        ]

        if missing:
            raise ValueError(
                f"Missing coordinate columns: {missing}"
            )

        coordinates = df[
            required_columns
        ].to_numpy(dtype=np.float64)

        if not np.isfinite(coordinates).all():
            raise ValueError(
                "Coordinates contain missing or invalid values."
            )

        return coordinates

    # --------------------------------------------------
    # Nearest geographic neighbors
    # --------------------------------------------------

    def find_nearest_neighbors(self, df):
        """
        Find the nearest geographic neighbors of every image.

        Each image is excluded from its own neighbor list.

        Returns:
            pd.DataFrame: Nearest-neighbor relationships and distances.
        """

        coordinates = self._get_coordinates(df)

        if len(df) < 2:
            raise ValueError(
                "At least two images are required."
            )

        n_neighbors = min(
            self.neighbors,
            len(df) - 1
        )

        model = NearestNeighbors(
            n_neighbors=n_neighbors,
            metric="euclidean"
        )

        model.fit(coordinates)

        # Omitting X makes sklearn query the training samples
        # while excluding each sample itself.
        distances, indices = model.kneighbors()

        records = []

        for image_idx in range(len(df)):

            image_row = df.iloc[image_idx]

            for neighbor_position in range(n_neighbors):

                neighbor_idx = indices[
                    image_idx,
                    neighbor_position
                ]

                neighbor_row = df.iloc[neighbor_idx]

                records.append({
                    "path": image_row["path"],
                    "neighbor_path": neighbor_row["path"],
                    "image": image_row["image"],
                    "neighbor": neighbor_row["image"],
                    "class_name": image_row["class_name"],
                    "neighbor_class": neighbor_row["class_name"],
                    "neighbor_position": neighbor_position + 1,
                    "distance_m": float(
                        distances[
                            image_idx,
                            neighbor_position
                        ]
                    )
                })

        return pd.DataFrame(records)

    # --------------------------------------------------
    # Unique nearest-neighbor pairs
    # --------------------------------------------------

    def find_closest_unique_pairs(self, df, limit=None):
        """
        Find unique unordered pairs among nearest-neighbor relationships.

        Each pair is included only once, regardless of direction.

        Args:
            df (pd.DataFrame): Image metadata and coordinates.
            limit (int, optional): Maximum number of pairs to return.

        Returns:
            pd.DataFrame: Unique pairs sorted by distance.
        """

        if limit is not None and limit < 1:
            raise ValueError(
                "limit must be a positive integer or None."
            )

        neighbors_df = self.find_nearest_neighbors(df)

        records = []
        seen_pairs = set()

        for _, row in neighbors_df.iterrows():

            path_1 = row["path"]
            path_2 = row["neighbor_path"]

            # Canonical key prevents duplicate unordered pairs.
            pair_key = tuple(sorted([path_1, path_2]))

            if pair_key in seen_pairs:
                continue

            seen_pairs.add(pair_key)

            records.append({
                "path_1": path_1,
                "path_2": path_2,
                "image_1": row["image"],
                "image_2": row["neighbor"],
                "class_1": row["class_name"],
                "class_2": row["neighbor_class"],
                "distance_m": row["distance_m"]
            })

        columns = [
            "path_1",
            "path_2",
            "image_1",
            "image_2",
            "class_1",
            "class_2",
            "distance_m"
        ]

        result = pd.DataFrame(
            records,
            columns=columns
        )

        result = (
            result
            .sort_values("distance_m")
            .reset_index(drop=True)
        )

        if limit is not None:
            result = result.head(limit)

        return result

    # --------------------------------------------------
    # Distance statistics
    # --------------------------------------------------

    def calculate_distance_statistics(self, neighbors_df):
        """
        Calculate descriptive distance statistics for each neighbor rank.

        Returns:
            pd.DataFrame: Minimum, mean, median, maximum and standard
                          deviation of geographic distances.
        """

        required_columns = [
            "neighbor_position",
            "distance_m"
        ]

        if not all(
            column in neighbors_df.columns
            for column in required_columns
        ):
            raise ValueError(
                "Missing columns required for distance statistics."
            )

        if neighbors_df.empty:
            raise ValueError(
                "Cannot calculate statistics from an empty DataFrame."
            )

        return (
            neighbors_df
            .groupby("neighbor_position")["distance_m"]
            .agg([
                "min",
                "mean",
                "median",
                "max",
                "std"
            ])
        )

    # --------------------------------------------------
    # Geographic footprint overlap
    # --------------------------------------------------

    def find_overlapping_patches(
        self,
        candidate_pairs,
        footprints
    ):
        """
        Calculate geometric overlap between candidate image pairs.

        Both footprints must be expressed in the same projected CRS,
        with coordinates measured in metres.

        Args:
            candidate_pairs (pd.DataFrame):
                Candidate pairs containing path_1 and path_2.
            footprints (dict):
                Mapping of image paths to Shapely geometries.

        Returns:
            pd.DataFrame:
                Pairs with positive intersection area and overlap ratios.
        """

        required_columns = [
            "path_1",
            "path_2",
            "image_1",
            "image_2",
            "distance_m"
        ]

        missing = [
            column
            for column in required_columns
            if column not in candidate_pairs.columns
        ]

        if missing:
            raise ValueError(
                f"Missing candidate-pair columns: {missing}"
            )

        records = []

        for _, pair in candidate_pairs.iterrows():

            path_1 = pair["path_1"]
            path_2 = pair["path_2"]

            geometry_1 = footprints.get(path_1)
            geometry_2 = footprints.get(path_2)

            if geometry_1 is None or geometry_2 is None:
                continue

            if geometry_1.is_empty or geometry_2.is_empty:
                continue

            if not geometry_1.is_valid or not geometry_2.is_valid:
                continue

            # Skip geometries that do not intersect.
            if not geometry_1.intersects(geometry_2):
                continue

            intersection = geometry_1.intersection(
                geometry_2
            )

            overlap_area = intersection.area

            # Touching boundaries have zero intersection area.
            if overlap_area <= 0:
                continue

            area_1 = geometry_1.area
            area_2 = geometry_2.area

            if area_1 <= 0 or area_2 <= 0:
                continue

            # Percentage of each image covered by the intersection.
            overlap_pct_image_1 = (
                overlap_area / area_1
            ) * 100

            overlap_pct_image_2 = (
                overlap_area / area_2
            ) * 100

            # Intersection over Union.
            union_area = area_1 + area_2 - overlap_area

            iou = (
                overlap_area / union_area
                if union_area > 0
                else 0.0
            )

            records.append({
                "path_1": path_1,
                "path_2": path_2,
                "image_1": pair["image_1"],
                "image_2": pair["image_2"],
                "distance_m": pair["distance_m"],
                "area_1_m2": area_1,
                "area_2_m2": area_2,
                "overlap_area_m2": overlap_area,
                "overlap_pct_image_1": overlap_pct_image_1,
                "overlap_pct_image_2": overlap_pct_image_2,
                "intersection_over_union": iou
            })

        columns = [
            "path_1",
            "path_2",
            "image_1",
            "image_2",
            "distance_m",
            "area_1_m2",
            "area_2_m2",
            "overlap_area_m2",
            "overlap_pct_image_1",
            "overlap_pct_image_2",
            "intersection_over_union"
        ]

        return pd.DataFrame(
            records,
            columns=columns
        )

    # --------------------------------------------------
    # Geographic clustering
    # --------------------------------------------------

    def cluster_geographic_locations(
        self,
        df,
        n_clusters=5
    ):
        """
        Apply K-Means to projected image center coordinates.

        Cluster labels represent exploratory geographic groups,
        not land-cover classes or predefined geographic regions.

        Returns:
            pd.DataFrame: Original metadata with an additional
                          cluster column.
        """

        coordinates = self._get_coordinates(df)

        if not 1 <= n_clusters <= len(df):
            raise ValueError(
                "Invalid number of clusters."
            )

        model = KMeans(
            n_clusters=n_clusters,
            random_state=self.random_state,
            n_init=10
        )

        labels = model.fit_predict(coordinates)

        result = df.copy()
        result["cluster"] = labels

        return result

    # --------------------------------------------------
    # Distances between dataset splits
    # --------------------------------------------------

    def analyze_split_distances(
        self,
        df,
        split_column="split"
    ):
        """
        Calculate the nearest geographic distance from every image
        to each dataset split.

        Self-matches are excluded when the source image and target
        split are the same.

        Args:
            df (pd.DataFrame):
                Image metadata, coordinates and split assignments.
            split_column (str):
                Column containing train, validation and test labels.

        Returns:
            pd.DataFrame:
                Nearest distance and nearest target image for every
                source image and target split.
        """

        if split_column not in df.columns:
            raise ValueError(
                f"Missing column: {split_column}"
            )

        coordinates = self._get_coordinates(df)

        if df[split_column].isna().any():
            raise ValueError(
                f"Column '{split_column}' contains missing values."
            )

        split_values = df[split_column].unique()

        records = []

        for target_split in split_values:

            # Original row indices belonging to the target split.
            target_indices = np.flatnonzero(
                df[split_column].to_numpy() == target_split
            )

            if len(target_indices) == 0:
                continue

            target_coordinates = coordinates[
                target_indices
            ]

            model = NearestNeighbors(
                n_neighbors=min(2, len(target_indices)),
                metric="euclidean"
            )

            model.fit(target_coordinates)

            distances, local_indices = model.kneighbors(
                coordinates
            )

            for source_idx in range(len(df)):

                source_row = df.iloc[source_idx]

                same_split = (
                    source_row[split_column] == target_split
                )

                nearest_position = None

                for position, local_idx in enumerate(
                    local_indices[source_idx]
                ):

                    target_idx = target_indices[local_idx]

                    # Exclude the image itself from its own split.
                    if same_split and target_idx == source_idx:
                        continue

                    nearest_position = position
                    break

                if nearest_position is None:

                    records.append({
                        "image": source_row["image"],
                        "path": source_row["path"],
                        "source_split": source_row[split_column],
                        "target_split": target_split,
                        "nearest_target_image": None,
                        "nearest_target_path": None,
                        "nearest_distance_m": np.nan
                    })

                    continue

                target_idx = target_indices[
                    local_indices[
                        source_idx,
                        nearest_position
                    ]
                ]

                target_row = df.iloc[target_idx]

                records.append({
                    "image": source_row["image"],
                    "path": source_row["path"],
                    "source_split": source_row[split_column],
                    "target_split": target_split,
                    "nearest_target_image": target_row["image"],
                    "nearest_target_path": target_row["path"],
                    "nearest_distance_m": float(
                        distances[
                            source_idx,
                            nearest_position
                        ]
                    )
                })

        return pd.DataFrame(records)
