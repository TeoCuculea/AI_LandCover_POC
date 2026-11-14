
from pathlib import Path

from shapely.geometry import shape

from src.config import (
    DATA_DIR,
    RESULTS_DIR,
    COORDINATES_FILE,
    CLOSEST_PAIRS_FILE,
    OVERLAPPING_PAIRS_FILE,
    CLUSTERS_FILE,
    MAP_FILE,
    DEFAULT_CLUSTER_COUNT,
    DEFAULT_NEIGHBORS,
    MAX_OVERLAP_CANDIDATES,
    RANDOM_STATE,
)

from src.geospatial.coordinates import CoordinateTransformer
from src.geospatial.spatial_analysis import SpatialAnalyzer
from src.geospatial.visualization import GeoVisualizer


def discover_images(data_dir):
    """
    Discover all TIFF images in the dataset directory.
    """

    data_dir = Path(data_dir)

    image_paths = sorted(
        data_dir.rglob("*.tif")
    )

    if not image_paths:
        raise FileNotFoundError(
            f"No TIFF images found in {data_dir}"
        )

    return image_paths


def main():
    """
    Execute the complete geospatial analysis workflow.
    """

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print("=" * 60)
    print("EUROSAT GEOSPATIAL ANALYSIS")
    print("=" * 60)

    # --------------------------------------------------
    # 1. Coordinate processing
    # --------------------------------------------------

    coordinate_transformer = CoordinateTransformer()

    if COORDINATES_FILE.exists():

        print("\nLoading existing coordinates...")

        coordinates_df = (
            coordinate_transformer
            .load_coordinates(COORDINATES_FILE)
        )

    else:

        print("\nDiscovering TIFF images...")

        image_paths = discover_images(DATA_DIR)

        print(f"Images discovered: {len(image_paths)}")

        coordinates_df = (
            coordinate_transformer
            .process_dataset(image_paths)
        )

    coordinates_df = (
        coordinate_transformer
        .add_geographic_coordinates(coordinates_df)
    )

    coordinate_transformer.save_coordinates(
        coordinates_df,
        COORDINATES_FILE
    )

    print(
        f"\nImages processed: {len(coordinates_df)}"
    )

    # --------------------------------------------------
    # 2. Nearest-neighbor analysis
    # --------------------------------------------------

    spatial_analyzer = SpatialAnalyzer(
        neighbors=DEFAULT_NEIGHBORS,
        random_state=RANDOM_STATE,
    )

    print("\nAnalyzing nearest neighbors...")

    neighbors_df = (
        spatial_analyzer
        .find_nearest_neighbors(coordinates_df)
    )

    statistics_df = (
        spatial_analyzer
        .calculate_distance_statistics(neighbors_df)
    )

    print("\nDistance statistics:")
    print(statistics_df)

    # --------------------------------------------------
    # 3. Closest unique pairs
    # --------------------------------------------------

    closest_pairs_df = (
        spatial_analyzer
        .find_closest_unique_pairs(
            coordinates_df,
            limit=MAX_OVERLAP_CANDIDATES,
        )
    )

    closest_pairs_df.to_csv(
        CLOSEST_PAIRS_FILE,
        index=False
    )

    print("\nClosest image pairs:")
    print(
        closest_pairs_df.head(10).to_string(index=False)
    )

    # --------------------------------------------------
    # 4. Geographic clustering
    # --------------------------------------------------

    print("\nRunning K-Means...")

    clusters_df = (
        spatial_analyzer
        .cluster_geographic_locations(
            coordinates_df,
            n_clusters=DEFAULT_CLUSTER_COUNT
        )
    )

    clusters_df.to_csv(
        CLUSTERS_FILE,
        index=False
    )

    print("\nCluster distribution:")
    print(
        clusters_df["cluster"]
        .value_counts()
        .sort_index()
    )

    # --------------------------------------------------
    # 5. Candidate footprint overlap analysis
    # --------------------------------------------------

    print("\nChecking candidate footprint overlaps...")

    candidate_pairs = closest_pairs_df.head(MAX_OVERLAP_CANDIDATES)

    candidate_paths = set(
        candidate_pairs["path_1"]
    ) | set(
        candidate_pairs["path_2"]
    )

    footprints = {}

    for image_path in candidate_paths:

        geometry = (
            coordinate_transformer
            .transform_image_footprint(image_path)
        )

        footprints[image_path] = shape(geometry)

    overlaps_df = (
        spatial_analyzer
        .find_overlapping_patches(
            candidate_pairs,
            footprints
        )
    )

    overlaps_df.to_csv(
        OVERLAPPING_PAIRS_FILE,
        index=False
    )

    print(
        f"Candidate overlaps: {len(overlaps_df)}"
    )

    # --------------------------------------------------
    # 6. Static visualizations
    # --------------------------------------------------

    visualizer = GeoVisualizer()

    visualizer.plot_europe_distribution(
        coordinates_df
    )

    visualizer.plot_geographic_clusters(
        clusters_df
    )

    visualizer.plot_nearest_neighbor_pairs(
        coordinates_df,
        closest_pairs_df
    )

    # --------------------------------------------------
    # 7. Interactive map
    # --------------------------------------------------

    visualizer.create_interactive_map(
        coordinates_df,
        closest_pairs_df,
        coordinate_transformer,
        output_path=MAP_FILE
    )

    # --------------------------------------------------
    # 8. Final summary
    # --------------------------------------------------

    print("\n" + "=" * 60)
    print("GEOSPATIAL ANALYSIS COMPLETED")
    print("=" * 60)

    print(f"Images: {len(coordinates_df)}")
    print(f"Candidate pairs: {len(closest_pairs_df)}")
    print(f"Candidate overlaps: {len(overlaps_df)}")

    print(f"\nCoordinates: {COORDINATES_FILE}")
    print(f"Pairs: {CLOSEST_PAIRS_FILE}")
    print(f"Overlaps: {OVERLAPPING_PAIRS_FILE}")
    print(f"Clusters: {CLUSTERS_FILE}")
    print(f"Map: {MAP_FILE}")


if __name__ == "__main__":
    main()
