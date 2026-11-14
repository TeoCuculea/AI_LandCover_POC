
from pathlib import Path

import folium
import matplotlib.pyplot as plt

from folium.plugins import FastMarkerCluster
from rasterio.warp import transform_geom
from shapely.geometry import shape

from src.config import (
    FIGURES_DIR,
    MAP_FILE,
    TARGET_CRS,
)


class GeoVisualizer:
    """
    Generate static geographic plots and interactive maps
    for the EuroSAT dataset.
    """

    def __init__(self, output_dir=FIGURES_DIR):
        """
        Initialize the visualizer and output directory.
        """

        self.output_dir = Path(output_dir)

        self.output_dir.mkdir(
            parents=True,
            exist_ok=True
        )

    def plot_europe_distribution(self, df):
        """
        Plot all image centers in the projected coordinate system.
        """

        fig, ax = plt.subplots(figsize=(12, 9))

        ax.scatter(
            df["center_x"],
            df["center_y"],
            s=3,
            alpha=0.5
        )

        ax.set_title(
            "Geographic Distribution of EuroSAT Images"
        )

        ax.set_xlabel(f"Easting ({TARGET_CRS})")
        ax.set_ylabel(f"Northing ({TARGET_CRS})")

        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)

        output_path = (
            self.output_dir / "eurosat_distribution.png"
        )

        fig.savefig(
            output_path,
            dpi=200,
            bbox_inches="tight"
        )

        plt.close(fig)

        print(f"Figure saved: {output_path}")

    def plot_geographic_clusters(self, df):
        """
        Plot geographic locations colored by K-Means cluster.
        """

        if "cluster" not in df.columns:
            raise ValueError(
                "Missing cluster column."
            )

        fig, ax = plt.subplots(figsize=(12, 9))

        scatter = ax.scatter(
            df["center_x"],
            df["center_y"],
            c=df["cluster"],
            cmap="tab10",
            s=5,
            alpha=0.6
        )

        ax.set_title("Geographic K-Means Clusters")
        ax.set_xlabel(f"Easting ({TARGET_CRS})")
        ax.set_ylabel(f"Northing ({TARGET_CRS})")

        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)

        fig.colorbar(
            scatter,
            ax=ax,
            label="Cluster"
        )

        output_path = (
            self.output_dir / "eurosat_clusters.png"
        )

        fig.savefig(
            output_path,
            dpi=200,
            bbox_inches="tight"
        )

        plt.close(fig)

        print(f"Figure saved: {output_path}")

    def plot_nearest_neighbor_pairs(
        self,
        df,
        pairs_df,
        limit=20
    ):
        """
        Plot selected nearest-neighbor image pairs.
        """

        fig, ax = plt.subplots(figsize=(12, 9))

        ax.scatter(
            df["center_x"],
            df["center_y"],
            s=2,
            alpha=0.15
        )

        image_lookup = df.set_index("path")

        for _, pair in pairs_df.head(limit).iterrows():

            image_1 = image_lookup.loc[pair["path_1"]]
            image_2 = image_lookup.loc[pair["path_2"]]

            ax.plot(
                [
                    image_1["center_x"],
                    image_2["center_x"]
                ],
                [
                    image_1["center_y"],
                    image_2["center_y"]
                ],
                linewidth=1
            )

            ax.scatter(
                image_1["center_x"],
                image_1["center_y"],
                s=20
            )

            ax.scatter(
                image_2["center_x"],
                image_2["center_y"],
                s=20
            )

        ax.set_title("Nearest Geographic Image Pairs")
        ax.set_xlabel(f"Easting ({TARGET_CRS})")
        ax.set_ylabel(f"Northing ({TARGET_CRS})")

        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)

        output_path = (
            self.output_dir / "nearest_neighbor_pairs.png"
        )

        fig.savefig(
            output_path,
            dpi=200,
            bbox_inches="tight"
        )

        plt.close(fig)

        print(f"Figure saved: {output_path}")

    def create_interactive_map(
        self,
        df,
        closest_pairs,
        coordinate_transformer,
        output_path=MAP_FILE
    ):
        """
        Generate an interactive OpenStreetMap.

        All image centers are displayed as clustered markers.
        The closest pair is highlighted with its raster footprint.
        """

        if df.empty:
            raise ValueError(
                "No geographic coordinates available."
            )

        center_lat = df["latitude"].mean()
        center_lon = df["longitude"].mean()

        map_object = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=5,
            tiles="OpenStreetMap",
            control_scale=True
        )

        # All image centers
        locations = df[
            ["latitude", "longitude"]
        ].values.tolist()

        marker_group = folium.FeatureGroup(
            name="All EuroSAT image centers"
        )

        FastMarkerCluster(
            data=locations
        ).add_to(marker_group)

        marker_group.add_to(map_object)

        # Closest pair
        if not closest_pairs.empty:

            pair = closest_pairs.iloc[0]

            pair_group = folium.FeatureGroup(
                name="Closest image pair",
                show=True
            )

            image_lookup = df.set_index("path")

            pair_locations = []

            image_paths = [
                pair["path_1"],
                pair["path_2"]
            ]

            colors = ["red", "blue"]

            for image_path, color in zip(
                image_paths,
                colors
            ):

                image_row = image_lookup.loc[image_path]

                latitude = image_row["latitude"]
                longitude = image_row["longitude"]

                pair_locations.append(
                    [latitude, longitude]
                )

                folium.Marker(
                    location=[latitude, longitude],
                    tooltip=image_row["image"],
                    popup=(
                        f"Image: {image_row['image']}<br>"
                        f"Class: {image_row['class_name']}"
                    ),
                    icon=folium.Icon(
                        color=color,
                        icon="info-sign"
                    )
                ).add_to(pair_group)

                footprint = (
                    coordinate_transformer
                    .transform_image_footprint(image_path)
                )

                geographic_geometry = transform_geom(
                    TARGET_CRS,
                    "EPSG:4326",
                    footprint
                )

                polygon = shape(geographic_geometry)

                polygons = (
                    list(polygon.geoms)
                    if polygon.geom_type == "MultiPolygon"
                    else [polygon]
                )

                for part in polygons:

                    locations_polygon = [
                        [lat, lon]
                        for lon, lat in part.exterior.coords
                    ]

                    folium.Polygon(
                        locations=locations_polygon,
                        color=color,
                        weight=3,
                        fill=True,
                        fill_color=color,
                        fill_opacity=0.25,
                        tooltip=image_row["image"]
                    ).add_to(pair_group)

            folium.PolyLine(
                locations=pair_locations,
                color="black",
                weight=2,
                tooltip=(
                    f"Center distance: "
                    f"{pair['distance_m']:.2f} m"
                )
            ).add_to(pair_group)

            pair_group.add_to(map_object)

        folium.LayerControl().add_to(map_object)

        output_path = Path(output_path)

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        map_object.save(str(output_path))

        print(f"Interactive map saved: {output_path}")
