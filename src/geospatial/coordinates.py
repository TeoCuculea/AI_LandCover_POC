
from pathlib import Path

import pandas as pd
import rasterio

from pyproj import Transformer
from shapely.geometry import box, mapping
from rasterio.warp import transform_geom

from src.config import TARGET_CRS


class CoordinateTransformer:
    """
    Extract and transform geographic metadata from
    EuroSAT multispectral TIFF images.

    All image centers and footprints are transformed
    into a common projected coordinate reference system.
    """

    def __init__(self, target_crs=TARGET_CRS):
        """
        Initialize the coordinate transformer.

        Args:
            target_crs: Target projected CRS.
        """

        self.target_crs = target_crs

    def extract_image_metadata(self, image_path):
        """
        Extract spatial metadata from a TIFF image.

        Returns:
            dict: CRS, bounds, resolution, dimensions,
            number of bands, and source center coordinates.
        """

        image_path = Path(image_path)

        with rasterio.open(image_path) as src:

            if src.crs is None:
                raise ValueError(
                    f"Missing CRS: {image_path}"
                )

            bounds = src.bounds

            center_x = (bounds.left + bounds.right) / 2
            center_y = (bounds.bottom + bounds.top) / 2

            return {
                "image": image_path.name,
                "path": str(image_path.resolve()),
                "class_name": image_path.parent.name,
                "crs": src.crs.to_string(),

                "left": bounds.left,
                "bottom": bounds.bottom,
                "right": bounds.right,
                "top": bounds.top,

                "resolution_x": src.res[0],
                "resolution_y": src.res[1],

                "width": src.width,
                "height": src.height,
                "bands": src.count,

                "center_x_source": center_x,
                "center_y_source": center_y
            }

    def transform_coordinates(self, x, y, source_crs):
        """
        Transform coordinates from source CRS to target CRS.

        Returns:
            tuple: Transformed X and Y coordinates.
        """

        transformer = Transformer.from_crs(
            source_crs,
            self.target_crs,
            always_xy=True
        )

        return transformer.transform(x, y)

    def transform_image_footprint(self, image_path):
        """
        Transform the image bounding polygon into target CRS.

        Returns:
            dict: Transformed GeoJSON geometry.
        """

        with rasterio.open(image_path) as src:

            if src.crs is None:
                raise ValueError(
                    f"Missing CRS: {image_path}"
                )

            source_polygon = box(*src.bounds)

            return transform_geom(
                src.crs,
                self.target_crs,
                mapping(source_polygon)
            )

    def process_image(self, image_path):
        """
        Extract metadata and transform the image center.
        """

        metadata = self.extract_image_metadata(image_path)

        center_x, center_y = self.transform_coordinates(
            metadata["center_x_source"],
            metadata["center_y_source"],
            metadata["crs"]
        )

        metadata["center_x"] = center_x
        metadata["center_y"] = center_y

        return metadata

    def process_dataset(self, image_paths):
        """
        Process all image paths.

        Returns:
            pandas.DataFrame: Geographic metadata.
        """

        image_paths = list(image_paths)

        records = []

        for index, image_path in enumerate(
            image_paths,
            start=1
        ):

            records.append(
                self.process_image(image_path)
            )

            if index % 1000 == 0:
                print(
                    f"Processed {index}/{len(image_paths)} images"
                )

        return pd.DataFrame(records)

    def add_geographic_coordinates(self, df):
        """
        Add longitude and latitude for map visualization.

        Input center coordinates must be in target CRS.
        """

        transformer = Transformer.from_crs(
            self.target_crs,
            "EPSG:4326",
            always_xy=True
        )

        longitude, latitude = transformer.transform(
            df["center_x"].values,
            df["center_y"].values
        )

        result = df.copy()

        result["longitude"] = longitude
        result["latitude"] = latitude

        return result

    def save_coordinates(self, df, output_path):
        """
        Save geographic metadata to CSV.
        """

        output_path = Path(output_path)

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        df.to_csv(output_path, index=False)

        print(f"Coordinates saved: {output_path}")

    def load_coordinates(self, input_path):
        """
        Load previously calculated geographic metadata.
        """

        return pd.read_csv(input_path)
