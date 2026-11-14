"""Create the interactive EuroSAT map from cached spatial results."""

import pandas as pd

from src.config import CLOSEST_PAIRS_FILE, COORDINATES_FILE, MAP_FILE
from src.geospatial.coordinates import CoordinateTransformer
from src.geospatial.visualization import GeoVisualizer


def main() -> None:
    """Load cached coordinates and render the interactive map."""
    if not COORDINATES_FILE.is_file():
        raise FileNotFoundError(
            f"Coordinates not found at {COORDINATES_FILE}. "
            "Run `python -m src.geospatial.run_analysis` first."
        )
    coordinates = pd.read_csv(COORDINATES_FILE)
    if CLOSEST_PAIRS_FILE.is_file():
        closest_pairs = pd.read_csv(CLOSEST_PAIRS_FILE)
    else:
        closest_pairs = pd.DataFrame(
            columns=["path_1", "path_2", "image_1", "image_2", "distance_m"]
        )
    GeoVisualizer().create_interactive_map(
        coordinates,
        closest_pairs,
        CoordinateTransformer(),
        output_path=MAP_FILE,
    )
    print(f"Interactive map saved: {MAP_FILE}")


if __name__ == "__main__":
    main()
