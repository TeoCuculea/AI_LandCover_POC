
from pathlib import Path


# Project directories
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = (
    PROJECT_ROOT
    / "data"
    / "EuroSAT_MS"
    / "EuroSAT_MS"
)

RESULTS_DIR = PROJECT_ROOT / "results"
RUNS_DIR = RESULTS_DIR / "runs"

FIGURES_DIR = RESULTS_DIR / "figures"

TABLES_DIR = RESULTS_DIR / "tables"

RANDOM_SPLIT_FILE = TABLES_DIR / "dataset_splits.csv"
SPATIAL_SPLIT_FILE = TABLES_DIR / "spatial_dataset_splits.csv"
NORMALIZATION_FILE = TABLES_DIR / "normalization_statistics.json"


# Coordinate reference systems
TARGET_CRS = "EPSG:3035"
MAP_CRS = "EPSG:4326"


# Dataset configuration
NUM_CLASSES = 10
NUM_BANDS = 13
IMAGE_SIZE = 64

BAND_NAMES = [
    "B01", "B02", "B03", "B04", "B05",
    "B06", "B07", "B08", "B8A", "B09",
    "B10", "B11", "B12"
]

RGB_BANDS = [3, 2, 1]


# Dataset split configuration
TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.15
TEST_RATIO = 0.15

RANDOM_STATE = 42
SPLIT_NAMES = ("train", "validation", "test")
SPLIT_STRATEGIES = ("random", "spatial")
REPRESENTATIONS = ("rgb", "multispectral")


# Spatial analysis configuration
DEFAULT_NEIGHBORS = 2
DEFAULT_CLUSTER_COUNT = 5

# Candidate pairs considered for footprint analysis.
MAX_OVERLAP_CANDIDATES = 100


# Output files
COORDINATES_FILE = (
    RESULTS_DIR / "eurosat_geographic_coordinates.csv"
)

CLOSEST_PAIRS_FILE = (
    RESULTS_DIR / "closest_geographic_pairs.csv"
)

OVERLAPPING_PAIRS_FILE = (
    RESULTS_DIR / "overlapping_image_pairs.csv"
)

CLUSTERS_FILE = (
    RESULTS_DIR / "geographic_clusters.csv"
)

MAP_FILE = RESULTS_DIR / "eurosat_map.html"
