
from pathlib import Path

import pandas as pd


from src.config import RESULTS_DIR, TABLES_DIR

UNIQUE_IMAGES_FILE = TABLES_DIR / "spatially_affected_images.csv"
TOP_PAIRS_FILE = TABLES_DIR / "top_spatial_overlaps.csv"


def find_input_file() -> Path:
    """
    Locate the spatial overlap report inside the results directory.
    """

    matches = list(
        RESULTS_DIR.rglob("spatial_leakage_pairs.csv")
    )

    if not matches:
        raise FileNotFoundError(
            f"Could not find spatial_leakage_pairs.csv inside {RESULTS_DIR}"
        )

    if len(matches) > 1:
        print("Multiple matching files found:")
        for path in matches:
            print(f"  {path}")

        print("\nUsing the first matching file.")

    return matches[0]


def inspect_spatial_overlaps() -> None:
    """
    Inspect cross-split spatial overlaps.

    Calculates the number of unique affected images per split,
    summarizes affected images by class, and identifies the
    image pairs with the largest intersection-over-union.
    """

    print("=" * 65)
    print("EUROSAT SPATIAL OVERLAP INSPECTION")
    print("=" * 65)

    input_file = find_input_file()

    print(f"\nReading input file:\n{input_file}")

    overlaps = pd.read_csv(input_file)

    if overlaps.empty:
        print("\nNo spatial overlaps were found.")
        return

    required_columns = {
        "image_1",
        "image_2",
        "path_1",
        "path_2",
        "class_1",
        "class_2",
        "split_1",
        "split_2",
        "overlap_area_m2",
        "overlap_pct_image_1",
        "overlap_pct_image_2",
        "intersection_over_union",
    }

    missing_columns = required_columns - set(overlaps.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    # Collect images from both sides of every overlapping pair.
    images_1 = overlaps[
        ["path_1", "image_1", "class_1", "split_1"]
    ].rename(
        columns={
            "path_1": "image_path",
            "image_1": "image_name",
            "class_1": "class_name",
            "split_1": "split",
        }
    )

    images_2 = overlaps[
        ["path_2", "image_2", "class_2", "split_2"]
    ].rename(
        columns={
            "path_2": "image_path",
            "image_2": "image_name",
            "class_2": "class_name",
            "split_2": "split",
        }
    )

    affected_images = pd.concat(
        [images_1, images_2],
        ignore_index=True,
    )

    # Remove duplicate image entries.
    affected_images = affected_images.drop_duplicates(
        subset=["image_path"]
    ).copy()

    # Count unique affected images per split.
    print("\n" + "=" * 65)
    print("UNIQUE AFFECTED IMAGES BY SPLIT")
    print("=" * 65)

    split_totals = (
        affected_images.groupby("split")["image_path"]
        .nunique()
        .sort_index()
    )

    for split, count in split_totals.items():
        print(f"{split}: {count} unique images")

    print(f"\nTotal unique affected images: {len(affected_images)}")

    # Summarize affected images by split and class.
    class_summary = (
        affected_images.groupby(["split", "class_name"])
        .size()
        .reset_index(name="affected_images")
        .sort_values(
            ["split", "affected_images"],
            ascending=[True, False],
        )
    )

    print("\n" + "=" * 65)
    print("AFFECTED IMAGES BY CLASS")
    print("=" * 65)

    print(class_summary.to_string(index=False))

    # Identify the 20 pairs with the highest IoU.
    top_pairs = overlaps.sort_values(
        "intersection_over_union",
        ascending=False,
    ).head(20).copy()

    display_columns = [
        "image_1",
        "image_2",
        "class_1",
        "class_2",
        "split_1",
        "split_2",
        "overlap_area_m2",
        "overlap_pct_image_1",
        "overlap_pct_image_2",
        "intersection_over_union",
    ]

    print("\n" + "=" * 65)
    print("TOP 20 SPATIAL OVERLAPS")
    print("=" * 65)

    print(top_pairs[display_columns].to_string(index=False))

    # Save the reports.
    UNIQUE_IMAGES_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    affected_images.to_csv(
        UNIQUE_IMAGES_FILE,
        index=False,
    )

    top_pairs.to_csv(
        TOP_PAIRS_FILE,
        index=False,
    )

    print("\n" + "=" * 65)
    print("OUTPUT FILES")
    print("=" * 65)

    print(f"Unique affected images:\n{UNIQUE_IMAGES_FILE}")
    print(f"\nTop overlapping pairs:\n{TOP_PAIRS_FILE}")

    print("\nInspection completed successfully.")


if __name__ == "__main__":
    inspect_spatial_overlaps()
