"""Cross-split geographic footprint overlap analysis."""

from pathlib import Path

import pandas as pd
import rasterio
from rasterio.warp import transform_geom
from shapely.geometry import Polygon, shape
from shapely.strtree import STRtree

from src.config import TABLES_DIR, TARGET_CRS


def analyze_split_overlaps(splits: pd.DataFrame):
    """Find positive-area overlaps between images assigned to different splits.

    Writes the same detailed and summary CSV artifacts used by the original
    dataset preparation workflow, and returns both tables.
    """
    required = {"image", "image_path", "label", "split"}
    missing = required - set(splits.columns)
    if missing:
        raise ValueError(f"Missing split columns: {sorted(missing)}")

    geometries, records = [], []
    for index, row in splits.reset_index(drop=True).iterrows():
        image_path = Path(row["image_path"])
        with rasterio.open(image_path) as source:
            if source.crs is None:
                raise ValueError(f"Missing CRS: {image_path}")
            footprint = Polygon([
                (source.bounds.left, source.bounds.bottom),
                (source.bounds.right, source.bounds.bottom),
                (source.bounds.right, source.bounds.top),
                (source.bounds.left, source.bounds.top),
            ])
            projected = shape(transform_geom(source.crs, TARGET_CRS, footprint.__geo_interface__))
        if projected.is_empty or not projected.is_valid:
            print(f"Skipping invalid footprint: {image_path.name}")
            continue
        geometries.append(projected)
        records.append({
            "image": row["image"],
            "image_path": str(image_path.resolve()),
            "label": row["label"],
            "split": row["split"],
        })
        if (index + 1) % 1000 == 0:
            print(f"Footprint processing: {index + 1}/{len(splits)}")

    if not geometries:
        raise ValueError("No valid image footprints found.")

    overlap_records = []
    tree = STRtree(geometries)
    for i, first in enumerate(geometries):
        for candidate in tree.query(first):
            j = int(candidate)
            if j <= i or records[i]["split"] == records[j]["split"]:
                continue
            second = geometries[j]
            intersection_area = first.intersection(second).area
            if intersection_area <= 0:
                continue
            area_1, area_2 = first.area, second.area
            union_area = first.union(second).area
            overlap_records.append({
                "image_1": records[i]["image"],
                "image_2": records[j]["image"],
                "path_1": records[i]["image_path"],
                "path_2": records[j]["image_path"],
                "class_1": records[i]["label"],
                "class_2": records[j]["label"],
                "split_1": records[i]["split"],
                "split_2": records[j]["split"],
                "distance_m": first.distance(second),
                "overlap_area_m2": intersection_area,
                "area_1_m2": area_1,
                "area_2_m2": area_2,
                "overlap_pct_image_1": intersection_area / area_1 * 100,
                "overlap_pct_image_2": intersection_area / area_2 * 100,
                "intersection_over_union": intersection_area / union_area,
            })

    columns = [
        "image_1", "image_2", "path_1", "path_2", "class_1", "class_2",
        "split_1", "split_2", "distance_m", "overlap_area_m2", "area_1_m2",
        "area_2_m2", "overlap_pct_image_1", "overlap_pct_image_2",
        "intersection_over_union",
    ]
    overlaps = pd.DataFrame(overlap_records, columns=columns)
    split_pairs = (("train", "validation"), ("train", "test"), ("validation", "test"))
    summaries = []
    for split_1, split_2 in split_pairs:
        if overlaps.empty:
            subset = overlaps
        else:
            mask = (
                (overlaps["split_1"].eq(split_1) & overlaps["split_2"].eq(split_2))
                | (overlaps["split_1"].eq(split_2) & overlaps["split_2"].eq(split_1))
            )
            subset = overlaps.loc[mask]
        summaries.append({
            "split_pair": f"{split_1}-{split_2}",
            "overlapping_pairs": len(subset),
            "mean_iou": subset["intersection_over_union"].mean() if not subset.empty else 0,
            "maximum_iou": subset["intersection_over_union"].max() if not subset.empty else 0,
        })

    summary = pd.DataFrame(summaries)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    overlaps.to_csv(TABLES_DIR / "spatial_leakage_pairs.csv", index=False)
    summary.to_csv(TABLES_DIR / "spatial_leakage_summary.csv", index=False)
    print("\nSpatial leakage summary:")
    print(summary.to_string(index=False))
    print(f"\nOverlapping pairs: {len(overlaps)}")
    print(f"Detailed results: {TABLES_DIR / 'spatial_leakage_pairs.csv'}")
    print(f"Summary: {TABLES_DIR / 'spatial_leakage_summary.csv'}")
    return overlaps, summary
