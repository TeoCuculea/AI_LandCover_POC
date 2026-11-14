"""Compare RGB and multispectral F1 scores for each EuroSAT class."""

import argparse
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.run_paths import run_directory


EXPERIMENTS = {
    "Random": {"RGB": "random_rgb", "Multispectral": "random_multispectral"},
    "Spatial": {"RGB": "spatial_rgb", "Multispectral": "spatial_multispectral"},
}
AGGREGATE_ROWS = {
    "accuracy",
    "macro avg",
    "weighted avg",
    "micro avg",
    "samples avg",
}
METRICS = ("RGB F1", "Multispectral F1", "F1 Improvement")


def _output_directories(run_id: str):
    """Return the experiment, table, and figure directories for a run."""
    run_root = run_directory(run_id)
    return run_root / "experiments", run_root / "tables", run_root / "figures"


def load_classification_report(experiment_name: str, run_id: str) -> dict:
    """Read a saved test classification report for one experiment."""
    experiment_directory, _, _ = _output_directories(run_id)
    report_path = experiment_directory / experiment_name / "test_classification_report.json"
    if not report_path.is_file():
        raise FileNotFoundError(f"Missing classification report: {report_path}")
    with report_path.open("r", encoding="utf-8") as report_file:
        return json.load(report_file)


def _class_names(report: dict) -> list[str]:
    """Return class entries in report order, excluding summary metrics."""
    return [
        name
        for name, values in report.items()
        if isinstance(values, dict)
        and "f1-score" in values
            and name.lower() not in AGGREGATE_ROWS
    ]


def load_class_performance(run_id: str) -> pd.DataFrame:
    """Combine per-class test F1 scores from the four experiment reports."""
    reports = {
        (split, representation): load_classification_report(experiment, run_id)
        for split, experiments in EXPERIMENTS.items()
        for representation, experiment in experiments.items()
    }
    classes = _class_names(reports[("Random", "RGB")])

    rows = []
    for split in EXPERIMENTS:
        for class_name in classes:
            rgb_report = reports[(split, "RGB")]
            multispectral_report = reports[(split, "Multispectral")]
            if class_name not in rgb_report or class_name not in multispectral_report:
                raise ValueError(f"Class {class_name!r} is missing from a {split} report")
            rgb_f1 = rgb_report[class_name]["f1-score"]
            multispectral_f1 = multispectral_report[class_name]["f1-score"]
            rows.append({
                "Split": split,
                "Class": class_name,
                "RGB F1": rgb_f1,
                "Multispectral F1": multispectral_f1,
                "F1 Improvement": multispectral_f1 - rgb_f1,
            })
    return pd.DataFrame(rows, columns=("Split", "Class", *METRICS))


def save_f1_comparison(data: pd.DataFrame, split: str, output_directory) -> None:
    """Save a grouped RGB and multispectral F1 chart for one split."""
    split_data = data.loc[data["Split"] == split]
    positions = np.arange(len(split_data))
    width = 0.38

    fig, ax = plt.subplots(figsize=(13, 7))
    rgb = ax.bar(positions - width / 2, split_data["RGB F1"], width, label="RGB")
    multispectral = ax.bar(
        positions + width / 2, split_data["Multispectral F1"], width,
        label="Multispectral",
    )
    ax.set(title=f"Per-class F1-score: RGB vs Multispectral ({split} split)",
           xlabel="EuroSAT class", ylabel="F1-score")
    ax.set_xticks(positions, split_data["Class"], rotation=35, ha="right")
    ax.set_ylim(0, 1.08)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend()
    ax.bar_label(rgb, fmt="%.2f", padding=2, fontsize=8)
    ax.bar_label(multispectral, fmt="%.2f", padding=2, fontsize=8)
    _save_figure(fig, output_directory / f"per_class_f1_{split.lower()}.png")


def save_improvement_chart(data: pd.DataFrame, output_directory) -> None:
    """Save the per-class difference between multispectral and RGB F1."""
    improvement = data.pivot(index="Class", columns="Split", values="F1 Improvement")
    positions = np.arange(len(improvement))
    width = 0.38

    fig, ax = plt.subplots(figsize=(13, 7))
    for offset, split in ((-width / 2, "Random"), (width / 2, "Spatial")):
        bars = ax.bar(positions + offset, improvement[split], width, label=split)
        ax.bar_label(bars, fmt="%+.3f", padding=2, fontsize=8)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set(title="Per-class F1 improvement: Multispectral minus RGB",
           xlabel="EuroSAT class", ylabel="F1-score difference")
    ax.set_xticks(positions, improvement.index, rotation=35, ha="right")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend(title="Split")
    _save_figure(fig, output_directory / "per_class_multispectral_improvement.png")


def _save_figure(fig: plt.Figure, output_path) -> None:
    """Write and close a chart consistently."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Chart saved: {output_path}")


def main(run_id: str | None = None) -> None:
    """Generate the class-level CSV and charts from saved reports."""
    if run_id is None:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--run-id", required=True,
                            help="Run folder under results/runs/ to read and write.")
        run_id = parser.parse_args().run_id
    _, table_directory, figure_directory = _output_directories(run_id)
    data = load_class_performance(run_id)
    table_directory.mkdir(parents=True, exist_ok=True)
    csv_path = table_directory / "per_class_comparison.csv"
    data.to_csv(csv_path, index=False, float_format="%.6f")
    print(f"Per-class metrics saved: {csv_path}")

    for split in EXPERIMENTS:
        save_f1_comparison(data, split, figure_directory)
    save_improvement_chart(data, figure_directory)
    print("\nPER-CLASS F1 IMPROVEMENT (MULTISPECTRAL - RGB)")
    print(data.pivot(index="Class", columns="Split", values="F1 Improvement")
          .to_string(float_format=lambda value: f"{value:+.4f}"))


if __name__ == "__main__":
    main()
