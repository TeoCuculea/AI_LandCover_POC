"""Summarize and plot the saved RGB and multispectral experiments."""

import argparse
import json

import matplotlib.pyplot as plt
import pandas as pd

from src.config import FIGURES_DIR, RESULTS_DIR, TABLES_DIR
from src.run_paths import run_directory


EXPERIMENTS = (
    ("random_rgb", "RGB", "Random"),
    ("random_multispectral", "Multispectral", "Random"),
    ("spatial_rgb", "RGB", "Spatial"),
    ("spatial_multispectral", "Multispectral", "Spatial"),
)


def _output_directories(run_id: str | None = None):
    """Return the experiment, table, and figure directories for a run."""
    if run_id is None:
        return RESULTS_DIR / "experiments", TABLES_DIR, FIGURES_DIR
    run_root = run_directory(run_id)
    return run_root / "experiments", run_root / "tables", run_root / "figures"


def load_experiment_results(run_id: str | None = None) -> pd.DataFrame:
    """Load final metrics for each experiment with a saved result file."""
    experiment_directory, _, _ = _output_directories(run_id)
    rows = []
    for name, representation, split in EXPERIMENTS:
        metrics_path = experiment_directory / name / "final_metrics.json"
        if not metrics_path.is_file():
            print(f"Missing results: {name}")
            continue
        with metrics_path.open("r", encoding="utf-8") as metrics_file:
            metrics = json.load(metrics_file)
        rows.append({
            "Experiment": name,
            "Representation": representation,
            "Split": split,
            "Test Accuracy": metrics["test_accuracy"],
            "Test Macro F1": metrics["test_macro_f1"],
            "Test Loss": metrics["test_loss"],
            "Best Epoch": metrics["best_epoch"],
            "Validation Accuracy at Selected Epoch": metrics["best_validation_accuracy"],
            "Best Validation Macro F1": metrics.get("best_validation_macro_f1"),
        })
    return pd.DataFrame(rows)


def save_comparison_table(results: pd.DataFrame, output_directory=TABLES_DIR) -> None:
    """Save the experiment metrics as a CSV table."""
    output_directory.mkdir(parents=True, exist_ok=True)
    output_path = output_directory / "experiment_comparison.csv"
    results.to_csv(output_path, index=False)
    print(f"Comparison table saved: {output_path}")


def _save_metric_chart(
    results: pd.DataFrame,
    column: str,
    title: str,
    filename: str,
    output_directory=FIGURES_DIR,
) -> None:
    """Render one metric comparison using a shared chart style."""
    plot_data = results.sort_values(["Split", "Representation"])
    values = plot_data[column].to_numpy()
    labels = [f"{row.Split}\n{row.Representation}" for row in plot_data.itertuples()]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(labels, values)
    ax.set(title=title, xlabel="Experiment Configuration", ylabel=column)
    ax.set_ylim(max(0, values.min() - 0.03), min(1.0, values.max() + 0.015))
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    for bar, value in zip(bars, values):
        ax.annotate(
            f"{value:.4f}",
            (bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 3), textcoords="offset points", ha="center", fontsize=9,
        )
    _save_figure(fig, output_directory / filename)


def save_accuracy_comparison(results: pd.DataFrame, output_directory=FIGURES_DIR) -> None:
    """Save the test accuracy comparison chart."""
    _save_metric_chart(results, "Test Accuracy", "Test Accuracy Comparison",
                       "test_accuracy_comparison.png", output_directory)


def save_macro_f1_comparison(results: pd.DataFrame, output_directory=FIGURES_DIR) -> None:
    """Save the test macro-F1 comparison chart."""
    _save_metric_chart(results, "Test Macro F1", "Test Macro F1 Comparison",
                       "test_macro_f1_comparison.png", output_directory)


def save_multispectral_improvement(results: pd.DataFrame, output_directory=FIGURES_DIR) -> None:
    """Plot multispectral gains over RGB in percentage points."""
    rows = []
    for split in ("Random", "Spatial"):
        subset = results.loc[results["Split"] == split]
        rgb = subset.loc[subset["Representation"] == "RGB"]
        multispectral = subset.loc[subset["Representation"] == "Multispectral"]
        if rgb.empty or multispectral.empty:
            continue
        rows.append({
            "Split": split,
            "Accuracy Improvement (pp)": (
                multispectral.iloc[0]["Test Accuracy"] - rgb.iloc[0]["Test Accuracy"]
            ) * 100,
            "Macro F1 Improvement (pp)": (
                multispectral.iloc[0]["Test Macro F1"] - rgb.iloc[0]["Test Macro F1"]
            ) * 100,
        })
    if not rows:
        print("Not enough results to calculate improvements.")
        return

    improvement = pd.DataFrame(rows)
    positions = range(len(improvement))
    width = 0.35
    fig, ax = plt.subplots(figsize=(9, 6))
    bars_by_metric = (
        ax.bar([i - width / 2 for i in positions], improvement["Accuracy Improvement (pp)"],
               width, label="Accuracy"),
        ax.bar([i + width / 2 for i in positions], improvement["Macro F1 Improvement (pp)"],
               width, label="Macro F1"),
    )
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set(title="Multispectral Improvement over RGB", xlabel="Data Split Strategy",
           ylabel="Improvement (percentage points)")
    ax.set_xticks(list(positions), improvement["Split"])
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    for bars in bars_by_metric:
        for bar in bars:
            value = bar.get_height()
            ax.annotate(f"{value:+.2f}", (bar.get_x() + bar.get_width() / 2, value),
                        xytext=(0, 3 if value >= 0 else -3), textcoords="offset points",
                        ha="center", va="bottom" if value >= 0 else "top", fontsize=9)
    _save_figure(fig, output_directory / "multispectral_improvement.png")
    print("\nMULTISPECTRAL IMPROVEMENT")
    print("=" * 60)
    print(improvement.to_string(index=False, float_format="%.3f"))


def _save_figure(fig: plt.Figure, output_path) -> None:
    """Save a chart to the configured figures directory and release it."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Chart saved: {output_path}")


def main(run_id: str | None = None) -> None:
    """Generate the experiment table and comparison charts."""
    if run_id is None:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--run-id", help="Read and write outputs under this run ID.")
        run_id = parser.parse_args().run_id
    _, table_directory, figure_directory = _output_directories(run_id)
    results = load_experiment_results(run_id)
    if results.empty:
        print("No experiment results found.")
        return
    results = results.sort_values("Test Accuracy", ascending=False)
    print("\nEUROSAT EXPERIMENT COMPARISON")
    print("=" * 90)
    print(results.to_string(index=False, formatters={
        "Test Accuracy": "{:.4f}".format,
        "Test Macro F1": "{:.4f}".format,
        "Test Loss": "{:.4f}".format,
        "Validation Accuracy at Selected Epoch": "{:.4f}".format,
        "Best Validation Macro F1": "{:.4f}".format,
    }))
    save_comparison_table(results, table_directory)
    save_accuracy_comparison(results, figure_directory)
    save_macro_f1_comparison(results, figure_directory)
    save_multispectral_improvement(results, figure_directory)
    print("\nAll comparison outputs generated successfully.")


if __name__ == "__main__":
    main()
