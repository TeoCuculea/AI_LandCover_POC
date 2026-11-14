"""Draw the figures of one finished run from its saved files. No training.

Reads, for each of the four experiments in ``results/runs/<run-id>/``:
- ``training_history.csv``      -> loss and accuracy curves
- ``test_confusion_matrix.csv`` -> confusion matrices
- ``test_classification_report.json`` -> per-class F1 comparison

Usage:
    python -m src.analysis.plot_run_figures --run-id disertatie_final_01
"""

import argparse
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.analysis.plot_style import (
    BLUE,
    INK,
    INK_SECONDARY,
    MUTED,
    ORANGE,
    PAGE_WIDTH,
    RED,
    SEQUENTIAL_BLUE,
    apply_style,
    save_figure,
)
from src.run_paths import run_directory


# Folder name of each experiment and the title shown above its panel.
EXPERIMENTS = (
    ("random_rgb", "Split aleator · RGB"),
    ("random_multispectral", "Split aleator · 13 benzi"),
    ("spatial_rgb", "Split spațial · RGB"),
    ("spatial_multispectral", "Split spațial · 13 benzi"),
)

SHORT_CLASS_NAMES = {
    "AnnualCrop": "AnnCrop",
    "Forest": "Forest",
    "HerbaceousVegetation": "HerbVeg",
    "Highway": "Highway",
    "Industrial": "Industr",
    "Pasture": "Pasture",
    "PermanentCrop": "PermCrop",
    "Residential": "Resid",
    "River": "River",
    "SeaLake": "SeaLake",
}


def _experiment_directory(run_id: str, experiment: str):
    return run_directory(run_id) / "experiments" / experiment


def _load_json(path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def plot_history_grid(run_id: str, train_column: str, validation_column: str,
                      ylabel: str, filename: str, ylim=None) -> None:
    """One panel per experiment: the training curve against the validation curve."""
    fig, axes = plt.subplots(2, 2, figsize=(PAGE_WIDTH, 4.6), sharex=True, sharey=True)

    for ax, (experiment, title) in zip(axes.flat, EXPERIMENTS):
        directory = _experiment_directory(run_id, experiment)
        history = pd.read_csv(directory / "training_history.csv")
        best_epoch = _load_json(directory / "final_metrics.json")["best_epoch"]

        ax.plot(history["epoch"], history[train_column], color=BLUE, label="antrenare")
        ax.plot(history["epoch"], history[validation_column], color=ORANGE,
                label="validare")

        # The epoch whose checkpoint was kept and later evaluated on the test set.
        ax.axvline(best_epoch, color=MUTED, linewidth=0.8)
        ax.annotate(
            f"epoca aleasă: {best_epoch}", xy=(best_epoch, 1), xycoords=("data", "axes fraction"),
            xytext=(4, -3), textcoords="offset points", va="top", fontsize=7.5,
            color=INK_SECONDARY,
        )
        ax.set_title(f"{title}  ({len(history)} epoci)")
        ax.set_xlim(1, 30)
        if ylim is not None:
            ax.set_ylim(*ylim)

    for ax in axes[-1]:
        ax.set_xlabel("Epocă")
    for ax in axes[:, 0]:
        ax.set_ylabel(ylabel)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", ncols=2, bbox_to_anchor=(0.99, 1.03))
    fig.tight_layout()
    save_figure(fig, run_directory(run_id) / "figures" / filename)


def plot_confusion_matrices(run_id: str) -> None:
    """One confusion matrix per experiment, as counts on the test set."""
    fig, axes = plt.subplots(2, 2, figsize=(PAGE_WIDTH, 6.5), sharex=True, sharey=True)

    for ax, (experiment, title) in zip(axes.flat, EXPERIMENTS):
        matrix = pd.read_csv(
            _experiment_directory(run_id, experiment) / "test_confusion_matrix.csv",
            index_col=0,
        )
        counts = matrix.to_numpy()
        labels = [SHORT_CLASS_NAMES.get(name, name) for name in matrix.index]

        # Colour shows the share of each true class (row), so the diagonal is
        # dark and every mistake is visible as a light tint.
        shares = counts / counts.sum(axis=1, keepdims=True)
        ax.imshow(shares, cmap=SEQUENTIAL_BLUE, vmin=0, vmax=1)

        for row in range(counts.shape[0]):
            for column in range(counts.shape[1]):
                value = counts[row, column]
                if value == 0:
                    continue
                on_diagonal = row == column
                ax.text(
                    column, row, str(value), ha="center", va="center", fontsize=6.3,
                    color="white" if on_diagonal else INK,
                    fontweight="normal" if on_diagonal else "bold",
                )

        ax.set_title(title)
        ax.set_xticks(range(len(labels)), labels, rotation=90, fontsize=7)
        ax.set_yticks(range(len(labels)), labels, fontsize=7)
        ax.grid(False)
        for spine in ax.spines.values():
            spine.set_visible(False)

    for ax in axes[-1]:
        ax.set_xlabel("Clasa prezisă")
    for ax in axes[:, 0]:
        ax.set_ylabel("Clasa reală")

    fig.tight_layout()
    save_figure(fig, run_directory(run_id) / "figures" / "confusion_matrices.png")


def plot_per_class_f1_difference(run_id: str) -> None:
    """Per class: F1 with 13 bands minus F1 with RGB, for each split strategy."""
    reports = {
        experiment: _load_json(
            _experiment_directory(run_id, experiment) / "test_classification_report.json"
        )
        for experiment, _ in EXPERIMENTS
    }
    classes = list(SHORT_CLASS_NAMES)
    panels = (
        ("Split aleator", "random_rgb", "random_multispectral"),
        ("Split spațial", "spatial_rgb", "spatial_multispectral"),
    )

    fig, axes = plt.subplots(1, 2, figsize=(PAGE_WIDTH, 3.4), sharex=True, sharey=True)
    positions = np.arange(len(classes))

    for ax, (title, rgb_name, multispectral_name) in zip(axes, panels):
        difference = np.array([
            (reports[multispectral_name][name]["f1-score"]
             - reports[rgb_name][name]["f1-score"]) * 100
            for name in classes
        ])
        colors = [BLUE if value >= 0 else RED for value in difference]
        ax.barh(positions, difference, height=0.62, color=colors)
        ax.axvline(0, color=INK_SECONDARY, linewidth=0.8)
        ax.set_title(title)
        ax.set_yticks(positions, classes)
        ax.grid(axis="y", visible=False)
        ax.set_xlabel("Diferență F1 (puncte procentuale)")

    axes[0].invert_yaxis()  # shared axis: first class at the top in both panels

    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, color=BLUE),
        plt.Rectangle((0, 0), 1, 1, color=RED),
    ]
    fig.legend(legend_handles, ["13 benzi mai bun", "RGB mai bun"],
               loc="upper right", ncols=2, bbox_to_anchor=(0.99, 1.06))
    fig.tight_layout()
    save_figure(fig, run_directory(run_id) / "figures" / "per_class_f1_difference.png")


def main() -> None:
    """Regenerate every figure of a run from its saved results."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-id", required=True,
                        help="Run folder under results/runs/ to read and write.")
    run_id = parser.parse_args().run_id

    apply_style()
    plot_history_grid(run_id, "train_loss", "validation_loss",
                      "Loss (cross-entropy)", "loss_curves.png")
    plot_history_grid(run_id, "train_accuracy", "validation_accuracy",
                      "Acuratețe", "accuracy_curves.png", ylim=(0.88, 1.003))
    plot_confusion_matrices(run_id)
    plot_per_class_f1_difference(run_id)


if __name__ == "__main__":
    main()
