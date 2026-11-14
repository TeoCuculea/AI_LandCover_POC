"""Training-history charts."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def save_training_plots(history: list[dict], output_directory: Path) -> None:
    """Render loss, accuracy, and learning-rate history charts."""
    output_directory.mkdir(parents=True, exist_ok=True)
    history_frame = pd.DataFrame(history)
    epochs = history_frame["epoch"]
    charts = (
        ("train_loss", "validation_loss", "Loss", "Training and Validation Loss",
         "training_curves.png"),
        ("train_accuracy", "validation_accuracy", "Accuracy",
         "Training and Validation Accuracy", "accuracy_curves.png"),
    )
    for train_column, validation_column, ylabel, title, filename in charts:
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(epochs, history_frame[train_column], label=f"Training {ylabel}")
        ax.plot(epochs, history_frame[validation_column], label=f"Validation {ylabel}")
        ax.set(xlabel="Epoch", ylabel=ylabel, title=title)
        ax.legend()
        ax.grid(alpha=0.3)
        _save_figure(fig, output_directory / filename)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(epochs, history_frame["learning_rate"], label="Learning Rate")
    ax.set(xlabel="Epoch", ylabel="Learning Rate", title="Learning Rate Schedule")
    ax.legend()
    ax.grid(alpha=0.3)
    _save_figure(fig, output_directory / "learning_rate_curve.png")
    print("Training plots saved.")


def _save_figure(fig: plt.Figure, output_path: Path) -> None:
    """Save and close one chart."""
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
