"""Model evaluation and classification artifact helpers."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch import nn
from torch.utils.data import DataLoader


def evaluate_model(model: nn.Module, data_loader: DataLoader,
                   criterion: nn.Module, device: torch.device):
    """Evaluate a model and return loss, scores, predictions, and labels."""
    model.eval()
    running_loss = 0.0
    predictions, labels = [], []

    with torch.no_grad():
        for images, batch_labels in data_loader:
            images = images.to(device, non_blocking=True)
            batch_labels = batch_labels.to(device, non_blocking=True)
            outputs = model(images)
            running_loss += criterion(outputs, batch_labels).item() * images.size(0)
            predictions.extend(torch.argmax(outputs, dim=1).cpu().numpy())
            labels.extend(batch_labels.cpu().numpy())

    if not len(data_loader.dataset):
        raise ValueError("Cannot evaluate an empty dataset.")
    predictions = np.asarray(predictions)
    labels = np.asarray(labels)
    average_loss = running_loss / len(data_loader.dataset)
    accuracy = accuracy_score(labels, predictions)
    macro_f1 = f1_score(labels, predictions, average="macro", zero_division=0)
    return average_loss, accuracy, macro_f1, predictions, labels


def save_evaluation_results(predictions: np.ndarray, labels: np.ndarray,
                            class_names: list[str], output_directory: Path,
                            prefix: str) -> None:
    """Write a classification report and confusion matrix to disk."""
    output_directory.mkdir(parents=True, exist_ok=True)
    class_ids = list(range(len(class_names)))
    report = classification_report(
        labels, predictions, labels=class_ids, target_names=class_names,
        output_dict=True, zero_division=0,
    )
    report_path = output_directory / f"{prefix}_classification_report.json"
    with report_path.open("w", encoding="utf-8") as report_file:
        json.dump(report, report_file, indent=4)

    matrix = confusion_matrix(labels, predictions, labels=class_ids)
    matrix_path = output_directory / f"{prefix}_confusion_matrix.csv"
    pd.DataFrame(matrix, index=class_names, columns=class_names).to_csv(
        matrix_path, encoding="utf-8"
    )
    print(f"Classification report saved: {report_path}")
    print(f"Confusion matrix saved: {matrix_path}")
