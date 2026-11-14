
"""
Training pipeline for RGB and multispectral EuroSAT classification.

The script supports:
- Random and spatial dataset splits.
- RGB and 13-band multispectral representations.
- ResNet18 transfer learning.
- GPU acceleration.
- Cosine annealing learning-rate scheduling.
- Best-model checkpointing.
- Validation and final test evaluation.
- Classification reports and confusion matrices.
"""

import argparse
import json
import random
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader

from src.config import NUM_CLASSES, RANDOM_STATE
from src.data.dataset import EuroSATDataset
from src.models.resnet import build_model
from src.run_paths import create_run_id, run_directory, validate_run_id
from src.training.metrics import evaluate_model, save_evaluation_results
from src.training.plots import save_training_plots


RANDOM_SEED = RANDOM_STATE

DEFAULT_BATCH_SIZE = 32
DEFAULT_EPOCHS = 30
DEFAULT_LEARNING_RATE = 1e-4
DEFAULT_WEIGHT_DECAY = 1e-4

DEFAULT_NUM_WORKERS = 0


def set_random_seed(seed: int = RANDOM_SEED) -> None:
    """
    Set random seeds for reproducible experiments.
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_data_loaders(
    representation: str,
    split_strategy: str,
    batch_size: int,
    num_workers: int,
):
    """
    Create DataLoaders for training, validation, and testing.
    """

    train_dataset = EuroSATDataset(
        split="train",
        representation=representation,
        split_strategy=split_strategy,
    )

    validation_dataset = EuroSATDataset(
        split="validation",
        representation=representation,
        split_strategy=split_strategy,
    )

    test_dataset = EuroSATDataset(
        split="test",
        representation=representation,
        split_strategy=split_strategy,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    return (
        train_loader,
        validation_loader,
        test_loader,
        train_dataset.class_names,
    )


def train_one_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
):
    """
    Train the model for one epoch.

    Returns:
        Average training loss and accuracy.
    """

    model.train()

    running_loss = 0.0
    all_predictions = []
    all_labels = []

    for images, labels in data_loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

        predictions = torch.argmax(outputs, dim=1)

        all_predictions.extend(predictions.detach().cpu().numpy())
        all_labels.extend(labels.detach().cpu().numpy())

    epoch_loss = running_loss / len(data_loader.dataset)
    epoch_accuracy = accuracy_score(all_labels, all_predictions)

    return epoch_loss, epoch_accuracy







def train_model(
    representation: str = "rgb",
    split_strategy: str = "random",
    batch_size: int = DEFAULT_BATCH_SIZE,
    epochs: int = DEFAULT_EPOCHS,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    weight_decay: float = DEFAULT_WEIGHT_DECAY,
    num_workers: int = DEFAULT_NUM_WORKERS,
    seed: int = RANDOM_SEED,
    run_id: str | None = None,
    patience: int = 6,
    min_delta: float = 0.001,
) -> None:
    """
    Train and evaluate a ResNet18 model.

    The best checkpoint is selected using validation macro F1.
    Training stops after ``patience`` epochs without a ``min_delta``
    improvement. The test set is evaluated only after model selection.
    """

    if epochs < 1:
        raise ValueError("epochs must be at least 1.")
    if patience < 0:
        raise ValueError("patience must be zero or greater.")
    if min_delta < 0:
        raise ValueError("min_delta must be zero or greater.")

    set_random_seed(seed)
    run_id = create_run_id("training") if run_id is None else validate_run_id(run_id)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    experiment_name = f"{split_strategy}_{representation}"

    run_root = run_directory(run_id)
    model_directory = run_root / "models"
    experiment_directory = run_root / "experiments" / experiment_name
    checkpoint_path = (
        model_directory / f"{experiment_name}_best.pth"
    )

    if experiment_directory.exists() or checkpoint_path.exists():
        raise FileExistsError(
            f"Outputs already exist for run '{run_id}', experiment "
            f"'{experiment_name}': {experiment_directory}. Choose a new run ID."
        )

    model_directory.mkdir(parents=True, exist_ok=True)
    experiment_directory.mkdir(parents=True, exist_ok=False)

    print("\n" + "=" * 65)
    print("EUROSAT TRANSFER LEARNING")
    print("=" * 65)

    print(f"Representation: {representation}")
    print(f"Split strategy: {split_strategy}")
    print(f"Run ID: {run_id}")
    print(f"Run output: {run_root}")
    print(f"Device: {device}")
    print(f"Batch size: {batch_size}")
    print(f"Epochs: {epochs}")
    print(f"Learning rate: {learning_rate}")
    print(f"Weight decay: {weight_decay}")
    print(f"Random seed: {seed}")

    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    (
        train_loader,
        validation_loader,
        test_loader,
        class_names,
    ) = create_data_loaders(
        representation=representation,
        split_strategy=split_strategy,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    print(f"\nTraining images: {len(train_loader.dataset)}")
    print(f"Validation images: {len(validation_loader.dataset)}")
    print(f"Test images: {len(test_loader.dataset)}")
    print(f"Classes: {class_names}")

    model = build_model(
        representation=representation,
        num_classes=NUM_CLASSES,
        pretrained=True,
    ).to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs,
        eta_min=1e-6,
    )

    best_validation_accuracy = 0.0
    best_validation_macro_f1 = float("-inf")
    patience_best_macro_f1 = float("-inf")
    epochs_without_improvement = 0
    best_epoch = 0

    history = []

    print("\nStarting training...\n")

    for epoch in range(1, epochs + 1):
        start_time = time.time()

        train_loss, train_accuracy = train_one_epoch(
            model=model,
            data_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )

        (
            validation_loss,
            validation_accuracy,
            validation_f1,
            _,
            _,
        ) = evaluate_model(
            model=model,
            data_loader=validation_loader,
            criterion=criterion,
            device=device,
        )

        current_learning_rate = optimizer.param_groups[0]["lr"]

        scheduler.step()

        epoch_duration = time.time() - start_time

        epoch_results = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "validation_loss": validation_loss,
            "validation_accuracy": validation_accuracy,
            "validation_macro_f1": validation_f1,
            "learning_rate": current_learning_rate,
            "duration_seconds": epoch_duration,
        }

        history.append(epoch_results)

        print(
            f"Epoch [{epoch:02d}/{epochs:02d}] | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_accuracy:.4f} | "
            f"Val Loss: {validation_loss:.4f} | "
            f"Val Acc: {validation_accuracy:.4f} | "
            f"Val F1: {validation_f1:.4f} | "
            f"Time: {epoch_duration:.1f}s"
        )

        if validation_f1 > best_validation_macro_f1:
            best_validation_macro_f1 = validation_f1
            best_validation_accuracy = validation_accuracy
            best_epoch = epoch

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "validation_accuracy": validation_accuracy,
                    "validation_macro_f1": validation_f1,
                    "representation": representation,
                    "split_strategy": split_strategy,
                    "class_names": class_names,
                },
                checkpoint_path,
            )

            print("  Best model checkpoint saved.")

        if validation_f1 > patience_best_macro_f1 + min_delta:
            patience_best_macro_f1 = validation_f1
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if patience > 0 and epochs_without_improvement >= patience:
            print(
                f"Early stopping: validation macro F1 did not improve by "
                f"{min_delta:.4f} for {patience} consecutive epochs."
            )
            break

    # Save training history.
    history_path = experiment_directory / "training_history.csv"

    pd.DataFrame(history).to_csv(
        history_path,
        index=False,
    )

    print("\nTraining completed.")
    print(f"Best epoch: {best_epoch}")
    print(f"Validation accuracy at selected epoch: {best_validation_accuracy:.4f}")
    print(f"Best validation macro F1: {best_validation_macro_f1:.4f}")
    print(f"Checkpoint: {checkpoint_path}")

    # Load the best model selected using validation performance.
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True,
    )

    model.load_state_dict(checkpoint["model_state_dict"])

    # Evaluate the selected model on the test set.
    (
        test_loss,
        test_accuracy,
        test_macro_f1,
        test_predictions,
        test_labels,
    ) = evaluate_model(
        model=model,
        data_loader=test_loader,
        criterion=criterion,
        device=device,
    )

    print("\n" + "=" * 65)
    print("FINAL TEST RESULTS")
    print("=" * 65)

    print(f"Test loss: {test_loss:.4f}")
    print(f"Test accuracy: {test_accuracy:.4f}")
    print(f"Test macro F1: {test_macro_f1:.4f}")

    save_evaluation_results(
        predictions=test_predictions,
        labels=test_labels,
        class_names=class_names,
        output_directory=experiment_directory,
        prefix="test",
    )

    final_results = {
        "run_id": run_id,
        "representation": representation,
        "split_strategy": split_strategy,
        "best_epoch": best_epoch,
        "best_validation_accuracy": best_validation_accuracy,
        "best_validation_macro_f1": best_validation_macro_f1,
        "checkpoint_selection_metric": "validation_macro_f1",
        "test_loss": test_loss,
        "test_accuracy": test_accuracy,
        "test_macro_f1": test_macro_f1,
        "batch_size": batch_size,
        "epochs": epochs,
        "epochs_completed": len(history),
        "early_stopping_patience": patience,
        "early_stopping_min_delta": min_delta,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "random_seed": seed,
        "device": str(device),
    }

    metrics_path = experiment_directory / "final_metrics.json"

    with open(metrics_path, "w", encoding="utf-8") as file:
        json.dump(final_results, file, indent=4)

    print(f"\nFinal metrics saved: {metrics_path}")
    print("\nExperiment completed successfully.")


def parse_arguments():
    """
    Parse command-line arguments.
    """

    parser = argparse.ArgumentParser(
        description="Train a ResNet18 model on EuroSAT."
    )

    parser.add_argument(
        "--representation",
        type=str,
        choices=["rgb", "multispectral"],
        default="rgb",
    )

    parser.add_argument(
        "--split",
        type=str,
        choices=["random", "spatial"],
        default="random",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_EPOCHS,
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=DEFAULT_LEARNING_RATE,
    )

    parser.add_argument(
        "--weight-decay",
        type=float,
        default=DEFAULT_WEIGHT_DECAY,
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=DEFAULT_NUM_WORKERS,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=RANDOM_SEED,
        help="Random seed used for reproducible training.",
    )

    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Shared output ID. If omitted, a unique timestamped ID is created.",
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=6,
        help="Early-stopping patience; use 0 to disable early stopping.",
    )

    parser.add_argument(
        "--min-delta",
        type=float,
        default=0.001,
        help="Minimum validation macro-F1 improvement for patience reset.",
    )

    return parser.parse_args()


def main() -> None:
    """Run one configured training experiment from command-line arguments."""
    arguments = parse_arguments()

    train_model(
        representation=arguments.representation,
        split_strategy=arguments.split,
        batch_size=arguments.batch_size,
        epochs=arguments.epochs,
        learning_rate=arguments.learning_rate,
        weight_decay=arguments.weight_decay,
        num_workers=arguments.num_workers,
        seed=arguments.seed,
        run_id=arguments.run_id,
        patience=arguments.patience,
        min_delta=arguments.min_delta,
    )


if __name__ == "__main__":
    main()
