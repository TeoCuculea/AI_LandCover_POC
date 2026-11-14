"""Run the four baseline configurations with isolated outputs."""

import argparse
from datetime import datetime, timezone
import subprocess
import sys

from src.run_paths import run_directory, validate_run_id


EXPERIMENTS = (
    ("rgb", "random"),
    ("multispectral", "random"),
    ("rgb", "spatial"),
    ("multispectral", "spatial"),
)


def _default_run_id() -> str:
    """Generate a timestamped ID shared by the complete experiment batch."""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    return validate_run_id(f"baseline_{timestamp}")


def _is_experiment_complete(run_root, representation: str, split: str) -> bool:
    """Check whether an experiment has its checkpoint and final artifacts."""
    experiment_name = f"{split}_{representation}"
    experiment_directory = run_root / "experiments" / experiment_name
    checkpoint = run_root / "models" / f"{experiment_name}_best.pth"
    expected = (
        experiment_directory / "final_metrics.json",
        experiment_directory / "test_classification_report.json",
        experiment_directory / "test_confusion_matrix.csv",
        experiment_directory / "training_history.csv",
        checkpoint,
    )
    return all(path.is_file() for path in expected)


def _experiment_has_partial_outputs(run_root, representation: str, split: str) -> bool:
    """Detect an interrupted experiment that must not be overwritten."""
    experiment_name = f"{split}_{representation}"
    experiment_directory = run_root / "experiments" / experiment_name
    checkpoint = run_root / "models" / f"{experiment_name}_best.pth"
    return experiment_directory.exists() or checkpoint.exists()


def _training_command(arguments, representation: str, split: str) -> list[str]:
    """Build one training command."""
    return [
        sys.executable, "-u", "-m", "src.training.runner",
        "--representation", representation,
        "--split", split,
        "--epochs", str(arguments.epochs),
        "--batch-size", str(arguments.batch_size),
        "--num-workers", str(arguments.num_workers),
        "--seed", str(arguments.seed),
        "--run-id", arguments.run_id,
        "--patience", str(arguments.patience),
        "--min-delta", str(arguments.min_delta),
    ]


def _report_commands(arguments) -> list[list[str]]:
    """Build the aggregate and per-class report commands."""
    return [
        [sys.executable, "-u", "-m", "src.analysis.compare_experiments",
         "--run-id", arguments.run_id],
        [sys.executable, "-u", "-m", "src.analysis.compare_class_performance",
         "--run-id", arguments.run_id],
    ]


def _commands(arguments, pending_experiments) -> list[list[str]]:
    """Build commands for pending experiments followed by both reports."""
    commands = []
    for representation, split in pending_experiments:
        commands.append([
            *_training_command(arguments, representation, split)
        ])
    commands.extend(_report_commands(arguments))
    return commands


def parse_arguments():
    """Parse batch settings and dry-run options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default=None,
                        help="Shared output ID; defaults to a unique UTC timestamp.")
    parser.add_argument("--epochs", type=int, default=30,
                        help="Maximum epochs for each experiment.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--patience", type=int, default=6,
                        help="Early stopping patience; 0 disables early stopping.")
    parser.add_argument("--min-delta", type=float, default=0.001,
                        help="Minimum validation macro-F1 increase for patience reset.")
    parser.add_argument("--what-if", action="store_true",
                        help="Print commands without starting training.")
    parser.add_argument("--continue-run", action="store_true",
                        help="Skip completed experiments in an existing run ID.")
    return parser.parse_args()


def main() -> None:
    """Run each experiment sequentially, then generate run-scoped reports."""
    arguments = parse_arguments()
    arguments.run_id = (
        _default_run_id()
        if arguments.run_id is None
        else validate_run_id(arguments.run_id)
    )
    output_root = run_directory(arguments.run_id)
    if arguments.continue_run and not output_root.is_dir():
        raise FileNotFoundError(
            f"Cannot continue: run directory does not exist: {output_root}"
        )
    if not arguments.continue_run and output_root.exists():
        raise FileExistsError(
            f"Run directory already exists: {output_root}. Choose a new --run-id."
        )

    pending_experiments = []
    for representation, split in EXPERIMENTS:
        if _is_experiment_complete(output_root, representation, split):
            print(f"Already complete; skipping {split} / {representation}.")
        elif _experiment_has_partial_outputs(output_root, representation, split):
            raise RuntimeError(
                f"Found incomplete outputs for {split} / {representation} in "
                f"{output_root}. They will not be overwritten; choose a new run ID "
                "or remove that experiment's partial files yourself."
            )
        else:
            pending_experiments.append((representation, split))

    commands = _commands(arguments, pending_experiments)

    print(f"Run ID: {arguments.run_id}")
    print(f"Output root: {output_root}")
    print(f"Maximum epochs: {arguments.epochs}; patience: {arguments.patience}")
    if arguments.what_if:
        print("Dry run only; no training was started.")
        for command in commands:
            print(" ".join(command))
        return

    for command in commands:
        print(f"\nExecuting: {' '.join(command)}", flush=True)
        subprocess.run(command, check=True)
    print(f"\nAll experiments and reports completed: {output_root}")


if __name__ == "__main__":
    main()
