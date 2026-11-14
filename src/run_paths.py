"""Helpers for safe, isolated experiment output directories."""

from datetime import datetime, timezone
import re

from src.config import RUNS_DIR


_RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")


def create_run_id(prefix: str = "run") -> str:
    """Create a UTC timestamped run identifier."""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    return validate_run_id(f"{prefix}_{timestamp}")


def validate_run_id(run_id: str) -> str:
    """Reject run identifiers that could escape the runs directory."""
    if not _RUN_ID_PATTERN.fullmatch(run_id):
        raise ValueError(
            "Run ID must be 1–64 characters and contain only letters, "
            "numbers, underscores, or hyphens; it must start with a letter or number."
        )
    return run_id


def run_directory(run_id: str):
    """Return the output root for a validated run ID."""
    return RUNS_DIR / validate_run_id(run_id)
