"""Compatibility entrypoint; training orchestration lives in src.training."""

from src.training.runner import *  # noqa: F403
from src.training.runner import main

if __name__ == "__main__":
    main()
