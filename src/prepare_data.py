"""Compatibility entrypoint for dataset preparation."""

from src.data.prepare import EuroSATDataManager, main

__all__ = ["EuroSATDataManager", "main"]

if __name__ == "__main__":
    main()
