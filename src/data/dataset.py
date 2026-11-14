
"""PyTorch dataset implementation for EuroSAT multispectral imagery."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import torch

from torch.utils.data import Dataset

from src.config import (
    DATA_DIR,
    IMAGE_SIZE,
    NORMALIZATION_FILE,
    NUM_BANDS,
    RANDOM_SPLIT_FILE,
    RGB_BANDS,
    SPATIAL_SPLIT_FILE,
)


# Dataset split files
# Supported dataset representations
SUPPORTED_REPRESENTATIONS = {
    "rgb": 3,
    "multispectral": NUM_BANDS,
}

# Supported split strategies
SUPPORTED_SPLIT_STRATEGIES = {
    "random",
    "spatial",
}


class EuroSATDataset(Dataset):
    """Load EuroSAT images as RGB or multispectral PyTorch tensors."""

    def __init__(
        self,
        split="train",
        representation="multispectral",
        split_strategy="random",
        transform=None,
    ):
        """
        Initialize the EuroSAT dataset.

        Args:
            split: Dataset subset ('train', 'validation', or 'test').
            representation: 'rgb' or 'multispectral'.
            split_strategy: 'random' or 'spatial'.
            transform: Optional image transformation.
        """

        self.split = split
        self.representation = representation.lower()
        self.split_strategy = split_strategy.lower()
        self.transform = transform

        self._validate_configuration()

        self.samples = self._load_samples()

        self.mean, self.std = self._load_normalization_parameters()

        self.class_names = sorted(
            path.name
            for path in DATA_DIR.iterdir()
            if path.is_dir()
        )

        self.num_classes = len(self.class_names)

        if not self.samples:
            raise ValueError(
                f"No images found for split '{self.split}' "
                f"using strategy '{self.split_strategy}'."
            )

    def _validate_configuration(self):
        """Validate dataset configuration parameters."""

        if self.split not in {"train", "validation", "test"}:
            raise ValueError(
                f"Unsupported split: {self.split}"
            )

        if self.representation not in SUPPORTED_REPRESENTATIONS:
            raise ValueError(
                f"Unsupported representation: "
                f"{self.representation}"
            )

        if self.split_strategy not in SUPPORTED_SPLIT_STRATEGIES:
            raise ValueError(
                f"Unsupported split strategy: "
                f"{self.split_strategy}"
            )

        if not DATA_DIR.exists():
            raise FileNotFoundError(
                f"Dataset directory not found: {DATA_DIR}"
            )

        if not NORMALIZATION_FILE.exists():
            raise FileNotFoundError(
                f"Normalization file not found: "
                f"{NORMALIZATION_FILE}"
            )

    def _load_samples(self):
        """Load image paths and labels from the selected split CSV."""

        split_file, split_column = {
            "random": (RANDOM_SPLIT_FILE, "split"),
            "spatial": (SPATIAL_SPLIT_FILE, "spatial_split"),
        }[self.split_strategy]

        if not split_file.exists():
            raise FileNotFoundError(
                f"Split file not found: {split_file}"
            )

        dataframe = pd.read_csv(split_file)

        required_columns = {
            "image",
            "label",
            "label_id",
            split_column,
        }

        missing_columns = required_columns - set(dataframe.columns)

        if missing_columns:
            raise ValueError(
                f"Missing columns in {split_file.name}: "
                f"{missing_columns}"
            )

        selected_samples = dataframe[
            dataframe[split_column] == self.split
        ].copy()

        samples = []

        for _, row in selected_samples.iterrows():

            image_path = DATA_DIR / str(row["label"]) / str(row["image"])

            if not image_path.exists():
                raise FileNotFoundError(
                    f"Image not found: {image_path}"
                )

            samples.append(
                {
                    "image_path": image_path,
                    "label": int(row["label_id"]),
                    "class_name": str(row["label"]),
                }
            )

        return samples

    def _load_normalization_parameters(self):
        """Load normalization parameters for the selected experiment."""

        with NORMALIZATION_FILE.open("r", encoding="utf-8") as file:
            normalization_data = json.load(file)

        experiment_statistics = normalization_data[
            self.split_strategy
        ]

        representation_statistics = experiment_statistics[
            self.representation
        ]

        mean = np.asarray(
            representation_statistics["mean"],
            dtype=np.float32,
        )

        std = np.asarray(
            representation_statistics["std"],
            dtype=np.float32,
        )

        if np.any(std <= 0):
            raise ValueError(
                "Normalization standard deviations must be positive."
            )

        expected_channels = SUPPORTED_REPRESENTATIONS[
            self.representation
        ]

        if len(mean) != expected_channels:
            raise ValueError(
                f"Expected {expected_channels} means, "
                f"received {len(mean)}."
            )

        if len(std) != expected_channels:
            raise ValueError(
                f"Expected {expected_channels} standard deviations, "
                f"received {len(std)}."
            )

        return mean, std

    def _read_image(self, image_path):
        """Read a multispectral TIFF image and select the requested bands."""

        with rasterio.open(image_path) as source:

            if source.count != NUM_BANDS:
                raise ValueError(
                    f"Expected {NUM_BANDS} bands in {image_path}, "
                    f"received {source.count}."
                )

            image = source.read(
                masked=True,
                out_dtype="float32",
            )

        image = np.ma.filled(
            image,
            fill_value=np.nan,
        )

        if self.representation == "rgb":
            image = image[RGB_BANDS]

        if image.shape[1:] != (IMAGE_SIZE, IMAGE_SIZE):
            raise ValueError(
                f"Unexpected image dimensions in {image_path}: "
                f"{image.shape[1:]}"
            )

        if not np.isfinite(image).all():
            raise ValueError(
                f"Image contains invalid pixel values: {image_path}"
            )

        return image

    def __len__(self):
        """Return the number of samples in the dataset."""

        return len(self.samples)

    def __getitem__(self, index):
        """Return a normalized image tensor and its class label."""

        sample = self.samples[index]

        image = self._read_image(
            sample["image_path"]
        )

        mean = self.mean[:, np.newaxis, np.newaxis]
        std = self.std[:, np.newaxis, np.newaxis]

        image = (image - mean) / std

        image = np.ascontiguousarray(
            image,
            dtype=np.float32,
        )

        image_tensor = torch.from_numpy(image)

        if self.transform is not None:
            image_tensor = self.transform(image_tensor)

        label = torch.tensor(
            sample["label"],
            dtype=torch.long,
        )

        return image_tensor, label
