
"""Test the EuroSAT PyTorch dataset implementation."""

from torch.utils.data import DataLoader

from src.data.dataset import EuroSATDataset


def inspect_dataset(representation, split_strategy):
    """Inspect one dataset configuration."""

    print("\n" + "=" * 60)
    print(f"Representation: {representation}")
    print(f"Split strategy: {split_strategy}")
    print("=" * 60)

    dataset = EuroSATDataset(
        split="train",
        representation=representation,
        split_strategy=split_strategy,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=16,
        shuffle=True,
        num_workers=0,
    )

    images, labels = next(iter(dataloader))

    print(f"Dataset size: {len(dataset)}")
    print(f"Image batch shape: {images.shape}")
    print(f"Label batch shape: {labels.shape}")
    print(f"Image data type: {images.dtype}")
    print(f"Label data type: {labels.dtype}")
    print(f"Image minimum: {images.min().item():.4f}")
    print(f"Image maximum: {images.max().item():.4f}")
    print(f"Image mean: {images.mean().item():.4f}")
    print(f"Image standard deviation: {images.std().item():.4f}")
    print(f"Unique labels: {labels.unique().tolist()}")


def main():
    """Test RGB and multispectral dataset configurations."""

    configurations = [
        ("rgb", "random"),
        ("multispectral", "random"),
        ("rgb", "spatial"),
        ("multispectral", "spatial"),
    ]

    for representation, split_strategy in configurations:
        inspect_dataset(
            representation,
            split_strategy,
        )


if __name__ == "__main__":
    main()
