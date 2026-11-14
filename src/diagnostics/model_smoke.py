
"""
Test ResNet18 models with RGB and multispectral input tensors.
"""

import torch

from src.models.resnet import build_model
from src.config import NUM_CLASSES


def test_model(representation: str, input_channels: int) -> None:
    """
    Initialize a model and verify its output dimensions.
    """

    print("\n" + "=" * 60)
    print(f"Representation: {representation}")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_model(
        representation=representation,
        num_classes=NUM_CLASSES,
        pretrained=True,
    )

    model = model.to(device)
    model.eval()

    sample_batch = torch.randn(
        4,
        input_channels,
        64,
        64,
        device=device,
    )

    with torch.no_grad():
        predictions = model(sample_batch)

    print(f"Device: {device}")
    print(f"Input shape: {sample_batch.shape}")
    print(f"Output shape: {predictions.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    assert predictions.shape == (4, NUM_CLASSES)

    print("Test passed!")


if __name__ == "__main__":
    test_model("rgb", 3)
    test_model("multispectral", 13)
