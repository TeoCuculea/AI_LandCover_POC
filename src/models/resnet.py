
"""
ResNet18 transfer learning models for RGB and multispectral EuroSAT data.
"""

import torch
import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18

from src.config import NUM_CLASSES, NUM_BANDS, RGB_BANDS


class ResNet18Classifier(nn.Module):
    """
    ResNet18 classifier adapted for RGB or multispectral Sentinel-2 images.

    Args:
        representation: Either "rgb" or "multispectral".
        num_classes: Number of target land-cover classes.
        pretrained: Whether to load ImageNet-pretrained weights.
    """

    def __init__(
        self,
        representation: str = "rgb",
        num_classes: int = NUM_CLASSES,
        pretrained: bool = True,
    ) -> None:
        super().__init__()

        if representation not in {"rgb", "multispectral"}:
            raise ValueError(
                "Representation must be 'rgb' or 'multispectral'."
            )

        self.representation = representation

        weights = ResNet18_Weights.DEFAULT if pretrained else None
        self.backbone = resnet18(weights=weights)

        if representation == "multispectral":
            self._adapt_first_convolution()

        # Replace the original ImageNet classification layer.
        input_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(input_features, num_classes)

    def _adapt_first_convolution(self) -> None:
        """
        Adapt the first convolution from 3 RGB channels to 13 Sentinel-2 bands.

        Sentinel-2 band order:
        B01, B02, B03, B04, B05, B06, B07,
        B08, B8A, B09, B10, B11, B12.

        Pretrained RGB filters are mapped to:
        B04 -> Red, B03 -> Green, B02 -> Blue.

        The remaining spectral-band filters are initialized to zero.
        They are subsequently learned during training.
        """

        original_conv = self.backbone.conv1

        if original_conv.in_channels != 3:
            raise ValueError(
                "Expected the original ResNet18 convolution to have 3 channels."
            )

        new_conv = nn.Conv2d(
            in_channels=NUM_BANDS,
            out_channels=original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            dilation=original_conv.dilation,
            groups=original_conv.groups,
            bias=original_conv.bias is not None,
            padding_mode=original_conv.padding_mode,
        )

        new_conv = new_conv.to(
            device=original_conv.weight.device,
            dtype=original_conv.weight.dtype,
        )

        with torch.no_grad():
            # Initialize all spectral channels to zero.
            new_conv.weight.zero_()

            # RGB_BANDS is ordered red, green, blue; ImageNet filters use
            # that same channel order.
            for rgb_channel, spectral_band in enumerate(RGB_BANDS):
                new_conv.weight[:, spectral_band, :, :] = original_conv.weight[
                    :, rgb_channel, :, :
                ]

            if original_conv.bias is not None:
                new_conv.bias.copy_(original_conv.bias)

        self.backbone.conv1 = new_conv

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Perform a forward pass.

        Args:
            images: Input tensor with shape [batch, channels, height, width].

        Returns:
            Classification logits with shape [batch, num_classes].
        """

        return self.backbone(images)


def build_model(
    representation: str = "rgb",
    num_classes: int = NUM_CLASSES,
    pretrained: bool = True,
) -> ResNet18Classifier:
    """
    Build a ResNet18 classifier for the selected image representation.

    Args:
        representation: "rgb" or "multispectral".
        num_classes: Number of output classes.
        pretrained: Whether to use ImageNet-pretrained weights.

    Returns:
        An initialized ResNet18Classifier.
    """

    return ResNet18Classifier(
        representation=representation,
        num_classes=num_classes,
        pretrained=pretrained,
    )
