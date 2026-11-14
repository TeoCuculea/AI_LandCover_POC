"""Compatibility imports; model implementations live in src.models."""

from src.models.resnet import ResNet18Classifier, build_model

__all__ = ["ResNet18Classifier", "build_model"]
