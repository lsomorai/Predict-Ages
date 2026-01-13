"""
Age Prediction - Deep Learning for Facial Age Classification

A production-ready facial age classification system using transfer learning
with MobileNetV2, ResNet50, and EfficientNet-B0 architectures.
"""

__version__ = "1.0.0"
__author__ = "Lucien Somorai"

from .config import AGE_GROUPS, Config
from .dataset import AgeDataset, create_dataloaders, get_transforms
from .gradcam import GradCAM, apply_gradcam
from .inference import AgePredictor
from .models import EnsembleModel, create_model, get_available_models

__all__ = [
    "Config",
    "AGE_GROUPS",
    "AgeDataset",
    "get_transforms",
    "create_dataloaders",
    "create_model",
    "get_available_models",
    "EnsembleModel",
    "AgePredictor",
    "GradCAM",
    "apply_gradcam",
]
