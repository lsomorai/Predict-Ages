"""
Model architectures for age prediction.
"""

from typing import Optional

import torch
import torch.nn as nn
from torchvision import models

from .config import NUM_CLASSES, Config

# Available model architectures
AVAILABLE_MODELS = ["mobilenet_v2", "resnet50", "efficientnet_b0"]


def get_available_models() -> list[str]:
    """Get list of available model architectures."""
    return AVAILABLE_MODELS.copy()


def _create_classifier(in_features: int, config: Config) -> nn.Sequential:
    """
    Create the custom classifier head.

    Args:
        in_features: Number of input features from backbone
        config: Configuration object

    Returns:
        Sequential classifier module
    """
    return nn.Sequential(
        nn.Linear(in_features, 512),
        nn.ReLU(),
        nn.Dropout(config.dropout_rate),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Dropout(config.dropout_rate),
        nn.Linear(256, config.num_classes)
    )


def create_model(
    model_name: str,
    config: Optional[Config] = None,
    pretrained: bool = True,
    freeze_backbone: bool = True
) -> nn.Module:
    """
    Create a model for age prediction.

    Args:
        model_name: One of "mobilenet_v2", "resnet50", "efficientnet_b0"
        config: Configuration object (uses defaults if None)
        pretrained: Whether to use ImageNet pretrained weights
        freeze_backbone: Whether to freeze the feature extractor

    Returns:
        PyTorch model ready for training/inference
    """
    if config is None:
        config = Config()

    if model_name not in AVAILABLE_MODELS:
        raise ValueError(
            f"Invalid model name: {model_name}. "
            f"Available: {AVAILABLE_MODELS}"
        )

    # Create base model with pretrained weights
    if model_name == "mobilenet_v2":
        weights = models.MobileNet_V2_Weights.DEFAULT if pretrained else None
        model = models.mobilenet_v2(weights=weights)
        in_features = model.classifier[1].in_features
        model.classifier = _create_classifier(in_features, config)
        classifier_name = "classifier"

    elif model_name == "resnet50":
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        model = models.resnet50(weights=weights)
        in_features = model.fc.in_features
        model.fc = _create_classifier(in_features, config)
        classifier_name = "fc"

    elif model_name == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        in_features = model.classifier[1].in_features
        model.classifier = _create_classifier(in_features, config)
        classifier_name = "classifier"

    # Freeze backbone if requested
    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

        # Unfreeze classifier
        classifier = getattr(model, classifier_name)
        for param in classifier.parameters():
            param.requires_grad = True

    return model


def get_target_layer(model: nn.Module, model_name: str) -> nn.Module:
    """
    Get the target layer for Grad-CAM visualization.

    Args:
        model: The model instance
        model_name: Name of the model architecture

    Returns:
        The target convolutional layer
    """
    if model_name == "resnet50":
        return model.layer4[-1].conv3
    elif model_name == "mobilenet_v2":
        return model.features[-1][0]
    elif model_name == "efficientnet_b0":
        return model.features[-1][0]
    else:
        raise ValueError(f"Unknown model: {model_name}")


class EnsembleModel(nn.Module):
    """
    Ensemble of multiple age prediction models.

    Combines predictions from multiple models using averaging.
    """

    def __init__(
        self,
        model_names: Optional[list[str]] = None,
        config: Optional[Config] = None,
        weights_dir: Optional[str] = None
    ):
        """
        Initialize ensemble model.

        Args:
            model_names: List of model names to include
            config: Configuration object
            weights_dir: Directory containing model weights
        """
        super().__init__()

        if model_names is None:
            model_names = AVAILABLE_MODELS

        if config is None:
            config = Config()

        self.model_names = model_names
        self.models = nn.ModuleList()

        for name in model_names:
            model = create_model(name, config, pretrained=True, freeze_backbone=True)

            # Load weights if available
            if weights_dir:
                weights_path = f"{weights_dir}/best_{name}.pth"
                try:
                    state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
                    model.load_state_dict(state_dict)
                    print(f"Loaded weights for {name}")
                except FileNotFoundError:
                    print(f"No weights found for {name}, using pretrained")

            self.models.append(model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through ensemble.

        Averages the softmax probabilities from all models.
        """
        outputs = []
        for model in self.models:
            output = model(x)
            probs = torch.softmax(output, dim=1)
            outputs.append(probs)

        # Average probabilities
        avg_probs = torch.stack(outputs).mean(dim=0)

        # Convert back to logits for compatibility with CrossEntropyLoss
        return torch.log(avg_probs + 1e-8)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Get averaged probability predictions."""
        outputs = []
        for model in self.models:
            model.eval()
            with torch.no_grad():
                output = model(x)
                probs = torch.softmax(output, dim=1)
                outputs.append(probs)

        return torch.stack(outputs).mean(dim=0)


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    """
    Count model parameters.

    Args:
        model: PyTorch model
        trainable_only: Count only trainable parameters

    Returns:
        Number of parameters
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def get_model_info(model_name: str, config: Optional[Config] = None) -> dict:
    """
    Get information about a model architecture.

    Returns:
        Dictionary with model information
    """
    model = create_model(model_name, config)

    return {
        "name": model_name,
        "total_params": count_parameters(model, trainable_only=False),
        "trainable_params": count_parameters(model, trainable_only=True),
        "num_classes": config.num_classes if config else NUM_CLASSES,
    }
