"""
Inference utilities for age prediction.
"""

import os
import time
from typing import Optional, Union

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from .config import AGE_GROUPS, NUM_CLASSES, Config
from .gradcam import GradCAM
from .models import EnsembleModel, create_model, get_target_layer


class AgePredictor:
    """
    High-level interface for age prediction inference.

    Handles model loading, preprocessing, and prediction with optional Grad-CAM.
    """

    def __init__(
        self,
        model_name: str = "resnet50",
        weights_path: Optional[str] = None,
        device: Optional[str] = None,
        config: Optional[Config] = None,
    ):
        """
        Initialize the age predictor.

        Args:
            model_name: Model architecture ("mobilenet_v2", "resnet50", "efficientnet_b0", "ensemble")
            weights_path: Path to model weights file
            device: Device to use ("cuda", "cpu", "mps", or "auto")
            config: Configuration object
        """
        self.config = config or Config()
        self.model_name = model_name

        # Set device
        if device is None:
            device = self.config.get_device()
        self.device = torch.device(device)

        # Load model
        if model_name == "ensemble":
            weights_dir = os.path.dirname(weights_path) if weights_path else None
            self.model = EnsembleModel(weights_dir=weights_dir)
        else:
            self.model = create_model(
                model_name, self.config, pretrained=True, freeze_backbone=True
            )

            if weights_path and os.path.exists(weights_path):
                state_dict = torch.load(weights_path, map_location=self.device, weights_only=True)
                self.model.load_state_dict(state_dict)
                print(f"Loaded weights from: {weights_path}")

        self.model = self.model.to(self.device)
        self.model.eval()

        # Setup preprocessing
        self.transform = transforms.Compose(
            [
                transforms.Resize((self.config.image_size, self.config.image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.config.mean, std=self.config.std),
            ]
        )

        # Setup Grad-CAM (only for single models, not ensemble)
        self.gradcam = None
        if model_name != "ensemble":
            target_layer = get_target_layer(self.model, model_name)
            self.gradcam = GradCAM(self.model, target_layer)

    def preprocess(self, image: Union[str, Image.Image, np.ndarray]) -> torch.Tensor:
        """
        Preprocess an image for inference.

        Args:
            image: Image path, PIL Image, or numpy array

        Returns:
            Preprocessed tensor
        """
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image).convert("RGB")

        tensor = self.transform(image)
        return tensor.unsqueeze(0).to(self.device)

    def predict(
        self, image: Union[str, Image.Image, np.ndarray], return_probs: bool = True
    ) -> dict:
        """
        Predict age group for an image.

        Args:
            image: Input image (path, PIL Image, or numpy array)
            return_probs: Whether to return all class probabilities

        Returns:
            Dictionary with prediction results
        """
        tensor = self.preprocess(image)

        with torch.no_grad():
            outputs = self.model(tensor)
            probs = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probs, 1)

        pred_class = predicted.item()
        pred_confidence = confidence.item()

        result = {
            "class_idx": pred_class,
            "class_label": AGE_GROUPS[pred_class]["label"],
            "age_range": AGE_GROUPS[pred_class]["display"],
            "confidence": pred_confidence,
        }

        if return_probs:
            result["probabilities"] = {
                AGE_GROUPS[i]["display"]: probs[0, i].item() for i in range(NUM_CLASSES)
            }

        return result

    def predict_with_gradcam(
        self, image: Union[str, Image.Image, np.ndarray]
    ) -> tuple[dict, np.ndarray]:
        """
        Predict age with Grad-CAM visualization.

        Args:
            image: Input image

        Returns:
            Tuple of (prediction dict, Grad-CAM overlay image)
        """
        if self.gradcam is None:
            raise ValueError("Grad-CAM not available for ensemble models")

        tensor = self.preprocess(image)

        # Get prediction
        prediction = self.predict(image)

        # Generate Grad-CAM
        heatmap, _ = self.gradcam(tensor[0], prediction["class_idx"])
        overlay = self.gradcam.overlay_heatmap(tensor[0], heatmap)

        return prediction, overlay

    def predict_batch(self, images: list[Union[str, Image.Image, np.ndarray]]) -> list[dict]:
        """
        Predict age for multiple images.

        Args:
            images: List of images

        Returns:
            List of prediction dictionaries
        """
        results = []
        for image in images:
            result = self.predict(image)
            results.append(result)
        return results

    def benchmark(
        self, image: Union[str, Image.Image, np.ndarray], num_runs: int = 100, warmup_runs: int = 10
    ) -> dict:
        """
        Benchmark inference latency.

        Args:
            image: Input image
            num_runs: Number of timed runs
            warmup_runs: Number of warmup runs

        Returns:
            Dictionary with timing statistics
        """
        tensor = self.preprocess(image)

        # Warmup
        for _ in range(warmup_runs):
            with torch.no_grad():
                _ = self.model(tensor)

        # Synchronize if using CUDA
        if self.device.type == "cuda":
            torch.cuda.synchronize()

        # Timed runs
        times = []
        for _ in range(num_runs):
            start = time.perf_counter()
            with torch.no_grad():
                _ = self.model(tensor)
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            end = time.perf_counter()
            times.append((end - start) * 1000)  # Convert to ms

        times = np.array(times)

        return {
            "mean_ms": float(np.mean(times)),
            "std_ms": float(np.std(times)),
            "min_ms": float(np.min(times)),
            "max_ms": float(np.max(times)),
            "median_ms": float(np.median(times)),
            "p95_ms": float(np.percentile(times, 95)),
            "p99_ms": float(np.percentile(times, 99)),
            "throughput_fps": float(1000 / np.mean(times)),
        }


def load_predictor(
    model_name: str = "resnet50", checkpoint_dir: str = "./checkpoints"
) -> AgePredictor:
    """
    Convenience function to load a predictor with default settings.

    Args:
        model_name: Model architecture
        checkpoint_dir: Directory containing model weights

    Returns:
        AgePredictor instance
    """
    if model_name == "ensemble":
        weights_path = checkpoint_dir
    else:
        weights_path = os.path.join(checkpoint_dir, f"best_{model_name}.pth")

    return AgePredictor(model_name=model_name, weights_path=weights_path)
