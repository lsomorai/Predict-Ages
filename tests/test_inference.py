"""
Unit tests for inference functionality.
"""

import numpy as np
import pytest
import torch
from PIL import Image

from src.age_prediction import AGE_GROUPS, AgePredictor, Config
from src.age_prediction.gradcam import GradCAM, denormalize_image
from src.age_prediction.models import create_model, get_target_layer


class TestAgePredictor:
    """Tests for AgePredictor class."""

    @pytest.fixture
    def predictor(self):
        """Create a predictor with pretrained weights only."""
        return AgePredictor(
            model_name="mobilenet_v2",
            weights_path=None,  # Use pretrained only
            device="cpu"
        )

    @pytest.fixture
    def dummy_image(self):
        """Create a dummy test image."""
        return Image.new("RGB", (256, 256), color="gray")

    def test_predictor_initialization(self, predictor):
        """Test predictor initializes correctly."""
        assert predictor.model is not None
        assert predictor.device == torch.device("cpu")
        assert predictor.transform is not None

    def test_predict_from_pil_image(self, predictor, dummy_image):
        """Test prediction from PIL Image."""
        result = predictor.predict(dummy_image)

        assert "class_idx" in result
        assert "class_label" in result
        assert "age_range" in result
        assert "confidence" in result
        assert "probabilities" in result

        assert 0 <= result["class_idx"] < len(AGE_GROUPS)
        assert 0 <= result["confidence"] <= 1

    def test_predict_from_numpy(self, predictor):
        """Test prediction from numpy array."""
        numpy_image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        result = predictor.predict(numpy_image)

        assert "class_idx" in result
        assert 0 <= result["class_idx"] < len(AGE_GROUPS)

    def test_predict_probabilities_sum_to_one(self, predictor, dummy_image):
        """Test that probabilities sum to approximately 1."""
        result = predictor.predict(dummy_image, return_probs=True)

        probs = list(result["probabilities"].values())
        assert abs(sum(probs) - 1.0) < 1e-5

    def test_predict_batch(self, predictor, dummy_image):
        """Test batch prediction."""
        images = [dummy_image, dummy_image, dummy_image]
        results = predictor.predict_batch(images)

        assert len(results) == 3
        for result in results:
            assert "class_idx" in result

    def test_predict_with_gradcam(self, predictor, dummy_image):
        """Test prediction with Grad-CAM visualization."""
        result, overlay = predictor.predict_with_gradcam(dummy_image)

        assert "class_idx" in result
        assert overlay is not None
        assert isinstance(overlay, np.ndarray)
        assert overlay.ndim == 3  # (H, W, C)

    def test_benchmark(self, predictor, dummy_image):
        """Test benchmarking functionality."""
        benchmark = predictor.benchmark(dummy_image, num_runs=5, warmup_runs=2)

        assert "mean_ms" in benchmark
        assert "std_ms" in benchmark
        assert "throughput_fps" in benchmark
        assert benchmark["mean_ms"] > 0

    def test_different_model_architectures(self, dummy_image):
        """Test predictor works with different architectures."""
        for model_name in ["mobilenet_v2", "resnet50", "efficientnet_b0"]:
            predictor = AgePredictor(
                model_name=model_name,
                weights_path=None,
                device="cpu"
            )
            result = predictor.predict(dummy_image)
            assert "class_idx" in result


class TestGradCAM:
    """Tests for Grad-CAM functionality."""

    @pytest.fixture
    def model_and_layer(self):
        """Create model and target layer for testing."""
        config = Config()
        model_name = "mobilenet_v2"
        model = create_model(model_name, config)
        model.eval()
        target_layer = get_target_layer(model, model_name)
        return model, target_layer

    @pytest.fixture
    def dummy_tensor(self):
        """Create a dummy image tensor."""
        return torch.randn(3, 224, 224)

    def test_gradcam_heatmap_shape(self, model_and_layer, dummy_tensor):
        """Test Grad-CAM produces valid heatmap."""
        model, target_layer = model_and_layer
        gradcam = GradCAM(model, target_layer)

        try:
            heatmap, pred_class = gradcam(dummy_tensor)

            assert isinstance(heatmap, np.ndarray)
            assert heatmap.ndim == 2  # 2D heatmap
            assert 0 <= pred_class < len(AGE_GROUPS)
        finally:
            gradcam.remove_hooks()

    def test_gradcam_overlay(self, model_and_layer, dummy_tensor):
        """Test Grad-CAM overlay generation."""
        model, target_layer = model_and_layer
        gradcam = GradCAM(model, target_layer)

        try:
            heatmap, _ = gradcam(dummy_tensor)
            overlay = gradcam.overlay_heatmap(dummy_tensor, heatmap)

            assert isinstance(overlay, np.ndarray)
            assert overlay.shape[2] == 3  # RGB
        finally:
            gradcam.remove_hooks()

    def test_gradcam_with_target_class(self, model_and_layer, dummy_tensor):
        """Test Grad-CAM with specified target class."""
        model, target_layer = model_and_layer
        gradcam = GradCAM(model, target_layer)

        try:
            target_class = 2
            heatmap, returned_class = gradcam(dummy_tensor, class_idx=target_class)

            assert returned_class == target_class
        finally:
            gradcam.remove_hooks()


class TestDenormalize:
    """Tests for image denormalization."""

    def test_denormalize_shape(self):
        """Test denormalized image has correct shape."""
        # Normalized tensor (C, H, W)
        tensor = torch.randn(3, 224, 224)
        result = denormalize_image(tensor)

        assert isinstance(result, np.ndarray)
        assert result.shape == (224, 224, 3)
        assert result.dtype == np.uint8

    def test_denormalize_range(self):
        """Test denormalized values are in valid range."""
        tensor = torch.randn(3, 224, 224)
        result = denormalize_image(tensor)

        assert result.min() >= 0
        assert result.max() <= 255

    def test_denormalize_with_batch(self):
        """Test denormalization handles batch dimension."""
        tensor = torch.randn(1, 3, 224, 224)
        result = denormalize_image(tensor)

        assert result.shape == (224, 224, 3)
