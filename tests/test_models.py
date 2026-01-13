"""
Unit tests for model architectures.
"""

import pytest
import torch

from src.age_prediction import Config, create_model
from src.age_prediction.models import (
    AVAILABLE_MODELS,
    EnsembleModel,
    count_parameters,
    get_available_models,
    get_model_info,
    get_target_layer,
)


class TestModelCreation:
    """Tests for model creation and architecture."""

    @pytest.fixture
    def config(self):
        return Config()

    @pytest.mark.parametrize("model_name", AVAILABLE_MODELS)
    def test_create_model_returns_module(self, model_name, config):
        """Test that create_model returns a valid PyTorch module."""
        model = create_model(model_name, config)
        assert isinstance(model, torch.nn.Module)

    @pytest.mark.parametrize("model_name", AVAILABLE_MODELS)
    def test_model_output_shape(self, model_name, config):
        """Test that model outputs correct shape for 4 classes."""
        model = create_model(model_name, config)
        model.eval()

        # Create dummy input
        batch_size = 2
        dummy_input = torch.randn(batch_size, 3, 224, 224)

        with torch.no_grad():
            output = model(dummy_input)

        assert output.shape == (batch_size, config.num_classes)

    @pytest.mark.parametrize("model_name", AVAILABLE_MODELS)
    def test_model_frozen_backbone(self, model_name, config):
        """Test that backbone parameters are frozen when specified."""
        model = create_model(model_name, config, freeze_backbone=True)

        # Count frozen vs trainable parameters
        frozen = sum(1 for p in model.parameters() if not p.requires_grad)
        trainable = sum(1 for p in model.parameters() if p.requires_grad)

        # Should have both frozen and trainable parameters
        assert frozen > 0, "Should have frozen backbone parameters"
        assert trainable > 0, "Should have trainable classifier parameters"

    @pytest.mark.parametrize("model_name", AVAILABLE_MODELS)
    def test_model_unfrozen_backbone(self, model_name, config):
        """Test that all parameters are trainable when backbone not frozen."""
        model = create_model(model_name, config, freeze_backbone=False)

        # All parameters should be trainable
        for param in model.parameters():
            assert param.requires_grad

    def test_invalid_model_name_raises_error(self, config):
        """Test that invalid model name raises ValueError."""
        with pytest.raises(ValueError, match="Invalid model name"):
            create_model("invalid_model", config)

    def test_get_available_models_returns_list(self):
        """Test that get_available_models returns expected models."""
        models = get_available_models()
        assert isinstance(models, list)
        assert len(models) > 0
        assert "resnet50" in models
        assert "mobilenet_v2" in models
        assert "efficientnet_b0" in models


class TestTargetLayer:
    """Tests for Grad-CAM target layer extraction."""

    @pytest.fixture
    def config(self):
        return Config()

    @pytest.mark.parametrize("model_name", AVAILABLE_MODELS)
    def test_get_target_layer(self, model_name, config):
        """Test that target layer is a valid module."""
        model = create_model(model_name, config)
        target_layer = get_target_layer(model, model_name)

        assert isinstance(target_layer, torch.nn.Module)

    @pytest.mark.parametrize("model_name", AVAILABLE_MODELS)
    def test_target_layer_is_conv(self, model_name, config):
        """Test that target layer is a convolutional layer."""
        model = create_model(model_name, config)
        target_layer = get_target_layer(model, model_name)

        # Should be Conv2d for Grad-CAM
        assert isinstance(target_layer, torch.nn.Conv2d)


class TestParameterCounting:
    """Tests for parameter counting utilities."""

    @pytest.fixture
    def config(self):
        return Config()

    @pytest.mark.parametrize("model_name", AVAILABLE_MODELS)
    def test_count_parameters_total(self, model_name, config):
        """Test total parameter counting."""
        model = create_model(model_name, config)
        total = count_parameters(model, trainable_only=False)

        assert total > 0
        assert isinstance(total, int)

    @pytest.mark.parametrize("model_name", AVAILABLE_MODELS)
    def test_count_parameters_trainable(self, model_name, config):
        """Test trainable parameter counting."""
        model = create_model(model_name, config, freeze_backbone=True)
        trainable = count_parameters(model, trainable_only=True)
        total = count_parameters(model, trainable_only=False)

        assert trainable > 0
        assert trainable < total  # Trainable should be less when backbone frozen

    @pytest.mark.parametrize("model_name", AVAILABLE_MODELS)
    def test_get_model_info(self, model_name, config):
        """Test model info retrieval."""
        info = get_model_info(model_name, config)

        assert "name" in info
        assert "total_params" in info
        assert "trainable_params" in info
        assert "num_classes" in info
        assert info["name"] == model_name
        assert info["num_classes"] == config.num_classes


class TestEnsembleModel:
    """Tests for ensemble model."""

    @pytest.fixture
    def config(self):
        return Config()

    def test_ensemble_creation(self, config):
        """Test ensemble model creation."""
        ensemble = EnsembleModel(config=config)
        assert isinstance(ensemble, torch.nn.Module)
        assert len(ensemble.models) == len(AVAILABLE_MODELS)

    def test_ensemble_forward(self, config):
        """Test ensemble forward pass."""
        ensemble = EnsembleModel(config=config)
        ensemble.eval()

        batch_size = 2
        dummy_input = torch.randn(batch_size, 3, 224, 224)

        with torch.no_grad():
            output = ensemble(dummy_input)

        assert output.shape == (batch_size, config.num_classes)

    def test_ensemble_predict_proba(self, config):
        """Test ensemble probability prediction."""
        ensemble = EnsembleModel(config=config)

        batch_size = 2
        dummy_input = torch.randn(batch_size, 3, 224, 224)

        probs = ensemble.predict_proba(dummy_input)

        assert probs.shape == (batch_size, config.num_classes)
        # Probabilities should sum to 1
        assert torch.allclose(probs.sum(dim=1), torch.ones(batch_size), atol=1e-5)

    def test_ensemble_with_subset_models(self, config):
        """Test ensemble with subset of models."""
        model_names = ["resnet50", "mobilenet_v2"]
        ensemble = EnsembleModel(model_names=model_names, config=config)

        assert len(ensemble.models) == 2
