"""
Unit tests for dataset and data loading.
"""

import pytest
import torch
from PIL import Image
from torchvision import transforms

from src.age_prediction import AGE_GROUPS, Config
from src.age_prediction.config import NUM_CLASSES, get_age_group, get_age_label
from src.age_prediction.dataset import AgeDataset, get_transforms


class TestAgeGrouping:
    """Tests for age group utilities."""

    @pytest.mark.parametrize(
        "age,expected_group",
        [
            (0, 0),
            (15, 0),
            (25, 0),
            (26, 1),
            (40, 1),
            (50, 1),
            (51, 2),
            (65, 2),
            (75, 2),
            (76, 3),
            (100, 3),
            (116, 3),
        ],
    )
    def test_get_age_group(self, age, expected_group):
        """Test age to group conversion."""
        assert get_age_group(age) == expected_group

    def test_get_age_group_invalid(self):
        """Test invalid age returns -1."""
        assert get_age_group(-1) == -1
        assert get_age_group(200) == -1

    @pytest.mark.parametrize(
        "group_idx,expected_label",
        [
            (0, "0-25"),
            (1, "26-50"),
            (2, "51-75"),
            (3, "76-116"),
        ],
    )
    def test_get_age_label(self, group_idx, expected_label):
        """Test group index to label conversion."""
        assert get_age_label(group_idx) == expected_label

    def test_get_age_label_invalid(self):
        """Test invalid group returns Unknown."""
        assert get_age_label(-1) == "Unknown"
        assert get_age_label(10) == "Unknown"

    def test_num_classes(self):
        """Test number of classes matches age groups."""
        assert NUM_CLASSES == len(AGE_GROUPS)
        assert NUM_CLASSES == 4


class TestTransforms:
    """Tests for data transforms."""

    @pytest.fixture
    def config(self):
        return Config()

    def test_train_transform_with_augmentation(self, config):
        """Test training transform includes augmentation."""
        config.use_augmentation = True
        transform = get_transforms(config, mode="train")

        assert transform is not None
        # Create a dummy image and verify transform works
        dummy_image = Image.new("RGB", (256, 256), color="red")
        transformed = transform(dummy_image)

        assert isinstance(transformed, torch.Tensor)
        assert transformed.shape == (3, config.image_size, config.image_size)

    def test_val_transform_no_augmentation(self, config):
        """Test validation transform has no augmentation."""
        transform = get_transforms(config, mode="val")

        dummy_image = Image.new("RGB", (256, 256), color="blue")
        transformed = transform(dummy_image)

        assert isinstance(transformed, torch.Tensor)
        assert transformed.shape == (3, config.image_size, config.image_size)

    def test_test_transform_equals_val(self, config):
        """Test that test and val transforms produce same shape."""
        val_transform = get_transforms(config, mode="val")
        test_transform = get_transforms(config, mode="test")

        dummy_image = Image.new("RGB", (256, 256), color="green")

        val_out = val_transform(dummy_image)
        test_out = test_transform(dummy_image)

        assert val_out.shape == test_out.shape

    def test_transform_normalization(self, config):
        """Test that transforms apply ImageNet normalization."""
        transform = get_transforms(config, mode="val")

        # White image should have values around (1 - mean) / std after normalization
        white_image = Image.new("RGB", (224, 224), color="white")
        transformed = transform(white_image)

        # Values should not be in [0, 1] range after normalization
        assert transformed.min() < 0 or transformed.max() > 1


class TestAgeDataset:
    """Tests for AgeDataset class."""

    @pytest.fixture
    def temp_dataset(self, tmp_path):
        """Create a temporary dataset with valid image files."""
        # Create some dummy images with valid UTKFace naming
        test_images = [
            "20_0_0_12345.jpg",  # Age 20 -> Group 0
            "35_1_1_12346.jpg",  # Age 35 -> Group 1
            "60_0_2_12347.jpg",  # Age 60 -> Group 2
            "80_1_3_12348.jpg",  # Age 80 -> Group 3
        ]

        for filename in test_images:
            img = Image.new("RGB", (100, 100), color="gray")
            img.save(tmp_path / filename)

        return tmp_path

    def test_dataset_loading(self, temp_dataset):
        """Test dataset loads images correctly."""
        dataset = AgeDataset(str(temp_dataset), verbose=False)

        assert len(dataset) == 4
        assert len(dataset.files) == 4
        assert len(dataset.labels) == 4

    def test_dataset_labels(self, temp_dataset):
        """Test dataset assigns correct labels."""
        dataset = AgeDataset(str(temp_dataset), verbose=False)

        # Sort by filename to have predictable order
        sorted_pairs = sorted(zip(dataset.files, dataset.labels), key=lambda x: x[0])
        labels = [label for _, label in sorted_pairs]

        # Labels should be 0, 1, 2, 3 for our test files
        assert sorted(labels) == [0, 1, 2, 3]

    def test_dataset_getitem(self, temp_dataset):
        """Test dataset __getitem__ returns correct types."""
        transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ]
        )
        dataset = AgeDataset(str(temp_dataset), transform=transform, verbose=False)

        image, label = dataset[0]

        assert isinstance(image, torch.Tensor)
        assert image.shape == (3, 224, 224)
        assert isinstance(label, int)
        assert 0 <= label < NUM_CLASSES

    def test_dataset_class_weights(self, temp_dataset):
        """Test class weights calculation."""
        dataset = AgeDataset(str(temp_dataset), verbose=False)
        weights = dataset.get_class_weights()

        assert isinstance(weights, torch.Tensor)
        assert len(weights) == NUM_CLASSES
        # With equal distribution, weights should be similar
        assert all(w > 0 for w in weights)

    def test_dataset_skips_invalid_filenames(self, tmp_path):
        """Test dataset skips files with invalid naming."""
        # Create files with invalid names
        invalid_files = [
            "not_a_valid_name.jpg",
            "abc_0_0_123.jpg",  # Non-numeric age
            ".hidden.jpg",  # Hidden file
        ]

        for filename in invalid_files:
            img = Image.new("RGB", (50, 50), color="white")
            img.save(tmp_path / filename)

        dataset = AgeDataset(str(tmp_path), verbose=False)

        assert len(dataset) == 0

    def test_dataset_empty_directory(self, tmp_path):
        """Test dataset handles empty directory."""
        dataset = AgeDataset(str(tmp_path), verbose=False)
        assert len(dataset) == 0
