"""
Dataset classes and data loading utilities for age prediction.
"""

import glob
import os
from collections import Counter
from typing import Optional

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms

from .config import NUM_CLASSES, Config, get_age_group


class AgeDataset(Dataset):
    """
    Custom dataset for facial age classification.

    Expects images with filenames in format: [age]_[gender]_[race]_[datetime].jpg
    (UTKFace dataset format)
    """

    def __init__(
        self, data_dir: str, transform: Optional[transforms.Compose] = None, verbose: bool = True
    ):
        """
        Initialize the dataset.

        Args:
            data_dir: Root directory containing images
            transform: Torchvision transforms to apply
            verbose: Whether to print dataset statistics
        """
        self.data_dir = data_dir
        self.transform = transform
        self.files: list[str] = []
        self.labels: list[int] = []

        self._load_dataset(verbose)

    def _load_dataset(self, verbose: bool) -> None:
        """Load and parse the dataset."""
        # Find all image files recursively
        image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]
        all_images = []

        for ext in image_extensions:
            all_images.extend(glob.glob(os.path.join(self.data_dir, "**", ext), recursive=True))

        if verbose:
            print(f"Found {len(all_images)} total images")

        # Parse each image
        for img_path in all_images:
            filename = os.path.basename(img_path)

            # Skip hidden files
            if filename.startswith("."):
                continue

            try:
                # Extract age from filename (format: age_gender_race_datetime.jpg)
                age = int(filename.split("_")[0])
                group_idx = get_age_group(age)

                if group_idx != -1:
                    self.files.append(img_path)
                    self.labels.append(group_idx)
            except (ValueError, IndexError):
                # Skip files that don't match the expected format
                continue

        if verbose:
            print(f"Loaded {len(self.files)} valid images with age labels")
            self._print_distribution()

    def _print_distribution(self) -> None:
        """Print class distribution."""
        from .config import AGE_GROUPS

        dist = Counter(self.labels)
        print("Class distribution:")
        for idx in sorted(dist.keys()):
            info = AGE_GROUPS[idx]
            print(f"  Class {idx} ({info['display']}): {dist[idx]} images")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        img_path = self.files[idx]
        image = Image.open(img_path).convert("RGB")
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

    def get_class_weights(self) -> torch.Tensor:
        """
        Calculate class weights for imbalanced dataset handling.

        Returns:
            Tensor of weights inversely proportional to class frequency
        """
        counts = Counter(self.labels)
        total = len(self.labels)
        weights = []

        for i in range(NUM_CLASSES):
            if counts[i] > 0:
                weights.append(total / (NUM_CLASSES * counts[i]))
            else:
                weights.append(1.0)

        return torch.FloatTensor(weights)


def get_transforms(config: Config, mode: str = "train") -> transforms.Compose:
    """
    Get data transforms based on mode.

    Args:
        config: Configuration object
        mode: One of "train", "val", or "test"

    Returns:
        Composed transforms
    """
    if mode == "train" and config.use_augmentation:
        transform_list = [
            transforms.Resize((config.image_size, config.image_size)),
            transforms.RandomHorizontalFlip(p=config.horizontal_flip_prob),
            transforms.RandomRotation(degrees=config.rotation_degrees),
        ]

        if config.color_jitter:
            transform_list.append(
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
            )

        transform_list.extend(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=config.mean, std=config.std),
            ]
        )

        if config.random_erasing_prob > 0:
            transform_list.append(transforms.RandomErasing(p=config.random_erasing_prob))

        return transforms.Compose(transform_list)
    else:
        # Validation and test transforms (no augmentation)
        return transforms.Compose(
            [
                transforms.Resize((config.image_size, config.image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=config.mean, std=config.std),
            ]
        )


def create_dataloaders(config: Config, data_dir: Optional[str] = None) -> dict[str, DataLoader]:
    """
    Create train, validation, and test dataloaders.

    Args:
        config: Configuration object
        data_dir: Override data directory from config

    Returns:
        Dictionary with "train", "val", and "test" dataloaders
    """
    data_dir = data_dir or config.data_dir

    # Load full dataset with training transforms
    full_dataset = AgeDataset(data_dir, transform=get_transforms(config, "train"))

    # Calculate split sizes
    total_size = len(full_dataset)
    train_size = int(config.train_split * total_size)
    val_size = int(config.val_split * total_size)
    test_size = total_size - train_size - val_size

    # Split dataset
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42),  # Reproducible splits
    )

    print(f"\nDataset splits: Train={train_size}, Val={val_size}, Test={test_size}")

    # Create dataloaders
    dataloaders = {
        "train": DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            pin_memory=True,
        ),
        "val": DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=True,
        ),
        "test": DataLoader(
            test_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=True,
        ),
    }

    return dataloaders
