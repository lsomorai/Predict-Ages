"""
Configuration settings for age prediction models.
"""

from dataclasses import dataclass

# Age group definitions
AGE_GROUPS = {
    0: {"range": (0, 25), "label": "Young", "display": "0-25"},
    1: {"range": (26, 50), "label": "Adult", "display": "26-50"},
    2: {"range": (51, 75), "label": "Middle-aged", "display": "51-75"},
    3: {"range": (76, 116), "label": "Senior", "display": "76-116"},
}

NUM_CLASSES = len(AGE_GROUPS)


@dataclass
class Config:
    """Training and model configuration."""

    # Model settings
    model_name: str = "resnet50"
    num_classes: int = NUM_CLASSES
    pretrained: bool = True
    freeze_backbone: bool = True

    # Training settings
    epochs: int = 10
    batch_size: int = 32
    learning_rate: float = 0.001
    weight_decay: float = 0.0

    # Learning rate scheduler
    use_scheduler: bool = True
    scheduler_type: str = "cosine"  # "cosine", "step", "plateau"

    # Data settings
    image_size: int = 224
    train_split: float = 0.7
    val_split: float = 0.15
    test_split: float = 0.15
    num_workers: int = 2

    # Augmentation settings
    use_augmentation: bool = True
    horizontal_flip_prob: float = 0.5
    rotation_degrees: int = 15
    color_jitter: bool = True
    random_erasing_prob: float = 0.1

    # ImageNet normalization
    mean: tuple[float, float, float] = (0.485, 0.456, 0.406)
    std: tuple[float, float, float] = (0.229, 0.224, 0.225)

    # Dropout
    dropout_rate: float = 0.3

    # Paths
    data_dir: str = "./data"
    checkpoint_dir: str = "./checkpoints"

    # Weights & Biases
    wandb_project: str = "age-prediction"
    wandb_entity: str = None

    # Device
    device: str = "auto"  # "auto", "cuda", "cpu", "mps"

    def get_device(self) -> str:
        """Get the appropriate device for training."""
        import torch

        if self.device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
            else:
                return "cpu"
        return self.device


def get_age_group(age: int) -> int:
    """Convert an age to its group index."""
    for group_idx, group_info in AGE_GROUPS.items():
        min_age, max_age = group_info["range"]
        if min_age <= age <= max_age:
            return group_idx
    return -1


def get_age_label(group_idx: int) -> str:
    """Get the display label for an age group."""
    if group_idx in AGE_GROUPS:
        return AGE_GROUPS[group_idx]["display"]
    return "Unknown"
