"""
Training utilities with Weights & Biases integration.
"""

import os
from dataclasses import asdict
from typing import Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau, StepLR
from torch.utils.data import DataLoader

from .config import Config


class TrainingHistory:
    """Stores training metrics history."""

    def __init__(self):
        self.train_loss: list[float] = []
        self.train_acc: list[float] = []
        self.val_loss: list[float] = []
        self.val_acc: list[float] = []
        self.learning_rates: list[float] = []

    def append(
        self, train_loss: float, train_acc: float, val_loss: float, val_acc: float, lr: float
    ):
        self.train_loss.append(train_loss)
        self.train_acc.append(train_acc)
        self.val_loss.append(val_loss)
        self.val_acc.append(val_acc)
        self.learning_rates.append(lr)

    def to_dict(self) -> dict:
        return {
            "train_loss": self.train_loss,
            "train_acc": self.train_acc,
            "val_loss": self.val_loss,
            "val_acc": self.val_acc,
            "learning_rates": self.learning_rates,
        }


def get_optimizer(model: nn.Module, config: Config) -> optim.Optimizer:
    """
    Create optimizer for training.

    Args:
        model: PyTorch model
        config: Configuration object

    Returns:
        Optimizer instance
    """
    # Only optimize parameters that require gradients
    params = filter(lambda p: p.requires_grad, model.parameters())

    return optim.Adam(params, lr=config.learning_rate, weight_decay=config.weight_decay)


def get_scheduler(optimizer: optim.Optimizer, config: Config, num_batches: int) -> Optional[object]:
    """
    Create learning rate scheduler.

    Args:
        optimizer: Optimizer instance
        config: Configuration object
        num_batches: Number of batches per epoch

    Returns:
        Scheduler instance or None
    """
    if not config.use_scheduler:
        return None

    if config.scheduler_type == "cosine":
        return CosineAnnealingLR(
            optimizer, T_max=config.epochs, eta_min=config.learning_rate * 0.01
        )
    elif config.scheduler_type == "step":
        return StepLR(optimizer, step_size=5, gamma=0.5)
    elif config.scheduler_type == "plateau":
        return ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2, verbose=True)
    else:
        return None


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    """
    Train for one epoch.

    Returns:
        Tuple of (average loss, accuracy)
    """
    model.train()
    running_loss = 0.0
    running_corrects = 0
    total_samples = 0

    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        _, preds = torch.max(outputs, 1)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data).item()
        total_samples += inputs.size(0)

    epoch_loss = running_loss / total_samples
    epoch_acc = running_corrects / total_samples

    return epoch_loss, epoch_acc


def validate_epoch(
    model: nn.Module, dataloader: DataLoader, criterion: nn.Module, device: torch.device
) -> tuple[float, float]:
    """
    Validate for one epoch.

    Returns:
        Tuple of (average loss, accuracy)
    """
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    total_samples = 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data).item()
            total_samples += inputs.size(0)

    epoch_loss = running_loss / total_samples
    epoch_acc = running_corrects / total_samples

    return epoch_loss, epoch_acc


def train_model(
    model: nn.Module,
    dataloaders: dict[str, DataLoader],
    config: Config,
    model_name: str,
    use_wandb: bool = False,
    checkpoint_dir: Optional[str] = None,
) -> tuple[nn.Module, TrainingHistory]:
    """
    Train a model with full training loop.

    Args:
        model: PyTorch model to train
        dataloaders: Dictionary with "train" and "val" dataloaders
        config: Configuration object
        model_name: Name of the model (for saving)
        use_wandb: Whether to log to Weights & Biases
        checkpoint_dir: Directory to save checkpoints

    Returns:
        Tuple of (trained model, training history)
    """
    device = torch.device(config.get_device())
    model = model.to(device)

    # Setup checkpoint directory
    if checkpoint_dir is None:
        checkpoint_dir = config.checkpoint_dir
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Initialize W&B if requested
    wandb_run = None
    if use_wandb:
        try:
            import wandb

            wandb_run = wandb.init(
                project=config.wandb_project,
                entity=config.wandb_entity,
                name=f"{model_name}-run",
                config=asdict(config),
            )
        except ImportError:
            print("Warning: wandb not installed. Skipping W&B logging.")
            use_wandb = False

    # Setup training components
    criterion = nn.CrossEntropyLoss()
    optimizer = get_optimizer(model, config)
    scheduler = get_scheduler(optimizer, config, len(dataloaders["train"]))

    history = TrainingHistory()
    best_acc = 0.0
    best_model_path = os.path.join(checkpoint_dir, f"best_{model_name}.pth")

    print(f"\nTraining {model_name} on {device}")
    print(f"{'=' * 50}")

    for epoch in range(config.epochs):
        print(f"\nEpoch {epoch + 1}/{config.epochs}")
        print("-" * 30)

        # Training phase
        train_loss, train_acc = train_epoch(
            model, dataloaders["train"], criterion, optimizer, device
        )
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")

        # Validation phase
        val_loss, val_acc = validate_epoch(model, dataloaders["val"], criterion, device)
        print(f"Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f}")

        # Get current learning rate
        current_lr = optimizer.param_groups[0]["lr"]

        # Update history
        history.append(train_loss, train_acc, val_loss, val_acc, current_lr)

        # Log to W&B
        if use_wandb and wandb_run:
            import wandb

            wandb.log(
                {
                    "epoch": epoch + 1,
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "learning_rate": current_lr,
                }
            )

        # Update scheduler
        if scheduler is not None:
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(val_acc)
            else:
                scheduler.step()

        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), best_model_path)
            print(f"  -> New best model saved! (Val Acc: {val_acc:.4f})")

    print(f"\n{'=' * 50}")
    print(f"Training complete! Best Val Acc: {best_acc:.4f}")
    print(f"Best model saved to: {best_model_path}")

    # Load best model
    model.load_state_dict(torch.load(best_model_path, weights_only=True))

    # Finish W&B run
    if use_wandb and wandb_run:
        import wandb

        wandb.finish()

    return model, history


def plot_training_curves(history: TrainingHistory, save_path: Optional[str] = None):
    """
    Plot training and validation curves.

    Args:
        history: TrainingHistory object
        save_path: Path to save the figure (optional)
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    epochs = range(1, len(history.train_loss) + 1)

    # Loss curves
    axes[0].plot(epochs, history.train_loss, "b-", label="Train")
    axes[0].plot(epochs, history.val_loss, "r-", label="Validation")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training & Validation Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Accuracy curves
    axes[1].plot(epochs, history.train_acc, "b-", label="Train")
    axes[1].plot(epochs, history.val_acc, "r-", label="Validation")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Training & Validation Accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Learning rate
    axes[2].plot(epochs, history.learning_rates, "g-")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Learning Rate")
    axes[2].set_title("Learning Rate Schedule")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Training curves saved to: {save_path}")

    return fig
