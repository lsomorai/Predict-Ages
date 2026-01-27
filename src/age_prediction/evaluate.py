"""
Evaluation utilities and metrics for age prediction models.
"""

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .config import AGE_GROUPS, NUM_CLASSES


def evaluate_model(
    model: nn.Module, dataloader: DataLoader, device: Optional[torch.device] = None
) -> tuple[float, np.ndarray, np.ndarray]:
    """
    Evaluate a model on a dataloader.

    Args:
        model: PyTorch model
        dataloader: DataLoader for evaluation
        device: Device to use (auto-detected if None)

    Returns:
        Tuple of (accuracy, all_predictions, all_labels)
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    accuracy = np.mean(all_preds == all_labels)

    return accuracy, all_preds, all_labels, all_probs


def compute_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, y_probs: Optional[np.ndarray] = None
) -> dict:
    """
    Compute comprehensive classification metrics.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_probs: Prediction probabilities (optional, for ROC-AUC)

    Returns:
        Dictionary with metrics
    """
    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
    )

    class_names = [AGE_GROUPS[i]["display"] for i in range(NUM_CLASSES)]

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "precision_weighted": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall_weighted": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred),
        "classification_report": classification_report(
            y_true, y_pred, target_names=class_names, zero_division=0
        ),
    }

    # Per-class metrics
    precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
    recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
    f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)

    metrics["per_class"] = {
        class_names[i]: {
            "precision": precision_per_class[i],
            "recall": recall_per_class[i],
            "f1": f1_per_class[i],
        }
        for i in range(len(class_names))
    }

    # ROC-AUC if probabilities provided
    if y_probs is not None:
        try:
            from sklearn.metrics import roc_auc_score

            # One-vs-rest ROC-AUC
            metrics["roc_auc_ovr"] = roc_auc_score(
                y_true, y_probs, multi_class="ovr", average="macro"
            )
        except ValueError:
            metrics["roc_auc_ovr"] = None

    return metrics


def print_metrics(metrics: dict) -> None:
    """Print metrics in a formatted way."""
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)

    print(f"\nOverall Accuracy: {metrics['accuracy']:.4f} ({metrics['accuracy'] * 100:.2f}%)")

    print("\nMacro-averaged Metrics:")
    print(f"  Precision: {metrics['precision_macro']:.4f}")
    print(f"  Recall:    {metrics['recall_macro']:.4f}")
    print(f"  F1-Score:  {metrics['f1_macro']:.4f}")

    if metrics.get("roc_auc_ovr"):
        print(f"  ROC-AUC:   {metrics['roc_auc_ovr']:.4f}")

    print("\nPer-Class Performance:")
    for class_name, class_metrics in metrics["per_class"].items():
        print(f"  {class_name}:")
        print(f"    Precision: {class_metrics['precision']:.4f}")
        print(f"    Recall:    {class_metrics['recall']:.4f}")
        print(f"    F1-Score:  {class_metrics['f1']:.4f}")

    print("\n" + "=" * 60)


def plot_confusion_matrix(
    confusion_matrix: np.ndarray, title: str = "Confusion Matrix", save_path: Optional[str] = None
):
    """
    Plot a confusion matrix.

    Args:
        confusion_matrix: Confusion matrix array
        title: Plot title
        save_path: Path to save the figure
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    class_names = [AGE_GROUPS[i]["display"] for i in range(NUM_CLASSES)]

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        confusion_matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.xlabel("Predicted Age Group")
    plt.ylabel("True Age Group")
    plt.title(title)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Confusion matrix saved to: {save_path}")

    return plt.gcf()


def plot_roc_curves(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    title: str = "ROC Curves",
    save_path: Optional[str] = None,
):
    """
    Plot ROC curves for all classes.

    Args:
        y_true: True labels
        y_probs: Prediction probabilities
        title: Plot title
        save_path: Path to save the figure
    """
    import matplotlib.pyplot as plt
    from sklearn.metrics import auc, roc_curve
    from sklearn.preprocessing import label_binarize

    class_names = [AGE_GROUPS[i]["display"] for i in range(NUM_CLASSES)]

    # Binarize labels for multi-class ROC
    y_true_bin = label_binarize(y_true, classes=list(range(NUM_CLASSES)))

    plt.figure(figsize=(10, 8))

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    for i in range(NUM_CLASSES):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_probs[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color=colors[i], lw=2, label=f"{class_names[i]} (AUC = {roc_auc:.3f})")

    plt.plot([0, 1], [0, 1], "k--", lw=2, label="Random (AUC = 0.500)")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"ROC curves saved to: {save_path}")

    return plt.gcf()


def compare_models(
    models: dict[str, nn.Module], dataloader: DataLoader, device: Optional[torch.device] = None
) -> dict[str, dict]:
    """
    Compare multiple models on the same dataset.

    Args:
        models: Dictionary mapping model names to model instances
        dataloader: DataLoader for evaluation
        device: Device to use

    Returns:
        Dictionary mapping model names to their metrics
    """
    results = {}

    for name, model in models.items():
        print(f"\nEvaluating {name}...")
        accuracy, preds, labels, probs = evaluate_model(model, dataloader, device)
        metrics = compute_metrics(labels, preds, probs)
        results[name] = metrics
        print(f"  Accuracy: {accuracy:.4f}")

    # Summary comparison
    print("\n" + "=" * 60)
    print("MODEL COMPARISON SUMMARY")
    print("=" * 60)
    print(f"{'Model':<20} {'Accuracy':<12} {'F1 (Macro)':<12} {'ROC-AUC':<12}")
    print("-" * 56)

    for name, metrics in results.items():
        roc_auc = metrics.get("roc_auc_ovr", "N/A")
        if isinstance(roc_auc, float):
            roc_auc = f"{roc_auc:.4f}"
        print(f"{name:<20} {metrics['accuracy']:<12.4f} {metrics['f1_macro']:<12.4f} {roc_auc:<12}")

    return results
