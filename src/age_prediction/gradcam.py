"""
Grad-CAM (Gradient-weighted Class Activation Mapping) visualization.

Based on: "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization"
Selvaraju et al., 2017 (https://arxiv.org/abs/1610.02391)
"""

from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn as nn


def denormalize_image(
    image_tensor: torch.Tensor,
    mean: tuple[float, ...] = (0.485, 0.456, 0.406),
    std: tuple[float, ...] = (0.229, 0.224, 0.225),
) -> np.ndarray:
    """
    Convert a normalized tensor image back to displayable format.

    Args:
        image_tensor: Normalized image tensor (C, H, W)
        mean: Normalization mean
        std: Normalization std

    Returns:
        Denormalized image as numpy array (H, W, C) in uint8 format
    """
    mean = np.array(mean)
    std = np.array(std)

    image = image_tensor.cpu().detach().numpy()

    # Handle batch dimension if present
    if image.ndim == 4:
        image = image[0]

    # Transpose from (C, H, W) to (H, W, C)
    image = image.transpose(1, 2, 0)

    # Denormalize
    image = (image * std) + mean
    image = np.clip(image, 0, 1)

    return (image * 255).astype(np.uint8)


class GradCAM:
    """
    Grad-CAM implementation for visualizing CNN activations.

    Usage:
        gradcam = GradCAM(model, target_layer)
        heatmap, prediction = gradcam(image_tensor)
        overlay = gradcam.overlay_heatmap(image_tensor, heatmap)
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        """
        Initialize Grad-CAM.

        Args:
            model: PyTorch model
            target_layer: Target convolutional layer for visualization
        """
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # Register hooks
        self._register_hooks()

    def _register_hooks(self):
        """Register forward and backward hooks on target layer."""

        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.forward_handle = self.target_layer.register_forward_hook(forward_hook)
        self.backward_handle = self.target_layer.register_full_backward_hook(backward_hook)

    def remove_hooks(self):
        """Remove registered hooks."""
        self.forward_handle.remove()
        self.backward_handle.remove()

    def __call__(
        self, image_tensor: torch.Tensor, class_idx: Optional[int] = None
    ) -> tuple[np.ndarray, int]:
        """
        Generate Grad-CAM heatmap for an image.

        Args:
            image_tensor: Input image tensor (C, H, W) or (1, C, H, W)
            class_idx: Target class index (uses predicted class if None)

        Returns:
            Tuple of (heatmap array, predicted/target class index)
        """
        self.model.eval()

        # Ensure batch dimension
        if image_tensor.dim() == 3:
            image_tensor = image_tensor.unsqueeze(0)

        # Get device
        device = next(self.model.parameters()).device
        image_tensor = image_tensor.to(device).requires_grad_(True)

        # Forward pass
        output = self.model(image_tensor)

        # Get target class
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        # Backward pass
        self.model.zero_grad()
        target = output[0, class_idx]
        target.backward()

        if self.gradients is None:
            raise RuntimeError("Gradients not captured. Check target layer.")

        # Compute Grad-CAM
        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])

        # Weight activations by gradients
        for i in range(self.activations.shape[1]):
            self.activations[:, i, :, :] *= pooled_gradients[i]

        # Generate heatmap
        heatmap = torch.mean(self.activations, dim=1).squeeze()
        heatmap = heatmap.cpu().numpy()

        # ReLU and normalize
        heatmap = np.maximum(heatmap, 0)
        heatmap = heatmap / (np.max(heatmap) + 1e-8)

        return heatmap, class_idx

    def overlay_heatmap(
        self,
        image_tensor: torch.Tensor,
        heatmap: np.ndarray,
        alpha: float = 0.4,
        colormap: int = cv2.COLORMAP_JET,
    ) -> np.ndarray:
        """
        Overlay heatmap on original image.

        Args:
            image_tensor: Original image tensor
            heatmap: Grad-CAM heatmap
            alpha: Heatmap transparency (0-1)
            colormap: OpenCV colormap

        Returns:
            Blended image as numpy array (H, W, 3) in RGB format
        """
        # Get original image
        original = denormalize_image(image_tensor)
        h, w = original.shape[:2]

        # Resize heatmap to image size
        heatmap_resized = cv2.resize(heatmap, (w, h))
        heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), colormap)

        # Blend
        blended = cv2.addWeighted(original, 1 - alpha, heatmap_colored, alpha, 0)

        # Convert BGR to RGB
        blended = cv2.cvtColor(blended, cv2.COLOR_BGR2RGB)

        return blended


def apply_gradcam(
    model: nn.Module,
    image_tensor: torch.Tensor,
    target_layer: nn.Module,
    class_idx: Optional[int] = None,
    show: bool = True,
    save_path: Optional[str] = None,
) -> tuple[np.ndarray, int]:
    """
    Apply Grad-CAM to an image and optionally display/save.

    Args:
        model: PyTorch model
        image_tensor: Input image tensor
        target_layer: Target convolutional layer
        class_idx: Target class (uses prediction if None)
        show: Whether to display the result
        save_path: Path to save the visualization

    Returns:
        Tuple of (overlay image, predicted/target class)
    """
    gradcam = GradCAM(model, target_layer)

    try:
        heatmap, pred_class = gradcam(image_tensor, class_idx)
        overlay = gradcam.overlay_heatmap(image_tensor, heatmap)
    finally:
        gradcam.remove_hooks()

    if show or save_path:
        import matplotlib.pyplot as plt

        from .config import AGE_GROUPS

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))

        # Original image
        original = denormalize_image(image_tensor)
        axes[0].imshow(original)
        axes[0].set_title("Original Image")
        axes[0].axis("off")

        # Heatmap
        heatmap_resized = cv2.resize(heatmap, (original.shape[1], original.shape[0]))
        axes[1].imshow(heatmap_resized, cmap="jet")
        axes[1].set_title("Grad-CAM Heatmap")
        axes[1].axis("off")

        # Overlay
        axes[2].imshow(overlay)
        pred_label = AGE_GROUPS[pred_class]["display"]
        axes[2].set_title(f"Overlay (Pred: {pred_label})")
        axes[2].axis("off")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Grad-CAM visualization saved to: {save_path}")

        if show:
            plt.show()

        plt.close()

    return overlay, pred_class


def batch_gradcam(
    model: nn.Module,
    images: list,
    target_layer: nn.Module,
    num_cols: int = 5,
    save_path: Optional[str] = None,
):
    """
    Apply Grad-CAM to multiple images and display in a grid.

    Args:
        model: PyTorch model
        images: List of image tensors
        target_layer: Target convolutional layer
        num_cols: Number of columns in the grid
        save_path: Path to save the visualization
    """
    import matplotlib.pyplot as plt

    from .config import AGE_GROUPS

    gradcam = GradCAM(model, target_layer)
    num_images = len(images)
    num_rows = (num_images + num_cols - 1) // num_cols

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(3 * num_cols, 3 * num_rows))
    axes = axes.flatten() if num_images > 1 else [axes]

    try:
        for i, image_tensor in enumerate(images):
            heatmap, pred_class = gradcam(image_tensor)
            overlay = gradcam.overlay_heatmap(image_tensor, heatmap)

            axes[i].imshow(overlay)
            pred_label = AGE_GROUPS[pred_class]["display"]
            axes[i].set_title(f"Pred: {pred_label}", fontsize=10)
            axes[i].axis("off")

        # Hide unused axes
        for i in range(num_images, len(axes)):
            axes[i].axis("off")

    finally:
        gradcam.remove_hooks()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Batch Grad-CAM saved to: {save_path}")

    plt.show()
    plt.close()
