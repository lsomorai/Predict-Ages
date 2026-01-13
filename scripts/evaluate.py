#!/usr/bin/env python3
"""
CLI script for evaluating age prediction models.

Usage:
    python scripts/evaluate.py --model resnet50 --data-dir ./data
    python scripts/evaluate.py --model all --compare
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from src.age_prediction import Config, create_dataloaders, create_model
from src.age_prediction.evaluate import (
    compare_models,
    compute_metrics,
    evaluate_model,
    plot_confusion_matrix,
    plot_roc_curves,
    print_metrics,
)
from src.age_prediction.models import get_available_models


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate age prediction models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--model", "-m",
        type=str,
        default="resnet50",
        choices=get_available_models() + ["all"],
        help="Model to evaluate"
    )

    parser.add_argument(
        "--data-dir", "-d",
        type=str,
        default="./data",
        help="Path to dataset"
    )

    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="./checkpoints",
        help="Directory with model checkpoints"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="./outputs",
        help="Directory to save outputs"
    )

    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare all models side by side"
    )

    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu", "mps"],
        help="Device to use"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    config = Config(device=args.device, data_dir=args.data_dir)
    device = torch.device(config.get_device())

    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading dataset...")
    dataloaders = create_dataloaders(config, args.data_dir)

    if args.compare or args.model == "all":
        # Load all models for comparison
        models = {}
        for model_name in get_available_models():
            weights_path = os.path.join(args.checkpoint_dir, f"best_{model_name}.pth")
            model = create_model(model_name, config)

            if os.path.exists(weights_path):
                state_dict = torch.load(weights_path, map_location=device, weights_only=True)
                model.load_state_dict(state_dict)
                print(f"Loaded weights for {model_name}")
            else:
                print(f"No weights found for {model_name}, using pretrained")

            model = model.to(device)
            model.eval()
            models[model_name] = model

        # Compare models
        compare_models(models, dataloaders["test"], device)

        # Generate visualizations for each
        for model_name, model in models.items():
            accuracy, preds, labels, probs = evaluate_model(model, dataloaders["test"], device)

            # Confusion matrix
            metrics = compute_metrics(labels, preds, probs)
            cm_path = os.path.join(args.output_dir, f"{model_name}_confusion_matrix.png")
            plot_confusion_matrix(metrics["confusion_matrix"], f"{model_name} Confusion Matrix", cm_path)

            # ROC curves
            roc_path = os.path.join(args.output_dir, f"{model_name}_roc_curves.png")
            plot_roc_curves(labels, probs, f"{model_name} ROC Curves", roc_path)

    else:
        # Single model evaluation
        model_name = args.model
        weights_path = os.path.join(args.checkpoint_dir, f"best_{model_name}.pth")

        model = create_model(model_name, config)

        if os.path.exists(weights_path):
            state_dict = torch.load(weights_path, map_location=device, weights_only=True)
            model.load_state_dict(state_dict)
            print(f"Loaded weights from {weights_path}")
        else:
            print(f"No weights found at {weights_path}, using pretrained only")

        model = model.to(device)
        model.eval()

        print(f"\nEvaluating {model_name}...")
        accuracy, preds, labels, probs = evaluate_model(model, dataloaders["test"], device)
        metrics = compute_metrics(labels, preds, probs)
        print_metrics(metrics)

        # Save visualizations
        cm_path = os.path.join(args.output_dir, f"{model_name}_confusion_matrix.png")
        plot_confusion_matrix(metrics["confusion_matrix"], f"{model_name} Confusion Matrix", cm_path)

        roc_path = os.path.join(args.output_dir, f"{model_name}_roc_curves.png")
        plot_roc_curves(labels, probs, f"{model_name} ROC Curves", roc_path)

    print(f"\nOutputs saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
