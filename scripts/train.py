#!/usr/bin/env python3
"""
CLI script for training age prediction models.

Usage:
    python scripts/train.py --model resnet50 --epochs 10 --wandb
    python scripts/train.py --model all --data-dir ./data
"""

import argparse
import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.age_prediction import Config, create_dataloaders, create_model
from src.age_prediction.evaluate import (
    compute_metrics,
    evaluate_model,
    plot_confusion_matrix,
    print_metrics,
)
from src.age_prediction.models import get_available_models
from src.age_prediction.train import plot_training_curves, train_model


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train age prediction models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Model settings
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default="resnet50",
        choices=get_available_models() + ["all"],
        help="Model architecture to train",
    )

    # Data settings
    parser.add_argument(
        "--data-dir", "-d", type=str, default="./data", help="Path to dataset directory"
    )

    # Training settings
    parser.add_argument("--epochs", "-e", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch-size", "-b", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", "--learning-rate", type=float, default=0.001, help="Learning rate")

    # Output settings
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="./checkpoints",
        help="Directory to save model checkpoints",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./outputs",
        help="Directory to save training outputs (plots, logs)",
    )

    # Weights & Biases
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument(
        "--wandb-project", type=str, default="age-prediction", help="W&B project name"
    )

    # Device
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu", "mps"],
        help="Device to use for training",
    )

    # Augmentation
    parser.add_argument("--no-augmentation", action="store_true", help="Disable data augmentation")

    return parser.parse_args()


def main():
    args = parse_args()

    # Create config from arguments
    config = Config(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        data_dir=args.data_dir,
        checkpoint_dir=args.checkpoint_dir,
        wandb_project=args.wandb_project,
        device=args.device,
        use_augmentation=not args.no_augmentation,
    )

    # Create output directories
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    # Determine which models to train
    if args.model == "all":
        models_to_train = get_available_models()
    else:
        models_to_train = [args.model]

    print(f"\n{'=' * 60}")
    print("AGE PREDICTION MODEL TRAINING")
    print(f"{'=' * 60}")
    print(f"Models: {', '.join(models_to_train)}")
    print(f"Device: {config.get_device()}")
    print(f"Epochs: {config.epochs}")
    print(f"Batch size: {config.batch_size}")
    print(f"Learning rate: {config.learning_rate}")
    print(f"Data directory: {config.data_dir}")
    print(f"{'=' * 60}\n")

    # Create dataloaders
    print("Loading dataset...")
    dataloaders = create_dataloaders(config, args.data_dir)

    # Train each model
    results = {}
    for model_name in models_to_train:
        print(f"\n{'=' * 60}")
        print(f"Training {model_name.upper()}")
        print(f"{'=' * 60}")

        # Create model
        model = create_model(model_name, config)

        # Train
        model, history = train_model(
            model=model,
            dataloaders=dataloaders,
            config=config,
            model_name=model_name,
            use_wandb=args.wandb,
            checkpoint_dir=args.checkpoint_dir,
        )

        # Plot training curves
        curves_path = os.path.join(args.output_dir, f"{model_name}_training_curves.png")
        plot_training_curves(history, save_path=curves_path)

        # Evaluate on test set
        print(f"\nEvaluating {model_name} on test set...")
        accuracy, preds, labels, probs = evaluate_model(model, dataloaders["test"])
        metrics = compute_metrics(labels, preds, probs)
        print_metrics(metrics)

        # Plot confusion matrix
        cm_path = os.path.join(args.output_dir, f"{model_name}_confusion_matrix.png")
        plot_confusion_matrix(
            metrics["confusion_matrix"], title=f"{model_name} Confusion Matrix", save_path=cm_path
        )

        results[model_name] = {
            "accuracy": accuracy,
            "f1_macro": metrics["f1_macro"],
            "history": history.to_dict(),
        }

    # Summary
    print(f"\n{'=' * 60}")
    print("TRAINING COMPLETE - SUMMARY")
    print(f"{'=' * 60}")
    print(f"{'Model':<20} {'Test Accuracy':<15} {'F1 (Macro)':<15}")
    print("-" * 50)
    for model_name, result in results.items():
        print(f"{model_name:<20} {result['accuracy']:<15.4f} {result['f1_macro']:<15.4f}")
    print(f"{'=' * 60}")

    print(f"\nCheckpoints saved to: {args.checkpoint_dir}")
    print(f"Outputs saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
