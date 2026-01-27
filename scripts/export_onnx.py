#!/usr/bin/env python3
"""
Export PyTorch models to ONNX format for cross-platform deployment.

Usage:
    python scripts/export_onnx.py --model resnet50
    python scripts/export_onnx.py --model all --quantize
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from src.age_prediction import Config, create_model
from src.age_prediction.models import get_available_models


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export models to ONNX format",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default="resnet50",
        choices=get_available_models() + ["all"],
        help="Model to export",
    )

    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="./checkpoints",
        help="Directory with model checkpoints",
    )

    parser.add_argument(
        "--output-dir", type=str, default="./exports", help="Directory to save ONNX models"
    )

    parser.add_argument(
        "--quantize", action="store_true", help="Apply INT8 quantization to ONNX model"
    )

    parser.add_argument(
        "--benchmark", action="store_true", help="Benchmark inference speed after export"
    )

    parser.add_argument(
        "--opset-version",
        type=int,
        default=18,
        help="ONNX opset version (minimum 18 for PyTorch 2.x)",
    )

    return parser.parse_args()


def export_to_onnx(
    model: torch.nn.Module, model_name: str, output_path: str, opset_version: int = 18
) -> str:
    """Export a PyTorch model to ONNX format."""
    model.eval()

    # Create dummy input
    dummy_input = torch.randn(1, 3, 224, 224)

    # Export using legacy exporter (dynamo=False) for better compatibility
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        dynamo=False,  # Use legacy exporter for quantization compatibility
    )

    print(f"Exported {model_name} to: {output_path}")

    # Get file size
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"  Model size: {size_mb:.2f} MB")

    return output_path


def quantize_onnx(input_path: str, output_path: str) -> str:
    """Apply INT8 quantization to an ONNX model."""
    try:
        from onnxruntime.quantization import QuantType, quantize_dynamic
        from onnxruntime.quantization.shape_inference import quant_pre_process
    except ImportError:
        print("  Warning: onnxruntime.quantization not available, skipping quantization")
        return input_path

    # Preprocess model for quantization (fixes shape inference issues)
    preprocessed_path = input_path.replace(".onnx", "_preprocessed.onnx")
    try:
        quant_pre_process(input_path, preprocessed_path, skip_symbolic_shape=True)
        model_to_quantize = preprocessed_path
    except Exception as e:
        print(f"  Warning: Preprocessing failed ({e}), trying direct quantization")
        model_to_quantize = input_path

    quantize_dynamic(model_to_quantize, output_path, weight_type=QuantType.QUInt8)

    # Clean up preprocessed file
    if os.path.exists(preprocessed_path):
        os.remove(preprocessed_path)

    print(f"Quantized model saved to: {output_path}")

    # Compare sizes
    original_size = os.path.getsize(input_path) / (1024 * 1024)
    quantized_size = os.path.getsize(output_path) / (1024 * 1024)
    reduction = (1 - quantized_size / original_size) * 100

    print(f"  Original size: {original_size:.2f} MB")
    print(f"  Quantized size: {quantized_size:.2f} MB")
    print(f"  Size reduction: {reduction:.1f}%")

    return output_path


def benchmark_onnx(onnx_path: str, num_runs: int = 100) -> dict:
    """Benchmark ONNX model inference speed."""
    try:
        import onnxruntime as ort
    except ImportError:
        print("  Warning: onnxruntime not installed, skipping benchmark")
        return {}

    # Create session
    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])

    # Create dummy input
    input_name = session.get_inputs()[0].name
    dummy_input = np.random.randn(1, 3, 224, 224).astype(np.float32)

    # Warmup
    for _ in range(10):
        _ = session.run(None, {input_name: dummy_input})

    # Benchmark
    times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        _ = session.run(None, {input_name: dummy_input})
        end = time.perf_counter()
        times.append((end - start) * 1000)

    times = np.array(times)

    results = {
        "mean_ms": float(np.mean(times)),
        "std_ms": float(np.std(times)),
        "min_ms": float(np.min(times)),
        "max_ms": float(np.max(times)),
        "p95_ms": float(np.percentile(times, 95)),
        "throughput_fps": float(1000 / np.mean(times)),
    }

    print(f"\n  Benchmark Results ({num_runs} runs):")
    print(f"    Mean latency: {results['mean_ms']:.2f} ms")
    print(f"    Std dev: {results['std_ms']:.2f} ms")
    print(f"    Min/Max: {results['min_ms']:.2f} / {results['max_ms']:.2f} ms")
    print(f"    P95 latency: {results['p95_ms']:.2f} ms")
    print(f"    Throughput: {results['throughput_fps']:.1f} FPS")

    return results


def main():
    args = parse_args()

    config = Config()
    os.makedirs(args.output_dir, exist_ok=True)

    # Determine models to export
    if args.model == "all":
        models_to_export = get_available_models()
    else:
        models_to_export = [args.model]

    print(f"\n{'=' * 60}")
    print("ONNX MODEL EXPORT")
    print(f"{'=' * 60}")
    print(f"Models: {', '.join(models_to_export)}")
    print(f"Output directory: {args.output_dir}")
    print(f"Quantization: {'Enabled' if args.quantize else 'Disabled'}")
    print(f"{'=' * 60}\n")

    results = {}

    for model_name in models_to_export:
        print(f"\n--- {model_name.upper()} ---")

        # Load model
        model = create_model(model_name, config)
        weights_path = os.path.join(args.checkpoint_dir, f"best_{model_name}.pth")

        if os.path.exists(weights_path):
            state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
            model.load_state_dict(state_dict)
            print(f"Loaded weights from {weights_path}")
        else:
            print("Using pretrained weights only (no fine-tuned weights found)")

        # Export to ONNX
        onnx_path = os.path.join(args.output_dir, f"{model_name}.onnx")
        export_to_onnx(model, model_name, onnx_path, args.opset_version)

        model_results = {"onnx_path": onnx_path}

        # Quantize if requested
        if args.quantize:
            quantized_path = os.path.join(args.output_dir, f"{model_name}_quantized.onnx")
            quantize_onnx(onnx_path, quantized_path)
            model_results["quantized_path"] = quantized_path

        # Benchmark if requested
        if args.benchmark:
            print("\nBenchmarking original model:")
            model_results["benchmark"] = benchmark_onnx(onnx_path)

            if args.quantize:
                print("\nBenchmarking quantized model:")
                model_results["benchmark_quantized"] = benchmark_onnx(quantized_path)

        results[model_name] = model_results

    print(f"\n{'=' * 60}")
    print("EXPORT COMPLETE")
    print(f"{'=' * 60}")
    print(f"ONNX models saved to: {args.output_dir}")

    # List exported files
    print("\nExported files:")
    for f in sorted(os.listdir(args.output_dir)):
        if f.endswith(".onnx"):
            size_mb = os.path.getsize(os.path.join(args.output_dir, f)) / (1024 * 1024)
            print(f"  - {f} ({size_mb:.2f} MB)")


if __name__ == "__main__":
    main()
