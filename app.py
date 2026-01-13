"""
Gradio web interface for facial age prediction.

Run with: python app.py
"""

import os
from typing import Optional

import gradio as gr
import numpy as np
from PIL import Image

# Check if running in HF Spaces
IS_HF_SPACE = os.environ.get("SPACE_ID") is not None

# Import our modules (after env check)
from src.age_prediction import (  # noqa: E402
    AGE_GROUPS,
    AgePredictor,
)
from src.age_prediction.models import get_available_models  # noqa: E402

# Global predictor cache
_predictors = {}


def get_predictor(model_name: str) -> AgePredictor:
    """Get or create a predictor for the given model."""
    if model_name not in _predictors:
        checkpoint_dir = "./checkpoints"
        weights_path = os.path.join(checkpoint_dir, f"best_{model_name}.pth")

        # Check if weights exist
        if not os.path.exists(weights_path):
            weights_path = None  # Use pretrained weights only

        _predictors[model_name] = AgePredictor(
            model_name=model_name,
            weights_path=weights_path
        )

    return _predictors[model_name]


def predict_age(
    image: np.ndarray,
    model_name: str,
    show_gradcam: bool
) -> tuple[dict, Optional[np.ndarray], str]:
    """
    Predict age from an image.

    Args:
        image: Input image as numpy array
        model_name: Model to use
        show_gradcam: Whether to generate Grad-CAM visualization

    Returns:
        Tuple of (confidence_dict, gradcam_image, result_text)
    """
    if image is None:
        return {}, None, "Please upload an image"

    try:
        predictor = get_predictor(model_name)

        # Convert to PIL Image
        pil_image = Image.fromarray(image)

        # Get prediction
        result = predictor.predict(pil_image, return_probs=True)

        # Format confidence scores for Gradio
        confidences = {
            f"{AGE_GROUPS[i]['display']} ({AGE_GROUPS[i]['label']})": result["probabilities"][AGE_GROUPS[i]["display"]]
            for i in range(len(AGE_GROUPS))
        }

        # Generate result text
        result_text = (
            f"**Predicted Age Group:** {result['age_range']} ({result['class_label']})\n\n"
            f"**Confidence:** {result['confidence']:.1%}"
        )

        # Generate Grad-CAM if requested
        gradcam_image = None
        if show_gradcam and model_name != "ensemble":
            try:
                _, gradcam_image = predictor.predict_with_gradcam(pil_image)
            except Exception as e:
                result_text += f"\n\n*Grad-CAM error: {str(e)}*"

        return confidences, gradcam_image, result_text

    except Exception as e:
        return {}, None, f"Error: {str(e)}"


def create_demo() -> gr.Blocks:
    """Create the Gradio demo interface."""

    # Custom CSS for better styling
    css = """
    .gradio-container {
        max-width: 1200px !important;
    }
    .result-text {
        font-size: 1.2em;
        padding: 1em;
        background: #f7f7f7;
        border-radius: 8px;
    }
    """

    with gr.Blocks(css=css, title="Facial Age Prediction") as demo:
        gr.Markdown(
            """
            # Facial Age Prediction with Deep Learning

            Upload a face image to predict the age group using state-of-the-art CNN models.

            **Models Available:**
            - **MobileNetV2** - Fast, lightweight model
            - **ResNet50** - Best accuracy (72.38%)
            - **EfficientNet-B0** - Efficient and accurate

            **Age Groups:** 0-25 (Young), 26-50 (Adult), 51-75 (Middle-aged), 76-116 (Senior)
            """
        )

        with gr.Row():
            with gr.Column(scale=1):
                # Input section
                image_input = gr.Image(
                    label="Upload Face Image",
                    type="numpy",
                    height=300
                )

                model_dropdown = gr.Dropdown(
                    choices=get_available_models(),
                    value="resnet50",
                    label="Select Model",
                    info="ResNet50 has the best accuracy"
                )

                gradcam_checkbox = gr.Checkbox(
                    label="Show Grad-CAM Visualization",
                    value=True,
                    info="Highlights regions the model focuses on"
                )

                predict_btn = gr.Button("Predict Age", variant="primary", size="lg")

            with gr.Column(scale=1):
                # Output section
                result_text = gr.Markdown(
                    label="Result",
                    elem_classes=["result-text"]
                )

                confidence_output = gr.Label(
                    label="Confidence Scores",
                    num_top_classes=4
                )

                gradcam_output = gr.Image(
                    label="Grad-CAM Visualization",
                    height=300
                )

        # Example images (if available)
        gr.Markdown("### Examples")
        gr.Markdown("*Upload your own face image to try the model*")

        # Connect the predict button
        predict_btn.click(
            fn=predict_age,
            inputs=[image_input, model_dropdown, gradcam_checkbox],
            outputs=[confidence_output, gradcam_output, result_text]
        )

        # Also predict on image upload
        image_input.change(
            fn=predict_age,
            inputs=[image_input, model_dropdown, gradcam_checkbox],
            outputs=[confidence_output, gradcam_output, result_text]
        )

        gr.Markdown(
            """
            ---
            ### About

            This project uses **transfer learning** with ImageNet-pretrained CNNs.
            The classifier head is trained on the [UTKFace dataset](https://www.kaggle.com/datasets/arashnic/faces-age-detection-dataset).

            **Features:**
            - 3 model architectures compared
            - Grad-CAM visualizations for interpretability
            - Production-ready codebase with tests and CI/CD

            [View on GitHub](https://github.com/lsomorai/Predict-Ages) |
            Built with PyTorch and Gradio
            """
        )

    return demo


# Create the demo
demo = create_demo()

if __name__ == "__main__":
    # Launch settings
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=IS_HF_SPACE,  # Auto-share if running in HF Spaces
        show_error=True
    )
