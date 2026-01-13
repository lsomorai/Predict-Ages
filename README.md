# Facial Age Prediction with Deep Learning

[![Python 3.9+](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![CI](https://github.com/lsomorai/Predict-Ages/actions/workflows/ci.yml/badge.svg)](https://github.com/lsomorai/Predict-Ages/actions/workflows/ci.yml)
[![Docker](https://github.com/lsomorai/Predict-Ages/actions/workflows/docker.yml/badge.svg)](https://github.com/lsomorai/Predict-Ages/actions/workflows/docker.yml)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lsomorai/Predict-Ages/blob/main/Model-Colab.ipynb)

A production-ready deep learning system for facial age classification using **transfer learning** with state-of-the-art CNN architectures. Features include model comparison, Grad-CAM visualizations, experiment tracking with Weights & Biases, and a Gradio web interface.

![Demo](https://via.placeholder.com/800x400?text=Age+Prediction+Demo)

---

## Highlights

- **72.38% accuracy** on 4-class age classification
- **3 model architectures** compared: MobileNetV2, ResNet50, EfficientNet-B0
- **Grad-CAM visualizations** for model interpretability
- **Weights & Biases** integration for experiment tracking
- **Gradio web interface** for easy demos
- **Docker support** for deployment
- **ONNX export** for cross-platform inference
- **CI/CD pipeline** with GitHub Actions
- **Comprehensive test suite** with pytest

---

## Quick Start

### Option 1: Try the Demo

```bash
# Clone the repository
git clone https://github.com/lsomorai/Predict-Ages.git
cd Predict-Ages

# Install dependencies
pip install -r requirements.txt

# Run the Gradio demo
python app.py
```

Open http://localhost:7860 in your browser to try the model!

### Option 2: Google Colab

Click the **Open in Colab** badge above to run training in the cloud with free GPU.

### Option 3: Docker

```bash
# Build and run with Docker
docker build -t age-prediction .
docker run -p 7860:7860 age-prediction
```

---

## Results

### Model Performance

| Model | Test Accuracy | F1 (Macro) | Parameters (Trainable) |
|:------|:-------------:|:----------:|:----------------------:|
| MobileNetV2 | 70.58% | 0.68 | 526K |
| **ResNet50** | **72.38%** | **0.71** | 1.05M |
| EfficientNet-B0 | 72.27% | 0.70 | 527K |

> ResNet50 achieves the best performance with 72.38% test accuracy.

### Age Group Classification

| Class | Age Range | Description |
|:-----:|:---------:|:------------|
| 0 | 0-25 | Young |
| 1 | 26-50 | Adult |
| 2 | 51-75 | Middle-aged |
| 3 | 76-116 | Senior |

---

## Project Structure

```
Predict-Ages/
├── src/
│   └── age_prediction/          # Main package
│       ├── __init__.py
│       ├── config.py            # Configuration & hyperparameters
│       ├── dataset.py           # Dataset & data loading
│       ├── models.py            # Model architectures
│       ├── train.py             # Training with W&B
│       ├── evaluate.py          # Evaluation & metrics
│       ├── gradcam.py           # Grad-CAM visualization
│       └── inference.py         # Prediction API
├── tests/                       # Unit tests
│   ├── test_models.py
│   ├── test_dataset.py
│   └── test_inference.py
├── scripts/                     # CLI scripts
│   ├── train.py                 # Training script
│   ├── evaluate.py              # Evaluation script
│   ├── export_onnx.py           # ONNX export
│   └── download_data.py         # Dataset downloader
├── app.py                       # Gradio web interface
├── Dockerfile                   # Container definition
├── docker-compose.yml           # Docker orchestration
├── .github/workflows/           # CI/CD pipelines
├── Model-Colab.ipynb           # Colab notebook
├── requirements.txt             # Dependencies
├── requirements-dev.txt         # Dev dependencies
├── pyproject.toml              # Project configuration
└── README.md
```

---

## Installation

### Prerequisites

- Python 3.9 or higher
- CUDA (optional, for GPU acceleration)

### Install from Source

```bash
# Clone the repository
git clone https://github.com/lsomorai/Predict-Ages.git
cd Predict-Ages

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install dev dependencies (for testing/development)
pip install -r requirements-dev.txt
```

---

## Usage

### Training

```bash
# Train a single model
python scripts/train.py --model resnet50 --epochs 10

# Train all models
python scripts/train.py --model all --epochs 10

# Train with Weights & Biases logging
python scripts/train.py --model resnet50 --wandb

# Custom training
python scripts/train.py \
    --model resnet50 \
    --epochs 20 \
    --batch-size 64 \
    --lr 0.0005 \
    --data-dir ./data \
    --wandb
```

### Evaluation

```bash
# Evaluate a single model
python scripts/evaluate.py --model resnet50

# Compare all models
python scripts/evaluate.py --model all --compare
```

### ONNX Export

```bash
# Export to ONNX
python scripts/export_onnx.py --model resnet50

# Export with quantization
python scripts/export_onnx.py --model all --quantize --benchmark
```

### Inference API

```python
from src.age_prediction import AgePredictor

# Load predictor
predictor = AgePredictor(
    model_name="resnet50",
    weights_path="./checkpoints/best_resnet50.pth"
)

# Predict from image file
result = predictor.predict("face.jpg")
print(f"Age group: {result['age_range']}")
print(f"Confidence: {result['confidence']:.1%}")

# Predict with Grad-CAM visualization
result, gradcam_image = predictor.predict_with_gradcam("face.jpg")
```

---

## Web Interface

Run the Gradio demo locally:

```bash
python app.py
```

Features:
- Upload face images for age prediction
- Select between different models
- View confidence scores for all age groups
- Grad-CAM visualization to see what the model focuses on

---

## Training from Scratch

### Dataset

This project uses the [UTKFace Dataset](https://www.kaggle.com/datasets/jangedoo/utkface-new) from Kaggle.

**Option 1: Automatic download (recommended)**

```bash
# Set up Kaggle credentials first (see https://www.kaggle.com/settings)
mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json

# Download dataset
python scripts/download_data.py
```

**Option 2: Manual download**

1. Download from [Kaggle](https://www.kaggle.com/datasets/jangedoo/utkface-new)
2. Extract to `./data` directory

**Run training:**

```bash
python scripts/train.py --model all --data-dir ./data
```

### With Weights & Biases

1. Create a [W&B account](https://wandb.ai/)
2. Login: `wandb login`
3. Run training with `--wandb` flag

```bash
python scripts/train.py --model resnet50 --wandb --wandb-project my-age-prediction
```

---

## Docker

### Run the App

```bash
# Build image
docker build -t age-prediction .

# Run container
docker run -p 7860:7860 age-prediction
```

### Docker Compose

```bash
# Run the app
docker-compose up

# Run training
docker-compose --profile training up train

# Run evaluation
docker-compose --profile evaluate up evaluate

# Export to ONNX
docker-compose --profile export up export
```

---

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ -v --cov=src/age_prediction --cov-report=html

# Run specific test file
pytest tests/test_models.py -v
```

---

## Model Architecture

All models use **transfer learning** with ImageNet-pretrained weights. The feature extractor is frozen, and a custom classifier head is trained:

```
┌─────────────────────────────────────┐
│     Pretrained Backbone (Frozen)    │
│  (MobileNetV2 / ResNet50 / EfficientNet) │
└─────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────┐
│        Custom Classifier Head        │
│  ┌─────────────────────────────────┐│
│  │ Linear(features → 512) + ReLU   ││
│  │ Dropout(0.3)                    ││
│  │ Linear(512 → 256) + ReLU        ││
│  │ Dropout(0.3)                    ││
│  │ Linear(256 → 4)                 ││
│  └─────────────────────────────────┘│
└─────────────────────────────────────┘
                  │
                  ▼
         4 Age Class Predictions
```

### Training Configuration

| Parameter | Value |
|:----------|:------|
| Optimizer | Adam |
| Learning Rate | 0.001 |
| LR Scheduler | CosineAnnealing |
| Loss Function | CrossEntropyLoss |
| Batch Size | 32 |
| Epochs | 10 |
| Image Size | 224 × 224 |
| Augmentation | Flip, Rotate, ColorJitter, RandomErasing |

---

## Grad-CAM Visualization

Gradient-weighted Class Activation Mapping (Grad-CAM) highlights the regions of an image that are most important for the model's prediction:

```python
from src.age_prediction import AgePredictor

predictor = AgePredictor("resnet50")
result, overlay = predictor.predict_with_gradcam("face.jpg")

# overlay is a numpy array with the Grad-CAM heatmap overlaid on the image
```

This helps understand what facial features the model uses to determine age:
- Wrinkles and skin texture
- Facial structure
- Hair characteristics
- Eye and mouth regions

---

## Technologies

| Category | Technology |
|:---------|:-----------|
| Deep Learning | PyTorch, torchvision |
| Web Interface | Gradio |
| Experiment Tracking | Weights & Biases |
| Model Export | ONNX, ONNX Runtime |
| Testing | pytest, pytest-cov |
| Linting | Ruff |
| CI/CD | GitHub Actions |
| Containerization | Docker |

---

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Run tests (`pytest tests/`)
4. Commit your changes (`git commit -m 'Add amazing feature'`)
5. Push to the branch (`git push origin feature/amazing-feature`)
6. Open a Pull Request

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- Dataset: [Faces Age Detection Dataset](https://www.kaggle.com/datasets/arashnic/faces-age-detection-dataset) on Kaggle
- Pre-trained models: [torchvision](https://pytorch.org/vision/stable/models.html)
- Grad-CAM: [Selvaraju et al., 2017](https://arxiv.org/abs/1610.02391)

---

## Citation

If you use this project in your research, please cite:

```bibtex
@software{age_prediction,
  author = {Lucien Somorai},
  title = {Facial Age Prediction with Deep Learning},
  year = {2026},
  url = {https://github.com/lsomorai/Predict-Ages}
}
```
