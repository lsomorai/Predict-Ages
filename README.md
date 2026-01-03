# Facial Age Prediction with Deep Learning

[![Python 3.8+](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/Predict-Ages/blob/main/Model-Colab.ipynb)

A deep learning project that classifies human faces into age groups using **transfer learning** with state-of-the-art CNN architectures. Features model comparison across MobileNetV2, ResNet50, and EfficientNet-B0, with **Grad-CAM visualizations** for interpretability.

---

## Highlights

- **72.38% accuracy** on 4-class age classification
- **3 model architectures** compared: MobileNetV2, ResNet50, EfficientNet-B0
- **Grad-CAM visualizations** to understand model predictions
- **Google Colab ready** - run with free GPU, no setup required

---

## Results

### Model Performance

| Model | Test Accuracy | Best Validation Acc |
|:------|:-------------:|:-------------------:|
| MobileNetV2 | 70.58% | 70.89% |
| ResNet50 | **72.38%** | 72.54% |
| EfficientNet-B0 | 72.27% | 72.45% |

> ResNet50 achieved the best performance with 72.38% test accuracy.

### Age Group Classification

| Class | Age Range | Description |
|:-----:|:---------:|:------------|
| 0 | 0-25 | Young |
| 1 | 26-50 | Adult |
| 2 | 51-75 | Middle-aged |
| 3 | 76-116 | Senior |

---

## Quick Start

### Option 1: Google Colab (Recommended)

The easiest way to run this project - no local setup required!

1. Click the **Open in Colab** badge above
2. Set runtime to **GPU** (Runtime → Change runtime type → T4 GPU)
3. Run all cells

### Option 2: Local Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/Predict-Ages.git
cd Predict-Ages

# Install dependencies
pip install -r requirements.txt

# Open Jupyter notebook
jupyter notebook
```

---

## Project Structure

```
Predict-Ages/
├── Model-Colab.ipynb           # Google Colab version (recommended)
├── age_prediction_gradcam.ipynb # Local version with Grad-CAM
├── requirements.txt            # Python dependencies
├── LICENSE                     # MIT License
├── .gitignore
└── README.md
```

---

## Dataset

This project uses the [Faces Age Detection Dataset](https://www.kaggle.com/datasets/arashnic/faces-age-detection-dataset) from Kaggle, which includes:

- **20,000+ face images** from UTKFace
- Images named with embedded metadata: `[age]_[gender]_[race]_[datetime].jpg`
- High variability in pose, expression, illumination, and ethnicity

### Data Split
- **Training**: 70%
- **Validation**: 15%
- **Test**: 15%

---

## Model Architecture

All models use **transfer learning** with ImageNet-pretrained weights. The feature extractor is frozen, and a custom classifier head is trained:

```python
nn.Sequential(
    nn.Linear(in_features, 512),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(512, 256),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(256, 4)  # 4 age classes
)
```

### Training Configuration

| Parameter | Value |
|:----------|:------|
| Optimizer | Adam |
| Learning Rate | 0.001 |
| Loss Function | CrossEntropyLoss |
| Batch Size | 32 |
| Epochs | 10 |
| Image Size | 224 × 224 |
| Normalization | ImageNet (mean, std) |

---

## Grad-CAM Visualization

**Gradient-weighted Class Activation Mapping (Grad-CAM)** highlights the regions of an image that are most important for the model's prediction. This helps us understand what facial features the model uses to determine age:

- Wrinkles and skin texture
- Facial structure and bone definition
- Hair characteristics (color, density)
- Eye and mouth regions

The `age_prediction_gradcam.ipynb` notebook includes Grad-CAM visualizations for all three models.

---

## Key Features

| Feature | Description |
|:--------|:------------|
| **Transfer Learning** | Leverages ImageNet-pretrained weights for robust feature extraction |
| **Multi-Architecture** | Compare MobileNetV2, ResNet50, and EfficientNet-B0 side-by-side |
| **Model Interpretability** | Grad-CAM visualizations explain model decisions |
| **Confusion Matrices** | Visual evaluation of per-class performance |
| **Model Checkpointing** | Automatically saves best model based on validation accuracy |
| **Colab Integration** | Ready-to-run notebook with Kaggle API integration |

---

## Technologies Used

- **PyTorch** - Deep learning framework
- **torchvision** - Pre-trained models and transforms
- **NumPy** - Numerical computing
- **Matplotlib / Seaborn** - Visualization
- **OpenCV** - Image processing for Grad-CAM
- **scikit-learn** - Metrics and evaluation

---

## Future Improvements

- [ ] Add data augmentation (rotation, color jitter)
- [ ] Implement age regression instead of classification
- [ ] Fine-tune more layers of the backbone
- [ ] Add ensemble methods
- [ ] Deploy as web application

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- Dataset: [Faces Age Detection Dataset](https://www.kaggle.com/datasets/arashnic/faces-age-detection-dataset) on Kaggle
- Pre-trained models from [torchvision](https://pytorch.org/vision/stable/models.html)
- Grad-CAM based on [Selvaraju et al., 2017](https://arxiv.org/abs/1610.02391)
