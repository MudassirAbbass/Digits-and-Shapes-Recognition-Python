# 🤖 AI Recognition System
An advanced, AI-powered system designed for real-time handwritten digit classification and geometric shape recognition. Built with **TensorFlow / Keras** and **Tkinter**, it features a modular architecture supporting single-model inference, advanced convolutional neural networks (CNNs), and ensemble prediction schemes for maximum reliability.
---
## 📋 Table of Contents
- [Key Features](#-key-features)
- [Bugs Fixed & Improvements](#-bugs-fixed--improvements)
- [Technology Stack](#-technology-stack)
- [Project Architecture](#-project-architecture)
- [Installation & Setup](#-installation--setup)
- [Usage Guide](#-usage-guide)
  - [Interactive GUI Options](#1-interactive-gui-options)
  - [Model Training](#2-model-training)
  - [Programmatic Prediction](#3-programmatic-prediction)
- [Model Architectures](#-model-architectures)
  - [Basic CNN Architecture](#basic-cnn-architecture)
  - [Advanced CNN Architecture](#advanced-cnn-architecture)
- [Performance & Comparison](#-performance--comparison)
- [Future Roadmap](#-future-roadmap)
---
## ✨ Key Features
- 🎨 **Dynamic Drawing Canvas**: 280×280 pixel drawing canvases with support for both black-on-white and white-on-black painting modes.
- 🧠 **Dual Mode Recognition**: Classify handwritten digits (0-9) trained on the classic MNIST dataset and geometric shapes (Circles, Squares, Triangles) trained on procedurally generated shape datasets.
- 🎯 **Ensemble Predictions**: Combines predictions from multiple trained CNNs using soft voting (probability averaging) to achieve up to **99.5% accuracy**.
- 🛠️ **Real-Time Preprocessing Visualization**: Interactive debugging step-by-step pipeline views displaying extraction, centering, resizing, and normalization stages.
- ⚙️ **Automatic Inversion Logic**: Smart brightness calculation that auto-detects drawing format and translates canvas representations to standard dark backgrounds.
---
## 🐛 Bugs Fixed & Improvements
The preprocessing and GUI pipelines have been fully updated to resolve critical bugs:
1. **Normalization Range Mismatch**: Fixed a bug where the GUI fed `[0.0, 1.0]` normalized inputs to the digit model which expected `[-0.5, 0.5]`. Normalization is now dynamically applied based on the mode:
   - **Digit Mode**: Normalized to `[-0.5, 0.5]` range.
   - **Shape Mode**: Normalized to `[0.0, 1.0]` range.
2. **Double Color Inversion**: Resolved color inversion bugs that occurred on black-background canvases. Drawing inputs are now automatically normalized to white-on-black (MNIST format) regardless of canvas backgrounds.
3. **Noisy/Destructive Preprocessing**: Removed destructive binarization thresholds and morphology filters that deleted gradient anti-aliasing. The model now processes smooth, anti-aliased hand-drawn pixels.
4. **Deterministic Inference**: Removed random noise injection steps during prediction, ensuring identical drawings yield consistent predictions.
---
## 🛠️ Technology Stack
- **Core Language**: Python 3.8+
- **Deep Learning Framework**: TensorFlow 2.x & Keras
- **Image Processing**: Pillow (PIL), NumPy, SciPy (scipy.ndimage)
- **GUI Engine**: Tkinter (Standard Library)
- **Plotting & Visuals**: Matplotlib, Seaborn
---
## 📁 Project Architecture
```
AI-Recognition-Python/
│
├── src/                          # Source code
│   ├── main.py                   # Main entry point (White canvas, basic GUI)
│   ├── gui_app.py                # Basic GUI (Black canvas, single-model predictions)
│   ├── gui_enhanced.py           # Enhanced GUI (Ensemble prediction, visualization panels)
│   ├── train_models.py           # Multi-tiered model training pipeline
│   ├── utils.py                  # Math helper functions, augmenters, and preprocessing
│   └── test_preprocessing.py     # Diagnostic & validation tests for range checks
│
├── models/                       # Saved models & diagnostics
│   ├── digit_model.h5            # Basic digit model
│   ├── shape_model.h5            # Basic shape model
│   ├── digit_model_ensemble_*.h5 # Ensemble components for digits
│   ├── shape_model_ensemble_*.h5 # Ensemble components for shapes
│   └── *_training_history.png    # Matplotlib accuracy/loss graphs
│
└── requirements.txt              # Project dependencies
```
---
## 📥Installation & Setup
### Prerequisites
- Python 3.8 or above
- Pip (Python Package Installer)
### Setup Steps
1. **Clone the repository**:
   ```bash
   git clone https://github.com/MudassirAbbas/AI-Recognition-Python.git
   cd AI-Recognition-Python
   ```
2. **Create a virtual environment**:
   ```bash
   python -m venv .venv
   # Windows:
   .venv\Scripts\activate
   # Mac/Linux:
   source .venv/bin/activate
   ```
3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
---
##  Usage Guide
### 1. Interactive GUI Options
There are three different Tkinter GUI applications available depending on your requirements:
- **Option A: Basic White Canvas GUI**
  ```bash
  python src/main.py
  ```
  Traditional drawing app utilizing basic single-model predictions with a white drawing canvas.
- **Option B: Basic Black Canvas GUI**
  ```bash
  python src/gui_app.py
  ```
  Dark-themed canvas layout utilizing single-model predictions and manual preprocessing debugging.
- **Option C: Enhanced Ensemble GUI (Recommended)**
  ```bash
  python src/gui_enhanced.py
  ```
  Premium interface displaying confidence gauges, detailed parameters, custom visual logs, and optional **multi-model ensemble voting**.
---
### 2. Model Training
Launch the interactive training CLI to train custom models:
```bash
python src/train_models.py
```
You will be prompted to choose:
- **[1] Ensemble Mode (60-90 min)**: Trains 3 parallel advanced CNNs for digits and shapes.
- **[2] Single Enhanced Mode (20-30 min)**: Trains a single advanced CNN with regularization.
- **[3] Basic Mode (10-15 min)**: Quickly builds a basic convolutional neural network.
---
### 3. Programmatic Prediction
Incorporate the trained models directly into external Python pipelines:
```python
import numpy as np
from PIL import Image
from tensorflow import keras
from utils import preprocess_image
# Load model
model = keras.models.load_model('models/digit_model.h5')
# Preprocess image for digit recognition (returns shape: 1, 28, 28, 1)
img = Image.open('my_digit.png')
img_array = preprocess_image(img, target_size=(28, 28), enhance=True, mode='digit')
# Run prediction
predictions = model.predict(img_array)
predicted_class = np.argmax(predictions[0])
confidence = predictions[0][predicted_class]
print(f"Predicted class: {predicted_class} ({confidence * 100:.2f}%)")
```
---
## 🏗️ Model Architectures
### Basic CNN Architecture
A simple, fast-converging network:
1. **Conv2D** (32 filters, 3x3 kernel, ReLU)
2. **MaxPooling2D** (2x2 pool)
3. **Conv2D** (64 filters, 3x3 kernel, ReLU)
4. **MaxPooling2D** (2x2 pool)
5. **Flatten**
6. **Dense** (128 neurons, ReLU)
7. **Dropout** (25%)
8. **Dense / Output** (Softmax activation)
### Advanced CNN Architecture
A deeper, highly regularized network containing:
- **Convolutional Blocks** with double 3x3 Conv2D layers.
- **Batch Normalization** on every block to stabilize activations.
- **L2 Regularization** weights penalty (`0.001`) to reduce overfitting.
- **Global Average Pooling 2D** to minimize parameter counts and retain spatial invariants.
- **Deep Dense Layer Classifier** with `50%` dropout rates.
---
## 📊 Performance & Comparison
|
 Mode / Model Type 
|
 Parameter Count 
|
 Training Dataset 
|
 Expected Accuracy 
|
|
:---
|
:---
|
:---
|
:---
|
|
**
Basic CNN (Digit)
**
|
 ~225,000 
|
 MNIST Standard 
|
**
98.5%
**
|
|
**
Advanced CNN (Digit)
**
|
 ~250,000 
|
 Augmented MNIST 
|
**
99.1%
**
|
|
**
Ensemble Model (Digit)
**
|
 ~750,000 (3x) 
|
 Multi-Augment MNIST 
|
**
99.6%
**
|
|
**
Basic CNN (Shape)
**
|
 ~220,000 
|
 Procedural Shapes 
|
**
96.2%
**
|
|
**
Advanced CNN (Shape)
**
|
 ~240,000 
|
 Procedural Shapes 
|
**
98.0%
**
|
---
## 🔮 Future Roadmap
- [ ] **Alphanumeric Expansion**: Support uppercase/lowercase English alphabet recognition (A-Z, a-z).
- [ ] **Web Deployment**: Port models to TensorFlow.js for in-browser canvas drawings.
- [ ] **On-Device Quantization**: Convert `.h5` weights to lightweight `.tflite` formats for mobile applications.
- [ ] **Complex Shapes**: Train dataset to recognize polygons, ellipses, and hand-drawn stars.
---
**Developed for Machine Learning Project | Made with ❤️ using Python & TensorFlow**
