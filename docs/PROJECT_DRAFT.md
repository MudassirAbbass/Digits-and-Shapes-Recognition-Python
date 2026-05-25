# 📄 AI Recognition System (Digits + Shapes) — Detailed Project Draft
**Date:** January 27, 2026  
**Course:** Machine Learning / Artificial Intelligence (Semester 5)  
**Project Type:** Computer Vision Classification + Desktop GUI (Tkinter)  

---

## 1) Abstract

This project delivers an end-to-end **AI recognition system** that classifies:
- **Handwritten digits (0–9)** using a Convolutional Neural Network trained on the **MNIST** dataset.
- **Geometric shapes (Circle, Square, Triangle)** using a CNN trained on a **custom-generated synthetic dataset**.

The system includes:
- A **training pipeline** that generates/loads datasets, trains CNN models, evaluates performance, and saves the models.
- An interactive **Tkinter desktop GUI** that lets a user draw on a canvas and obtain real-time predictions with confidence scores.

The solution demonstrates the complete ML lifecycle: data → preprocessing → model design → training → evaluation → deployment (GUI inference).

---

## 2) Problem Statement

Manual recognition of handwritten digits and simple shapes is:
- Slow when scaled to large volumes
- Error-prone and inconsistent across humans
- Difficult to integrate into automated workflows

This project addresses these issues by providing a deep-learning-based classifier with a usable interface suitable for demonstrations and educational use.

---

## 3) Project Objectives

**Primary objectives**
- Build a CNN-based classifier for **digit recognition (10 classes)**.
- Build a CNN-based classifier for **shape recognition (3 classes)**.
- Provide a GUI that supports drawing input and displays prediction probabilities.

**Secondary objectives**
- Save models to disk and load them for inference.
- Provide evaluation outputs (accuracy, loss curves, optional confusion matrix).
- Keep the system structured and reproducible.

---

## 4) High-Level System Overview

The system is composed of three layers:

### 4.1 Data Layer
- **Digits:** MNIST (built into Keras).
- **Shapes:** synthetic dataset generated programmatically using PIL drawing primitives.

### 4.2 Model/Training Layer
- CNN architectures for both tasks.
- Training includes:
  - train/validation split (where applicable)
  - callbacks (early stopping, LR reduction, checkpointing) in enhanced training
  - saving `.h5` artifacts to `models/`

### 4.3 Application (Inference) Layer
- Tkinter GUI canvas captures user strokes.
- Preprocessing converts canvas drawing → 28×28 grayscale tensor.
- Keras model predicts class probabilities.
- GUI displays top prediction + probability bars.

---

## 5) Repository Structure (What Each Folder Does)

### 5.1 Top-level directories
- `src/`: main application source code (GUI, training, utilities).
- `models/`: saved trained models (`.h5`) and training artifacts (plots/logs).
- `docs/`: documentation (including this draft and the existing report).
- `data/`: optional dataset storage (not required for MNIST; can be used for exports).
- `notebooks/`: optional experiments and analysis.

### 5.2 Key source files
- `src/train_models.py`: training entrypoint for digit + shape models (basic/enhanced/ensemble routines).
- `src/utils.py`: preprocessing, augmentation, synthetic shape generation, plotting utilities.
- `src/gui_app.py`: GUI app (black canvas + white drawing; loads saved models).
- `src/main.py`: alternate GUI version (similar functionality).
- `src/gui_enhanced.py`: enhanced GUI supporting enhanced preprocessing + optional ensemble predictions.

---

## 6) Dependencies and Environment

### 6.1 Core dependencies (`requirements.txt`)
- TensorFlow / Keras
- NumPy
- Matplotlib
- Pillow (PIL)
- scikit-learn

### 6.2 Suggested environment setup (Windows)

Create and activate a virtual environment:

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

---

## 7) Dataset Details

### 7.1 MNIST digit dataset
**Source:** `tensorflow.keras.datasets.mnist`  
**Classes:** 10 (0–9)  
**Input:** 28×28 grayscale images  

Typical MNIST representation is a digit in the center of a dark background. The training pipeline normalizes data for stable training.

### 7.2 Synthetic shapes dataset
**Generator:** `generate_enhanced_shapes()` in `src/utils.py`  
**Classes:** 3
- 0 → Circle
- 1 → Square
- 2 → Triangle

**How images are produced**
- Blank 28×28 grayscale canvas is created.
- One shape is drawn in **white (255)** on **black (0)** background using PIL primitives:
  - circle/ellipse via `draw.ellipse`
  - square/rectangle via `draw.rectangle` or `draw.rounded_rectangle`
  - triangle via `draw.polygon`

**Augmentation at generation time**
The generator includes random variation in:
- size
- position shifts
- triangle rotation
- slight brightness/contrast changes
- optional noise injection

This improves generalization so the model can handle hand-drawn variations.

---

## 8) Data Preprocessing

Preprocessing is critical because the GUI drawing is **280×280** while models expect **28×28**.

### 8.1 GUI input acquisition
- User draws strokes on Tkinter canvas.
- The application also draws into a PIL image (`self.image`) that mirrors the canvas.

### 8.2 Resizing and normalization (basic GUI)
Both `src/gui_app.py` and `src/main.py`:
- convert PIL image → NumPy array
- resize to 28×28 using `Image.Resampling.LANCZOS`
- normalize pixel values to `[0, 1]`
- apply a small threshold to drop faint pixels
- add batch and channel dimensions to get shape `(1, 28, 28, 1)`

### 8.3 Important note: Digit vs Shape color convention
Digits and shapes can have different “expected polarity” (white foreground vs black foreground) depending on training.

In this project:
- **Shapes** were trained on **white shape on black background** (from generator).
- **Digits** are aligned to MNIST expectations (handled by inversion logic in GUI).

This separation prevents systematic misclassification (e.g., predicting one class for everything).

### 8.4 Enhanced preprocessing (enhanced GUI)
`src/gui_enhanced.py` uses `preprocess_image()` from `src/utils.py`, which optionally:
- extracts the non-zero region and centers it
- smooths edges via Gaussian blur
- applies morphological operations to clean noise
- normalizes to `[-0.5, 0.5]`

This is intended to increase robustness on real drawings.

---

## 9) Model Architectures

The project includes “enhanced” CNN architectures with regularization and normalization.

### 9.1 Digit recognition model
**Task:** 10-class classification (0–9)  
**Output:** softmax over 10 classes  

Typical CNN components used:
- multiple Conv2D layers (feature extraction)
- MaxPooling (spatial downsampling)
- BatchNormalization (training stability)
- Dropout (reduce overfitting)
- Dense layers (classification head)

### 9.2 Shape recognition model
**Task:** 3-class classification (Circle, Square, Triangle)  
**Output:** softmax over 3 classes  

This model is similar in structure to the digit CNN but trained on the synthetic dataset.

---

## 10) Training Pipeline

Training is managed primarily in `src/train_models.py`.

### 10.1 Digit model training flow (enhanced)
1. Load MNIST via Keras.
2. Normalize and reshape.
3. Optional augmentation (via `augment_data()`).
4. Build CNN model.
5. Train with callbacks (early stopping, LR scheduling, checkpoints).
6. Evaluate on held-out test set.
7. Save model to `models/`.

### 10.2 Shape model training flow (enhanced)
1. Generate dataset using `generate_enhanced_shapes()`.
2. Convert labels to one-hot (3 classes).
3. Train/test split using scikit-learn.
4. Optional augmentation to increase training data.
5. Build CNN model.
6. Train with callbacks.
7. Evaluate and save artifacts.

### 10.3 Training artifacts
The pipeline can generate:
- `.h5` model files (primary inference artifacts)
- training history plots (accuracy/loss over epochs)
- optional confusion matrix images
- text summaries in `models/`

---

## 11) Model Saving and Loading

### 11.1 Saved model format
Models are saved in Keras H5 format (`.h5`), e.g.:
- `models/digit_model.h5`
- `models/shape_model.h5`

Enhanced runs may also produce:
- `models/digit_model_enhanced.h5`
- `models/shape_model_enhanced.h5`
- `models/digit_model_ensemble_*.h5`
- `models/shape_model_ensemble_*.h5`

### 11.2 Robust path handling (important for Windows)
The GUIs load models using a path derived from `__file__`, so launching from different working directories still succeeds.

---

## 12) GUI Design and User Workflow

### 12.1 GUI features
- Canvas drawing area (280×280)
- Mode selection:
  - Digit recognition
  - Shape recognition
- Buttons:
  - Clear canvas
  - Predict
  - Optional debug (in some GUI variants)
- Output panel:
  - top class prediction
  - probability bars for all classes

### 12.2 User workflow
1. Start GUI.
2. Choose recognition mode (digit or shape).
3. Draw on the canvas.
4. Click Predict.
5. Read the predicted label and confidence scores.

---

## 13) Inference Flow (Step-by-Step)

At inference time:
1. GUI converts the PIL canvas image into a NumPy array.
2. The image is resized to 28×28.
3. Normalization is applied.
4. The model outputs softmax probabilities.
5. The UI sorts probabilities and displays results.

This mirrors the training input format to avoid distribution shift.

---

## 14) Evaluation and Metrics

### 14.1 Accuracy
Accuracy is the main reported metric for both tasks.

### 14.2 Additional metrics (enhanced training)
Some training configs include:
- Precision
- Recall

### 14.3 Confusion matrix (optional)
When enabled, a confusion matrix can be computed and saved for analysis of which classes are confused most often.

---

## 15) Troubleshooting Guide

### 15.1 “Model not found”
- Ensure `.h5` files exist in the project `models/` directory.
- If missing, run training from the project root:

```bash
python src/train_models.py
```

### 15.2 Shape prediction always outputs one class
Common causes:
- Preprocessing mismatch (inversion/polarity different from training)
- Drawing too small / off-center
- Model not trained well or overwritten

Fix:
- Ensure shapes are preprocessed consistently with training (white foreground on black background).

### 15.3 Low confidence predictions
- Draw larger, centered shapes/digits.
- Clear canvas and retry.
- Retrain with more samples/epochs for shapes.

---

## 16) How to Run (Quick Start)

### 16.1 Train models (if needed)

```bash
python src/train_models.py
```

### 16.2 Run GUI

```bash
python src/gui_app.py
```

Alternative GUI variants:
- `python src/main.py`
- `python src/gui_enhanced.py`

---

## 17) Results Summary (Template)

Use this section to insert your final measured results after training on your machine:

- **Digit Model (MNIST)**  
  - Test Accuracy: ___ %
  - Notes: (epochs, augmentation, model variant)

- **Shape Model (Synthetic)**  
  - Test Accuracy: ___ %
  - Notes: (num_samples, augmentation, typical confusions)

---

## 18) Future Improvements

Suggested next steps:
- Expand shapes: pentagon, star, rectangle, etc.
- Add stroke-thickness randomization to shape generator to match hand-drawn lines better.
- Add automatic re-centering/deskewing for shapes as done for digits.
- Export models to TensorFlow SavedModel / ONNX for broader deployment.
- Add a small test suite for preprocessing invariants (polarity, shape, normalization).

---

## 19) Conclusion

This project demonstrates a complete ML workflow and a practical deployment path (desktop GUI) for computer vision classification. By supporting both MNIST digits and synthetic shapes, it highlights how dataset design and preprocessing consistency directly impact real-world inference performance.

