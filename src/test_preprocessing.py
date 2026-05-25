"""
Diagnostic test script for AI Recognition System preprocessing and models.
"""
import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow import keras
from PIL import Image, ImageDraw

# Add src folder to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import preprocess_image

# Resolve project paths
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SRC_DIR)
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

def run_tests():
    print("=" * 70)
    print("🧪 RUNNING PREPROCESSING DIAGNOSTIC TESTS")
    print("=" * 70)
    
    # 1. Load models
    digit_model_path = os.path.join(MODELS_DIR, "digit_model.h5")
    shape_model_path = os.path.join(MODELS_DIR, "shape_model.h5")
    
    if not os.path.exists(digit_model_path):
        print(f"❌ Digit model not found at {digit_model_path}")
        return
    if not os.path.exists(shape_model_path):
        print(f"❌ Shape model not found at {shape_model_path}")
        return
        
    print("Loading models...")
    digit_model = keras.models.load_model(digit_model_path)
    shape_model = keras.models.load_model(shape_model_path)
    print("✅ Models loaded successfully!\n")
    
    # 2. Test digit preprocessing range
    print("--- Test 1: Digit Preprocessing Range ---")
    # Create dummy black-on-white image (like drawing on white canvas)
    img_white_bg = Image.new('L', (280, 280), 255)
    draw_white_bg = ImageDraw.Draw(img_white_bg)
    # Draw a line (digit simulation) in black (0)
    draw_white_bg.line([140, 50, 140, 230], fill=0, width=15)
    
    processed_digit = preprocess_image(img_white_bg, target_size=(28, 28), enhance=True, mode='digit')
    print(f"Processed shape: {processed_digit.shape}")
    print(f"Value range: [{processed_digit.min():.4f}, {processed_digit.max():.4f}]")
    print(f"Mean value: {processed_digit.mean():.4f}")
    
    # Verify range is in [-0.5, 0.5]
    if -0.51 <= processed_digit.min() <= -0.45 and 0.45 <= processed_digit.max() <= 0.51:
        print("✅ Digit preprocessing range is correctly [-0.5, 0.5]!")
    else:
        print(f"❌ Digit preprocessing range error! Expected [-0.5, 0.5], got [{processed_digit.min():.4f}, {processed_digit.max():.4f}]")
        
    # 3. Test shape preprocessing range
    print("\n--- Test 2: Shape Preprocessing Range ---")
    # Create dummy white-on-black image (like drawing on black canvas)
    img_black_bg = Image.new('L', (280, 280), 0)
    draw_black_bg = ImageDraw.Draw(img_black_bg)
    # Draw a circle in white (255)
    draw_black_bg.ellipse([50, 50, 230, 230], fill=255)
    
    processed_shape = preprocess_image(img_black_bg, target_size=(28, 28), enhance=True, mode='shape')
    print(f"Processed shape: {processed_shape.shape}")
    print(f"Value range: [{processed_shape.min():.4f}, {processed_shape.max():.4f}]")
    print(f"Mean value: {processed_shape.mean():.4f}")
    
    # Verify range is in [0.0, 1.0]
    if 0.0 <= processed_shape.min() <= 0.05 and 0.95 <= processed_shape.max() <= 1.05:
        print("✅ Shape preprocessing range is correctly [0.0, 1.0]!")
    else:
        print(f"❌ Shape preprocessing range error! Expected [0.0, 1.0], got [{processed_shape.min():.4f}, {processed_shape.max():.4f}]")

    # 4. Test predictions on synthetic hand-drawn inputs
    print("\n--- Test 3: Predictions on Mock Drawings ---")
    
    # Mock Digit 1 (vertical line)
    img_digit1 = Image.new('L', (280, 280), 0)
    draw_d1 = ImageDraw.Draw(img_digit1)
    draw_d1.line([140, 40, 140, 240], fill=255, width=20) # White vertical line
    
    # Mock Digit 0 (circle)
    img_digit0 = Image.new('L', (280, 280), 0)
    draw_d0 = ImageDraw.Draw(img_digit0)
    draw_d0.ellipse([80, 60, 200, 220], fill=0, outline=255, width=20) # White circle outline
    
    prep_d1 = preprocess_image(img_digit1, target_size=(28, 28), enhance=True, mode='digit')
    prep_d0 = preprocess_image(img_digit0, target_size=(28, 28), enhance=True, mode='digit')
    
    pred_d1 = digit_model.predict(prep_d1, verbose=0)[0]
    pred_d0 = digit_model.predict(prep_d0, verbose=0)[0]
    
    class_d1 = np.argmax(pred_d1)
    class_d0 = np.argmax(pred_d0)
    
    print(f"Mock '1' Prediction: Predicted digit {class_d1} with {pred_d1[class_d1]*100:.2f}% confidence")
    print(f"Mock '0' Prediction: Predicted digit {class_d0} with {pred_d0[class_d0]*100:.2f}% confidence")
    
    if class_d1 != class_d0:
        print("✅ Success! Predictions are different for different inputs!")
    else:
        print("❌ Warning: Model predicted the same digit for both distinct inputs.")
        
    # 5. Test Shape model predictions
    print("\n--- Test 4: Predictions on Mock Shapes ---")
    
    # Circle
    img_circle = Image.new('L', (280, 280), 0)
    draw_c = ImageDraw.Draw(img_circle)
    draw_c.ellipse([60, 60, 220, 220], fill=255)
    
    # Square
    img_square = Image.new('L', (280, 280), 0)
    draw_s = ImageDraw.Draw(img_square)
    draw_s.rectangle([60, 60, 220, 220], fill=255)
    
    prep_c = preprocess_image(img_circle, target_size=(28, 28), enhance=True, mode='shape')
    prep_s = preprocess_image(img_square, target_size=(28, 28), enhance=True, mode='shape')
    
    shape_labels = ['Circle', 'Square', 'Triangle']
    
    pred_c = shape_model.predict(prep_c, verbose=0)[0]
    pred_s = shape_model.predict(prep_s, verbose=0)[0]
    
    class_c = shape_labels[np.argmax(pred_c)]
    class_s = shape_labels[np.argmax(pred_s)]
    
    print(f"Mock Circle Prediction: Predicted {class_c} with {pred_c[np.argmax(pred_c)]*100:.2f}% confidence")
    print(f"Mock Square Prediction: Predicted {class_s} with {pred_s[np.argmax(pred_s)]*100:.2f}% confidence")
    
    if class_c != class_s:
        print("✅ Success! Shape predictions are different for circle and square!")
    else:
        print("❌ Warning: Shape model predicted the same shape for both circle and square.")

if __name__ == "__main__":
    run_tests()
