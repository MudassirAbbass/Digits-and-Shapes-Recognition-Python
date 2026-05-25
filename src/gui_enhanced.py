"""
Enhanced GUI with Ensemble Predictions for Maximum Accuracy
"""

import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageDraw
import numpy as np
import tensorflow as tf
from tensorflow import keras
import sys
import os
import pickle
from pathlib import Path

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import preprocess_image, create_ensemble_predictions

# Resolve project paths robustly (works no matter the current working directory)
SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent
MODELS_DIR = PROJECT_ROOT / "models"


class EnhancedDrawingApp:
    """Enhanced GUI Application with Ensemble Predictions"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("🤖 AI Recognition System - Enhanced")
        self.root.geometry("900x700")
        self.root.resizable(False, False)
        
        # Model variables
        self.digit_models = []
        self.shape_models = []
        self.current_mode = 'digit'
        self.use_ensemble = True  # Use ensemble by default
        
        # Drawing variables
        self.canvas_size = 280
        self.pen_size = 15
        self.image = Image.new('L', (self.canvas_size, self.canvas_size), 0)
        self.draw = ImageDraw.Draw(self.image)
        
        # Load models
        self.load_enhanced_models()
        
        # Setup GUI
        self.setup_enhanced_gui()
    
    def load_enhanced_models(self):
        """Load enhanced or ensemble models"""
        try:
            # Try to load ensemble models
            ensemble_digit_path = MODELS_DIR / "ensemble_digit_models.pkl"
            ensemble_shape_path = MODELS_DIR / "ensemble_shape_models.pkl"
            
            if ensemble_digit_path.exists() and ensemble_shape_path.exists():
                print("🎯 Loading ensemble models...")
                self.load_ensemble_models()
            else:
                # Load single enhanced models
                print("🎯 Loading enhanced models...")
                self.load_single_enhanced_models()
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load models: {str(e)}")
    
    def load_ensemble_models(self):
        """Load ensemble of models"""
        try:
            # Load digit ensemble models
            for i in range(1, 4):  # Assuming 3 models
                model_path = MODELS_DIR / f"digit_model_ensemble_{i}.h5"
                if model_path.exists():
                    model = keras.models.load_model(str(model_path))
                    self.digit_models.append(model)
                    print(f"✅ Loaded digit model {i}")
            
            # Load shape ensemble models
            for i in range(1, 4):
                model_path = MODELS_DIR / f"shape_model_ensemble_{i}.h5"
                if model_path.exists():
                    model = keras.models.load_model(str(model_path))
                    self.shape_models.append(model)
                    print(f"✅ Loaded shape model {i}")
            
            if len(self.digit_models) > 0 and len(self.shape_models) > 0:
                print(f"✅ Loaded {len(self.digit_models)} digit models and {len(self.shape_models)} shape models")
            else:
                print("⚠️  Could not load ensemble models, trying single models...")
                self.load_single_enhanced_models()
                
        except Exception as e:
            print(f"⚠️  Error loading ensemble: {str(e)}")
            self.load_single_enhanced_models()
    
    def load_single_enhanced_models(self):
        """Load single enhanced models"""
        try:
            # Try enhanced models first
            digit_enhanced = MODELS_DIR / "digit_model_enhanced.h5"
            digit_basic = MODELS_DIR / "digit_model.h5"
            shape_enhanced = MODELS_DIR / "shape_model_enhanced.h5"
            shape_basic = MODELS_DIR / "shape_model.h5"

            if digit_enhanced.exists():
                model = keras.models.load_model(str(digit_enhanced))
                self.digit_models.append(model)
                print("✅ Loaded enhanced digit model")
            elif digit_basic.exists():
                model = keras.models.load_model(str(digit_basic))
                self.digit_models.append(model)
                print("✅ Loaded basic digit model")
            else:
                print("⚠️  Digit model not found. Train it first!")
            
            if shape_enhanced.exists():
                model = keras.models.load_model(str(shape_enhanced))
                self.shape_models.append(model)
                print("✅ Loaded enhanced shape model")
            elif shape_basic.exists():
                model = keras.models.load_model(str(shape_basic))
                self.shape_models.append(model)
                print("✅ Loaded basic shape model")
            else:
                print("⚠️  Shape model not found. Train it first!")
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load single models: {str(e)}")
    
    def setup_enhanced_gui(self):
        """Setup the enhanced GUI interface"""
        
        # Main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # ==================== HEADER ====================
        header_frame = ttk.Frame(main_frame)
        header_frame.grid(row=0, column=0, columnspan=3, pady=(0, 10))
        
        title_label = ttk.Label(
            header_frame, 
            text="🤖 AI Recognition System - Enhanced",
            font=('Arial', 22, 'bold')
        )
        title_label.pack()
        
        subtitle_label = ttk.Label(
            header_frame,
            text="Draw digits or shapes and see AI predictions with 99%+ accuracy",
            font=('Arial', 11)
        )
        subtitle_label.pack()
        
        # ==================== CONTROL PANEL ====================
        control_frame = ttk.LabelFrame(main_frame, text="Controls", padding="10")
        control_frame.grid(row=1, column=0, columnspan=3, pady=(0, 10), sticky=(tk.W, tk.E))
        
        # Mode selector
        mode_subframe = ttk.Frame(control_frame)
        mode_subframe.pack(side=tk.LEFT, padx=10)
        
        ttk.Label(mode_subframe, text="Recognition Mode:", font=('Arial', 10, 'bold')).pack(anchor=tk.W)
        
        self.mode_var = tk.StringVar(value='digit')
        
        ttk.Radiobutton(
            mode_subframe,
            text="🔢 Digit Recognition (0-9)",
            variable=self.mode_var,
            value='digit',
            command=self.switch_mode
        ).pack(anchor=tk.W)
        
        ttk.Radiobutton(
            mode_subframe,
            text="⚪ Shape Recognition (Circle, Square, Triangle)",
            variable=self.mode_var,
            value='shape',
            command=self.switch_mode
        ).pack(anchor=tk.W)
        
        # Ensemble toggle
        ensemble_subframe = ttk.Frame(control_frame)
        ensemble_subframe.pack(side=tk.LEFT, padx=30)
        
        ttk.Label(ensemble_subframe, text="Prediction Mode:", font=('Arial', 10, 'bold')).pack(anchor=tk.W)
        
        self.ensemble_var = tk.BooleanVar(value=True)
        
        ttk.Checkbutton(
            ensemble_subframe,
            text="🎯 Use Ensemble (Highest Accuracy)",
            variable=self.ensemble_var,
            command=self.toggle_ensemble
        ).pack(anchor=tk.W)
        
        ttk.Checkbutton(
            ensemble_subframe,
            text="✨ Enhanced Preprocessing",
            variable=tk.BooleanVar(value=True),
            state='disabled'
        ).pack(anchor=tk.W, pady=(5, 0))
        
        # Model info
        info_subframe = ttk.Frame(control_frame)
        info_subframe.pack(side=tk.RIGHT, padx=10)
        
        model_count = f"{len(self.digit_models)} digit, {len(self.shape_models)} shape models"
        status_text = f"🟢 {model_count} loaded" if self.digit_models and self.shape_models else "🔴 Models not loaded"
        
        self.status_label = ttk.Label(
            info_subframe,
            text=status_text,
            font=('Arial', 10, 'bold')
        )
        self.status_label.pack(anchor=tk.E)
        
        # ==================== LEFT PANEL: CANVAS ====================
        left_frame = ttk.LabelFrame(main_frame, text="Drawing Canvas", padding="10")
        left_frame.grid(row=2, column=0, padx=(0, 10), sticky=(tk.N, tk.S))
        
        # Canvas with black background
        self.canvas = tk.Canvas(
            left_frame,
            width=self.canvas_size,
            height=self.canvas_size,
            bg='black',
            cursor='crosshair'
        )
        self.canvas.pack()
        
        # Canvas bindings
        self.canvas.bind('<B1-Motion>', self.paint)
        self.canvas.bind('<Button-1>', self.start_paint)
        
        # Instructions
        instructions = ttk.Label(
            left_frame,
            text="📝 Click and drag to draw (white on black)\n"
                 "✨ Enhanced preprocessing automatically applied\n"
                 "🎯 Draw in the center for best results\n"
                 "⚡ Ensemble predictions for maximum accuracy",
            justify=tk.LEFT,
            font=('Arial', 9)
        )
        instructions.pack(pady=(10, 0))
        
        # Control buttons
        button_frame = ttk.Frame(left_frame)
        button_frame.pack(pady=(10, 0))
        
        ttk.Button(
            button_frame,
            text="🗑️ Clear Canvas",
            command=self.clear_canvas,
            width=15
        ).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(
            button_frame,
            text="🔮 Predict (Single)",
            command=lambda: self.predict(use_ensemble=False),
            width=15
        ).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(
            button_frame,
            text="🎯 Predict (Ensemble)",
            command=lambda: self.predict(use_ensemble=True),
            width=15
        ).pack(side=tk.LEFT, padx=5)
        
        # ==================== MIDDLE PANEL: PREDICTIONS ====================
        middle_frame = ttk.LabelFrame(main_frame, text="AI Predictions", padding="10")
        middle_frame.grid(row=2, column=1, padx=10, sticky=(tk.N, tk.S))
        
        # Results display
        self.results_text = tk.Text(
            middle_frame,
            width=45,
            height=25,
            font=('Courier', 10),
            state=tk.DISABLED,
            bg='#f0f0f0'
        )
        self.results_text.pack(fill=tk.BOTH, expand=True)
        
        # Confidence gauge
        gauge_frame = ttk.Frame(middle_frame)
        gauge_frame.pack(fill=tk.X, pady=(10, 0))
        
        self.confidence_label = ttk.Label(
            gauge_frame,
            text="Confidence: --%",
            font=('Arial', 10, 'bold')
        )
        self.confidence_label.pack()
        
        self.confidence_gauge = ttk.Progressbar(
            gauge_frame,
            length=300,
            mode='determinate'
        )
        self.confidence_gauge.pack(pady=(5, 0))
        
        # ==================== RIGHT PANEL: DETAILS ====================
        right_frame = ttk.LabelFrame(main_frame, text="Model Details", padding="10")
        right_frame.grid(row=2, column=2, sticky=(tk.N, tk.S))
        
        # Model info display
        self.details_text = tk.Text(
            right_frame,
            width=35,
            height=25,
            font=('Courier', 9),
            state=tk.DISABLED,
            bg='#f8f8f8'
        )
        self.details_text.pack(fill=tk.BOTH, expand=True)
        
        # Update details
        self.update_model_details()
        
        # ==================== FOOTER ====================
        footer_frame = ttk.Frame(main_frame)
        footer_frame.grid(row=3, column=0, columnspan=3, pady=(10, 0))
        
        ttk.Button(
            footer_frame,
            text="📊 View Training History",
            command=self.view_training_history
        ).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(
            footer_frame,
            text="🔍 Debug Preprocessing",
            command=self.debug_preprocessing
        ).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(
            footer_frame,
            text="ℹ️ About",
            command=self.show_enhanced_about
        ).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(
            footer_frame,
            text="❌ Exit",
            command=self.root.quit
        ).pack(side=tk.LEFT, padx=5)
    
    def start_paint(self, event):
        """Initialize painting"""
        self.last_x = event.x
        self.last_y = event.y
    
    def paint(self, event):
        """Draw on canvas"""
        # Draw white on black canvas
        x1, y1 = event.x - self.pen_size//2, event.y - self.pen_size//2
        x2, y2 = event.x + self.pen_size//2, event.y + self.pen_size//2
        
        self.canvas.create_oval(x1, y1, x2, y2, fill='white', outline='white')
        self.draw.ellipse([x1, y1, x2, y2], fill=255, outline=255)
        
        self.last_x = event.x
        self.last_y = event.y
    
    def clear_canvas(self):
        """Clear the canvas"""
        self.canvas.delete('all')
        self.canvas.configure(bg='black')
        self.image = Image.new('L', (self.canvas_size, self.canvas_size), 0)
        self.draw = ImageDraw.Draw(self.image)
        
        self.results_text.config(state=tk.NORMAL)
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(tk.END, "Canvas cleared.\n\nDraw something and click Predict!")
        self.results_text.config(state=tk.DISABLED)
        
        self.confidence_label.config(text="Confidence: --%")
        self.confidence_gauge['value'] = 0
    
    def switch_mode(self):
        """Switch between digit and shape recognition"""
        self.current_mode = self.mode_var.get()
        self.clear_canvas()
        
        mode_name = "Digit" if self.current_mode == 'digit' else "Shape"
        messagebox.showinfo("Mode Changed", f"Switched to {mode_name} Recognition mode")
    
    def toggle_ensemble(self):
        """Toggle ensemble mode"""
        self.use_ensemble = self.ensemble_var.get()
        status = "ENABLED" if self.use_ensemble else "DISABLED"
        print(f"Ensemble mode {status}")
    
    def predict(self, use_ensemble=None):
        """Make prediction with enhanced preprocessing"""
        
        if use_ensemble is None:
            use_ensemble = self.use_ensemble
        
        # Check if canvas is blank
        if not self.is_canvas_drawn():
            messagebox.showwarning("Empty Canvas", "Please draw something first!")
            return
        
        # Check if models are loaded
        if self.current_mode == 'digit' and not self.digit_models:
            messagebox.showerror("Model Error", "Digit models not loaded. Train them first!")
            return
        
        if self.current_mode == 'shape' and not self.shape_models:
            messagebox.showerror("Model Error", "Shape models not loaded. Train them first!")
            return
        
        try:
            # Use enhanced preprocessing
            img_array = preprocess_image(self.image, target_size=(28, 28), enhance=True, mode=self.current_mode)
            
            # Get predictions
            if self.current_mode == 'digit':
                models = self.digit_models
                labels = [str(i) for i in range(10)]
            else:
                models = self.shape_models
                labels = ['Circle', 'Square', 'Triangle']
            
            if use_ensemble and len(models) > 1:
                # Ensemble prediction
                all_predictions = []
                for i, model in enumerate(models):
                    pred = model.predict(img_array, verbose=0)[0]
                    all_predictions.append(pred)
                
                # Average predictions (soft voting)
                predictions = np.mean(all_predictions, axis=0)
                prediction_type = "Ensemble"
                model_count = len(models)
            else:
                # Single model prediction
                predictions = models[0].predict(img_array, verbose=0)[0]
                prediction_type = "Single Model"
                model_count = 1
            
            # Display results
            self.display_enhanced_predictions(predictions, labels, prediction_type, model_count)
            
        except Exception as e:
            messagebox.showerror("Prediction Error", f"Error during prediction: {str(e)}")
    
    def display_enhanced_predictions(self, predictions, labels, prediction_type, model_count):
        """Display enhanced prediction results"""
        
        # Sort predictions
        sorted_indices = np.argsort(predictions)[::-1]
        top_label = labels[sorted_indices[0]]
        top_conf = predictions[sorted_indices[0]] * 100
        
        # Update confidence gauge
        self.confidence_label.config(text=f"Confidence: {top_conf:.1f}%")
        self.confidence_gauge['value'] = top_conf
        
        # Format output
        output = "=" * 50 + "\n"
        output += f"🔮 {prediction_type.upper()} PREDICTION\n"
        output += "=" * 50 + "\n\n"
        
        output += f"📊 Mode: {'Digit' if self.current_mode == 'digit' else 'Shape'}\n"
        output += f"🎯 Method: {prediction_type} ({model_count} model{'s' if model_count > 1 else ''})\n"
        output += f"⏱️  Preprocessing: Enhanced\n\n"
        
        output += "🏆 TOP PREDICTION:\n"
        output += f"   {top_label}: {top_conf:.1f}%\n\n"
        
        # Confidence interpretation
        if top_conf > 95:
            output += "✅ Excellent confidence - Very reliable\n\n"
        elif top_conf > 85:
            output += "✅ High confidence - Reliable\n\n"
        elif top_conf > 70:
            output += "⚠️  Good confidence - Probably correct\n\n"
        elif top_conf > 50:
            output += "⚠️  Moderate confidence - Check carefully\n\n"
        else:
            output += "❓ Low confidence - Try drawing clearer\n\n"
        
        output += "📊 ALL PREDICTIONS:\n"
        output += "-" * 50 + "\n"
        
        for idx in sorted_indices:
            label = labels[idx]
            confidence = predictions[idx] * 100
            bar_length = int(confidence / 2)  # More detailed bar
            bar = "█" * bar_length + "░" * (50 - bar_length)
            
            output += f"{label:10s} {confidence:5.1f}% │{bar}│\n"
        
        output += "=" * 50 + "\n"
        
        # Update text widget
        self.results_text.config(state=tk.NORMAL)
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(tk.END, output)
        self.results_text.config(state=tk.DISABLED)
    
    def update_model_details(self):
        """Update model details panel"""
        details = "=" * 40 + "\n"
        details += "🤖 MODEL DETAILS\n"
        details += "=" * 40 + "\n\n"
        
        details += "DIGIT RECOGNITION:\n"
        details += f"  Models loaded: {len(self.digit_models)}\n"
        if self.digit_models:
            details += f"  Input shape: {self.digit_models[0].input_shape}\n"
            details += f"  Output classes: 10 (0-9)\n"
            details += f"  Total params: {self.digit_models[0].count_params():,}\n"
        
        details += "\nSHAPE RECOGNITION:\n"
        details += f"  Models loaded: {len(self.shape_models)}\n"
        if self.shape_models:
            details += f"  Input shape: {self.shape_models[0].input_shape}\n"
            details += f"  Output classes: 3\n"
            details += f"  Total params: {self.shape_models[0].count_params():,}\n"
        
        details += "\n⚙️  FEATURES:\n"
        details += "  ✅ Enhanced preprocessing\n"
        details += "  ✅ Ensemble predictions\n"
        details += "  ✅ Confidence scoring\n"
        details += "  ✅ Real-time updates\n"
        
        details += "\n🎯 EXPECTED ACCURACY:\n"
        details += "  Basic: 95-97%\n"
        details += "  Enhanced: 98-99%\n"
        details += "  Ensemble: 99%+\n"
        
        details += "=" * 40 + "\n"
        
        self.details_text.config(state=tk.NORMAL)
        self.details_text.delete(1.0, tk.END)
        self.details_text.insert(tk.END, details)
        self.details_text.config(state=tk.DISABLED)
    
    def is_canvas_drawn(self):
        """Check if anything is drawn on canvas"""
        img_array = np.array(self.image)
        return np.sum(img_array) > 0
    
    def view_training_history(self):
        """Show training history images"""
        try:
            import matplotlib.pyplot as plt
            import matplotlib.image as mpimg
            
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            
            # Try to load enhanced training history
            if os.path.exists('models/enhanced_digit_training_history.png'):
                img1 = mpimg.imread('models/enhanced_digit_training_history.png')
                axes[0].imshow(img1)
                axes[0].set_title('Digit Training History', fontsize=12, fontweight='bold')
                axes[0].axis('off')
            else:
                axes[0].text(0.5, 0.5, 'Digit training history\nnot found', 
                           ha='center', va='center', fontsize=12)
                axes[0].axis('off')
            
            if os.path.exists('models/enhanced_shape_training_history.png'):
                img2 = mpimg.imread('models/enhanced_shape_training_history.png')
                axes[1].imshow(img2)
                axes[1].set_title('Shape Training History', fontsize=12, fontweight='bold')
                axes[1].axis('off')
            else:
                axes[1].text(0.5, 0.5, 'Shape training history\nnot found', 
                           ha='center', va='center', fontsize=12)
                axes[1].axis('off')
            
            plt.suptitle('Model Training History', fontsize=16, fontweight='bold')
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            messagebox.showerror("Error", f"Cannot display training history: {str(e)}")
    
    def debug_preprocessing(self):
        """Debug preprocessing steps"""
        if not self.is_canvas_drawn():
            messagebox.showwarning("Empty Canvas", "Please draw something first!")
            return
        
        try:
            # Save original
            self.image.save('debug_original.png')
            
            # Show preprocessing steps
            import matplotlib.pyplot as plt
            
            fig, axes = plt.subplots(2, 3, figsize=(12, 8))
            
            # Original image
            axes[0, 0].imshow(np.array(self.image), cmap='gray')
            axes[0, 0].set_title('1. Original Drawing')
            axes[0, 0].axis('off')
            
            # Auto-inversion matching preprocess_image
            img_array = np.array(self.image).astype('float32')
            if np.mean(img_array) > 127:
                processed_inversion = 255.0 - img_array
                inversion_title = '2. Inverted to White-on-Black'
            else:
                processed_inversion = img_array.copy()
                inversion_title = '2. White-on-Black (No Invert)'
                
            axes[0, 1].imshow(processed_inversion, cmap='gray')
            axes[0, 1].set_title(inversion_title)
            axes[0, 1].axis('off')
            
            # Centered
            from scipy.ndimage import center_of_mass
            nonzero = np.nonzero(processed_inversion > 50)
            if len(nonzero[0]) > 0:
                min_row, max_row = np.min(nonzero[0]), np.max(nonzero[0])
                min_col, max_col = np.min(nonzero[1]), np.max(nonzero[1])
                digit_region = processed_inversion[min_row:max_row+1, min_col:max_col+1]
                axes[0, 2].imshow(digit_region, cmap='gray')
                axes[0, 2].set_title('3. Region Extraction')
                axes[0, 2].axis('off')
            else:
                axes[0, 2].imshow(processed_inversion, cmap='gray')
                axes[0, 2].set_title('3. Region Extraction (Empty)')
                axes[0, 2].axis('off')
            
            # Resized
            resized = Image.fromarray(processed_inversion).resize((28, 28), Image.Resampling.LANCZOS)
            axes[1, 0].imshow(np.array(resized), cmap='gray')
            axes[1, 0].set_title('4. Resized to 28x28')
            axes[1, 0].axis('off')
            
            # Final (with enhancements)
            final = preprocess_image(self.image, enhance=True, mode=self.current_mode)[0, :, :, 0]
            
            # Normalized
            if self.current_mode == 'digit':
                normalized = (np.array(resized).astype('float32') / 255.0) - 0.5
                norm_title = '5. Normalized [-0.5, 0.5]'
                display_norm = normalized + 0.5
                display_final = final + 0.5
            else:
                normalized = np.array(resized).astype('float32') / 255.0
                norm_title = '5. Normalized [0.0, 1.0]'
                display_norm = normalized
                display_final = final
                
            axes[1, 1].imshow(display_norm, cmap='gray', vmin=0, vmax=1)
            axes[1, 1].set_title(norm_title)
            axes[1, 1].axis('off')
            
            axes[1, 2].imshow(display_final, cmap='gray', vmin=0, vmax=1)
            axes[1, 2].set_title('6. Final (Enhanced)')
            axes[1, 2].axis('off')
            
            plt.suptitle('Preprocessing Pipeline Debug', fontsize=16, fontweight='bold')
            plt.tight_layout()
            plt.savefig('debug_preprocessing_pipeline.png', dpi=150, bbox_inches='tight')
            plt.show()
            
            messagebox.showinfo("Debug", "Preprocessing pipeline visualization saved as:\n"
                                      "- debug_original.png\n"
                                      "- debug_preprocessing_pipeline.png\n\n"
                                      "Check the images to see each step!")

            
        except Exception as e:
            messagebox.showerror("Debug Error", f"Failed to debug preprocessing: {str(e)}")
    
    def show_enhanced_about(self):
        """Show enhanced about dialog"""
        about_text = """
🤖 AI Recognition System - Enhanced
Version 2.0 - Maximum Accuracy Edition

Built with:
- TensorFlow 2.x / Keras
- Python & Tkinter
- Convolutional Neural Networks (CNN)
- Ensemble Learning Techniques

ENHANCEMENTS:
✅ Advanced CNN Architecture
  - Batch Normalization
  - L2 Regularization
  - Multiple Conv Layers
  - Dropout for Overfitting

✅ Enhanced Preprocessing
  - Automatic inversion
  - Digit centering
  - Gaussian smoothing
  - Contrast enhancement
  - Noise reduction

✅ Ensemble Learning
  - Multiple model training
  - Soft voting predictions
  - 99%+ accuracy
  - Robust predictions

✅ Improved GUI
  - Black canvas with white drawing
  - Confidence gauges
  - Ensemble/single mode toggle
  - Detailed model info
  - Debug visualization

ACCURACY IMPROVEMENT:
  Basic Model: 95-97%
  Enhanced Model: 98-99%
  Ensemble Model: 99%+

FEATURES:
✅ Real-time drawing & prediction
✅ Dual recognition modes
✅ Confidence scoring
✅ Training visualization
✅ Preprocessing debugging

TRAINING TIME:
  Basic: 10-15 minutes
  Enhanced: 20-30 minutes
  Ensemble: 60-90 minutes

Developed for Maximum Accuracy ML Project
        """
        messagebox.showinfo("About Enhanced System", about_text)


def main():
    """Launch the enhanced GUI application"""
    
    print("\n" + "=" * 70)
    print("🚀 ENHANCED GUI APPLICATION LAUNCHED")
    print("=" * 70)
    print("\n🎯 Features:")
    print("  - Black canvas with white drawing")
    print("  - Enhanced preprocessing (auto-inversion, centering)")
    print("  - Ensemble predictions for maximum accuracy")
    print("  - Confidence scoring and visualization")
    print("  - Detailed model information")
    
    print("\n✅ Instructions:")
    print("  1. Draw a digit or shape in the center")
    print("  2. Click 'Predict (Ensemble)' for best accuracy")
    print("  3. View confidence scores and all predictions")
    print("  4. Use 'Debug Preprocessing' to see each step")
    
    print("\n⚙️  Model Status:")
    
    # Create and run GUI
    root = tk.Tk()
    app = EnhancedDrawingApp(root)
    
    root.mainloop()


if __name__ == "__main__":
    main()