"""
AI Recognition System - GUI Application
Tkinter-based drawing interface for real-time predictions
"""

import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageDraw, ImageTk
import numpy as np
import tensorflow as tf
from tensorflow import keras
import sys
import os
from pathlib import Path

# Import utilities
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import preprocess_image

# Resolve project paths robustly (works no matter the current working directory)
SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent
MODELS_DIR = PROJECT_ROOT / "models"


class DrawingApp:
    """Main GUI Application for AI Recognition"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("🤖 AI Recognition System")
        self.root.geometry("800x600")
        self.root.resizable(False, False)
        
        # Model variables
        self.digit_model = None
        self.shape_model = None
        self.current_mode = 'digit'  # 'digit' or 'shape'
        
        # Drawing variables
        self.canvas_size = 280
        self.pen_size = 15
        self.image = Image.new('L', (self.canvas_size, self.canvas_size), 0)
        self.draw = ImageDraw.Draw(self.image)
        
        # Load models
        self.load_models()
        
        # Setup GUI
        self.setup_gui()
        
    def load_models(self):
        """Load pre-trained models"""
        try:
            digit_path = MODELS_DIR / "digit_model.h5"
            shape_path = MODELS_DIR / "shape_model.h5"

            if digit_path.exists():
                self.digit_model = keras.models.load_model(str(digit_path))
                print("✅ Digit model loaded successfully")
            else:
                print(f"⚠️  Digit model not found at: {digit_path}")
            
            if shape_path.exists():
                self.shape_model = keras.models.load_model(str(shape_path))
                print("✅ Shape model loaded successfully")
            else:
                print(f"⚠️  Shape model not found at: {shape_path}")
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load models: {str(e)}")
    
    def setup_gui(self):
        """Setup the GUI interface"""
        
        # Main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # ==================== HEADER ====================
        header_frame = ttk.Frame(main_frame)
        header_frame.grid(row=0, column=0, columnspan=2, pady=(0, 10))
        
        title_label = ttk.Label(
            header_frame, 
            text="🤖 AI Recognition System",
            font=('Arial', 20, 'bold')
        )
        title_label.pack()
        
        subtitle_label = ttk.Label(
            header_frame,
            text="Draw digits or shapes and see AI predictions in real-time",
            font=('Arial', 10)
        )
        subtitle_label.pack()
        
        # ==================== MODE SELECTOR ====================
        mode_frame = ttk.LabelFrame(main_frame, text="Recognition Mode", padding="10")
        mode_frame.grid(row=1, column=0, columnspan=2, pady=(0, 10), sticky=(tk.W, tk.E))
        
        self.mode_var = tk.StringVar(value='digit')
        
        digit_radio = ttk.Radiobutton(
            mode_frame,
            text="🔢 Digit Recognition (0-9)",
            variable=self.mode_var,
            value='digit',
            command=self.switch_mode
        )
        digit_radio.pack(side=tk.LEFT, padx=10)
        
        shape_radio = ttk.Radiobutton(
            mode_frame,
            text="⚪ Shape Recognition (Circle, Square, Triangle)",
            variable=self.mode_var,
            value='shape',
            command=self.switch_mode
        )
        shape_radio.pack(side=tk.LEFT, padx=10)
        
        # ==================== LEFT PANEL: CANVAS ====================
        left_frame = ttk.LabelFrame(main_frame, text="Drawing Canvas", padding="10")
        left_frame.grid(row=2, column=0, padx=(0, 10), sticky=(tk.N, tk.S))
        
        # Canvas - CHANGED TO BLACK BACKGROUND
        self.canvas = tk.Canvas(
            left_frame,
            width=self.canvas_size,
            height=self.canvas_size,
            bg='black',  # CHANGED FROM 'white' TO 'black'
            cursor='crosshair'
        )
        self.canvas.pack()
        
        # Canvas bindings
        self.canvas.bind('<B1-Motion>', self.paint)
        self.canvas.bind('<Button-1>', self.start_paint)
        
        # Instructions - UPDATED TEXT
        instructions = ttk.Label(
            left_frame,
            text="📝 Click and drag to draw (white on black)\n✨ Draw will automatically predict\n🎯 Draw digits in the CENTER",
            justify=tk.CENTER,
            font=('Arial', 9)
        )
        instructions.pack(pady=(10, 0))
        
        # Control buttons
        button_frame = ttk.Frame(left_frame)
        button_frame.pack(pady=(10, 0))
        
        ttk.Button(
            button_frame,
            text="🗑️ Clear",
            command=self.clear_canvas,
            width=12
        ).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(
            button_frame,
            text="🔮 Predict",
            command=self.predict,
            width=12
        ).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(
            button_frame,
            text="🐛 Debug",
            command=self.debug_preprocessing,
            width=12
        ).pack(side=tk.LEFT, padx=5)
        
        # ==================== RIGHT PANEL: PREDICTIONS ====================
        right_frame = ttk.LabelFrame(main_frame, text="AI Predictions", padding="10")
        right_frame.grid(row=2, column=1, sticky=(tk.N, tk.S))
        
        # Model status
        status_frame = ttk.Frame(right_frame)
        status_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.status_label = ttk.Label(
            status_frame,
            text="🟢 Models Ready" if self.digit_model and self.shape_model else "🔴 Models Not Loaded",
            font=('Arial', 10, 'bold')
        )
        self.status_label.pack()
        
        # Results display
        self.results_text = tk.Text(
            right_frame,
            width=40,
            height=20,
            font=('Courier', 10),
            state=tk.DISABLED,
            bg='#f0f0f0'
        )
        self.results_text.pack(fill=tk.BOTH, expand=True)
        
        # Model info
        info_frame = ttk.Frame(right_frame)
        info_frame.pack(fill=tk.X, pady=(10, 0))
        
        self.info_label = ttk.Label(
            info_frame,
            text="Model: CNN with 2 Conv2D layers\nAccuracy: 98.5% (digits) | 96.2% (shapes)\n\n💡 TIP: Draw digits in the center",
            font=('Arial', 8),
            justify=tk.LEFT
        )
        self.info_label.pack()
        
        # ==================== FOOTER ====================
        footer_frame = ttk.Frame(main_frame)
        footer_frame.grid(row=3, column=0, columnspan=2, pady=(10, 0))
        
        ttk.Button(
            footer_frame,
            text="ℹ️ About",
            command=self.show_about
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
        # Draw on canvas widget - CHANGED TO WHITE ON BLACK
        x1, y1 = event.x - self.pen_size//2, event.y - self.pen_size//2
        x2, y2 = event.x + self.pen_size//2, event.y + self.pen_size//2
        
        # CHANGED FROM 'black' TO 'white'
        self.canvas.create_oval(x1, y1, x2, y2, fill='white', outline='white')
        
        # Draw on PIL image - Use 255 (white) which will become black after inversion
        self.draw.ellipse([x1, y1, x2, y2], fill=255, outline=255)
        
        self.last_x = event.x
        self.last_y = event.y
    
    def clear_canvas(self):
        """Clear the canvas"""
        self.canvas.delete('all')
        # Reset canvas background to black
        self.canvas.configure(bg='black')
        self.image = Image.new('L', (self.canvas_size, self.canvas_size), 0)
        self.draw = ImageDraw.Draw(self.image)
        
        # Clear results
        self.results_text.config(state=tk.NORMAL)
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(tk.END, "Canvas cleared. Draw something to predict!")
        self.results_text.config(state=tk.DISABLED)
    
    def switch_mode(self):
        """Switch between digit and shape recognition"""
        self.current_mode = self.mode_var.get()
        self.clear_canvas()
        
        mode_name = "Digit" if self.current_mode == 'digit' else "Shape"
        messagebox.showinfo("Mode Changed", f"Switched to {mode_name} Recognition mode")
    
    def predict(self):
        """Make prediction on current drawing"""
        
        # Check if canvas is blank
        if not self.is_canvas_drawn():
            messagebox.showwarning("Empty Canvas", "Please draw something first!")
            return
        
        # Check if models are loaded
        if self.current_mode == 'digit' and not self.digit_model:
            messagebox.showerror("Model Error", "Digit model not loaded. Train it first!")
            return
        
        if self.current_mode == 'shape' and not self.shape_model:
            messagebox.showerror("Model Error", "Shape model not loaded. Train it first!")
            return
        
        try:
            # Use preprocess_image from utils.py with enhance=False for simple preprocessing
            img_array = preprocess_image(self.image, target_size=(28, 28), enhance=False, mode=self.current_mode)
            
            # Predict
            if self.current_mode == 'digit':
                predictions = self.digit_model.predict(img_array, verbose=0)[0]
                labels = [str(i) for i in range(10)]
            else:
                predictions = self.shape_model.predict(img_array, verbose=0)[0]
                labels = ['Circle', 'Square', 'Triangle']
            
            # Display results
            self.display_predictions(predictions, labels)
            
        except Exception as e:
            messagebox.showerror("Prediction Error", f"Error during prediction: {str(e)}")
    
    def debug_preprocessing(self):
        """Save current image for debugging"""
        if not self.is_canvas_drawn():
            messagebox.showwarning("Empty Canvas", "Please draw something first!")
            return
        
        try:
            # Save original
            self.image.save('debug_original.png')
            
            # Process like in predict()
            img_array = preprocess_image(self.image, target_size=(28, 28), enhance=False, mode=self.current_mode)
            
            # Rescale back to 0-255 for display/saving
            if self.current_mode == 'digit':
                # Convert back from [-0.5, 0.5] to [0, 255]
                processed_img = ((img_array[0, :, :, 0] + 0.5) * 255.0).astype(np.uint8)
            else:
                # Convert back from [0.0, 1.0] to [0, 255]
                processed_img = (img_array[0, :, :, 0] * 255.0).astype(np.uint8)
                
            Image.fromarray(processed_img).save('debug_processed.png')
            
            # Show info
            messagebox.showinfo(
                "Debug", 
                "Debug images saved as:\n"
                "- debug_original.png (what you drew - white on black)\n"
                "- debug_processed.png (what model sees - normalized & centered correctly)\n\n"
                "Check if processed image looks like MNIST digits!\n"
                "It should be a white digit on black background."
            )
            
        except Exception as e:
            messagebox.showerror("Debug Error", f"Failed to save debug images: {str(e)}")

    
    def display_predictions(self, predictions, labels):
        """Display prediction results"""
        
        # Sort predictions
        sorted_indices = np.argsort(predictions)[::-1]
        
        # Format output
        output = "=" * 40 + "\n"
        output += "🔮 AI PREDICTION RESULTS\n"
        output += "=" * 40 + "\n\n"
        
        output += "🏆 TOP PREDICTION:\n"
        top_label = labels[sorted_indices[0]]
        top_conf = predictions[sorted_indices[0]] * 100
        output += f"   {top_label}: {top_conf:.1f}%\n\n"
        
        # Confidence interpretation
        if top_conf > 80:
            output += "✅ High confidence prediction\n\n"
        elif top_conf > 50:
            output += "⚠️  Medium confidence prediction\n\n"
        else:
            output += "❓ Low confidence - try drawing clearer\n\n"
        
        output += "📊 ALL PREDICTIONS:\n"
        output += "-" * 40 + "\n"
        
        for idx in sorted_indices:
            label = labels[idx]
            confidence = predictions[idx] * 100
            bar_length = int(confidence / 5)
            bar = "█" * bar_length
            
            output += f"{label:10s} │ {bar:<20s} {confidence:5.1f}%\n"
        
        output += "=" * 40 + "\n"
        
        # Update text widget
        self.results_text.config(state=tk.NORMAL)
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(tk.END, output)
        self.results_text.config(state=tk.DISABLED)
    
    def is_canvas_drawn(self):
        """Check if anything is drawn on canvas"""
        img_array = np.array(self.image)
        return np.sum(img_array) > 0
    
    def show_about(self):
        """Show about dialog"""
        about_text = """
AI Recognition System
Version 1.2 - Black Canvas Update

Built with:
- TensorFlow / Keras
- Python & Tkinter
- Convolutional Neural Networks

NEW FEATURE:
🎨 Black canvas with white drawing
✨ Better visual experience

Features:
✅ Digit Recognition (0-9)
✅ Shape Recognition (Circle, Square, Triangle)
✅ Real-time Predictions
✅ 98.5% Accuracy

Drawing Tips:
1. Draw digits in the CENTER
2. Make digits reasonably sized (medium)
3. Draw clearly and consistently
4. White on black background

Troubleshooting:
- Use Debug button to see what model sees
- Clear canvas and try again if confused
- Make sure models are trained first

Developed for Machine Learning Project
        """
        messagebox.showinfo("About", about_text)


def main():
    """Launch the GUI application"""
    
    # Check if models exist
    digit_path = MODELS_DIR / "digit_model.h5"
    shape_path = MODELS_DIR / "shape_model.h5"

    if not digit_path.exists() or not shape_path.exists():
        print("\n" + "=" * 70)
        print("⚠️  WARNING: Models not found!")
        print("=" * 70)
        print("\nPlease train the models first by running:")
        print("   python src/train_models.py")
        print("\nContinuing anyway (models won't work until trained)...\n")
    
    # Create and run GUI
    root = tk.Tk()
    app = DrawingApp(root)
    
    print("\n" + "=" * 70)
    print("🚀 GUI APPLICATION LAUNCHED - BLACK CANVAS VERSION")
    print("=" * 70)
    print("\n✅ Canvas: BLACK background with WHITE drawing")
    print("✅ Draw digits in the CENTER of the canvas")
    print("✅ Click Predict to see accuracy")
    print("✅ Use Debug button to check preprocessing")
    print("✅ Switch modes to test digit or shape recognition")
    print("✅ Close the window to exit\n")
    print("🎯 DRAWING TIPS:")
    print("   - Draw digits clearly")
    print("   - Center your drawing")
    print("   - Make digits medium-sized")
    print("   - For '7': horizontal top + diagonal down\n")
    print("🔧 FEATURES:")
    print("   - Black canvas with white drawing")
    print("   - Color inversion fixed")
    print("   - Better preprocessing\n")
    
    root.mainloop()


if __name__ == "__main__":
    main()