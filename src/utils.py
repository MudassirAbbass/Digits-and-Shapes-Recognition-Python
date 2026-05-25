"""
AI Recognition System - Enhanced Utility Functions
Helper functions for data processing, visualization, and model operations
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFilter
import os
import tensorflow as tf
from tensorflow import keras
from scipy.ndimage import gaussian_filter, binary_dilation, binary_erosion


def create_directories():
    """Create necessary project directories if they don't exist"""
    directories = ['models', 'data', 'docs', 'notebooks', 'models/logs', 'models/ensemble']
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"✅ Created directory: {directory}")


def preprocess_image(image, target_size=(28, 28), enhance=True, mode='digit'):
    """
    Enhanced preprocessing for model input with accuracy improvements
    
    WHAT THIS FUNCTION DOES:
    ========================
    Converts a drawing/image to the exact format the CNN model expects.
    This is CRITICAL - wrong preprocessing = wrong predictions!
    
    PREPROCESSING STEPS:
    ===================
    1. Convert to grayscale (if color image)
    2. Resize to 28×28 (model input size)
    3. Auto-invert colors (if needed for MNIST format)
    4. Center the digit/shape (if enhance=True)
    5. Apply Gaussian blur (smooth edges)
    6. Normalize depending on mode ('digit': [-0.5, 0.5], 'shape': [0.0, 1.0])
    7. Add batch dimension: (28, 28) → (1, 28, 28, 1)
    
    Args:
        image: PIL Image or numpy array (the drawing/image to preprocess)
        target_size: Tuple (height, width) - default (28, 28) for MNIST
        enhance: Boolean - if True, applies advanced enhancements (centering, blur, etc.)
        mode: String - 'digit' or 'shape' (determines normalization range)
    
    Returns:
        Preprocessed numpy array ready for model
        Shape: (1, 28, 28, 1) - batch of 1 image, 28×28, 1 channel (grayscale)
    """
    # STEP 1: Convert to PIL Image if it's a numpy array
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    
    # STEP 2: Convert to grayscale
    image = image.convert('L')
    
    # STEP 3: Resize to target size (28×28)
    image = image.resize(target_size, Image.Resampling.LANCZOS)
    
    # STEP 4: Convert to numpy array for processing
    img_array = np.array(image).astype('float32')
    
    # STEP 5: Auto-invert colors if needed
    # MNIST and shape generator format: White digit/shape on black background.
    # If image is mostly white (mean > 127), invert it.
    if np.mean(img_array) > 127:
        img_array = 255.0 - img_array
    
    # STEP 6: Apply enhancements (if enabled)
    if enhance:
        # ENHANCEMENT 1: Center the digit/shape
        nonzero = np.nonzero(img_array > 50)
        if len(nonzero[0]) > 0 and len(nonzero[1]) > 0:
            min_row, max_row = np.min(nonzero[0]), np.max(nonzero[0])
            min_col, max_col = np.min(nonzero[1]), np.max(nonzero[1])
            
            digit_region = img_array[min_row:max_row+1, min_col:max_col+1]
            
            height, width = digit_region.shape
            pad_height_top = max(0, (target_size[0] - height) // 2)
            pad_height_bottom = target_size[0] - height - pad_height_top
            pad_width_left = max(0, (target_size[1] - width) // 2)
            pad_width_right = target_size[1] - width - pad_width_left
            
            centered = np.zeros(target_size)
            centered[pad_height_top:pad_height_top+height, 
                     pad_width_left:pad_width_left+width] = digit_region
            img_array = centered
        
        # ENHANCEMENT 2: Apply Gaussian blur to smooth edges
        img_array = gaussian_filter(img_array, sigma=0.5)
        
        # ENHANCEMENT 3: Normalize and Apply Contrast Enhancement depending on mode
        if mode == 'digit':
            img_array = (img_array / 255.0) - 0.5
            img_array = np.clip(img_array * 1.2, -0.5, 0.5)
        else:
            img_array = img_array / 255.0
            img_array = np.clip(img_array * 1.2, 0.0, 1.0)
            
    else:
        # BASIC MODE: Just normalize (no enhancements)
        if mode == 'digit':
            img_array = (img_array / 255.0) - 0.5
        else:
            img_array = img_array / 255.0
    
    # STEP 7: Add batch and channel dimensions
    img_array = np.expand_dims(img_array, axis=(0, -1))
    
    return img_array


def generate_enhanced_shapes(num_samples=10000, image_size=28, augment=True):
    """
    Generate enhanced custom dataset of geometric shapes with variations
    
    Args:
        num_samples: Total number of samples to generate
        image_size: Size of each image (square)
        augment: Apply augmentation during generation
    
    Returns:
        x_data: Images array
        y_data: Labels array (0=circle, 1=square, 2=triangle)
    """
    print(f"\n📊 Generating {num_samples} enhanced shape images...")
    
    x_data = []
    y_data = []
    
    samples_per_class = num_samples // 3
    
    for i in range(num_samples):
        # Create blank image
        img = Image.new('L', (image_size, image_size), 0)
        draw = ImageDraw.Draw(img)
        
        # Add more randomness for better generalization
        margin = np.random.randint(3, 6)
        max_size = image_size - 2 * margin
        
        # Random size variation
        size_variation = np.random.randint(-3, 4)
        actual_size = max_size + size_variation
        
        # Random position variation
        pos_variation_x = np.random.randint(-3, 4)
        pos_variation_y = np.random.randint(-3, 4)
        
        # Random rotation for triangles
        rotation = np.random.randint(0, 360) if augment else 0
        
        if i < samples_per_class:
            # Circle with variations
            x1 = margin + pos_variation_x
            y1 = margin + pos_variation_y
            x2 = margin + actual_size + pos_variation_x
            y2 = margin + actual_size + pos_variation_y
            
            # Sometimes make ellipse instead of perfect circle
            if augment and np.random.random() > 0.7:
                # Ellipse
                x2 = x2 + np.random.randint(-2, 3)
                draw.ellipse([x1, y1, x2, y2], fill=255)
            else:
                # Circle
                draw.ellipse([x1, y1, x2, y2], fill=255)
            label = 0
            
        elif i < 2 * samples_per_class:
            # Square with variations
            x1 = margin + pos_variation_x
            y1 = margin + pos_variation_y
            x2 = margin + actual_size + pos_variation_x
            y2 = margin + actual_size + pos_variation_y
            
            # Sometimes make rectangle instead of square
            if augment and np.random.random() > 0.7:
                x2 = x2 + np.random.randint(-3, 4)
            
            # Sometimes rounded corners
            if augment and np.random.random() > 0.8:
                radius = np.random.randint(1, 4)
                draw.rounded_rectangle([x1, y1, x2, y2], radius=radius, fill=255)
            else:
                draw.rectangle([x1, y1, x2, y2], fill=255)
            label = 1
            
        else:
            # Triangle with variations
            center_x = image_size // 2 + pos_variation_x
            center_y = image_size // 2 + pos_variation_y
            
            size_factor = actual_size / max_size
            
            # Triangle points
            if rotation == 0:
                points = [
                    (center_x, margin + pos_variation_y),  # Top
                    (margin + pos_variation_x, margin + actual_size + pos_variation_y),  # Bottom left
                    (margin + actual_size + pos_variation_x, margin + actual_size + pos_variation_y)  # Bottom right
                ]
            else:
                # Rotated triangle
                radius = actual_size / 2
                angles = [rotation, rotation + 120, rotation + 240]
                points = []
                for angle in angles:
                    rad = np.deg2rad(angle)
                    x = center_x + radius * np.cos(rad)
                    y = center_y + radius * np.sin(rad)
                    points.append((x, y))
            
            draw.polygon(points, fill=255)
            label = 2
        
        # Convert to numpy array
        img_array = np.array(img).astype('float32') / 255.0
        
        # Apply augmentation during generation
        if augment:
            # Random brightness adjustment
            brightness = np.random.uniform(0.8, 1.2)
            img_array = np.clip(img_array * brightness, 0, 1)
            
            # Random contrast adjustment
            contrast = np.random.uniform(0.8, 1.2)
            mean = np.mean(img_array)
            img_array = np.clip((img_array - mean) * contrast + mean, 0, 1)
            
            # Add slight noise
            if np.random.random() > 0.5:
                noise = np.random.normal(0, 0.05, img_array.shape)
                img_array = np.clip(img_array + noise, 0, 1)
        
        x_data.append(img_array)
        y_data.append(label)
        
        # Progress indicator
        if (i + 1) % 1000 == 0:
            print(f"  Generated {i + 1}/{num_samples} images...")
    
    # Convert to numpy arrays
    x_data = np.array(x_data, dtype='float32')
    x_data = np.expand_dims(x_data, -1)
    y_data = np.array(y_data)
    
    print(f"✅ Enhanced shape dataset generated: {x_data.shape}")
    
    return x_data, y_data


def augment_data(x_data, y_data, augmentation_factor=3):
    """
    Enhanced data augmentation for training
    
    Args:
        x_data: Input images
        y_data: Labels
        augmentation_factor: How many augmented samples per original
    
    Returns:
        Augmented dataset
    """
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    
    print(f"📊 Augmenting data by factor {augmentation_factor}...")
    
    # Create data generator with various augmentations
    datagen = ImageDataGenerator(
        rotation_range=12,          # Rotate up to 12 degrees
        width_shift_range=0.12,     # Shift horizontally up to 12%
        height_shift_range=0.12,    # Shift vertically up to 12%
        zoom_range=0.12,            # Zoom in/out up to 12%
        shear_range=0.12,           # Shear transformation
        brightness_range=[0.85, 1.15],  # Brightness adjustment
        fill_mode='nearest'         # Fill missing pixels
    )
    
    # Fit the data generator
    datagen.fit(x_data)
    
    augmented_images = []
    augmented_labels = []
    
    # Generate augmented samples
    for i in range(len(x_data)):
        img = x_data[i]
        label = y_data[i]
        
        # Reshape for flow method
        img_reshaped = img.reshape(1, *img.shape)
        
        # Generate augmented samples
        count = 0
        for batch in datagen.flow(img_reshaped, batch_size=1):
            augmented_images.append(batch[0])
            augmented_labels.append(label)
            count += 1
            if count >= augmentation_factor:
                break
    
    # Combine original and augmented data
    if len(augmented_images) > 0:
        all_images = np.vstack([x_data, np.array(augmented_images)])
        all_labels = np.concatenate([y_data, np.array(augmented_labels)])
    else:
        all_images = x_data
        all_labels = y_data
    
    print(f"✅ Data augmented: {len(x_data)} → {len(all_images)} samples")
    
    return all_images, all_labels


def plot_sample_images(x_data, y_data, n_samples=10, class_names=None, save_path=None):
    """
    Plot sample images from dataset
    
    Args:
        x_data: Images array
        y_data: Labels array
        n_samples: Number of samples to plot
        class_names: List of class names
        save_path: Path to save figure
    """
    fig, axes = plt.subplots(2, 5, figsize=(12, 6))
    axes = axes.flatten()
    
    for i in range(min(n_samples, len(x_data))):
        # Handle different image formats
        img_to_show = x_data[i].squeeze()
        
        # If image is normalized to [-0.5, 0.5], convert to [0, 1] for display
        if img_to_show.min() < 0:
            img_to_show = (img_to_show + 0.5)  # Convert from [-0.5, 0.5] to [0, 1]
        
        axes[i].imshow(img_to_show, cmap='gray', vmin=0, vmax=1)
        
        if class_names:
            label = class_names[y_data[i]]
        else:
            label = y_data[i]
        
        axes[i].set_title(f'Label: {label}')
        axes[i].axis('off')
    
    plt.suptitle('Sample Images from Dataset', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ Sample images saved to {save_path}")
    
    plt.show()


def plot_training_history(history, model_name='Model', save_path=None):
    """
    Enhanced training history visualization
    
    Args:
        history: Keras History object
        model_name: Name for plot title
        save_path: Path to save figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot accuracy
    axes[0].plot(history.history['accuracy'], label='Training Accuracy', linewidth=2, marker='o', markersize=4)
    axes[0].plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2, marker='s', markersize=4)
    axes[0].set_title(f'{model_name} - Accuracy', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Accuracy', fontsize=12)
    axes[0].legend(fontsize=10, loc='lower right')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim([0, 1.05])
    
    # Add final accuracy values
    final_train_acc = history.history['accuracy'][-1]
    final_val_acc = history.history['val_accuracy'][-1]
    axes[0].axhline(y=final_train_acc, color='blue', linestyle='--', alpha=0.3)
    axes[0].axhline(y=final_val_acc, color='orange', linestyle='--', alpha=0.3)
    
    # Plot loss
    axes[1].plot(history.history['loss'], label='Training Loss', linewidth=2, marker='o', markersize=4)
    axes[1].plot(history.history['val_loss'], label='Validation Loss', linewidth=2, marker='s', markersize=4)
    axes[1].set_title(f'{model_name} - Loss', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Loss', fontsize=12)
    axes[1].legend(fontsize=10, loc='upper right')
    axes[1].grid(True, alpha=0.3)
    
    plt.suptitle(f'Training History - {model_name}', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ Training history saved to {save_path}")
    
    plt.show()


def print_model_summary(model, model_name='Model'):
    """
    Print detailed model summary with formatted output
    
    Args:
        model: Keras model
        model_name: Name of the model
    """
    print("\n" + "=" * 70)
    print(f"📊 {model_name.upper()} ARCHITECTURE")
    print("=" * 70)
    
    model.summary()
    
    total_params = model.count_params()
    trainable_params = sum([np.prod(v.get_shape()) for v in model.trainable_weights])
    non_trainable_params = total_params - trainable_params
    
    print("\n" + "-" * 70)
    print(f"📈 Total Parameters: {total_params:,}")
    print(f"📈 Trainable Parameters: {trainable_params:,}")
    print(f"📈 Non-Trainable Parameters: {non_trainable_params:,}")
    print("-" * 70)


def evaluate_model(model, x_test, y_test, class_names=None, verbose=1):
    """
    Enhanced model evaluation with confusion matrix
    
    Args:
        model: Trained Keras model
        x_test: Test images
        y_test: Test labels (one-hot encoded)
        class_names: List of class names
        verbose: Print detailed results
    
    Returns:
        Dictionary with evaluation metrics
    """
    if verbose:
        print("\n" + "=" * 70)
        print("📊 ENHANCED MODEL EVALUATION")
        print("=" * 70)
    
    # Evaluate
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    
    if verbose:
        print(f"\n✅ Test Loss:     {test_loss:.4f}")
        print(f"✅ Test Accuracy: {test_accuracy * 100:.2f}%")
    
    # Get predictions
    predictions = model.predict(x_test, verbose=0)
    pred_classes = np.argmax(predictions, axis=1)
    true_classes = np.argmax(y_test, axis=1)
    
    # Calculate confidence scores
    confidence_scores = np.max(predictions, axis=1)
    avg_confidence = np.mean(confidence_scores) * 100
    
    if verbose:
        print(f"✅ Average Confidence: {avg_confidence:.2f}%")
    
    # Per-class accuracy
    if class_names and verbose:
        print("\n📈 Per-Class Accuracy and Confidence:")
        for i, name in enumerate(class_names):
            mask = true_classes == i
            if np.sum(mask) > 0:
                class_acc = np.mean(pred_classes[mask] == true_classes[mask]) * 100
                class_conf = np.mean(confidence_scores[mask]) * 100
                print(f"   {name:12s}: Accuracy={class_acc:6.2f}% | Confidence={class_conf:6.2f}%")
    
    # Confusion matrix
    if class_names and verbose and len(class_names) <= 10:
        from sklearn.metrics import confusion_matrix
        import seaborn as sns
        
        cm = confusion_matrix(true_classes, pred_classes)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
        plt.xlabel('Predicted Label', fontsize=12)
        plt.ylabel('True Label', fontsize=12)
        plt.tight_layout()
        
        # Save confusion matrix
        cm_path = 'models/confusion_matrix.png'
        plt.savefig(cm_path, dpi=150, bbox_inches='tight')
        print(f"✅ Confusion matrix saved to {cm_path}")
        plt.show()
    
    if verbose:
        print("=" * 70 + "\n")
    
    return {
        'loss': test_loss,
        'accuracy': test_accuracy,
        'avg_confidence': avg_confidence,
        'predictions': predictions,
        'pred_classes': pred_classes,
        'true_classes': true_classes,
        'confidence_scores': confidence_scores
    }


def save_model_info(model, history, save_path='models/model_info.txt'):
    """
    Save enhanced model information to text file
    
    Args:
        model: Keras model
        history: Training history
        save_path: Path to save file
    """
    with open(save_path, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("🤖 AI RECOGNITION SYSTEM - ENHANCED MODEL INFORMATION\n")
        f.write("=" * 70 + "\n\n")
        
        # Model architecture info
        f.write("MODEL ARCHITECTURE:\n")
        f.write("-" * 40 + "\n")
        f.write(f"Total Parameters: {model.count_params():,}\n")
        f.write(f"Number of Layers: {len(model.layers)}\n")
        f.write(f"Input Shape: {model.input_shape}\n")
        f.write(f"Output Shape: {model.output_shape}\n")
        
        # Training results
        f.write("\nTRAINING RESULTS:\n")
        f.write("-" * 40 + "\n")
        f.write(f"Final Training Accuracy: {history.history['accuracy'][-1]*100:.2f}%\n")
        f.write(f"Final Validation Accuracy: {history.history['val_accuracy'][-1]*100:.2f}%\n")
        f.write(f"Final Training Loss: {history.history['loss'][-1]:.4f}\n")
        f.write(f"Final Validation Loss: {history.history['val_loss'][-1]:.4f}\n")
        
        # Best epoch
        best_val_acc = max(history.history['val_accuracy'])
        best_epoch = history.history['val_accuracy'].index(best_val_acc) + 1
        f.write(f"\nBest Validation Accuracy: {best_val_acc*100:.2f}% (Epoch {best_epoch})\n")
        
        # Learning trends
        f.write("\nLEARNING TRENDS:\n")
        f.write("-" * 40 + "\n")
        accuracy_gain = (history.history['accuracy'][-1] - history.history['accuracy'][0]) * 100
        loss_reduction = (history.history['loss'][0] - history.history['loss'][-1]) * 100
        f.write(f"Accuracy Gain: {accuracy_gain:+.2f}%\n")
        f.write(f"Loss Reduction: {loss_reduction:+.2f}%\n")
        
        # Overfitting analysis
        overfit = history.history['accuracy'][-1] - history.history['val_accuracy'][-1]
        f.write(f"Overfitting (Train - Val Accuracy): {overfit*100:.2f}%\n")
        
        f.write("\n" + "=" * 70 + "\n")
    
    print(f"✅ Enhanced model info saved to {save_path}")


def create_ensemble_predictions(models, x_data):
    """
    Create ensemble predictions from multiple models
    
    Args:
        models: List of trained models
        x_data: Input data
    
    Returns:
        Ensemble predictions
    """
    all_predictions = []
    
    for i, model in enumerate(models):
        pred = model.predict(x_data, verbose=0)
        all_predictions.append(pred)
        print(f"  Model {i+1} predictions generated")
    
    # Average predictions (soft voting)
    ensemble_pred = np.mean(all_predictions, axis=0)
    
    return ensemble_pred


def analyze_failed_predictions(model, x_test, y_test, class_names=None, n_samples=10):
    """
    Analyze failed predictions to understand model weaknesses
    
    Args:
        model: Trained model
        x_test: Test images
        y_test: Test labels (one-hot encoded)
        class_names: List of class names
        n_samples: Number of failed samples to analyze
    """
    predictions = model.predict(x_test, verbose=0)
    pred_classes = np.argmax(predictions, axis=1)
    true_classes = np.argmax(y_test, axis=1)
    
    # Find failed predictions
    failed_indices = np.where(pred_classes != true_classes)[0]
    
    if len(failed_indices) == 0:
        print("🎉 No failed predictions! Model is perfect on test set.")
        return
    
    print(f"\n⚠️  Found {len(failed_indices)} failed predictions")
    print("Analyzing top confused classes...")
    
    # Show some failed predictions
    n_to_show = min(n_samples, len(failed_indices))
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    axes = axes.flatten()
    
    for i, idx in enumerate(failed_indices[:n_to_show]):
        img = x_test[idx].squeeze()
        
        # If image is normalized to [-0.5, 0.5], convert to [0, 1] for display
        if img.min() < 0:
            img = (img + 0.5)
        
        axes[i].imshow(img, cmap='gray', vmin=0, vmax=1)
        
        true_label = class_names[true_classes[idx]] if class_names else true_classes[idx]
        pred_label = class_names[pred_classes[idx]] if class_names else pred_classes[idx]
        confidence = np.max(predictions[idx]) * 100
        
        axes[i].set_title(f'True: {true_label}\nPred: {pred_label}\nConf: {confidence:.1f}%', 
                         fontsize=9)
        axes[i].axis('off')
    
    plt.suptitle('Failed Predictions Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('models/failed_predictions.png', dpi=150, bbox_inches='tight')
    print("✅ Failed predictions analysis saved to models/failed_predictions.png")
    plt.show()


if __name__ == "__main__":
    """Test enhanced utility functions"""
    print("🧪 Testing enhanced utility functions...")
    
    # Test directory creation
    create_directories()
    
    # Test enhanced shape generation
    print("\nTesting enhanced shape generation...")
    x_shapes, y_shapes = generate_enhanced_shapes(num_samples=30, augment=True)
    
    # Test enhanced preprocessing
    print("\nTesting enhanced preprocessing...")
    test_image = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
    processed = preprocess_image(test_image, enhance=True)
    print(f"Processed image shape: {processed.shape}")
    
    # Test plotting
    shape_names = ['Circle', 'Square', 'Triangle']
    plot_sample_images(x_shapes, y_shapes, class_names=shape_names)
    
    print("\n✅ All enhanced utility functions working correctly!")