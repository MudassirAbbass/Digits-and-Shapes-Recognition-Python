"""
AI Recognition System - Enhanced Model Training
Builds and trains advanced CNN models for improved accuracy

WHAT THIS FILE DOES:
====================
This file contains the complete training pipeline for both:
1. Digit Recognition Model (0-9) - trained on MNIST dataset
2. Shape Recognition Model (Circle, Square, Triangle) - trained on synthetic data

It handles:
- Loading/preprocessing data
- Building CNN architectures
- Training with callbacks (early stopping, learning rate scheduling)
- Evaluating model performance
- Saving trained models and training history
"""

# ============================================================================
# IMPORT STATEMENTS - What each library does:
# ============================================================================
import numpy as np  # Numerical operations on arrays (image data)
import tensorflow as tf  # Deep learning framework
from tensorflow import keras  # High-level API for building neural networks
from tensorflow.keras import layers  # Pre-built layer types (Conv2D, Dense, etc.)
from tensorflow.keras.datasets import mnist  # MNIST digit dataset (60K training, 10K test)
from sklearn.model_selection import train_test_split  # Split data into train/validation sets
import matplotlib.pyplot as plt  # Plotting graphs (training curves)
import os  # File system operations
import sys  # System-specific parameters

# Add current directory to Python path so we can import utils.py
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import helper functions from utils.py
from utils import (
    create_directories,  # Creates folders (models/, data/, etc.)
    generate_enhanced_shapes,  # Generates synthetic shape images
    augment_data,  # Creates variations of images (rotate, shift, zoom)
    plot_training_history,  # Plots accuracy/loss curves
    plot_sample_images,  # Shows example images from dataset
    print_model_summary,  # Prints model architecture details
    evaluate_model,  # Calculates test accuracy and other metrics
    save_model_info,  # Saves training results to text file
    analyze_failed_predictions  # Shows which predictions were wrong
)


# ============================================================================
# CLASS: EnhancedDigitRecognitionModel
# ============================================================================
# PURPOSE: This class handles everything related to digit recognition model
# - Loading MNIST dataset
# - Building CNN architecture
# - Training the model
# - Saving the trained model
# ============================================================================
class EnhancedDigitRecognitionModel:
    """
    Advanced CNN Model for MNIST Digit Recognition (0-9)
    
    This class encapsulates the complete workflow for training a digit recognition model:
    1. Loads MNIST dataset (60,000 training images, 10,000 test images)
    2. Preprocesses data (normalize, reshape, one-hot encode)
    3. Builds CNN architecture with multiple convolutional layers
    4. Trains the model with callbacks (early stopping, learning rate scheduling)
    5. Evaluates performance and saves the model
    """
    
    def __init__(self):
        """
        Initialize the model class
        
        Attributes:
        - self.model: The Keras CNN model (built later)
        - self.history: Training history (accuracy, loss over epochs)
        - self.class_names: List of class names ['0', '1', '2', ..., '9']
        """
        self.model = None  # Will store the built CNN model
        self.history = None  # Will store training history (accuracy, loss per epoch)
        self.class_names = [str(i) for i in range(10)]  # ['0', '1', '2', ..., '9']
    
    def load_data(self, augment=True):
        """
        Load and preprocess MNIST dataset with optional augmentation
        
        WHAT THIS FUNCTION DOES:
        ========================
        1. Downloads/loads MNIST dataset (if not already downloaded)
        2. Normalizes pixel values from [0, 255] to [-0.5, 0.5]
        3. Reshapes images to add channel dimension (required for CNN)
        4. Converts labels to one-hot encoding (required for categorical crossentropy)
        5. Optionally applies data augmentation (creates more training samples)
        
        WHY NORMALIZE TO [-0.5, 0.5]?
        - Centers data around zero, which helps neural networks converge faster
        - Better than [0, 1] normalization for some optimizers
        
        WHY ONE-HOT ENCODING?
        - Label 3 becomes [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
        - Required for categorical_crossentropy loss function
        
        WHY DATA AUGMENTATION?
        - Creates variations of images (rotated, shifted, zoomed)
        - Increases dataset size without collecting new data
        - Helps model generalize better (recognize digits in different positions)
        
        Args:
            augment (bool): If True, apply data augmentation (default: True)
        
        Returns:
            tuple: ((x_train, y_train), (x_test, y_test))
                - x_train: Training images (numpy array)
                - y_train: Training labels (one-hot encoded)
                - x_test: Test images (numpy array)
                - y_test: Test labels (one-hot encoded)
        """
        print("\n" + "=" * 70)
        print("📥 LOADING AND ENHANCING MNIST DATASET")
        print("=" * 70)
        
        # STEP 1: Load MNIST dataset
        # MNIST is built into Keras, so this downloads it automatically if needed
        # Returns: (x_train, y_train), (x_test, y_test)
        # - x_train: 60,000 images of shape (28, 28) with pixel values 0-255
        # - y_train: 60,000 labels (0-9)
        # - x_test: 10,000 images for testing
        # - y_test: 10,000 labels for testing
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        
        print(f"\n📊 Dataset Statistics:")
        print(f"  Training samples: {x_train.shape[0]:,}")  # 60,000
        print(f"  Test samples: {x_test.shape[0]:,}")  # 10,000
        print(f"  Image shape: {x_train.shape[1:]}")  # (28, 28)
        
        # STEP 2: Normalize pixel values
        # Convert from uint8 (0-255) to float32 and normalize to [-0.5, 0.5]
        # Why float32? Neural networks work better with floating point numbers
        # Why -0.5? Centers data around zero for better convergence
        x_train = x_train.astype('float32') / 255.0 - 0.5
        x_test = x_test.astype('float32') / 255.0 - 0.5
        
        # STEP 3: Reshape to add channel dimension
        # CNN expects: (batch, height, width, channels)
        # MNIST gives: (batch, height, width)
        # We need: (batch, height, width, 1) for grayscale images
        # expand_dims adds dimension at position -1 (last position)
        x_train = np.expand_dims(x_train, -1)  # (60000, 28, 28) -> (60000, 28, 28, 1)
        x_test = np.expand_dims(x_test, -1)  # (10000, 28, 28) -> (10000, 28, 28, 1)
        
        # STEP 4: Convert labels to one-hot encoding
        # Label 3 becomes [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
        # Required for categorical_crossentropy loss function
        # Example: y_train[0] = 5 -> becomes [0,0,0,0,0,1,0,0,0,0]
        y_train = keras.utils.to_categorical(y_train, 10)  # (60000,) -> (60000, 10)
        y_test = keras.utils.to_categorical(y_test, 10)  # (10000,) -> (10000, 10)
        
        # STEP 5: Data augmentation (optional but recommended)
        # Creates variations: rotated, shifted, zoomed versions of images
        # Doubles the dataset size (augmentation_factor=2)
        # Helps model learn to recognize digits in different positions/angles
        if augment:
            print("\n🔄 Applying data augmentation...")
            x_train_aug, y_train_aug = augment_data(x_train, y_train, augmentation_factor=2)
            print(f"  Augmented: {len(x_train)} → {len(x_train_aug)} samples")
            x_train, y_train = x_train_aug, y_train_aug
        
        print(f"\n✅ Final dataset shapes:")
        print(f"  x_train: {x_train.shape}")  # e.g., (120000, 28, 28, 1) after augmentation
        print(f"  y_train: {y_train.shape}")  # e.g., (120000, 10)
        print(f"  x_test: {x_test.shape}")  # (10000, 28, 28, 1)
        print(f"  y_test: {y_test.shape}")  # (10000, 10)
        
        return (x_train, y_train), (x_test, y_test)
    
    def build_advanced_model(self):
        """
        Build advanced CNN architecture for digit recognition
        
        WHAT THIS FUNCTION DOES:
        ========================
        Creates a Convolutional Neural Network (CNN) with:
        - 3 Convolutional Blocks (feature extraction)
        - Global Average Pooling (reduces overfitting)
        - 2 Dense Layers (classification)
        - Output Layer (10 classes for digits 0-9)
        
        ARCHITECTURE EXPLANATION:
        =========================
        
        CONVOLUTIONAL LAYERS (Conv2D):
        - Purpose: Extract features from images (edges, curves, patterns)
        - How it works: Slides a 3x3 filter over the image to detect patterns
        - 32 filters: Learns 32 different feature patterns
        - 64 filters: Learns 64 more complex patterns
        - 128 filters: Learns 128 high-level patterns
        
        BATCH NORMALIZATION:
        - Purpose: Normalizes inputs to each layer
        - Benefit: Faster training, more stable, allows higher learning rates
        
        MAX POOLING:
        - Purpose: Reduces image size (28x28 -> 14x14 -> 7x7)
        - Benefit: Fewer parameters, faster training, translation invariance
        
        DROPOUT:
        - Purpose: Prevents overfitting (memorizing training data)
        - How: Randomly sets 25-50% of neurons to zero during training
        - Benefit: Forces model to learn more general patterns
        
        GLOBAL AVERAGE POOLING:
        - Purpose: Reduces spatial dimensions to single values
        - Alternative to: Flatten + Dense (but uses fewer parameters)
        - Benefit: Less overfitting, fewer parameters
        
        DENSE LAYERS:
        - Purpose: Final classification based on extracted features
        - 256 neurons: First dense layer (high capacity)
        - 128 neurons: Second dense layer (refined features)
        
        OUTPUT LAYER:
        - 10 neurons: One for each digit (0-9)
        - Softmax activation: Converts to probabilities (sum = 1.0)
        
        Returns:
            keras.Model: Compiled CNN model ready for training
        """
        print("\n" + "=" * 70)
        print("🏗️  BUILDING ADVANCED DIGIT RECOGNITION MODEL")
        print("=" * 70)
        
        # Create Sequential model (layers stacked one after another)
        model = keras.Sequential([
            # ====================================================================
            # INPUT LAYER: 28x28x1 grayscale images
            # ====================================================================
            # No explicit input layer needed - first Conv2D defines input shape
            
            # ====================================================================
            # CONVOLUTIONAL BLOCK 1: First feature extraction
            # ====================================================================
            # Conv2D: 32 filters, each 3x3, ReLU activation
            # - 32 filters: Learns 32 different feature patterns (edges, curves)
            # - (3, 3): Filter size (standard for small images)
            # - activation='relu': Rectified Linear Unit (f(x) = max(0, x))
            # - input_shape=(28, 28, 1): First layer must specify input size
            # - kernel_initializer='he_normal': Weight initialization method
            # - padding='same': Keeps output size same as input (adds padding)
            # - kernel_regularizer: L2 regularization (penalizes large weights)
            layers.Conv2D(32, (3, 3), activation='relu', 
                         input_shape=(28, 28, 1),  # Input: 28x28 grayscale
                         kernel_initializer='he_normal',  # Weight initialization
                         padding='same',  # Keep same size (28x28)
                         kernel_regularizer=keras.regularizers.l2(0.001),  # Prevent overfitting
                         name='conv2d_1'),
            
            # BatchNormalization: Normalizes the output of previous layer
            # - Speeds up training, allows higher learning rates
            layers.BatchNormalization(name='batch_norm_1'),
            
            # Second Conv2D in same block: Learns more complex patterns
            layers.Conv2D(32, (3, 3), activation='relu',
                         kernel_initializer='he_normal',
                         padding='same',  # Still 28x28
                         kernel_regularizer=keras.regularizers.l2(0.001),
                         name='conv2d_2'),
            layers.BatchNormalization(name='batch_norm_2'),
            
            # MaxPooling2D: Reduces size from 28x28 to 14x14
            # - Takes maximum value in each 2x2 region
            # - Reduces parameters, provides translation invariance
            layers.MaxPooling2D((2, 2), name='maxpool_1'),  # 28x28 -> 14x14
            
            # Dropout: Randomly disables 25% of neurons during training
            # - Prevents overfitting (memorizing training data)
            layers.Dropout(0.25, name='dropout_1'),
            
            # ====================================================================
            # CONVOLUTIONAL BLOCK 2: Deeper feature extraction
            # ====================================================================
            # Now using 64 filters to learn more complex patterns
            # Input: 14x14x32 (from previous block)
            layers.Conv2D(64, (3, 3), activation='relu',  # 64 filters (more complex features)
                         kernel_initializer='he_normal',
                         padding='same',  # Still 14x14
                         kernel_regularizer=keras.regularizers.l2(0.001),
                         name='conv2d_3'),
            layers.BatchNormalization(name='batch_norm_3'),
            
            # Second Conv2D in block 2
            layers.Conv2D(64, (3, 3), activation='relu',
                         kernel_initializer='he_normal',
                         padding='same',
                         kernel_regularizer=keras.regularizers.l2(0.001),
                         name='conv2d_4'),
            layers.BatchNormalization(name='batch_norm_4'),
            
            # MaxPooling: Reduces from 14x14 to 7x7
            layers.MaxPooling2D((2, 2), name='maxpool_2'),  # 14x14 -> 7x7
            layers.Dropout(0.25, name='dropout_2'),
            
            # ====================================================================
            # CONVOLUTIONAL BLOCK 3: High-level feature extraction
            # ====================================================================
            # 128 filters: Learns very complex, high-level patterns
            # Input: 7x7x64 (from previous block)
            layers.Conv2D(128, (3, 3), activation='relu',  # 128 filters (high-level features)
                         kernel_initializer='he_normal',
                         padding='same',  # Still 7x7
                         kernel_regularizer=keras.regularizers.l2(0.001),
                         name='conv2d_5'),
            layers.BatchNormalization(name='batch_norm_5'),
            layers.Dropout(0.25, name='dropout_3'),
            
            # ====================================================================
            # GLOBAL AVERAGE POOLING: Reduces spatial dimensions
            # ====================================================================
            # Converts 7x7x128 feature maps to 128 single values
            # Alternative to Flatten + Dense, but uses fewer parameters
            # Reduces overfitting risk
            layers.GlobalAveragePooling2D(name='global_avg_pool'),  # 7x7x128 -> 128
            
            # ====================================================================
            # DENSE LAYERS: Final classification
            # ====================================================================
            # Dense Layer 1: 256 neurons (high capacity for complex decisions)
            # Input: 128 values (from Global Average Pooling)
            layers.Dense(256, activation='relu',  # 256 neurons
                        kernel_initializer='he_normal',
                        kernel_regularizer=keras.regularizers.l2(0.001),
                        name='dense_1'),
            layers.BatchNormalization(name='batch_norm_6'),
            layers.Dropout(0.5, name='dropout_4'),  # 50% dropout (high, prevents overfitting)
            
            # Dense Layer 2: 128 neurons (refined features)
            # Input: 256 values (from previous dense layer)
            layers.Dense(128, activation='relu',  # 128 neurons
                        kernel_initializer='he_normal',
                        kernel_regularizer=keras.regularizers.l2(0.001),
                        name='dense_2'),
            layers.BatchNormalization(name='batch_norm_7'),
            layers.Dropout(0.3, name='dropout_5'),  # 30% dropout
            
            # ====================================================================
            # OUTPUT LAYER: Final prediction
            # ====================================================================
            # 10 neurons: One for each digit (0-9)
            # Softmax activation: Converts to probabilities (all sum to 1.0)
            # Example output: [0.01, 0.02, 0.05, 0.80, 0.01, ...] for digit "3"
            layers.Dense(10, activation='softmax', name='output')  # 10 classes
        ])
        
        # ====================================================================
        # OPTIMIZER SETUP: Adam with Learning Rate Scheduling
        # ====================================================================
        # WHY ADAM OPTIMIZER?
        # - Adaptive learning rate (adjusts for each parameter)
        # - Works well with default settings
        # - Faster convergence than basic SGD
        
        # LEARNING RATE SCHEDULE:
        # - Starts at 0.001
        # - Decays exponentially every 10,000 steps
        # - Decay rate: 0.96 (multiplies by 0.96 each time)
        # - Why decay? Smaller steps near end of training (fine-tuning)
        initial_learning_rate = 0.001  # Starting learning rate
        lr_schedule = keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate,  # Start at 0.001
            decay_steps=10000,  # Decay every 10,000 steps
            decay_rate=0.96,  # Multiply by 0.96 (4% reduction)
            staircase=True  # Step-wise decay (not smooth)
        )
        
        # ADAM OPTIMIZER PARAMETERS:
        # - learning_rate: Uses schedule (starts at 0.001, decreases over time)
        # - beta_1: Momentum term (0.9 = standard)
        # - beta_2: RMSprop term (0.999 = standard)
        # - epsilon: Small number to prevent division by zero
        # - amsgrad: Variant of Adam (more stable)
        optimizer = keras.optimizers.Adam(
            learning_rate=lr_schedule,  # Use learning rate schedule
            beta_1=0.9,  # Momentum decay rate
            beta_2=0.999,  # RMSprop decay rate
            epsilon=1e-07,  # Numerical stability constant
            amsgrad=True  # Use AMSGrad variant (more stable)
        )
        
        # ====================================================================
        # COMPILE MODEL: Set optimizer, loss, and metrics
        # ====================================================================
        # WHY COMPILE?
        # - Configures model for training
        # - Defines how to calculate loss and update weights
        # - Specifies which metrics to track
        
        # LOSS FUNCTION: categorical_crossentropy
        # - Used for multi-class classification
        # - Compares predicted probabilities with true one-hot labels
        # - Penalizes wrong predictions more than right ones
        
        # METRICS: Track during training
        # - accuracy: Percentage of correct predictions
        # - precision: How many predicted positives are actually positive
        # - recall: How many actual positives are found
        # - auc: Area under ROC curve (classification quality)
        model.compile(
            optimizer=optimizer,  # Adam with learning rate schedule
            loss='categorical_crossentropy',  # Multi-class classification loss
            metrics=['accuracy',  # Main metric: % correct
                    keras.metrics.Precision(name='precision'),  # Precision metric
                    keras.metrics.Recall(name='recall'),  # Recall metric
                    keras.metrics.AUC(name='auc')]  # AUC metric
        )
        
        self.model = model
        print_model_summary(model, "Enhanced Digit Recognition Model")
        
        return model
    
    def train(self, x_train, y_train, x_val=None, y_val=None, epochs=20):
        """Train the model with advanced techniques"""
        print("\n" + "=" * 70)
        print("🚀 TRAINING ENHANCED DIGIT RECOGNITION MODEL")
        print("=" * 70)
        
        print(f"\n⚙️  Training Configuration:")
        print(f"   Epochs: {epochs}")
        print(f"   Batch Size: 64")
        print(f"   Optimizer: Adam with Exponential Decay")
        print(f"   Learning Rate: 0.001 → 0.0001")
        print(f"   Regularization: L2 (0.001)")
        print(f"   Dropout: Yes (25-50%)")
        
        # ====================================================================
        # CALLBACKS: Functions called during training
        # ====================================================================
        # WHY CALLBACKS?
        # - Automate training process
        # - Save best model automatically
        # - Stop early if not improving
        # - Adjust learning rate dynamically
        
        callbacks = [
            # CALLBACK 1: Early Stopping
            # - Stops training if validation loss doesn't improve for 10 epochs
            # - Prevents overfitting (training too long)
            # - Saves time
            # - restore_best_weights: Keeps the best model (not the last one)
            keras.callbacks.EarlyStopping(
                monitor='val_loss',  # Watch validation loss
                patience=10,  # Wait 10 epochs without improvement
                restore_best_weights=True,  # Keep best weights (not last)
                verbose=1,  # Print messages
                mode='min'  # Lower is better
            ),
            
            # CALLBACK 2: Reduce Learning Rate on Plateau
            # - Reduces learning rate if validation loss stops improving
            # - Helps fine-tune when stuck
            # - factor=0.5: Cuts learning rate in half
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',  # Watch validation loss
                factor=0.5,  # Multiply learning rate by 0.5 (halve it)
                patience=5,  # Wait 5 epochs without improvement
                min_lr=0.00001,  # Don't go below this learning rate
                verbose=1,  # Print when reducing LR
                mode='min'  # Lower is better
            ),
            
            # CALLBACK 3: Model Checkpoint
            # - Saves model automatically when validation accuracy improves
            # - save_best_only: Only saves when better than previous
            # - Prevents saving worse models
            keras.callbacks.ModelCheckpoint(
                filepath='models/best_digit_model.h5',  # Where to save
                monitor='val_accuracy',  # Watch validation accuracy
                save_best_only=True,  # Only save if better
                save_weights_only=False,  # Save entire model (not just weights)
                verbose=1,  # Print when saving
                mode='max'  # Higher is better
            ),
            
            # CALLBACK 4: TensorBoard Logger
            # - Logs metrics for visualization in TensorBoard
            # - Can view training curves in browser
            # - Useful for debugging and analysis
            keras.callbacks.TensorBoard(
                log_dir='models/logs/digit',  # Where to save logs
                histogram_freq=1,  # Log histograms every epoch
                write_graph=True,  # Save computation graph
                write_images=True  # Save sample images
            ),
            
            # CALLBACK 5: CSV Logger
            # - Saves training metrics to CSV file
            # - Can open in Excel for analysis
            # - Useful for tracking progress
            keras.callbacks.CSVLogger(
                'models/digit_training_log.csv',  # CSV file path
                separator=',',  # Comma-separated values
                append=False  # Overwrite existing file
            )
        ]
        
        # ====================================================================
        # VALIDATION SPLIT: Separate training and validation data
        # ====================================================================
        # WHY VALIDATION SET?
        # - Used to monitor training progress (not used for training)
        # - Detects overfitting (if training accuracy >> validation accuracy)
        # - Used by callbacks (early stopping, learning rate reduction)
        
        # If validation data not provided, split from training data
        # - test_size=0.15: Use 15% for validation (85% for training)
        # - random_state=42: Same random split every time (reproducible)
        # - stratify: Maintains class distribution (same % of each digit)
        if x_val is None or y_val is None:
            x_train, x_val, y_train, y_val = train_test_split(
                x_train, y_train, 
                test_size=0.15,  # 15% for validation
                random_state=42,  # For reproducibility
                stratify=np.argmax(y_train, axis=1)  # Maintain class balance
            )
        
        print(f"\n📊 Training Set: {x_train.shape[0]:,} samples")
        print(f"📊 Validation Set: {x_val.shape[0]:,} samples")
        
        # ====================================================================
        # TRAIN MODEL: The actual training process
        # ====================================================================
        # WHAT HAPPENS DURING TRAINING:
        # 1. Model processes batches of images (64 at a time)
        # 2. Calculates predictions
        # 3. Compares with true labels (calculates loss)
        # 4. Updates weights using optimizer (backpropagation)
        # 5. Repeats for all batches = 1 epoch
        # 6. Repeats for all epochs
        
        # PARAMETERS:
        # - x_train, y_train: Training data and labels
        # - validation_data: Used to monitor progress (not for training)
        # - epochs: Number of complete passes through training data
        # - batch_size: Number of samples processed at once (64)
        # - callbacks: Functions called during training (early stopping, etc.)
        # - verbose=1: Print progress (0=silent, 1=progress bar, 2=one line per epoch)
        # - shuffle=True: Randomize order of samples each epoch
        
        # RETURNS:
        # - history: Dictionary with training metrics (accuracy, loss per epoch)
        self.history = self.model.fit(
            x_train, y_train,  # Training data
            validation_data=(x_val, y_val),  # Validation data (for monitoring)
            epochs=epochs,  # Number of training cycles
            batch_size=64,  # Process 64 images at once
            callbacks=callbacks,  # Early stopping, checkpointing, etc.
            verbose=1,  # Show progress bar
            shuffle=True  # Randomize sample order each epoch
        )
        
        print("\n✅ Training completed!")
        
        # Return training history (contains accuracy, loss for each epoch)
        return self.history
    
    def save(self, filepath='models/digit_model_enhanced.h5'):
        """Save trained model"""
        self.model.save(filepath)
        print(f"\n💾 Enhanced model saved to: {filepath}")
        
        # Also save in TensorFlow SavedModel format
        saved_model_path = filepath.replace('.h5', '_savedmodel')
        tf.saved_model.save(self.model, saved_model_path)
        print(f"💾 SavedModel format saved to: {saved_model_path}")


class EnhancedShapeRecognitionModel:
    """Advanced CNN Model for Shape Recognition (Circle, Square, Triangle)"""
    
    def __init__(self):
        self.model = None
        self.history = None
        self.class_names = ['Circle', 'Square', 'Triangle']
    
    def load_data(self, num_samples=15000, augment=True):
        """Generate and prepare enhanced shape dataset"""
        print("\n" + "=" * 70)
        print("📥 GENERATING ENHANCED SHAPE DATASET")
        print("=" * 70)
        
        # Generate enhanced shapes
        x_data, y_data = generate_enhanced_shapes(
            num_samples=num_samples, 
            image_size=28,
            augment=augment
        )
        
        print(f"\n📊 Dataset Statistics:")
        print(f"  Total samples: {x_data.shape[0]:,}")
        print(f"  Image shape: {x_data.shape[1:]}")
        print(f"  Class distribution:")
        for i, name in enumerate(self.class_names):
            count = np.sum(y_data == i)
            print(f"    {name}: {count:,} samples ({count/len(y_data)*100:.1f}%)")
        
        # Convert labels to one-hot encoding
        y_data_onehot = keras.utils.to_categorical(y_data, 3)
        
        # Split into train and test
        x_train, x_test, y_train, y_test = train_test_split(
            x_data, y_data_onehot, 
            test_size=0.2, 
            random_state=42,
            stratify=y_data
        )
        
        # Data augmentation for training set
        if augment:
            print("\n🔄 Applying additional data augmentation...")
            x_train_aug, y_train_aug = augment_data(x_train, y_train, augmentation_factor=2)
            print(f"  Augmented: {len(x_train)} → {len(x_train_aug)} samples")
            x_train, y_train = x_train_aug, y_train_aug
        
        print(f"\n✅ Final dataset shapes:")
        print(f"  x_train: {x_train.shape}")
        print(f"  y_train: {y_train.shape}")
        print(f"  x_test: {x_test.shape}")
        print(f"  y_test: {y_test.shape}")
        
        return (x_train, y_train), (x_test, y_test)
    
    def build_advanced_model(self):
        """Build advanced CNN architecture for shape recognition"""
        print("\n" + "=" * 70)
        print("🏗️  BUILDING ENHANCED SHAPE RECOGNITION MODEL")
        print("=" * 70)
        
        model = keras.Sequential([
            # Input: 28x28x1 grayscale images
            
            # Convolutional Block 1
            layers.Conv2D(32, (3, 3), activation='relu',
                         input_shape=(28, 28, 1),
                         kernel_initializer='he_normal',
                         padding='same',
                         kernel_regularizer=keras.regularizers.l2(0.001),
                         name='conv2d_1'),
            layers.BatchNormalization(name='batch_norm_1'),
            layers.Conv2D(32, (3, 3), activation='relu',
                         kernel_initializer='he_normal',
                         padding='same',
                         kernel_regularizer=keras.regularizers.l2(0.001),
                         name='conv2d_2'),
            layers.BatchNormalization(name='batch_norm_2'),
            layers.MaxPooling2D((2, 2), name='maxpool_1'),
            layers.Dropout(0.25, name='dropout_1'),
            
            # Convolutional Block 2
            layers.Conv2D(64, (3, 3), activation='relu',
                         kernel_initializer='he_normal',
                         padding='same',
                         kernel_regularizer=keras.regularizers.l2(0.001),
                         name='conv2d_3'),
            layers.BatchNormalization(name='batch_norm_3'),
            layers.Conv2D(64, (3, 3), activation='relu',
                         kernel_initializer='he_normal',
                         padding='same',
                         kernel_regularizer=keras.regularizers.l2(0.001),
                         name='conv2d_4'),
            layers.BatchNormalization(name='batch_norm_4'),
            layers.MaxPooling2D((2, 2), name='maxpool_2'),
            layers.Dropout(0.25, name='dropout_2'),
            
            # Global Average Pooling
            layers.GlobalAveragePooling2D(name='global_avg_pool'),
            
            # Dense Layers
            layers.Dense(128, activation='relu',
                        kernel_initializer='he_normal',
                        kernel_regularizer=keras.regularizers.l2(0.001),
                        name='dense_1'),
            layers.BatchNormalization(name='batch_norm_5'),
            layers.Dropout(0.4, name='dropout_3'),
            
            layers.Dense(64, activation='relu',
                        kernel_initializer='he_normal',
                        kernel_regularizer=keras.regularizers.l2(0.001),
                        name='dense_2'),
            layers.BatchNormalization(name='batch_norm_6'),
            layers.Dropout(0.3, name='dropout_4'),
            
            # Output Layer: 3 classes (circle, square, triangle)
            layers.Dense(3, activation='softmax', name='output')
        ])
        
        # Optimizer
        optimizer = keras.optimizers.Adam(
            learning_rate=0.001,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-07
        )
        
        # Compile model
        model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy',
                    keras.metrics.Precision(name='precision'),
                    keras.metrics.Recall(name='recall')]
        )
        
        self.model = model
        print_model_summary(model, "Enhanced Shape Recognition Model")
        
        return model
    
    def train(self, x_train, y_train, x_val=None, y_val=None, epochs=15):
        """Train the model with advanced techniques"""
        print("\n" + "=" * 70)
        print("🚀 TRAINING ENHANCED SHAPE RECOGNITION MODEL")
        print("=" * 70)
        
        print(f"\n⚙️  Training Configuration:")
        print(f"   Epochs: {epochs}")
        print(f"   Batch Size: 32")
        print(f"   Optimizer: Adam")
        print(f"   Regularization: L2 (0.001)")
        print(f"   Dropout: Yes (25-40%)")
        
        # Callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=8,
                restore_best_weights=True,
                verbose=1
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=4,
                min_lr=0.00001,
                verbose=1
            ),
            keras.callbacks.ModelCheckpoint(
                filepath='models/best_shape_model.h5',
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            )
        ]
        
        # Split validation data if not provided
        if x_val is None or y_val is None:
            x_train, x_val, y_train, y_val = train_test_split(
                x_train, y_train, 
                test_size=0.15, 
                random_state=42,
                stratify=np.argmax(y_train, axis=1)
            )
        
        print(f"\n📊 Training Set: {x_train.shape[0]:,} samples")
        print(f"📊 Validation Set: {x_val.shape[0]:,} samples")
        
        # Train model
        self.history = self.model.fit(
            x_train, y_train,
            validation_data=(x_val, y_val),
            epochs=epochs,
            batch_size=32,
            callbacks=callbacks,
            verbose=1,
            shuffle=True
        )
        
        print("\n✅ Training completed!")
        
        return self.history
    
    def save(self, filepath='models/shape_model_enhanced.h5'):
        """Save trained model"""
        self.model.save(filepath)
        print(f"\n💾 Enhanced model saved to: {filepath}")


def train_ensemble_models():
    """Train multiple models for ensemble learning"""
    
    create_directories()
    
    print("\n" + "=" * 70)
    print("🤖 ENHANCED AI RECOGNITION SYSTEM - ENSEMBLE TRAINING")
    print("=" * 70)
    
    models_to_train = 3  # Number of models for ensemble
    
    digit_models = []
    shape_models = []
    
    # ========================================
    # TRAIN MULTIPLE DIGIT MODELS
    # ========================================
    
    print("\n\n### PART 1: ENSEMBLE DIGIT MODELS ###\n")
    
    for i in range(models_to_train):
        print(f"\n🎯 Training Digit Model {i+1}/{models_to_train}")
        
        digit_model = EnhancedDigitRecognitionModel()
        
        # Load data (different augmentation each time)
        (x_train_digit, y_train_digit), (x_test_digit, y_test_digit) = digit_model.load_data(
            augment=True
        )
        
        # Build model
        digit_model.build_advanced_model()
        
        # Train model
        digit_model.train(x_train_digit, y_train_digit, epochs=15)
        
        # Evaluate
        results = evaluate_model(digit_model.model, x_test_digit, y_test_digit,
                               class_names=digit_model.class_names)
        
        # Save model
        model_path = f'models/digit_model_ensemble_{i+1}.h5'
        digit_model.save(model_path)
        digit_models.append(digit_model.model)
        
        # Plot training history
        plot_training_history(digit_model.history,
                            model_name=f'Digit Model {i+1}',
                            save_path=f'models/digit_training_history_{i+1}.png')
        
        # Save model info
        save_model_info(digit_model.model, digit_model.history,
                       f'models/digit_model_info_{i+1}.txt')
    
    # ========================================
    # TRAIN MULTIPLE SHAPE MODELS
    # ========================================
    
    print("\n\n### PART 2: ENSEMBLE SHAPE MODELS ###\n")
    
    for i in range(models_to_train):
        print(f"\n🎯 Training Shape Model {i+1}/{models_to_train}")
        
        shape_model = EnhancedShapeRecognitionModel()
        
        # Load data
        (x_train_shape, y_train_shape), (x_test_shape, y_test_shape) = shape_model.load_data(
            num_samples=15000,
            augment=True
        )
        
        # Plot sample shapes from first model only
        if i == 0:
            plot_sample_images(x_train_shape[:15],
                             np.argmax(y_train_shape[:15], axis=1),
                             class_names=shape_model.class_names,
                             save_path='models/enhanced_shape_samples.png')
        
        # Build model
        shape_model.build_advanced_model()
        
        # Train model
        shape_model.train(x_train_shape, y_train_shape, epochs=12)
        
        # Evaluate
        results = evaluate_model(shape_model.model, x_test_shape, y_test_shape,
                               class_names=shape_model.class_names)
        
        # Save model
        model_path = f'models/shape_model_ensemble_{i+1}.h5'
        shape_model.save(model_path)
        shape_models.append(shape_model.model)
        
        # Plot training history
        plot_training_history(shape_model.history,
                            model_name=f'Shape Model {i+1}',
                            save_path=f'models/shape_training_history_{i+1}.png')
        
        # Save model info
        save_model_info(shape_model.model, shape_model.history,
                       f'models/shape_model_info_{i+1}.txt')
    
    # ========================================
    # CREATE ENSEMBLE MODELS
    # ========================================
    
    print("\n\n### PART 3: CREATING ENSEMBLE MODELS ###\n")
    
    # Save ensemble models list
    import pickle
    
    with open('models/ensemble_digit_models.pkl', 'wb') as f:
        pickle.dump([m.get_config() for m in digit_models], f)
    
    with open('models/ensemble_shape_models.pkl', 'wb') as f:
        pickle.dump([m.get_config() for m in shape_models], f)
    
    print("✅ Ensemble models configuration saved")
    
    # ========================================
    # FINAL SUMMARY
    # ========================================
    
    print("\n\n" + "=" * 70)
    print("🎉 ENSEMBLE TRAINING COMPLETE!")
    print("=" * 70)
    print("\n✅ Models saved:")
    print("\n  Digit Models:")
    for i in range(models_to_train):
        print(f"    - models/digit_model_ensemble_{i+1}.h5")
    
    print("\n  Shape Models:")
    for i in range(models_to_train):
        print(f"    - models/shape_model_ensemble_{i+1}.h5")
    
    print("\n✅ Training graphs saved:")
    print("    - models/digit_training_history_*.png")
    print("    - models/shape_training_history_*.png")
    
    print("\n✅ Model info saved:")
    print("    - models/digit_model_info_*.txt")
    print("    - models/shape_model_info_*.txt")
    
    print("\n✅ Ensemble configuration saved:")
    print("    - models/ensemble_digit_models.pkl")
    print("    - models/ensemble_shape_models.pkl")
    
    print("\n📊 Expected Accuracy Improvement:")
    print("    Basic Model: 95-97%")
    print("    Enhanced Model: 98-99%")
    print("    Ensemble Model: 99%+")
    
    print("\n🚀 Ready to test! Use the enhanced GUI with ensemble predictions")
    print("=" * 70 + "\n")


def train_single_enhanced_models():
    """Train single enhanced models (quicker option)"""
    
    create_directories()
    
    print("\n" + "=" * 70)
    print("🤖 ENHANCED AI RECOGNITION SYSTEM - SINGLE MODEL TRAINING")
    print("=" * 70)
    
    # ========================================
    # TRAIN ENHANCED DIGIT MODEL
    # ========================================
    
    print("\n\n### PART 1: ENHANCED DIGIT MODEL ###\n")
    
    digit_model = EnhancedDigitRecognitionModel()
    
    # Load data
    (x_train_digit, y_train_digit), (x_test_digit, y_test_digit) = digit_model.load_data(
        augment=True
    )
    
    # Build model
    digit_model.build_advanced_model()
    
    # Train model
    digit_model.train(x_train_digit, y_train_digit, epochs=20)
    
    # Evaluate
    results = evaluate_model(digit_model.model, x_test_digit, y_test_digit,
                           class_names=digit_model.class_names)
    
    # Analyze failed predictions
    analyze_failed_predictions(digit_model.model, x_test_digit, y_test_digit,
                             class_names=digit_model.class_names)
    
    # Save model
    digit_model.save('models/digit_model_enhanced.h5')
    
    # Plot training history
    plot_training_history(digit_model.history,
                         model_name='Enhanced Digit Recognition',
                         save_path='models/enhanced_digit_training_history.png')
    
    # Save model info
    save_model_info(digit_model.model, digit_model.history,
                   'models/enhanced_digit_model_info.txt')
    
    # ========================================
    # TRAIN ENHANCED SHAPE MODEL
    # ========================================
    
    print("\n\n### PART 2: ENHANCED SHAPE MODEL ###\n")
    
    shape_model = EnhancedShapeRecognitionModel()
    
    # Load data
    (x_train_shape, y_train_shape), (x_test_shape, y_test_shape) = shape_model.load_data(
        num_samples=15000,
        augment=True
    )
    
    # Plot sample shapes
    plot_sample_images(x_train_shape[:15],
                      np.argmax(y_train_shape[:15], axis=1),
                      class_names=shape_model.class_names,
                      save_path='models/enhanced_shape_samples.png')
    
    # Build model
    shape_model.build_advanced_model()
    
    # Train model
    shape_model.train(x_train_shape, y_train_shape, epochs=15)
    
    # Evaluate
    results = evaluate_model(shape_model.model, x_test_shape, y_test_shape,
                           class_names=shape_model.class_names)
    
    # Save model
    shape_model.save('models/shape_model_enhanced.h5')
    
    # Plot training history
    plot_training_history(shape_model.history,
                         model_name='Enhanced Shape Recognition',
                         save_path='models/enhanced_shape_training_history.png')
    
    # Save model info
    save_model_info(shape_model.model, shape_model.history,
                   'models/enhanced_shape_model_info.txt')
    
    # ========================================
    # FINAL SUMMARY
    # ========================================
    
    print("\n\n" + "=" * 70)
    print("🎉 ENHANCED TRAINING COMPLETE!")
    print("=" * 70)
    print("\n✅ Models saved:")
    print("    - models/digit_model_enhanced.h5")
    print("    - models/shape_model_enhanced.h5")
    
    print("\n✅ Training graphs saved:")
    print("    - models/enhanced_digit_training_history.png")
    print("    - models/enhanced_shape_training_history.png")
    
    print("\n✅ Model info saved:")
    print("    - models/enhanced_digit_model_info.txt")
    print("    - models/enhanced_shape_model_info.txt")
    
    print("\n📊 Expected Accuracy:")
    print("    Digit Recognition: 98-99%")
    print("    Shape Recognition: 97-98%")
    
    print("\n🚀 Ready to use enhanced GUI!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    """Run training when executed directly"""
    
    print("\n" + "=" * 70)
    print("🤖 SELECT TRAINING MODE")
    print("=" * 70)
    print("\n[1] Train Ensemble Models (Highest Accuracy, 3 models each)")
    print("    ⏱️  Time: 60-90 minutes")
    print("    📊 Accuracy: 99%+")
    
    print("\n[2] Train Single Enhanced Models (Good Accuracy)")
    print("    ⏱️  Time: 20-30 minutes")
    print("    📊 Accuracy: 98-99%")
    
    print("\n[3] Train Basic Models (Quick Test)")
    print("    ⏱️  Time: 10-15 minutes")
    print("    📊 Accuracy: 95-97%")
    
    choice = input("\nSelect option (1-3): ").strip()
    
    if choice == '1':
        train_ensemble_models()
    elif choice == '2':
        train_single_enhanced_models()
    elif choice == '3':
        # You can keep your original training function here
        print("\n⚠️  Basic training not implemented. Using enhanced training instead.")
        train_single_enhanced_models()
    else:
        print("\n❌ Invalid choice. Using enhanced single model training.")
        train_single_enhanced_models()